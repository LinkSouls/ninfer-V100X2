#pragma once

// Self-contained (no CUDA, no engine types): builds and unit-tests in seconds.
//
// Disk third tier for historical KV pages. Fixed-size, 4 KiB-aligned slots live in one data
// file; each slot carries a 48-byte self-describing header (magic, content identity, CRC32C of
// the payload, recency stamp) followed by the payload, so a slot is addressable by content
// identity alone and an index loss can be repaired by scanning slot headers. A separate index
// file maps identity -> slot and is rewritten atomically (tmp + rename); data slots are only
// ever overwritten in place. Eviction is page-granular LRU ordered by recency stamp, located
// through an ordered index (no linear scan of slot headers).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer {

// Disk-tier configuration carried through the plan to the Program. An empty `dir` (or zero
// capacity) disables the tier: no files are created and no demotion occurs. When enabled, the
// Program owns one store under `dir`, bounded by `capacity_bytes`.
struct DiskKvTierConfig {
    std::filesystem::path dir; // empty => tier disabled
    std::size_t capacity_bytes = 0;

    [[nodiscard]] bool enabled() const noexcept { return !dir.empty() && capacity_bytes > 0; }
};

// Content identity of one disk-resident KV chunk: a rolling 128-bit digest pair over the
// licensed prefix up to the chunk's page-tail position, a program-level identity tag (the
// model/format/width class the payload was produced under), and the chunk's page-tail frontier.
// Two chunks with equal identities carry equal payload bytes.
struct DiskKVIdentity {
    std::uint64_t lo       = 0;
    std::uint64_t hi       = 0;
    std::uint64_t tag      = 0;
    std::uint32_t frontier = 0; // chunk page-tail position (first_page + pages)

    [[nodiscard]] bool operator==(const DiskKVIdentity& other) const noexcept {
        return lo == other.lo && hi == other.hi && tag == other.tag && frontier == other.frontier;
    }
    [[nodiscard]] bool operator!=(const DiskKVIdentity& other) const noexcept {
        return !(*this == other);
    }
    [[nodiscard]] bool operator<(const DiskKVIdentity& other) const noexcept {
        if (lo != other.lo) { return lo < other.lo; }
        if (hi != other.hi) { return hi < other.hi; }
        if (tag != other.tag) { return tag < other.tag; }
        return frontier < other.frontier;
    }
    [[nodiscard]] bool empty() const noexcept { return lo == 0 && hi == 0 && tag == 0 && frontier == 0; }
};

struct DiskKVIdentityHash {
    [[nodiscard]] std::size_t operator()(const DiskKVIdentity& id) const noexcept {
        std::uint64_t h = id.lo;
        h = (h ^ (id.hi + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2))) * 0x85EBCA6BULL;
        h ^= id.tag + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<std::uint64_t>(id.frontier) + (h << 6) + (h >> 2);
        return static_cast<std::size_t>(h);
    }
};

namespace detail {

inline constexpr std::uint32_t kDiskKvSlotMagic = 0x4E44564B;
inline constexpr std::uint32_t kDiskKvIndexMagic = 0x4944564B;
inline constexpr std::uint32_t kDiskKvMetaMagic  = 0x4D4D564B;
inline constexpr std::size_t kDiskKvAlignment    = 4096;

// Self-describing slot header. The on-disk layout is a fixed 48-byte little-endian field
// serialization (defined in disk_kv_store.cpp), NOT this struct's in-memory layout:
//   u32 magic | u64 lo | u64 hi | u64 tag | u32 frontier | u32 crc | u64 last_used | u32 payload_len
// `payload_len` records the payload written into the slot (a slot may hold a shorter payload
// than its capacity).
struct DiskKvSlotHeader {
    std::uint32_t magic;
    std::uint64_t lo;
    std::uint64_t hi;
    std::uint64_t tag;
    std::uint32_t frontier;
    std::uint32_t crc;         // CRC32C (Castagnoli) over the payload bytes
    std::uint64_t last_used;   // recency stamp; the LRU eviction key
    std::uint32_t payload_len; // payload bytes written into this slot
};

[[nodiscard]] std::uint32_t disk_kv_crc32c(const std::byte* data, std::size_t size) noexcept;

} // namespace detail

// Cumulative event counters for observability (serving startup JSON and runtime occupancy
// queries). All counters are monotonic for the lifetime of the open store.
struct DiskKVStats {
    std::uint64_t writes               = 0; // successful writes
    std::uint64_t write_failures       = 0; // rejected writes (oversized, full, I/O error)
    std::uint64_t reads                = 0; // successful reads
    std::uint64_t read_failures        = 0; // failed reads (missing, torn, CRC, I/O)
    std::uint64_t crc_mismatches       = 0; // reads dropped on a payload CRC mismatch
    std::uint64_t identity_mismatches  = 0; // reads dropped on a header/identity mismatch
    std::uint64_t evictions            = 0; // LRU slots evicted to make room
    std::uint64_t index_rebuilds       = 0; // full slot-scan index rebuilds (open recovery)
    std::uint64_t bytes_written        = 0; // payload bytes written
    std::uint64_t bytes_read           = 0; // payload bytes read
};

// One on-disk KV page store. All public operations are thread-safe (a single internal lock);
// the engine uses one writer thread for demotion and the plan thread for reads.
class DiskKVStore {
public:
    struct OpenStatus {
        bool ok         = false;
        std::string error;
    };

    // Opens the store under `dir`, creating the directory, data file, and index when absent.
    // `kv_format` is the stable identifier of the KV storage format the payloads were produced
    // under (the engine passes its KvCacheStorage raw value); a store created under a different
    // format is rejected at open with a clear reason. `max_payload_bytes` sizes the slots (a
    // multiple of 4 KiB, header included); `capacity_bytes` bounds the data file, slot-aligned
    // downward.
    [[nodiscard]] static OpenStatus
    open(const std::filesystem::path& dir, std::uint32_t kv_format, std::size_t max_payload_bytes,
         std::size_t capacity_bytes, std::unique_ptr<DiskKVStore>& out);

    ~DiskKVStore();
    DiskKVStore(const DiskKVStore&)            = delete;
    DiskKVStore& operator=(const DiskKVStore&) = delete;

    // Writes `data` under `identity`, overwriting in place when the identity is already
    // resident and evicting least-recently-used unprotected slots when a fresh slot is needed.
    // Returns false (with a reason in last_error()) when the payload exceeds the slot capacity
    // or every in-use slot is protected.
    bool write(const DiskKVIdentity& identity, std::span<const std::byte> data);

    // Reads the payload recorded under `identity` into `out`, verifying the slot magic, the full
    // identity, and the payload CRC. A missing identity, a torn header, or a CRC/identity
    // mismatch reports false; the failed slot is dropped while every other identity is left
    // untouched, and foreign bytes are never returned. A successful read refreshes the slot's
    // recency stamp.
    bool read(const DiskKVIdentity& identity, std::span<std::byte> out);

    [[nodiscard]] bool contains(const DiskKVIdentity& identity) const;
    // Refreshes the recency stamp of a resident identity (in-memory; it persists with the next
    // index flush). No-op for a missing identity.
    void touch(const DiskKVIdentity& identity);
    // Evicts the least recently used unprotected slot. False when no unprotected slot is in use.
    bool evict_oldest();
    // Evicts until `slots` free slots exist, or until nothing more is evictable.
    bool make_room(std::size_t slots);
    // Excludes a resident identity from LRU eviction (e.g. while a restore is reading it).
    void protect(const DiskKVIdentity& identity);
    void unprotect(const DiskKVIdentity& identity);
    [[nodiscard]] std::vector<DiskKVIdentity> active_identities() const;
    // Removes a resident identity and frees its slot (whole-branch eviction; other identities
    // are untouched). False when the identity is not resident.
    bool erase(const DiskKVIdentity& identity);
    // Rebuilds the identity -> slot index by scanning every slot header, keeping only entries
    // whose magic and CRC verify. Used at open when the index file is missing or corrupt, and
    // after out-of-band damage.
    void refresh_index();
    // Atomically rewrites the index file (tmp + rename). Mutating operations flush it
    // themselves; this is the explicit hook for persisting recency updates made by reads.
    void flush_index();

    [[nodiscard]] std::size_t capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t used_bytes() const noexcept;
    [[nodiscard]] std::size_t active_pages() const noexcept;
    [[nodiscard]] std::size_t slot_size() const noexcept;
    [[nodiscard]] std::size_t slot_count() const noexcept;
    [[nodiscard]] std::uint32_t kv_format() const noexcept;
    [[nodiscard]] const std::string& last_error() const noexcept;
    // Cumulative event counters (see DiskKVStats); a consistent snapshot under the store lock.
    [[nodiscard]] DiskKVStats stats() const noexcept;

private:
    struct Impl;
    explicit DiskKVStore(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer