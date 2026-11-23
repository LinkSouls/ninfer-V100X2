// Self-contained (no CUDA, no engine types): builds and unit-tests in seconds.
// See disk_kv_store.h for the slot/index design.

#include "core/disk_kv_store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace ninfer {
namespace {

inline constexpr std::uint32_t kDiskKvCrcPolynomial = 0x82F63B78; // CRC32C (Castagnoli), reflected
inline constexpr std::uint32_t kDiskKvIndexVersion  = 1;
inline constexpr std::size_t kDiskKvHeaderBytes     = 48;

std::uint32_t crc32c_bytes(const std::byte* data, std::size_t size) {
    struct Crc32cTable {
        std::uint32_t values[256];
        Crc32cTable() {
            for (std::uint32_t i = 0; i < 256; ++i) {
                std::uint32_t crc = i;
                for (int bit = 0; bit < 8; ++bit) {
                    crc = (crc & 1U) != 0U ? (crc >> 1) ^ kDiskKvCrcPolynomial : crc >> 1;
                }
                values[i] = crc;
            }
        }
    };
    static const Crc32cTable table;
    const auto* t = table.values;
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < size; ++i) {
        crc = t[(crc ^ static_cast<std::uint32_t>(data[i])) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

// Little-endian field (de)serialization so the on-disk layout never depends on host endianness:
// on a little-endian host the raw value bytes are already in on-disk order; on a big-endian
// host they must be swapped.
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define NINFER_DISK_KV_SWAP32(value) __builtin_bswap32(value)
#define NINFER_DISK_KV_SWAP64(value) __builtin_bswap64(value)
#else
#define NINFER_DISK_KV_SWAP32(value) (value)
#define NINFER_DISK_KV_SWAP64(value) (value)
#endif

void put_u32(std::string& out, std::uint32_t value) {
    const std::uint32_t le = NINFER_DISK_KV_SWAP32(value);
    out.append(reinterpret_cast<const char*>(&le), sizeof(le));
}
void put_u64(std::string& out, std::uint64_t value) {
    const std::uint64_t le = NINFER_DISK_KV_SWAP64(value);
    out.append(reinterpret_cast<const char*>(&le), sizeof(le));
}
bool get_u32(const std::string& in, std::size_t& offset, std::uint32_t& value) {
    if (offset + 4 > in.size()) { return false; }
    std::uint32_t le;
    std::memcpy(&le, in.data() + offset, 4);
    value = NINFER_DISK_KV_SWAP32(le);
    offset += 4;
    return true;
}
bool get_u64(const std::string& in, std::size_t& offset, std::uint64_t& value) {
    if (offset + 8 > in.size()) { return false; }
    std::uint64_t le;
    std::memcpy(&le, in.data() + offset, 8);
    value = NINFER_DISK_KV_SWAP64(le);
    offset += 8;
    return true;
}

bool read_whole_file(const std::filesystem::path& path, std::string& out) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { return false; }
    out.clear();
    char buffer[65536];
    for (;;) {
        const ssize_t n = ::read(fd, buffer, sizeof(buffer));
        if (n < 0) { ::close(fd); return false; }
        if (n == 0) { break; }
        out.append(buffer, static_cast<std::size_t>(n));
    }
    ::close(fd);
    return true;
}

// Writes `data` to a temp file in the same directory and renames it over `path`: a reader of
// `path` only ever sees the old complete file or the new complete file, never a partial write.
bool atomic_replace_file(const std::filesystem::path& path, const std::string& data) {
    auto tmp = path;
    tmp += ".tmp." + std::to_string(::getpid());
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { return false; }
    bool ok = true;
    std::size_t written = 0;
    while (written < data.size()) {
        const ssize_t n = ::write(fd, data.data() + written, data.size() - written);
        if (n <= 0) { ok = false; break; }
        written += static_cast<std::size_t>(n);
    }
    if (ok) { ok = ::fsync(fd) == 0; }
    const int close_result = ::close(fd);
    if (close_result != 0) { ok = false; }
    if (!ok) { (void)::unlink(tmp.c_str()); return false; }
    if (::rename(tmp.c_str(), path.c_str()) != 0) { (void)::unlink(tmp.c_str()); return false; }
    return true;
}

struct MetaRecord {
    std::uint32_t kv_format = 0;
    std::size_t slot_size   = 0;
    std::size_t slot_count  = 0;
};

bool read_meta(const std::filesystem::path& path, MetaRecord& out) {
    std::string buf;
    if (!read_whole_file(path, buf) || buf.size() < 24) { return false; }
    std::size_t offset = 0;
    std::uint32_t magic = 0;
    if (!get_u32(buf, offset, magic) || magic != detail::kDiskKvMetaMagic) { return false; }
    if (!get_u32(buf, offset, out.kv_format)) { return false; }
    std::uint64_t slot_size = 0;
    std::uint64_t slot_count = 0;
    if (!get_u64(buf, offset, slot_size) || !get_u64(buf, offset, slot_count)) { return false; }
    out.slot_size  = slot_size;
    out.slot_count = slot_count;
    return true;
}

bool write_meta(const std::filesystem::path& path, std::uint32_t kv_format, std::size_t slot_size,
                std::size_t slot_count) {
    std::string buf;
    put_u32(buf, detail::kDiskKvMetaMagic);
    put_u32(buf, kv_format);
    put_u64(buf, slot_size);
    put_u64(buf, slot_count);
    return atomic_replace_file(path, buf);
}

struct IndexRecord {
    std::vector<std::pair<DiskKVIdentity, std::uint32_t>> entries; // identity -> slot
    std::vector<std::uint64_t> last_used;                          // parallel stamps
    std::uint64_t max_stamp = 0;
};

// Structural validation only: the fast path trusts the index and leaves per-slot content checks
// to the read CRC.
bool parse_index(const std::string& buf, std::size_t slot_count, IndexRecord& out) {
    std::size_t offset = 0;
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    if (!get_u32(buf, offset, magic) || magic != detail::kDiskKvIndexMagic) { return false; }
    if (!get_u32(buf, offset, version) || version != kDiskKvIndexVersion) { return false; }
    std::uint64_t entry_count = 0;
    if (!get_u64(buf, offset, entry_count)) { return false; }
    if (!get_u64(buf, offset, out.max_stamp)) { return false; }
    if (entry_count > slot_count) { return false; }
    if (offset + entry_count * 40 != buf.size()) { return false; }
    out.entries.clear();
    out.last_used.clear();
    out.max_stamp = 0;
    std::vector<bool> slot_seen(slot_count, false);
    for (std::uint64_t i = 0; i < entry_count; ++i) {
        std::uint64_t lo = 0;
        std::uint64_t hi = 0;
        std::uint64_t tag = 0;
        std::uint32_t frontier = 0;
        std::uint32_t slot = 0;
        std::uint64_t last_used = 0;
        if (!get_u64(buf, offset, lo) || !get_u64(buf, offset, hi) || !get_u64(buf, offset, tag) ||
            !get_u32(buf, offset, frontier) || !get_u32(buf, offset, slot) ||
            !get_u64(buf, offset, last_used)) {
            return false;
        }
        if (slot >= slot_count || slot_seen[slot]) { return false; }
        slot_seen[slot] = true;
        out.entries.emplace_back(DiskKVIdentity{lo, hi, tag, frontier}, slot);
        out.last_used.push_back(last_used);
        out.max_stamp = std::max(out.max_stamp, last_used);
    }
    return true;
}

bool serialize_index(const IndexRecord& record, std::string& out) {
    out.clear();
    put_u32(out, detail::kDiskKvIndexMagic);
    put_u32(out, kDiskKvIndexVersion);
    put_u64(out, record.entries.size());
    put_u64(out, record.max_stamp);
    for (std::size_t i = 0; i < record.entries.size(); ++i) {
        const auto& [identity, slot] = record.entries[i];
        put_u64(out, identity.lo);
        put_u64(out, identity.hi);
        put_u64(out, identity.tag);
        put_u32(out, identity.frontier);
        put_u32(out, slot);
        put_u64(out, record.last_used[i]);
    }
    return true;
}

std::string format_name(std::uint32_t kv_format) {
    switch (kv_format) {
        case 0: return "bfloat16";
        case 1: return "int8-group64";
        case 2: return "fp8-e4m3-group64";
        case 3: return "nvfp4-group16";
        case 4: return "fp8-key-nvfp4-value";
        default: return "unknown";
    }
}

// Fixed 48-byte little-endian slot-header (de)serialization (see the struct's header comment).
bool serialize_slot_header(const detail::DiskKvSlotHeader& header, std::string& out) {
    out.clear();
    put_u32(out, header.magic);
    put_u64(out, header.lo);
    put_u64(out, header.hi);
    put_u64(out, header.tag);
    put_u32(out, header.frontier);
    put_u32(out, header.crc);
    put_u64(out, header.last_used);
    put_u32(out, header.payload_len);
    return out.size() == kDiskKvHeaderBytes;
}

bool parse_slot_header(const std::string& in, detail::DiskKvSlotHeader& out) {
    if (in.size() != kDiskKvHeaderBytes) { return false; }
    std::size_t offset = 0;
    if (!get_u32(in, offset, out.magic) || !get_u64(in, offset, out.lo) ||
        !get_u64(in, offset, out.hi) || !get_u64(in, offset, out.tag) ||
        !get_u32(in, offset, out.frontier) || !get_u32(in, offset, out.crc) ||
        !get_u64(in, offset, out.last_used) || !get_u32(in, offset, out.payload_len)) {
        return false;
    }
    return true;
}

} // namespace

std::uint32_t detail::disk_kv_crc32c(const std::byte* data, std::size_t size) noexcept {
    return crc32c_bytes(data, size);
}

struct DiskKVStore::Impl {
    std::filesystem::path dir;
    int data_fd = -1;
    std::string data_path;
    std::string index_path;
    std::string meta_path;
    std::uint32_t kv_format = 0;
    std::size_t slot_size        = 0;
    std::size_t slot_count       = 0;
    std::size_t payload_capacity = 0;
    std::size_t capacity_bytes   = 0;

    std::mutex mutex;
    std::string last_error;

    struct Entry {
        std::uint32_t slot      = 0;
        std::uint64_t last_used = 0;
    };
    using Lru = std::map<std::uint64_t, std::uint32_t>; // last_used -> slot, oldest first
    std::unordered_map<DiskKVIdentity, Entry, DiskKVIdentityHash> index;
    Lru lru;
    std::unordered_map<std::uint32_t, Lru::iterator> lru_pos; // slot -> Lru position
    std::vector<DiskKVIdentity> slot_identity;
    std::vector<bool> slot_in_use;
    std::vector<std::uint32_t> free_slots;
    std::set<DiskKVIdentity> protected_;
    std::uint64_t stamp = 0; // monotonic recency counter; zero means "never used"
    DiskKVStats stats;

    // ---- locked helpers (caller holds mutex) -------------------------------------------

    void free_slot(std::uint32_t slot) {
        const auto identity = slot_identity[slot];
        index.erase(identity);
        const auto position = lru_pos.find(slot);
        if (position != lru_pos.end()) {
            lru.erase(position->second);
            lru_pos.erase(position);
        }
        slot_in_use[slot] = false;
        free_slots.push_back(slot);
    }

    void move_lru(std::uint32_t slot, std::uint64_t new_last_used) {
        const auto position = lru_pos.find(slot);
        if (position != lru_pos.end()) { lru.erase(position->second); }
        lru_pos[slot] = lru.emplace(new_last_used, slot).first;
    }

    bool evict_oldest_locked() {
        for (const auto& [last_used, slot] : lru) {
            (void)last_used;
            if (protected_.count(slot_identity[slot]) != 0) { continue; }
            free_slot(slot);
            ++stats.evictions;
            return true;
        }
        return false;
    }

    bool flush_index_locked() {
        IndexRecord record;
        record.max_stamp = stamp;
        for (const auto& [identity, entry] : index) {
            record.entries.emplace_back(identity, entry.slot);
            record.last_used.push_back(entry.last_used);
        }
        std::string buf;
        if (!serialize_index(record, buf)) {
            last_error = "index serialization failed";
            return false;
        }
        if (!atomic_replace_file(index_path, buf)) {
            last_error = "index atomic replace failed: " + index_path;
            return false;
        }
        return true;
    }

    // Scans every slot header and keeps only entries whose magic and CRC verify.
    void refresh_index_locked() {
        ++stats.index_rebuilds;
        index.clear();
        lru.clear();
        lru_pos.clear();
        slot_identity.assign(slot_count, DiskKVIdentity{});
        slot_in_use.assign(slot_count, false);
        free_slots.clear();
        std::vector<std::byte> payload(payload_capacity);
        std::uint64_t max_last_used = 0;
        for (std::size_t slot = 0; slot < slot_count; ++slot) {
            const std::uint64_t base = slot * slot_size;
            std::string header_buf(kDiskKvHeaderBytes, '\0');
            if (::pread(data_fd, header_buf.data(), header_buf.size(), static_cast<off_t>(base)) !=
                static_cast<ssize_t>(header_buf.size())) {
                continue;
            }
            detail::DiskKvSlotHeader head;
            if (!parse_slot_header(header_buf, head)) { continue; }
            if (head.magic != detail::kDiskKvSlotMagic) { continue; }
            const DiskKVIdentity identity{head.lo, head.hi, head.tag, head.frontier};
            if (identity.empty() || head.payload_len > payload_capacity) { continue; }
            std::size_t payload_len = head.payload_len;
            if (payload_len > 0 &&
                ::pread(data_fd, payload.data(), payload_len, static_cast<off_t>(base + kDiskKvHeaderBytes)) !=
                    static_cast<ssize_t>(payload_len)) {
                continue;
            }
            if (detail::disk_kv_crc32c(payload.data(), payload_len) != head.crc) { continue; }
            // Recency stamps are unique in a healthy store; out-of-band damage can leave ties,
            // so bump the later slot past the earlier one (monotonicity is the invariant).
            std::uint64_t last_used = head.last_used;
            while (lru.count(last_used) != 0) { ++last_used; }
            index[identity] = Entry{static_cast<std::uint32_t>(slot), last_used};
            lru_pos[static_cast<std::uint32_t>(slot)] = lru.emplace(last_used, static_cast<std::uint32_t>(slot)).first;
            slot_identity[slot] = identity;
            slot_in_use[slot]   = true;
            max_last_used       = std::max(max_last_used, last_used);
        }
        for (std::size_t slot = 0; slot < slot_count; ++slot) {
            if (!slot_in_use[slot]) { free_slots.push_back(static_cast<std::uint32_t>(slot)); }
        }
        stamp = max_last_used;
        (void)flush_index_locked();
    }

    void adopt_index(const IndexRecord& record) {
        index.clear();
        lru.clear();
        lru_pos.clear();
        slot_identity.assign(slot_count, DiskKVIdentity{});
        slot_in_use.assign(slot_count, false);
        free_slots.clear();
        for (std::size_t i = 0; i < record.entries.size(); ++i) {
            const auto& [identity, slot] = record.entries[i];
            std::uint64_t last_used = record.last_used[i];
            while (lru.count(last_used) != 0) { ++last_used; }
            index[identity] = Entry{slot, last_used};
            lru_pos[slot]   = lru.emplace(last_used, slot).first;
            slot_identity[slot] = identity;
            slot_in_use[slot]   = true;
        }
        for (std::size_t slot = 0; slot < slot_count; ++slot) {
            if (!slot_in_use[slot]) { free_slots.push_back(static_cast<std::uint32_t>(slot)); }
        }
        stamp = record.max_stamp;
    }
};

DiskKVStore::DiskKVStore(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

DiskKVStore::~DiskKVStore() {
    {
        std::lock_guard lock(impl_->mutex);
        (void)impl_->flush_index_locked();
    }
    if (impl_->data_fd >= 0) { ::close(impl_->data_fd); }
}

DiskKVStore::OpenStatus
DiskKVStore::open(const std::filesystem::path& dir, std::uint32_t kv_format,
                  std::size_t max_payload_bytes, std::size_t capacity_bytes,
                  std::unique_ptr<DiskKVStore>& out) {
    auto fail = [&out](std::string message) -> OpenStatus {
        out.reset();
        return OpenStatus{false, std::move(message)};
    };
    if (dir.empty()) { return fail("disk KV directory is empty"); }
    if (capacity_bytes == 0) { return fail("disk KV capacity is zero"); }
    const std::size_t slot_size =
        (kDiskKvHeaderBytes + max_payload_bytes + detail::kDiskKvAlignment - 1) &
        ~(detail::kDiskKvAlignment - 1);
    if (slot_size <= kDiskKvHeaderBytes) {
        return fail("disk KV slot size underflows the header");
    }
    if (capacity_bytes < slot_size) {
        return fail("disk KV capacity below one slot (" + std::to_string(capacity_bytes) +
                    " < " + std::to_string(slot_size) + " bytes)");
    }
    const std::size_t slot_count = capacity_bytes / slot_size;

    auto impl = std::make_unique<Impl>();
    impl->dir            = dir;
    impl->kv_format      = kv_format;
    impl->slot_size      = slot_size;
    impl->slot_count     = slot_count;
    impl->payload_capacity = slot_size - kDiskKvHeaderBytes;
    impl->capacity_bytes = slot_count * slot_size;
    impl->data_path      = (dir / "ninfer_disk_kv.dat").string();
    impl->index_path     = (dir / "ninfer_disk_kv.idx").string();
    impl->meta_path      = (dir / "ninfer_disk_kv.meta").string();

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const bool data_exists = std::filesystem::exists(impl->data_path);
    if (data_exists) {
        if (!std::filesystem::exists(impl->meta_path)) {
            return fail("disk KV meta file is missing for the existing data file; refusing to "
                        "adopt a store of unknown KV format");
        }
        MetaRecord meta;
        if (!read_meta(impl->meta_path, meta)) {
            return fail("disk KV meta file is unreadable or corrupt");
        }
        if (meta.kv_format != kv_format) {
            return fail("disk KV format mismatch: store records " + format_name(meta.kv_format) +
                        ", engine requires " + format_name(kv_format) +
                        "; remove the store directory or match the KV format");
        }
        // slot_size is the payload class (a function of KV format and max payload); a different
        // size means the data cannot be mixed, so it stays a hard fail. slot_count is pure
        // capacity and is adopted below (resize + meta rewrite), never an error.
        if (meta.slot_size != slot_size) {
            return fail("disk KV slot size mismatch: store records " +
                        std::to_string(meta.slot_size) + ", this open requires " +
                        std::to_string(slot_size) +
                        "; remove the store directory or match the payload class");
        }
        const bool adopt_capacity = meta.slot_count != slot_count;
        impl->data_fd = ::open(impl->data_path.c_str(), O_RDWR);
        if (impl->data_fd < 0) { return fail("cannot open disk KV data file: " + impl->data_path); }
        if (adopt_capacity) {
            // Capacity changed since the store was written: adopt it. Expanding sparse-extends the
            // data file (the new region reads as zero => empty slot headers); shrinking truncates
            // it. Both cases persist the new slot_count. The index below then reconciles: a
            // shrunken index is rejected by parse_index (entry_count > slot_count, or a slot >=
            // slot_count) and rebuilt by the existing slow path, which prunes out-of-bounds entries
            // and persists the result atomically; an expanded index is adopted on the fast path and
            // the new empty slots enter the free set. No new index-side code is needed.
            if (::ftruncate(impl->data_fd, static_cast<off_t>(impl->capacity_bytes)) != 0) {
                ::close(impl->data_fd);
                impl->data_fd = -1;
                return fail("cannot resize disk KV data file to the requested capacity");
            }
            if (!write_meta(impl->meta_path, kv_format, slot_size, slot_count)) {
                ::close(impl->data_fd);
                impl->data_fd = -1;
                return fail("cannot rewrite disk KV meta file");
            }
        } else {
            // Same capacity: keep the strict size check (the p5 corruption-detection contract is
            // unchanged); the size was not just established by ftruncate.
            struct stat st;
            if (::fstat(impl->data_fd, &st) != 0 ||
                static_cast<std::size_t>(st.st_size) != impl->capacity_bytes) {
                ::close(impl->data_fd);
                impl->data_fd = -1;
                return fail("disk KV data file size does not match the recorded geometry");
            }
        }
    } else {
        impl->data_fd = ::open(impl->data_path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0644);
        if (impl->data_fd < 0) { return fail("cannot create disk KV data file: " + impl->data_path); }
        if (::ftruncate(impl->data_fd, static_cast<off_t>(impl->capacity_bytes)) != 0) {
            ::close(impl->data_fd);
            impl->data_fd = -1;
            return fail("cannot size disk KV data file");
        }
        if (!write_meta(impl->meta_path, kv_format, slot_size, slot_count)) {
            ::close(impl->data_fd);
            impl->data_fd = -1;
            return fail("cannot write disk KV meta file");
        }
    }

    {
        std::lock_guard lock(impl->mutex);
        IndexRecord record;
        std::string buf;
        if (read_whole_file(impl->index_path, buf) && parse_index(buf, impl->slot_count, record)) {
            impl->adopt_index(record); // fast path: the index is complete and structurally valid
        } else {
            // Slow one-shot path: the index is missing or corrupt; rebuild it from the
            // self-describing slot headers (magic + CRC gate) and persist it atomically.
            impl->refresh_index_locked();
        }
    }
    out = std::unique_ptr<DiskKVStore>(new DiskKVStore(std::move(impl)));
    return OpenStatus{true, {}};
}

bool DiskKVStore::write(const DiskKVIdentity& identity, std::span<const std::byte> data) {
    std::lock_guard lock(impl_->mutex);
    if (identity.empty()) {
        ++impl_->stats.write_failures;
        impl_->last_error = "disk KV identity is empty";
        return false;
    }
    if (data.size() > impl_->payload_capacity) {
        ++impl_->stats.write_failures;
        impl_->last_error = "disk KV payload of " + std::to_string(data.size()) +
                            " bytes exceeds the slot capacity of " +
                            std::to_string(impl_->payload_capacity) + " bytes";
        return false;
    }
    std::uint32_t slot = 0;
    const auto existing = impl_->index.find(identity);
    if (existing != impl_->index.end()) {
        slot = existing->second.slot;
    } else {
        if (impl_->free_slots.empty() && !impl_->evict_oldest_locked()) {
            ++impl_->stats.write_failures;
            impl_->last_error = "disk KV store is full and every in-use slot is protected";
            return false;
        }
        slot = impl_->free_slots.back();
        impl_->free_slots.pop_back();
    }
    ++impl_->stamp;
    detail::DiskKvSlotHeader header{};
    header.magic       = detail::kDiskKvSlotMagic;
    header.lo          = identity.lo;
    header.hi          = identity.hi;
    header.tag         = identity.tag;
    header.frontier    = identity.frontier;
    header.crc         = detail::disk_kv_crc32c(data.data(), data.size());
    header.last_used   = impl_->stamp;
    header.payload_len = static_cast<std::uint32_t>(data.size());
    std::string header_buf;
    if (!serialize_slot_header(header, header_buf)) {
        ++impl_->stats.write_failures;
        impl_->last_error = "slot header serialization failed";
        return false;
    }
    const std::uint64_t base = static_cast<std::uint64_t>(slot) * impl_->slot_size;
    if (::pwrite(impl_->data_fd, header_buf.data(), header_buf.size(), static_cast<off_t>(base)) !=
        static_cast<ssize_t>(header_buf.size())) {
        impl_->free_slot(slot);
        ++impl_->stats.write_failures;
        impl_->last_error = "disk KV header write failed";
        return false;
    }
    if (data.size() > 0 &&
        ::pwrite(impl_->data_fd, data.data(), data.size(),
                 static_cast<off_t>(base + kDiskKvHeaderBytes)) != static_cast<ssize_t>(data.size())) {
        impl_->free_slot(slot);
        ++impl_->stats.write_failures;
        impl_->last_error = "disk KV payload write failed";
        return false;
    }
    impl_->move_lru(slot, impl_->stamp);
    impl_->slot_identity[slot] = identity;
    impl_->slot_in_use[slot]   = true;
    impl_->index[identity]     = Impl::Entry{slot, impl_->stamp};
    ++impl_->stats.writes;
    impl_->stats.bytes_written += data.size();
    (void)impl_->flush_index_locked();
    return true;
}

bool DiskKVStore::read(const DiskKVIdentity& identity, std::span<std::byte> out) {
    std::lock_guard lock(impl_->mutex);
    const auto existing = impl_->index.find(identity);
    if (existing == impl_->index.end()) {
        ++impl_->stats.read_failures;
        impl_->last_error = "disk KV identity is not resident";
        return false;
    }
    const std::uint32_t slot = existing->second.slot;
    const std::uint64_t base = static_cast<std::uint64_t>(slot) * impl_->slot_size;
    std::string header_buf(kDiskKvHeaderBytes, '\0');
    if (::pread(impl_->data_fd, header_buf.data(), header_buf.size(), static_cast<off_t>(base)) !=
        static_cast<ssize_t>(header_buf.size())) {
        impl_->free_slot(slot);
        (void)impl_->flush_index_locked();
        ++impl_->stats.read_failures;
        impl_->last_error = "disk KV header read failed";
        return false;
    }
    detail::DiskKvSlotHeader head;
    if (!parse_slot_header(header_buf, head) || head.magic != detail::kDiskKvSlotMagic ||
        head.lo != identity.lo || head.hi != identity.hi || head.tag != identity.tag ||
        head.frontier != identity.frontier) {
        // Torn header or foreign content: report as missing, drop only this slot, keep every
        // other identity.
        impl_->free_slot(slot);
        (void)impl_->flush_index_locked();
        ++impl_->stats.read_failures;
        ++impl_->stats.identity_mismatches;
        impl_->last_error = "disk KV slot failed identity verification";
        return false;
    }
    if (head.payload_len > impl_->payload_capacity || head.payload_len > out.size()) {
        impl_->free_slot(slot);
        (void)impl_->flush_index_locked();
        ++impl_->stats.read_failures;
        impl_->last_error = "disk KV payload length is inconsistent";
        return false;
    }
    const std::size_t payload_len = head.payload_len;
    if (payload_len > 0 &&
        ::pread(impl_->data_fd, out.data(), payload_len,
                static_cast<off_t>(base + kDiskKvHeaderBytes)) != static_cast<ssize_t>(payload_len)) {
        impl_->free_slot(slot);
        (void)impl_->flush_index_locked();
        ++impl_->stats.read_failures;
        impl_->last_error = "disk KV payload read failed";
        return false;
    }
    if (detail::disk_kv_crc32c(out.data(), payload_len) != head.crc) {
        impl_->free_slot(slot);
        (void)impl_->flush_index_locked();
        ++impl_->stats.read_failures;
        ++impl_->stats.crc_mismatches;
        impl_->last_error = "disk KV payload CRC mismatch";
        return false;
    }
    // A successful read refreshes the recency stamp (in-memory; it persists with the next index
    // flush, and the rebuild path falls back to the slot's persisted stamp).
    ++impl_->stamp;
    impl_->move_lru(slot, impl_->stamp);
    existing->second.last_used = impl_->stamp;
    ++impl_->stats.reads;
    impl_->stats.bytes_read += payload_len;
    return true;
}

bool DiskKVStore::contains(const DiskKVIdentity& identity) const {
    std::lock_guard lock(impl_->mutex);
    return impl_->index.count(identity) != 0;
}

void DiskKVStore::touch(const DiskKVIdentity& identity) {
    std::lock_guard lock(impl_->mutex);
    auto existing = impl_->index.find(identity);
    if (existing == impl_->index.end()) { return; }
    ++impl_->stamp;
    impl_->move_lru(existing->second.slot, impl_->stamp);
    existing->second.last_used = impl_->stamp;
}

bool DiskKVStore::evict_oldest() {
    std::lock_guard lock(impl_->mutex);
    if (!impl_->evict_oldest_locked()) {
        impl_->last_error = "no evictable disk KV slot";
        return false;
    }
    (void)impl_->flush_index_locked();
    return true;
}

bool DiskKVStore::make_room(std::size_t slots) {
    std::lock_guard lock(impl_->mutex);
    for (;;) {
        if (impl_->free_slots.size() >= slots) { return true; }
        if (!impl_->evict_oldest_locked()) {
            impl_->last_error = "cannot free disk KV slots: in-use slots are protected";
            return false;
        }
    }
}

void DiskKVStore::protect(const DiskKVIdentity& identity) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->index.count(identity) != 0) { impl_->protected_.insert(identity); }
}

void DiskKVStore::unprotect(const DiskKVIdentity& identity) {
    std::lock_guard lock(impl_->mutex);
    impl_->protected_.erase(identity);
}

std::vector<DiskKVIdentity> DiskKVStore::active_identities() const {
    std::lock_guard lock(impl_->mutex);
    std::vector<DiskKVIdentity> identities;
    identities.reserve(impl_->index.size());
    for (const auto& [identity, entry] : impl_->index) { (void)entry; identities.push_back(identity); }
    return identities;
}

bool DiskKVStore::erase(const DiskKVIdentity& identity) {
    std::lock_guard lock(impl_->mutex);
    const auto existing = impl_->index.find(identity);
    if (existing == impl_->index.end()) {
        impl_->last_error = "disk KV identity is not resident";
        return false;
    }
    impl_->index.erase(existing);
    impl_->free_slot(existing->second.slot);
    impl_->protected_.erase(identity);
    (void)impl_->flush_index_locked();
    return true;
}

void DiskKVStore::refresh_index() {
    std::lock_guard lock(impl_->mutex);
    impl_->refresh_index_locked();
}

void DiskKVStore::flush_index() {
    std::lock_guard lock(impl_->mutex);
    (void)impl_->flush_index_locked();
}

std::size_t DiskKVStore::capacity_bytes() const noexcept { return impl_->capacity_bytes; }
std::size_t DiskKVStore::used_bytes() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->index.size() * impl_->slot_size;
}
std::size_t DiskKVStore::active_pages() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->index.size();
}
std::size_t DiskKVStore::slot_size() const noexcept { return impl_->slot_size; }
std::size_t DiskKVStore::slot_count() const noexcept { return impl_->slot_count; }
std::uint32_t DiskKVStore::kv_format() const noexcept { return impl_->kv_format; }
const std::string& DiskKVStore::last_error() const noexcept { return impl_->last_error; }
DiskKVStats DiskKVStore::stats() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->stats;
}

} // namespace ninfer