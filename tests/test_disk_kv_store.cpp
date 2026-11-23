// Independent unit tests for the self-contained disk KV store (src/core/disk_kv_store.{h,cpp}).
// The module has no CUDA and no engine-type dependencies; this test compiles the module source
// directly and links no library, which proves that property.

#include "core/disk_kv_store.h"
#include "core/disk_kv_digest.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            ++failures;                                                        \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #condition \
                      << std::endl;                                            \
        }                                                                      \
    } while (0)

using ninfer::DiskKVIdentity;
using ninfer::DiskKVStore;

DiskKVIdentity make_id(std::uint64_t lo, std::uint32_t frontier = 1) {
    return DiskKVIdentity{lo, lo ^ 0x9E3779B97F4A7C15ULL, 42, frontier};
}

std::vector<std::byte> make_payload(std::uint64_t seed, std::size_t size) {
    std::vector<std::byte> payload(size);
    std::uint64_t state = seed == 0 ? 0x8E3C297D43B2B2B7ULL : seed;
    for (auto& byte : payload) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        byte = static_cast<std::byte>(state & 0xFFu);
    }
    return payload;
}

std::filesystem::path test_dir(const char* name) {
    const auto dir =
        std::filesystem::temp_directory_path() /
        ("ninfer_disk_kv_test_" + std::to_string(::getpid()) + "_" + name);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

std::filesystem::path data_file(const std::filesystem::path& dir) { return dir / "ninfer_disk_kv.dat"; }
std::filesystem::path index_file(const std::filesystem::path& dir) { return dir / "ninfer_disk_kv.idx"; }
std::filesystem::path meta_file(const std::filesystem::path& dir) { return dir / "ninfer_disk_kv.meta"; }

// Reads the 24-byte meta file (u32 magic | u32 kv_format | u64 slot_size | u64 slot_count,
// little-endian) so capacity-adoption tests can assert the persisted geometry.
struct MetaInfo {
    std::uint32_t kv_format = 0;
    std::size_t slot_size   = 0;
    std::size_t slot_count  = 0;
    bool ok                 = false;
};

MetaInfo read_meta_file(const std::filesystem::path& dir) {
    MetaInfo info;
    std::ifstream in(meta_file(dir), std::ios::binary);
    unsigned char buf[24];
    if (in.read(reinterpret_cast<char*>(buf), sizeof(buf)) &&
        in.gcount() == static_cast<std::streamsize>(sizeof(buf))) {
        const auto u32 = [&](int off) {
            return std::uint32_t(buf[off]) | std::uint32_t(buf[off + 1]) << 8 |
                   std::uint32_t(buf[off + 2]) << 16 | std::uint32_t(buf[off + 3]) << 24;
        };
        const auto u64 = [&](int off) {
            std::uint64_t v = 0;
            for (int i = 7; i >= 0; --i) { v = (v << 8) | buf[off + i]; }
            return v;
        };
        info.kv_format = u32(4);
        info.slot_size  = u64(8);
        info.slot_count = u64(16);
        info.ok         = true;
    }
    return info;
}

std::unique_ptr<DiskKVStore> open_store(const std::filesystem::path& dir, std::uint32_t format,
                                        std::size_t payload, std::size_t capacity, std::string& error) {
    std::unique_ptr<DiskKVStore> store;
    const auto status = DiskKVStore::open(dir, format, payload, capacity, store);
    error = status.error;
    return status.ok ? std::move(store) : nullptr;
}

void test_round_trip() {
    const auto dir = test_dir("round_trip");
    std::string error;
    auto store = open_store(dir, 1, 100 * 1024, 4 * 1024 * 1024, error);
    CHECK(store != nullptr);
    if (store == nullptr) { std::cerr << "  open: " << error << std::endl; return; }
    CHECK(store->slot_size() % 4096 == 0);
    CHECK(store->slot_size() >= 48 + 100 * 1024);

    std::vector<DiskKVIdentity> ids = {make_id(1), make_id(2, 65), make_id(3, 128)};
    std::vector<std::vector<std::byte>> payloads;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        payloads.push_back(make_payload(i + 1, 100 * 1024 + static_cast<std::size_t>(i)));
        CHECK(store->write(ids[i], {payloads[i].data(), payloads[i].size()}));
    }
    for (std::size_t i = 0; i < ids.size(); ++i) {
        CHECK(store->contains(ids[i]));
        auto readback = payloads[i];
        CHECK(store->read(ids[i], {readback.data(), readback.size()}));
        CHECK(readback == payloads[i]);
    }
    CHECK(!store->contains(make_id(99)));
}

void test_crc_mismatch() {
    const auto dir = test_dir("crc");
    std::string error;
    const std::size_t payload = 64 * 1024;
    const std::size_t slot = (48 + payload + 4095) & ~std::size_t(4095);
    // Exactly two slots: slots are taken from the end, so id_a -> slot 1, id_b -> slot 0.
    auto store = open_store(dir, 1, payload, 2 * slot, error);
    CHECK(store != nullptr);
    if (store == nullptr) { return; }
    const auto id_a = make_id(1);
    const auto id_b = make_id(2);
    auto payload_a = make_payload(1, payload);
    auto payload_b = make_payload(2, payload);
    CHECK(store->write(id_a, {payload_a.data(), payload_a.size()}));
    CHECK(store->write(id_b, {payload_b.data(), payload_b.size()}));

    // Corrupt one payload byte of slot 0 (id_b) in the data file, out of band.
    int fd = ::open(data_file(dir).c_str(), O_RDWR);
    CHECK(fd >= 0);
    const std::uint64_t offset = 48 + 100; // into slot 0's payload
    const std::byte garbage = static_cast<std::byte>(0xA5);
    CHECK(::pwrite(fd, &garbage, 1, static_cast<off_t>(offset)) == 1);
    ::close(fd);

    auto read_b = payload_b;
    CHECK(!store->read(id_b, {read_b.data(), read_b.size()}));
    CHECK(!store->contains(id_b));
    // The other identity survives the single-page fault.
    auto read_a = payload_a;
    CHECK(store->read(id_a, {read_a.data(), read_a.size()}));
    CHECK(read_a == payload_a);
}

void test_identity_mismatch() {
    const auto dir = test_dir("identity");
    std::string error;
    const std::size_t payload = 64 * 1024;
    const std::size_t slot = (48 + payload + 4095) & ~std::size_t(4095);
    // Exactly one slot, so the identity lives in slot 0.
    auto store = open_store(dir, 1, payload, slot, error);
    CHECK(store != nullptr);
    if (store == nullptr) { return; }
    const auto id_a = make_id(7);
    auto bytes = make_payload(7, payload);
    CHECK(store->write(id_a, {bytes.data(), bytes.size()}));

    // Tamper with the slot header's digest: the recorded identity no longer matches the request.
    int fd = ::open(data_file(dir).c_str(), O_RDWR);
    CHECK(fd >= 0);
    // `lo` is the 4th field of the 48-byte header (offset 4).
    const std::uint64_t tampered_lo = id_a.lo + 1;
    CHECK(::pwrite(fd, &tampered_lo, sizeof(tampered_lo), 4) == static_cast<ssize_t>(sizeof(tampered_lo)));
    ::close(fd);

    auto readback = bytes;
    CHECK(!store->read(id_a, {readback.data(), readback.size()}));
    CHECK(!store->contains(id_a));
}

void test_index_rebuild() {
    const auto dir = test_dir("rebuild");
    std::string error;
    auto store = open_store(dir, 2, 64 * 1024, 4 * 1024 * 1024, error);
    CHECK(store != nullptr);
    if (store == nullptr) { return; }
    constexpr std::size_t kCount = 20;
    std::vector<DiskKVIdentity> ids;
    std::vector<std::vector<std::byte>> payloads;
    for (std::size_t i = 0; i < kCount; ++i) {
        ids.push_back(make_id(i + 1, static_cast<std::uint32_t>(i + 1)));
        payloads.push_back(make_payload(i + 1, 64 * 1024));
        CHECK(store->write(ids[i], {payloads[i].data(), payloads[i].size()}));
    }
    store.reset(); // close (flushes the index)

    // Missing index: reopening must rebuild it from the slot headers.
    std::error_code ec;
    std::filesystem::remove(index_file(dir), ec);
    store = open_store(dir, 2, 64 * 1024, 4 * 1024 * 1024, error);
    CHECK(store != nullptr);
    if (store == nullptr) { std::cerr << "  reopen: " << error << std::endl; return; }
    CHECK(store->active_pages() == kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        auto readback = payloads[i];
        CHECK(store->read(ids[i], {readback.data(), readback.size()}));
        CHECK(readback == payloads[i]);
    }
    store.reset();

    // Corrupt index (truncated): the rebuild path must recover the same content.
    {
        int fd = ::open(index_file(dir).c_str(), O_RDWR);
        CHECK(fd >= 0);
        CHECK(::ftruncate(fd, 64) == 0);
        ::close(fd);
    }
    store = open_store(dir, 2, 64 * 1024, 4 * 1024 * 1024, error);
    CHECK(store != nullptr);
    if (store == nullptr) { return; }
    CHECK(store->active_pages() == kCount);
    auto readback = payloads[5];
    CHECK(store->read(ids[5], {readback.data(), readback.size()}));
    CHECK(readback == payloads[5]);
}

void test_lru_order() {
    const auto dir = test_dir("lru");
    std::string error;
    // Capacity sized to exactly three slots: payload 64 KiB -> slot 65560? No:
    // slot = align4k(48 + 65536) = 69632; capacity 3 * 69632.
    const std::size_t payload = 64 * 1024;
    const std::size_t slot = (48 + payload + 4095) & ~std::size_t(4095);
    auto store = open_store(dir, 1, payload, 3 * slot, error);
    CHECK(store != nullptr);
    if (store == nullptr) { std::cerr << "  open: " << error << std::endl; return; }
    CHECK(store->slot_count() == 3);

    const auto a = make_id(1);
    const auto b = make_id(2);
    const auto c = make_id(3);
    CHECK(store->write(a, make_payload(1, payload)));
    CHECK(store->write(b, make_payload(2, payload)));
    CHECK(store->write(c, make_payload(3, payload)));

    // Stamps after the three writes: a=1, b=2, c=3. Refresh `a` -> a=4; the LRU order is
    // b < c < a, so the eviction victim is `b`.
    store->touch(a);
    CHECK(store->evict_oldest());
    CHECK(!store->contains(b));
    CHECK(store->contains(a));
    CHECK(store->contains(c));

    // A read also refreshes recency: c=5 now beats a=4.
    auto read_c = make_payload(3, payload);
    CHECK(store->read(c, {read_c.data(), read_c.size()}));
    const auto d = make_id(4);
    CHECK(store->write(d, make_payload(4, payload))); // fills the last free slot: a=4, c=5, d=6
    CHECK(store->contains(a));
    const auto e = make_id(5);
    CHECK(store->write(e, make_payload(5, payload))); // must evict `a` (oldest), not `c`
    CHECK(!store->contains(a));
    CHECK(store->contains(c));
    CHECK(store->contains(d));
    CHECK(store->contains(e));
}

void test_make_room_and_full_failure() {
    const auto dir = test_dir("full");
    std::string error;
    const std::size_t payload = 64 * 1024;
    const std::size_t slot = (48 + payload + 4095) & ~std::size_t(4095);
    auto store = open_store(dir, 1, payload, slot, error); // exactly one slot
    CHECK(store != nullptr);
    if (store == nullptr) { return; }
    const auto a = make_id(1);
    const auto b = make_id(2);
    CHECK(store->write(a, make_payload(1, payload)));
    CHECK(store->write(b, make_payload(2, payload))); // evicts `a`
    CHECK(!store->contains(a));
    CHECK(store->contains(b));

    // A protected in-use slot makes the store full: writes must fail, not overwrite.
    store->protect(b);
    const auto c = make_id(3);
    CHECK(!store->write(c, make_payload(3, payload)));
    CHECK(store->contains(b));
    store->unprotect(b);
    CHECK(store->write(c, make_payload(3, payload)));
    CHECK(!store->contains(b));
    CHECK(store->contains(c));

    // make_room evicts down to the requested free-slot count.
    CHECK(store->make_room(1));
    CHECK(store->active_pages() == 0);

    // Erasing a whole branch frees its slot without touching the rest.
    CHECK(store->write(a, make_payload(1, payload)));
    CHECK(store->write(b, make_payload(2, payload))); // evicts `a`
    CHECK(store->erase(b));
    CHECK(!store->contains(b));
    CHECK(!store->erase(b));
}

void test_format_mismatch() {
    const auto dir = test_dir("format");
    std::string error;
    auto store = open_store(dir, 1, 64 * 1024, 2 * 1024 * 1024, error);
    CHECK(store != nullptr);
    if (store == nullptr) { return; }
    store.reset();
    std::unique_ptr<DiskKVStore> wrong;
    const auto status = DiskKVStore::open(dir, 3, 64 * 1024, 2 * 1024 * 1024, wrong);
    CHECK(!status.ok);
    CHECK(wrong == nullptr);
    CHECK(status.error.find("format mismatch") != std::string::npos);
    // The matching format still opens the same store.
    store = open_store(dir, 1, 64 * 1024, 2 * 1024 * 1024, error);
    CHECK(store != nullptr);
}

void test_capacity_adopt_expand() {
    // Reopening with a larger capacity adopts the store: the meta is rewritten, the old entry
    // stays readable, and the new empty slots enter the free set.
    const auto dir = test_dir("adopt_expand");
    std::string error;
    const std::size_t payload = 64 * 1024;
    const std::size_t slot = (48 + payload + 4095) & ~std::size_t(4095);
    auto store = open_store(dir, 1, payload, 2 * slot, error);
    CHECK(store != nullptr);
    if (store == nullptr) { std::cerr << "  open: " << error << std::endl; return; }
    const auto id = make_id(1);
    auto bytes = make_payload(1, payload);
    CHECK(store->write(id, {bytes.data(), bytes.size()}));
    store.reset(); // close (flushes the index)

    store = open_store(dir, 1, payload, 8 * slot, error);
    CHECK(store != nullptr);
    if (store == nullptr) { std::cerr << "  expand: " << error << std::endl; return; }
    CHECK(store->slot_count() == 8);
    const auto meta = read_meta_file(dir);
    CHECK(meta.ok);
    CHECK(meta.slot_count == 8);
    auto readback = bytes;
    CHECK(store->read(id, {readback.data(), readback.size()}));
    CHECK(readback == bytes);
    // The six new slots are writable and fill the store to capacity.
    for (std::uint32_t i = 2; i <= 7; ++i) {
        const auto nid = make_id(i);
        CHECK(store->write(nid, make_payload(i, payload)));
        CHECK(store->contains(nid));
    }
    CHECK(store->active_pages() == 7);
    store.reset();

    // Reopening at the now-persisted capacity is a no-op.
    store = open_store(dir, 1, payload, 8 * slot, error);
    CHECK(store != nullptr);
    if (store == nullptr) { return; }
    CHECK(store->slot_count() == 8);
    CHECK(store->active_pages() == 7);
    store.reset();
}

void test_capacity_adopt_shrink() {
    // Reopening with a smaller capacity adopts the store: out-of-bounds slots are pruned by the
    // existing index-rebuild slow path, in-capacity entries survive, and the data file is
    // truncated to the new capacity with the meta rewritten.
    const auto dir = test_dir("adopt_shrink");
    std::string error;
    const std::size_t payload = 64 * 1024;
    const std::size_t slot = (48 + payload + 4095) & ~std::size_t(4095);
    auto store = open_store(dir, 1, payload, 4 * slot, error);
    CHECK(store != nullptr);
    if (store == nullptr) { std::cerr << "  open: " << error << std::endl; return; }
    // Fill all four slots. Slots are taken from the end, so the write order maps to
    // id1 -> slot 3, id2 -> slot 2, id3 -> slot 1, id4 -> slot 0.
    std::vector<std::vector<std::byte>> payloads;
    for (std::uint32_t i = 1; i <= 4; ++i) {
        payloads.push_back(make_payload(i, payload));
        CHECK(store->write(make_id(i), {payloads[i - 1].data(), payloads[i - 1].size()}));
    }
    store.reset();

    store = open_store(dir, 1, payload, 2 * slot, error);
    CHECK(store != nullptr);
    if (store == nullptr) { std::cerr << "  shrink: " << error << std::endl; return; }
    CHECK(store->slot_count() == 2);
    CHECK(store->active_pages() == 2);
    const auto meta = read_meta_file(dir);
    CHECK(meta.ok);
    CHECK(meta.slot_count == 2);
    CHECK(std::filesystem::file_size(data_file(dir)) == 2 * slot);
    // The in-capacity entries (slots 0 and 1) survive and are readable.
    auto read3 = payloads[2];
    CHECK(store->read(make_id(3), {read3.data(), read3.size()}));
    CHECK(read3 == payloads[2]);
    auto read4 = payloads[3];
    CHECK(store->read(make_id(4), {read4.data(), read4.size()}));
    CHECK(read4 == payloads[3]);
    // The out-of-capacity entries (slots 2 and 3) are gone.
    CHECK(!store->contains(make_id(1)));
    CHECK(!store->contains(make_id(2)));
    auto read1 = payloads[0];
    CHECK(!store->read(make_id(1), {read1.data(), read1.size()}));
    store.reset();
}

void test_capacity_adopt_same_capacity() {
    // Reopening at the same capacity is a no-op: the strict size check still runs (fast path),
    // the meta and data file are untouched, and the entries are unchanged.
    const auto dir = test_dir("adopt_same");
    std::string error;
    const std::size_t payload = 64 * 1024;
    const std::size_t slot = (48 + payload + 4095) & ~std::size_t(4095);
    auto store = open_store(dir, 1, payload, 2 * slot, error);
    CHECK(store != nullptr);
    if (store == nullptr) { std::cerr << "  open: " << error << std::endl; return; }
    const auto id = make_id(1);
    auto bytes = make_payload(1, payload);
    CHECK(store->write(id, {bytes.data(), bytes.size()}));
    const auto size_before = std::filesystem::file_size(data_file(dir));
    const auto meta_before = read_meta_file(dir);
    store.reset();

    store = open_store(dir, 1, payload, 2 * slot, error);
    CHECK(store != nullptr);
    if (store == nullptr) { std::cerr << "  same: " << error << std::endl; return; }
    CHECK(store->slot_count() == 2);
    CHECK(store->stats().index_rebuilds == 0); // fast path, no slot scan
    CHECK(std::filesystem::file_size(data_file(dir)) == size_before);
    const auto meta_after = read_meta_file(dir);
    CHECK(meta_after.slot_count == meta_before.slot_count);
    auto readback = bytes;
    CHECK(store->read(id, {readback.data(), readback.size()}));
    CHECK(readback == bytes);
    store.reset();
}

void test_capacity_adopt_slot_size_mismatch() {
    // A different max payload means a different slot size (a different payload class): the open
    // must hard-fail and leave the store's files untouched (no truncation, no meta rewrite).
    const auto dir = test_dir("adopt_slot_size");
    std::string error;
    const std::size_t payload = 64 * 1024;
    auto store = open_store(dir, 1, payload, 2 * 1024 * 1024, error);
    CHECK(store != nullptr);
    if (store == nullptr) { std::cerr << "  open: " << error << std::endl; return; }
    const auto id = make_id(1);
    auto bytes = make_payload(1, payload);
    CHECK(store->write(id, {bytes.data(), bytes.size()}));
    const auto size_before = std::filesystem::file_size(data_file(dir));
    const auto meta_before = read_meta_file(dir);
    store.reset();

    std::unique_ptr<DiskKVStore> wrong;
    const auto status = DiskKVStore::open(dir, 1, 128 * 1024, 2 * 1024 * 1024, wrong);
    CHECK(!status.ok);
    CHECK(wrong == nullptr);
    CHECK(status.error.find("slot size mismatch") != std::string::npos);
    CHECK(std::filesystem::file_size(data_file(dir)) == size_before);
    const auto meta_after = read_meta_file(dir);
    CHECK(meta_after.slot_count == meta_before.slot_count);
    CHECK(meta_after.slot_size == meta_before.slot_size);
    // The matching payload class still opens the same store.
    store = open_store(dir, 1, payload, 2 * 1024 * 1024, error);
    CHECK(store != nullptr);
    if (store == nullptr) { return; }
    auto readback = bytes;
    CHECK(store->read(id, {readback.data(), readback.size()}));
    CHECK(readback == bytes);
}

void test_index_reader_sees_complete_files() {
    // The index is replaced through tmp + rename, so a concurrent reader must only ever observe
    // a complete old or complete new index, never a partial write.
    const auto dir = test_dir("atomic");
    std::string error;
    const std::size_t payload = 64 * 1024;
    const std::size_t slot = (48 + payload + 4095) & ~std::size_t(4095);
    auto store = open_store(dir, 1, payload, 16 * slot, error);
    CHECK(store != nullptr);
    if (store == nullptr) { return; }

    std::atomic<bool> stop{false};
    std::atomic<bool> saw_torn{false};
    std::thread reader([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            std::string buf;
            int fd = ::open(index_file(dir).c_str(), O_RDONLY);
            if (fd < 0) { continue; }
            char chunk[65536];
            for (;;) {
                const ssize_t n = ::read(fd, chunk, sizeof(chunk));
                if (n < 0) {
                    if (errno == EINTR) { continue; }
                    break;
                }
                if (n == 0) { break; }
                buf.append(chunk, static_cast<std::size_t>(n));
            }
            ::close(fd);
            if (buf.size() < 12) { saw_torn.store(true); continue; }
            const auto* base = reinterpret_cast<const std::uint8_t*>(buf.data());
            const std::uint32_t magic = base[0] | (base[1] << 8) | (base[2] << 16) | (base[3] << 24);
            if (magic != 0x4944564B) { saw_torn.store(true); continue; }
            const std::uint64_t entry_count =
                (std::uint64_t)base[8] | (std::uint64_t)base[9] << 8 | (std::uint64_t)base[10] << 16 |
                (std::uint64_t)base[11] << 24 | (std::uint64_t)base[12] << 32 |
                (std::uint64_t)base[13] << 40 | (std::uint64_t)base[14] << 48 |
                (std::uint64_t)base[15] << 56;
            if (buf.size() != 24 + entry_count * 40) { saw_torn.store(true); }
        }
    });
    for (std::uint64_t i = 1; i <= 400; ++i) {
        const auto id = DiskKVIdentity{i, i ^ 0xDEAD, 42, static_cast<std::uint32_t>(i % 8 + 1)};
        (void)store->write(id, make_payload(i, payload));
    }
    stop.store(true);
    reader.join();
    CHECK(!saw_torn.load());
}

void test_reopen_fast_and_rebuild_paths() {
    // Reopening with the index file present takes the fast path (no slot scan); deleting the
    // index forces the rebuild path. The per-store stats counter distinguishes the two.
    const auto dir = test_dir("reopen_paths");
    std::string error;
    const std::size_t payload = 64 * 1024;
    auto store = open_store(dir, 1, payload, 4 * 1024 * 1024, error);
    CHECK(store != nullptr);
    if (store == nullptr) { return; }
    const auto id = make_id(11);
    auto bytes = make_payload(11, payload);
    CHECK(store->write(id, {bytes.data(), bytes.size()}));
    store.reset(); // closes and flushes the index

    // Fast path: the index file is intact, so no slot scan happens on open.
    store = open_store(dir, 1, payload, 4 * 1024 * 1024, error);
    CHECK(store != nullptr);
    if (store == nullptr) { std::cerr << "  reopen: " << error << std::endl; return; }
    CHECK(store->stats().index_rebuilds == 0);
    auto readback = bytes;
    CHECK(store->read(id, {readback.data(), readback.size()}));
    CHECK(readback == bytes);
    store.reset();

    // Rebuild path: the index file is gone, so open must rescan the slot headers.
    std::error_code ec;
    std::filesystem::remove(index_file(dir), ec);
    store = open_store(dir, 1, payload, 4 * 1024 * 1024, error);
    CHECK(store != nullptr);
    if (store == nullptr) { return; }
    CHECK(store->stats().index_rebuilds == 1);
    readback = bytes;
    CHECK(store->read(id, {readback.data(), readback.size()}));
    CHECK(readback == bytes);
}

void test_oversized_payload() {
    // A payload larger than the slot's payload capacity is rejected; nothing is written and the
    // identity is not resident.
    const auto dir = test_dir("oversized");
    std::string error;
    auto store = open_store(dir, 1, 64 * 1024, 2 * 1024 * 1024, error);
    CHECK(store != nullptr);
    if (store == nullptr) { return; }
    const auto id = make_id(21);
    auto too_big = make_payload(21, 64 * 1024 + 4096 + 48 + 1); // one byte past the slot capacity
    CHECK(!store->write(id, {too_big.data(), too_big.size()}));
    CHECK(!store->contains(id));
    CHECK(store->stats().write_failures == 1);
    // A payload that exactly fills the slot capacity is accepted.
    auto exact = make_payload(22, store->slot_size() - 48);
    CHECK(store->write(id, {exact.data(), exact.size()}));
    CHECK(store->contains(id));
}

void test_digest() {
    using ninfer::DiskKvPrefixDigest;
    // Determinism: the same prefix yields the same digest, computed in one shot or rolled.
    const std::int32_t tokens[] = {101, 2053, 13, 68005, 299, 1};
    DiskKvPrefixDigest whole;
    whole.absorb(tokens, 6);
    DiskKvPrefixDigest rolled;
    rolled.absorb(tokens, 2);
    rolled.absorb(tokens + 2, 2);
    rolled.absorb(tokens + 4, 2);
    CHECK(whole.lo() == rolled.lo());
    CHECK(whole.hi() == rolled.hi());

    // Stable change under prefix extension: a longer prefix differs from its own prefix.
    DiskKvPrefixDigest shorter;
    shorter.absorb(tokens, 3);
    CHECK(shorter.lo() != whole.lo() || shorter.hi() != whole.hi());

    // Content sensitivity: one changed token changes the digest of the shared tail.
    DiskKvPrefixDigest variant;
    const std::int32_t variant_tokens[] = {101, 2053, 14, 68005, 299, 1};
    variant.absorb(variant_tokens, 6);
    CHECK(variant.lo() != whole.lo() || variant.hi() != whole.hi());

    // The empty prefix has a fixed, non-zero digest (the offset bases).
    DiskKvPrefixDigest empty;
    CHECK(empty.lo() == 0xcbf29ce484222325ULL);
    CHECK(empty.hi() == 0x9dc8d177d94a1b9eULL);
}

} // namespace

int main() {
    test_digest();
    test_round_trip();
    test_crc_mismatch();
    test_identity_mismatch();
    test_index_rebuild();
    test_lru_order();
    test_make_room_and_full_failure();
    test_format_mismatch();
    test_capacity_adopt_expand();
    test_capacity_adopt_shrink();
    test_capacity_adopt_same_capacity();
    test_capacity_adopt_slot_size_mismatch();
    test_oversized_payload();
    test_reopen_fast_and_rebuild_paths();
    test_index_reader_sees_complete_files();
    if (failures != 0) {
        std::cerr << failures << " disk KV store check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "disk KV store: all checks passed" << std::endl;
    return 0;
}