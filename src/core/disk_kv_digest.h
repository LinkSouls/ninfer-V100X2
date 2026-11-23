#pragma once

// Pure-host rolling 128-bit content digest over an int32 token prefix. Two independent 64-bit
// FNV-1a hashes (distinct offset bases) over the little-endian token bytes. No CUDA, no engine
// types. A prefix's digest is deterministic; extending the prefix changes it. The engine feeds a
// chunk's licensed ledger prefix into this to derive the (lo, hi) half of the disk-tier content
// identity; the caller combines it with the program tag and the chunk's page-tail frontier.

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "disk_kv_store.h" // DiskKVIdentity

namespace ninfer {

class DiskKvPrefixDigest {
public:
    // Absorb a range of tokens into the running digest, continuing from the current state. This
    // is the "rolling" step: each archived chunk extends the digest by exactly the tokens its
    // page-tail newly covers, so the digest is a pure function of the licensed prefix.
    void absorb(const std::int32_t* tokens, std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint32_t value = static_cast<std::uint32_t>(tokens[i]);
            for (int byte = 0; byte < 4; ++byte) {
                const std::uint8_t b = static_cast<std::uint8_t>(value >> (8 * byte));
                lo_ = fnv(lo_, b);
                hi_ = fnv(hi_, b);
            }
        }
    }

    [[nodiscard]] std::uint64_t lo() const noexcept { return lo_; }
    [[nodiscard]] std::uint64_t hi() const noexcept { return hi_; }

private:
    static std::uint64_t fnv(std::uint64_t hash, std::uint8_t byte) noexcept {
        return (hash ^ static_cast<std::uint64_t>(byte)) * 0x00000100000001B3ULL;
    }
    std::uint64_t lo_ = 0xcbf29ce484222325ULL; // FNV-1a 64-bit offset basis
    std::uint64_t hi_ = 0x9dc8d177d94a1b9eULL; // independent second basis
};

// Cross-restart cold restore: re-derive the KV chunk identity chain from a token ledger using the
// EXACT chunking and rolling digest of the archival path (pages = ceil(n_tokens/page_size), chunks
// of `chunk_pages` pages, the digest advanced by each chunk's page-tail token count, each chunk's
// identity = {digest.lo, digest.hi, tag, first_page + pages}), then walk the chunks from the start
// and stop at the first whose identity fails `contains`. Returns the farthest recoverable page-tail
// frontier (in pages); 0 when `n_tokens` is 0 or no chunk is recoverable. This is a pure function
// of the ledger (no CUDA, no engine types), so it reproduces the archival identities chunk for
// chunk and is unit-testable in isolation. `contains` is the store residency predicate.
template <typename Contains>
std::uint32_t disk_kv_cold_recoverable_frontier(const std::int32_t* ledger, std::size_t n_tokens,
                                                std::uint64_t tag, std::uint32_t page_size,
                                                std::uint32_t chunk_pages, const Contains& contains) {
    if (n_tokens == 0) { return 0; }
    const std::uint32_t pages = 1 + static_cast<std::uint32_t>(n_tokens - 1) / page_size;
    DiskKvPrefixDigest disk_digest;
    std::size_t disk_absorbed = 0;
    std::uint32_t frontier = 0;
    for (std::uint32_t first = 0; first < pages; first += chunk_pages) {
        const std::uint32_t count = std::min(chunk_pages, pages - first);
        const std::size_t tail_tokens = std::min<std::uint64_t>(
            std::uint64_t(first + count) * page_size, n_tokens);
        disk_digest.absorb(ledger + disk_absorbed, tail_tokens - disk_absorbed);
        disk_absorbed = tail_tokens;
        const DiskKVIdentity identity{disk_digest.lo(), disk_digest.hi(), tag, first + count};
        if (!contains(identity)) { break; }
        frontier = first + count;
    }
    return frontier;
}

} // namespace ninfer