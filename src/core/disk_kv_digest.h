#pragma once

// Pure-host rolling 128-bit content digest over an int32 token prefix. Two independent 64-bit
// FNV-1a hashes (distinct offset bases) over the little-endian token bytes. No CUDA, no engine
// types. A prefix's digest is deterministic; extending the prefix changes it. The engine feeds a
// chunk's licensed ledger prefix into this to derive the (lo, hi) half of the disk-tier content
// identity; the caller combines it with the program tag and the chunk's page-tail frontier.

#include <cstddef>
#include <cstdint>

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

} // namespace ninfer