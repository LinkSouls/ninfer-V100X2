#include "ops/launcher/token_mask.h"

#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 128;

// One thread clears one (lane, word) pair: 32 vocabulary bits per UINT32 word. A cleared bit
// forbids the token; a fully zero word is a dead end (all 32 tokens forbidden) and must clear
// the whole word range. An all-ones row is the unconstrained no-op and touches no bits.
__global__ void apply_token_mask_kernel(std::uint16_t* logits, const std::uint32_t* masks,
                                        std::int64_t lanes, std::int64_t vocab,
                                        std::int64_t lane_stride, std::int64_t vocab_stride,
                                        std::int32_t words) {
    const std::int64_t item = static_cast<std::int64_t>(blockIdx.x) * kBlock + threadIdx.x;
    const std::int64_t total = lanes * words;
    if (item >= total) { return; }
    const std::int64_t lane = item / words;
    const std::int32_t word = static_cast<std::int32_t>(item % words);
    const std::uint32_t allowed = masks[item];
    if (allowed == 0xFFFFFFFFu) { return; } // all-ones word: every token allowed, no writes
    const std::int64_t base  = lane * lane_stride;
    const std::int64_t begin = static_cast<std::int64_t>(word) * 32;
    const std::int64_t count = std::min<std::int64_t>(32, vocab - begin);
    for (std::int64_t bit = 0; bit < count; ++bit) {
        if (((allowed >> bit) & 1u) == 0u) {
            logits[base + (begin + bit) * vocab_stride] = 0xFF80u; // BF16 -inf
        }
    }
}

} // namespace

void apply_token_mask_strided_launch(const Tensor& logits, const Tensor& masks,
                                     std::int64_t lane_stride, std::int64_t vocab_stride,
                                     std::int64_t vocab, cudaStream_t stream) {
    const std::int64_t lanes = masks.ne[0];
    const std::int64_t words = masks.ne[1];
    const std::int64_t total = lanes * words;
    if (total <= 0) { return; }
    const std::int64_t grid = (total + kBlock - 1) / kBlock;
    apply_token_mask_kernel<<<static_cast<unsigned>(grid), kBlock, 0, stream>>>(
        reinterpret_cast<std::uint16_t*>(logits.data),
        reinterpret_cast<const std::uint32_t*>(masks.data), lanes, vocab, lane_stride,
        vocab_stride, static_cast<std::int32_t>(words));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
