// Public-contract qualification for apply_token_mask().
//
// Three invariants: an all-ones mask leaves every BF16 bit untouched (including at the
// sampler), a cleared bit writes exactly the BF16 negative-infinity pattern at the forbidden
// entries only, and a fully zero word (dead end) forbids the whole word range. A fixed-seed
// repeated sampling run proves a forbidden token is never selected.
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/token_mask.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::uint16_t kBf16NegInf = 0xFF80u;
constexpr std::uint32_t kAllOnes    = 0xFFFFFFFFu;

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

int run_case(std::int32_t lanes, std::int32_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(lanes) * columns;
    const std::int32_t words = (columns + 31) / 32;
    std::vector<float> values(count);
    fill_uniform(values, seed, -4.0f, 4.0f);
    std::vector<std::uint16_t> bits = bf16_bits(values);
    std::vector<std::uint32_t> masks(static_cast<std::size_t>(lanes) * words, kAllOnes);

    DeviceBuffer dlogits = to_device(bits);
    Tensor logits(dlogits.p, DType::BF16, {lanes, columns, 1, 1});
    DeviceBuffer dones = to_device(masks);
    Tensor ones_tensor(dones.p, DType::I32, {lanes, words, 1, 1});

    // 1) All-ones mask is a bit-identical no-op.
    ops::apply_token_mask(logits, ones_tensor, 0);
    cuda_synchronize();
    {
        const auto after = from_device<std::uint16_t>(dlogits, count);
        if (after != bits) { std::cerr << "all-ones mask changed logits bits\n"; return 1; }
    }

    // 2) Partial mask: forbid the low half of word 0 on every lane and the whole of word 0
    // (dead end) on the first lane.
    for (std::int32_t r = 0; r < lanes; ++r) {
        const std::size_t at = static_cast<std::size_t>(r) * words;
        masks[at] &= ~0x0000FFFFu;
        if (r == 0) { masks[at] = 0u; }
    }
    DeviceBuffer dmasks = to_device(masks);
    Tensor mask_tensor(dmasks.p, DType::I32, {lanes, words, 1, 1});

    // 3) Partial mask writes -inf exactly at the cleared bits and touches nothing else.
    ops::apply_token_mask(logits, mask_tensor, 0);
    cuda_synchronize();
    const auto result = from_device<std::uint16_t>(dlogits, count);
    int failures = 0;
    for (std::int32_t r = 0; r < lanes; ++r) {
        for (std::int32_t v = 0; v < columns; ++v) {
            const std::uint32_t word =
                masks[static_cast<std::size_t>(r) * words + v / 32] >> (v % 32) & 1u;
            const std::size_t at = (static_cast<std::size_t>(r) * columns + v);
            if (word == 0u) {
                if (result[at] != kBf16NegInf) {
                    std::cerr << "lane " << r << " token " << v << " not masked to -inf\n";
                    ++failures;
                }
            } else if (result[at] != bits[at]) {
                std::cerr << "lane " << r << " token " << v << " touched by the mask\n";
                ++failures;
            }
        }
    }
    return failures;
}

// Column-major [V, lanes] layout plus a fixed-seed repeated sampling run: a forbidden token
// must never be selected, and an all-ones mask must be a no-op at the sampler as well.
int run_columns_case(std::int32_t lanes, std::int32_t columns) {
    const std::size_t count = static_cast<std::size_t>(lanes) * columns;
    const std::int32_t words = (columns + 31) / 32;
    std::vector<float> values(count);
    fill_uniform(values, 7, -4.0f, 4.0f);
    std::vector<std::uint16_t> bits = bf16_bits(values);
    std::vector<std::uint32_t> masks(static_cast<std::size_t>(lanes) * words, kAllOnes);
    for (std::int32_t r = 0; r < lanes; ++r) {
        const std::size_t at = static_cast<std::size_t>(r) * words;
        masks[at] &= ~0x0000FFFFu; // forbid tokens 0..31
        if (r == 0) { masks[at] = 0u; } // dead end: forbid tokens 32..63 as well
    }

    DeviceBuffer dlogits = to_device(bits);
    Tensor logits(dlogits.p, DType::BF16, {columns, lanes, 1, 1});

    // All-ones mask must leave the sampled output bit-identical to the unmasked draw.
    std::vector<ops::SamplingConfig> configs(lanes);
    std::vector<int> positions(lanes);
    for (std::int32_t r = 0; r < lanes; ++r) {
        configs[r].temperature = 1.0f;
        configs[r].seed        = 424242u;
        positions[r]           = 1000 + r;
    }
    DeviceBuffer dconfigs = to_device(configs);
    DeviceBuffer dpositions = to_device(positions);
    DeviceBuffer dout = to_device(std::vector<std::int32_t>(lanes, -1));
    Tensor out_tensor(dout.p, DType::I32, {lanes, 1, 1, 1});
    Tensor positions_tensor(dpositions.p, DType::I32, {lanes, 1, 1, 1});
    const std::size_t workspace_bytes =
        ops::sampling_workspace_capacity_bytes(columns, lanes, lanes);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::sample(logits, out_tensor, columns, static_cast<const ops::SamplingConfig*>(dconfigs.p),
                positions_tensor, ops::kSamplePurposeDecode, workspace, 0);
    cuda_synchronize();
    const auto unmasked = from_device<std::int32_t>(dout, lanes);

    DeviceBuffer dones = to_device(std::vector<std::uint32_t>(masks.size(), kAllOnes));
    Tensor ones_tensor(dones.p, DType::I32, {lanes, words, 1, 1});
    ops::apply_token_mask_columns(logits, ones_tensor, 0);
    ops::sample(logits, out_tensor, columns, static_cast<const ops::SamplingConfig*>(dconfigs.p),
                positions_tensor, ops::kSamplePurposeDecode, workspace, 0);
    cuda_synchronize();
    const auto masked_ones = from_device<std::int32_t>(dout, lanes);
    if (masked_ones != unmasked) {
        std::cerr << "all-ones column mask changed the sampled output\n";
        return 1;
    }

    // Partial mask: repeated fixed-seed draws must never select a forbidden token.
    DeviceBuffer dmasks = to_device(masks);
    Tensor mask_tensor(dmasks.p, DType::I32, {lanes, words, 1, 1});
    ops::apply_token_mask_columns(logits, mask_tensor, 0);
    int failures = 0;
    constexpr int kRepeats = 8;
    for (int rep = 0; rep < kRepeats; ++rep) {
        // Fresh logical position per repeat: the counter-based RNG keys on (seed, position,
        // purpose), so each repeat is an independent draw from the same masked distribution.
        for (std::int32_t r = 0; r < lanes; ++r) { positions[r] = 1000 + rep * lanes + r; }
        dpositions = to_device(positions);
        positions_tensor = Tensor(dpositions.p, DType::I32, {lanes, 1, 1, 1});
        ops::sample(logits, out_tensor, columns, static_cast<const ops::SamplingConfig*>(dconfigs.p),
                    positions_tensor, ops::kSamplePurposeDecode, workspace, 0);
        cuda_synchronize();
        const auto drawn = from_device<std::int32_t>(dout, lanes);
        for (std::int32_t r = 0; r < lanes; ++r) {
            const int token = drawn[static_cast<std::size_t>(r)];
            if (token < 0 || token >= columns) {
                std::cerr << "lane " << r << " sampled out-of-domain token " << token << "\n";
                ++failures;
                continue;
            }
            const std::uint32_t word =
                masks[static_cast<std::size_t>(r) * words + token / 32] >> (token % 32) & 1u;
            if (word == 0u) {
                std::cerr << "lane " << r << " sampled forbidden token " << token << "\n";
                ++failures;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case(1, 33, 1);      // one lane, width crossing the word boundary
    failures += run_case(4, 40, 2);      // two words per lane
    failures += run_case(8, 1024, 3);    // wide vocabulary, batch of lanes
    failures += run_columns_case(4, 96); // column-major layout plus repeated fixed-seed sampling
    std::cout << (failures == 0 ? "OK" : "FAIL") << " apply_token_mask public contract\n";
    return failures == 0 ? 0 : 1;
}