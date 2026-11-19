#pragma once

// Shared GQA attention test fixture: the paged-KV host/device cache model, the FP64 ideal
// attention oracle, the INT8-G64 codec, and the two registered numerical criteria.
//
// Owned by tests/ops/test_gqa_attention.cpp (the Op's public-contract conformance suite) and
// reused verbatim by tests/ops/test_attention_headlocal.cpp (the tp2 head-local parity suite),
// which must build its per-device 2-KV-head pools out of exactly the same cache model the tp1
// suite builds its 4-head pool from -- otherwise the two sides would not be comparable.
//
// Everything here is parameterized by `Geometry{q_heads, kv_heads}`, so a 12|2 head-local
// geometry is expressed by passing one, not by adding a code path.
//
// EVERY non-template free function here MUST be `inline` (op_tester.h in this directory is the
// precedent). This code was lifted out of an anonymous namespace inside a .cpp, where internal
// linkage made that unnecessary; at named-namespace scope in a header it is not optional, and the
// two current consumers each being a single-TU target would hide the defect until a third target
// included this from two TUs.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::test::gqa {

using ninfer::DType;
using ninfer::KvCacheStorage;
using ninfer::kPagedKVPageSize;
using ninfer::PagedKVBatchLayerView;
using ninfer::PagedKVLayerView;
using ninfer::Tensor;

constexpr std::int32_t kHeadDim       = 256;
constexpr std::int32_t kQuantGroup    = 64;
constexpr std::int32_t kQuantGroups   = kHeadDim / kQuantGroup;
// p4 nvfp4-g16: NVFP4 stores two E2M1 codes per byte plus one E4M3 scale byte per group of 16.
constexpr std::int32_t kNvfp4Group    = 16;
constexpr std::int32_t kNvfp4Groups   = kHeadDim / kNvfp4Group;  // 16
constexpr float kAttentionScale       = 0.0625f;
constexpr std::uint16_t kOutputCanary = 0x7fc1u;

// The Op has two registered compute profiles. A1 and A3 use the same criterion for a given
// profile; token count, geometry, execution envelope, and private launch route do not select it.
//
// Both were fitted at the short windows the conformance suite runs (<= ~1025 visible keys), and
// their relative-L2 terms sit only about 1.9x above the unavoidable BF16 OUTPUT STORAGE floor for
// those cases. That headroom is real but small, and a route with a longer reduction chain can
// consume it without being wrong: the 85-way split-KV decode route at a 400k-1M window measures
// 2.84e-3 to 3.18e-3 against an INT8 limit of 3.15e-3. The long-window suite therefore judges
// relative-L2 against a floor it MEASURES per case rather than against these constants -- see
// long_window_attention_criterion below. These two are unchanged and remain the registered
// contract for every case the conformance suite runs.
constexpr ReductionCriterion kAttentionBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

constexpr ReductionCriterion kAttentionInt8Criterion{
    /*relative_l2*/ 3.15e-3,
    // p3 kv-cache-rotation: gross_absolute history: 1.1e-3 (UNrotated, 61-case calibration) ->
    // 2.2e-3 (rotation FP-association cost, worst measured 2.01e-3) -> 3.2e-3. The 3.2e-3 step
    // is driven by the 128-token fresh-prefill case (16q, T=128, max_ref ~1.0): the ROTATED path
    // needs gross 2.92e-3 (max_abs 5.02e-3 at limit rel 2.2e-3), and the UNrotated baseline on the
    // same case already needs 1.25e-3 ABOVE its original 1.1e-3 -- so the new shape, not the
    // rotation, invalidated the old calibration. 3.2e-3 = 2.92e-3 with ~10% margin (same
    // convention as the 2.2e-3 step). relative_l2 is unchanged -- it is the primary accuracy
    // metric; the rotated path's kernel-level rel_l2 sits within 0.95x of the limit on all 64
    // INT8 cases, and the end-to-end "no degradation" judgment (spec requirement) is made at the
    // whole-model tier against a common oracle, not from this per-tree operator comparison.
    // This is the spec's "judge the rotated path by tolerance, not bit-identity" contract.
    /*gross_absolute*/ 3.2e-3,
    /*gross_relative_to_max_reference*/ 2.2e-3,
};

// p4 fp8-e4m3: the FP8 conformance oracle is scheme-relative -- ideal_attention reads the E4M3-
// quantized cache values, so the device-vs-oracle gap is only the low-precision arithmetic (fp16
// dequant + fp16 mma + BF16 output storage), NOT the E4M3 quantization error (that is judged by
// the outlier regression and the whole-model quality tier, not here). It therefore sits at the
// same order as the INT8 criterion.
//
// CALIBRATED (sm_70, 64 FP8 attention cases, task 1.5): the worst measured kernel-level rel_l2 is
// 3.024e-3; 3.4e-3 admits it with ~11% margin (the fixture's standing convention, cf. the INT8
// 3.2e-3 = 2.92e-3 + ~10% step), while staying 3x below a dropped-rotation defect (rel_l2 ~ 1.4),
// so it is a conformance bound, not a pass-everything. The gross terms are inherited from INT8 and
// already carry their own margin (worst measured gross_ratio 0.897). The E4M3 outlier-channel cost
// is judged separately by the 0.030 rel_l2/ref_rms regression bound (worst measured 2.67e-2).
constexpr ReductionCriterion kAttentionFp8Criterion{
    /*relative_l2*/ 3.4e-3,
    /*gross_absolute*/ 3.2e-3,
    /*gross_relative_to_max_reference*/ 2.2e-3,
};

// p4 nvfp4-g16: the NVFP4 conformance oracle is scheme-relative, like FP8 -- ideal_attention reads
// the E2M1-quantized cache values (two-per-byte E2M1 code x E4M3 group scale), so the device-vs-
// oracle gap is only the low-precision arithmetic (fp16 dequant + fp16 mma + BF16 output storage),
// NOT the coarse-E2M1 quantization error (that is judged by the outlier regression and the
// whole-model quality tier, not here). The coarser E2M1 dequant values feed the fp16 mma with a
// slightly wider rounding spread than E4M3, so the relative-l2 bound is set a step above FP8's
// 3.4e-3. CALIBRATED (sm_70, NVFP4 attention cases, task 2.5): see evidence/nvfp4-g16-oracle-
// tolerances.md for the measured worst-case and the final margin. The gross terms are inherited
// from INT8/FP8 and carry their own margin.
constexpr ReductionCriterion kAttentionNvfp4Criterion{
    /*relative_l2*/ 4.0e-3,
    /*gross_absolute*/ 3.2e-3,
    /*gross_relative_to_max_reference*/ 2.2e-3,
};

// p4 k8v4: the asymmetric format's conformance oracle is scheme-relative, like FP8 and NVFP4 --
// ideal_attention reads the per-plane quantized cache values (key: E4M3 code x FP16 group scale,
// value: E2M1 code x E4M3 group scale), so the device-vs-oracle gap is only the low-precision
// arithmetic (fp16 dequant + fp16 mma + BF16 output storage), NOT the quantization error (that is
// judged by the outlier regression and the whole-model quality tier, not here). CALIBRATED (sm_70,
// 64 K8V4 attention cases, task 3.4): the worst measured kernel-level rel_l2 is 2.9707e-3; 3.3e-3
// admits it with ~11% margin (the fixture's standing convention, cf. the FP8 3.4e-3 = 3.024e-3 +
// ~11% step), while staying 4x below a dropped-rotation defect (rel_l2 ~ 1.4), so it is a
// conformance bound, not a pass-everything. The measured worst case sits at the NVFP4 value-plane
// cost (the E2M1 value grid dominates; the finer E4M3 key plane adds little). The gross terms are
// inherited from INT8/FP8/NVFP4 and already carry their own margin (worst measured gross_ratio
// 0.883). The E2M1/E4M3 outlier-channel cost is judged separately by the 0.141 rel_l2/ref_rms
// regression bound (worst measured 0.1280, 6q geometry). See
// evidence/k8v4-asym-oracle-tolerances.md for the measured worst-cases and final margins.
constexpr ReductionCriterion kAttentionK8V4Criterion{
    /*relative_l2*/ 3.3e-3,
    /*gross_absolute*/ 3.2e-3,
    /*gross_relative_to_max_reference*/ 2.2e-3,
};

// --- long-window reduction domain -------------------------------------------------------------
//
// A suite-owned criterion, in the fixture rather than in one test's anonymous namespace for the
// same reason gdn_criteria.h exists: a second consumer of the long-window domain must be judged by
// THIS contract, not by a transcription of it that drifts.
//
// The Op writes BF16. Rounding the exact FP64 result to BF16 is therefore an error the kernel
// cannot avoid, and its relative-L2 contribution is a property of the output VALUES, not of the
// implementation: for a population whose values are spread across a binade, round-to-nearest gives
// a relative-L2 of roughly 2^-9/sqrt(3) ~ 1.1e-3, and for the concrete long-window outputs it
// measures ~1.66e-3. That floor does NOT grow with the window -- relative-L2 is a ratio and the
// storage error is relative, which is why the measured long-window numbers are flat in N
// (3.18e-3 at 400k, 2.96e-3 at 1M). What the long window costs is the REDUCTION: an 85-way split
// with ~12k keys per split accumulates more FP32 rounding than the <= 1025-key cases the
// registered constants were fitted at, and the registered ~1.9x headroom over the floor is not
// enough to absorb it.
//
// So the long-window criterion calibrates against the floor MEASURED for the exact case
// (`bf16_storage_floor_relative_l2` below, applied to that case's own FP64 reference) instead of
// carrying a fitted constant that would silently mean different things at different shapes. The
// multiplier is the ratio of total error to storage error that is accepted: 2.5x, against a
// measured ~1.78x for the shipped kernel, so it admits the observed accumulation with ~40% margin
// and nothing like a defect. The GROSS terms are inherited from the registered profile, but note
// they are nearly UNINFORMATIVE at these magnitudes -- with max|reference| ~ 1.8e-3 the INT8 gross
// limit is ~1.104e-3, i.e. ~60% of the largest reference value -- so the relative-L2 term is the
// live check and is why it is calibrated rather than rounded off.
inline constexpr double kLongWindowFloorMultiplier = 2.5;

// The relative-L2 that BF16 output storage alone contributes for this exact reference population:
// round the FP64 reference to BF16, promote it back, and measure it against the FP64 reference.
// Self-calibrating -- no constant to drift, and it reports the floor for the case actually run.
[[nodiscard]] inline double bf16_storage_floor_relative_l2(std::span<const double> reference) {
    std::vector<double> rounded(reference.size());
    for (std::size_t index = 0; index < reference.size(); ++index) {
        rounded[index] = static_cast<double>(
            bf16_to_f32(f32_to_bf16(static_cast<float>(reference[index]))));
    }
    return compute_reduction_stats(rounded.data(), reference.data(),
                                   static_cast<std::int64_t>(reference.size()))
        .relative_l2;
}

// The registered profile's gross terms with a relative-L2 term calibrated against `storage_floor`
// (which the caller obtains from `bf16_storage_floor_relative_l2` on that case's own reference).
[[nodiscard]] inline ReductionCriterion long_window_attention_criterion(DType dtype,
                                                                        double storage_floor);

struct Geometry {
    const char* name;
    std::int32_t q_heads;
    std::int32_t kv_heads;

    [[nodiscard]] std::int32_t query_group() const { return q_heads / kv_heads; }
};

enum class MappingPattern { Identity, Offset, Fragmented };

inline const char* mapping_name(MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return "identity";
    case MappingPattern::Offset:
        return "offset";
    case MappingPattern::Fragmented:
        return "fragmented";
    }
    return "unknown";
}

inline std::int32_t align_up_page(std::int32_t value) {
    constexpr std::int32_t kFixtureAlignment = 2 * kPagedKVPageSize;
    return ((value + kFixtureAlignment - 1) / kFixtureAlignment) * kFixtureAlignment;
}

inline std::int32_t physical_page_count(std::int32_t logical_pages, MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return logical_pages;
    case MappingPattern::Offset:
        return logical_pages + 2;
    case MappingPattern::Fragmented:
        return 2 * logical_pages + 1;
    }
    return 0;
}

inline std::vector<std::int32_t> make_block_table(std::int32_t logical_pages, MappingPattern pattern) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages));
    switch (pattern) {
    case MappingPattern::Identity:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page; }
        break;
    case MappingPattern::Offset:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page + 1; }
        break;
    case MappingPattern::Fragmented:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = 2 * page + 1; }
        break;
    }
    return table;
}

inline std::size_t q_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                    std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.q_heads) * static_cast<std::size_t>(token));
}

inline std::size_t kv_input_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                           std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.kv_heads) * static_cast<std::size_t>(token));
}

inline std::size_t cache_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

inline std::size_t scale_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t group) {
    (void)geometry;
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(kQuantGroups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

inline std::size_t cache_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

inline std::size_t scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kQuantGroups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

inline std::size_t paged_index(std::int32_t leading_extent, const Geometry& geometry,
                        std::int32_t physical_page, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(geometry.kv_heads) *
                                                          static_cast<std::size_t>(physical_page)));
}

template <typename T>
std::vector<T> scatter_paged(const std::vector<T>& logical, std::int32_t leading_extent,
                             const Geometry& geometry, std::int32_t logical_capacity,
                             std::span<const std::int32_t> block_table,
                             std::int32_t physical_pages) {
    std::vector<T> physical(static_cast<std::size_t>(leading_extent) * kPagedKVPageSize *
                            static_cast<std::size_t>(geometry.kv_heads) *
                            static_cast<std::size_t>(physical_pages));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
    return physical;
}

template <typename T>
void scatter_paged_into(const std::vector<T>& logical, std::int32_t leading_extent,
                        const Geometry& geometry, std::int32_t logical_capacity,
                        std::span<const std::int32_t> block_table, std::vector<T>& physical) {
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
}

template <typename T>
std::vector<T> gather_paged(std::span<const T> physical, std::int32_t leading_extent,
                            const Geometry& geometry, std::int32_t logical_capacity,
                            std::span<const std::int32_t> block_table) {
    std::vector<T> logical(static_cast<std::size_t>(leading_extent) * logical_capacity *
                           static_cast<std::size_t>(geometry.kv_heads));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t target = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                logical[target] =
                    physical[paged_index(leading_extent, geometry, page, head, position, leading)];
            }
        }
    }
    return logical;
}

inline std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

inline std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

inline std::vector<double> bf16_bits_to_double(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        values[i] = static_cast<double>(bf16_to_f32(bits[i]));
    }
    return values;
}

inline std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }

    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) { ++half_mantissa; }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

inline float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exp       = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exp == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exp - 15);
    }
    return negative ? -magnitude : magnitude;
}

inline std::int32_t round_even_to_i32(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    std::int32_t lower   = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

// --- host FP8 E4M3 (FN) codec model ---------------------------------------------------------
//
// p4: the FP8 group-64 path stores one E4M3 code byte per rotated dimension plus one FP16 scale
// (the group's abs-max), mirroring the device codec in kv_cache_fp8_g64_codec.cuh. This host model
// encodes with the SAME reference FP8 arithmetic (round-to-nearest-even, saturate |x| >= 448 to
// 448, NaN -> 0x7F) so the code/scale consistency check is meaningful: it BOUNDS the delta between
// the device's FP32 warp-shuffle rotation + E4M3 encode and this fixture's independent FP32
// association order + E4M3 encode (one E4M3 code step / one fp16 ulp at a rounding boundary) -- the
// same tolerance-equivalence contract the INT8 path uses, not bit identity.

// Decode one E4M3 (FN) code byte to double (oracle precision). 0x7F/0xFF decode to NaN; the top
// finite value is 448 (code 0x7E). Matches the device e4m3_decode.
[[nodiscard]] inline double fp8_e4m3_decode_host(std::uint8_t code) {
    const bool negative = (code & 0x80u) != 0;
    const int e4        = (code >> 3) & 0x0Fu;
    const int m3        = code & 0x07u;
    if (e4 == 15 && m3 == 7) { return std::numeric_limits<double>::quiet_NaN(); }
    const double magnitude = (e4 == 0)
                                 ? std::ldexp(static_cast<double>(m3) / 8.0, -6)  // subnormal m3*2^-9
                                 : (1.0 + static_cast<double>(m3) / 8.0) *
                                       std::ldexp(1.0, e4 - 7);
    return negative ? -magnitude : magnitude;
}

// Encode a finite float to one E4M3 (FN) code byte: nearest code in value space, round-to-nearest-
// even on a tie, saturate |x| >= 448 to 448 (0x7E), NaN -> 0x7F. Host mirror of
// __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3).
[[nodiscard]] inline std::uint8_t fp8_e4m3_encode_host(float x) {
    if (std::isnan(x)) { return 0x7Fu; }
    if (x == 0.0f) { return std::signbit(x) ? 0x80u : 0x00u; }  // preserve the sign of zero
    double v = x;
    if (v > 448.0) { v = 448.0; }
    if (v < -448.0) { v = -448.0; }
    // The 254 finite E4M3 values in ascending value order (negatives most-negative first, then
    // positives least first). Built once per TU (function-local static; a test fixture, so a
    // per-TU copy is harmless).
    struct Table {
        std::uint8_t code[254];
        double value[254];
        Table() {
            int n = 0;
            for (int c = 0xFE; c >= 0x80; --c) {  // -448 .. -0
                code[n] = static_cast<std::uint8_t>(c);
                value[n] = fp8_e4m3_decode_host(static_cast<std::uint8_t>(c));
                ++n;
            }
            for (int c = 0x00; c <= 0x7E; ++c) {  // +0 .. +448
                code[n] = static_cast<std::uint8_t>(c);
                value[n] = fp8_e4m3_decode_host(static_cast<std::uint8_t>(c));
                ++n;
            }
        }
    };
    static const Table table = Table();
    int lo = 0, hi = 254;
    while (lo < hi) {  // lower_bound: first index with value >= v
        const int mid = (lo + hi) / 2;
        if (table.value[mid] < v) { lo = mid + 1; } else { hi = mid; }
    }
    const int hi_idx = lo;
    const int lo_idx = (lo > 0) ? lo - 1 : 0;
    if (hi_idx == 254) { return table.code[253]; }  // v above all finite values -> 448
    const double d_lo = std::abs(v - table.value[lo_idx]);
    const double d_hi = std::abs(v - table.value[hi_idx]);
    int chosen;
    if (d_hi < d_lo) { chosen = hi_idx; }
    else if (d_lo < d_hi) { chosen = lo_idx; }
    else {  // exact tie: round to even mantissa
        const int m_lo = table.code[lo_idx] & 0x07u;
        const int m_hi = table.code[hi_idx] & 0x07u;
        chosen         = (m_lo % 2 == 0) ? lo_idx : hi_idx;
    }
    return table.code[chosen];
}

// Rank of a finite E4M3 code in ascending value order (0..253), so the code-step distance between
// two codes is |rank(a) - rank(b)|. NaN codes (0x7F/0xFF) rank -1 (a defect if ever seen).
[[nodiscard]] inline int fp8_e4m3_code_rank(std::uint8_t code) {
    if (code == 0x7Fu || code == 0xFFu) { return -1; }
    return (code < 0x80u) ? (static_cast<int>(code) + 127) : (254 - static_cast<int>(code));
}

// --- host NVFP4 (E2M1 group-16) codec model -------------------------------------------------
//
// p4: the NVFP4 group-16 path stores two E2M1 codes per byte (low nibble first) plus one E4M3
// scale byte per group of 16 D256-rotated values (scale = E4M3(absmax / 6)), mirroring the device
// codec in kv_cache_nvfp4_g16_codec.cuh. The host model encodes with the SAME E2M1 8-value grid
// {0, 0.5, 1, 1.5, 2, 3, 4, 6} (round-to-nearest-even on the midpoints, saturate at 6) and the SAME
// E4M3 scale (E4M3(absmax/6)) so the code/scale consistency check bounds the delta between the
// device's FP32 16-lane warp-shuffle rotation + E2M1 pack and this fixture's independent naive
// association order (one E2M1 code step / one E4M3 scale step at a rounding boundary) -- tolerance
// equivalence, not bit identity (the E2M1 grid is only 8 values, so the code delta is coarse but
// bounded).

// E2M1 magnitude grid by 3-bit magnitude index (sign is the top bit 0x08).
[[nodiscard]] inline double nvfp4_e2m1_decode_host(std::uint8_t code) {
    static constexpr double kMag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const bool negative  = (code & 0x08u) != 0;
    const int mag_index  = code & 0x07u;
    return negative ? -kMag[mag_index] : kMag[mag_index];
}

// Encode one signed float to a 4-bit E2M1 code (RN-even on the 8-value grid, saturate at 6).
// Host mirror of the device e2m1_encode (same float thresholds, same tie-to-even-mantissa).
[[nodiscard]] inline std::uint8_t nvfp4_e2m1_encode_host(float x) {
    const bool neg = x < 0.0f;
    const float a  = std::fabs(x);
    std::uint8_t mag;
    if (a <= 0.25f) { mag = 0u; }
    else if (a < 0.75f) { mag = 1u; }
    else if (a <= 1.25f) { mag = 2u; }
    else if (a < 1.75f) { mag = 3u; }
    else if (a <= 2.5f) { mag = 4u; }
    else if (a < 3.5f) { mag = 5u; }
    else if (a <= 5.0f) { mag = 6u; }
    else { mag = 7u; }
    return static_cast<std::uint8_t>(mag | (neg ? 0x08u : 0x00u));
}

// Value-order rank of a 4-bit E2M1 code (0..14); +0 (0x0) and -0 (0x8) both rank 7. A "one step"
// is an adjacent rank, so |rank(a) - rank(b)| bounds the number of grid values between two codes.
[[nodiscard]] inline int nvfp4_e2m1_code_rank(std::uint8_t nibble) {
    const bool neg = (nibble & 0x08u) != 0;
    const int mag  = nibble & 0x07u;
    return neg ? (7 - mag) : (7 + mag);  // neg: -6..-0 -> 0..7; pos: 0..6 -> 7..14
}

// NVFP4 per-row geometry: two E2M1 codes per byte, so dimension d lives in byte d/2 of the code
// plane (low nibble = even d, high nibble = odd d); one E4M3 scale byte per 16-dim group.
inline std::size_t nvfp4_code_byte_index(const Geometry& geometry, std::int32_t padded_context,
                            std::int32_t head, std::int32_t position, std::int32_t d) {
    return static_cast<std::size_t>(d / 2) +
           static_cast<std::size_t>(kHeadDim / 2) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

inline std::size_t nvfp4_scale_index(const Geometry& geometry, std::int32_t padded_context,
                            std::int32_t head, std::int32_t position, std::int32_t group16) {
    (void)geometry;
    return static_cast<std::size_t>(group16) +
           static_cast<std::size_t>(kNvfp4Groups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

inline std::size_t nvfp4_code_bytes(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kHeadDim / 2) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

inline std::size_t nvfp4_scale_bytes(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kNvfp4Groups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

struct HostCache {
    Geometry geometry;
    DType dtype;
    KvCacheStorage storage;  // p4 k8v4: dispatch key; the asymmetric format is FP8 dtype + this
    std::int32_t max_context;
    std::int32_t logical_capacity;
    std::vector<std::uint16_t> k_bf16;
    std::vector<std::uint16_t> v_bf16;
    std::vector<std::int8_t> k_i8;
    std::vector<std::int8_t> v_i8;
    std::vector<std::uint8_t> k_fp8;
    std::vector<std::uint8_t> v_fp8;
    std::vector<std::uint8_t> k_nvfp4;  // packed, two E2M1 codes per byte (kHeadDim/2 bytes/row)
    std::vector<std::uint8_t> v_nvfp4;
    std::vector<std::uint8_t> k_nvfp4_scale;  // one E4M3 byte per 16-dim group
    std::vector<std::uint8_t> v_nvfp4_scale;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;

    // p4 k8v4: the asymmetric format reuses the per-plane fields -- the KEY plane is FP8 group-64
    // (k_fp8 + FP16 k_scale) and the VALUE plane is NVFP4 group-16 (v_nvfp4 + E4M3 v_nvfp4_scale).
    // dtype stays FP8_E4M3FN (the key-plane code dtype the Op views carry); storage carries the
    // asymmetry.
    [[nodiscard]] bool k8v4() const { return storage == KvCacheStorage::Fp8KeyNvfp4Value; }
};

// The symmetric storage format recoverable from a code dtype (the interim map the public capacity
// entry point uses until the asymmetric Fp8KeyNvfp4Value is selected explicitly).
inline KvCacheStorage storage_for_dtype(DType dtype) {
    switch (dtype) {
    case DType::I8:         return KvCacheStorage::Int8Group64;
    case DType::FP8_E4M3FN: return KvCacheStorage::Fp8E4M3Group64;
    case DType::U8:         return KvCacheStorage::Nvfp4Group16;
    default:                return KvCacheStorage::BFloat16;
    }
}

// p3 kv-cache-rotation: the INT8-G64 path quantizes in the coordinates of a fixed normalized
// Sylvester D256 rotation (src/ops/kernel/hadamard_d256.cuh), and undoes it on the attention
// output. This host model applies the SAME transform with an independent naive association
// order (eight full-row butterfly stages, then the 2^-4 normalization), in the precision of
// the element type. It deliberately does NOT reuse the device warp-shuffle association order,
// so it is not bit-identical to the device result: at a .5 code boundary the predicted int8
// code can differ by one step. That is exactly why the INT8 cache checks below BOUND the
// code/scale delta (one step / one fp16 ulp) instead of demanding a bit-exact cache -- the
// kv-cache-rotation contract is tolerance equivalence, not bit identity.
template <typename T>
[[nodiscard]] inline std::vector<T> d256_rotate(const T* row) {
    std::vector<T> v(row, row + kHeadDim);
    for (int stage = 0; stage < 8; ++stage) {
        const int half = 1 << stage;
        for (int block = 0; block < kHeadDim; block += 2 * half) {
            for (int i = 0; i < half; ++i) {
                const T a = v[block + i];
                const T b = v[block + i + half];
                v[block + i]        = a + b;
                v[block + i + half] = a - b;
            }
        }
    }
    for (auto& x : v) { x = x * static_cast<T>(0.0625); }
    return v;
}

inline void encode_group(const std::vector<float>& source, std::size_t source_base,
                  std::vector<std::int8_t>& codes, std::size_t code_base,
                  std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }

    const float unrounded_scale    = absmax / 127.0f;
    const std::uint16_t scale_bits = f32_to_f16_bits(unrounded_scale);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale == 0.0f ? 0.0f : 1.0f / stored_scale;
    scales[scale_offset]           = scale_bits;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        std::int32_t code = 0;
        if (stored_scale != 0.0f) {
            const float scaled = source[source_base + static_cast<std::size_t>(i)] * inverse_scale;
            code               = std::clamp(round_even_to_i32(scaled), -127, 127);
        }
        codes[code_base + static_cast<std::size_t>(i)] = static_cast<std::int8_t>(code);
    }
}

// p4 fp8-e4m3: the group-64 FP8 encoder. Unlike INT8 (scale = absmax / 127, integer code), the FP16
// scale IS the abs-max and the code is the E4M3 byte E4M3(x / scale). Same D256-rotated input.
inline void encode_group_fp8(const std::vector<float>& source, std::size_t source_base,
                   std::vector<std::uint8_t>& codes, std::size_t code_base,
                   std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }
    const std::uint16_t scale_bits = f32_to_f16_bits(absmax);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale == 0.0f ? 0.0f : 1.0f / stored_scale;
    scales[scale_offset]           = scale_bits;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        const float scaled = source[source_base + static_cast<std::size_t>(i)] * inverse_scale;
        codes[code_base + static_cast<std::size_t>(i)] = fp8_e4m3_encode_host(scaled);
    }
}

// p4 nvfp4-g16: the group-16 NVFP4 encoder. scale = E4M3(absmax / 6) (one E4M3 byte); each of the
// 16 rotated values is E2M1(x / scale), packed two per byte (low nibble first) into 8 bytes.
// Mirrors the device group16_encode (same amax/6 E4M3 scale, same 8-value E2M1 grid).
inline void encode_group_nvfp4(const std::vector<float>& source, std::size_t source_base,
                   std::vector<std::uint8_t>& codes8, std::size_t byte_base,
                   std::vector<std::uint8_t>& scales, std::size_t scale_offset) {
    float amax = 0.0f;
    for (std::int32_t i = 0; i < kNvfp4Group; ++i) {
        amax = std::max(amax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }
    const std::uint8_t scale_byte = fp8_e4m3_encode_host(amax / 6.0f);
    scales[scale_offset]          = scale_byte;
    if (scale_byte == 0) {
        for (std::int32_t i = 0; i < 8; ++i) { codes8[byte_base + static_cast<std::size_t>(i)] = 0; }
        return;
    }
    const float scale_val = static_cast<float>(fp8_e4m3_decode_host(scale_byte));
    const float inv       = 1.0f / scale_val;
    for (std::int32_t i = 0; i < kNvfp4Group; i += 2) {
        const std::uint8_t lo = nvfp4_e2m1_encode_host(
            source[source_base + static_cast<std::size_t>(i)] * inv);
        const std::uint8_t hi = nvfp4_e2m1_encode_host(
            source[source_base + static_cast<std::size_t>(i + 1)] * inv);
        codes8[byte_base + static_cast<std::size_t>(i / 2)] =
            static_cast<std::uint8_t>(lo | (hi << 4));
    }
}

inline HostCache make_cache(const Geometry& geometry, DType dtype, std::int32_t max_context,
                     std::uint32_t seed, KvCacheStorage storage = KvCacheStorage::BFloat16) {
    // A default argument cannot name a preceding parameter of the same (non-template) function, so
    // BFloat16 is used as a "derive from dtype" sentinel; the derived storage equals the sentinel
    // for the BF16 dtype itself, so the resolve is a no-op there.
    if (storage == KvCacheStorage::BFloat16) { storage = storage_for_dtype(dtype); }
    const std::int32_t logical_capacity = align_up_page(max_context);
    const std::size_t elements          = cache_elements(geometry, logical_capacity);
    std::vector<float> logical_k        = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> logical_v        = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);

    HostCache cache;
    cache.geometry         = geometry;
    cache.dtype            = dtype;
    cache.storage          = storage;
    cache.max_context      = max_context;
    cache.logical_capacity = logical_capacity;
    if (dtype == DType::BF16) {
        cache.k_bf16 = to_bf16_bits(logical_k);
        cache.v_bf16 = to_bf16_bits(logical_v);
        return cache;
    }

    const bool k8v4    = (storage == KvCacheStorage::Fp8KeyNvfp4Value);
    const bool fp8     = (dtype == DType::FP8_E4M3FN) && !k8v4;
    const bool nvfp4   = (dtype == DType::U8);
    if (k8v4) {
        // Asymmetric: key plane FP8 group-64 (code + FP16 scale), value plane NVFP4 group-16
        // (packed E2M1 code + E4M3 scale).
        cache.k_fp8.assign(elements, 0);
        cache.v_nvfp4.assign(nvfp4_code_bytes(geometry, logical_capacity), 0);
        cache.v_nvfp4_scale.assign(nvfp4_scale_bytes(geometry, logical_capacity), 0);
    } else if (nvfp4) {
        cache.k_nvfp4.assign(nvfp4_code_bytes(geometry, logical_capacity), 0);
        cache.v_nvfp4.assign(nvfp4_code_bytes(geometry, logical_capacity), 0);
        cache.k_nvfp4_scale.assign(nvfp4_scale_bytes(geometry, logical_capacity), 0);
        cache.v_nvfp4_scale.assign(nvfp4_scale_bytes(geometry, logical_capacity), 0);
    } else if (fp8) {
        cache.k_fp8.assign(elements, 0);
        cache.v_fp8.assign(elements, 0);
    } else {
        cache.k_i8.assign(elements, 0);
        cache.v_i8.assign(elements, 0);
    }
    const std::size_t scales = scale_elements(geometry, logical_capacity);
    cache.k_scale.assign(scales, 0);
    cache.v_scale.assign(scales, 0);
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            // p3/p4: the quantized codecs run in the rotated coordinates; rotate each 256-dim row
            // before the group codec (INT8: absmax/127 integer code; FP8: absmax E4M3 code;
            // NVFP4: amax/6 E4M3 scale + 2-per-byte E2M1 codes).
            const std::size_t row = cache_index(geometry, logical_capacity, head, position, 0);
            const std::vector<float> k_rot = d256_rotate(logical_k.data() + row);
            const std::vector<float> v_rot = d256_rotate(logical_v.data() + row);
            if (k8v4) {
                // Key plane: FP8 group-64 (4 groups of 64, FP16 abs-max scale).
                for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                    const std::int32_t d   = group * kQuantGroup;
                    const std::size_t code = row + d;
                    const std::size_t scale =
                        scale_index(geometry, logical_capacity, head, position, group);
                    encode_group_fp8(k_rot, d, cache.k_fp8, code, cache.k_scale, scale);
                }
                // Value plane: NVFP4 group-16 (16 groups of 16, E4M3 amax/6 scale).
                for (std::int32_t group = 0; group < kNvfp4Groups; ++group) {
                    const std::int32_t d    = group * kNvfp4Group;
                    const std::size_t byte  = nvfp4_code_byte_index(geometry, logical_capacity, head,
                                                                    position, d);
                    const std::size_t scale = nvfp4_scale_index(geometry, logical_capacity, head,
                                                                position, group);
                    encode_group_nvfp4(v_rot, d, cache.v_nvfp4, byte, cache.v_nvfp4_scale, scale);
                }
                continue;
            }
            if (nvfp4) {
                for (std::int32_t group = 0; group < kNvfp4Groups; ++group) {
                    const std::int32_t d    = group * kNvfp4Group;
                    const std::size_t byte  = nvfp4_code_byte_index(geometry, logical_capacity, head,
                                                                    position, d);
                    const std::size_t scale = nvfp4_scale_index(geometry, logical_capacity, head,
                                                                position, group);
                    encode_group_nvfp4(k_rot, d, cache.k_nvfp4, byte, cache.k_nvfp4_scale, scale);
                    encode_group_nvfp4(v_rot, d, cache.v_nvfp4, byte, cache.v_nvfp4_scale, scale);
                }
                continue;
            }
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d   = group * kQuantGroup;
                const std::size_t code = row + d;
                const std::size_t scale =
                    scale_index(geometry, logical_capacity, head, position, group);
                if (fp8) {
                    encode_group_fp8(k_rot, d, cache.k_fp8, code, cache.k_scale, scale);
                    encode_group_fp8(v_rot, d, cache.v_fp8, code, cache.v_scale, scale);
                } else {
                    encode_group(k_rot, d, cache.k_i8, code, cache.k_scale, scale);
                    encode_group(v_rot, d, cache.v_i8, code, cache.v_scale, scale);
                }
            }
        }
    }
    return cache;
}

inline void append_cache(HostCache& cache, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    for (std::int32_t token = 0; token < static_cast<std::int32_t>(positions.size()); ++token) {
        const std::int32_t position = positions[static_cast<std::size_t>(token)];
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            if (cache.dtype == DType::BF16) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.logical_capacity, head, position, d);
                    cache.k_bf16[target] = f32_to_bf16(k[source]);
                    cache.v_bf16[target] = f32_to_bf16(v[source]);
                }
                continue;
            }

            // p3/p4: rotate each appended row before the group codec, matching the device write path.
            const bool k8v4    = cache.k8v4();
            const bool fp8     = (cache.dtype == DType::FP8_E4M3FN) && !k8v4;
            const bool nvfp4   = (cache.dtype == DType::U8);
            const std::size_t k_src = kv_input_index(geometry, head, 0, token);
            const std::vector<float> k_rot = d256_rotate(k.data() + k_src);
            const std::vector<float> v_rot = d256_rotate(v.data() + k_src);
            if (k8v4) {
                for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                    const std::int32_t d     = group * kQuantGroup;
                    const std::size_t target =
                        cache_index(geometry, cache.logical_capacity, head, position, d);
                    const std::size_t scale =
                        scale_index(geometry, cache.logical_capacity, head, position, group);
                    encode_group_fp8(k_rot, d, cache.k_fp8, target, cache.k_scale, scale);
                }
                for (std::int32_t group = 0; group < kNvfp4Groups; ++group) {
                    const std::int32_t d    = group * kNvfp4Group;
                    const std::size_t byte  = nvfp4_code_byte_index(geometry, cache.logical_capacity,
                                                                    head, position, d);
                    const std::size_t scale = nvfp4_scale_index(geometry, cache.logical_capacity,
                                                                head, position, group);
                    encode_group_nvfp4(v_rot, d, cache.v_nvfp4, byte, cache.v_nvfp4_scale, scale);
                }
                continue;
            }
            if (nvfp4) {
                for (std::int32_t group = 0; group < kNvfp4Groups; ++group) {
                    const std::int32_t d    = group * kNvfp4Group;
                    const std::size_t byte  = nvfp4_code_byte_index(geometry, cache.logical_capacity,
                                                                    head, position, d);
                    const std::size_t scale = nvfp4_scale_index(geometry, cache.logical_capacity,
                                                                head, position, group);
                    encode_group_nvfp4(k_rot, d, cache.k_nvfp4, byte, cache.k_nvfp4_scale, scale);
                    encode_group_nvfp4(v_rot, d, cache.v_nvfp4, byte, cache.v_nvfp4_scale, scale);
                }
                continue;
            }
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d     = group * kQuantGroup;
                const std::size_t target =
                    cache_index(geometry, cache.logical_capacity, head, position, d);
                const std::size_t scale =
                    scale_index(geometry, cache.logical_capacity, head, position, group);
                if (fp8) {
                    encode_group_fp8(k_rot, d, cache.k_fp8, target, cache.k_scale, scale);
                    encode_group_fp8(v_rot, d, cache.v_fp8, target, cache.v_scale, scale);
                } else {
                    encode_group(k_rot, d, cache.k_i8, target, cache.k_scale, scale);
                    encode_group(v_rot, d, cache.v_i8, target, cache.v_scale, scale);
                }
            }
        }
    }
}

inline double cache_value(const HostCache& cache, bool key, std::int32_t head, std::int32_t position,
                   std::int32_t d) {
    const std::size_t code = cache_index(cache.geometry, cache.logical_capacity, head, position, d);
    if (cache.dtype == DType::BF16) {
        return static_cast<double>(bf16_to_f32(key ? cache.k_bf16[code] : cache.v_bf16[code]));
    }

    const std::size_t scale =
        scale_index(cache.geometry, cache.logical_capacity, head, position, d / kQuantGroup);
    if (cache.k8v4()) {
        // Key plane: FP8 group-64; value plane: NVFP4 group-16 (independent code/scale dtypes).
        if (key) {
            return fp8_e4m3_decode_host(cache.k_fp8[code]) *
                   static_cast<double>(f16_bits_to_f32(cache.k_scale[scale]));
        }
        const std::size_t byte  = nvfp4_code_byte_index(cache.geometry, cache.logical_capacity, head,
                                                        position, d);
        const std::size_t nscale = nvfp4_scale_index(cache.geometry, cache.logical_capacity, head,
                                                     position, d / kNvfp4Group);
        const std::uint8_t packed = cache.v_nvfp4[byte];
        const std::uint8_t nibble =
            (d % 2 == 0) ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);  // low nibble = even d
        return nvfp4_e2m1_decode_host(nibble) *
               static_cast<double>(fp8_e4m3_decode_host(cache.v_nvfp4_scale[nscale]));
    }
    if (cache.dtype == DType::U8) {
        const std::size_t byte  = nvfp4_code_byte_index(cache.geometry, cache.logical_capacity,
                                                        head, position, d);
        const std::size_t nscale = nvfp4_scale_index(cache.geometry, cache.logical_capacity, head,
                                                     position, d / kNvfp4Group);
        const auto& codes  = key ? cache.k_nvfp4 : cache.v_nvfp4;
        const auto& scales = key ? cache.k_nvfp4_scale : cache.v_nvfp4_scale;
        const std::uint8_t packed = codes[byte];
        const std::uint8_t nibble =
            (d % 2 == 0) ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);  // low nibble = even d
        return nvfp4_e2m1_decode_host(nibble) *
               static_cast<double>(fp8_e4m3_decode_host(scales[nscale]));
    }
    if (cache.dtype == DType::FP8_E4M3FN) {
        const auto& codes  = key ? cache.k_fp8 : cache.v_fp8;
        const auto& scales = key ? cache.k_scale : cache.v_scale;
        return fp8_e4m3_decode_host(codes[code]) *
               static_cast<double>(f16_bits_to_f32(scales[scale]));
    }
    const auto& codes   = key ? cache.k_i8 : cache.v_i8;
    const auto& scales  = key ? cache.k_scale : cache.v_scale;
    const float decoded = static_cast<float>(codes[code]) * f16_bits_to_f32(scales[scale]);
    return static_cast<double>(decoded);
}

inline std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions,
                                    std::span<const std::int32_t> selected_pages = {}) {
    const Geometry& geometry  = cache.geometry;
    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) *
                               static_cast<std::size_t>(geometry.q_heads) *
                               static_cast<std::size_t>(tokens));

    std::vector<double> scores(static_cast<std::size_t>(positions.back()) + 1);
    std::vector<double> probabilities(scores.size());
    const auto selected = [&](std::int32_t position) {
        return selected_pages.empty() || std::find(selected_pages.begin(), selected_pages.end(),
            position / kPagedKVPageSize) != selected_pages.end();
    };
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t visible = positions[static_cast<std::size_t>(token)] + 1;
        for (std::int32_t q_head = 0; q_head < geometry.q_heads; ++q_head) {
            const std::int32_t kv_head = q_head / geometry.query_group();
            // p3/p4: quantized KV (INT8, FP8 and NVFP4) lives in rotated coordinates. Score in the
            // rotated space ((Hq)·(Hk) == q·k, H orthogonal) and undo the V rotation on the PV row
            // (the transform is an involution) so the output matches the un-rotated model.
            const bool rotated = (cache.dtype == DType::I8 || cache.dtype == DType::FP8_E4M3FN ||
                                  cache.dtype == DType::U8);
            const std::vector<float> q_rot =
                rotated ? d256_rotate(q.data() + q_index(geometry, q_head, 0, token))
                        : std::vector<float>();
            double max_score = -std::numeric_limits<double>::infinity();
            for (std::int32_t position = 0; position < visible; ++position) {
                if (!selected(position)) { continue; }
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const double qv = rotated
                                          ? q_rot[static_cast<std::size_t>(d)]
                                          : static_cast<double>(q[q_index(geometry, q_head, d, token)]);
                    dot += qv * cache_value(cache, true, kv_head, position, d);
                }
                const double score = dot * static_cast<double>(kAttentionScale);
                scores[static_cast<std::size_t>(position)] = score;
                max_score                                  = std::max(max_score, score);
            }

            double sum = 0.0;
            for (std::int32_t position = 0; position < visible; ++position) {
                if (!selected(position)) { probabilities[position] = 0; continue; }
                const double probability =
                    std::exp(scores[static_cast<std::size_t>(position)] - max_score);
                probabilities[static_cast<std::size_t>(position)] = probability;
                sum += probability;
            }
            for (std::int32_t position = 0; position < visible; ++position) {
                probabilities[static_cast<std::size_t>(position)] /= sum;
            }

            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double value = 0.0;
                for (std::int32_t position = 0; position < visible; ++position) {
                    value += probabilities[static_cast<std::size_t>(position)] *
                             cache_value(cache, false, kv_head, position, d);
                }
                output[q_index(geometry, q_head, d, token)] = value;
            }

            if (rotated) {
                const std::size_t out_base = q_index(geometry, q_head, 0, token);
                std::vector<double> row(static_cast<std::size_t>(kHeadDim));
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    row[static_cast<std::size_t>(d)] = output[out_base + static_cast<std::size_t>(d)];
                }
                const std::vector<double> undone = d256_rotate(row.data());
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    output[out_base + static_cast<std::size_t>(d)] =
                        undone[static_cast<std::size_t>(d)];
                }
            }
        }
    }
    return output;
}

template <typename T>
std::vector<T> copy_from_guarded(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(T));
    return values;
}

class DeviceCache {
public:
    DeviceCache(const HostCache& cache, MappingPattern mapping)
        : geometry_(cache.geometry), dtype_(cache.dtype), storage_(cache.storage),
          max_context_(cache.max_context),
          logical_capacity_(cache.logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(physical_page_count(logical_pages_, mapping)),
          block_table_host_(make_block_table(logical_pages_, mapping)),
          code_elements_(dtype_ == DType::U8
                             ? static_cast<std::size_t>(kHeadDim / 2) * kPagedKVPageSize *
                                   geometry_.kv_heads * physical_pages_
                             : static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                                   geometry_.kv_heads * physical_pages_),
          scale_elements_(dtype_ == DType::U8
                              ? static_cast<std::size_t>(kNvfp4Groups) * kPagedKVPageSize *
                                    geometry_.kv_heads * physical_pages_
                              : static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                                    geometry_.kv_heads * physical_pages_),
          // K8V4 value plane is NVFP4-sized (128 packed bytes/row, 16 E4M3 scale bytes/row) even
          // though the cache dtype is the key-plane FP8; every other format is symmetric.
          v_code_elements_((cache.k8v4() || dtype_ == DType::U8)
                               ? static_cast<std::size_t>(kHeadDim / 2) * kPagedKVPageSize *
                                     geometry_.kv_heads * physical_pages_
                               : static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                                     geometry_.kv_heads * physical_pages_),
          v_scale_elements_((cache.k8v4() || dtype_ == DType::U8)
                                ? static_cast<std::size_t>(kNvfp4Groups) * kPagedKVPageSize *
                                      geometry_.kv_heads * physical_pages_
                                : static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                                      geometry_.kv_heads * physical_pages_),
          k_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::uint8_t))),
          v_(v_code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::uint8_t))),
          k_scale_((dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN || dtype_ == DType::U8)
                       ? scale_elements_ *
                             (dtype_ == DType::U8 ? sizeof(std::uint8_t) : sizeof(std::uint16_t))
                       : 1),
          v_scale_((dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN || dtype_ == DType::U8)
                       ? v_scale_elements_ *
                             ((cache.k8v4() || dtype_ == DType::U8) ? sizeof(std::uint8_t)
                                                                    : sizeof(std::uint16_t))
                       : 1),
          block_table_(block_table_host_.size() * sizeof(std::int32_t)) {
        block_table_.copy_from_host(block_table_host_.data(),
                                    block_table_host_.size() * sizeof(std::int32_t));
        if (dtype_ == DType::BF16) {
            const auto k_physical =
                scatter_paged(cache.k_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint16_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint16_t));
        } else if (cache.k8v4()) {
            // K8V4: key plane FP8 group-64 (256 code bytes + 4 FP16 scales per row), value plane
            // NVFP4 group-16 (128 packed bytes + 16 E4M3 scale bytes per row).
            const auto k_physical =
                scatter_paged(cache.k_fp8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_nvfp4, kHeadDim / 2, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_nvfp4_scale, kNvfp4Groups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint8_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint8_t));
        } else if (dtype_ == DType::FP8_E4M3FN) {
            const auto k_physical =
                scatter_paged(cache.k_fp8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_fp8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint8_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        } else if (dtype_ == DType::U8) {
            // NVFP4: code plane is two E2M1 codes per byte (kHeadDim/2 bytes/row); the scale plane
            // is one E4M3 byte per 16-dim group (kNvfp4Groups bytes/row).
            const auto k_physical =
                scatter_paged(cache.k_nvfp4, kHeadDim / 2, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_nvfp4, kHeadDim / 2, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_nvfp4_scale, kNvfp4Groups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_nvfp4_scale, kNvfp4Groups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint8_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint8_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint8_t));
        } else {
            const auto k_physical =
                scatter_paged(cache.k_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::int8_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        }
    }

    PagedKVLayerView view() {
        PagedKVLayerView result;
        const bool k8v4 = storage_ == KvCacheStorage::Fp8KeyNvfp4Value;
        const std::int32_t v_code_extent = (k8v4 || dtype_ == DType::U8) ? kHeadDim / 2 : kHeadDim;
        // NVFP4 (U8) packs two E2M1 codes per byte on the key plane (kHeadDim/2 bytes/row); K8V4's
        // key plane is full-width FP8. Mirror the wrapper's k_code_extent so k_pages matches layout.
        const std::int32_t k_code_extent = (dtype_ == DType::U8 && !k8v4) ? kHeadDim / 2 : kHeadDim;
        result.k_pages      = Tensor(k_.data(), dtype_,
                                     {k_code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages      = Tensor(v_.data(), k8v4 ? DType::U8 : dtype_,
                                     {v_code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_table  = Tensor(block_table_.data(), DType::I32, {logical_pages_});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.dtype        = dtype_;
        result.storage      = k8v4 ? KvCacheStorage::Fp8KeyNvfp4Value
                                   : (dtype_ == DType::BF16)
                                         ? KvCacheStorage::BFloat16
                                         : (dtype_ == DType::FP8_E4M3FN)
                                               ? KvCacheStorage::Fp8E4M3Group64
                                               : (dtype_ == DType::U8)
                                                     ? KvCacheStorage::Nvfp4Group16
                                                     : KvCacheStorage::Int8Group64;
        if (dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            if (k8v4) {
                // Asymmetric: the value scale plane is E4M3 bytes, 16 groups of 16.
                result.v_scale_pages =
                    Tensor(v_scale_.data(), DType::FP8_E4M3FN,
                           {kNvfp4Groups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            } else {
                result.v_scale_pages =
                    Tensor(v_scale_.data(), DType::FP16,
                           {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            }
            result.quant_group = kQuantGroup;
        } else if (dtype_ == DType::U8) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP8_E4M3FN,
                       {kNvfp4Groups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP8_E4M3FN,
                       {kNvfp4Groups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.quant_group = kNvfp4Group;
        }
        return result;
    }

    PagedKVBatchLayerView batch_view() {
        const PagedKVLayerView direct = view();
        return {
            .k_pages       = direct.k_pages,
            .v_pages       = direct.v_pages,
            .k_scale_pages = direct.k_scale_pages,
            .v_scale_pages = direct.v_scale_pages,
            .block_tables  = direct.block_table.view({logical_pages_, 1}),
            .head_dim      = direct.head_dim,
            .num_kv_heads  = direct.num_kv_heads,
            .dtype         = direct.dtype,
            .quant_group   = direct.quant_group,
            .storage       = direct.storage,
        };
    }

    HostCache snapshot() const {
        HostCache cache;
        cache.geometry         = geometry_;
        cache.dtype            = dtype_;
        cache.storage          = storage_;
        cache.max_context      = max_context_;
        cache.logical_capacity = logical_capacity_;
        if (dtype_ == DType::BF16) {
            const auto k_physical = copy_from_guarded<std::uint16_t>(k_, code_elements_);
            const auto v_physical = copy_from_guarded<std::uint16_t>(v_, code_elements_);
            cache.k_bf16          = gather_paged<std::uint16_t>(k_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
            cache.v_bf16          = gather_paged<std::uint16_t>(v_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
        } else if (cache.k8v4()) {
            const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, code_elements_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_);
            cache.k_fp8 = gather_paged<std::uint8_t>(k_physical, kHeadDim, geometry_,
                                                     logical_capacity_, block_table_host_);
            cache.v_nvfp4 = gather_paged<std::uint8_t>(v_physical, kHeadDim / 2, geometry_,
                                                       logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_nvfp4_scale = gather_paged<std::uint8_t>(vs_physical, kNvfp4Groups, geometry_,
                                                             logical_capacity_, block_table_host_);
        } else if (dtype_ == DType::FP8_E4M3FN) {
            const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, code_elements_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_);
            cache.k_fp8 = gather_paged<std::uint8_t>(k_physical, kHeadDim, geometry_,
                                                     logical_capacity_, block_table_host_);
            cache.v_fp8 = gather_paged<std::uint8_t>(v_physical, kHeadDim, geometry_,
                                                     logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_scale = gather_paged<std::uint16_t>(vs_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
        } else if (dtype_ == DType::U8) {
            const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, code_elements_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint8_t>(k_scale_, scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint8_t>(v_scale_, scale_elements_);
            cache.k_nvfp4 = gather_paged<std::uint8_t>(k_physical, kHeadDim / 2, geometry_,
                                                       logical_capacity_, block_table_host_);
            cache.v_nvfp4 = gather_paged<std::uint8_t>(v_physical, kHeadDim / 2, geometry_,
                                                       logical_capacity_, block_table_host_);
            cache.k_nvfp4_scale = gather_paged<std::uint8_t>(ks_physical, kNvfp4Groups, geometry_,
                                                             logical_capacity_, block_table_host_);
            cache.v_nvfp4_scale = gather_paged<std::uint8_t>(vs_physical, kNvfp4Groups, geometry_,
                                                             logical_capacity_, block_table_host_);
        } else {
            const auto k_physical  = copy_from_guarded<std::int8_t>(k_, code_elements_);
            const auto v_physical  = copy_from_guarded<std::int8_t>(v_, code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_);
            cache.k_i8             = gather_paged<std::int8_t>(k_physical, kHeadDim, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.v_i8             = gather_paged<std::int8_t>(v_physical, kHeadDim, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_scale = gather_paged<std::uint16_t>(vs_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
        }
        return cache;
    }

    int verify_guards(const std::string& label) const {
        int failures = 0;
        failures += k_.verify_guards((label + " cache-k").c_str());
        failures += v_.verify_guards((label + " cache-v").c_str());
        if (dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN || dtype_ == DType::U8) {
            failures += k_scale_.verify_guards((label + " cache-k-scale").c_str());
            failures += v_scale_.verify_guards((label + " cache-v-scale").c_str());
        }
        failures += block_table_.verify_guards((label + " block-table").c_str());
        failures +=
            verify_exact((label + " block-table unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_table_, block_table_host_.size()),
                         block_table_host_);
        return failures;
    }

private:
    Geometry geometry_;
    DType dtype_;
    KvCacheStorage storage_;
    std::int32_t max_context_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_table_host_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    std::size_t v_code_elements_;  // == code_elements_ except K8V4 (NVFP4 value plane)
    std::size_t v_scale_elements_; // == scale_elements_ except K8V4 (NVFP4 value scale)
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_table_;
};

// p3: bounded INT8 cache comparison -- the device's FP32 shuffle rotation vs this fixture's
// FP64 ideal model can flip one code step / one fp16 scale ulp at a rounding boundary
// (definitions below, after BatchDeviceCache, which uses them in its verify()).
inline int verify_close_int8_codes(const std::string& label, const std::vector<std::int8_t>& got,
                     const std::vector<std::int8_t>& expected);
inline int verify_close_fp8_codes(const std::string& label, const std::vector<std::uint8_t>& got,
                     const std::vector<std::uint8_t>& expected);
inline int verify_close_nvfp4_codes(const std::string& label, const std::vector<std::uint8_t>& got,
                     const std::vector<std::uint8_t>& expected);
inline int verify_close_f16_scales(const std::string& label,
                     const std::vector<std::uint16_t>& got,
                     const std::vector<std::uint16_t>& expected);

class BatchDeviceCache {
public:
    BatchDeviceCache(std::span<const HostCache> rows, MappingPattern mapping)
        : geometry_(rows.front().geometry), dtype_(rows.front().dtype),
          storage_(rows.front().storage), rows_(rows.size()),
          logical_capacity_(rows.front().logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(mapping == MappingPattern::Fragmented
                              ? 2 * static_cast<std::int32_t>(rows_) * logical_pages_ + 1
                              : static_cast<std::int32_t>(rows_) * logical_pages_),
          block_tables_host_(rows_ * static_cast<std::size_t>(logical_pages_)),
          code_elements_(dtype_ == DType::U8
                             ? static_cast<std::size_t>(kHeadDim / 2) * kPagedKVPageSize *
                                   geometry_.kv_heads * physical_pages_
                             : static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                                   geometry_.kv_heads * physical_pages_),
          scale_elements_(dtype_ == DType::U8
                              ? static_cast<std::size_t>(kNvfp4Groups) * kPagedKVPageSize *
                                    geometry_.kv_heads * physical_pages_
                              : static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                                    geometry_.kv_heads * physical_pages_),
          v_code_elements_((storage_ == KvCacheStorage::Fp8KeyNvfp4Value || dtype_ == DType::U8)
                               ? static_cast<std::size_t>(kHeadDim / 2) * kPagedKVPageSize *
                                     geometry_.kv_heads * physical_pages_
                               : static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                                     geometry_.kv_heads * physical_pages_),
          v_scale_elements_((storage_ == KvCacheStorage::Fp8KeyNvfp4Value || dtype_ == DType::U8)
                                ? static_cast<std::size_t>(kNvfp4Groups) * kPagedKVPageSize *
                                      geometry_.kv_heads * physical_pages_
                                : static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                                      geometry_.kv_heads * physical_pages_),
          k_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::uint8_t))),
          v_(v_code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::uint8_t))),
          k_scale_((dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN || dtype_ == DType::U8)
                       ? scale_elements_ *
                             (dtype_ == DType::U8 ? sizeof(std::uint8_t) : sizeof(std::uint16_t))
                       : 1),
          v_scale_((dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN || dtype_ == DType::U8)
                       ? v_scale_elements_ *
                             ((storage_ == KvCacheStorage::Fp8KeyNvfp4Value ||
                               dtype_ == DType::U8)
                                  ? sizeof(std::uint8_t)
                                  : sizeof(std::uint16_t))
                       : 1),
          block_tables_(block_tables_host_.size() * sizeof(std::int32_t)) {
        for (std::size_t row = 0; row < rows_; ++row) {
            const HostCache& cache = rows[row];
            if (cache.geometry.q_heads != geometry_.q_heads ||
                cache.geometry.kv_heads != geometry_.kv_heads || cache.dtype != dtype_ ||
                cache.storage != storage_ ||
                cache.logical_capacity != logical_capacity_) {
                throw std::invalid_argument("batch cache rows must share one physical geometry");
            }
            for (std::int32_t logical = 0; logical < logical_pages_; ++logical) {
                const std::int32_t linear =
                    static_cast<std::int32_t>(row) * logical_pages_ + logical;
                block_tables_host_[row * static_cast<std::size_t>(logical_pages_) + logical] =
                    mapping == MappingPattern::Fragmented ? 2 * linear + 1 : linear;
            }
        }
        block_tables_.copy_from_host(block_tables_host_.data(),
                                     block_tables_host_.size() * sizeof(std::int32_t));
        upload_rows(rows);
    }

    PagedKVBatchLayerView view() {
        PagedKVBatchLayerView result;
        const bool k8v4 = storage_ == KvCacheStorage::Fp8KeyNvfp4Value;
        const std::int32_t v_code_extent = (k8v4 || dtype_ == DType::U8) ? kHeadDim / 2 : kHeadDim;
        // NVFP4 (U8) packs two E2M1 codes per byte on the key plane (kHeadDim/2 bytes/row); K8V4's
        // key plane is full-width FP8. Mirror the wrapper's k_code_extent so k_pages matches layout.
        const std::int32_t k_code_extent = (dtype_ == DType::U8 && !k8v4) ? kHeadDim / 2 : kHeadDim;
        result.k_pages      = Tensor(k_.data(), dtype_,
                                     {k_code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages      = Tensor(v_.data(), k8v4 ? DType::U8 : dtype_,
                                     {v_code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_tables = Tensor(block_tables_.data(), DType::I32,
                                     {logical_pages_, static_cast<std::int32_t>(rows_)});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.dtype        = dtype_;
        result.storage      = k8v4 ? KvCacheStorage::Fp8KeyNvfp4Value
                                   : (dtype_ == DType::BF16)
                                         ? KvCacheStorage::BFloat16
                                         : (dtype_ == DType::FP8_E4M3FN)
                                               ? KvCacheStorage::Fp8E4M3Group64
                                               : (dtype_ == DType::U8)
                                                     ? KvCacheStorage::Nvfp4Group16
                                                     : KvCacheStorage::Int8Group64;
        if (dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            if (k8v4) {
                result.v_scale_pages =
                    Tensor(v_scale_.data(), DType::FP8_E4M3FN,
                           {kNvfp4Groups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            } else {
                result.v_scale_pages =
                    Tensor(v_scale_.data(), DType::FP16,
                           {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            }
            result.quant_group = kQuantGroup;
        } else if (dtype_ == DType::U8) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP8_E4M3FN,
                       {kNvfp4Groups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP8_E4M3FN,
                       {kNvfp4Groups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.quant_group = kNvfp4Group;
        }
        return result;
    }

    int verify(const std::string& label, std::span<const HostCache> expected) const {
        if (expected.size() != rows_) {
            std::cerr << label << ": expected cache row count mismatch\n";
            return 1;
        }
        int failures = 0;
        if (dtype_ == DType::BF16) {
            std::vector<std::uint16_t> expected_k(code_elements_, 0);
            std::vector<std::uint16_t> expected_v(code_elements_, 0);
            scatter_bf16_rows(expected, expected_k, expected_v);
            failures +=
                verify_exact((label + " cache-k").c_str(),
                             copy_from_guarded<std::uint16_t>(k_, code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v").c_str(),
                             copy_from_guarded<std::uint16_t>(v_, code_elements_), expected_v);
        } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
            // K8V4: key plane FP8 group-64 (256 code bytes + 4 FP16 scales/row), value plane NVFP4
            // group-16 (128 packed bytes + 16 E4M3 scale bytes/row).
            std::vector<std::uint8_t> expected_k(code_elements_, 0);
            std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
            std::vector<std::uint16_t> expected_ks(scale_elements_, 0);
            std::vector<std::uint8_t> expected_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_fp8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_k);
                scatter_paged_into(expected[row].v_nvfp4, kHeadDim / 2, geometry_,
                                   logical_capacity_, table, expected_v);
                scatter_paged_into(expected[row].k_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_nvfp4_scale, kNvfp4Groups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_close_fp8_codes((label + " cache-k-code").c_str(),
                                       copy_from_guarded<std::uint8_t>(k_, code_elements_),
                                       expected_k);
            failures +=
                verify_close_nvfp4_codes((label + " cache-v-code").c_str(),
                                         copy_from_guarded<std::uint8_t>(v_, v_code_elements_),
                                         expected_v);
            failures += verify_close_f16_scales((label + " cache-k-scale").c_str(),
                                                copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_),
                                                expected_ks);
            failures += verify_close_fp8_codes((label + " cache-v-scale").c_str(),
                                               copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_),
                                               expected_vs);
        } else if (dtype_ == DType::FP8_E4M3FN) {
            std::vector<std::uint8_t> expected_k(code_elements_, 0);
            std::vector<std::uint8_t> expected_v(code_elements_, 0);
            std::vector<std::uint16_t> expected_ks(scale_elements_, 0);
            std::vector<std::uint16_t> expected_vs(scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_fp8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_k);
                scatter_paged_into(expected[row].v_fp8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_v);
                scatter_paged_into(expected[row].k_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_close_fp8_codes((label + " cache-k-code").c_str(),
                                       copy_from_guarded<std::uint8_t>(k_, code_elements_),
                                       expected_k);
            failures +=
                verify_close_fp8_codes((label + " cache-v-code").c_str(),
                                       copy_from_guarded<std::uint8_t>(v_, code_elements_),
                                       expected_v);
            failures += verify_close_f16_scales((label + " cache-k-scale").c_str(),
                                                copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_),
                                                expected_ks);
            failures += verify_close_f16_scales((label + " cache-v-scale").c_str(),
                                                copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_),
                                                expected_vs);
        } else if (dtype_ == DType::U8) {
            // NVFP4: packed two-per-byte E2M1 codes (kHeadDim/2 bytes/row) + one E4M3 scale byte
            // per 16-dim group (kNvfp4Groups/row).
            std::vector<std::uint8_t> expected_k(code_elements_, 0);
            std::vector<std::uint8_t> expected_v(code_elements_, 0);
            std::vector<std::uint8_t> expected_ks(scale_elements_, 0);
            std::vector<std::uint8_t> expected_vs(scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_nvfp4, kHeadDim / 2, geometry_,
                                   logical_capacity_, table, expected_k);
                scatter_paged_into(expected[row].v_nvfp4, kHeadDim / 2, geometry_,
                                   logical_capacity_, table, expected_v);
                scatter_paged_into(expected[row].k_nvfp4_scale, kNvfp4Groups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_nvfp4_scale, kNvfp4Groups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_close_nvfp4_codes((label + " cache-k-code").c_str(),
                                         copy_from_guarded<std::uint8_t>(k_, code_elements_),
                                         expected_k);
            failures +=
                verify_close_nvfp4_codes((label + " cache-v-code").c_str(),
                                         copy_from_guarded<std::uint8_t>(v_, code_elements_),
                                         expected_v);
            failures += verify_close_fp8_codes((label + " cache-k-scale").c_str(),
                                               copy_from_guarded<std::uint8_t>(k_scale_, scale_elements_),
                                               expected_ks);
            failures += verify_close_fp8_codes((label + " cache-v-scale").c_str(),
                                               copy_from_guarded<std::uint8_t>(v_scale_, scale_elements_),
                                               expected_vs);
        } else {
            std::vector<std::int8_t> expected_k(code_elements_, 0);
            std::vector<std::int8_t> expected_v(code_elements_, 0);
            std::vector<std::uint16_t> expected_ks(scale_elements_, 0);
            std::vector<std::uint16_t> expected_vs(scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_i8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_k);
                scatter_paged_into(expected[row].v_i8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_v);
                scatter_paged_into(expected[row].k_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_close_int8_codes((label + " cache-k-code").c_str(),
                                        copy_from_guarded<std::int8_t>(k_, code_elements_), expected_k);
            failures +=
                verify_close_int8_codes((label + " cache-v-code").c_str(),
                                        copy_from_guarded<std::int8_t>(v_, code_elements_), expected_v);
            failures += verify_close_f16_scales((label + " cache-k-scale").c_str(),
                                                copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_),
                                                expected_ks);
            failures += verify_close_f16_scales((label + " cache-v-scale").c_str(),
                                                copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_),
                                                expected_vs);
        }
        failures +=
            verify_exact((label + " block tables unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_tables_, block_tables_host_.size()),
                         block_tables_host_);
        failures += k_.verify_guards((label + " cache-k guard").c_str());
        failures += v_.verify_guards((label + " cache-v guard").c_str());
        if (dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN || dtype_ == DType::U8) {
            failures += k_scale_.verify_guards((label + " cache-k-scale guard").c_str());
            failures += v_scale_.verify_guards((label + " cache-v-scale guard").c_str());
        }
        failures += block_tables_.verify_guards((label + " block tables guard").c_str());
        return failures;
    }

private:
    [[nodiscard]] std::span<const std::int32_t> row_table(std::size_t row) const {
        return std::span<const std::int32_t>(block_tables_host_.data() +
                                                 row * static_cast<std::size_t>(logical_pages_),
                                             static_cast<std::size_t>(logical_pages_));
    }

    void scatter_bf16_rows(std::span<const HostCache> rows, std::vector<std::uint16_t>& k,
                           std::vector<std::uint16_t>& v) const {
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_bf16, kHeadDim, geometry_, logical_capacity_, table, k);
            scatter_paged_into(rows[row].v_bf16, kHeadDim, geometry_, logical_capacity_, table, v);
        }
    }

    void upload_rows(std::span<const HostCache> rows) {
        if (dtype_ == DType::BF16) {
            std::vector<std::uint16_t> physical_k(code_elements_, 0);
            std::vector<std::uint16_t> physical_v(code_elements_, 0);
            scatter_bf16_rows(rows, physical_k, physical_v);
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::uint16_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::uint16_t));
            return;
        }
        if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
            std::vector<std::uint8_t> physical_k(code_elements_, 0);
            std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
            std::vector<std::uint16_t> physical_ks(scale_elements_, 0);
            std::vector<std::uint8_t> physical_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(rows[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_k);
                scatter_paged_into(rows[row].v_nvfp4, kHeadDim / 2, geometry_, logical_capacity_,
                                   table, physical_v);
                scatter_paged_into(rows[row].k_scale, kQuantGroups, geometry_, logical_capacity_,
                                   table, physical_ks);
                scatter_paged_into(rows[row].v_nvfp4_scale, kNvfp4Groups, geometry_,
                                   logical_capacity_, table, physical_vs);
            }
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::uint8_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::uint8_t));
            k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() *
                                                               sizeof(std::uint16_t));
            v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() *
                                                               sizeof(std::uint8_t));
            return;
        }
        if (dtype_ == DType::FP8_E4M3FN) {
            std::vector<std::uint8_t> physical_k(code_elements_, 0);
            std::vector<std::uint8_t> physical_v(code_elements_, 0);
            std::vector<std::uint16_t> physical_ks(scale_elements_, 0);
            std::vector<std::uint16_t> physical_vs(scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(rows[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_k);
                scatter_paged_into(rows[row].v_fp8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_v);
                scatter_paged_into(rows[row].k_scale, kQuantGroups, geometry_, logical_capacity_,
                                   table, physical_ks);
                scatter_paged_into(rows[row].v_scale, kQuantGroups, geometry_, logical_capacity_,
                                   table, physical_vs);
            }
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::uint8_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::uint8_t));
            k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() *
                                                               sizeof(std::uint16_t));
            v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() *
                                                               sizeof(std::uint16_t));
            return;
        }
        if (dtype_ == DType::U8) {
            std::vector<std::uint8_t> physical_k(code_elements_, 0);
            std::vector<std::uint8_t> physical_v(code_elements_, 0);
            std::vector<std::uint8_t> physical_ks(scale_elements_, 0);
            std::vector<std::uint8_t> physical_vs(scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(rows[row].k_nvfp4, kHeadDim / 2, geometry_, logical_capacity_,
                                   table, physical_k);
                scatter_paged_into(rows[row].v_nvfp4, kHeadDim / 2, geometry_, logical_capacity_,
                                   table, physical_v);
                scatter_paged_into(rows[row].k_nvfp4_scale, kNvfp4Groups, geometry_,
                                   logical_capacity_, table, physical_ks);
                scatter_paged_into(rows[row].v_nvfp4_scale, kNvfp4Groups, geometry_,
                                   logical_capacity_, table, physical_vs);
            }
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::uint8_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::uint8_t));
            k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() *
                                                               sizeof(std::uint8_t));
            v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() *
                                                               sizeof(std::uint8_t));
            return;
        }
        std::vector<std::int8_t> physical_k(code_elements_, 0);
        std::vector<std::int8_t> physical_v(code_elements_, 0);
        std::vector<std::uint16_t> physical_ks(scale_elements_, 0);
        std::vector<std::uint16_t> physical_vs(scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_i8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_k);
            scatter_paged_into(rows[row].v_i8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_v);
            scatter_paged_into(rows[row].k_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_ks);
            scatter_paged_into(rows[row].v_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_vs);
        }
        k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::int8_t));
        v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::int8_t));
        k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
        v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
    }

    Geometry geometry_;
    DType dtype_;
    KvCacheStorage storage_;
    std::size_t rows_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_tables_host_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    std::size_t v_code_elements_;  // == code_elements_ except K8V4 (NVFP4 value plane)
    std::size_t v_scale_elements_; // == scale_elements_ except K8V4 (NVFP4 value scale)
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_tables_;
};

// p3: bounded INT8 cache comparison. The device's FP32 warp-shuffle rotation and this
// fixture's FP64 ideal rotation agree to a few ulps, and at a .5 code boundary that flips
// ONE code step (and at most one fp16 scale ulp). Bigger gaps are defects. Sizes are still
// checked exactly: the rotation contract is "same layout, different numbers".
inline int verify_close_int8_codes(const std::string& label, const std::vector<std::int8_t>& got,
                     const std::vector<std::int8_t>& expected) {
    if (got.size() != expected.size()) {
        std::cerr << label << ": code size mismatch " << got.size() << " vs "
                  << expected.size() << "\n";
        return 1;
    }
    int failures = 0;
    int max_delta = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const int delta = std::abs(static_cast<int>(got[i]) - static_cast<int>(expected[i]));
        max_delta       = std::max(max_delta, delta);
        if (delta > 1) {
            ++failures;
            if (failures <= 8) {
                std::cerr << label << " [" << i << "]: expected code "
                          << static_cast<int>(expected[i]) << " got " << static_cast<int>(got[i])
                          << " (delta " << delta << ")\n";
            }
        }
    }
    if (failures == 0) {
        std::cout << "ok      " << label << " (" << got.size() << " codes, max delta "
                  << max_delta << ")\n";
    } else {
        std::cerr << label << ": " << failures << " code(s) differ by more than one step\n";
    }
    return failures;
}

// p4 fp8-e4m3: bounded FP8 cache comparison. The device's FP32 warp-shuffle rotation and this
// fixture's FP32 ideal rotation agree to a few ulps; at an E4M3 rounding boundary that flips ONE
// code step (adjacent in value space) and at most one fp16 scale ulp. Bigger gaps are defects.
// E4M3 codes are a floating-point lattice, not a contiguous integer one, so "one step" is the
// number of E4M3 values between the two codes (fp8_e4m3_code_rank), not a byte delta.
inline int verify_close_fp8_codes(const std::string& label, const std::vector<std::uint8_t>& got,
                     const std::vector<std::uint8_t>& expected) {
    if (got.size() != expected.size()) {
        std::cerr << label << ": code size mismatch " << got.size() << " vs "
                  << expected.size() << "\n";
        return 1;
    }
    int failures = 0;
    int max_steps = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const int gr = fp8_e4m3_code_rank(got[i]);
        const int er = fp8_e4m3_code_rank(expected[i]);
        if (gr < 0 || er < 0) {
            ++failures;
            if (failures <= 8) {
                std::cerr << label << " [" << i << "]: NaN E4M3 code seen (got " << got[i]
                          << ", expected " << expected[i] << ")\n";
            }
            continue;
        }
        const int steps = std::abs(gr - er);
        max_steps       = std::max(max_steps, steps);
        if (steps > 1) {
            ++failures;
            if (failures <= 8) {
                std::cerr << label << " [" << i << "]: expected code " << static_cast<int>(expected[i])
                          << " got " << static_cast<int>(got[i]) << " (" << steps
                          << " E4M3 step(s))\n";
            }
        }
    }
    if (failures == 0) {
        std::cout << "ok      " << label << " (" << got.size() << " codes, max " << max_steps
                  << " E4M3 step)\n";
    } else {
        std::cerr << label << ": " << failures << " code(s) differ by more than one E4M3 step\n";
    }
    return failures;
}

// p4 nvfp4-g16: bounded NVFP4 code comparison. Each packed byte holds two E2M1 codes (low nibble
// first); the device's 16-lane warp-shuffle rotation vs this fixture's naive association order can
// flip one E2M1 grid step (per nibble) at a rounding boundary, so bound the per-nibble value-rank
// delta to one step (the NVFP4 E4M3 scale bytes are checked with verify_close_fp8_codes, which
// already ranks E4M3 codes).
inline int verify_close_nvfp4_codes(const std::string& label, const std::vector<std::uint8_t>& got,
                     const std::vector<std::uint8_t>& expected) {
    if (got.size() != expected.size()) {
        std::cerr << label << ": code size mismatch " << got.size() << " vs "
                  << expected.size() << "\n";
        return 1;
    }
    int failures = 0;
    int max_steps = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        for (int n = 0; n < 2; ++n) {
            const std::uint8_t gn = n == 0 ? (got[i] & 0x0Fu) : ((got[i] >> 4) & 0x0Fu);
            const std::uint8_t en = n == 0 ? (expected[i] & 0x0Fu) : ((expected[i] >> 4) & 0x0Fu);
            const int steps = std::abs(nvfp4_e2m1_code_rank(gn) - nvfp4_e2m1_code_rank(en));
            max_steps       = std::max(max_steps, steps);
            if (steps > 1) {
                ++failures;
                if (failures <= 8) {
                    std::cerr << label << " [" << i << ", nibble " << n << "]: expected code "
                              << static_cast<int>(en) << " got " << static_cast<int>(gn) << " ("
                              << steps << " E2M1 step(s))\n";
                }
            }
        }
    }
    if (failures == 0) {
        std::cout << "ok      " << label << " (" << got.size() << " bytes / "
                  << (got.size() * 2) << " E2M1 codes, max " << max_steps << " E2M1 step)\n";
    } else {
        std::cerr << label << ": " << failures << " code(s) differ by more than one E2M1 step\n";
    }
    return failures;
}

inline int verify_close_f16_scales(const std::string& label,
                     const std::vector<std::uint16_t>& got,
                     const std::vector<std::uint16_t>& expected) {
    if (got.size() != expected.size()) {
        std::cerr << label << ": scale size mismatch " << got.size() << " vs "
                  << expected.size() << "\n";
        return 1;
    }
    // The stored scales are non-negative fp16, so the bit-pattern order IS the numeric order
    // and a one-ulp difference is a one-bit difference in the pattern.
    int failures = 0;
    int max_ulp  = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const int ulp = std::abs(static_cast<int>(got[i]) - static_cast<int>(expected[i]));
        max_ulp       = std::max(max_ulp, ulp);
        if (ulp > 1) {
            ++failures;
            if (failures <= 8) {
                std::cerr << label << " [" << i << "]: expected scale bits " << expected[i]
                          << " got " << got[i] << " (" << ulp << " ulp)\n";
            }
        }
    }
    if (failures == 0) {
        std::cout << "ok      " << label << " (" << got.size() << " scales, max delta "
                  << max_ulp << " ulp)\n";
    } else {
        std::cerr << label << ": " << failures << " scale(s) differ by more than one ulp\n";
    }
    return failures;
}

inline int verify_cache(const std::string& label, const HostCache& got, const HostCache& expected) {
    int failures = 0;
    if (expected.k8v4()) {
        // p4 k8v4: asymmetric planes -- key FP8 group-64 (code + FP16 scale), value NVFP4 group-16
        // (packed E2M1 code + E4M3 scale).
        failures += verify_close_fp8_codes((label + " cache-k-code").c_str(), got.k_fp8, expected.k_fp8);
        failures += verify_close_nvfp4_codes((label + " cache-v-code").c_str(), got.v_nvfp4, expected.v_nvfp4);
        failures += verify_close_f16_scales((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        failures += verify_close_fp8_codes((label + " cache-v-scale").c_str(), got.v_nvfp4_scale, expected.v_nvfp4_scale);
    } else if (expected.dtype == DType::BF16) {
        failures += verify_exact((label + " cache-k").c_str(), got.k_bf16, expected.k_bf16);
        failures += verify_exact((label + " cache-v").c_str(), got.v_bf16, expected.v_bf16);
    } else if (expected.dtype == DType::FP8_E4M3FN) {
        failures += verify_close_fp8_codes((label + " cache-k-code").c_str(), got.k_fp8, expected.k_fp8);
        failures += verify_close_fp8_codes((label + " cache-v-code").c_str(), got.v_fp8, expected.v_fp8);
        failures += verify_close_f16_scales((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        failures += verify_close_f16_scales((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    } else if (expected.dtype == DType::U8) {
        failures += verify_close_nvfp4_codes((label + " cache-k-code").c_str(), got.k_nvfp4, expected.k_nvfp4);
        failures += verify_close_nvfp4_codes((label + " cache-v-code").c_str(), got.v_nvfp4, expected.v_nvfp4);
        failures += verify_close_fp8_codes((label + " cache-k-scale").c_str(), got.k_nvfp4_scale, expected.k_nvfp4_scale);
        failures += verify_close_fp8_codes((label + " cache-v-scale").c_str(), got.v_nvfp4_scale, expected.v_nvfp4_scale);
    } else {
        failures += verify_close_int8_codes((label + " cache-k-code").c_str(), got.k_i8, expected.k_i8);
        failures += verify_close_int8_codes((label + " cache-v-code").c_str(), got.v_i8, expected.v_i8);
        failures += verify_close_f16_scales((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        failures += verify_close_f16_scales((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    }
    return failures;
}

inline int verify_input(const std::string& label, const GuardedDeviceBuffer& device,
                 const std::vector<std::uint16_t>& expected) {
    int failures = verify_exact(
        label.c_str(), copy_from_guarded<std::uint16_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

inline int verify_positions(const std::string& label, const GuardedDeviceBuffer& device,
                     const std::vector<std::int32_t>& expected) {
    int failures = verify_exact(label.c_str(),
                                copy_from_guarded<std::int32_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

inline const char* cache_name(DType dtype) {
    if (dtype == DType::BF16) { return "bf16"; }
    if (dtype == DType::FP8_E4M3FN) { return "fp8-e4m3-g64"; }
    if (dtype == DType::U8) { return "nvfp4-g16"; }
    return "int8-g64";
}

// p4 k8v4: the asymmetric format shares the FP8 dtype, so a storage-aware name is needed to keep
// the K8V4 cases distinguishable from the symmetric FP8 ones in labels and failure output.
inline const char* cache_name(DType dtype, KvCacheStorage storage) {
    if (storage == KvCacheStorage::Fp8KeyNvfp4Value) { return "k8v4(k8-v4)"; }
    return cache_name(dtype);
}

inline ReductionCriterion attention_criterion(DType dtype) {
    if (dtype == DType::BF16) { return kAttentionBf16Criterion; }
    if (dtype == DType::FP8_E4M3FN) { return kAttentionFp8Criterion; }
    if (dtype == DType::U8) { return kAttentionNvfp4Criterion; }
    return kAttentionInt8Criterion;
}

inline ReductionCriterion attention_criterion(DType dtype, KvCacheStorage storage) {
    if (storage == KvCacheStorage::Fp8KeyNvfp4Value) { return kAttentionK8V4Criterion; }
    return attention_criterion(dtype);
}

inline ReductionCriterion long_window_attention_criterion(DType dtype, double storage_floor) {
    ReductionCriterion criterion = attention_criterion(dtype);
    criterion.relative_l2        = kLongWindowFloorMultiplier * storage_floor;
    return criterion;
}

inline int verify_attention(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& reference, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, reference, criterion);
}

inline void inject_codec_edges(const Geometry& geometry, std::int32_t tokens, std::vector<float>& k,
                        std::vector<float>& v) {
    if (tokens == 0) return;
    for (std::int32_t d = 0; d < kQuantGroup; ++d) {
        k[kv_input_index(geometry, 0, d, 0)]               = 0.0f;
        v[kv_input_index(geometry, 0, kQuantGroup + d, 0)] = 0.0f;
    }
    k[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = -1.0f;
    v[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = 1.0f;
}

} // namespace ninfer::test::gqa
