#pragma once

// NVFP4 group-16 KV cache codec (device helpers).
//
// Each group of 16 D256-rotated values is stored as 8 packed bytes (two E2M1 codes per byte,
// low nibble first) plus one E4M3 scale byte (the group's abs-max divided by the E2M1 max of 6,
// matching the weight-side quantize_nvfp4_k16 convention).
//
// Dequantization reuses the validated integer E2M1 decode (decode_nvfp4_e2m1_byte, bit-exact with
// the hardware path on all 16 codes) and the E4M3 scale decode (decode_nvfp4_e4m3). Quantization
// uses a SOFTWARE E2M1 packer: the hardware cvt.rn.satfinite.e2m1x2.f32 is Blackwell-only and
// __trap()s on sm_70, so the code bytes are produced by round-to-nearest-even against the 8-value
// E2M1 grid {0, 0.5, 1, 1.5, 2, 3, 4, 6} (ties to the even mantissa).
//
// As with the FP8 codec, quantization (append) and dequantization (stage) are FUSED into the
// attention kernels; this header has NO standalone kernel launch of its own (gated by the
// kernel-launch-token assertion in t0-assertions.txt).

#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer {
namespace kv_nvfp4 {

inline constexpr int kGroup = 16;
inline constexpr std::uint8_t kE2M1Max = 6u;

using ::ninfer::ops::detail::decode_nvfp4_e2m1_byte;
using ::ninfer::ops::detail::decode_nvfp4_e4m3;

// Round a non-negative finite float to the nearest E2M1 code (0..7), ties to the even mantissa.
// The E2M1 grid is {0, 0.5, 1, 1.5, 2, 3, 4, 6}; the midpoints are
// {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0} and each tie goes to the even-mantissa code (0, 2, 4, 6):
//   x <= 0.25 -> 0     (tie 0|0.5 -> code 0)
//   x <  0.75 -> 0.5   (tie 0.5|1 -> code 2)
//   x <= 1.25 -> 1     (tie 1|1.5 -> code 2)
//   x <  1.75 -> 1.5   (tie 1.5|2 -> code 4)
//   x <= 2.5  -> 2     (tie 2|3 -> code 4)
//   x <  3.5  -> 3     (tie 3|4 -> code 6)
//   x <= 5.0  -> 4     (tie 4|6 -> code 6)
//   else      -> 6     (saturate)
__device__ __forceinline__ std::uint32_t e2m1_encode_magnitude(float x) {
    if (x <= 0.25f) { return 0u; }
    if (x < 0.75f) { return 1u; }
    if (x <= 1.25f) { return 2u; }
    if (x < 1.75f) { return 3u; }
    if (x <= 2.5f) { return 4u; }
    if (x < 3.5f) { return 5u; }
    if (x <= 5.0f) { return 6u; }
    return 7u;
}

// Encode one signed float to a 4-bit E2M1 code (RN-even, saturating at 6 in magnitude).
__device__ __forceinline__ std::uint8_t e2m1_encode(float x) {
    const bool neg = x < 0.0f;
    const std::uint8_t mag = static_cast<std::uint8_t>(e2m1_encode_magnitude(fabsf(x)));
    return static_cast<std::uint8_t>(mag | (neg ? 0x8u : 0x0u));
}

// Encode 16 rotated values to 8 packed bytes (low nibble first) with the group's E4M3 scale.
// scale = E4M3(absmax / 6); codes[i] = E2M1(x[i] / scale). When the group abs-max is 0 (or the
// E4M3 scale rounds to 0) all codes are 0.
__device__ __forceinline__ void group16_encode(const float* rotated16, std::uint8_t* codes8,
                                               std::uint8_t* scale) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kGroup; ++i) {
        amax = fmaxf(amax, fabsf(rotated16[i]));
    }
    const std::uint8_t s = __nv_cvt_float_to_fp8(__fdiv_rn(amax, 6.0f), __NV_SATFINITE, __NV_E4M3);
    *scale = s;
    if (s == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            codes8[i] = 0;
        }
        return;
    }
    const float inv = __fdiv_rn(1.0f, decode_nvfp4_e4m3(s));
#pragma unroll
    for (int i = 0; i < kGroup; i += 2) {
        const std::uint8_t lo = e2m1_encode(__fmul_rn(rotated16[i], inv));
        const std::uint8_t hi = e2m1_encode(__fmul_rn(rotated16[i + 1], inv));
        codes8[i / 2] = static_cast<std::uint8_t>(lo | (hi << 4));
    }
}

// Decode one packed byte (two E2M1 codes, low nibble first) times the group scale to float2.
__device__ __forceinline__ float2 decode2(std::uint8_t packed, std::uint8_t scale) {
    const float2 code = decode_nvfp4_e2m1_byte(packed);
    const float s     = decode_nvfp4_e4m3(scale);
    return make_float2(__fmul_rn(code.x, s), __fmul_rn(code.y, s));
}

// Decode 8 packed bytes (16 E2M1 values, low nibble first) times the group scale to 16 FP16
// values (for mma staging).
__device__ __forceinline__ void decode16_to_f16(const std::uint8_t* codes8, std::uint8_t scale,
                                                __half* out16) {
    const float s = decode_nvfp4_e4m3(scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 v = decode_nvfp4_e2m1_byte(codes8[i]);
        out16[2 * i]     = __float2half_rn(__fmul_rn(v.x, s));
        out16[2 * i + 1] = __float2half_rn(__fmul_rn(v.y, s));
    }
}

} // namespace kv_nvfp4
} // namespace ninfer