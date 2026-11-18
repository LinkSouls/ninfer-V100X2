#pragma once

// FP8 E4M3 group-64 KV cache codec (device helpers).
//
// The KV rows are D256-rotated before quantization (the P3 rotation coordinate contract), so the
// group-64 abs-max is well-behaved and the E4M3 code is well-conditioned. Each group of 64 rotated
// values is stored as 64 E4M3 code bytes plus one FP16 scale (the group's abs-max, per the p4
// design: "逐组绝对值最大值导出 FP16 scale").
//
// Quantization (append) and dequantization (stage) are FUSED into the attention kernels, exactly
// like the INT8 path in gqa_attention_kv_quant.cuh. This header deliberately provides only
// __device__ helpers, with NO standalone kernel launch of its own -- the T0 assertion on the
// kernel-launch token in t0-assertions.txt gates that.
//
// E4M3 encode/decode reuses the CUDA __nv_cvt_float_to_fp8 / __nv_fp8x2_e4m3 primitives (the same
// software FP8 arithmetic the FP8 linear path uses), so this codec is bit-exact with the reference
// FP8 numbers on sm_70 (where, unlike the NVFP4 E2M1 pack, E4M3 has no hardware requirement).

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer {
namespace kv_fp8 {

inline constexpr std::uint8_t kE4M3NanCode = 0x7Fu;
inline constexpr int kGroup = 64;

// Encode a finite float to one E4M3 (FN) code byte: round-to-nearest-even, saturate |x| >= 448 to
// 0x7E (the max finite value 448), NaN to 0x7F. This is the reference FP8 encode.
__device__ __forceinline__ std::uint8_t e4m3_encode(float x) {
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

// Decode one E4M3 (FN) code byte to float. 0x7F/0xFF (exp==15, mant==7) decode to NaN; the top
// finite range (exp==15, mant<=6) reaches 448. Matches the FP8 linear path's decode.
__device__ __forceinline__ float e4m3_decode(std::uint8_t code) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(code) | (static_cast<std::uint16_t>(code) << 8);
    return static_cast<float2>(value).x;
}

// Decode 8 adjacent E4M3 code bytes to 8 FP16 values (for the mma staging), matching the value
// order of the INT8 gqa_kv_dequant_i8x8_f16 helper.
__device__ __forceinline__ void e4m3_decode8_to_f16(const std::uint8_t* codes8, __half* out8) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out8[i] = __float2half_rn(e4m3_decode(codes8[i]));
    }
}

// The group's abs-max, as the FP16 scale.
__device__ __forceinline__ __half group64_scale(const float* rotated64) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kGroup; ++i) {
        amax = fmaxf(amax, fabsf(rotated64[i]));
    }
    return __float2half_rn(amax);
}

// Quantize a rotated group of 64 values: FP16 scale = abs-max, codes = E4M3(x / scale). The scale
// is the FP16-rounded abs-max, so the dequant (e4m3_decode * scale) is self-consistent.
__device__ __forceinline__ void group64_encode(const float* rotated64, std::uint8_t* codes,
                                               __half* scale) {
    const __half s = group64_scale(rotated64);
    *scale = s;
    const float inv = (__half2float(s) > 0.0f) ? __fdiv_rn(1.0f, __half2float(s)) : 0.0f;
#pragma unroll
    for (int i = 0; i < kGroup; ++i) {
        codes[i] = e4m3_encode(__fmul_rn(rotated64[i], inv));
    }
}

// Dequantize one code: value = E4M3(code) * scale.
__device__ __forceinline__ float decode(std::uint8_t code, __half scale) {
    return __fmul_rn(e4m3_decode(code), __half2float(scale));
}

// Dequantize 8 codes to 8 FP16 values with the group scale (for mma staging).
__device__ __forceinline__ void decode8_to_f16(const std::uint8_t* codes8, __half scale,
                                               __half* out8) {
    const float s = __half2float(scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out8[i] = __float2half_rn(__fmul_rn(e4m3_decode(codes8[i]), s));
    }
}

} // namespace kv_fp8
} // namespace ninfer