#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Kept for reference only: no production call site uses it any more (every NVFP4 weight decode
// entry point now goes through decode_nvfp4_e2m1_byte, or through a kernel that already decodes
// with integer bit math of its own). It stays so the hardware-backed decode remains available for
// bit-equivalence checks and for architectures that do have FP4 hardware.
__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

// Volta has no FP4 hardware: __nv_fp4x2_e2m1 decodes through an FP6 path that costs several
// instructions per value. The E2M1 grid (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives) is
// exactly representable in FP16, so a nibble decodes with pure integer bit ops that build the
// FP16 bit pattern directly from sign(1) | exp(2) | mant(1):
//   exp == 0: subnormal, mant ? 0.5 (0x3800) : 0.0 (0x0000)
//   exp != 0: (1 + mant/2) * 2^(exp-1) -> 1.0 (0x3C00) with exponent (exp-1) and mantissa mant
__device__ __forceinline__ std::uint16_t decode_nvfp4_e2m1_nibble(std::uint32_t code) {
    const std::uint32_t sign = (code >> 3) & 1u;
    const std::uint32_t exp  = (code >> 1) & 3u;
    const std::uint32_t mant = code & 1u;
    const std::uint32_t magnitude =
        (exp == 0) ? (mant ? 0x3800u : 0x0000u) : (0x3C00u + ((exp - 1) << 10) + (mant << 9));
    return static_cast<std::uint16_t>(magnitude | (sign << 15));
}

// Decode one packed byte holding two E2M1 values (low nibble first) to a float2, matching the
// value order of decode_nvfp4_e2m1x2.
__device__ __forceinline__ float2 decode_nvfp4_e2m1_byte(std::uint8_t storage) {
    const __half lo{__half_raw{decode_nvfp4_e2m1_nibble(storage & 0x0Fu)}};
    const __half hi{__half_raw{decode_nvfp4_e2m1_nibble(storage >> 4)}};
    return make_float2(__half2float(lo), __half2float(hi));
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
// cvt.e2m1x2 is native FP4 hardware, Blackwell-only -- Volta has no FP4 hardware at all,
// so unlike the mma.cuh helpers this isn't "no SIMT replacement built yet", it's simply
// never going to have one. Already permanently out of scope (see the V100 performance summary);
// this only needs a guard so ptxas doesn't reject the PTX outright at compile time.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
#else
    (void)values;
    codes_lo = 0;
    codes_hi = 0;
    __trap();
#endif
}

__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x * input_scale_divisor, decoded_scale);
        values[pair].y = __fdiv_rn(values[pair].y * input_scale_divisor, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

} // namespace ninfer::ops::detail
