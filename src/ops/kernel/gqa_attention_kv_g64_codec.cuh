#pragma once

// Group-64 KV codec policies for the GQA attention kernels (INT8 and FP8-E4M3).
//
// Both the INT8 and FP8 group-64 formats share the cache geometry: one code byte per rotated
// dimension plus one FP16 scale per group of 64. The ONLY thing that differs is the code<->value
// arithmetic, so the attention kernels (decode partial, SIMT tiled, QPN) carry a single codec
// policy template parameter and select the arithmetic from it -- the p4 design's "kernel 内按模板
// 参数选择 codec". Quantization input is always the D256-rotated row (the P3 rotation coordinate
// contract), exactly as in the INT8 path.
//
// A policy exposes:
//   Code / Scale            cache element types (code byte, FP16 scale)
//   scale_from_amax(amax)   the group's FP16 scale from its abs-max
//   encode(x, inv_scale)    one code byte for a rotated value x, given 1/scale
//   dequant8(codes8, scale) 8 code bytes + scale -> 8 packed FP16 (an int4), for mma staging
//   dequant8_from(ptr,scale) the same, loading the 8 bytes from global/shared memory
//
// The INT8 policy reproduces the original gqa_kv_quant_code / gqa_kv_dequant_i8x8_f16 helpers
// verbatim, so the INT8 path stays bit-identical to the pre-p4 kernel; the FP8 policy swaps in the
// E4M3 encode (kv_fp8) and E4M3->fp16 dequant. No standalone quant/dequant kernel is introduced:
// these are __device__ helpers fused into the attention kernels (gated by the !__global__ T0
// assertion).

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/kv_cache_fp8_g64_codec.cuh"

#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

// Shared group-64 quantize body for the INT8 and FP8 policies: one D256-rotated row (lane l holds
// dims l + 32*r) -> four per-64-group codes (two per lane) + one FP16 scale per group (lane 0).
// This is the append-block arithmetic that the attention kernels used to inline; it lives here so
// the kernels select it through the policy's quantize_row, exactly as the NVFP4 policy does.
template <typename KvCodec, typename Geometry>
__device__ __forceinline__ void gqa_g64_quantize_row(const float* k_vals, const float* v_vals,
                                                     int physical_page, int kv_head,
                                                     int page_offset,
                                                     typename KvCodec::Code* cache_k,
                                                     typename KvCodec::Code* cache_v,
                                                     typename KvCodec::Scale* cache_k_scale,
                                                     typename KvCodec::Scale* cache_v_scale) {
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    constexpr unsigned FullMask = 0xffffffffu;
#pragma unroll
    for (int grp = 0; grp < kGqaKvQuantGroups; ++grp) {
        const int d0   = grp * kGqaKvQuantGroup + lane;
        const int d1   = d0 + 32;
        const float kv0 = k_vals[2 * grp];
        const float kv1 = k_vals[2 * grp + 1];
        const float vv0 = v_vals[2 * grp];
        const float vv1 = v_vals[2 * grp + 1];
        float kamax     = warp_max(fmaxf(fabsf(kv0), fabsf(kv1)), FullMask);
        float vamax     = warp_max(fmaxf(fabsf(vv0), fabsf(vv1)), FullMask);
        const auto ksh  = KvCodec::scale_from_amax(kamax);
        const auto vsh  = KvCodec::scale_from_amax(vamax);
        const float k_inv  = __half2float(ksh) > 0.0f ? 1.0f / __half2float(ksh) : 0.0f;
        const float v_inv  = __half2float(vsh) > 0.0f ? 1.0f / __half2float(vsh) : 0.0f;
        cache_k[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
            KvCodec::encode(kv0, k_inv);
        cache_k[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
            KvCodec::encode(kv1, k_inv);
        cache_v[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
            KvCodec::encode(vv0, v_inv);
        cache_v[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
            KvCodec::encode(vv1, v_inv);
        if (lane == 0) {
            const std::int64_t so =
                gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, grp, page_offset);
            cache_k_scale[so] = ksh;
            cache_v_scale[so] = vsh;
        }
    }
}

// Signed int8, per-64-group scale = abs-max / 127, code = clamp(x / scale) (the original P3 path).
struct KvI8G64 {
    using Code       = std::int8_t;
    using Scale      = __half;
    using StagingBuf = int2;  // 8 code bytes (8 rotated dimensions)
    // K/V-plane type aliases (the asymmetric K8V4 policy needs separate K and V plane types; the
    // symmetric policies alias them to the same type so every call site stays bit-identical).
    using KScale      = Scale;
    using VScale      = Scale;
    using KStagingBuf = StagingBuf;
    using VStagingBuf = StagingBuf;

    static __device__ __forceinline__ Scale scale_from_amax(float amax) {
        return __float2half_rn(amax > 0.0f ? amax / 127.0f : 0.0f);
    }
    static __device__ __forceinline__ Code encode(float x, float inv_scale) {
        return gqa_kv_quant_code(x, inv_scale);
    }
    static __device__ __forceinline__ int4 dequant8(StagingBuf codes, Scale scale) {
        return gqa_kv_dequant_i8x8_f16(codes, scale);
    }
    static __device__ __forceinline__ int4 dequant8_from(const Code* codes, Scale scale) {
        return gqa_kv_dequant_i8x8_f16_from(codes, scale);
    }
    static __device__ __forceinline__ void dequant16_from(const Code* codes16, Scale scale,
                                                          int4& lo, int4& hi) {
        const uint4 raw = __ldcs(reinterpret_cast<const uint4*>(codes16));
        lo = dequant8(make_int2(raw.x, raw.y), scale);
        hi = dequant8(make_int2(raw.z, raw.w), scale);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t code_index(int physical_page, int kv_head, int d,
                                                              int page_offset) {
        return gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t scale_index(int physical_page, int kv_head,
                                                               int d, int page_offset) {
        return gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, d / kGqaKvQuantGroup,
                                                  page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ void quantize_row(const float* k_vals, const float* v_vals,
                                                        int physical_page, int kv_head,
                                                        int page_offset, Code* cache_k,
                                                        Code* cache_v, Scale* cache_k_scale,
                                                        Scale* cache_v_scale) {
        gqa_g64_quantize_row<KvI8G64, Geometry>(k_vals, v_vals, physical_page, kv_head, page_offset,
                                                cache_k, cache_v, cache_k_scale, cache_v_scale);
    }

    // K/V-plane variants: the symmetric policy serves both planes with the same arithmetic, so
    // each alias delegates to the shared method above (the K8V4 policy overrides the split).
    static __device__ __forceinline__ int4 k_dequant8(StagingBuf codes, Scale scale) {
        return dequant8(codes, scale);
    }
    static __device__ __forceinline__ int4 v_dequant8(StagingBuf codes, Scale scale) {
        return dequant8(codes, scale);
    }
    static __device__ __forceinline__ int4 k_dequant8_from(const Code* codes, Scale scale) {
        return dequant8_from(codes, scale);
    }
    static __device__ __forceinline__ int4 v_dequant8_from(const Code* codes, Scale scale) {
        return dequant8_from(codes, scale);
    }
    static __device__ __forceinline__ void k_dequant16_from(const Code* codes16, Scale scale,
                                                            int4& lo, int4& hi) {
        dequant16_from(codes16, scale, lo, hi);
    }
    static __device__ __forceinline__ void v_dequant16_from(const Code* codes16, Scale scale,
                                                            int4& lo, int4& hi) {
        dequant16_from(codes16, scale, lo, hi);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t k_code_index(int physical_page, int kv_head, int d,
                                                                int page_offset) {
        return code_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t v_code_index(int physical_page, int kv_head, int d,
                                                                int page_offset) {
        return code_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t k_scale_index(int physical_page, int kv_head,
                                                                 int d, int page_offset) {
        return scale_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t v_scale_index(int physical_page, int kv_head,
                                                                 int d, int page_offset) {
        return scale_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
};

// FP8 E4M3, per-64-group scale = abs-max (FP16), code = E4M3(x / scale). The E4M3 code byte is the
// same width as the int8 code byte, so the shared staging/dequant shape (8 codes -> 8 fp16) holds.
struct KvFp8G64 {
    using Code       = std::uint8_t;
    using Scale      = __half;
    using StagingBuf = int2;  // 8 code bytes (8 rotated dimensions)
    using KScale      = Scale;
    using VScale      = Scale;
    using KStagingBuf = StagingBuf;
    using VStagingBuf = StagingBuf;

    static __device__ __forceinline__ Scale scale_from_amax(float amax) {
        return __float2half_rn(amax);
    }
    static __device__ __forceinline__ Code encode(float x, float inv_scale) {
        return ninfer::kv_fp8::e4m3_encode(__fmul_rn(x, inv_scale));
    }
    static __device__ __forceinline__ int4 dequant8(StagingBuf codes, Scale scale) {
        const std::uint8_t* c = reinterpret_cast<const std::uint8_t*>(&codes);
        __half out8[8];
        ninfer::kv_fp8::decode8_to_f16(c, scale, out8);
        return make_int4(*reinterpret_cast<const int*>(out8 + 0),
                         *reinterpret_cast<const int*>(out8 + 2),
                         *reinterpret_cast<const int*>(out8 + 4),
                         *reinterpret_cast<const int*>(out8 + 6));
    }
    static __device__ __forceinline__ int4 dequant8_from(const Code* codes, Scale scale) {
        return dequant8(load_vec<StagingBuf>(codes), scale);
    }
    static __device__ __forceinline__ void dequant16_from(const Code* codes16, Scale scale,
                                                          int4& lo, int4& hi) {
        const uint4 raw = __ldcs(reinterpret_cast<const uint4*>(codes16));
        lo = dequant8(make_int2(raw.x, raw.y), scale);
        hi = dequant8(make_int2(raw.z, raw.w), scale);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t code_index(int physical_page, int kv_head, int d,
                                                              int page_offset) {
        return gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t scale_index(int physical_page, int kv_head,
                                                               int d, int page_offset) {
        return gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, d / kGqaKvQuantGroup,
                                                  page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ void quantize_row(const float* k_vals, const float* v_vals,
                                                        int physical_page, int kv_head,
                                                        int page_offset, Code* cache_k,
                                                        Code* cache_v, Scale* cache_k_scale,
                                                        Scale* cache_v_scale) {
        gqa_g64_quantize_row<KvFp8G64, Geometry>(k_vals, v_vals, physical_page, kv_head, page_offset,
                                                 cache_k, cache_v, cache_k_scale, cache_v_scale);
    }

    static __device__ __forceinline__ int4 k_dequant8(StagingBuf codes, Scale scale) {
        return dequant8(codes, scale);
    }
    static __device__ __forceinline__ int4 v_dequant8(StagingBuf codes, Scale scale) {
        return dequant8(codes, scale);
    }
    static __device__ __forceinline__ int4 k_dequant8_from(const Code* codes, Scale scale) {
        return dequant8_from(codes, scale);
    }
    static __device__ __forceinline__ int4 v_dequant8_from(const Code* codes, Scale scale) {
        return dequant8_from(codes, scale);
    }
    static __device__ __forceinline__ void k_dequant16_from(const Code* codes16, Scale scale,
                                                            int4& lo, int4& hi) {
        dequant16_from(codes16, scale, lo, hi);
    }
    static __device__ __forceinline__ void v_dequant16_from(const Code* codes16, Scale scale,
                                                            int4& lo, int4& hi) {
        dequant16_from(codes16, scale, lo, hi);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t k_code_index(int physical_page, int kv_head, int d,
                                                                int page_offset) {
        return code_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t v_code_index(int physical_page, int kv_head, int d,
                                                                int page_offset) {
        return code_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t k_scale_index(int physical_page, int kv_head,
                                                                 int d, int page_offset) {
        return scale_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t v_scale_index(int physical_page, int kv_head,
                                                                 int d, int page_offset) {
        return scale_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
};

// Trait for the launcher's format-specific guards (e.g. FP8 decode is Volta-TC-only).
template <typename KvCodec>
struct kv_codec_is_fp8 : std::false_type {};
template <>
struct kv_codec_is_fp8<KvFp8G64> : std::true_type {};

} // namespace ninfer::ops