#pragma once

// K8V4 (asymmetric FP8-key / NVFP4-value) KV codec policy for the GQA attention kernels.
//
// The key plane uses the FP8 E4M3 group-64 geometry (1-byte E4M3 code per rotated dimension +
// one FP16 scale per 64) and the value plane uses the NVFP4 group-16 geometry (two E2M1 codes
// per byte + one E4M3 scale byte per 16). The two planes are INDEPENDENT: each carries its own
// code/scale dtype and extent (kv_cache_plane_geometry(Fp8KeyNvfp4Value)), so a single symmetric
// codec policy cannot express the format. This policy composes the two existing policies:
//   - K-side methods (k_dequant*, k_code_index, k_scale_index) delegate to KvFp8G64,
//   - V-side methods (v_dequant*, v_code_index, v_scale_index) delegate to KvNvfp4G16.
// The attention kernels select the codec through the single policy template parameter (the p4
// design's "kernel 内按模板参数选择 codec") and address each plane through its K/V variant, so
// the symmetric formats (INT8/FP8/NVFP4) keep bit-identical behavior (their K/V variants are
// aliases of the shared method).
//
// quantize_row (append) quantizes the K row with the FP8 group-64 arithmetic (full-warp abs-max
// reduce, 4 groups of 64, FP16 scale) and the V row with the NVFP4 group-16 arithmetic (16-lane
// block abs-max reduce, 16 groups of 16, E4M3 scale, E2M1 pack via the block-restricted
// __shfl_sync -- the same Volta-safe pattern as KvNvfp4G16::quantize_row). As with the other
// codecs, quant/dequant stay FUSED into the attention kernels: no standalone kernel launch (the
// !__global__ T0 assertion holds).

#include "ops/kernel/gqa_attention_kv_g64_codec.cuh"
#include "ops/kernel/gqa_attention_nvfp4_g16_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

// K-only FP8 group-64 quantize body (the K half of gqa_g64_quantize_row): one D256-rotated row
// (lane l holds dims l + 32*r) -> four per-64-group E4M3 codes (two per lane) + one FP16 scale per
// group (lane 0). Full-warp collectives only -- no branch divergence around the warp_max.
template <typename Geometry>
__device__ __forceinline__ void k8v4_quantize_row_k_fp8(const float* k_vals, int physical_page,
                                                        int kv_head, int page_offset,
                                                        std::uint8_t* cache_k,
                                                        __half* cache_k_scale) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    constexpr unsigned FullMask = 0xffffffffu;
#pragma unroll
    for (int grp = 0; grp < kGqaKvQuantGroups; ++grp) {
        const int d0   = grp * kGqaKvQuantGroup + lane;
        const int d1   = d0 + 32;
        const float kv0 = k_vals[2 * grp];
        const float kv1 = k_vals[2 * grp + 1];
        const float kamax = warp_max(fmaxf(fabsf(kv0), fabsf(kv1)), FullMask);
        const __half ksh = __float2half_rn(kamax);
        const float k_inv = __half2float(ksh) > 0.0f ? 1.0f / __half2float(ksh) : 0.0f;
        cache_k[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
            ninfer::kv_fp8::e4m3_encode(__fmul_rn(kv0, k_inv));
        cache_k[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
            ninfer::kv_fp8::e4m3_encode(__fmul_rn(kv1, k_inv));
        if (lane == 0) {
            const std::int64_t so =
                gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, grp, page_offset);
            cache_k_scale[so] = ksh;
        }
    }
}

// V-only NVFP4 group-16 quantize body (the V half of KvNvfp4G16::quantize_row): 16 groups of 16,
// each -> 8 packed E2M1 bytes (low nibble first) + 1 E4M3 scale byte. The abs-max reduce is a
// full-warp 16-lane-block fold (all 32 lanes participate); the E2M1 pack fetches the high nibble
// with a __shfl_sync restricted to the ACTIVE 16-lane block (a full-warp mask there is undefined
// behavior on Volta: only the even-position lanes reach the pack, so the diverged lanes never
// arrive and the participants deadlock).
template <typename Geometry>
__device__ __forceinline__ void k8v4_quantize_row_v_nvfp4(const float* v_vals, int physical_page,
                                                          int kv_head, int page_offset,
                                                          std::uint8_t* cache_v,
                                                          std::uint8_t* cache_v_scale) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const unsigned mask = 0xffffffffu;
#pragma unroll
    for (int g = 0; g < kGqaNvfp4Groups; ++g) {
        const int h    = g >> 1;
        const int base = (g & 1) * 16;
        const bool active = (lane >= base) && (lane < base + 16);
        const float vv   = active ? v_vals[h] : 0.0f;
        float vamax = fabsf(vv);
#pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
            vamax = fmaxf(vamax, __shfl_xor_sync(mask, vamax, off));
        }
        const std::uint8_t vs =
            __nv_cvt_float_to_fp8(vamax > 0.0f ? vamax / 6.0f : 0.0f, __NV_SATFINITE, __NV_E4M3);
        const float vinc = vs != 0 ? 1.0f / ninfer::kv_nvfp4::decode_nvfp4_e4m3(vs) : 0.0f;
        if (!active) { continue; }
        const int pos = lane - base; // 0..15, within-group position
        const unsigned block_mask = 0xFFFFu << base;
        const int nsrc            = (lane + 1 < base + 16) ? (lane + 1) : lane;
        const float nb_v = __shfl_sync(block_mask, vv, nsrc);
        if ((pos & 1) == 0) {
            const std::uint8_t vb =
                static_cast<std::uint8_t>(ninfer::kv_nvfp4::e2m1_encode(__fmul_rn(vv, vinc)) |
                                          (static_cast<std::uint8_t>(
                                               ninfer::kv_nvfp4::e2m1_encode(__fmul_rn(nb_v, vinc)))
                                           << 4));
            const int dim = g * kGqaNvfp4Group + pos; // pos is even = 2j
            const int cb  = static_cast<int>(gqa_nvfp4_code_index<Geometry>(physical_page, kv_head,
                                                                           dim, page_offset));
            cache_v[cb] = vb;
            if (pos == 0) {
                const int sb = static_cast<int>(gqa_nvfp4_scale_index<Geometry>(physical_page,
                                                                               kv_head, dim,
                                                                               page_offset));
                cache_v_scale[sb] = vs;
            }
        }
    }
}

// Asymmetric K8V4 codec policy: key plane = FP8 E4M3 group-64 (KvFp8G64), value plane = NVFP4
// group-16 (KvNvfp4G16). Both code planes are 1 byte wide (E4M3 byte / packed-E2M1 byte), so a
// single Code type suffices; the scale and staging types differ per plane.
struct KvK8V4 {
    using Code       = std::uint8_t;
    using KScale     = __half;      // FP8 group-64 key scale
    using VScale     = std::uint8_t; // NVFP4 group-16 value scale (E4M3 byte)
    using Scale      = KScale; // back-compat: the key-plane scale
    using KStagingBuf = int2;  // 8 E4M3 code bytes -> 8 fp16
    using VStagingBuf = std::uint32_t;  // 4 packed E2M1 bytes (8 values)
    using StagingBuf  = KStagingBuf; // back-compat: the key-plane staging

    // --- key plane: FP8 E4M3 group-64 (delegates to KvFp8G64) --------------------------------
    static __device__ __forceinline__ int4 k_dequant8(KStagingBuf codes, KScale scale) {
        return KvFp8G64::dequant8(codes, scale);
    }
    static __device__ __forceinline__ int4 k_dequant8_from(const Code* codes, KScale scale) {
        return KvFp8G64::dequant8_from(codes, scale);
    }
    static __device__ __forceinline__ void k_dequant16_from(const Code* codes16, KScale scale,
                                                            int4& lo, int4& hi) {
        KvFp8G64::dequant16_from(codes16, scale, lo, hi);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t k_code_index(int physical_page, int kv_head, int d,
                                                                int page_offset) {
        return KvFp8G64::code_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t k_scale_index(int physical_page, int kv_head,
                                                                 int d, int page_offset) {
        return KvFp8G64::scale_index<Geometry>(physical_page, kv_head, d, page_offset);
    }

    // --- value plane: NVFP4 group-16 (delegates to KvNvfp4G16) ------------------------------
    static __device__ __forceinline__ int4 v_dequant8(VStagingBuf codes, VScale scale) {
        return KvNvfp4G16::dequant8(codes, scale);
    }
    static __device__ __forceinline__ int4 v_dequant8_from(const Code* codes, VScale scale) {
        return KvNvfp4G16::dequant8_from(codes, scale);
    }
    static __device__ __forceinline__ void v_dequant16_from(const Code* codes8, VScale scale,
                                                            int4& lo, int4& hi) {
        KvNvfp4G16::dequant16_from(codes8, scale, lo, hi);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t v_code_index(int physical_page, int kv_head, int d,
                                                                int page_offset) {
        return KvNvfp4G16::code_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t v_scale_index(int physical_page, int kv_head,
                                                                 int d, int page_offset) {
        return KvNvfp4G16::scale_index<Geometry>(physical_page, kv_head, d, page_offset);
    }

    // --- append: K row via FP8 g64, V row via NVFP4 g16 (both fused, no separate kernels) ----
    template <typename Geometry>
    static __device__ __forceinline__ void quantize_row(const float* k_vals, const float* v_vals,
                                                        int physical_page, int kv_head,
                                                        int page_offset, Code* cache_k,
                                                        Code* cache_v, KScale* cache_k_scale,
                                                        VScale* cache_v_scale) {
        k8v4_quantize_row_k_fp8<Geometry>(k_vals, physical_page, kv_head, page_offset, cache_k,
                                          cache_k_scale);
        k8v4_quantize_row_v_nvfp4<Geometry>(v_vals, physical_page, kv_head, page_offset, cache_v,
                                            cache_v_scale);
    }
};

// Trait for the launcher's format-specific guards (K8V4 decode is Volta-TC-only, like FP8/NVFP4).
template <typename KvCodec>
struct kv_codec_is_k8v4 : std::false_type {};
template <>
struct kv_codec_is_k8v4<KvK8V4> : std::true_type {};

} // namespace ninfer::ops