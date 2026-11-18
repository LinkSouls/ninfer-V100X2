#pragma once

// NVFP4 group-16 KV codec policy + staging/quantize helpers for the GQA attention kernels.
//
// NVFP4 is structurally different from the group-64 INT8/FP8 codecs, so it does NOT reuse the g64
// staging (which assumes one code byte per rotated dimension, 8-byte->8-fp16 dequant, and an FP16
// per-64 scale). NVFP4 packs TWO E2M1 codes per byte (code extent = head_dim/2), carries one E4M3
// scale byte per group of 16 (scale extent = head_dim/16), and dequantizes 4 packed bytes -> 8 fp16.
//
// The attention kernels select the codec through a single policy template parameter (the p4 design's
// "kernel 内按模板参数选择 codec"). This header exposes that policy (KvNvfp4G16) with the SAME
// interface as the group-64 policies (gqa_attention_kv_g64_codec.cuh): Code/Scale/StagingBuf types,
// dequant8/dequant8_from/dequant16_from staging, per-dimension code/scale index, and a fused
// quantize_row. The INT8/FP8 policies keep their group-64 bodies verbatim, so those paths stay
// bit-identical; only NVFP4's arithmetic differs.
//
// As with the FP8/INT8 codecs, quantization (append) and dequantization (stage) are FUSED into the
// attention kernels; this header has NO standalone kernel launch (gated by the !__global__ T0
// assertion). The E2M1/E4M3 arithmetic reuses the validated helpers in kv_cache_nvfp4_g16_codec.cuh.

#include "ops/kernel/kv_cache_nvfp4_g16_codec.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/common/memory.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

inline constexpr int kGqaNvfp4HeadDim  = 256;
inline constexpr int kGqaNvfp4Group    = 16;
inline constexpr int kGqaNvfp4Groups   = kGqaNvfp4HeadDim / kGqaNvfp4Group;  // 16
inline constexpr int kGqaNvfp4CodeExt  = kGqaNvfp4HeadDim / 2;              // 128 (2 E2M1/byte)
inline constexpr int kGqaNvfp4ScaleExt = kGqaNvfp4HeadDim / kGqaNvfp4Group; // 16 (E4M3 per 16)

// NVFP4 code plane: 2 E2M1 codes per byte, so dimension d lives in code byte d/2.
template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_nvfp4_code_index(int physical_page, int kv_head,
                                                             int d, int page_offset) {
    return paged_kv_element_offset<kGqaNvfp4CodeExt, Geometry::KVHeads>(physical_page, kv_head,
                                                                        page_offset, d / 2);
}

// NVFP4 scale plane: one E4M3 scale byte per group of 16, so dimension d uses scale group d/16.
template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_nvfp4_scale_index(int physical_page, int kv_head,
                                                              int d, int page_offset) {
    return paged_kv_element_offset<kGqaNvfp4ScaleExt, Geometry::KVHeads>(physical_page, kv_head,
                                                                         page_offset, d / 16);
}

// NVFP4 group-16 codec policy. Code = packed E2M1 bytes (2/byte), Scale = E4M3 byte, StagingBuf =
// 4 packed bytes (8 E2M1 values). dequant8 widens 4 packed bytes times the group E4M3 scale to
// 8 fp16 (an int4) for mma staging; dequant16_from covers the 16-dimension staging chunk of the
// single-QPN / verify5 kernels (two 4-byte halves).
struct KvNvfp4G16 {
    using Code       = std::uint8_t;
    using Scale      = std::uint8_t;
    using StagingBuf = std::uint32_t;  // 4 packed E2M1 bytes (8 values)
    using KScale      = Scale;
    using VScale      = Scale;
    using KStagingBuf = StagingBuf;
    using VStagingBuf = StagingBuf;

    static __device__ __forceinline__ int4 dequant8(StagingBuf codes4, Scale scale) {
        const std::uint8_t* c = reinterpret_cast<const std::uint8_t*>(&codes4);
        __half out8[8];
        const float s = ninfer::kv_nvfp4::decode_nvfp4_e4m3(scale);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 v = ninfer::kv_nvfp4::decode_nvfp4_e2m1_byte(c[i]);
            out8[2 * i]     = __float2half_rn(__fmul_rn(v.x, s));
            out8[2 * i + 1] = __float2half_rn(__fmul_rn(v.y, s));
        }
        return make_int4(*reinterpret_cast<const int*>(out8 + 0),
                         *reinterpret_cast<const int*>(out8 + 2),
                         *reinterpret_cast<const int*>(out8 + 4),
                         *reinterpret_cast<const int*>(out8 + 6));
    }
    static __device__ __forceinline__ int4 dequant8_from(const Code* codes4, Scale scale) {
        return dequant8(load_vec<StagingBuf>(codes4), scale);
    }
    static __device__ __forceinline__ void dequant16_from(const Code* codes8, Scale scale,
                                                          int4& lo, int4& hi) {
        lo = dequant8(load_vec<StagingBuf>(codes8), scale);
        hi = dequant8(load_vec<StagingBuf>(codes8 + 4), scale);
    }

    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t code_index(int physical_page, int kv_head, int d,
                                                              int page_offset) {
        return gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, d, page_offset);
    }
    template <typename Geometry>
    static __device__ __forceinline__ std::int64_t scale_index(int physical_page, int kv_head,
                                                               int d, int page_offset) {
        return gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, d, page_offset);
    }

    // Quantize one D256-rotated row (lane L holds k_vals[r] = dim L + 32*r) into the NVFP4
    // code/scale planes for one (page, kv_head, token): 16 groups of 16, each -> 8 packed bytes
    // (low nibble first) + 1 E4M3 scale byte. For group g the 16 values sit in register h = g>>1
    // of the 16-lane block base = (g&1)*16 (even groups in lanes 0-15, odd in 16-31).
    template <typename Geometry>
    static __device__ __forceinline__ void quantize_row(const float* k_vals, const float* v_vals,
                                                        int physical_page, int kv_head,
                                                        int page_offset, Code* cache_k,
                                                        Code* cache_v, Scale* scale_k,
                                                        Scale* scale_v) {
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const unsigned mask = 0xffffffffu;
#pragma unroll
        for (int g = 0; g < kGqaNvfp4Groups; ++g) {
            const int h    = g >> 1;
            const int base = (g & 1) * 16;
            const bool active = (lane >= base) && (lane < base + 16);
            const float kv   = active ? k_vals[h] : 0.0f;
            const float vv   = active ? v_vals[h] : 0.0f;
            // 16-lane (16-aligned block) abs-max reduce: shfl_xor 1,2,4,8 folds exactly the block.
            float kamax = fabsf(kv), vamax = fabsf(vv);
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) {
                kamax = fmaxf(kamax, __shfl_xor_sync(mask, kamax, off));
                vamax = fmaxf(vamax, __shfl_xor_sync(mask, vamax, off));
            }
            const std::uint8_t ks =
                __nv_cvt_float_to_fp8(kamax > 0.0f ? kamax / 6.0f : 0.0f, __NV_SATFINITE, __NV_E4M3);
            const std::uint8_t vs =
                __nv_cvt_float_to_fp8(vamax > 0.0f ? vamax / 6.0f : 0.0f, __NV_SATFINITE, __NV_E4M3);
            const float kinv = ks != 0 ? 1.0f / ninfer::kv_nvfp4::decode_nvfp4_e4m3(ks) : 0.0f;
            const float vinc = vs != 0 ? 1.0f / ninfer::kv_nvfp4::decode_nvfp4_e4m3(vs) : 0.0f;
            if (!active) { continue; }
            const int pos = lane - base; // 0..15, within-group position
            // The high nibble (position pos+1) lives in lane+1. Fetch it with a shuffle restricted to
            // the ACTIVE 16-lane block. All 16 active lanes call __shfl_sync while converged, with a
            // mask that exactly matches them -- a full-warp (0xffffffff) mask here is undefined
            // behavior on Volta: only the even-position lanes reach the pack, so the diverged
            // odd/inactive lanes never arrive at the shfl and the participating ones deadlock
            // (observed as a 100%-GPU spin on the first NVFP4 append). The source is clamped to the
            // block so the last active lane (base+15) is safe.
            const unsigned block_mask = 0xFFFFu << base;
            const int nsrc            = (lane + 1 < base + 16) ? (lane + 1) : lane;
            const float nb_k = __shfl_sync(block_mask, kv, nsrc);
            const float nb_v = __shfl_sync(block_mask, vv, nsrc);
            if ((pos & 1) == 0) {
                // This lane holds position 2j = pos; the high nibble (position 2j+1) came in via nb_*.
                const std::uint8_t kb =
                    static_cast<std::uint8_t>(ninfer::kv_nvfp4::e2m1_encode(__fmul_rn(kv, kinv)) |
                                              (static_cast<std::uint8_t>(
                                                   ninfer::kv_nvfp4::e2m1_encode(__fmul_rn(nb_k, kinv)))
                                               << 4));
                const std::uint8_t vb =
                    static_cast<std::uint8_t>(ninfer::kv_nvfp4::e2m1_encode(__fmul_rn(vv, vinc)) |
                                              (static_cast<std::uint8_t>(
                                                   ninfer::kv_nvfp4::e2m1_encode(__fmul_rn(nb_v, vinc)))
                                               << 4));
                const int dim = g * kGqaNvfp4Group + pos; // pos is even = 2j
                const int cb  = static_cast<int>(gqa_nvfp4_code_index<Geometry>(physical_page, kv_head,
                                                                               dim, page_offset));
                cache_k[cb] = kb;
                cache_v[cb] = vb;
                if (pos == 0) {
                    const int sb = static_cast<int>(gqa_nvfp4_scale_index<Geometry>(physical_page,
                                                                                   kv_head, dim,
                                                                                   page_offset));
                    scale_k[sb] = ks;
                    scale_v[sb] = vs;
                }
            }
        }
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
    static __device__ __forceinline__ void k_dequant16_from(const Code* codes8, Scale scale,
                                                            int4& lo, int4& hi) {
        dequant16_from(codes8, scale, lo, hi);
    }
    static __device__ __forceinline__ void v_dequant16_from(const Code* codes8, Scale scale,
                                                            int4& lo, int4& hi) {
        dequant16_from(codes8, scale, lo, hi);
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

// Trait for the launcher's format-specific guards (NVFP4 decode is Volta-TC-only, like FP8).
template <typename KvCodec>
struct kv_codec_is_nvfp4 : std::false_type {};
template <>
struct kv_codec_is_nvfp4<KvNvfp4G16> : std::true_type {};

} // namespace ninfer::ops