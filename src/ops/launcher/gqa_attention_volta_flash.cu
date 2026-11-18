// ninfer::ops::detail - Volta (sm_70) flash-attention prefill route.
//
// Drives the vendored llama.cpp MMA flash-attention kernel
// (third_party/llama_cpp_fattn, pinned at 62bf73d25) from a ninfer-native
// launcher. Rationale, phase gates and measurements: the V100 implementation and
// the V100 implementation.
//
// This is the only translation unit that sees the vendored headers, which is
// why the vendored include directory is scoped to this file in CMake: the
// vendored tree contains `common.cuh` and `mma.cuh`, names that also exist
// under src/ops.
//
// The vendored kernel reads K/V as plain strided FP16 and Q as strided FP32,
// while ninfer stores BF16 Q/out and a paged BF16 KV cache. Rather than inject
// page lookups into the kernel's hot loop, this file stages: append the whole
// width into the paged cache, gather the visible key range into a contiguous
// FP16 buffer once per layer, convert Q per Q-block, run the kernel, convert the
// FP32 result back to BF16.

#include "fattn-mma-f16.cuh" // vendored; must precede ninfer headers (defines WARP_SIZE)

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/kv_cache_fp8_g64_codec.cuh"      // ninfer::kv_fp8 (E4M3 encode/decode)
#include "ops/kernel/kv_cache_nvfp4_g16_codec.cuh"    // ninfer::kv_nvfp4 (E2M1/E4M3)
#include "ops/kernel/gqa_attention_kv_g64_codec.cuh"   // KvFp8G64 codec policy
#include "ops/kernel/gqa_attention_nvfp4_g16_codec.cuh" // KvNvfp4G16 codec policy
#include "ops/kernel/gqa_attention_k8v4_codec.cuh"     // KvK8V4 codec policy
#include "ops/kernel/hadamard_d256.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/launcher/gqa_attention.h"
#include "ops/launcher/gqa_attention_volta_splitd.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Spelled out rather than taken from gqa_attention_decode.cuh's kGqaHeadDim:
// including that header would pull in ops/common/mma.cuh and ops/common/warp.cuh,
// whose WARP_SIZE collides with the vendored common.cuh. Guarded by the
// static_asserts in VoltaFlashTiling below.
constexpr int kHeadDim = 256;
constexpr int kDKQ     = 256;
constexpr int kDV      = 256;

// The kernel picks ncols2 from the GQA ratio, and this table mirrors that choice
// per registered geometry:
//   24q/4kv -> ratio 6: 6 % 8 != 0, 6 % 4 != 0, then DKQ <= 256 and 6 % 2 == 0
//                       -> ncols2 = 2
//   16q/2kv -> ratio 8: 8 % 8 == 0                              -> ncols2 = 8
// ncols1 is then chosen so ncols1*ncols2 == 32, the smallest tile Volta compiles.
// A causal mask tensor is mandatory in both cases: mask == nullptr is only legal
// at ncols2 == 1.
template <typename Geometry>
struct VoltaFlashTiling;

template <>
struct VoltaFlashTiling<Gqa27Geometry> {
    static constexpr int ncols2 = 2;
    static constexpr int ncols1 = 16;
};

template <>
struct VoltaFlashTiling<Gqa27Tp2Geometry> : VoltaFlashTiling<Gqa27Geometry> {};
template <>
struct VoltaFlashTiling<Gqa27Tp4Geometry> : VoltaFlashTiling<Gqa27Geometry> {};

template <>
struct VoltaFlashTiling<Gqa35Geometry> {
    static constexpr int ncols2 = 8;
    static constexpr int ncols1 = 4;
};

template <typename Geometry>
struct VoltaFlashParams {
    static constexpr int kQHeads  = Geometry::QHeads;
    static constexpr int kKVHeads = Geometry::KVHeads;
    static constexpr int kGroup   = Geometry::GroupSize;
    static constexpr int kNcols2  = VoltaFlashTiling<Geometry>::ncols2;
    static constexpr int kNcols1  = VoltaFlashTiling<Geometry>::ncols1;
    static constexpr int kNcols   = kNcols1 * kNcols2;

    static_assert(kQHeads / kKVHeads == kGroup, "gqa ratio mismatch");
    static_assert(kGroup % kNcols2 == 0, "ncols2 must divide the GQA group");
    static_assert(kNcols >= 32, "Volta compiles no tile smaller than 32 columns");
};

// The key extent handed to the kernel is padded to FATTN_KQ_STRIDE. Upstream
// always satisfies this (llama.cpp pads its KV cache to 256 whenever FA is on), so
// the kernel's last key tile is free to read a whole tile past the true key count.
// Without the padding those reads would run off the end of a mask row and into the
// next one, which a multiple-of-32 key count happens to hide.
constexpr int kKeyPad = FATTN_KQ_STRIDE;

constexpr int round_up_keys(int n) { return ((n + kKeyPad - 1) / kKeyPad) * kKeyPad; }

// ---------------------------------------------------------------------------
// Staging kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ const std::int32_t* select_block_table(
        const std::int32_t* block_tables, const std::int32_t* table_rows, int logical_pages) {
    return block_tables + static_cast<std::int64_t>(table_rows[0]) * logical_pages;
}

// Append the whole width's K/V into the paged cache in one launch. The chunked
// route does this per 5-token chunk; a flash tiling needs the full range resident
// before the first Q-block runs, and the causal mask is what keeps a query from
// seeing a key it must not.
template <int kKVHeads>
__global__ void volta_flash_append_kv_kernel(
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ table_rows,
        __nv_bfloat16* __restrict__ k_pages, __nv_bfloat16* __restrict__ v_pages,
        const std::int32_t* __restrict__ block_tables, int logical_pages, int width) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (t >= width) { return; }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const std::int32_t position     = positions[t];
    const std::int64_t dst =
        paged_kv_element_offset<kHeadDim, kKVHeads>(block_table, h, position, d);
    const std::int64_t src =
        static_cast<std::int64_t>(d) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * t);

    k_pages[dst] = k[src];
    v_pages[dst] = v[src];
}

__device__ __forceinline__ float volta_flash_warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

// BF16 input -> paged INT8-G64 cache. One warp owns the full D256 (token, kv-head)
// row: lane l holds dims l + 32*r in values[r] (the hadamard_d256.cuh mapping). K and V
// are rotated in registers, then quantized per 64-d group with the cache's public
// absmax/127 contract -- group g's components are values[2g] (dim 64g+lane) and
// values[2g+1] (dim 64g+32+lane), so each group's absmax is a full-warp reduction over
// two per-lane values. The code/scale layout and counts are the unrotated path's; only
// the numerics differ (see hadamard_d256.cuh).
template <int kKVHeads>
__launch_bounds__(256) __global__ void volta_flash_append_kv_i8_kernel(
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ table_rows,
        std::int8_t* __restrict__ k_pages, std::int8_t* __restrict__ v_pages,
        half* __restrict__ k_scales, half* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, int logical_pages, int width) {
    constexpr int kWarps = 8;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int unit       = static_cast<int>(blockIdx.x) * kWarps + warp;
    const int units      = width * kKVHeads;
    if (unit >= units) { return; }

    const int h     = unit % kKVHeads;
    const int token = unit / kKVHeads;
    const std::int64_t row_base = kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * token);
    float k_vals[8], v_vals[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_vals[r]   = __bfloat162float(k[row_base + d]);
        v_vals[r]   = __bfloat162float(v[row_base + d]);
    }
    gqa_hadamard_d256_inplace(k_vals, lane);
    gqa_hadamard_d256_inplace(v_vals, lane);

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const int position              = positions[token];
    int physical_page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page     = __shfl_sync(0xffffffffu, physical_page, 0);
    const int page_offset = position & kPagedKVPageMask;
#pragma unroll
    for (int group = 0; group < kGqaKvQuantGroups; ++group) {
        const int d0 = group * kGqaKvQuantGroup + lane;
        const float kv0 = k_vals[2 * group];
        const float kv1 = k_vals[2 * group + 1];
        const float vv0 = v_vals[2 * group];
        const float vv1 = v_vals[2 * group + 1];
        const float kamax = volta_flash_warp_max(fmaxf(fabsf(kv0), fabsf(kv1)));
        const float vamax = volta_flash_warp_max(fmaxf(fabsf(vv0), fabsf(vv1)));
        const half ks  = __float2half_rn(kamax > 0.0f ? kamax / 127.0f : 0.0f);
        const half vs  = __float2half_rn(vamax > 0.0f ? vamax / 127.0f : 0.0f);
        const float ksf = __half2float(ks);
        const float vsf = __half2float(vs);
        const float ki  = ksf > 0.0f ? 1.0f / ksf : 0.0f;
        const float vi  = vsf > 0.0f ? 1.0f / vsf : 0.0f;

        const std::int64_t code_base = paged_kv_element_offset<kHeadDim, kKVHeads>(
            physical_page, h, page_offset, group * kGqaKvQuantGroup);
        k_pages[code_base + lane]      = gqa_kv_quant_code(kv0, ki);
        k_pages[code_base + lane + 32] = gqa_kv_quant_code(kv1, ki);
        v_pages[code_base + lane]      = gqa_kv_quant_code(vv0, vi);
        v_pages[code_base + lane + 32] = gqa_kv_quant_code(vv1, vi);
        if (lane == 0) {
            const std::int64_t scale_offset = paged_kv_element_offset<kGqaKvQuantGroups, kKVHeads>(
                physical_page, h, page_offset, group);
            k_scales[scale_offset] = ks;
            v_scales[scale_offset] = vs;
        }
    }
}

// The codec policies address their code/scale planes through a Geometry's KVHeads member; the
// flash tiling only carries the head count, so this minimal adapter feeds their index/quantize
// helpers (KvFp8G64 / KvNvfp4G16 / KvK8V4) without pulling in a full registered geometry.
template <int NHeads>
struct FlashGeo {
    static constexpr int KVHeads = NHeads;
};

// FP8/NVFP4/K8V4 append: the same warp-per-row D256 structure as the INT8 kernel above (one warp
// rotates and quantizes one (token, kv-head) row in registers), but the per-format quantize is
// delegated to the codec policy's quantize_row -- K8V4 splits K->FP8 g64 / V->NVFP4 g16, the
// symmetric policies do both planes identically. The code/scale layout and counts come from the
// policy's index functions, so they match that format's layout plan (kv_cache_plane_geometry).
// Quant stays fused into the append (no standalone kernel); the BF16 and INT8 branches above are
// untouched, so those formats stay bit-identical.
template <int kKVHeads, typename KvCodec>
__launch_bounds__(256) __global__ void volta_flash_append_kv_codec_kernel(
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ table_rows,
        typename KvCodec::Code* __restrict__ k_pages, typename KvCodec::Code* __restrict__ v_pages,
        typename KvCodec::KScale* __restrict__ k_scales, typename KvCodec::VScale* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, int logical_pages, int width) {
    constexpr int kWarps = 8;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int unit       = static_cast<int>(blockIdx.x) * kWarps + warp;
    const int units      = width * kKVHeads;
    if (unit >= units) { return; }

    const int h     = unit % kKVHeads;
    const int token = unit / kKVHeads;
    const std::int64_t row_base = kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * token);
    float k_vals[8], v_vals[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_vals[r]   = __bfloat162float(k[row_base + d]);
        v_vals[r]   = __bfloat162float(v[row_base + d]);
    }
    gqa_hadamard_d256_inplace(k_vals, lane);
    gqa_hadamard_d256_inplace(v_vals, lane);

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const int position              = positions[token];
    int physical_page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page     = __shfl_sync(0xffffffffu, physical_page, 0);
    const int page_offset = position & kPagedKVPageMask;

    KvCodec::template quantize_row<FlashGeo<kKVHeads>>(k_vals, v_vals, physical_page, h, page_offset,
                                              k_pages, v_pages, k_scales, v_scales);
}

// Paged BF16 -> contiguous FP16, one launch per layer over the whole visible key
// range. ~50 MB of traffic per layer at 12K against the seconds the chunked route
// spends re-walking the key range; see the cost note in the plan.
template <int kKVHeads>
__global__ void volta_flash_gather_kv_kernel(
        const __nv_bfloat16* __restrict__ k_pages, const __nv_bfloat16* __restrict__ v_pages,
        const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
        int logical_pages, half* __restrict__ k_out, half* __restrict__ v_out, int n_kv,
        int n_kv_padded) {
    const int key = blockIdx.x;
    const int h   = blockIdx.y;
    const int d   = threadIdx.x;
    if (key >= n_kv_padded) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * key);

    // Padding rows are written as zero rather than left uninitialised: the mask
    // gives them weight zero, and 0 * NaN would still be NaN.
    if (key >= n_kv) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const std::int64_t src = paged_kv_element_offset<kHeadDim, kKVHeads>(block_table, h, key, d);

    k_out[dst] = __float2half(__bfloat162float(k_pages[src]));
    v_out[dst] = __float2half(__bfloat162float(v_pages[src]));
}

// Paged INT8-G64 -> contiguous FP16. Dequantizing once here preserves the
// flash kernel as the already-qualified arithmetic boundary and avoids the
// ChunkedSmallT path re-reading the full prefix for every five query tokens.
template <int kKVHeads>
__global__ void volta_flash_gather_kv_i8_kernel(
        const std::int8_t* __restrict__ k_pages, const std::int8_t* __restrict__ v_pages,
        const half* __restrict__ k_scales, const half* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
        int logical_pages, half* __restrict__ k_out, half* __restrict__ v_out, int n_kv,
        int n_kv_padded) {
    const int key = blockIdx.x;
    const int h   = blockIdx.y;
    const int d   = threadIdx.x;
    if (key >= n_kv_padded) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * key);
    if (key >= n_kv) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const std::int64_t code = paged_kv_element_offset<kHeadDim, kKVHeads>(block_table, h, key, d);
    const int group         = d / kGqaKvQuantGroup;
    const std::int64_t scale =
        paged_kv_element_offset<kGqaKvQuantGroups, kKVHeads>(block_table, h, key, group);
    k_out[dst] = __float2half(static_cast<float>(k_pages[code]) * __half2float(k_scales[scale]));
    v_out[dst] = __float2half(static_cast<float>(v_pages[code]) * __half2float(v_scales[scale]));
}

// Paged FP8-E4M3-G64 -> contiguous FP16. Both planes share the INT8 group-64 geometry (one code
// byte per rotated dimension + one FP16 scale per 64); only the code<->value arithmetic differs:
// value = E4M3(code) * scale, matching the validated FP8 mma staging bit-for-bit.
template <int kKVHeads>
__global__ void volta_flash_gather_kv_fp8_kernel(
        const std::uint8_t* __restrict__ k_pages, const std::uint8_t* __restrict__ v_pages,
        const half* __restrict__ k_scales, const half* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
        int logical_pages, half* __restrict__ k_out, half* __restrict__ v_out, int n_kv,
        int n_kv_padded) {
    const int key = blockIdx.x;
    const int h   = blockIdx.y;
    const int d   = threadIdx.x;
    if (key >= n_kv_padded) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * key);
    if (key >= n_kv) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const std::int64_t code = paged_kv_element_offset<kHeadDim, kKVHeads>(block_table, h, key, d);
    const int group         = d / kGqaKvQuantGroup;
    const std::int64_t scale =
        paged_kv_element_offset<kGqaKvQuantGroups, kKVHeads>(block_table, h, key, group);
    k_out[dst] = __float2half_rn(
        __fmul_rn(ninfer::kv_fp8::e4m3_decode(k_pages[code]), __half2float(k_scales[scale])));
    v_out[dst] = __float2half_rn(
        __fmul_rn(ninfer::kv_fp8::e4m3_decode(v_pages[code]), __half2float(v_scales[scale])));
}

// Paged NVFP4-G16 -> contiguous FP16. Each plane packs two E2M1 codes per byte (dimension d lives
// in byte d/2, nibble d&1) plus one E4M3 scale byte per 16; value = E2M1(nibble) * E4M3(scale),
// the same decode the validated NVFP4 mma staging uses.
template <int kKVHeads>
__global__ void volta_flash_gather_kv_nvfp4_kernel(
        const std::uint8_t* __restrict__ k_pages, const std::uint8_t* __restrict__ v_pages,
        const std::uint8_t* __restrict__ k_scales, const std::uint8_t* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
        int logical_pages, half* __restrict__ k_out, half* __restrict__ v_out, int n_kv,
        int n_kv_padded) {
    const int key = blockIdx.x;
    const int h   = blockIdx.y;
    const int d   = threadIdx.x;
    if (key >= n_kv_padded) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * key);
    if (key >= n_kv) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    const std::int64_t code =
        paged_kv_element_offset<kGqaNvfp4CodeExt, kKVHeads>(block_table, h, key, d / 2);
    const int group16 = d / kGqaNvfp4Group;
    const std::int64_t scale =
        paged_kv_element_offset<kGqaNvfp4ScaleExt, kKVHeads>(block_table, h, key, group16);
    const float ksf = ninfer::kv_nvfp4::decode_nvfp4_e4m3(k_scales[scale]);
    const float2 kc = ninfer::kv_nvfp4::decode_nvfp4_e2m1_byte(k_pages[code]);
    k_out[dst] = __float2half_rn(__fmul_rn((d & 1) ? kc.y : kc.x, ksf));
    const float vsf = ninfer::kv_nvfp4::decode_nvfp4_e4m3(v_scales[scale]);
    const float2 vc = ninfer::kv_nvfp4::decode_nvfp4_e2m1_byte(v_pages[code]);
    v_out[dst] = __float2half_rn(__fmul_rn((d & 1) ? vc.y : vc.x, vsf));
}

// Paged K8V4 (asymmetric) -> contiguous FP16. The key plane is FP8 E4M3 group-64 (code byte at
// dimension d, FP16 scale at d/64) and the value plane is NVFP4 group-16 (packed byte at d/2,
// E4M3 scale byte at d/16); each plane dequantizes with its own codec, independently.
template <int kKVHeads>
__global__ void volta_flash_gather_kv_k8v4_kernel(
        const std::uint8_t* __restrict__ k_pages, const std::uint8_t* __restrict__ v_pages,
        const half* __restrict__ k_scales, const std::uint8_t* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
        int logical_pages, half* __restrict__ k_out, half* __restrict__ v_out, int n_kv,
        int n_kv_padded) {
    const int key = blockIdx.x;
    const int h   = blockIdx.y;
    const int d   = threadIdx.x;
    if (key >= n_kv_padded) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kHeadDim * (h + static_cast<std::int64_t>(kKVHeads) * key);
    if (key >= n_kv) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, logical_pages);
    // Key plane: FP8 E4M3 group-64.
    const std::int64_t kcode = paged_kv_element_offset<kHeadDim, kKVHeads>(block_table, h, key, d);
    const int kgroup         = d / kGqaKvQuantGroup;
    const std::int64_t kscale =
        paged_kv_element_offset<kGqaKvQuantGroups, kKVHeads>(block_table, h, key, kgroup);
    k_out[dst] = __float2half_rn(
        __fmul_rn(ninfer::kv_fp8::e4m3_decode(k_pages[kcode]), __half2float(k_scales[kscale])));
    // Value plane: NVFP4 group-16.
    const std::int64_t vcode =
        paged_kv_element_offset<kGqaNvfp4CodeExt, kKVHeads>(block_table, h, key, d / 2);
    const int vgroup16 = d / kGqaNvfp4Group;
    const std::int64_t vscale =
        paged_kv_element_offset<kGqaNvfp4ScaleExt, kKVHeads>(block_table, h, key, vgroup16);
    const float vsf = ninfer::kv_nvfp4::decode_nvfp4_e4m3(v_scales[vscale]);
    const float2 vc = ninfer::kv_nvfp4::decode_nvfp4_e2m1_byte(v_pages[vcode]);
    v_out[dst] = __float2half_rn(__fmul_rn((d & 1) ? vc.y : vc.x, vsf));
}

// BF16 [256, QHeads, T] -> FP32, same element order (the kernel's Q and ninfer's q
// agree on layout; only the element type differs).
__global__ void volta_flash_convert_q_kernel(const __nv_bfloat16* __restrict__ q,
                                             float* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = __bfloat162float(q[i]);
}

// INT8-only Q staging: same layout, but each (q_head, token) row is a D256 row that
// must carry the same rotation the K rows carry (Q·(HK)ᵀ == (HQ)·Kᵀ), so one warp
// rotates one row in registers before the FP32 store. Row `row` of the block slice is
// (qh, t) with qh == row % QHeads, t == row / QHeads; its 256 elements are contiguous
// at row * kHeadDim. BF16 uses the plain elementwise kernel above.
__global__ void volta_flash_convert_q_i8_kernel(const __nv_bfloat16* __restrict__ q,
                                                float* __restrict__ out, int rows) {
    constexpr int kWarpsPerBlock = 8;
    const int row  = static_cast<int>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    if (row >= rows) { return; }
    const int lane = threadIdx.x & 31;
    const std::int64_t base = static_cast<std::int64_t>(row) * kHeadDim;
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { values[r] = __bfloat162float(q[base + lane + 32 * r]); }
    gqa_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) { out[base + lane + 32 * r] = values[r]; }
}

// FP32 -> BF16 on the way back out, same element order.
__global__ void volta_flash_convert_out_kernel(const float* __restrict__ in,
                                               __nv_bfloat16* __restrict__ out,
                                               std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = __float2bfloat16(in[i]);
}

// INT8-only output epilogue: the K/V rows were rotated before quantization, so this
// attention result is in rotated-V coordinates. The transform is an involution, so the
// same in-place map undoes it; one warp rotates one (q_head, token) row, fused into the
// existing FP32 -> BF16 conversion (no extra kernel, no intermediate tensor). It runs
// after every stream-K branch / split-D output has been merged and normalized, so the
// undo happens exactly once. BF16 uses the plain elementwise kernel above.
__global__ void volta_flash_convert_out_i8_kernel(const float* __restrict__ in,
                                                  __nv_bfloat16* __restrict__ out, int rows) {
    constexpr int kWarpsPerBlock = 8;
    const int row  = static_cast<int>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    if (row >= rows) { return; }
    const int lane = threadIdx.x & 31;
    const std::int64_t base = static_cast<std::int64_t>(row) * kHeadDim;
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { values[r] = in[base + lane + 32 * r]; }
    gqa_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) { out[base + lane + 32 * r] = __float2bfloat16(values[r]); }
}

// Causal mask for one Q-block, built from the device-side positions rather than
// inferred from the envelope, so a non-sequential position layout cannot silently
// produce a wrong mask. Padding rows are fully masked; their outputs are discarded.
__global__ void volta_flash_build_mask_kernel(const std::int32_t* __restrict__ positions,
                                              int token_begin, int rows_valid,
                                              half* __restrict__ mask, int n_kv, int mask_stride) {
    const int row = blockIdx.x;
    const std::int32_t last_visible = row < rows_valid ? positions[token_begin + row] : -1;

    half* mask_row = mask + static_cast<std::int64_t>(row) * mask_stride;
    for (int key = threadIdx.x; key < n_kv; key += blockDim.x) {
        mask_row[key] = key <= last_visible ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

// ---------------------------------------------------------------------------
// Kernel launch: launch_fattn's stream-K decomposition, without the ggml types.
// ---------------------------------------------------------------------------

struct FlashLaunchConfig {
    int    nthreads       = 0;
    int    nwarps         = 0;
    int    nbatch_fa      = 0;
    size_t nbytes_shared  = 0;
    int    blocks_per_sm  = 0;
    int    nsm            = 0;
};

// One cached config per geometry: the shared-memory and occupancy figures depend
// only on the tiling, so they are resolved once per instantiation.
template <typename Geometry>
const FlashLaunchConfig& flash_launch_config() {
    using P = VoltaFlashParams<Geometry>;

    static std::mutex mutex;
    static std::map<int, FlashLaunchConfig> configs;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = configs.find(device); found != configs.end()) { return found->second; }
    const FlashLaunchConfig config = [device] {
        FlashLaunchConfig c;

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        c.nsm = prop.multiProcessorCount;

        const int cc = prop.major * 100 + prop.minor * 10;

        const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (kDKQ, kDV, P::kNcols, cc);
        const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (kDKQ, kDV, P::kNcols, cc);
        const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(kDKQ, kDV, P::kNcols, cc);
        const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (kDKQ, kDV, P::kNcols, cc);
        const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (kDKQ, kDV, P::kNcols1, P::kNcols2, cc);

        c.nthreads  = ggml_cuda_fattn_mma_get_nthreads (kDKQ, kDV, P::kNcols, cc);
        c.nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(kDKQ, kDV, P::kNcols, cc);
        c.nwarps    = c.nthreads / WARP_SIZE;

        const int cols_per_warp = std::min(P::kNcols, get_cols_per_warp(cc));

        const size_t shared_KV_1stage = size_t(c.nbatch_fa) * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
        const size_t shared_KV_2stage = size_t(c.nbatch_fa) *         (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
        const size_t shared_Q         = size_t(P::kNcols)      * (kDKQ/2 + 4)                            * sizeof(half2);
        const size_t shared_mask      = size_t(P::kNcols1)     * (c.nbatch_fa/2 + 4)                     * sizeof(half2);
        const size_t shared_combine   = size_t(c.nwarps)*cols_per_warp * (nbatch_combine + 4)         * sizeof(half2);
        const size_t shared_KV        = nstages <= 1 ? shared_KV_1stage : shared_KV_2stage;

        c.nbytes_shared = std::max(shared_combine, Q_in_reg
            ? std::max(shared_Q, shared_KV + shared_mask)
            :          shared_Q + shared_KV + shared_mask);

        auto kernel = flash_attn_ext_f16<kDKQ, kDV, P::kNcols1, P::kNcols2, false, false>;
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, c.nbytes_shared));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &c.blocks_per_sm, reinterpret_cast<const void *>(kernel), c.nthreads, c.nbytes_shared));
        if (c.blocks_per_sm <= 0) { c.blocks_per_sm = 1; }
        return c;
    }();
    return configs.emplace(device, config).first->second;
}

// One Q-block through the vendored kernel plus its stream-K fixup.
template <typename Geometry>
void launch_flash_block(const float* q_f32, const half* k_f16, const half* v_f16, const half* mask,
                        float* dst, float2* dst_meta, int tokens, int n_kv, int mask_stride,
                        float scale, cudaStream_t stream) {
    using P                    = VoltaFlashParams<Geometry>;
    constexpr int kQHeads      = P::kQHeads;
    constexpr int kKVHeads     = P::kKVHeads;
    constexpr int kGroup       = P::kGroup;
    constexpr int kNcols1      = P::kNcols1;
    constexpr int kNcols2      = P::kNcols2;

    const FlashLaunchConfig& c = flash_launch_config<Geometry>();

    const int ntiles_x     = (tokens + kNcols1 - 1) / kNcols1;
    const int ntiles_z_gqa = (kGroup + kNcols2 - 1) / kNcols2;
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * kKVHeads;
    const int ntiles_KV    = (n_kv + c.nbatch_fa - 1) / c.nbatch_fa;

    const int max_blocks = c.blocks_per_sm * c.nsm;

    const int raw     = std::min(max_blocks, ntiles_KV * ntiles_dst);
    const int rounded = (raw / ntiles_dst) * ntiles_dst;
    const int loss    = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
    const int nblocks = loss <= 5 ? rounded : raw;

    const dim3 block_dim(WARP_SIZE, c.nwarps, 1);
    const dim3 blocks_num(nblocks, 1, 1);

    const std::int32_t nb01 = kDKQ * kQHeads  * sizeof(float);
    const std::int32_t nb02 = kDKQ            * sizeof(float);
    const std::int32_t nb11 = kDKQ * kKVHeads * sizeof(half);
    const std::int32_t nb12 = kDKQ            * sizeof(half);
    const std::int32_t nb31 = mask_stride     * sizeof(half);

    const uint3 ne01_fd = init_fastdiv_values(tokens);

    flash_attn_ext_f16<kDKQ, kDV, kNcols1, kNcols2, false, false>
        <<<blocks_num, block_dim, c.nbytes_shared, stream>>>(
            reinterpret_cast<const char *>(q_f32),
            reinterpret_cast<const char *>(k_f16),
            reinterpret_cast<const char *>(v_f16),
            reinterpret_cast<const char *>(mask),
            nullptr, nullptr, dst, dst_meta,
            scale, 0.0f, 1.0f, 1.0f, /*n_head_log2=*/16u, 0.0f,
            kDKQ, ne01_fd, kQHeads, 1, nb01, nb02, 0,
            kDKQ, n_kv, kKVHeads, 1, nb11, nb12, 0,
            nb11, nb12, 0,
            tokens, 1, 1,
            nb31, 0, 0);

    if (nblocks % ntiles_dst == 0 && nblocks > ntiles_dst) {
        const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * kKVHeads);
        const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
        const uint3 fd2 = init_fastdiv_values(ntiles_x);

        flash_attn_stream_k_fixup_uniform<kDV, kNcols1, kNcols2>
            <<<dim3((unsigned) ntiles_dst, kNcols1, kNcols2), dim3(kDV, 1, 1), 0, stream>>>(
                dst, dst_meta, tokens, kQHeads, kKVHeads, nblocks, kGroup,
                nblocks / ntiles_dst, fd0, fd1, fd2);
    } else if (ntiles_dst % nblocks != 0) {
        const int total_work = ntiles_KV * ntiles_dst;

        const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * kKVHeads);
        const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
        const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
        const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

        flash_attn_stream_k_fixup_general<kDV, kNcols1, kNcols2>
            <<<dim3((unsigned) nblocks, kNcols1, kNcols2), dim3(kDV, 1, 1), 0, stream>>>(
                dst, dst_meta, tokens, kQHeads, kGroup, total_work,
                fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
    }
}

// ---------------------------------------------------------------------------

template <typename Geometry>
std::size_t meta_elements_impl(std::int32_t tokens) {
    using P                    = VoltaFlashParams<Geometry>;
    const FlashLaunchConfig& c = flash_launch_config<Geometry>();
    const int ntiles_x         = (tokens + P::kNcols1 - 1) / P::kNcols1;
    const int ntiles_dst = ntiles_x * ((P::kGroup + P::kNcols2 - 1) / P::kNcols2) * P::kKVHeads;
    const int max_blocks = c.blocks_per_sm * c.nsm;
    // Upper bound: the launch never uses more blocks than this.
    const int nblocks = std::max(max_blocks, ntiles_dst);
    return static_cast<std::size_t>(nblocks) * P::kNcols * (2 + kDV / 2);
}

template <typename Geometry>
void volta_flash_launch_impl(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& table_rows, float scale,
                             PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                             std::int32_t q_block_tokens, Tensor& k_gathered, Tensor& v_gathered,
                             Tensor& mask, Tensor& q_f32, Tensor& out_f32, Tensor& dst_meta,
                             Tensor& out, cudaStream_t stream) {
    using P                          = VoltaFlashParams<Geometry>;
    constexpr int kQHeads            = P::kQHeads;
    constexpr int kKVHeads           = P::kKVHeads;
    const std::int32_t width         = q.ne[2];
    const std::int32_t n_kv_total    = static_cast<std::int32_t>(envelope.max_visible_keys);
    const std::int32_t logical_pages = cache.block_tables.ne[0];
    if constexpr (kQHeads == 6) { q_block_tokens = width; }

    auto* block_tables = static_cast<const std::int32_t*>(cache.block_tables.data);
    auto* rows         = static_cast<const std::int32_t*>(table_rows.data);
    auto* position_ptr = static_cast<const std::int32_t*>(positions.data);

    // Every quantized storage format (INT8/FP8/NVFP4/K8V4) carries the D256 rotation on its K/V
    // rows, so Q staging must apply the same rotation and the output epilogue must undo it exactly
    // once; BF16 leaves both untouched. The storage field is the p4 dispatch key (0.2): BF16 and
    // INT8 keep their original elementwise/INT8 branches bit-identical, the new formats take the
    // codec-dispatched branches below.
    const bool rotated = cache.storage != KvCacheStorage::BFloat16;

    // 1. Append this call's K/V for the whole width.
    if (cache.dtype == DType::I8) {
        constexpr int kWarpsPerBlock = 8;
        const int units              = width * kKVHeads;
        const int blocks             = (units + kWarpsPerBlock - 1) / kWarpsPerBlock;
        volta_flash_append_kv_i8_kernel<kKVHeads><<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            position_ptr, rows, static_cast<std::int8_t*>(cache.k_pages.data),
            static_cast<std::int8_t*>(cache.v_pages.data),
            static_cast<half*>(cache.k_scale_pages.data),
            static_cast<half*>(cache.v_scale_pages.data), block_tables, logical_pages, width);
    } else if (cache.dtype == DType::FP8_E4M3FN) {
        constexpr int kWarpsPerBlock = 8;
        const int units              = width * kKVHeads;
        const int blocks             = (units + kWarpsPerBlock - 1) / kWarpsPerBlock;
        if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
            volta_flash_append_kv_codec_kernel<kKVHeads, KvK8V4>
                <<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data), position_ptr, rows,
                    static_cast<KvK8V4::Code*>(cache.k_pages.data),
                    static_cast<KvK8V4::Code*>(cache.v_pages.data),
                    static_cast<KvK8V4::KScale*>(cache.k_scale_pages.data),
                    static_cast<KvK8V4::VScale*>(cache.v_scale_pages.data), block_tables,
                    logical_pages, width);
        } else {
            volta_flash_append_kv_codec_kernel<kKVHeads, KvFp8G64>
                <<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data), position_ptr, rows,
                    static_cast<KvFp8G64::Code*>(cache.k_pages.data),
                    static_cast<KvFp8G64::Code*>(cache.v_pages.data),
                    static_cast<KvFp8G64::KScale*>(cache.k_scale_pages.data),
                    static_cast<KvFp8G64::VScale*>(cache.v_scale_pages.data), block_tables,
                    logical_pages, width);
        }
    } else if (cache.dtype == DType::U8) {
        constexpr int kWarpsPerBlock = 8;
        const int units              = width * kKVHeads;
        const int blocks             = (units + kWarpsPerBlock - 1) / kWarpsPerBlock;
        volta_flash_append_kv_codec_kernel<kKVHeads, KvNvfp4G16>
            <<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
                position_ptr, rows, static_cast<KvNvfp4G16::Code*>(cache.k_pages.data),
                static_cast<KvNvfp4G16::Code*>(cache.v_pages.data),
                static_cast<KvNvfp4G16::KScale*>(cache.k_scale_pages.data),
                static_cast<KvNvfp4G16::VScale*>(cache.v_scale_pages.data), block_tables,
                logical_pages, width);
    } else {
        volta_flash_append_kv_kernel<kKVHeads><<<dim3(width, kKVHeads), kHeadDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            position_ptr, rows, static_cast<__nv_bfloat16*>(cache.k_pages.data),
            static_cast<__nv_bfloat16*>(cache.v_pages.data), block_tables, logical_pages, width);
    }
    CUDA_CHECK(cudaGetLastError());

    // 2. Gather the visible key range paged -> contiguous FP16, once for every
    //    Q-block of this layer.
    const std::int32_t n_kv_alloc = round_up_keys(n_kv_total);
    if (cache.dtype == DType::I8) {
        volta_flash_gather_kv_i8_kernel<kKVHeads>
            <<<dim3(n_kv_alloc, kKVHeads), kHeadDim, 0, stream>>>(
                static_cast<const std::int8_t*>(cache.k_pages.data),
                static_cast<const std::int8_t*>(cache.v_pages.data),
                static_cast<const half*>(cache.k_scale_pages.data),
                static_cast<const half*>(cache.v_scale_pages.data), block_tables, rows,
                logical_pages, static_cast<half*>(k_gathered.data),
                static_cast<half*>(v_gathered.data), n_kv_total, n_kv_alloc);
    } else if (cache.dtype == DType::FP8_E4M3FN) {
        if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
            volta_flash_gather_kv_k8v4_kernel<kKVHeads>
                <<<dim3(n_kv_alloc, kKVHeads), kHeadDim, 0, stream>>>(
                    static_cast<const std::uint8_t*>(cache.k_pages.data),
                    static_cast<const std::uint8_t*>(cache.v_pages.data),
                    static_cast<const half*>(cache.k_scale_pages.data),
                    static_cast<const std::uint8_t*>(cache.v_scale_pages.data), block_tables, rows,
                    logical_pages, static_cast<half*>(k_gathered.data),
                    static_cast<half*>(v_gathered.data), n_kv_total, n_kv_alloc);
        } else {
            volta_flash_gather_kv_fp8_kernel<kKVHeads>
                <<<dim3(n_kv_alloc, kKVHeads), kHeadDim, 0, stream>>>(
                    static_cast<const std::uint8_t*>(cache.k_pages.data),
                    static_cast<const std::uint8_t*>(cache.v_pages.data),
                    static_cast<const half*>(cache.k_scale_pages.data),
                    static_cast<const half*>(cache.v_scale_pages.data), block_tables, rows,
                    logical_pages, static_cast<half*>(k_gathered.data),
                    static_cast<half*>(v_gathered.data), n_kv_total, n_kv_alloc);
        }
    } else if (cache.dtype == DType::U8) {
        volta_flash_gather_kv_nvfp4_kernel<kKVHeads>
            <<<dim3(n_kv_alloc, kKVHeads), kHeadDim, 0, stream>>>(
                static_cast<const std::uint8_t*>(cache.k_pages.data),
                static_cast<const std::uint8_t*>(cache.v_pages.data),
                static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
                static_cast<const std::uint8_t*>(cache.v_scale_pages.data), block_tables, rows,
                logical_pages, static_cast<half*>(k_gathered.data),
                static_cast<half*>(v_gathered.data), n_kv_total, n_kv_alloc);
    } else {
        volta_flash_gather_kv_kernel<kKVHeads>
            <<<dim3(n_kv_alloc, kKVHeads), kHeadDim, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(cache.k_pages.data),
                static_cast<const __nv_bfloat16*>(cache.v_pages.data), block_tables, rows,
                logical_pages, static_cast<half*>(k_gathered.data),
                static_cast<half*>(v_gathered.data), n_kv_total, n_kv_alloc);
    }
    CUDA_CHECK(cudaGetLastError());

    // 3. Q-blocks. Bounding the block bounds mask memory, and lets earlier blocks
    //    attend over a shorter key range than later ones.
    const std::int32_t base = n_kv_total - width;
    for (std::int32_t begin = 0; begin < width; begin += q_block_tokens) {
        const std::int32_t tokens = std::min(q_block_tokens, width - begin);
        const std::int32_t n_kv   = round_up_keys(base + begin + tokens);
        const std::int32_t rows_padded =
            ((tokens + kVoltaFlashMaskRowPad - 1) / kVoltaFlashMaskRowPad) * kVoltaFlashMaskRowPad;

        const std::int64_t q_count = static_cast<std::int64_t>(tokens) * kQHeads * kHeadDim;

        constexpr int kConvertThreads = 256;
        const int convert_blocks =
            static_cast<int>((q_count + kConvertThreads - 1) / kConvertThreads);

        if constexpr (kQHeads == 6) {
            const bool rotate = rotated;
            gqa_attention_volta_splitd_launch(
                static_cast<const __nv_bfloat16*>(q.data) +
                    static_cast<std::int64_t>(begin) * kQHeads * kHeadDim,
                static_cast<const half*>(k_gathered.data),
                static_cast<const half*>(v_gathered.data), position_ptr + begin,
                static_cast<half*>(q_f32.data), static_cast<float*>(out_f32.data),
                tokens, n_kv_total, kQHeads, kKVHeads, scale, stream, rotate);
            if (rotate) {
                const int rows = tokens * kQHeads;
                volta_flash_convert_out_i8_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
                    static_cast<const float*>(out_f32.data),
                    static_cast<__nv_bfloat16*>(out.data) +
                        static_cast<std::int64_t>(begin) * kQHeads * kHeadDim,
                    rows);
            } else {
                volta_flash_convert_out_kernel<<<convert_blocks, kConvertThreads, 0, stream>>>(
                    static_cast<const float*>(out_f32.data),
                    static_cast<__nv_bfloat16*>(out.data) +
                        static_cast<std::int64_t>(begin) * kQHeads * kHeadDim,
                    q_count);
            }
            CUDA_CHECK(cudaGetLastError());
            continue;
        }

        if (rotated) {
            const int rows = tokens * kQHeads;
            volta_flash_convert_q_i8_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data) +
                    static_cast<std::int64_t>(begin) * kQHeads * kHeadDim,
                static_cast<float*>(q_f32.data), rows);
        } else {
            volta_flash_convert_q_kernel<<<convert_blocks, kConvertThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data) +
                    static_cast<std::int64_t>(begin) * kQHeads * kHeadDim,
                static_cast<float*>(q_f32.data), q_count);
        }

        // The mask extent is the padded key count, so every key the kernel can touch
        // has a defined mask entry; keys past a row's causal limit -- padding included
        // -- get -inf from the same comparison.
        volta_flash_build_mask_kernel<<<rows_padded, 256, 0, stream>>>(
            position_ptr, begin, tokens, static_cast<half*>(mask.data), n_kv, n_kv);

        cudaMemsetAsync(out_f32.data, 0, static_cast<std::size_t>(q_count) * sizeof(float), stream);

        launch_flash_block<Geometry>(static_cast<const float*>(q_f32.data),
                           static_cast<const half*>(k_gathered.data),
                           static_cast<const half*>(v_gathered.data),
                           static_cast<const half*>(mask.data),
                           static_cast<float*>(out_f32.data),
                           static_cast<float2*>(dst_meta.data), tokens, n_kv, n_kv, scale, stream);

        if (rotated) {
            const int rows = tokens * kQHeads;
            volta_flash_convert_out_i8_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
                static_cast<const float*>(out_f32.data),
                static_cast<__nv_bfloat16*>(out.data) +
                    static_cast<std::int64_t>(begin) * kQHeads * kHeadDim,
                rows);
        } else {
            volta_flash_convert_out_kernel<<<convert_blocks, kConvertThreads, 0, stream>>>(
                static_cast<const float*>(out_f32.data),
                static_cast<__nv_bfloat16*>(out.data) +
                    static_cast<std::int64_t>(begin) * kQHeads * kHeadDim,
                q_count);
        }
    }
}

} // namespace

// The two registered geometries differ only in tiling; both are instantiated so
// the route can serve 27B (24q/4kv) and 35B-A3B (16q/2kv) from one launcher.
std::size_t gqa_attention_volta_flash_meta_elements(std::int32_t q_heads, std::int32_t tokens) {
    if (q_heads == Gqa27Geometry::QHeads) { return meta_elements_impl<Gqa27Geometry>(tokens); }
    if (q_heads == Gqa27Tp2Geometry::QHeads) { return meta_elements_impl<Gqa27Tp2Geometry>(tokens); }
    if (q_heads == Gqa27Tp4Geometry::QHeads) { return meta_elements_impl<Gqa27Tp4Geometry>(tokens); }
    if (q_heads == Gqa35Geometry::QHeads) { return meta_elements_impl<Gqa35Geometry>(tokens); }
    throw std::invalid_argument("gqa_attention volta flash: unsupported Q head geometry");
}

void gqa_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& positions, const Tensor& table_rows,
                                      float scale, PagedKVBatchLayerView cache,
                                      GqaExecutionEnvelope envelope, std::int32_t q_block_tokens,
                                      Tensor& k_gathered, Tensor& v_gathered, Tensor& mask,
                                      Tensor& q_f32, Tensor& out_f32, Tensor& dst_meta, Tensor& out,
                                      cudaStream_t stream) {
    if (q.ne[1] == Gqa27Tp4Geometry::QHeads) {
        volta_flash_launch_impl<Gqa27Tp4Geometry>(q, k, v, positions, table_rows, scale, cache,
                                                  envelope, q_block_tokens, k_gathered, v_gathered,
                                                  mask, q_f32, out_f32, dst_meta, out, stream);
        return;
    }
    if (q.ne[1] == Gqa27Tp2Geometry::QHeads) {
        volta_flash_launch_impl<Gqa27Tp2Geometry>(q, k, v, positions, table_rows, scale, cache,
                                                  envelope, q_block_tokens, k_gathered, v_gathered,
                                                  mask, q_f32, out_f32, dst_meta, out, stream);
        return;
    }
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        volta_flash_launch_impl<Gqa27Geometry>(q, k, v, positions, table_rows, scale, cache,
                                               envelope, q_block_tokens, k_gathered, v_gathered,
                                               mask, q_f32, out_f32, dst_meta, out, stream);
        return;
    }
    if (q.ne[1] == Gqa35Geometry::QHeads) {
        volta_flash_launch_impl<Gqa35Geometry>(q, k, v, positions, table_rows, scale, cache,
                                               envelope, q_block_tokens, k_gathered, v_gathered,
                                               mask, q_f32, out_f32, dst_meta, out, stream);
        return;
    }
    throw std::invalid_argument("gqa_attention volta flash: unsupported Q head geometry");
}

} // namespace ninfer::ops::detail
