#include "ops/launcher/gqa_attention_qpn_volta.h"

#ifdef NINFER_VOLTA_BUILD
#include "core/device.h"
#include "ops/kernel/gqa_attention_decode_i8_qpn_verify_volta.cuh"
#include "ops/kernel/gqa_attention_decode_i8_qpn_verify5_volta.cuh"
#include "ops/kernel/gqa_attention_decode_i8_qpn_volta.cuh"
#include "ops/kernel/gqa_attention_nvfp4_g16_codec.cuh"
#include "ops/kernel/gqa_attention_k8v4_codec.cuh"
#include "ops/launcher/kernel_attr_once.h"

namespace ninfer::ops::detail {

template <typename KvCodec>
void gqa_attention_single_qpn_volta_launch(
    const Tensor& q, const __nv_bfloat16* new_k, const __nv_bfloat16* new_v,
    const Tensor& positions, float scale, PagedKVBatchLayerView cache,
    const GqaSmallTInvocation& invocation, std::int32_t logical_capacity,
    bool long_profile, std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, cudaStream_t stream) {
    const auto launch = [&]<typename Geometry, bool MultiBatch, bool Masked, typename CacheInput>(CacheInput input) {
        const auto profile = [&]<bool LongProfile>() {
            gqa_attention_single_qpn_volta_partial_i8_kernel<Geometry, MultiBatch, Masked, CacheInput, LongProfile, KvCodec>
                <<<dim3(Geometry::KVHeads, splits, invocation.batch_size), 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data), input,
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<KvCodec::Code*>(cache.k_pages.data),
                    static_cast<KvCodec::Code*>(cache.v_pages.data),
                    static_cast<KvCodec::KScale*>(cache.k_scale_pages.data),
                    static_cast<KvCodec::VScale*>(cache.v_scale_pages.data),
                    static_cast<const std::int32_t*>(cache.block_tables.data),
                    invocation.valid_columns == nullptr ? nullptr
                        : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                    invocation.table_rows == nullptr ? nullptr
                        : static_cast<const std::int32_t*>(invocation.table_rows->data),
                    cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                    logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
        };
        if (long_profile) { profile.template operator()<true>(); }
        else { profile.template operator()<false>(); }
        CUDA_CHECK(cudaGetLastError());
    };
    const auto dispatch_geometry = [&]<typename Geometry, typename CacheInput>(CacheInput input) {
        if (invocation.batch_size == 1) {
            if (invocation.valid_columns != nullptr) {
                launch.template operator()<Geometry, false, true>(input);
            } else {
                launch.template operator()<Geometry, false, false>(input);
            }
        } else if (invocation.valid_columns != nullptr) {
            launch.template operator()<Geometry, true, true>(input);
        } else {
            launch.template operator()<Geometry, true, false>(input);
        }
    };
    const auto dispatch = [&]<typename CacheInput>(CacheInput input) {
        if (q.ne[1] == Gqa27Tp4Geometry::QHeads) {
            dispatch_geometry.template operator()<Gqa27Tp4Geometry>(input);
        } else {
            dispatch_geometry.template operator()<Gqa27Tp2Geometry>(input);
        }
    };
    if (new_k != nullptr) { dispatch(GqaAppendInput{new_k, new_v}); }
    else { dispatch(GqaCachedInput{}); }
}

template <typename KvCodec>
void gqa_attention_verify_qpn_volta_launch(
    const Tensor& q, const __nv_bfloat16* new_k, const __nv_bfloat16* new_v,
    const Tensor& positions, float scale, PagedKVBatchLayerView cache,
    const GqaSmallTInvocation& invocation, std::int32_t logical_capacity,
    std::int32_t splits, Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
    cudaStream_t stream) {
    const auto launch = [&]<typename Geometry, bool MultiBatch, bool Masked, typename CacheInput>(CacheInput input) {
        const auto issue = [&](auto kernel, int dynamic_bytes) {
            if (dynamic_bytes != 0) {
                ensure_func_attr_per_device(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             dynamic_bytes);
            }
            kernel<<<dim3(Geometry::KVHeads, splits, invocation.batch_size),
                       256, dynamic_bytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(positions.data),
                static_cast<KvCodec::Code*>(cache.k_pages.data),
                static_cast<KvCodec::Code*>(cache.v_pages.data),
                static_cast<KvCodec::KScale*>(cache.k_scale_pages.data),
                static_cast<KvCodec::VScale*>(cache.v_scale_pages.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
        };
        if (invocation.width == 5) {
            issue(gqa_attention_verify5_qpn_volta_partial_i8_kernel<Geometry, MultiBatch, Masked, CacheInput, KvCodec>, 0);
        } else {
            issue(gqa_attention_verify_qpn_volta_partial_i8_kernel<Geometry, MultiBatch, Masked, CacheInput, KvCodec>,
                  kGqaVerifyQpnDynamicBytes);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    const auto dispatch_geometry = [&]<typename Geometry, typename CacheInput>(CacheInput input) {
        if (invocation.batch_size == 1) {
            if (invocation.valid_columns != nullptr) {
                launch.template operator()<Geometry, false, true>(input);
            } else {
                launch.template operator()<Geometry, false, false>(input);
            }
        } else if (invocation.valid_columns != nullptr) {
            launch.template operator()<Geometry, true, true>(input);
        } else {
            launch.template operator()<Geometry, true, false>(input);
        }
    };
    const auto dispatch = [&]<typename CacheInput>(CacheInput input) {
        if (q.ne[1] == Gqa27Tp4Geometry::QHeads) {
            dispatch_geometry.template operator()<Gqa27Tp4Geometry>(input);
        } else {
            dispatch_geometry.template operator()<Gqa27Tp2Geometry>(input);
        }
    };
    if (new_k != nullptr) {
        dispatch(GqaAppendInput{new_k, new_v});
    } else {
        dispatch(GqaCachedInput{});
    }
}

// The QPN long-context launchers are codec-generic and are called from a DIFFERENT translation
// unit (gqa_attention_decode.cu's small-T path), so their concrete specializations must be emitted
// here explicitly. INT8 and FP8 share the same kernel body, differing only in the group-64 codec
// (int8 codes vs E4M3 codes with an FP16 abs-max scale); NVFP4 uses the group-16 E2M1/E4M3 codec.
template void gqa_attention_single_qpn_volta_launch<KvI8G64>(
    const Tensor&, const __nv_bfloat16*, const __nv_bfloat16*, const Tensor&, float,
    PagedKVBatchLayerView, const GqaSmallTInvocation&, std::int32_t, bool, std::int32_t,
    Tensor&, Tensor&, Tensor&, cudaStream_t);
template void gqa_attention_single_qpn_volta_launch<KvFp8G64>(
    const Tensor&, const __nv_bfloat16*, const __nv_bfloat16*, const Tensor&, float,
    PagedKVBatchLayerView, const GqaSmallTInvocation&, std::int32_t, bool, std::int32_t,
    Tensor&, Tensor&, Tensor&, cudaStream_t);
template void gqa_attention_single_qpn_volta_launch<KvNvfp4G16>(
    const Tensor&, const __nv_bfloat16*, const __nv_bfloat16*, const Tensor&, float,
    PagedKVBatchLayerView, const GqaSmallTInvocation&, std::int32_t, bool, std::int32_t,
    Tensor&, Tensor&, Tensor&, cudaStream_t);
template void gqa_attention_single_qpn_volta_launch<KvK8V4>(
    const Tensor&, const __nv_bfloat16*, const __nv_bfloat16*, const Tensor&, float,
    PagedKVBatchLayerView, const GqaSmallTInvocation&, std::int32_t, bool, std::int32_t,
    Tensor&, Tensor&, Tensor&, cudaStream_t);
template void gqa_attention_verify_qpn_volta_launch<KvI8G64>(
    const Tensor&, const __nv_bfloat16*, const __nv_bfloat16*, const Tensor&, float,
    PagedKVBatchLayerView, const GqaSmallTInvocation&, std::int32_t, std::int32_t,
    Tensor&, Tensor&, Tensor&, cudaStream_t);
template void gqa_attention_verify_qpn_volta_launch<KvFp8G64>(
    const Tensor&, const __nv_bfloat16*, const __nv_bfloat16*, const Tensor&, float,
    PagedKVBatchLayerView, const GqaSmallTInvocation&, std::int32_t, std::int32_t,
    Tensor&, Tensor&, Tensor&, cudaStream_t);
template void gqa_attention_verify_qpn_volta_launch<KvNvfp4G16>(
    const Tensor&, const __nv_bfloat16*, const __nv_bfloat16*, const Tensor&, float,
    PagedKVBatchLayerView, const GqaSmallTInvocation&, std::int32_t, std::int32_t,
    Tensor&, Tensor&, Tensor&, cudaStream_t);
template void gqa_attention_verify_qpn_volta_launch<KvK8V4>(
    const Tensor&, const __nv_bfloat16*, const __nv_bfloat16*, const Tensor&, float,
    PagedKVBatchLayerView, const GqaSmallTInvocation&, std::int32_t, std::int32_t,
    Tensor&, Tensor&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
#endif
