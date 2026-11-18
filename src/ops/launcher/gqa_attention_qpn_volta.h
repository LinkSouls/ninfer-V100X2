#pragma once

#include "ops/kernel/gqa_attention_kv_g64_codec.cuh"
#include "ops/launcher/gqa_attention.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

// The group-64 QPN long-context launchers are codec-generic (INT8 / FP8): the kernel body is
// selected by KvCodec, exactly as in the small-T decode path. See gqa_attention_kv_g64_codec.cuh.
template <typename KvCodec>
void gqa_attention_single_qpn_volta_launch(
    const Tensor& q, const __nv_bfloat16* new_k, const __nv_bfloat16* new_v,
    const Tensor& positions, float scale, PagedKVBatchLayerView cache,
    const GqaSmallTInvocation& invocation, std::int32_t logical_capacity,
    bool long_profile, std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, cudaStream_t stream);

template <typename KvCodec>
void gqa_attention_verify_qpn_volta_launch(
    const Tensor& q, const __nv_bfloat16* new_k, const __nv_bfloat16* new_v,
    const Tensor& positions, float scale, PagedKVBatchLayerView cache,
    const GqaSmallTInvocation& invocation, std::int32_t logical_capacity,
    std::int32_t splits, Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
    cudaStream_t stream);

} // namespace ninfer::ops::detail
