#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// lanes = masks.ne[0]; the caller supplies the lane and vocab strides for the logits layout.
void apply_token_mask_strided_launch(const Tensor& logits, const Tensor& masks,
                                     std::int64_t lane_stride, std::int64_t vocab_stride,
                                     std::int64_t vocab, cudaStream_t stream);

} // namespace ninfer::ops::detail
