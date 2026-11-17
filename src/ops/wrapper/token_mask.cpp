#include "ninfer/ops/token_mask.h"

#include "ops/launcher/token_mask.h"

#include <stdexcept>

namespace ninfer::ops {

namespace {
void check_pair(const Tensor& logits, const Tensor& masks) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("apply_token_mask: logits must be BF16");
    }
    if (masks.dtype != DType::I32) {
        throw std::invalid_argument("apply_token_mask: masks must be I32 bit patterns");
    }
    if (logits.ne[2] != 1 || logits.ne[3] != 1 || masks.ne[2] != 1 || masks.ne[3] != 1) {
        throw std::invalid_argument("apply_token_mask: tensors must be rank-2");
    }
    if (masks.ne[0] <= 0) { throw std::invalid_argument("apply_token_mask: lanes must be positive"); }
    if (!logits.is_contiguous() || !masks.is_contiguous() || logits.data == nullptr ||
        masks.data == nullptr) {
        throw std::invalid_argument("apply_token_mask: tensors must be contiguous and non-null");
    }
}
} // namespace

void apply_token_mask(const Tensor& logits, const Tensor& masks, cudaStream_t stream) {
    check_pair(logits, masks);
    if (logits.ne[1] <= 0) { throw std::invalid_argument("apply_token_mask: vocabulary must be positive"); }
    const std::int32_t words = (static_cast<std::int32_t>(logits.ne[1]) + 31) / 32;
    if (masks.ne[1] != words) { throw std::invalid_argument("apply_token_mask: mask width must be ceil(V/32)"); }
    // Row-major [lanes, V]: lanes along ne[0], vocab contiguous.
    detail::apply_token_mask_strided_launch(logits, masks, logits.ne[1], 1, logits.ne[1], stream);
}

void apply_token_mask_columns(const Tensor& logits, const Tensor& masks, cudaStream_t stream) {
    check_pair(logits, masks);
    if (logits.ne[1] <= 0) { throw std::invalid_argument("apply_token_mask_columns: lanes must be positive"); }
    if (masks.ne[0] != logits.ne[1]) {
        throw std::invalid_argument("apply_token_mask_columns: lane counts must match");
    }
    const std::int32_t words = (static_cast<std::int32_t>(logits.ne[0]) + 31) / 32;
    if (masks.ne[1] != words) {
        throw std::invalid_argument("apply_token_mask_columns: mask width must be ceil(V/32)");
    }
    // Column-major [V, lanes]: ne[0] (vocab) is the fastest-varying dimension, so lane l's
    // column starts at l * ne[0] elements and vocabulary entries are contiguous.
    detail::apply_token_mask_strided_launch(logits, masks, logits.ne[0], 1, logits.ne[0], stream);
}

} // namespace ninfer::ops
