#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Op: apply_token_mask (row-major lanes)
 *
 * Math / indexing:
 *   For lane r and vocabulary token v, with word index w = v / 32 and bit i = v % 32:
 *
 *     logits[r, v] = (masks[r, w] has bit i set) ? logits[r, v] : -inf
 *
 * Logical shapes:
 *   logits is contiguous BF16 [lanes, V]; masks is contiguous I32 [lanes, words] (uint32 bit
 *   patterns) with words = (V + 31) / 32. A cleared bit forbids the token for that lane. An
 *   all-ones row is the unconstrained no-op and leaves the lane bit-identical; a fully zero
 *   word is a dead end and forbids all 32 of its tokens.
 *
 * Numeric:
 *   Forbidden entries are written as the BF16 negative-infinity bit pattern (0xFF80); every
 *   other entry is left bit-identical, so an all-ones mask is a no-op.
 *
 * Effects:
 *   In-place on logits. masks is read-only. No workspace. The call is a no-op when lanes is 0.
 *   Intended as the pre-sampling constraint step: an unconstrained lane is simply not masked.
 */
void apply_token_mask(const Tensor& logits, const Tensor& masks, cudaStream_t stream);

/**
 * Op: apply_token_mask_columns (column-major lanes)
 *
 * Same semantics as apply_token_mask, but for the layout the decode/prefill logits use:
 *   logits is contiguous BF16 [V, lanes] (ne[0] = V, ne[1] = lanes, ne[0] fastest-varying);
 *   masks is contiguous I32 [lanes, words]. Lane l's vocabulary column starts at l * ne[0]
 *   elements with contiguous vocabulary entries, so this is the strided form with
 *   lane_stride = ne[0] and vocab_stride = 1.
 */
void apply_token_mask_columns(const Tensor& logits, const Tensor& masks, cudaStream_t stream);

} // namespace ninfer::ops
