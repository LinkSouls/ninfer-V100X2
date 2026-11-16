#pragma once

// ninfer::ops - normalized D256 Sylvester Hadamard rotation for the INT8 group-64 KV cache
// (and, later, the FP8/NVFP4 storage formats).
//
// Why it exists: the per-group absmax/127 symmetric quantization of the INT8 KV cache is
// extremely sensitive to a small number of high-energy key/value channels. Because the
// attention score is Q·K, energy concentrated in a few channels is squared into the score and
// the quantization error lands exactly where it hurts long-context attention. Applying the same
// orthogonal transform H to Q, K and V before quantization flattens the per-channel energy, so
// the (unchanged) symmetric quantizer has less to lose; the score is invariant because H is
// orthogonal (Q·K == (HQ)·(HK)ᵀ in exact arithmetic), and the V rotation is undone on the
// attention output so the external result is equivalent to the un-rotated one.
//
// The transform is a fixed normalized Sylvester Hadamard of order 256, ported from the upstream
// reference implementation. It is pure warp shuffles + register butterflies: no shared memory,
// no extra memory traffic, no Blackwell-only intrinsic (it is Volta/sm_70-safe).
//
// Lane mapping (the one this project's kernels already use): one full warp owns one D256 row;
// lane `l` carries dimension `l + 32*r` in `values[r]` for r in [0, 8). This is exactly the
// per-group layout the INT8 kernels use (group g, lane l holds dims g*64+l and g*64+l+32), so the
// rotation slots in with no re-mapping.
//
// The transform is its own inverse (an involution): it is a normalized orthogonal Hadamard, so
// applying it twice yields the identity (up to FP32 rounding). The output-side "undo" of the V
// rotation therefore reuses the very same function.

#include <cuda_runtime.h>

namespace ninfer::ops {

// Apply the five lane dimensions of a D32 Sylvester transform to `Columns` independent columns.
// Keeping the column loop inside each stage exposes the independent shuffle instructions while
// preserving the butterfly and rounding order of every column.
template <int Columns>
__device__ __forceinline__ void gqa_hadamard_d32_columns_inplace(float (&values)[Columns],
                                                                 int lane) {
    constexpr unsigned FullMask = 0xffffffffu;
#pragma unroll
    for (int stride = 1; stride <= 16; stride <<= 1) {
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            const float value = values[column];
            const float peer  = __shfl_xor_sync(FullMask, value, stride);
            values[column]    = (lane & stride) == 0 ? __fadd_rn(value, peer)
                                                    : __fsub_rn(peer, value);
        }
    }
}

// One full warp owns one D256 row. Lane `l` carries dimensions `l + 32*r` in `values[r]`. The
// fixed normalized Sylvester transform is shared by the persistent K/V write path and the
// transient Q path; it never materializes an intermediate row outside registers.
//
// Structure: (1) the 5-level 32-lane butterfly across each of the 8 columns, (2) the 3-level
// in-register butterfly over the 8 column values, (3) a final 2^-4 normalization. Steps (1)-(2)
// together are the unnormalized 256-point Hadamard B with B·Bᵀ = 256·I and B symmetric, so the
// 2^-4 factor makes the whole map an involution: T(T(x)) == x.
__device__ __forceinline__ void gqa_hadamard_d256_inplace(float (&values)[8], int lane) {
    gqa_hadamard_d32_columns_inplace<8>(values, lane);

#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float low  = values[base + offset];
                const float high = values[base + offset + span];
                values[base + offset]        = __fadd_rn(low, high);
                values[base + offset + span] = __fsub_rn(low, high);
            }
        }
    }

#pragma unroll
    for (float& value : values) { value = __fmul_rn(value, 0x1p-4f); }
}

// Contiguous eight-feature lane layout: `values[j]` is feature `8*lane + j`. Kernels that stage a
// contiguous 8-wide d-chunk per lane (e.g. the Volta SIMT decode's Q read) use this variant. The
// low three butterflies stay in registers; the remaining five use warp lanes. The logical bit
// order and the final 2^-4 normalization match the strided layout, so both variants compute the
// same transform (in a different FP32 association order).
__device__ __forceinline__ void gqa_hadamard_d256_contiguous_inplace(float (&values)[8], int lane) {
#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * span) {
#pragma unroll
            for (int i = 0; i < span; ++i) {
                const float low  = values[base + i];
                const float high = values[base + i + span];
                values[base + i]        = __fadd_rn(low, high);
                values[base + i + span] = __fsub_rn(low, high);
            }
        }
    }
    gqa_hadamard_d32_columns_inplace<8>(values, lane);
#pragma unroll
    for (float& x : values) { x = __fmul_rn(x, 0x1p-4f); }
}

} // namespace ninfer::ops