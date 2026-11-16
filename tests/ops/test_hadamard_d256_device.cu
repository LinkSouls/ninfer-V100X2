// Device bit-match test for the D256 rotation primitive (task 1.1, algorithm level).
//
// Two independent bit-exact comparisons, both on identical inputs and lane mappings:
//
//   1. device vs the UPSTREAM reference: `tests/ops/upstream/hadamard_upstream.cuh` vendors
//      Neroued/ninfer `normalized_hadamard_d256_inplace` / `..._contiguous_inplace`
//      (Apache-2.0, rev d9ad9dea4cdbc64ec44b5b18704b46b5db2613dc) verbatim. Both transforms are
//      compiled in this translation unit, run in separate kernels on the same rows, and their
//      outputs must agree BIT-FOR-BIT.
//   2. device vs a HOST model that simulates the documented association order EXACTLY -- the
//      5-level 32-lane `__shfl_xor_sync` butterfly (IEEE-RN FP32 add/sub), the 3-level in-register
//      butterfly, and the final 2^-4 multiply, in that order, for both the strided
//      (lane l holds dims l + 32*r) and the contiguous (lane l holds dims 8l..8l+7) layouts.
//
// Together with the CPU oracle (test_hadamard_d256.cpp: orthogonality, involution, and
// agreement with the naive Sylvester D256 within FP32 tolerance) this proves the kernel
// implements the documented normalized D256 transform, bit-identically to the upstream
// reference. The op tests (gqa_attention_fixture.h) then judge the attention-level contract
// against the independent FP64 oracle within the registered tolerance, as the kv-cache-rotation
// spec requires.
//
// The target is compiled with -fno-fast-math -ffp-contract=off (ninfer_add_op_test), so the
// host `a + b` / `a - b` / `a * b` on float are plain IEEE-RN `addss`/`subss`/`mulss` --
// bit-identical to `__fadd_rn` / `__fsub_rn` / `__fmul_rn`.

#include "ops/kernel/hadamard_d256.cuh"
#include "upstream/hadamard_upstream.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ninfer;

namespace {

// --- host model: exact simulation of the device association order -------------------

// The 5-level 32-lane butterfly across each of 8 independent columns. `__shfl_xor_sync`
// snapshots every lane's value for the whole instruction, so the per-lane update must read
// the PRE-stride peer value -- a separate output array, never an in-place sequential write
// (that one silently corrupts the peer read, the classic butterfly ordering trap).
void lane_butterfly(std::array<std::array<float, 8>, 32>& v) {
    for (int stride = 1; stride <= 16; stride <<= 1) {
        for (int col = 0; col < 8; ++col) {
            std::array<float, 32> c, n;
            for (int l = 0; l < 32; ++l) { c[l] = v[l][col]; }
            for (int l = 0; l < 32; ++l) {
                const float a = c[l];
                const float b = c[l ^ stride];
                n[l]          = (l & stride) == 0 ? a + b : b - a;
            }
            for (int l = 0; l < 32; ++l) { v[l][col] = n[l]; }
        }
    }
}

// The 3-level in-register butterfly over a lane's 8 values.
void register_butterfly(std::array<std::array<float, 8>, 32>& v) {
    for (int l = 0; l < 32; ++l) {
        for (int span = 1; span < 8; span <<= 1) {
            for (int base = 0; base < 8; base += 2 * span) {
                for (int offset = 0; offset < span; ++offset) {
                    const float low  = v[l][base + offset];
                    const float high = v[l][base + offset + span];
                    v[l][base + offset]        = low + high;
                    v[l][base + offset + span] = low - high;
                }
            }
        }
    }
}

void normalize(std::array<std::array<float, 8>, 32>& v) {
    for (auto& lane : v) {
        for (auto& x : lane) { x = x * 0x1p-4f; }
    }
}

// Strided layout: lane l carries dims l + 32*r in values[r].
std::array<std::array<float, 8>, 32> simulate_strided(const float* row) {
    std::array<std::array<float, 8>, 32> v;
    for (int l = 0; l < 32; ++l) {
        for (int r = 0; r < 8; ++r) { v[l][r] = row[l + 32 * r]; }
    }
    lane_butterfly(v);    // (1)
    register_butterfly(v); // (2)
    normalize(v);          // (3)
    return v;
}

// Contiguous layout: lane l carries dims 8l..8l+7 in values[0..7]. The low three butterflies
// stay in registers; the remaining five use warp lanes; then the 2^-4 normalization.
std::array<std::array<float, 8>, 32> simulate_contiguous(const float* row) {
    std::array<std::array<float, 8>, 32> v;
    for (int l = 0; l < 32; ++l) {
        for (int j = 0; j < 8; ++j) { v[l][j] = row[8 * l + j]; }
    }
    register_butterfly(v); // (1)
    lane_butterfly(v);     // (2)
    normalize(v);          // (3)
    return v;
}

// --- device kernels: one warp per row, the same mapping the production kernels use ---

__global__ void strided_row_kernel(const float* in, float* out) {
    const int lane   = threadIdx.x;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 256;
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { values[r] = in[row + lane + 32 * r]; }
    ninfer::ops::gqa_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) { out[row + lane + 32 * r] = values[r]; }
}

__global__ void contiguous_row_kernel(const float* in, float* out) {
    const int lane   = threadIdx.x;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 256;
    float values[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { values[j] = in[row + 8 * lane + j]; }
    ninfer::ops::gqa_hadamard_d256_contiguous_inplace(values, lane);
#pragma unroll
    for (int j = 0; j < 8; ++j) { out[row + 8 * lane + j] = values[j]; }
}

// --- upstream reference kernels: same mapping, the upstream transform verbatim ---

__global__ void upstream_strided_row_kernel(const float* in, float* out) {
    const int lane   = threadIdx.x;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 256;
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { values[r] = in[row + lane + 32 * r]; }
    ninfer::ops::detail::kv::normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) { out[row + lane + 32 * r] = values[r]; }
}

__global__ void upstream_contiguous_row_kernel(const float* in, float* out) {
    const int lane   = threadIdx.x;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 256;
    float values[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { values[j] = in[row + 8 * lane + j]; }
    ninfer::ops::detail::kv::normalized_hadamard_d256_contiguous_inplace(values, lane);
#pragma unroll
    for (int j = 0; j < 8; ++j) { out[row + 8 * lane + j] = values[j]; }
}

// --- inputs: deterministic randoms plus the edges the codec suites inject -------------

std::vector<float> make_rows() {
    std::vector<float> rows;
    auto push = [&](const std::vector<float>& r) { rows.insert(rows.end(), r.begin(), r.end()); };

    std::vector<float> zeros(256, 0.0f), ones(256, 1.0f), neg_one(256, -1.0f);
    std::vector<float> small(256, 1e-38f), large(256, 1e30f);
    std::vector<float> alternating(256);
    for (int i = 0; i < 256; ++i) { alternating[i] = (i & 1) ? 1.0f : -1.0f; }
    std::vector<float> spike_lo(256, 0.0f), spike_hi(256, 0.0f);
    spike_lo[0] = -1.0f;
    spike_hi[3] = 1.0f;

    push(zeros);
    push(ones);
    push(neg_one);
    push(small);
    push(large);
    push(alternating);
    push(spike_lo);
    push(spike_hi);

    // Deterministic pseudo-random rows (LCG; no <random> so the input set is stable).
    std::uint64_t state = 0x9E3779B97F4A7C15ull;
    auto next = [&]() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return state;
    };
    for (int row = 0; row < 64; ++row) {
        std::vector<float> r(256);
        for (auto& x : r) {
            // Uniform in [-4, 4), with a few hot channels like the distortion suite uses.
            x = static_cast<float>(static_cast<int64_t>((next() >> 11) % 8192) - 4096) *
                (1.0f / 1024.0f);
        }
        r[5]   *= 6.0f;
        r[90]  *= 5.0f;
        r[177] *= 7.0f;
        push(r);
    }
    return rows;
}

// Compare one layout's device output against the exact host model, bit for bit.
int check_layout(const char* label, const char* layout, const float* in, int n_rows,
                 std::array<std::array<float, 8>, 32> (*simulate)(const float*),
                 void (*kernel)(const float*, float*), int lane_dim) {
    const std::size_t bytes = static_cast<std::size_t>(n_rows) * 256 * sizeof(float);
    float *d_in = nullptr, *d_out = nullptr;
    if (cudaMalloc(&d_in, bytes) != cudaSuccess || cudaMalloc(&d_out, bytes) != cudaSuccess) {
        std::cerr << label << ": cudaMalloc failed\n";
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }
    if (cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << label << ": copy-in failed\n";
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }
    kernel<<<n_rows, 32>>>(d_in, d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << label << " " << layout << ": " << cudaGetErrorString(cudaGetLastError())
                  << "\n";
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }
    std::vector<float> got(256 * n_rows);
    cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    int failures = 0;
    for (int row = 0; row < n_rows; ++row) {
        const auto expect = simulate(in + 256 * row);
        for (int l = 0; l < 32; ++l) {
            for (int r = 0; r < 8; ++r) {
                const int dim = (lane_dim == 0) ? (l + 32 * r) : (8 * l + r);
                const float want = expect[l][r];
                const float have = got[256 * row + dim];
                if (std::memcmp(&want, &have, sizeof(float)) != 0) {
                    ++failures;
                    if (failures <= 8) {
                        std::cerr << label << " " << layout << " row " << row << " dim " << dim
                                  << ": want " << want << " got " << have << "\n";
                    }
                }
            }
        }
    }
    if (failures == 0) {
        std::cout << "  " << label << " " << layout << ": " << n_rows
                  << " row(s) bit-identical to the exact host model\n";
    } else {
        std::cerr << "  " << label << " " << layout << ": " << failures << " bit mismatch(es)\n";
    }
    return failures;
}

// Run the project's kernel and the upstream reference kernel on the same rows and require
// bit-identical outputs (task 1.1: 同一输入、同一 lane 映射, 逐位一致).
int check_upstream_match(const char* label, const char* layout, const float* in, int n_rows,
                         void (*ours_kernel)(const float*, float*),
                         void (*upstream_kernel)(const float*, float*)) {
    const std::size_t bytes = static_cast<std::size_t>(n_rows) * 256 * sizeof(float);
    float *d_in = nullptr, *d_ours = nullptr, *d_upstream = nullptr;
    if (cudaMalloc(&d_in, bytes) != cudaSuccess || cudaMalloc(&d_ours, bytes) != cudaSuccess ||
        cudaMalloc(&d_upstream, bytes) != cudaSuccess) {
        std::cerr << label << ": cudaMalloc failed\n";
        cudaFree(d_in);
        cudaFree(d_ours);
        cudaFree(d_upstream);
        return 1;
    }
    if (cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << label << ": copy-in failed\n";
        cudaFree(d_in);
        cudaFree(d_ours);
        cudaFree(d_upstream);
        return 1;
    }
    ours_kernel<<<n_rows, 32>>>(d_in, d_ours);
    upstream_kernel<<<n_rows, 32>>>(d_in, d_upstream);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << label << " " << layout << ": " << cudaGetErrorString(cudaGetLastError()) << "\n";
        cudaFree(d_in);
        cudaFree(d_ours);
        cudaFree(d_upstream);
        return 1;
    }
    std::vector<float> got_ours(256 * n_rows), got_upstream(256 * n_rows);
    cudaMemcpy(got_ours.data(), d_ours, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(got_upstream.data(), d_upstream, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_ours);
    cudaFree(d_upstream);

    int failures = 0;
    for (std::size_t i = 0; i < got_ours.size(); ++i) {
        if (std::memcmp(&got_ours[i], &got_upstream[i], sizeof(float)) != 0) {
            ++failures;
            if (failures <= 8) {
                std::cerr << label << " " << layout << " element " << i << ": project "
                          << got_ours[i] << " upstream " << got_upstream[i] << "\n";
            }
        }
    }
    if (failures == 0) {
        std::cout << "  " << label << " " << layout << ": " << got_ours.size()
                  << " element(s) bit-identical to the upstream reference\n";
    } else {
        std::cerr << "  " << label << " " << layout << ": " << failures
                  << " bit mismatch(es) vs upstream\n";
    }
    return failures;
}

} // namespace

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "no CUDA device: skipping\n";
        return 77; // skip: no GPU
    }

    const std::vector<float> rows = make_rows();
    const int n_rows              = static_cast<int>(rows.size() / 256);

    int failures = 0;
    // (1) literal device-vs-upstream bit-match (task 1.1).
    failures += check_upstream_match("hadamard_d256", "strided", rows.data(), n_rows,
                                     &strided_row_kernel, &upstream_strided_row_kernel);
    failures += check_upstream_match("hadamard_d256", "contiguous", rows.data(), n_rows,
                                     &contiguous_row_kernel, &upstream_contiguous_row_kernel);
    // (2) device-vs-exact-host-model bit-match (association order).
    failures += check_layout("hadamard_d256", "strided", rows.data(), n_rows, &simulate_strided,
                             &strided_row_kernel, 0);
    failures += check_layout("hadamard_d256", "contiguous", rows.data(), n_rows,
                             &simulate_contiguous, &contiguous_row_kernel, 1);

    if (failures == 0) {
        std::cout << "hadamard_d256 device bit-match: all checks passed\n";
        return 0;
    }
    std::cerr << "hadamard_d256 device bit-match: " << failures << " check(s) failed\n";
    return 1;
}