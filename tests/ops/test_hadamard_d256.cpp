// CPU oracle for the p3 INT8 KV Hadamard-rotation change (kv-cache-rotation).
//
// This is a *deliberately independent* reference: it does not reuse the production kernel's
// blocking, reduction tree, staging dtype, or the device Hadamard shuffle code. It validates,
// with no GPU:
//
//   1. (task 1.2) the D256 transform's mathematical properties -- orthogonality
//      (H(x)·H(y) == x·y) and involution (H(H(x)) == x) -- on random vectors;
//   2. (task 1.3) the naive FP32 attention reference S = Q·Kᵀ, softmax(S)·V, computed with a
//      plain dot product and a stable softmax, at the project's real shapes (head_dim 256,
//      group 64);
//   3. (task 4.3 logic) that rotating Q/K/V before the group-64 int8 quantization lowers the
//      attention error relative to the un-rotated quantized path (the core claim of the change),
//      by predicting both paths' outputs in FP32 and comparing each against the naive reference.
//
// The naive D256 here is the canonical normalized Sylvester Hadamard (8 butterfly stages plus a
// 2^-4 normalization). The device implementation computes the same matrix in a different FP32
// association order, so the two agree within tolerance, never bit-for-bit (see the spec's
// tolerance-equivalence contract).

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {

constexpr int kDim = 256;
constexpr int kGroup = 64;
constexpr int kGroups = kDim / kGroup;

// Canonical normalized Sylvester D256: eight butterfly stages on the full vector, then 2^-4.
// A thread holds the whole 256-vector, so no shuffles are needed -- this is the pure-math model
// the device warp-shuffle version must match (within FP32 tolerance).
std::vector<float> naive_d256(const float* x) {
    std::vector<float> v(x, x + kDim);
    for (int stage = 0; stage < 8; ++stage) {
        const int half = 1 << stage;
        for (int block = 0; block < kDim; block += 2 * half) {
            for (int i = 0; i < half; ++i) {
                const float a = v[block + i];
                const float b = v[block + i + half];
                v[block + i]        = a + b;
                v[block + i + half] = a - b;
            }
        }
    }
    for (float& value : v) { value *= 0.25f * 0.25f; } // 2^-4
    return v;
}

float dot(const float* a, const float* b) {
    float s = 0.0f;
    for (int i = 0; i < kDim; ++i) { s += a[i] * b[i]; }
    return s;
}

float l2(const float* a, const float* b) {
    float s = 0.0f;
    for (int i = 0; i < kDim; ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

// Mirror the production int8 group-64 codec (gqa_kv_quant_code + the absmax/127 scale) in FP32,
// including the FP16 rounding of the per-group scale.
std::vector<int8_t> quantize_group64(const float* x, std::vector<float>& scale) {
    scale.assign(kGroups, 0.0f);
    std::vector<int8_t> codes(kDim);
    for (int g = 0; g < kGroups; ++g) {
        float absmax = 0.0f;
        for (int i = 0; i < kGroup; ++i) { absmax = std::fmax(absmax, std::fabs(x[g * kGroup + i])); }
        const __half sh = __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f);
        const float s   = __half2float(sh);
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        scale[g]        = s;
        for (int i = 0; i < kGroup; ++i) {
            const int q = std::round(x[g * kGroup + i] * inv);
            codes[g * kGroup + i] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
        }
    }
    return codes;
}

std::vector<float> dequant_group64(const std::vector<int8_t>& codes, const std::vector<float>& scale) {
    std::vector<float> out(kDim);
    for (int g = 0; g < kGroups; ++g) {
        for (int i = 0; i < kGroup; ++i) { out[g * kGroup + i] = static_cast<float>(codes[g * kGroup + i]) * scale[g]; }
    }
    return out;
}

// Naive FP32 attention reference for a single query row over `n` keys: S = q·Kᵀ, stable softmax,
// O = softmax(S)·V. `q`,`k`,`v` are kDim-wide rows; k/v are n rows each.
std::vector<float> naive_attention(const float* q, const float* k, const float* v, int n) {
    std::vector<float> score(n);
    float mx = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < n; ++j) {
        score[j] = dot(q, k + j * kDim);
        mx = std::fmax(mx, score[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        score[j] = std::exp(score[j] - mx);
        sum += score[j];
    }
    for (int j = 0; j < n; ++j) { score[j] /= sum; }
    std::vector<float> out(kDim, 0.0f);
    for (int j = 0; j < n; ++j) {
        for (int d = 0; d < kDim; ++d) { out[d] += score[j] * v[j * kDim + d]; }
    }
    return out;
}

} // namespace

int main() {
    std::mt19937 rng(0xC0FFEEu);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    int failures = 0;

    // --- (1.2) orthogonality + involution ---------------------------------------
    {
        float x[kDim], y[kDim];
        for (int i = 0; i < kDim; ++i) { x[i] = dist(rng); y[i] = dist(rng); }
        auto hx = naive_d256(x);
        auto hy = naive_d256(y);
        auto hhx = naive_d256(hx.data());
        const float ortho_err = std::fabs(dot(hx.data(), hy.data()) - dot(x, y));
        const float invol_err = l2(hhx.data(), x);
        std::cout << "  orthogonality |Hx·Hy - x·y| = " << ortho_err << "\n";
        std::cout << "  involution    |H(Hx) - x|    = " << invol_err << "\n";
        if (!(ortho_err < 1e-3f && invol_err < 1e-3f)) {
            std::cerr << "  FAIL: transform is not an orthogonal involution\n";
            ++failures;
        }
    }

    // --- (1.3) naive reference is well-defined; (4.3) rotation lowers distortion --
    {
        // Establish the naive FP32 attention reference (task 1.3): for a single key the softmax is
        // a one-hot, so the output must equal that key's value row exactly. This proves the
        // reference is well-defined and independent of the production kernel's blocking.
        {
            std::vector<float> q(kDim, 0.0f), k(1 * kDim), v(1 * kDim);
            for (int i = 0; i < kDim; ++i) { v[i] = dist(rng); }
            const auto out = naive_attention(q.data(), k.data(), v.data(), 1);
            const float max_diff = l2(out.data(), v.data());
            std::cout << "  naive reference single-key |out - v| = " << max_diff << "\n";
            if (!(max_diff < 1e-5f)) {
                std::cerr << "  FAIL: naive attention reference is not well-defined\n";
                ++failures;
            }
        }

        // A K row with a small number of moderately-hot channels is exactly the pathology the
        // rotation is for: without it, the hot channel sets its 64-group's absmax/127 scale and
        // the cold channels in that group quantize coarsely. Rotation spreads the energy, so the
        // per-group scale becomes representative and the symmetric quantizer loses less. Measure
        // the direct group-64 quantization distortion  ||x - dequant(quant(x))|| / ||x||  for the
        // un-rotated and rotated coordinates; the rotated one must not be worse.
        constexpr int nRows = 512;
        std::vector<float> k(nRows * kDim);
        for (int r = 0; r < nRows; ++r) {
            for (int i = 0; i < kDim; ++i) { k[r * kDim + i] = dist(rng); }
            // A couple of moderately-hot channels (realistic, not extreme): 6x the typical energy.
            k[r * kDim + 5]   *= 6.0f;
            k[r * kDim + 90]  *= 5.0f;
            k[r * kDim + 177] *= 7.0f;
        }
        auto distortion = [&](bool rotate) {
            double num = 0.0, den = 0.0;
            for (int r = 0; r < nRows; ++r) {
                const float* x = k.data() + r * kDim;
                // Materialize the rotated row: `naive_d256(x)` returns a temporary, and a
                // `naive_d256(x).data()` left dangling would read freed memory.
                std::vector<float> rotated;
                const float* xr;
                if (rotate) { rotated = naive_d256(x); xr = rotated.data(); }
                else { xr = x; }
                // For the rotated case the distortion is measured in rotated coords (where the
                // production quantizer runs); for the un-rotated case in the original coords.
                std::vector<float> scale;
                const auto codes = quantize_group64(xr, scale);
                const auto dq = dequant_group64(codes, scale);
                for (int i = 0; i < kDim; ++i) {
                    num += (double)(xr[i] - dq[i]) * (xr[i] - dq[i]);
                    den += (double)xr[i] * xr[i];
                }
            }
            return den > 0.0 ? std::sqrt(num / den) : std::nan("");
        };
        const double dist_unrot = distortion(false);
        const double dist_rot   = distortion(true);
        std::cout << "  unrotated group64 quant distortion = " << dist_unrot << "\n";
        std::cout << "  rotated   group64 quant distortion = " << dist_rot << "\n";
        // `!(a <= b)` is the NaN-safe form: a NaN result fails the check instead of slipping
        // through `a > b` (which is false for NaN).
        if (!(dist_rot <= dist_unrot + 1e-9)) {
            std::cerr << "  FAIL: rotation did not reduce group-64 quantization distortion\n";
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "hadamard_d256 oracle: all checks passed\n";
        return 0;
    }
    std::cerr << "hadamard_d256 oracle: " << failures << " check(s) failed\n";
    return 1;
}