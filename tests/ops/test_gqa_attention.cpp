#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/gqa_attention_fixture.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The cache model, oracle, codec, and criteria live in the shared fixture so the tp2 head-local
// parity suite (tests/ops/test_attention_headlocal.cpp) builds its per-device pools from exactly
// the same model this suite builds its tp1 pool from.
using namespace ninfer::test::gqa;

constexpr Geometry kGeometries[] = {
    {"qwen3_6_27b", 24, 4},
    {"qwen3_6_35b_a3b", 16, 2},
    {"qwen3_6_27b_tp4", 6, 1},
};

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    std::uint32_t seed;
};
std::string case_label(const char* entry, const Geometry& geometry, DType dtype,
                       const AttentionCase& test_case, MappingPattern mapping,
                       KvCacheStorage storage = KvCacheStorage::BFloat16) {
    if (storage == KvCacheStorage::BFloat16) { storage = storage_for_dtype(dtype); }
    return std::string(entry) + " " + geometry.name + " " + cache_name(dtype, storage) +
           " mapping=" + mapping_name(mapping) + " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max);
}


int run_append_case(const Geometry& geometry, DType dtype, MappingPattern mapping,
                    std::uint32_t seed, std::int32_t tokens = 3, std::int32_t base = 63,
                    KvCacheStorage storage = KvCacheStorage::BFloat16) {
    if (storage == KvCacheStorage::BFloat16) { storage = storage_for_dtype(dtype); }
    const std::int32_t max_context = base + tokens + 4;
    const std::size_t elements =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(geometry.kv_heads) * tokens;
    std::vector<float> k = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);
    inject_codec_edges(geometry, tokens, k, v);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base + token;
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, seed + 10u, storage);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    DeviceCache cache(initial, mapping);

    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dpositions(positions.size() * sizeof(std::int32_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dpositions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dpositions.data(), DType::I32, {tokens});

    ops::gqa_kv_append(tk, tv, tp, cache.view(), nullptr);
    cuda_synchronize();

    const std::string label = std::string("gqa_kv_append ") + geometry.name + " " +
                              cache_name(dtype, storage) + " mapping=" + mapping_name(mapping);
    int failures = verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dpositions, positions);
    failures += cache.verify_guards(label);
    return failures;
}

int run_a1_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping, KvCacheStorage storage = KvCacheStorage::BFloat16) {
    if (storage == KvCacheStorage::BFloat16) { storage = storage_for_dtype(dtype); }
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, test_case.tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, test_case.seed + 10u, storage);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference = ideal_attention(q, expected, positions);
    DeviceCache cache(initial, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    // Positive finite poison makes a producer's unwritten partials visible to the oracle:
    // negative or zero partial_l values would let the reducer silently skip them.
    workspace_buffer.fill(0x3f);
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kAttentionScale, cache.batch_view(),
                       envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label = case_label("gqa_attention", geometry, dtype, test_case, mapping, storage);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype, storage));
    failures += verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += verify_positions(label + " table row unchanged", dtable_row, {table_row});
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

int run_a3_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping, KvCacheStorage storage = KvCacheStorage::BFloat16) {
    if (storage == KvCacheStorage::BFloat16) { storage = storage_for_dtype(dtype); }
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache cache_host = make_cache(geometry, dtype, max_context, test_case.seed + 10u, storage);
    const std::vector<double> reference = ideal_attention(q, cache_host, positions);
    DeviceCache cache(cache_host, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    workspace_buffer.fill(0x3f);
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                              nullptr);
    cuda_synchronize();

    const std::string label =
        case_label("gqa_attention_cached", geometry, dtype, test_case, mapping, storage);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype, storage));
    failures += verify_cache(label + " cache unchanged", cache.snapshot(), cache_host);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

struct BatchAttentionCase {
    std::int32_t width;
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    MappingPattern mapping;
    std::uint32_t seed;
};

std::vector<float> extract_request_columns(const std::vector<float>& source,
                                           std::size_t column_elements, std::int32_t width,
                                           std::int32_t request, std::int32_t valid) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::vector<float> result(static_cast<std::size_t>(valid) * column_elements);
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(begin), result.size(), result.begin());
    return result;
}

void insert_request_columns(const std::vector<double>& source, std::size_t column_elements,
                            std::int32_t width, std::int32_t request,
                            std::vector<double>& destination) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::copy(source.begin(), source.end(),
              destination.begin() + static_cast<std::ptrdiff_t>(begin));
}

int verify_invalid_columns_zero(const std::string& label, std::span<const std::uint16_t> output,
                                const Geometry& geometry, std::int32_t width,
                                std::span<const std::int32_t> valid_columns) {
    int failures                      = 0;
    const std::size_t column_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    for (std::size_t batch = 0; batch < valid_columns.size(); ++batch) {
        for (std::int32_t token = valid_columns[batch]; token < width; ++token) {
            const std::size_t begin =
                (batch * static_cast<std::size_t>(width) + token) * column_elements;
            for (std::size_t element = 0; element < column_elements; ++element) {
                if (output[begin + element] != 0) {
                    if (failures == 0) {
                        std::cerr << label << ": invalid output column is not BF16 zero at row "
                                  << batch << " column " << token << '\n';
                    }
                    ++failures;
                }
            }
        }
    }
    return failures;
}

int run_batch_case(const Geometry& geometry, DType dtype, const BatchAttentionCase& test_case,
                   KvCacheStorage storage = KvCacheStorage::BFloat16) {
    if (storage == KvCacheStorage::BFloat16) { storage = storage_for_dtype(dtype); }
    const std::int32_t batch = static_cast<std::int32_t>(test_case.contexts.size());
    if (batch <= 0 || test_case.valid_columns.size() != static_cast<std::size_t>(batch) ||
        test_case.table_rows.size() != static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("invalid GQA batch test profile");
    }

    std::int32_t maximum_visible = 1;
    for (std::int32_t row = 0; row < batch; ++row) {
        maximum_visible =
            std::max(maximum_visible, test_case.contexts[static_cast<std::size_t>(row)] +
                                          test_case.valid_columns[static_cast<std::size_t>(row)]);
    }
    const std::int32_t max_context       = maximum_visible + 3;
    const std::size_t q_column_elements  = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    const std::size_t kv_column_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads;
    const std::size_t columns            = static_cast<std::size_t>(test_case.width) * batch;
    std::vector<float> q =
        make_bf16_values(q_column_elements * columns, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, static_cast<std::int32_t>(columns), k, v);

    std::vector<std::int32_t> positions(columns, 0);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < valid; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] =
                test_case.contexts[static_cast<std::size_t>(row)] + token;
        }
        const std::int32_t padding_position =
            valid == 0 ? 0 : test_case.contexts[static_cast<std::size_t>(row)] + valid - 1;
        for (std::int32_t token = valid; token < test_case.width; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] = padding_position;
        }
    }

    std::vector<HostCache> initial;
    initial.reserve(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial.push_back(
            make_cache(geometry, dtype, max_context, test_case.seed + 20u + 3u * row, storage));
    }
    std::vector<HostCache> expected = initial;
    std::vector<double> reference(q_column_elements * columns, 0.0);
    for (std::int32_t request = 0; request < batch; ++request) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(request)];
        if (valid == 0) { continue; }
        const std::int32_t table_row = test_case.table_rows[static_cast<std::size_t>(request)];
        std::vector<std::int32_t> row_positions(static_cast<std::size_t>(valid));
        std::copy_n(positions.begin() + static_cast<std::ptrdiff_t>(request * test_case.width),
                    valid, row_positions.begin());
        const std::vector<float> row_q =
            extract_request_columns(q, q_column_elements, test_case.width, request, valid);
        const std::vector<float> row_k =
            extract_request_columns(k, kv_column_elements, test_case.width, request, valid);
        const std::vector<float> row_v =
            extract_request_columns(v, kv_column_elements, test_case.width, request, valid);
        append_cache(expected[static_cast<std::size_t>(table_row)], row_k, row_v, row_positions);
        insert_request_columns(
            ideal_attention(row_q, expected[static_cast<std::size_t>(table_row)], row_positions),
            q_column_elements, test_case.width, request, reference);
    }

    BatchDeviceCache cache(initial, test_case.mapping);
    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvalid(test_case.valid_columns.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_rows(test_case.table_rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dvalid.copy_from_host(test_case.valid_columns.data(),
                          test_case.valid_columns.size() * sizeof(std::int32_t));
    dtable_rows.copy_from_host(test_case.table_rows.data(),
                               test_case.table_rows.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tp(dp.data(), DType::I32, {test_case.width, batch});
    Tensor tvalid(dvalid.data(), DType::I32, {batch});
    Tensor ttable_rows(dtable_rows.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(maximum_visible),
                                             static_cast<std::uint32_t>(maximum_visible)};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, batch, test_case.width, test_case.width);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    const bool masked = std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](std::int32_t valid) { return valid != test_case.width; });
    ops::gqa_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, ttable_rows, kAttentionScale,
                       cache.view(), envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label = std::string("gqa_attention batch ") + geometry.name + " " +
                              cache_name(dtype, storage) + " mapping=" + mapping_name(test_case.mapping) +
                              " B=" + std::to_string(batch) +
                              " W=" + std::to_string(test_case.width);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype, storage));
    failures += verify_invalid_columns_zero(label, output_bits, geometry, test_case.width,
                                            test_case.valid_columns);
    failures += cache.verify(label, expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    if (masked) {
        failures +=
            verify_positions(label + " valid columns unchanged", dvalid, test_case.valid_columns);
    }
    failures +=
        verify_positions(label + " table rows unchanged", dtable_rows, test_case.table_rows);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_batch_cases() {
    int failures = 0;
    failures += run_batch_case(kGeometries[0], DType::I8,
                               {6, {127}, {3}, {0}, MappingPattern::Identity, 499u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {16, {49}, {7}, {0}, MappingPattern::Identity, 500u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {1, {63, 2048}, {1, 1}, {1, 0}, MappingPattern::Fragmented, 501u});
    failures += run_batch_case(kGeometries[1], DType::I8,
                               {1,
                                {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Identity,
                                502u});
    failures +=
        run_batch_case(kGeometries[0], DType::I8,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 503u});
    failures += run_batch_case(kGeometries[1], DType::BF16,
                               {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Identity, 504u});
    // p4 fp8-e4m3: FP8 batched coverage mirrors the INT8 batch shapes (batched small-T decode,
    // the wide per-row table, and the fragmented masked case) so the FP8 dispatch is checked on the
    // same per-path surface as INT8.
    failures += run_batch_case(kGeometries[0], DType::FP8_E4M3FN,
                               {6, {127}, {3}, {0}, MappingPattern::Identity, 505u});
    failures += run_batch_case(kGeometries[1], DType::FP8_E4M3FN,
                               {1,
                                {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Identity,
                                506u});
    failures +=
        run_batch_case(kGeometries[0], DType::FP8_E4M3FN,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 507u});
    // p4 nvfp4-g16: NVFP4 batched coverage mirrors the FP8 batch shapes (batched small-T decode,
    // the wide per-row table, and the fragmented masked case) so the NVFP4 dispatch (packed
    // two-per-byte E2M1 codes + E4M3 group-16 scale) is checked on the same per-path surface.
    failures += run_batch_case(kGeometries[0], DType::U8,
                               {6, {127}, {3}, {0}, MappingPattern::Identity, 508u});
    failures += run_batch_case(kGeometries[1], DType::U8,
                               {1,
                                {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Identity,
                                509u});
    failures +=
        run_batch_case(kGeometries[0], DType::U8,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 510u});
    // p4 k8v4: asymmetric batched coverage mirrors the NVFP4 batch shapes (batched small-T decode,
    // the wide per-row table, and the fragmented masked case) so the K8V4 dispatch is checked on
    // the same per-path surface with the key plane as FP8 and the value plane as NVFP4.
    failures += run_batch_case(kGeometries[0], DType::FP8_E4M3FN,
                               {6, {127}, {3}, {0}, MappingPattern::Identity, 511u},
                               KvCacheStorage::Fp8KeyNvfp4Value);
    failures += run_batch_case(kGeometries[1], DType::FP8_E4M3FN,
                               {1,
                                {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Identity,
                                512u},
                               KvCacheStorage::Fp8KeyNvfp4Value);
    failures += run_batch_case(kGeometries[0], DType::FP8_E4M3FN,
                               {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented,
                                513u},
                               KvCacheStorage::Fp8KeyNvfp4Value);
    return failures;
}

int run_geometry(const Geometry& geometry) {
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::I8, DType::FP8_E4M3FN, DType::U8}) {
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            failures += run_append_case(geometry, dtype, mapping, 100u + geometry.q_heads);
            failures += run_a1_case(geometry, dtype, {6, 61, 67, 190u}, mapping);
            failures += run_a3_case(geometry, dtype, {1, 128, 129, 191u}, mapping);
        }
        if (dtype == DType::I8 || dtype == DType::FP8_E4M3FN || dtype == DType::U8) {
            failures += run_append_case(geometry, dtype, MappingPattern::Fragmented,
                                        150u + geometry.q_heads, 129, 61);
        }

        const AttentionCase a1_cases[] = {
            {1, 0, 1, 201u},    {6, 17, 23, 202u},   {7, 17, 512, 203u},
            {17, 31, 48, 204u}, {66, 63, 129, 205u},
            // Fresh prefill at the flash-route width frontier: 128 tokens from column 0
            // (min == max visible keys, width >= 64), spanning two 64-token pages. 24q/6q
            // geometries take the flash route (6q through split-D); 16q falls back to
            // chunked small-T, which keeps this case a cross-route cache-layout check.
            {128, 0, 128, 206u},
        };
        for (const AttentionCase& test_case : a1_cases) {
            failures += run_a1_case(geometry, dtype, test_case, MappingPattern::Identity);
        }

        const AttentionCase a3_cases[] = {
            {1, 31, 32, 301u},
            {7, 17, 512, 302u},
            {17, 31, 48, 303u},
        };
        for (const AttentionCase& test_case : a3_cases) {
            failures += run_a3_case(geometry, dtype, test_case, MappingPattern::Identity);
        }

        if (geometry.q_heads == 16) {
            // Loose execution envelopes straddle the two registered host-resource frontiers.
            // Device positions, not these bounds, continue to define the oracle result.
            failures += run_a1_case(geometry, dtype, {7, 17, 513, 401u}, MappingPattern::Identity);
            failures += run_a3_case(geometry, dtype, {7, 17, 513, 402u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1024, 403u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1025, 404u}, MappingPattern::Identity);
        }
    }

    // p4 k8v4: the asymmetric format (key plane FP8 group-64, value plane NVFP4 group-16) mirrors
    // the symmetric FP8/NVFP4 per-path surface -- append, a1, a3, the quantized extra-append, the
    // prefill frontier, and the 16q loose-envelope cases -- so the K8V4 dispatch (key decoded via
    // the FP8 codec, value via the NVFP4 codec) is checked on the same paths.
    {
        const DType k8v4_dtype = DType::FP8_E4M3FN;
        const KvCacheStorage k8v4 = KvCacheStorage::Fp8KeyNvfp4Value;
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            failures += run_append_case(geometry, k8v4_dtype, mapping, 200u + geometry.q_heads,
                                        3, 63, k8v4);
            failures += run_a1_case(geometry, k8v4_dtype, {6, 61, 67, 290u}, mapping, k8v4);
            failures += run_a3_case(geometry, k8v4_dtype, {1, 128, 129, 291u}, mapping, k8v4);
        }
        failures += run_append_case(geometry, k8v4_dtype, MappingPattern::Fragmented,
                                    250u + geometry.q_heads, 129, 61, k8v4);

        const AttentionCase k8v4_a1_cases[] = {
            {1, 0, 1, 292u},    {6, 17, 23, 293u},   {7, 17, 512, 294u},
            {17, 31, 48, 295u}, {66, 63, 129, 296u},
            // Fresh prefill at the flash-route width frontier, as in the symmetric suites.
            {128, 0, 128, 297u},
        };
        for (const AttentionCase& test_case : k8v4_a1_cases) {
            failures += run_a1_case(geometry, k8v4_dtype, test_case, MappingPattern::Identity, k8v4);
        }

        const AttentionCase k8v4_a3_cases[] = {
            {1, 31, 32, 298u},
            {7, 17, 512, 299u},
            {17, 31, 48, 300u},
        };
        for (const AttentionCase& test_case : k8v4_a3_cases) {
            failures += run_a3_case(geometry, k8v4_dtype, test_case, MappingPattern::Identity, k8v4);
        }

        if (geometry.q_heads == 16) {
            failures += run_a1_case(geometry, k8v4_dtype, {7, 17, 513, 405u}, MappingPattern::Identity,
                                    k8v4);
            failures += run_a3_case(geometry, k8v4_dtype, {7, 17, 513, 406u}, MappingPattern::Identity,
                                    k8v4);
            failures += run_a3_case(geometry, k8v4_dtype, {16, 17, 1024, 407u},
                                    MappingPattern::Identity, k8v4);
            failures += run_a3_case(geometry, k8v4_dtype, {16, 17, 1025, 408u},
                                    MappingPattern::Identity, k8v4);
        }
    }
    return failures;
}

int run_quantized_split_policy_cases() {
    // Volta's TP2 quantized producer (INT8 and FP8 share the tensor-core decode path) must use the
    // same active split policy as its reducer. The short windows demand more splits than BF16; the
    // loose 6K envelope permits more launches than the policy consumes, exposing the opposite
    // mismatch. A1 and A3 both compare every output directly with the fixture's FP64 oracle.
    constexpr Geometry geometry{"qwen3_6_27b_tp2", 12, 2};
    constexpr AttentionCase cases[] = {
        {5, 252, 257, 601u},
        {6, 139, 145, 602u},
        {6, 6138, 8199, 603u},
        {5, 32764, 32769, 604u},
        {1, 16390, 16391, 605u},
        // TP2 MTP4 verifier: fragmented pages, an incomplete final key tile and
        // enough tiles per split to exercise the packed-KV lookahead repeatedly.
        {5, 85000, 85005, 607u},
    };
    int failures = 0;
    for (const DType dtype : {DType::I8, DType::FP8_E4M3FN, DType::U8}) {
        for (const AttentionCase& test_case : cases) {
            failures += run_a1_case(geometry, dtype, test_case, MappingPattern::Fragmented);
            failures += run_a3_case(geometry, dtype, test_case, MappingPattern::Fragmented);
        }
    }
    // p4 k8v4: the asymmetric format takes the same tensor-core decode path on TP2, so it gets the
    // same split-policy surface (key plane FP8, value plane NVFP4).
    for (const AttentionCase& test_case : cases) {
        failures += run_a1_case(geometry, DType::FP8_E4M3FN, test_case, MappingPattern::Fragmented,
                                KvCacheStorage::Fp8KeyNvfp4Value);
        failures += run_a3_case(geometry, DType::FP8_E4M3FN, test_case, MappingPattern::Fragmented,
                                KvCacheStorage::Fp8KeyNvfp4Value);
    }
    return failures;
}

int verify_workspace_capacity_contract() {
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::I8, DType::FP8_E4M3FN, DType::U8}) {
        constexpr ops::GqaExecutionEnvelope envelope{1, 1025};
        const std::size_t interval =
            ops::gqa_attention_workspace_capacity_bytes(16, dtype, envelope, 1, 1, 17);
        std::size_t witness = 0;
        for (std::int32_t tokens = 1; tokens <= 17; ++tokens) {
            witness = std::max(witness, ops::gqa_attention_workspace_capacity_bytes(
                                            16, dtype, envelope, 1, tokens, tokens));
        }
        if (interval != witness) {
            std::cerr << "gqa_attention interval capacity has no exact route witness\n";
            ++failures;
        }
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys}, 1, 1, 1);
    } catch (const std::invalid_argument&) {
        std::cerr << "gqa_attention rejected its maximum visible-key envelope\n";
        ++failures;
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys + 1}, 1, 1, 1);
        std::cerr << "gqa_attention accepted an envelope outside the launcher domain\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

} // namespace

// p3 kv-cache-rotation: regression guard for the rotation's designed benefit. On a
// high-energy-channel (outlier) distribution the rotated INT8 path must keep its attention
// output close to the scheme-independent BF16-derived FP64 reference. The unrotated path crushes
// the non-outlier dims of an outlier-dominated group (its group scale is set by the outlier) and
// lands ~0.0105 relative-L2 here, so the 0.008 bound passes for the rotated path (~0.0058) and
// FAILS if the rotation is dropped -- guarding the benefit, not merely the tolerance.
int run_outlier_regression_cases() {
    int failures = 0;
    struct C {
        Geometry g;
        std::int32_t tokens;
        std::uint32_t seed;
        DType dtype;
        double rel_bound;
        // BFloat16 here means "derive from dtype" (storage_for_dtype); only the K8V4 cases set it
        // explicitly, since they share the FP8 dtype with the symmetric FP8 cases.
        KvCacheStorage storage = KvCacheStorage::BFloat16;
    };
    const C cases[] = {
        // INT8: rotated ~0.0058 passes; unrotated ~0.0105 fails.
        {kGeometries[0], 128, 910u, DType::I8, 0.008},
        {kGeometries[1], 128, 911u, DType::I8, 0.008},
        {kGeometries[2], 128, 912u, DType::I8, 0.008},
        // p4 fp8-e4m3: E4M3 is coarser than INT8, so the outlier-channel cost is higher; the bound
        // is set loose enough to pass the rotated FP8 path while still rejecting a dropped rotation
        // (which lands at rel_l2/ref_rms ~ 1.4, i.e. 45x above this bound). CALIBRATED (sm_70, task
        // 1.5): the three rotated FP8 geometries measure 2.21e-2 / 2.48e-2 / 2.67e-2, so 0.030
        // carries ~11% margin over the worst.
        {kGeometries[0], 128, 913u, DType::FP8_E4M3FN, 0.030},
        {kGeometries[1], 128, 914u, DType::FP8_E4M3FN, 0.030},
        {kGeometries[2], 128, 915u, DType::FP8_E4M3FN, 0.030},
        // p4 nvfp4-g16: the E2M1 grid (8 values, max 6) is coarser than E4M3, so the outlier-
        // channel cost is higher still; the bound stays far below a dropped-rotation defect
        // (rel_l2/ref_rms ~ 1.4, ~14x the worst measured here). CALIBRATED (sm_70, task 2.5): the
        // three rotated NVFP4 geometries measure 9.91e-2 / 9.14e-2 / 9.11e-2, so 0.110 carries
        // ~11% margin over the worst (9.91e-2); a dropped rotation lands at ~1.4, far above the
        // bound, so it still discriminates the rotation contract. See
        // evidence/nvfp4-g16-oracle-tolerances.md for the measured worst-case and final margin.
        {kGeometries[0], 128, 916u, DType::U8, 0.110},
        {kGeometries[1], 128, 917u, DType::U8, 0.110},
        {kGeometries[2], 128, 918u, DType::U8, 0.110},
        // p4 k8v4: the value plane is the coarser E2M1 grid (as in NVFP4, worst ~9.91e-2) and the key
        // plane adds E4M3 rounding to the attention weights; the 6q geometry (fewer heads, wider
        // per-head reduction) carries the worst outlier-channel cost. CALIBRATED (sm_70, task 3.4):
        // the three rotated K8V4 geometries measure 9.22e-2 (24q) / 1.046e-1 (16q) / 1.280e-1 (6q),
        // so 0.141 carries ~11% margin over the worst (1.280e-1) and stays ~10x below a
        // dropped-rotation defect (rel_l2/ref_rms ~ 1.4), so it still discriminates the rotation
        // contract. See evidence/k8v4-asym-oracle-tolerances.md for the measured worst-case.
        {kGeometries[0], 128, 919u, DType::FP8_E4M3FN, 0.141, KvCacheStorage::Fp8KeyNvfp4Value},
        {kGeometries[1], 128, 920u, DType::FP8_E4M3FN, 0.141, KvCacheStorage::Fp8KeyNvfp4Value},
        {kGeometries[2], 128, 921u, DType::FP8_E4M3FN, 0.141, KvCacheStorage::Fp8KeyNvfp4Value},
    };
    constexpr int kOutlierDims = 8;
    constexpr float kAmp       = 32.0f;
    for (const C& c : cases) {
        const Geometry& geometry = c.g;
        const std::int32_t tokens = c.tokens;
        const std::int32_t base   = 1;
        const std::size_t q_elements =
            static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(geometry.q_heads) *
            static_cast<std::size_t>(tokens);
        const std::size_t kv_elements =
            static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(geometry.kv_heads) *
            static_cast<std::size_t>(tokens);
        std::vector<float> q = make_bf16_values(q_elements, c.seed, -0.25f, 0.25f);
        const auto make_outlier = [&](std::uint32_t s) {
            std::vector<float> x = make_bf16_values(kv_elements, s, -0.25f, 0.25f);
            for (std::size_t row = 0; row * static_cast<std::size_t>(kHeadDim) < kv_elements;
                 ++row) {
                for (int d = 0; d < kOutlierDims; ++d) {
                    x[row * static_cast<std::size_t>(kHeadDim) + static_cast<std::size_t>(d)] *= kAmp;
                }
            }
            return x;
        };
        std::vector<float> k = make_outlier(c.seed + 1u);
        std::vector<float> v = make_outlier(c.seed + 2u);
        std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
        for (std::int32_t t = 0; t < tokens; ++t) { positions[t] = base + t; }
        const std::int32_t max_context = base + tokens + 3;
        const KvCacheStorage storage =
            (c.storage == KvCacheStorage::BFloat16) ? storage_for_dtype(c.dtype) : c.storage;
        // Scheme-independent reference: FP64 ideal over the BF16 cache (bit-identical across trees).
        const HostCache bf16_initial = make_cache(geometry, DType::BF16, max_context, c.seed + 10u);
        HostCache bf16_expected      = bf16_initial;
        append_cache(bf16_expected, k, v, positions);
        const std::vector<double> reference = ideal_attention(q, bf16_expected, positions);
        // This tree's quantized codec (rotated in the rotated tree).
        const HostCache quant_initial = make_cache(geometry, c.dtype, max_context, c.seed + 10u, storage);
        DeviceCache cache(quant_initial, MappingPattern::Identity);

        const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
        const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
        const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
        GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
        GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
        GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
        dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
        dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
        dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
        dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
        const std::int32_t table_row = 0;
        dtable_row.copy_from_host(&table_row, sizeof(std::int32_t));
        const std::vector<std::uint16_t> canary(q_bits.size(), kOutputCanary);
        dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));
        Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});
        Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
        Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
        Tensor tp(dp.data(), DType::I32, {tokens});
        Tensor ttable_row(dtable_row.data(), DType::I32, {1});
        Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});
        const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(base + tokens),
                                                 static_cast<std::uint32_t>(base + tokens)};
        const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
            geometry.q_heads, c.dtype, envelope, 1, tokens, tokens);
        GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
        WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});
        ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kAttentionScale, cache.batch_view(),
                           envelope, workspace, tout, nullptr);
        cuda_synchronize();
        const std::vector<std::uint16_t> bits = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
        const std::vector<double> actual = bf16_bits_to_double(bits);

        long double sq_err = 0, sq_ref = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const double d = actual[i] - reference[i];
            sq_err += static_cast<long double>(d * d);
            sq_ref += static_cast<long double>(reference[i] * reference[i]);
        }
        const double rel_l2  = std::sqrt(static_cast<double>(sq_err / static_cast<long double>(actual.size())));
        const double ref_rms = std::sqrt(static_cast<double>(sq_ref / static_cast<long double>(actual.size())));
        const double ratio   = ref_rms > 0 ? rel_l2 / ref_rms : 0.0;
        const std::string label = std::string("gqa_attention_outlier ") + geometry.name + " " +
                                  cache_name(c.dtype, storage) + " T=" + std::to_string(tokens) +
                                  " dims=" + std::to_string(kOutlierDims);
        if (ratio > c.rel_bound) {
            std::cerr << label << ": outlier-channel attention rel_l2/ref_rms " << ratio
                      << " exceeds " << c.rel_bound << " (rotation benefit lost)\n";
            ++failures;
        } else {
            std::cout << "ok      " << label << " (rel_l2/ref_rms " << ratio << " <= " << c.rel_bound
                      << ")\n";
        }
    }
    return failures;
}

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += verify_workspace_capacity_contract();
    for (const Geometry& geometry : kGeometries) { failures += run_geometry(geometry); }
    failures += run_quantized_split_policy_cases();
    failures += run_batch_cases();
    failures += run_outlier_regression_cases();
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " gqa_attention public-contract correctness\n";
    return failures == 0 ? 0 : 1;
}
