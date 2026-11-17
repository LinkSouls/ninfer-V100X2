#pragma once

// p9-vision-prefix-bridge-nvfp4: opt-in numerical dump for the visual prefix-bridge
// investigation (first-token divergence between the prefix-reuse MTP-bridge path and the
// cold-prefill path on the official NVFP4 artifact).
//
// Compile with -DNINFER_VISION_BRIDGE_DUMP; activate at runtime by setting
// NINFER_BRIDGE_DUMP_DIR to an existing directory. Without the macro every entry point
// compiles away and the execution paths are untouched (no getenv, no files).
//
// Files written (synchronous device->host copies; <label> discriminates the path):
//   <dir>/<label>_prefill_c<chunk_begin>_L<layer>_hidden.bin   per-layer rank-0 residual
//   <dir>/<label>_prefill_c<chunk_begin>_final_hidden.bin      post final-norm hidden
//   <dir>/<label>_prefill_c<chunk_begin>_logits.bin            last-chunk target logits
//   <dir>/<label>_prefill_c<chunk_begin>_mtp_align_embeddings.bin  cold-path MTP input
//   <dir>/<label>_mtp_bridge_pos<position>_input_hidden.bin    bridge restored hidden
//   <dir>/<label>_mtp_bridge_pos<position>_draft_hidden.bin    bridge MTP draft hidden
//   <dir>/<label>_mtp_bridge_pos<position>_logits.bin          bridge target logits
//   <dir>/<label>_checkpoint_frontier<f>_image.bin             GDN state + tail hidden
//   <dir>/manifest.jsonl                                       one JSON record per dump

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <memory>
#include <atomic>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::bridge_dump {

#ifdef NINFER_VISION_BRIDGE_DUMP

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("bridge_dump: ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

void manifest_line(const std::string& dir, const std::string& record) {
    std::ofstream out(dir + "/manifest.jsonl", std::ios::app);
    out << record << '\n';
}

void dump_tensor_file(const std::string& dir, const std::string& name, const ::ninfer::Tensor& t,
                      cudaStream_t stream) {
    if (t.data == nullptr || t.bytes() == 0) { return; }
    std::vector<std::byte> host(static_cast<std::size_t>(t.bytes()));
    check_cuda(cudaMemcpyAsync(host.data(), t.data, static_cast<std::size_t>(t.bytes()),
                               cudaMemcpyDeviceToHost, stream), "dump d2h");
    check_cuda(cudaStreamSynchronize(stream), "dump sync");
    const std::string path = dir + "/" + name + ".bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size()));
    manifest_line(dir, "{\"file\": \"" + name + ".bin\", \"bytes\": " + std::to_string(host.size()) +
                     ", \"dtype\": " + std::to_string(static_cast<int>(t.dtype)) + ", \"ne\": [" +
                     std::to_string(t.ne[0]) + ", " + std::to_string(t.ne[1]) + ", " +
                     std::to_string(t.ne[2]) + ", " + std::to_string(t.ne[3]) + "]}");
}

} // namespace

struct DumpSink {
    std::string dir;
    std::string label;
    std::int32_t chunk_begin = 0;

    DumpSink(std::string dir_, std::string label_) : dir(std::move(dir_)), label(std::move(label_)) {}

    void capture_layer(int layer, const ::ninfer::Tensor& x0, cudaStream_t stream) const {
        dump_tensor_file(dir, label + "_prefill_c" + std::to_string(chunk_begin) + "_L" +
                             std::to_string(layer) + "_hidden",
                         x0, stream);
    }

    void end(const ::ninfer::Tensor& final_hidden, cudaStream_t stream) const {
        dump_tensor_file(dir, label + "_prefill_c" + std::to_string(chunk_begin) + "_final_hidden",
                         final_hidden, stream);
    }

    void dump_logits(const ::ninfer::Tensor& logits, cudaStream_t stream) const {
        dump_tensor_file(dir, label + "_prefill_c" + std::to_string(chunk_begin) + "_logits",
                         logits, stream);
    }

    void dump_mtp_align_embeddings(const ::ninfer::Tensor& embeddings, cudaStream_t stream) const {
        dump_tensor_file(dir, label + "_prefill_c" + std::to_string(chunk_begin) +
                             "_mtp_align_embeddings",
                         embeddings, stream);
    }
};

inline void dump_mtp_bridge(const std::string& dir, const std::string& label,
                            const ::ninfer::Tensor& input_hidden, const ::ninfer::Tensor& draft_hidden,
                            const ::ninfer::Tensor& logits, std::int32_t position, cudaStream_t stream) {
    const std::string stem = label + "_mtp_bridge_pos" + std::to_string(position);
    dump_tensor_file(dir, stem + "_input_hidden", input_hidden, stream);
    dump_tensor_file(dir, stem + "_draft_hidden", draft_hidden, stream);
    dump_tensor_file(dir, stem + "_logits", logits, stream);
}

inline void dump_checkpoint_image(const std::string& dir, const std::string& label, std::uint32_t frontier,
                                  const std::byte* image, std::size_t bytes) {
    if (image == nullptr || bytes == 0) { return; }
    // A sequence suffix keeps every capture event: the same frontier is captured by both the
    // text-only source request and the cold visual request, and both images must survive for
    // the GDN-state comparison.
    static std::atomic<std::int32_t> seq{0};
    const std::string name = label + "_checkpoint_frontier" + std::to_string(frontier) +
                             "_seq" + std::to_string(seq.fetch_add(1)) + "_image";
    const std::string path = dir + "/" + name + ".bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(image), static_cast<std::streamsize>(bytes));
    manifest_line(dir, "{\"file\": \"" + name + ".bin\", \"bytes\": " + std::to_string(bytes) +
                     ", \"frontier\": " + std::to_string(frontier) + ", \"label\": " + label + "}");
}

// Fresh sink when NINFER_BRIDGE_DUMP_DIR is set; nullptr otherwise. The caller owns it.
inline std::unique_ptr<DumpSink> make_sink(const std::string& label) {
    const char* dir = std::getenv("NINFER_BRIDGE_DUMP_DIR");
    if (dir == nullptr || *dir == '\0') { return nullptr; }
    return std::make_unique<DumpSink>(dir, label);
}

#else  // NINFER_VISION_BRIDGE_DUMP undefined: compile-out stubs with identical signatures.

struct DumpSink {
    std::string dir;
    std::string label;
    std::int32_t chunk_begin = 0;
    DumpSink(std::string, std::string) {}
    void capture_layer(int, const ::ninfer::Tensor&, cudaStream_t) const {}
    void end(const ::ninfer::Tensor&, cudaStream_t) const {}
    void dump_logits(const ::ninfer::Tensor&, cudaStream_t) const {}
    void dump_mtp_align_embeddings(const ::ninfer::Tensor&, cudaStream_t) const {}
};

inline void dump_mtp_bridge(const std::string&, const std::string&, const ::ninfer::Tensor&,
                            const ::ninfer::Tensor&, const ::ninfer::Tensor&, std::int32_t,
                            cudaStream_t) {}
inline void dump_checkpoint_image(const std::string&, const std::string&, std::uint32_t,
                                  const std::byte*, std::size_t) {}
inline std::unique_ptr<DumpSink> make_sink(const std::string&) { return nullptr; }

#endif  // NINFER_VISION_BRIDGE_DUMP

} // namespace ninfer::targets::qwen3_6::detail::bridge_dump