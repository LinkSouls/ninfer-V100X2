#pragma once

// Product-side adapter between HTTP protocol requests and the public NInfer
// engine. It owns one Engine and keeps protocol concerns (aliases, usage,
// streaming callbacks, and tool-call parsing) outside the target package.

#include "ninfer/engine.h"
#include "serve/anonymous_residency.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::serve {

struct RequestLifetime;
struct RequestCapacity;

struct GenerationMetrics {
    double prepare_seconds = 0.0;
    double ttft_seconds    = 0.0;
    double vision_seconds  = 0.0;
    double prefill_seconds = 0.0;
    double decode_seconds  = 0.0;
    double total_seconds   = 0.0;

    SpeculativeBackend speculative_backend    = SpeculativeBackend::None;
    std::uint32_t speculative_draft_window    = 0;
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    std::vector<std::uint64_t> speculative_accepted_per_position;
    std::uint32_t prefix_cache_hit_tokens     = 0;
    ninfer::PrefixReusePath prefix_reuse_path = ninfer::PrefixReusePath::FullReset;
};

struct GenerationOutcome {
    std::string text;
    std::string reasoning;
    std::vector<ToolCall> tool_calls;
    int prompt_tokens                  = 0;
    int completion_tokens              = 0;
    int reasoning_tokens               = 0;
    std::size_t streamed_content_bytes = 0;
    ninfer::FinishReason finish_reason = ninfer::FinishReason::OutputLimit;
    GenerationMetrics metrics;

    [[nodiscard]] CompletionUsage usage() const {
        return {prompt_tokens, completion_tokens,
                static_cast<int>(metrics.prefix_cache_hit_tokens)};
    }
};

struct StreamSink {
    std::function<void(const std::string& delta_text)> on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    std::function<bool()> is_cancelled;
};

// Translate Engine request failures into the shared protocol-neutral HTTP error contract.
ApiError request_error_to_api_error(const ninfer::RequestError& exception);

// Preparation ends by synchronously submitting the owning prompt to the Engine FIFO. The returned
// request keeps its ingress/response lifetime reservation until the HTTP response is released and
// is consumed exactly once by run().
struct PreparedRequest {
    // Destroy the generation handle before releasing the lifecycle reservation.
    std::shared_ptr<RequestLifetime> lifetime;
    ninfer::GenerationHandle generation;
    ninfer::ResolvedSamplingParameters sampling;
    double prepare_seconds     = 0.0;
    double acquisition_seconds = 0.0;
    PromptPreparationStats preparation;
    int prompt_tokens                      = 0;
    bool include_usage                     = false;
    bool tool_capable                      = false;
    std::size_t tool_name_max_length       = 64;
    bool enable_thinking                   = true;
    bool preserve_thinking                 = false;
    bool preserve_thinking_semantic_change = false;
    // Media parts rewritten to placeholders because the server runs without
    // vision (degrade_media_history). Zero unless the degradation fired; the
    // request log records it only when non-zero.
    std::size_t media_degraded_count = 0;
};

struct ModelResidency {
    std::string state = "gpu"; // gpu, offloading, ram, loading
    std::uint64_t retained_host_bytes = 0;
    std::string phase;
    std::uint64_t done = 0;
    std::uint64_t total = 0;
    std::size_t active_requests = 0;
};

// What process-level hardening is actually in force, as opposed to what the command
// line asked for. Every field is observed at the point the effect exists: the stderr
// mode is read from the installed replacement, the lock figures from the warmup-time
// attempt, so a failed or skipped step is never reported as enabled.
struct ProcessHardeningSummary {
    // Command-line request (`--no-process-hardening` clears both).
    bool requested = true;
    // stderr is a non-blocking pipe with the line-committing sink installed.
    bool nonblocking_stderr = false;
    // Why it is not installed, when the request was made and the install failed.
    std::string stderr_fallback_reason;
    std::uint64_t stderr_dropped_lines  = 0;
    std::uint64_t stderr_forwarded_bytes = 0;
    std::uint32_t stderr_pipe_bytes      = 0;
    // Anonymous-page locking, executed once after warmup().
    bool anonymous_lock_attempted      = false;
    std::size_t anonymous_locked_regions = 0;
    std::uint64_t anonymous_locked_bytes = 0;
    std::uint64_t anonymous_lock_budget_bytes = 0;
    std::size_t anonymous_skipped_regions     = 0;
    std::size_t anonymous_failed_regions      = 0;
    std::string anonymous_lock_note; // first failure or scan error, empty on success
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options, LoadProgress load_progress = {});

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }

    [[nodiscard]] ninfer::LoadSummary load_summary() const { return load_; }

    [[nodiscard]] ninfer::MemorySummary memory_summary() const;

    [[nodiscard]] ninfer::DiskTierStats disk_tier_stats() const;

    [[nodiscard]] ninfer::RuntimeStats runtime_stats() const;

    [[nodiscard]] ninfer::MediaCacheSummary media_cache_summary() const;

    [[nodiscard]] ninfer::ModelSamplingDefaults sampling_defaults() const {
        return sampling_defaults_;
    }

    [[nodiscard]] ModelResidency model_residency() const;
    void offload_to_memory();
    void load_from_memory();

    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;

    // Consumes prepared.generation. A PreparedRequest is single-use.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled = {});

    void warmup();

    // Locks the eligible anonymous mappings of this process. Must be called after
    // warmup(), when the host arenas are allocated and touched. Never throws; the
    // result is kept for process_hardening_summary().
    void lock_anonymous_pages();

    // Observed process hardening state. The single place that answers "what is
    // actually in force", so logging never has to infer it from ServeOptions alone.
    [[nodiscard]] ProcessHardeningSummary process_hardening_summary() const;

private:
    [[nodiscard]] std::shared_ptr<RequestLifetime> acquire_request_lifetime() const;

    ServeOptions options_;
    std::unique_ptr<ninfer::Engine> engine_;
    ninfer::PromptCapabilities prompt_capabilities_;
    std::shared_ptr<RequestCapacity> request_capacity_;
    ninfer::EngineOptions engine_options_;
    ninfer::LoadSummary load_;
    ninfer::ModelSamplingDefaults sampling_defaults_;
    ninfer::ModelMemory host_model_;
    ModelResidency residency_;
    ninfer::RuntimeStats completed_stats_;
    // The lock report is written once by the post-warmup pass; the mutex is mutable so
    // the read-side accessor stays const.
    mutable std::mutex anonymous_lock_mutex_;
    AnonymousPageLockReport anonymous_lock_;
    void update_load_progress(std::string_view phase, std::uint64_t done, std::uint64_t total);
};

} // namespace ninfer::serve
