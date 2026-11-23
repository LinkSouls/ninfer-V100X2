#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/host_worker_pool.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include "core/disk_kv_store.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"
#include "targets/qwen3_6/impl/vision/host.h"

#include <cstdint>
#include <array>
#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData    = qwen3_6::PreparedPromptData;
using RewriteCheckpointKind = qwen3_6::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_6::RewriteCheckpointSpec;

struct ArchivedPrefix;

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return path == ReusePath::RestoreTurnCheckpoint || path == ReusePath::RestoreResponseCheckpoint;
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? ReusePath::RestoreTurnCheckpoint
                                                      : ReusePath::RestoreResponseCheckpoint;
}

enum class RewriteCheckpointAction : std::uint8_t {
    Drop,
    KeepExisting,
    ReclassifyExisting,
    CaptureNew,
    DeferCapture,
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct RequestBasePlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_6::VisionControl> vision_control;
    std::size_t vision_transient_bytes = 0;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint;
    bool allow_prefix_reuse = false;
    bool capture_prefix_checkpoints = false;
};

template <>
struct RequestPlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode::None;
    std::shared_ptr<const NINFER_QWEN36_RUNTIME_NS::ArchivedPrefix> archived_prefix;
    bool prepare_mtp = false;
    bool capture_prefix_checkpoints = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction rewrite_checkpoint_action =
        NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction::Drop;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint_capture;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using RequestPlanImpl     = qwen3_6::detail::RequestPlanImpl<Variant>;
using RequestBasePlanImpl = qwen3_6::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Complete,
};

struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

struct PrefixCheckpoint {
    std::uint32_t frontier = 0;
    // All ranks' exact convolution/recurrent state followed by rank 0's tail hidden.
    std::shared_ptr<std::byte[]> image;
};

inline constexpr std::size_t kPrefixCheckpointBudget = 2ULL * 1024 * 1024 * 1024;

struct SequenceKVBundle {
    PagedKVAllocation text;
    std::optional<PagedKVAllocation> backend;
    // Each peer owns its text KV pool: TP splits heads; PP splits layers. Lockstep pool
    // operations preserve matching page identifiers and block tables across compute devices.
    std::array<std::optional<PagedKVAllocation>, kMaximumExecutionDevices - 1> text_peers;
    // Peer backend allocations use the same lockstep page ownership. In PP2 the peer's
    // symmetric MTP reservation is unused by computation.
    std::array<std::optional<PagedKVAllocation>, kMaximumExecutionDevices - 1> backend_peers;
};

// Immutable host chunks form a persistent prefix tree. Descendants share only complete
// chunks restored from this exact lineage; numerically recomputed prefixes never alias.
struct PrefixKVChunk {
    std::uint32_t first_page = 0;
    std::uint32_t backend_pages = 0;
    std::uint32_t pages = 0;
    // Mutable only for the disk tier: chunks are shared through const handles, and the demotion
    // worker releases the host payload (swap-with-empty) once the durable disk write succeeds.
    // No other path mutates these vectors after archival.
    mutable std::array<std::vector<std::byte>, kMaximumExecutionDevices> text;
    mutable std::array<std::vector<std::byte>, kMaximumExecutionDevices> backend;
    // Content identity for the optional disk tier: rolling (lo,hi) digest over the licensed
    // ledger prefix up to this chunk's page-tail, the program tag, and the page-tail frontier
    // (first_page + pages). Two chunks with equal identities carry equal payload bytes.
    DiskKVIdentity disk_identity;
    // Disk-tier demotion marker: set (by the demotion worker, through the const shared handle)
    // once the chunk's payload is durably in the disk store and the host vectors above have been
    // released. A demoted chunk is restored by reading its identity back from the store.
    mutable std::atomic<bool> demoted = false;
};

// Decoded disk-tier image of one demoted chunk: the per-rank text/backend payload in exactly the
// shape of PrefixKVChunk's host vectors, imported through the same import_page_prefix path.
struct DiskChunkImage {
    std::array<std::vector<std::byte>, kMaximumExecutionDevices> text;
    std::array<std::vector<std::byte>, kMaximumExecutionDevices> backend;
};

struct ArchivedPrefix {
    std::uint32_t frontier = 0;
    std::uint32_t mtp_valid = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity identity;
    std::vector<PrefixCheckpoint> checkpoints;
    std::vector<std::shared_ptr<const PrefixKVChunk>> chunks;
    // Monotonic stamp of the last plan hit on this branch. Zero means "never hit": such a branch
    // is the coldest candidate under LRU eviction.
    std::uint64_t last_used_tick = 0;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    bool greedy_target                   = false;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool retained                 = false;
    RewriteCheckpoint rewrite_checkpoint;
    std::vector<PrefixCheckpoint> prefix_checkpoints;
    bool history_enabled = false;
    std::vector<std::shared_ptr<const PrefixKVChunk>> prefix_chunks;
    std::uint32_t shared_prefix_tokens = 0;
    std::weak_ptr<ArchivedPrefix> archived_snapshot;
    // When set, the committed GDN state lives in this lane's complete owned RAM
    // checkpoint at this frontier; the current GPU GDN slot is not readable yet.
    std::optional<std::uint32_t> deferred_prefix_state;
    TokenSpan ram_query;
    std::uint32_t ram_rank_frontier = 0;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;
    // The executor-owned output session for this lane, bound at admission. Non-null only when the
    // request carries an output constraint; the program reads its per-position mask words during
    // the round and never owns it (the session outlives the lane's program state).
    const qwen3_6::OutputSession* output_session = nullptr;

    // Exact-hit output selection needs only the lane's saved tail hidden. Restore the
    // recurrent state and rebuild MTP continuation when decode first consumes them.
    // These inputs belong to this request/lane, never to shared prefill scratch.
    struct ExactHitContinuation {
        std::array<std::int32_t, 3> bridge_rope;
        std::uint32_t proposal_extent = 0;
    };
    std::optional<ExactHitContinuation> exact_hit;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        runtime::TransientRegion transient;
        std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        bool prepare_mtp                 = false;
        bool capture_prefix_checkpoints  = false;
        ReusePath reuse                  = ReusePath::FullReset;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;
};

// Rank 1's complete runtime mirror: its own arenas, its own shard of the weights, its own halved
// decoder state, and its own RoundState. It owns no bookkeeping -- lanes, page accounting,
// sampling and the pinned host round buffers all live once, on rank 0.
struct PeerRuntime {
    PeerRuntime(DeviceContext& peer_device, const LoadedModelData& peer_model,
                const SequencePlanImpl& plan, int rank);

    PeerRuntime(const PeerRuntime&)            = delete;
    PeerRuntime& operator=(const PeerRuntime&) = delete;

    const int rank;
    DeviceContext& device;
    const LoadedModelData& model;
    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::optional<DFlashPersistentState> dflash;
    // Rank 1's own GDN replay records: the speculative verify round records this device's own
    // head/channel shard and folds it here, so the two devices commit the same accepted prefix
    // from records neither ever exchanges.
    std::optional<GdnReplayRecords> replay_records;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    // Rank 1's OWN penalty counters. `ops::SamplingConfig::token_counts` is a raw device pointer,
    // and the MTP round's acceptance runs on both devices (see the replicated-accept note in
    // mtp_impl.h), so rank 1 must never be handed rank 0's. It is not a cache: rank 0 remains the
    // source of truth (ProgramImplCore::install_sampling zeroes both, and the one increment that
    // happens on rank 0 alone -- prefill's bonus token -- is copied across before the first decode
    // round), and thereafter both are advanced by the same Op over bit-identical inputs.
    Tensor token_counts;
    std::optional<DecodeGraphPeerBridge> graph_bridge;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    // Rank 1's pinned mirror of rank 0's MTP mask-word staging. The masks depend only on the
    // mirrored drafts, so the bytes are identical; published with the MTP ingress.
    std::optional<PinnedHostBuffer> mtp_mask_host;
    std::uint32_t* mtp_mask_words = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, std::span<const LoadedModelData* const> peer_models,
                    const SequencePlanImpl& plan, ExecutionContext& execution);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPromptData& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPromptData&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient,
                                                                const qwen3_6::OutputSession* output_session =
                                                                    nullptr);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane);
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] const SpeculativeStats& speculative_stats_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeBackend speculative_backend() const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    [[nodiscard]] DiskTierStats disk_tier_stats() const noexcept;

    void reset_memory_peaks() noexcept;

    const LoadedModelData& model;
    mutable qwen3_6::HostVisionCache host_vision_cache;
    ExecutionContext& execution;
    DeviceContext& device;
    const int device_count;
    const int pp;
    const std::uint32_t pipeline_chunks;
    std::optional<PinnedHostBuffer> pipeline_upload;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefix_cache_tokens;
    const std::uint32_t prefill_chunk;
    // Effective prefill sub-step bounding one prefill execution unit (<= prefill_chunk).
    const std::uint32_t prefill_step;
    const std::uint32_t draft_window;
    const RamKvOptions ram_kv;
    const bool volta_long_attention;
    const SpeculativeBackend speculative_backend_;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const KvCacheStorage kv_storage;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t gdn_state_bytes;
    const std::size_t graph_allowance_bytes;
    // Measured graph residency per device (index = rank). At tp1 only [0] is populated. At tp2 the
    // single cross-device graph materializes driver state on BOTH devices, and each is checked
    // against the SAME per-device allowance -- graph_allowance_bytes is a per-device budget, like
    // every other field in device_reservation_bytes.
    std::array<std::size_t, kMaximumExecutionDevices> graph_observed_bytes{0, 0};
    // Node count of ONE captured decode graph (the first profile of the captured family). At tp2
    // one graph holds both devices' nodes, so this is the direct measurement of whether the peer's
    // half of the schedule was captured rather than left out.
    std::size_t graph_node_count = 0;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    // YaRN rotary state, resolved once at construction.
    //
    // `rope_frequency_storage[rank]` is a dedicated 32-float device allocation on rank `rank`'s OWN
    // device, made once here and never touched again: CUDA Graph capture bakes the pointer into the
    // replayed rope launch node, so it must outlive every replay, and at device_count 2 each rank ropes its
    // own head-local q/k on its own device, where a pointer into the other rank's allocation is not
    // addressable. It is deliberately NOT part of the planned persistent arena: under
    // `RopeMode::Native` nothing is allocated at all, which is what keeps a native plan and its
    // memory summary byte-identical to the pre-YaRN engine.
    //
    // `rope_frequency[rank]` is the descriptor every text rope call site reads (through
    // `ExecutionCore::rope_frequency` -> `TextContext`); a null `inv_frequency` IS the native path.
    std::array<DeviceBuffer, kMaximumExecutionDevices> rope_frequency_storage;
    std::array<ops::RopeFrequencyOverride, kMaximumExecutionDevices> rope_frequency{};
    const RopeMode rope_mode;
    const std::uint32_t effective_max_context;
    const double yarn_mscale;
    std::vector<std::unique_ptr<PeerRuntime>> peers;
    std::optional<ops::PeerEvents> peer_events;
    std::array<schedule::PeerCore, kMaximumExecutionDevices - 1> peer_cores{};
    [[nodiscard]] std::span<const schedule::PeerCore> peer_views() const noexcept {
        return std::span(peer_cores).first(pp == 1 ? peers.size() : 0);
    }
    [[nodiscard]] const schedule::PeerCore* pipeline_view() const noexcept {
        return pp == 2 ? &peer_cores[0] : nullptr;
    }
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<DFlashPersistentState> dflash;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden_store;
    Tensor rewrite_checkpoint_hidden_store;

    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;
    std::vector<std::shared_ptr<ArchivedPrefix>> prefix_history;
    // Monotonic clock for branch recency. Zero is reserved for "never hit", so the counter starts
    // at 0 and every refresh takes the next value.
    std::uint64_t prefix_history_tick = 0;
    // Refreshes the recency stamp of an archived branch itself (not of a plan snapshot copy).
    void touch_archived_prefix(ArchivedPrefix& archive) {
        archive.last_used_tick = ++prefix_history_tick;
        // A host-tier hit on a branch that carries demoted chunks also refreshes those chunks'
        // disk recency stamps, so the branch sits at the back of the LRU in BOTH tiers; a host
        // hit must not leave the branch's disk pages as the next eviction victims.
        if (disk_kv_store != nullptr) {
            for (const auto& chunk : archive.chunks) {
                if (chunk->demoted.load(std::memory_order_relaxed)) {
                    disk_kv_store->touch(chunk->disk_identity);
                }
            }
        }
    }
    void archive_prefix(SequenceState& sequence);
    void restore_archived_prefix(SequenceState& sequence, const ArchivedPrefix& archive,
                                 const RequestPlanImpl& plan,
                                 const std::vector<DiskChunkImage>* disk_images = nullptr);
    void trim_prefix_history();

    // Optional disk tier. `disk_kv_tag` is the program-level content-identity class (model/format/
    // width) folded into every archived chunk's disk identity; `disk_kv_match_misses` is the
    // observable count of disk-tier prefix matches that found no recoverable frontier.
    //
    // Digest production/consumption coverage (all four archive paths):
    //   PRODUCTION  archive_prefix() rolls the ledger digest and stamps chunk->disk_identity for
    //               every chunk. A chunk carries every rank's payload (text[rank]/backend[rank])
    //               under that ONE identity, so the history-branch and peer-lane archive paths are
    //               both covered here; the ledger (and hence the digest) is rank-independent.
    //   CONSUMPTION disk_recoverable_frontier() checks each chunk's identity against the store for
    //               a prefix-reuse match; the restore path (restore_archived_prefix, disk tier)
    //               reads a demoted chunk back by the same chunk->disk_identity.
    std::uint64_t disk_kv_tag = 0;
    // GDN endpoint-state payload class: disk_kv_tag XOR kDiskKvGdnStateTagXor. The branch's
    // endpoint checkpoint image (linear-attention recurrent state + tail hidden) is written into
    // the same content-addressed store under this distinct tag, so it never aliases a KV chunk
    // payload. It is a pure function of the program class, so it survives a process restart.
    std::uint64_t disk_kv_gdn_tag = 0;
    std::uint64_t disk_kv_match_misses = 0;
    // Farthest page-tail frontier (in pages) at which every chunk from the start of `archive` up
    // to and including that frontier is present in `store` under the chunk's content identity. A
    // null `store` (disk tier off) yields 0, and a requested match that recovers no chunk counts
    // as a miss.
    std::uint32_t disk_recoverable_frontier(const ArchivedPrefix& archive,
                                             const DiskKVStore* store);
    // Opened in the constructor from the plan's DiskKvTierConfig when enabled; null when the tier
    // is off (the default) or the open failed. Bound to the KV storage format at open.
    std::unique_ptr<DiskKVStore> disk_kv_store;
    // Separate content-addressed store for the GDN endpoint state images, opened under
    // `dir / "gdn"` with a slot sized to prefix_checkpoint_bytes(). It is distinct from
    // disk_kv_store (whose slots are sized to a single KV chunk) so the KV store's geometry is
    // unchanged and pre-existing KV directories stay openable. Null when the tier is off or the
    // open failed; a missing GDN store simply disables cold restore (full recompute fallback).
    std::unique_ptr<DiskKVStore> disk_gdn_store;
    // Per-page payload bytes of the text and backend (MTP) pools, fixed at construction; they size
    // the disk slots and the restore read buffers for a chunk's exact page count.
    std::size_t disk_kv_text_page_bytes = 0;
    std::size_t disk_kv_backend_page_bytes = 0;
    // Bounded FIFO of asynchronous demotion writes. The pool is declared LAST so it is destroyed
    // FIRST: its join drains in-flight tasks while the store and bookkeeping below are alive.
    std::mutex disk_demotion_mutex;
    // Branches whose demotion write is in flight, keyed by the canonical archive pointer in
    // prefix_history. In-flight chunks stay host-resident (their bytes are released only on write
    // success) and are excluded from further demotion until the write completes.
    std::unordered_set<const ArchivedPrefix*> disk_demotion_inflight;
    // Branches whose last demotion write FAILED (persistent I/O error or oversized payload). They
    // stay host-resident and are excluded from victim selection, so a broken store cannot turn
    // the trim loop into a hot resubmission cycle; the budget overrun is tolerated.
    std::unordered_set<const ArchivedPrefix*> disk_demotion_failed;
    // Chunks whose host vectors a restore is importing right now. The demotion worker releases a
    // chunk's host bytes only under disk_demotion_mutex and only for chunks NOT in this set, so a
    // host-resident chunk can never be emptied out from under an in-flight restore.
    std::unordered_set<const PrefixKVChunk*> disk_restore_inflight;
    std::vector<std::future<void>> pending_disk_demotions;
    std::unique_ptr<HostWorkerPool> disk_demotion_pool;
    // Observable count of restore attempts that fell back to full recompute because a demoted
    // chunk was unavailable (missing, torn, CRC/identity mismatch, or write not yet complete).
    std::uint64_t disk_kv_restore_failures = 0;
    // Submits the branch's exclusive, not-yet-demotion chunks to the async writer. The branch is
    // marked in-flight under disk_demotion_mutex; the worker releases its host bytes only after
    // every chunk write succeeded.
    void submit_disk_demotion(ArchivedPrefix& branch,
                              std::span<const std::shared_ptr<const PrefixKVChunk>> chunks);
    // Collects completed demotion futures (memory hygiene); the in-flight set is cleared by the
    // worker itself under the same mutex.
    void reap_disk_demotions();
    // Reads every demoted chunk of `archive` covered by `base` pages back from the disk store into
    // `out` (aligned with archive.chunks; host chunks get an empty image). Returns false when any
    // read fails; the failed slot is dropped by the store while every other identity is kept.
    bool load_disk_restore_images(const ArchivedPrefix& archive, std::uint32_t base,
                                  std::vector<DiskChunkImage>& out);
    // Cross-restart cold restore. Re-derives the KV chunk identity chain from a re-issued token
    // ledger (the same chunking and rolling digest as archive_prefix) and returns the farthest
    // page-tail frontier (in pages) at which every chunk is present in `store` under the
    // re-derived identity. A null `store` (disk tier off) or a ledger with no recoverable chunk
    // yields 0 and counts as a miss. `tag` is the program-level KV content-identity class (the
    // caller passes disk_kv_tag).
    std::uint32_t disk_cold_recoverable_frontier(const TokenId* ledger, std::size_t n_tokens,
                                                 std::uint64_t tag, const DiskKVStore* store);
    // The reusable (prompt) boundary of an archived branch: the frontier of its largest inner
    // checkpoint (strictly below the branch's execution frontier). The disk-tier identity chain
    // and the GDN endpoint are keyed to this boundary -- a pure function of the licensed prompt
    // ledger -- not to the execution frontier, which includes generated tokens and is therefore
    // not re-derivable from a re-issued prompt. Returns 0 when the branch has no inner
    // checkpoint (it is then not cold-restorable; the KV chunks fall back to full-ledger
    // identities for the in-memory path).
    std::uint32_t disk_reuse_boundary(const ArchivedPrefix& archive) const;
    // The re-derivable content identity of a branch's GDN endpoint state image (the checkpoint at
    // `n_tokens`): the rolling digest of ledger[0..n_tokens) under the GDN tag, with the page-tail
    // frontier. `n_tokens` is the reusable boundary (disk_reuse_boundary). A pure function of the
    // token ledger; no branch structure is persisted, so the identity is the same before and after
    // a process restart.
    DiskKVIdentity disk_gdn_endpoint_identity(const TokenId* ledger, std::size_t n_tokens) const;
    // Reads the GDN endpoint state image recorded under `identity` (prefix_checkpoint_bytes()
    // bytes) into `out`. Returns false when the read fails; the store drops only the failed slot.
    bool load_disk_gdn_endpoint(const DiskKVIdentity& identity, std::vector<std::byte>& out);
    // Cross-restart cold restore spec: a synthetic ArchivedPrefix whose chunks are re-derived from
    // the re-issued token ledger (the same chunking/digest as archive_prefix, all marked demoted so
    // the restore path reads them from the disk store) and whose single checkpoint (at `n_tokens`)
    // carries no GDN image yet (the execution path loads it from the GDN store). Returns nullptr
    // when the KV chain is not fully recoverable to the full-prompt frontier or the GDN endpoint
    // is absent, so the caller falls back to a full recompute.
    std::shared_ptr<ArchivedPrefix>
    build_disk_cold_restore_spec(const PreparedPromptData& prompt, const TokenId* ledger,
                                 std::size_t n_tokens);

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;

    PinnedHostBuffer round_host;
    TokenId* host_tokens = nullptr;
    // Debug-only capture of the round's full-vocabulary logits, raw BF16 bits. OFF by default and
    // completely inert until enable_logits_capture(true) allocates the host buffer: with it off,
    // copy_round_logits() is a predictable branch and nothing is allocated or copied, so no
    // production path (tp1 or tp2) is changed. With it on, the buffer is overwritten every time a
    // token is sampled from io.logits at prefill finalization, for both tp1 and tp2 (io is rank
    // 0's window; logits_tp2 gathers into it the same way the tp1 leaf writes it), so no
    // device_count-specific code is needed. Prefill only -- the decode path is CUDA-graph-capturable and an
    // unconditional memcpy inside a captured region is rejected at capture time; a probe that
    // requests exactly one output token completes on prefill's own sampled token and never runs a
    // decode round, which is what the parity harness (tools/tp2/parity.cpp) uses. Valid only
    // immediately after Engine::generate()/wait() returns to the calling thread for a single-lane
    // engine -- concurrent lanes would overwrite it out of order.
    std::vector<std::uint16_t> logits_capture;
    bool logits_capture_enabled = false;
    // Output-constraint mask-word staging: one ceil(kTokenDomain/32) uint32 row per compact batch
    // row for the ordinary round (indexed by row, matching the ingress), one
    // (kMtpDecodeMaximumWidth * ceil(kTokenDomain/32)) block per lane for the MTP verify round.
    // Pinned so the CUDA-graph decode body can upload it, in a buffer that outlives every graph.
    // Filled on the host from the lane's output session just before the round; unconstrained
    // rows are all-ones (a bit-identical no-op for the mask op), so unconstrained requests stay
    // bit-identical and the decode body can run the mask op unconditionally inside the graph.
    std::optional<PinnedHostBuffer> mask_words_host;
    std::uint32_t* ordinary_mask_words = nullptr;
    std::uint32_t* mtp_mask_words      = nullptr;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    // Rank 1's own pinned ordinary ingress: rank 0's record with every row's
    // `sampling[row].token_counts` nulled. Rank 1 mirrors the ordinary round for its half of the
    // weights and never samples (the output head is vocabulary-split and sampling belongs to rank
    // 0), so the sampling configs in its frame are inert -- but they must not carry a rank-0
    // DEVICE address into rank 1's frame: dereferencing one from rank 1 is a silent cross-device
    // fault. A separate pinned buffer rather than a patched copy at issue time, for the same
    // reason as `mtp_peer_host_ingress`: the upload is inside the captured decode graph, which
    // re-reads this exact host address at every replay.

    bool peer_egress_check_enabled     = false;
    std::uint64_t peer_egress_rounds     = 0;
    std::uint64_t peer_egress_mismatches = 0;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_6::MtpDecodeEgress* mtp_host_egress   = nullptr;
    // Rank 1's own pinned MTP ingress: byte-for-byte rank 0's record except that each row's
    // `sampling[row].token_counts` names rank 1's counter lane. It has to be a separate pinned
    // buffer rather than a patched copy made at issue time, because the ingress upload is inside
    // the captured decode graph and the graph re-reads this exact host address at every replay.

    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_6::DFlashDecodeEgress* dflash_host_egress   = nullptr;


    std::size_t workspace_logical_peak_bytes = 0;

    // See logits_capture's comment. Read-only; the caller owns thread-safety (single-lane, read
    // after wait() returns). Empty while capture is disabled.
    [[nodiscard]] std::span<const std::uint16_t> last_round_logits_bf16() const noexcept {
        return logits_capture;
    }

    // Turns the debug capture above on or off. Enabling sizes the host buffer to the round's
    // logits row count; disabling releases it. Must not be called while a round is in flight (the
    // executor holds its execution mutex across this call).
    void enable_logits_capture(bool enabled);

    // Debug-only, OFF by default: after each MTP decode round at device_count == 2, read rank 1's MTP egress
    // back and compare it field for field with rank 0's. The two ranks run the acceptance Op over
    // bit-identical inputs, so their egress records are argued to agree; enabling this turns that
    // induction into a measurement. Costs one ~1 KiB device-to-host copy plus a host compare per
    // round while enabled, and nothing at all while off. Counters are cumulative over the
    // Program's lifetime; a mismatch is counted (per row, per field) rather than thrown, so a test
    // can read the totals after a clean run.
    void enable_peer_egress_check(bool enabled) noexcept;
    [[nodiscard]] std::uint64_t peer_egress_check_rounds() const noexcept {
        return peer_egress_rounds;
    }
    [[nodiscard]] std::uint64_t peer_egress_check_mismatches() const noexcept {
        return peer_egress_mismatches;
    }

private:
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    void prepare_graphs();
    // Copies rank 0's counter lane onto rank 1. Called once, after prefill's bonus token is
    // sampled -- the only counter increment that happens on rank 0 and not on rank 1.
    [[nodiscard]] static Tensor token_counts_lane(const Tensor& storage, std::uint32_t lane);
    void publish_peer_token_counts(const SequenceState& sequence);
    // Mirrors `mtp_host_ingress` into `mtp_peer_host_ingress`, swapping every row's counter
    // pointer for rank 1's. No-op at tp1 or without MTP.
    void publish_peer_mtp_ingress(std::span<const std::uint32_t> lanes);
    void publish_peer_dflash_ingress(std::span<const std::uint32_t> lanes);
    // Mirrors `ordinary_host_ingress` into `ordinary_peer_host_ingress` with every row's counter
    // pointer nulled. No-op at tp1 or without an ordinary frame.
    void publish_peer_ordinary_ingress();
    // Debug-only: reads rank 1's MTP egress back and compares it, field for field, with rank 0's.
    // No-op unless the check is enabled and a peer exists.
    void check_peer_mtp_egress(std::size_t rows);
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    static void set_peer_i32(Tensor& tensor, std::int32_t value, DeviceContext& rank_device);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void copy_round_logits();
    void resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t accepted_tokens, bool terminal);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill(SequenceState& sequence,
                                                             RequestControl& request);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets);
    void reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                             std::uint32_t backend_pages);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    void refresh_ram_kv(SequenceState& sequence, std::uint32_t frontier);
    [[nodiscard]] std::size_t prefix_checkpoint_bytes() const;
    void save_prefix_checkpoint(SequenceState& sequence, std::uint32_t frontier, const Tensor& hidden);
    PrefixCheckpoint capture_prefix_checkpoint(SequenceState& sequence, std::uint32_t frontier,
                                                const Tensor& hidden,
                                                std::shared_ptr<std::byte[]> image = {});
    void restore_prefix_checkpoint(SequenceState& sequence, std::uint32_t frontier,
                                   bool restore_linear_state = true);
    void finish_exact_hit(SequenceState& sequence, RequestControl& request, bool decode);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    [[nodiscard]] qwen3_6::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_6::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_6::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] std::array<qwen3_6::PagedKVCacheView, kMaximumExecutionDevices - 1>
    mtp_kv_views_peer(const SequenceState& sequence) const;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
