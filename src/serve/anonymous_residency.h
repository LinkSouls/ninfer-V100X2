#pragma once

// Keeps the serving process's large anonymous arenas resident under host memory
// pressure by mlocking them after warmup.
//
// Only anonymous mappings are eligible: a named mapping (weight mmap, driver shared
// memory) is either backed by a file the kernel can re-fault cheaply or owned by a
// component that manages its own residency, and locking one would inflate
// unreclaimable memory for no latency benefit. The eligible size band and the total
// budget come from this machine's measured layout, not from a generic constant; see
// the defaults below and `docs/changes/p2-serve-process-resilience.md`.

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// One `/proc/self/smaps` mapping reduced to the facts selection needs.
struct AnonymousRegion {
    std::uintptr_t start = 0;
    std::uintptr_t end   = 0;
    std::uint64_t size_bytes     = 0; // virtual size: what RLIMIT_MEMLOCK charges
    std::uint64_t resident_bytes = 0; // Rss: what locking actually protects
    std::uint64_t locked_bytes   = 0; // Locked
    std::uint64_t inode          = 0;
    std::string permissions; // "rw-p", "---p", ...
    std::string name;        // empty for an anonymous mapping

    [[nodiscard]] bool anonymous() const noexcept { return name.empty() && inode == 0; }
    [[nodiscard]] bool readable_writable() const noexcept {
        return permissions.size() >= 2 && permissions[0] == 'r' && permissions[1] == 'w';
    }
};

// Measured on this host (2x V100, 31 GiB RAM, one production service resident):
//
//   eligible anonymous rw mappings    108 regions / 13.99 GiB virtual, 12.54 GiB RSS
//   of these, size >= 64 MiB           28 regions / 12.61 GiB virtual, 11.40 GiB RSS
//   of these, size >= 512 MiB          11 regions /  8.81 GiB virtual,  7.61 GiB RSS
//   largest eligible mapping          1452 MiB
//   RLIMIT_MEMLOCK                    3.90 GiB (no CAP_IPC_LOCK in this deployment)
//
// The 64 MiB floor sits in the gap between the ~60 MiB staging cluster (15 regions,
// `nr` flags, transient) and the 66 MiB arenas, so it excludes that noise without
// losing an arena. The 2 GiB ceiling admits every real arena while refusing a single
// mapping large enough to consume the whole locked budget. The budget is
// min(3/4 x RLIMIT_MEMLOCK, 4 GiB): the first factor keeps a quarter of the hard
// limit in reserve for other locked allocations, the second stops an unlimited
// (root/CAP_IPC_LOCK) deployment from locking every anonymous page and starving the
// co-resident service to which this host's memory is already committed.
inline constexpr std::uint64_t kAnonymousLockMinRegionBytes = 64ULL << 20;
inline constexpr std::uint64_t kAnonymousLockMaxRegionBytes = 2ULL << 30;
inline constexpr std::uint64_t kAnonymousLockBudgetCeilingBytes = 4ULL << 30;

struct AnonymousPageLockPolicy {
    std::uint64_t min_region_bytes = kAnonymousLockMinRegionBytes;
    std::uint64_t max_region_bytes = kAnonymousLockMaxRegionBytes;
    // Upper bound on total locked *virtual* bytes; the caller supplies the effective
    // budget after consulting RLIMIT_MEMLOCK.
    std::uint64_t budget_bytes = kAnonymousLockBudgetCeilingBytes;
    // A mapping that was reserved but never touched would consume budget while
    // protecting nothing, so it is only eligible once this much of it is resident.
    std::uint64_t min_resident_bytes = kAnonymousLockMinRegionBytes;
};

struct AnonymousPageLockReport {
    // What was considered, before the budget.
    std::size_t scanned_regions        = 0;
    std::size_t eligible_regions       = 0;
    std::uint64_t eligible_bytes       = 0;
    std::uint64_t eligible_resident    = 0;
    // What the selection and the mlock() calls actually achieved.
    std::size_t attempted_regions      = 0;
    std::size_t locked_regions         = 0;
    std::uint64_t locked_bytes         = 0;
    std::size_t already_locked_regions = 0;
    std::size_t skipped_regions        = 0; // eligible but over budget
    std::size_t failed_regions         = 0;
    std::uint64_t budget_bytes         = 0;
    std::uint64_t memlock_limit_bytes  = 0;
    // One entry per failed region, and the scan-level reason when nothing ran.
    std::vector<std::string> failures;
    std::string scan_error;

    [[nodiscard]] bool attempted() const noexcept { return attempted_regions != 0; }
};

// Parses an smaps body. Malformed trailing data is ignored rather than thrown: the
// caller is reporting on a live process, not validating a file.
[[nodiscard]] std::vector<AnonymousRegion> parse_smaps(std::istream& input);

// Reads and parses /proc/self/smaps. Returns false and fills `error` on failure.
[[nodiscard]] bool read_self_smaps(std::vector<AnonymousRegion>& regions, std::string& error);

// The region subset that passes the anonymity, permission, size and residency
// filters, in the given policy band. Exposed for tests.
[[nodiscard]] std::vector<AnonymousRegion> select_eligible_regions(
    const std::vector<AnonymousRegion>& regions, const AnonymousPageLockPolicy& policy);

// Soft RLIMIT_MEMLOCK of this process; zero when the resource is unlimited.
[[nodiscard]] std::uint64_t memlock_limit_bytes();

// The default budget for this process: min(3/4 x RLIMIT_MEMLOCK, policy ceiling),
// floored at the per-region minimum so a tiny limit still attempts one arena.
[[nodiscard]] std::uint64_t effective_lock_budget(const AnonymousPageLockPolicy& policy);

// Scans, selects, and mlock()s largest-first until the budget is spent. Never
// throws and never fails the caller: every problem lands in the report.
[[nodiscard]] AnonymousPageLockReport lock_anonymous_pages(const AnonymousPageLockPolicy& policy);

// One console-sized line summarizing `report`, including the skip/failure reason.
[[nodiscard]] std::string format_anonymous_page_lock_report(const AnonymousPageLockReport& report);

} // namespace ninfer::serve
