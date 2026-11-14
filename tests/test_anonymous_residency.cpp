// Host-only: /proc/self/smaps selection rules for the post-warmup anonymous-page
// lock. The lock itself needs a real process image, so these cases drive the pure
// parser and selector with synthetic smaps bodies; the lock path is covered by the
// serving startup record.

#include "serve/anonymous_residency.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;

int failures = 0;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// One mapping, formatted the way smaps prints it, including a few fields the
// selector must ignore.
std::string mapping(const char* range, const char* perms, const char* inode, const char* path,
                    std::uint64_t size_kib, std::uint64_t rss_kib, std::uint64_t locked_kib) {
    std::ostringstream out;
    out << range << ' ' << perms << " 00000000 00:00 " << inode << (path[0] ? " " : "") << path
        << '\n';
    out << "Size: " << size_kib << " kB\n";
    out << "KernelPageSize: 4 kB\n";
    out << "MMUPageSize: 4 kB\n";
    out << "Rss: " << rss_kib << " kB\n";
    out << "Pss: " << rss_kib << " kB\n";
    out << "Shared_Clean: 0 kB\n";
    out << "Anonymous: " << rss_kib << " kB\n";
    out << "Locked: " << locked_kib << " kB\n";
    out << "VmFlags: rd wr mr mw me ac sd\n";
    return out.str();
}

} // namespace

int main() {
    // A body that mixes every category the selector distinguishes.
    std::string body;
    body += mapping("00005600-00005700", "rw-p", "0", "", 4096, 4096, 0);             // heap
    body += mapping("00007f00-00007f10", "r--p", "0", "", 4096, 4096, 0);             // read-only, out
    body += mapping("00007f10-00007f20", "rw-p", "0", "", 4096, 4096, 0);             // too small
    body += mapping("00007f20-00008000", "rw-p", "0", "", 200 * 1024, 200 * 1024, 0); // arena
    body += mapping("00008000-00009000", "rw-p", "0", "", 8 * 1024 * 1024, 100, 0);   // untouched
    body += mapping("00009000-0000a000", "rw-p", "1234", "/weights.bin", 400 * 1024,
                    400 * 1024, 0); // named mmap
    body += mapping("0000a000-0000b000", "rw-s", "7", "/dev/shm/kv", 400 * 1024, 400 * 1024,
                    0); // shared named mapping
    body += mapping("0000b000-0000c000", "rw-p", "0", "", 3ULL * 1024 * 1024, 3ULL * 1024 * 1024,
                    0); // over the ceiling

    std::istringstream input(body);
    const std::vector<AnonymousRegion> regions = parse_smaps(input);
    failures += check(regions.size() == 8, "parser lost or merged mappings");

    int anonymous = 0;
    for (const AnonymousRegion& region : regions) {
        if (region.anonymous()) { ++anonymous; }
    }
    failures += check(anonymous == 6, "an anonymity decision was wrong");

    // Field parsing: the first mapping is a 4 MiB heap with everything resident.
    failures += check(regions.front().size_bytes == 4ULL * 1024 * 1024,
                      "Size did not parse into bytes");
    failures += check(regions.front().resident_bytes == 4ULL * 1024 * 1024,
                      "Rss did not parse into bytes");
    failures += check(regions.front().permissions == "rw-p", "permissions did not parse");
    failures += check(regions[5].name == "/weights.bin", "named mapping path did not parse");
    failures += check(regions[5].inode == 1234, "named mapping inode did not parse");

    const AnonymousPageLockPolicy policy;
    const std::vector<AnonymousRegion> eligible = select_eligible_regions(regions, policy);
    failures += check(eligible.size() == 1, "selector did not pick exactly the one real arena");
    if (!eligible.empty()) {
        failures += check(eligible.front().start == 0x7f20, "wrong resident arena selected");
    }

    // Named mappings are excluded even when they are the right size and resident:
    // locking a weight mmap would hold unreclaimable memory for no latency benefit.
    for (const AnonymousRegion& region : eligible) {
        failures += check(region.anonymous(), "a named mapping passed the anonymity filter");
    }

    // An already-locked region is not re-locked; a partially locked one is charged
    // only for the remainder by the lock pass, which parses these fields.
    std::istringstream locked_input(
        mapping("0000c000-0000d000", "rw-p", "0", "", 512 * 1024, 512 * 1024, 512 * 1024));
    const std::vector<AnonymousRegion> locked_regions = parse_smaps(locked_input);
    failures += check(locked_regions.size() == 1 &&
                          locked_regions.front().locked_bytes == 512ULL * 1024 * 1024,
                      "Locked did not parse into bytes");

    // Malformed trailing data must not fabricate mappings or throw: the caller is
    // reporting on a live process, not validating a file.
    std::istringstream ragged("not a mapping\nSize: nonsense kB\n0000e000-0000f000 rw-p 0 0 0\n");
    const std::vector<AnonymousRegion> ragged_regions = parse_smaps(ragged);
    failures += check(ragged_regions.size() == 1 && ragged_regions.front().size_bytes == 0,
                      "a malformed body produced a bogus region");

    // The report line is what an operator sees in the startup log, so it must name
    // the outcome and the first failure reason.
    AnonymousPageLockReport report;
    report.budget_bytes         = 1ULL << 30;
    report.memlock_limit_bytes  = 4ULL << 30;
    report.eligible_regions     = 3;
    report.eligible_resident    = 2ULL << 30;
    report.attempted_regions    = 2;
    report.locked_regions       = 1;
    report.locked_bytes         = 512ULL << 20;
    report.skipped_regions      = 1;
    report.failed_regions       = 1;
    report.failures.push_back("mlock(1.00 GiB at 0x7f00) failed: Cannot allocate memory");
    const std::string line = format_anonymous_page_lock_report(report);
    failures += check(line.find("locked 1 region(s)") != std::string::npos,
                      "report does not state what was locked");
    failures += check(line.find("skipped 1") != std::string::npos,
                      "report does not state the budget skip");
    failures += check(line.find("Cannot allocate memory") != std::string::npos,
                      "report does not carry the failure reason");

    // The effective budget never exceeds the policy ceiling and keeps a quarter of
    // RLIMIT_MEMLOCK free; a tiny limit still attempts the smallest eligible arena.
    const std::uint64_t limit = memlock_limit_bytes();
    const std::uint64_t budget = effective_lock_budget(policy);
    if (limit != 0) {
        failures += check(budget <= limit - limit / 4 || budget == policy.min_region_bytes,
                          "effective budget does not respect RLIMIT_MEMLOCK");
    }
    AnonymousPageLockPolicy tiny = policy;
    tiny.budget_bytes            = 1;
    failures += check(effective_lock_budget(tiny) == tiny.min_region_bytes,
                      "a tiny budget did not fall back to one attempt");

    if (failures == 0) { std::cerr << "anonymous_residency: ok\n"; }
    return failures == 0 ? 0 : 1;
}
