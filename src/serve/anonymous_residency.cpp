#include "serve/anonymous_residency.h"

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <istream>
#include <numeric>
#include <sstream>
#include <system_error>

namespace ninfer::serve {
namespace {

std::uint64_t parse_bytes_field(std::string_view value) {
    // smaps prints "<kib> kB" for every byte field; older kernels omit the unit.
    while (!value.empty() && value.back() == ' ') { value.remove_suffix(1); }
    const std::size_t space = value.find(' ');
    const std::string_view number = value.substr(0, space);
    std::uint64_t kib = 0;
    const auto parsed = std::from_chars(number.data(), number.data() + number.size(), kib);
    if (parsed.ec != std::errc{}) { return 0; }
    return kib * 1024;
}

std::string format_bytes(std::uint64_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (bytes >= (1ULL << 30)) {
        out << static_cast<double>(bytes) / static_cast<double>(1ULL << 30) << " GiB";
    } else {
        out << static_cast<double>(bytes) / static_cast<double>(1ULL << 20) << " MiB";
    }
    return out.str();
}

std::string errno_text(int error) { return std::string(std::strerror(error)); }

} // namespace

std::vector<AnonymousRegion> parse_smaps(std::istream& input) {
    std::vector<AnonymousRegion> regions;
    std::string line;
    AnonymousRegion current;
    bool in_mapping = false;

    auto flush = [&] {
        if (in_mapping) { regions.push_back(std::move(current)); }
        current     = AnonymousRegion{};
        in_mapping  = false;
    };

    while (std::getline(input, line)) {
        if (!line.empty() && line.front() != ' ' && line.find('-') != std::string::npos) {
            // Mapping header: "start-end perms offset dev inode [path]".
            flush();
            const std::size_t dash = line.find('-');
            const std::size_t sp1  = line.find(' ', dash);
            if (dash == std::string::npos || sp1 == std::string::npos) { continue; }
            std::uintptr_t start = 0;
            std::uintptr_t end   = 0;
            const auto start_parse =
                std::from_chars(line.data(), line.data() + dash, start, 16);
            const auto end_parse = std::from_chars(line.data() + dash + 1,
                                                   line.data() + sp1, end, 16);
            if (start_parse.ec != std::errc{} || end_parse.ec != std::errc{}) { continue; }
            current.start = start;
            current.end   = end;
            std::istringstream header(line.substr(sp1 + 1));
            std::string offset;
            std::string device;
            header >> current.permissions >> offset >> device >> current.inode;
            std::string path;
            std::getline(header, path);
            while (!path.empty() && path.front() == ' ') { path.erase(path.begin()); }
            while (!path.empty() && path.back() == ' ') { path.pop_back(); }
            current.name = path;
            in_mapping   = true;
            continue;
        }
        if (!in_mapping) { continue; }

        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) { continue; }
        const std::string_view key(line.data(), colon);
        const std::string_view value(line.data() + colon + 1,
                                     line.size() - colon - 1);
        std::string_view trimmed = value;
        while (!trimmed.empty() && trimmed.front() == ' ') { trimmed.remove_prefix(1); }

        if (key == "Size") {
            current.size_bytes = parse_bytes_field(trimmed);
        } else if (key == "Rss") {
            current.resident_bytes = parse_bytes_field(trimmed);
        } else if (key == "Locked") {
            current.locked_bytes = parse_bytes_field(trimmed);
        }
    }
    flush();
    return regions;
}

bool read_self_smaps(std::vector<AnonymousRegion>& regions, std::string& error) {
    std::ifstream input("/proc/self/smaps");
    if (!input) {
        error = "cannot open /proc/self/smaps: " + errno_text(errno);
        return false;
    }
    regions = parse_smaps(input);
    if (regions.empty()) {
        error = "/proc/self/smaps produced no parsable mappings";
        return false;
    }
    return true;
}

std::vector<AnonymousRegion> select_eligible_regions(
    const std::vector<AnonymousRegion>& regions, const AnonymousPageLockPolicy& policy) {
    std::vector<AnonymousRegion> eligible;
    for (const AnonymousRegion& region : regions) {
        if (!region.anonymous() || !region.readable_writable()) { continue; }
        if (region.size_bytes < policy.min_region_bytes ||
            region.size_bytes > policy.max_region_bytes) {
            continue;
        }
        if (region.resident_bytes < policy.min_resident_bytes) { continue; }
        eligible.push_back(region);
    }
    return eligible;
}

std::uint64_t memlock_limit_bytes() {
    rlimit limit{};
    if (::getrlimit(RLIMIT_MEMLOCK, &limit) != 0) { return 0; }
    if (limit.rlim_cur == RLIM_INFINITY) { return 0; }
    return static_cast<std::uint64_t>(limit.rlim_cur);
}

std::uint64_t effective_lock_budget(const AnonymousPageLockPolicy& policy) {
    const std::uint64_t limit = memlock_limit_bytes();
    std::uint64_t budget      = policy.budget_bytes;
    if (limit != 0) { budget = std::min(budget, limit - limit / 4); }
    // A tiny RLIMIT_MEMLOCK must not turn into "lock nothing" silently: attempt the
    // smallest eligible arena and let the failure be reported with its errno.
    return std::max(budget, policy.min_region_bytes);
}

AnonymousPageLockReport lock_anonymous_pages(const AnonymousPageLockPolicy& policy) {
    AnonymousPageLockReport report;
    report.budget_bytes        = effective_lock_budget(policy);
    report.memlock_limit_bytes = memlock_limit_bytes();

    std::vector<AnonymousRegion> regions;
    if (!read_self_smaps(regions, report.scan_error)) {
        report.failures.push_back(report.scan_error);
        return report;
    }
    report.scanned_regions = regions.size();

    std::vector<AnonymousRegion> eligible = select_eligible_regions(regions, policy);
    report.eligible_regions = eligible.size();
    for (const AnonymousRegion& region : eligible) {
        report.eligible_bytes += region.size_bytes;
        report.eligible_resident += region.resident_bytes;
    }

    // Largest resident first: the budget is the scarce resource, and the whole point
    // of locking is to protect the arenas whose pages are actually touched.
    std::sort(eligible.begin(), eligible.end(), [](const AnonymousRegion& left,
                                                   const AnonymousRegion& right) {
        return left.resident_bytes > right.resident_bytes;
    });

    for (const AnonymousRegion& region : eligible) {
        if (region.locked_bytes >= region.size_bytes) {
            ++report.already_locked_regions;
            continue;
        }
        // RLIMIT_MEMLOCK is charged by virtual size, so locking a partially locked
        // region costs only the remainder.
        const std::uint64_t already_locked = std::min(region.locked_bytes, region.size_bytes);
        const std::uint64_t additional     = region.size_bytes - already_locked;
        if (report.locked_bytes + additional > report.budget_bytes) {
            ++report.skipped_regions;
            continue;
        }
        ++report.attempted_regions;
        void* const address = reinterpret_cast<void*>(region.start);
        const std::size_t length = static_cast<std::size_t>(region.size_bytes);
        if (::mlock(address, length) == 0) {
            ++report.locked_regions;
            report.locked_bytes += additional;
            continue;
        }
        ++report.failed_regions;
        std::ostringstream message;
        message << "mlock(" << format_bytes(region.size_bytes) << " at 0x" << std::hex
                << region.start << std::dec << ") failed: " << errno_text(errno);
        report.failures.push_back(message.str());
    }
    return report;
}

std::string format_anonymous_page_lock_report(const AnonymousPageLockReport& report) {
    std::ostringstream out;
    out << "anonymous page lock: locked " << report.locked_regions << " region(s) / "
        << format_bytes(report.locked_bytes) << " of budget " << format_bytes(report.budget_bytes);
    if (report.memlock_limit_bytes != 0) { out << " (RLIMIT_MEMLOCK " << format_bytes(report.memlock_limit_bytes) << ')'; }
    out << "; eligible " << report.eligible_regions << " region(s) / "
        << format_bytes(report.eligible_resident) << " resident";
    if (report.skipped_regions != 0) { out << ", skipped " << report.skipped_regions << " over budget"; }
    if (report.already_locked_regions != 0) {
        out << ", already locked " << report.already_locked_regions;
    }
    if (report.failed_regions != 0) { out << ", failed " << report.failed_regions; }
    if (!report.failures.empty()) { out << "; first failure: " << report.failures.front(); }
    if (!report.scan_error.empty()) { out << "; scan error: " << report.scan_error; }
    return out.str();
}

} // namespace ninfer::serve
