#include "serve/console_log.h"

#include "product/logging/nonblocking_stderr.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace ninfer::serve {
namespace {

const char* level_name(ConsoleLogLevel level) noexcept {
    switch (level) {
    case ConsoleLogLevel::Info:
        return "info";
    case ConsoleLogLevel::Warning:
        return "warning";
    case ConsoleLogLevel::Error:
        return "error";
    }
    return "info";
}

std::mutex& console_mutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace

std::string format_console_log_prefix(std::chrono::system_clock::time_point timestamp,
                                      ConsoleLogLevel level) {
    const auto since_epoch   = timestamp.time_since_epoch();
    const auto whole_seconds = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch - whole_seconds).count();
    const std::time_t wall_seconds =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::time_point(whole_seconds));
    std::tm local{};
    localtime_r(&wall_seconds, &local);

    std::ostringstream out;
    out << '[' << std::put_time(&local, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0')
        << std::setw(3) << milliseconds << "] [" << level_name(level) << "] ninfer-serve: ";
    return out.str();
}

std::string current_console_log_prefix(ConsoleLogLevel level) {
    return format_console_log_prefix(std::chrono::system_clock::now(), level);
}

void write_console_log(ConsoleLogLevel level, std::string_view message) {
    // Prefix format, level vocabulary, and the mutex that keeps prefix and message in
    // one piece are identical in both modes. Only the sink differs: the non-blocking
    // sink commits the whole line or drops it, so a backed-up downstream can neither
    // park this thread nor split a line. `message` may itself contain newlines; each
    // embedded line is prefixed, matching the blocking std::cerr form below.
    std::string line = current_console_log_prefix(level);
    line.reserve(line.size() + message.size() + 2);

    std::lock_guard lock(console_mutex());
    if (product::nonblocking_stderr_enabled()) {
        const bool single_line = message.find('\n') == std::string_view::npos;
        if (single_line) {
            line.append(message);
            line.push_back('\n');
            (void)product::write_nonblocking_stderr_line(line);
            return;
        }
        const std::string prefix = line;
        std::size_t start        = 0;
        while (start <= message.size()) {
            const std::size_t newline = message.find('\n', start);
            const std::string_view segment =
                newline == std::string_view::npos
                    ? message.substr(start)
                    : message.substr(start, newline - start);
            std::string embedded = prefix;
            embedded.append(segment);
            embedded.push_back('\n');
            (void)product::write_nonblocking_stderr_line(embedded);
            if (newline == std::string_view::npos) { break; }
            start = newline + 1;
        }
        return;
    }
    line.append(message);
    line.push_back('\n');
    // Blocking fallback: one insertion, preserving the previous behaviour exactly.
    std::cerr << line;
}

} // namespace ninfer::serve
