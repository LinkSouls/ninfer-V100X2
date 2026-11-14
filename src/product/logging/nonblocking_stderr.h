#pragma once

// Process-level stderr resilience.
//
// A serving process writes diagnostics to stderr. When stderr's downstream stops
// consuming (journald back-pressure, a slow log driver, an unread FIFO), a blocking
// write parks the writing thread; because serving serializes console lines through
// one mutex, every other logging thread parks behind it and can eventually park the
// request path. This module replaces the process stderr with a non-blocking pipe
// drained by one forwarding thread. A line is committed whole or dropped whole, so
// no caller ever waits on the downstream.
//
// The replacement is process-wide, one-way, and established once during early
// startup, before any worker thread exists.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::product {

struct NonblockingStderrOptions {
    // Requested kernel pipe capacity. The kernel clamps this to
    // /proc/sys/fs/pipe-max-size; a clamp is not a failure. Zero keeps the
    // kernel default. Capacity bounds burst absorption only; the atomic commit
    // limit stays at PIPE_BUF.
    std::size_t pipe_bytes = 1U << 20;
};

// Replaces fd 2 with a non-blocking pipe and installs a line-committing buffer on
// std::cerr. Returns an empty string on success. On failure it returns a
// human-readable reason and leaves the original stderr descriptor in place and
// blocking.
[[nodiscard]] std::string enable_nonblocking_stderr(const NonblockingStderrOptions& options = {});

// True once the replacement is installed and not yet restored.
[[nodiscard]] bool nonblocking_stderr_enabled() noexcept;

// Number of lines dropped because the whole line did not fit in the pipe (or
// exceeded PIPE_BUF, or the pipe was closed).
[[nodiscard]] std::uint64_t nonblocking_stderr_dropped_lines() noexcept;

// Bytes the forwarding thread has handed to the original stderr.
[[nodiscard]] std::uint64_t nonblocking_stderr_forwarded_bytes() noexcept;

// Largest line that can be committed atomically: PIPE_BUF. Zero when the
// replacement is not installed.
[[nodiscard]] std::size_t nonblocking_stderr_max_line_bytes() noexcept;

// Actual kernel pipe capacity in bytes; zero when the replacement is not installed.
[[nodiscard]] std::size_t nonblocking_stderr_pipe_capacity_bytes() noexcept;

// Why the most recent enable_nonblocking_stderr() call returned a reason. Empty
// before any attempt and after a successful one.
[[nodiscard]] std::string nonblocking_stderr_last_error();

// Bytes of a std::cerr insertion that had to be discarded because they pushed the
// pending frame past the commit limit. A discarded fragment flushes itself and
// starts a fresh frame, so std::cerr's stream state is never poisoned.
[[nodiscard]] std::uint64_t nonblocking_stderr_dropped_bytes() noexcept;

// Whether the descriptor that was replaced referred to a terminal. Interactive load
// progress must render against this, not against the pipe's isatty().
[[nodiscard]] bool nonblocking_stderr_original_is_tty() noexcept;

// Commits one complete line (the caller supplies the trailing '\n'). Never blocks:
// a line that does not fit entirely is dropped and counted, so downstream never
// receives a fragment. Safe from any thread. Returns false when the line was
// dropped, and also when the replacement is not installed, so callers check
// nonblocking_stderr_enabled() first.
[[nodiscard]] bool write_nonblocking_stderr_line(std::string_view line) noexcept;

// Restores the original fd 2. Intended for exit and fatal-error paths only: it does
// not join the forwarding thread, so lines still in the pipe may be lost when the
// process exits immediately afterwards.
void restore_original_stderr() noexcept;

} // namespace ninfer::product
