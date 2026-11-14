#include "product/logging/nonblocking_stderr.h"

#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <exception>
#include <ios>
#include <iostream>
#include <mutex>
#include <streambuf>
#include <string>
#include <thread>

namespace ninfer::product {
namespace {

// Process-lifetime state. The replacement is installed before any worker thread
// exists and is torn down only on exit paths, so nothing here needs destruction
// ordering beyond "still alive while writers exist".

std::atomic<bool> g_enabled{false};
std::atomic<std::uint64_t> g_dropped_lines{0};
std::atomic<std::uint64_t> g_dropped_bytes{0};
std::atomic<std::uint64_t> g_forwarded_bytes{0};
std::atomic<std::size_t> g_max_line_bytes{0};
std::atomic<std::size_t> g_pipe_capacity_bytes{0};
std::atomic<bool> g_original_is_tty{false};

std::mutex& error_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string g_last_error;
int g_original_stderr_fd = -1; // dup of the pre-replacement fd 2
int g_pipe_read_fd       = -1;
int g_pipe_write_fd      = -1; // the pipe write end; fd 2 is a duplicate of it
std::streambuf* g_original_cerr_buffer = nullptr;

std::mutex& install_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string errno_reason(const char* operation) {
    return std::string(operation) + " failed: " + std::strerror(errno);
}

// Copies everything the pipe holds to the original stderr. Only this thread writes
// to g_original_stderr_fd, so a backed-up downstream parks this forwarder and no
// logging caller.
void forward_loop() {
    char buffer[64 * 1024];
    for (;;) {
        const ssize_t read_bytes = ::read(g_pipe_read_fd, buffer, sizeof buffer);
        if (read_bytes > 0) {
            const auto total   = static_cast<std::size_t>(read_bytes);
            std::size_t offset = 0;
            while (offset < total) {
                const ssize_t written =
                    ::write(g_original_stderr_fd, buffer + offset, total - offset);
                if (written > 0) {
                    offset += static_cast<std::size_t>(written);
                    continue;
                }
                if (written < 0 && errno == EINTR) { continue; }
                // The original stderr is unusable (closed, EIO). Dropping the rest
                // beats spinning on a dead descriptor.
                g_forwarded_bytes.fetch_add(offset, std::memory_order_relaxed);
                return;
            }
            g_forwarded_bytes.fetch_add(total, std::memory_order_relaxed);
            continue;
        }
        if (read_bytes < 0 && errno == EINTR) { continue; }
        return; // every write end closed, or an unrecoverable read error.
    }
}

// --- std::cerr bridge --------------------------------------------------------
//
// Once fd 2 is a non-blocking pipe, an unmodified std::cerr would (a) be able to
// stop mid-line on EAGAIN and (b) latch badbit, silently discarding every later
// line. This buffer accumulates a line and commits it whole through the same drop
// path, so std::cerr output obeys the same contract as write_console_log(). It
// never reports failure, so cerr's state stays clean.
class LineCommittingStreamBuffer final : public std::streambuf {
public:
    // Deliberately no put area: with one, the base class would copy characters into
    // that area and only call sync() on flush, so a line would reach commit() in
    // pieces. Without one every insertion funnels through xsputn()/overflow().
    //
    // A partial line must never reach the downstream, so an explicit flush — which
    // std::cerr performs after every insertion, because it is unit-buffered — is a
    // no-op rather than a commit.
    int sync() override { return 0; }

protected:
    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) { return traits_type::not_eof(ch); }
        const char one = traits_type::to_char_type(ch);
        return consume(&one, 1) == 1 ? ch : traits_type::eof();
    }

    std::streamsize xsputn(const char* text, std::streamsize count) override {
        return static_cast<std::streamsize>(consume(text, static_cast<std::size_t>(count)));
    }

private:
    // Interactive progress redraws with '\r' and no newline. A bare '\r' starts a new
    // frame; the frame is committed when the next '\r' or the final '\n' arrives, so
    // a terminal receives "\rframe" sequences and a log file receives whole lines.
    std::size_t consume(const char* text, std::size_t count) {
        const std::size_t max_line = g_max_line_bytes.load(std::memory_order_relaxed);
        for (std::size_t i = 0; i < count; ++i) {
            const char ch = text[i];
            if (ch == '\r') {
                if (!pending_.empty()) { commit(); }
                pending_.push_back(ch);
                continue;
            }
            pending_.push_back(ch);
            if (ch == '\n') {
                commit();
            } else if (max_line != 0 && pending_.size() >= max_line) {
                // This frame can never be committed, so drop it now instead of letting
                // it grow without bound. Committing it would fail for the same reason.
                g_dropped_bytes.fetch_add(pending_.size(), std::memory_order_relaxed);
                pending_.clear();
            }
        }
        return count;
    }

    void commit() {
        (void)write_nonblocking_stderr_line(std::string_view(pending_.data(), pending_.size()));
        pending_.clear();
    }

    std::string pending_;
};

LineCommittingStreamBuffer& stream_buffer() {
    // Leaked on purpose: std::cerr is destroyed after function-local statics, so an
    // automatic object here would die while cerr still points at it.
    static LineCommittingStreamBuffer* buffer = new LineCommittingStreamBuffer();
    return *buffer;
}

} // namespace

std::string enable_nonblocking_stderr(const NonblockingStderrOptions& options) {
    std::lock_guard lock(install_mutex());
    if (g_enabled.load(std::memory_order_acquire)) { return {}; }

    auto record_and_return = [](std::string reason) {
        {
            std::lock_guard error_lock(error_mutex());
            g_last_error = reason;
        }
        return reason;
    };

    int fds[2];
    if (::pipe2(fds, O_CLOEXEC) != 0) { return record_and_return(errno_reason("pipe2")); }
    const int read_fd  = fds[0];
    const int write_fd = fds[1];

    auto fail = [&](const std::string& reason) {
        ::close(read_fd);
        ::close(write_fd);
        return record_and_return(reason);
    };

    long pipe_buf_bytes = ::fpathconf(write_fd, _PC_PIPE_BUF);
    if (pipe_buf_bytes <= 0) { pipe_buf_bytes = 4096; }

    if (options.pipe_bytes != 0) {
        // A clamp to /proc/sys/fs/pipe-max-size is normal for an unprivileged process
        // and is not an error: capacity only decides how much burst survives.
        (void)::fcntl(write_fd, F_SETPIPE_SZ, static_cast<int>(options.pipe_bytes));
    }

    const int write_flags = ::fcntl(write_fd, F_GETFL);
    if (write_flags < 0 || ::fcntl(write_fd, F_SETFL, write_flags | O_NONBLOCK) != 0) {
        return fail(errno_reason("fcntl(O_NONBLOCK)"));
    }

    const bool original_is_tty = ::isatty(STDERR_FILENO) == 1;
    const int saved_fd         = ::dup(STDERR_FILENO);
    if (saved_fd < 0) { return fail(errno_reason("dup(stderr)")); }
    (void)::fcntl(saved_fd, F_SETFD, FD_CLOEXEC);

    // The forwarder's write to the original stderr can raise SIGPIPE if that
    // downstream closed while the server was running. A dead log consumer must not
    // kill a serving process; the HTTP server ignores SIGPIPE for the same reason.
    ::signal(SIGPIPE, SIG_IGN);

    if (::dup2(write_fd, STDERR_FILENO) < 0) {
        const std::string reason = errno_reason("dup2(stderr)");
        ::close(saved_fd);
        return fail(reason);
    }

    g_original_stderr_fd = saved_fd;
    g_pipe_read_fd       = read_fd;
    g_pipe_write_fd      = write_fd;
    g_original_is_tty.store(original_is_tty, std::memory_order_relaxed);
    // fcntl(pipe?) reports the kernel's actual capacity even when F_SETPIPE_SZ was
    // clamped; record it so startup can report what the pipe really holds.
    const long actual_capacity = ::fcntl(write_fd, F_GETPIPE_SZ);
    g_pipe_capacity_bytes.store(
        actual_capacity > 0 ? static_cast<std::size_t>(actual_capacity) : 0,
        std::memory_order_relaxed);
    // Atomicity on an O_NONBLOCK pipe is guaranteed only below PIPE_BUF, so that
    // boundary — not the larger capacity — is the commit limit.
    g_max_line_bytes.store(static_cast<std::size_t>(pipe_buf_bytes),
                           std::memory_order_relaxed);

    try {
        std::thread(forward_loop).detach();
    } catch (const std::exception& exception) {
        (void)::dup2(saved_fd, STDERR_FILENO);
        ::close(saved_fd);
        g_original_stderr_fd = -1;
        g_pipe_read_fd       = -1;
        g_pipe_write_fd      = -1;
        g_max_line_bytes.store(0, std::memory_order_relaxed);
        g_pipe_capacity_bytes.store(0, std::memory_order_relaxed);
        return record_and_return(std::string("stderr forwarder thread could not start: ") +
                                 exception.what());
    }

    // Every std::cerr insertion now commits whole lines, so a stray cerr user can
    // neither block on nor fragment the pipe.
    g_original_cerr_buffer = std::cerr.rdbuf(&stream_buffer());
    g_enabled.store(true, std::memory_order_release);
    {
        std::lock_guard error_lock(error_mutex());
        g_last_error.clear();
    }
    return {};
}

bool nonblocking_stderr_enabled() noexcept { return g_enabled.load(std::memory_order_acquire); }

std::uint64_t nonblocking_stderr_dropped_lines() noexcept {
    return g_dropped_lines.load(std::memory_order_relaxed);
}

std::uint64_t nonblocking_stderr_dropped_bytes() noexcept {
    return g_dropped_bytes.load(std::memory_order_relaxed);
}

std::uint64_t nonblocking_stderr_forwarded_bytes() noexcept {
    return g_forwarded_bytes.load(std::memory_order_relaxed);
}

std::size_t nonblocking_stderr_max_line_bytes() noexcept {
    return g_max_line_bytes.load(std::memory_order_relaxed);
}

std::size_t nonblocking_stderr_pipe_capacity_bytes() noexcept {
    return g_pipe_capacity_bytes.load(std::memory_order_relaxed);
}

std::string nonblocking_stderr_last_error() {
    std::lock_guard lock(error_mutex());
    return g_last_error;
}

bool nonblocking_stderr_original_is_tty() noexcept {
    return g_original_is_tty.load(std::memory_order_relaxed);
}

bool write_nonblocking_stderr_line(std::string_view line) noexcept {
    if (!g_enabled.load(std::memory_order_acquire) || g_pipe_write_fd < 0) { return false; }
    const std::size_t max_line = g_max_line_bytes.load(std::memory_order_relaxed);
    if (line.empty() || line.size() > max_line) {
        // Dropped for the same reason as a full pipe: anything partly written would
        // put a fragment of the line downstream.
        g_dropped_lines.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // One write call of at most PIPE_BUF on a non-blocking pipe either lands whole or
    // fails with EAGAIN. A short count is treated as a drop rather than retried: a
    // retry is the blocking behaviour this module removes.
    for (;;) {
        const ssize_t written = ::write(g_pipe_write_fd, line.data(), line.size());
        if (written == static_cast<ssize_t>(line.size())) { return true; }
        if (written < 0 && errno == EINTR) { continue; }
        g_dropped_lines.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
}

void restore_original_stderr() noexcept {
    std::lock_guard lock(install_mutex());
    if (!g_enabled.exchange(false, std::memory_order_acq_rel)) { return; }
    if (g_original_cerr_buffer != nullptr) {
        std::cerr.rdbuf(g_original_cerr_buffer);
        g_original_cerr_buffer = nullptr;
    }
    if (g_original_stderr_fd < 0) { return; }

    (void)::dup2(g_original_stderr_fd, STDERR_FILENO);
    // The forwarder is not joined: closing the last write end gives it EOF after
    // draining what the pipe still holds into the restored descriptor.
    if (g_pipe_write_fd >= 0) { ::close(g_pipe_write_fd); }
    g_pipe_write_fd = -1;
}

} // namespace ninfer::product
