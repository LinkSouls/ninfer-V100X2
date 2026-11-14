// Host-only: non-blocking stderr replacement.
//
// The contract under test is the one serving depends on: a committed line reaches
// the downstream whole, an uncommittable line is dropped whole and counted, an
// installation failure leaves fd 2 untouched and blocking, and console lines keep
// their prefix format in both modes.
//
// The test never writes through the terminal: it points fd 2 at a FIFO it owns, so
// the assertions are made on bytes rather than on a human reader's impression.

#include "product/logging/nonblocking_stderr.h"
#include "serve/console_log.h"

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;

int failures = 0;

// Failure reports go to a descriptor saved before fd 2 is redirected, so a failing
// assertion is visible even while the test is exercising the redirect.
int g_report_fd = STDERR_FILENO;

void report(const std::string& text) {
    const std::string line = text + "\n";
    const ssize_t written  = ::write(g_report_fd, line.data(), line.size());
    (void)written;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    report(std::string("FAIL: ") + message);
    return 1;
}

// Reads whatever the FIFO/pipe currently holds, without waiting for EOF. Returns the
// bytes read and whether the descriptor had nothing left.
std::string drain(int fd, std::chrono::milliseconds settle) {
    std::string collected;
    const int flags = ::fcntl(fd, F_GETFL);
    (void)::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    char buffer[64 * 1024];
    const auto deadline = std::chrono::steady_clock::now() + settle;
    while (std::chrono::steady_clock::now() < deadline) {
        const ssize_t got = ::read(fd, buffer, sizeof buffer);
        if (got > 0) {
            collected.append(buffer, static_cast<std::size_t>(got));
            continue;
        }
        if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        break;
    }
    (void)::fcntl(fd, F_SETFL, flags);
    return collected;
}

// Every byte a downstream received must belong to a complete line. A trailing
// fragment is exactly the defect the module exists to prevent, so it is checked on
// every capture rather than only on the drop path.
bool only_whole_lines(const std::string& received, std::vector<std::string>& lines) {
    bool whole = true;
    std::size_t start = 0;
    while (start < received.size()) {
        const std::size_t newline = received.find('\n', start);
        if (newline == std::string::npos) {
            whole = false; // a fragment was delivered
            break;
        }
        lines.push_back(received.substr(start, newline - start + 1));
        start = newline + 1;
    }
    return whole;
}

// --- installation failure ----------------------------------------------------

// Runs enable_nonblocking_stderr() in a child whose descriptor limit is already
// exhausted, so pipe2() fails with EMFILE. The child reports only through its exit
// status, because its stderr is part of what is being asserted.
int run_capacity_failure_child() {
    const pid_t pid = ::fork();
    if (pid < 0) { return -1; }
    if (pid == 0) {
        struct stat before {};
        if (::fstat(STDERR_FILENO, &before) != 0) { ::_exit(10); }
        // Clear every descriptor above the three standard ones is not needed: the
        // limit is lowered below the first free descriptor, so pipe2() cannot get
        // the two it needs and fails with EMFILE. Lowering the limit does not close
        // descriptors that are already open.
        rlimit limit{};
        limit.rlim_cur = 4;
        limit.rlim_max = 4;
        if (::setrlimit(RLIMIT_NOFILE, &limit) != 0) { ::_exit(11); }
        int probe_fds[2] = {-1, -1};
        if (::pipe(probe_fds) == 0) {
            // Two descriptors were available, so a failure below would not be about
            // the limit; keep the child from leaving them behind either way.
            ::close(probe_fds[0]);
            ::close(probe_fds[1]);
            ::_exit(18);
        }

        const std::string error = product::enable_nonblocking_stderr();
        if (error.empty()) { ::_exit(12); }               // must report a reason
        if (product::nonblocking_stderr_enabled()) { ::_exit(13); }
        if (product::nonblocking_stderr_last_error() != error) { ::_exit(14); }
        struct stat after {};
        if (::fstat(STDERR_FILENO, &after) != 0) { ::_exit(15); }
        if (after.st_dev != before.st_dev || after.st_ino != before.st_ino) { ::_exit(16); }
        if (product::nonblocking_stderr_max_line_bytes() != 0) { ::_exit(17); }
        ::_exit(0);
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) { return -1; }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// --- console line format -----------------------------------------------------

std::string capture_console_line(serve::ConsoleLogLevel level, const std::string& message) {
    int fds[2];
    if (::pipe(fds) != 0) { return {}; }
    const int saved = ::dup(STDERR_FILENO);
    if (saved < 0) { return {}; }
    (void)::dup2(fds[1], STDERR_FILENO);
    serve::write_console_log(level, message);
    std::string captured = drain(fds[0], std::chrono::milliseconds(200));
    (void)::dup2(saved, STDERR_FILENO);
    ::close(saved);
    ::close(fds[0]);
    ::close(fds[1]);
    return captured;
}

bool matches_console_format(const std::string& line, const char* level,
                            const std::string& message) {
    // "[YYYY-MM-DD HH:MM:SS.mmm] [level] ninfer-serve: message\n"
    if (line.size() < 45 || line.back() != '\n') { return false; }
    if (line.front() != '[') { return false; }
    if (line[5] != '-' || line[8] != '-' || line[11] != ' ' || line[14] != ':' ||
        line[17] != ':' || line[20] != '.') {
        return false;
    }
    for (int i : {1, 2, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 23}) {
        if (line[static_cast<std::size_t>(i)] < '0' || line[static_cast<std::size_t>(i)] > '9') {
            return false;
        }
    }
    const std::string expected_tail =
        std::string("] [") + level + "] ninfer-serve: " + message + "\n";
    return line.compare(line.size() - expected_tail.size(), expected_tail.size(),
                        expected_tail) == 0;
}

} // namespace

int main() {
    g_report_fd = ::dup(STDERR_FILENO);
    if (g_report_fd < 0) { g_report_fd = STDERR_FILENO; }

    failures += check(run_capacity_failure_child() == 0,
                      "an installation failure did not report a reason and leave fd 2 intact");

    // --- the installed path --------------------------------------------------
    // $TMPDIR is the sandbox-writable scratch directory; /tmp itself may be a
    // read-only mount in some of the environments this test runs in.
    std::string fifo_path;
    if (const char* tmp = std::getenv("TMPDIR"); tmp != nullptr && tmp[0] != '\0') {
        fifo_path = std::string(tmp) + "/ninfer-stderr-test.fifo";
    } else {
        fifo_path = "/tmp/ninfer-stderr-test.fifo";
    }
    ::unlink(fifo_path.c_str());
    if (::mkfifo(fifo_path.c_str(), 0600) != 0) {
        std::perror("mkfifo");
        return 1;
    }
    // O_RDWR keeps the FIFO from ever reaching EOF, so the forwarder blocks once the
    // kernel FIFO buffer is full and the inner pipe backs up deterministically.
    const int sink = ::open(fifo_path.c_str(), O_RDWR | O_CLOEXEC);
    if (sink < 0) {
        std::perror("open fifo");
        return 1;
    }
    const int saved_stderr = ::dup(STDERR_FILENO);
    (void)::dup2(sink, STDERR_FILENO);

    const std::string install_error = product::enable_nonblocking_stderr();
    failures += check(install_error.empty(), "non-blocking stderr did not install");
    failures += check(product::nonblocking_stderr_enabled(), "enabled() disagrees with install");
    failures += check(product::nonblocking_stderr_max_line_bytes() == 4096,
                      "commit limit is not PIPE_BUF");
    failures += check(product::nonblocking_stderr_pipe_capacity_bytes() >= 4096,
                      "pipe capacity was not recorded");
    failures += check(product::nonblocking_stderr_dropped_lines() == 0,
                      "drop counter did not start at zero");

    // Every commit below is observable on the FIFO once drained, so nothing is
    // examined while the pipeline still holds data: each capture drains the FIFO
    // first, and the whole-line judgement is made only on a stream whose writer has
    // exited.

    // --- console line format, non-blocking mode ------------------------------
    // The line travels fd 2 -> inner pipe -> forwarder -> FIFO, so the format is
    // checked on the bytes the downstream actually receives. Drain first: an empty
    // pipeline means the line cannot be one of the drops.
    (void)drain(sink, std::chrono::milliseconds(400));
    serve::write_console_log(serve::ConsoleLogLevel::Warning, "hardening check");
    std::string line_capture = drain(sink, std::chrono::milliseconds(400));
    failures += check(matches_console_format(line_capture, "warning", "hardening check"),
                      "non-blocking console line format changed");

    // A line at the commit limit must pass; one byte more must be dropped whole.
    const std::uint64_t drops_before = product::nonblocking_stderr_dropped_lines();
    const std::string limit_line(4095, 'x');
    failures += check(product::write_nonblocking_stderr_line(limit_line + "\n"),
                      "a PIPE_BUF-sized line was rejected");
    const std::string oversized(4096, 'y');
    failures += check(!product::write_nonblocking_stderr_line(oversized + "\n"),
                      "an oversized line was accepted");
    failures += check(product::nonblocking_stderr_dropped_lines() == drops_before + 1,
                      "the oversized line was not counted as one drop");

    // Fill until the downstream stops consuming. The FIFO holds 64 KiB and the inner
    // pipe 1 MiB, so 300k short lines overflow both and force drops.
    constexpr int kLines = 300000;
    std::uint64_t attempted = 0;
    std::uint64_t accepted  = 0;
    for (int i = 0; i < kLines; ++i) {
        const std::string line = "line " + std::to_string(i) + "\n";
        ++attempted;
        if (product::write_nonblocking_stderr_line(line)) { ++accepted; }
    }
    const std::uint64_t dropped = product::nonblocking_stderr_dropped_lines();
    failures += check(dropped > 1, "no line was dropped while the downstream was stalled");
    failures += check(dropped == attempted - accepted + 1,
                      "drop counter does not match the rejected writes");
    failures += check(product::nonblocking_stderr_forwarded_bytes() > 0,
                      "nothing was forwarded to the original stderr");

    // --- std::cerr goes through the same whole-line path ----------------------
    // Two insertions, one line: the bridge must join them before committing, which
    // is the property that keeps a stray `std::cerr << a << b << '\n'` from being
    // split by an EAGAIN in the middle.
    (void)drain(sink, std::chrono::milliseconds(400));
    std::cerr << "cerr fragment";
    std::cerr << " completed" << '\n';
    failures += check(std::cerr.good(), "std::cerr latched a failure on the non-blocking pipe");

    // --- restore -------------------------------------------------------------
    product::restore_original_stderr();
    failures += check(!product::nonblocking_stderr_enabled(), "restore did not disable the mode");
    failures += check(!product::write_nonblocking_stderr_line("after restore\n"),
                      "writes still succeed after restore");
    struct stat restored {};
    struct stat sink_stat {};
    if (::fstat(STDERR_FILENO, &restored) == 0 && ::fstat(sink, &sink_stat) == 0) {
        failures += check(restored.st_ino == sink_stat.st_ino,
                          "restore did not put the original descriptor back on fd 2");
    }
    // Collect everything the forwarder had left; it sees EOF on the inner pipe once
    // the last write end closes, so this drain returns the complete stream.
    line_capture += drain(sink, std::chrono::milliseconds(1500));

    // Whole-line invariant over the complete received stream: every byte a downstream
    // got belongs to a line that was committed whole. A trailing fragment would be
    // exactly the defect this module exists to prevent.
    std::vector<std::string> lines;
    failures += check(only_whole_lines(line_capture, lines),
                      "the downstream received a partial line");
    bool saw_cerr_line = false;
    for (const std::string& line : lines) {
        if (line == "cerr fragment completed\n") { saw_cerr_line = true; }
        failures += check(line.size() <= 4096, "a delivered line exceeded PIPE_BUF");
    }
    // The cerr line is only guaranteed delivered when it was not one of the drops.
    // If it is absent, it must be absent whole: the partial fragment must never
    // appear on its own.
    failures += check(saw_cerr_line || line_capture.find("cerr fragment") == std::string::npos,
                      "std::cerr delivered a partial line");
    // The console line, the FIFO-limit probe and the cerr join all travelled the same
    // path, and the complete stream is now in `line_capture`; the console line must be
    // present whole.
    failures += check(line_capture.find("] [warning] ninfer-serve: hardening check\n") !=
                          std::string::npos,
                      "the console line did not reach the downstream");

    // Blocking-mode console line, on the restored descriptor.
    const std::string blocking_line =
        capture_console_line(serve::ConsoleLogLevel::Error, "blocking check");
    failures += check(matches_console_format(blocking_line, "error", "blocking check"),
                      "blocking console line format changed");

    (void)::dup2(saved_stderr, STDERR_FILENO);
    ::close(saved_stderr);
    ::close(sink);
    ::unlink(fifo_path.c_str());

    if (failures == 0) { report("nonblocking_stderr: ok"); }
    return failures == 0 ? 0 : 1;
}
