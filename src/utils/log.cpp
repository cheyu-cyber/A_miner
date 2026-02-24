/*
 * log.cpp – Simple thread-safe logging utility.
 */
#include "log.h"
#include <cstdarg>
#include <ctime>
#include <mutex>

namespace util {

static LogLevel g_level = LogLevel::INFO;
static std::mutex g_log_mutex;

void set_log_level(LogLevel level) {
    g_level = level;
}

static void vlog(const char* tag, const char* fmt, va_list ap) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    time_t now = std::time(nullptr);
    struct tm tm_buf;
    localtime_r(&now, &tm_buf);
    char time_str[20];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
    std::fprintf(stderr, "[%s] %s: ", time_str, tag);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
}

void log_debug(const char* fmt, ...) {
    if (g_level > LogLevel::DEBUG) return;
    va_list ap; va_start(ap, fmt); vlog("DBG", fmt, ap); va_end(ap);
}

void log_info(const char* fmt, ...) {
    if (g_level > LogLevel::INFO) return;
    va_list ap; va_start(ap, fmt); vlog("INF", fmt, ap); va_end(ap);
}

void log_warn(const char* fmt, ...) {
    if (g_level > LogLevel::WARN) return;
    va_list ap; va_start(ap, fmt); vlog("WRN", fmt, ap); va_end(ap);
}

void log_error(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); vlog("ERR", fmt, ap); va_end(ap);
}

} // namespace util
