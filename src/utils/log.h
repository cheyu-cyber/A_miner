/*
 * log.h – Simple thread-safe logging utility.
 */
#pragma once

#include <string>
#include <cstdio>

namespace util {

enum class LogLevel { DEBUG, INFO, WARN, ERR };

void set_log_level(LogLevel level);

void log_debug(const char* fmt, ...);
void log_info(const char* fmt, ...);
void log_warn(const char* fmt, ...);
void log_error(const char* fmt, ...);

} // namespace util
