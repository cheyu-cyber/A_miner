/*
 * hex.h – Hex encoding / decoding utilities.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace util {

std::string to_hex(const uint8_t* data, size_t len);
std::vector<uint8_t> from_hex(const std::string& hex);

} // namespace util
