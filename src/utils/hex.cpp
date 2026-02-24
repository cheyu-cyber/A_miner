/*
 * hex.cpp – Hex encoding / decoding utilities.
 */
#include "hex.h"
#include <stdexcept>

namespace util {

static const char hex_chars[] = "0123456789abcdef";

std::string to_hex(const uint8_t* data, size_t len) {
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out += hex_chars[(data[i] >> 4) & 0x0f];
        out += hex_chars[data[i] & 0x0f];
    }
    return out;
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::invalid_argument("invalid hex character");
}

std::vector<uint8_t> from_hex(const std::string& hex) {
    if (hex.size() % 2 != 0)
        throw std::invalid_argument("hex string has odd length");
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2)
        out.push_back(static_cast<uint8_t>((hex_val(hex[i]) << 4) | hex_val(hex[i + 1])));
    return out;
}

} // namespace util
