/*
 * keccak.h – Keccak (SHA-3) hash functions used by Ethash / KawPow.
 *
 * Provides keccak-256 and keccak-512 (the *original* Keccak submission,
 * NOT the NIST SHA-3 standard which uses a different padding).
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <array>

namespace kawpow {

/* Fixed-size output types */
using hash256 = std::array<uint8_t, 32>;
using hash512 = std::array<uint8_t, 64>;

/* Keccak-f[1600] permutation (operates on 25 x uint64 state) */
void keccak_f1600(uint64_t state[25]);

/* Keccak-256: arbitrary input  →  32-byte digest */
hash256 keccak256(const uint8_t* data, size_t len);

/* Keccak-512: arbitrary input  →  64-byte digest */
hash512 keccak512(const uint8_t* data, size_t len);

/* Convenience overloads for hash types */
hash256 keccak256(const hash256& h);
hash512 keccak512(const hash256& h);
hash512 keccak512(const hash512& h);

} // namespace kawpow
