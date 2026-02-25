/*
 * keccak.cpp – Keccak-f[1600] permutation and keccak-256 / keccak-512 hashes.
 *
 * This is the original Keccak submission (padding 0x01) used by Ethereum and
 * Ravencoin, NOT the NIST SHA-3 standard (padding 0x06).
 */
#include "keccak.h"
#include <cstring>
#include <algorithm>

namespace kawpow {

/* Round constants for Keccak-f[1600] */
static constexpr uint64_t keccak_round_constants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

/* Rotation offsets */
static constexpr int keccak_rot[24] = {
     1,  3,  6, 10, 15, 21, 28, 36,
    45, 55,  2, 14, 27, 41, 56,  8,
    25, 43, 62, 18, 39, 61, 20, 44,
};

/* Pi-lane permutation indices */
static constexpr int keccak_pi[24] = {
    10,  7, 11, 17, 18, 3,  5, 16,
     8, 21, 24,  4, 15, 23, 19, 13,
    12,  2, 20, 14, 22,  9,  6,  1,
};

static inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccak_f1600(uint64_t state[25]) {
    for (int round = 0; round < 24; ++round) {
        /* θ step */
        uint64_t C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];

        uint64_t D[5];
        for (int x = 0; x < 5; ++x)
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);

        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 25; y += 5)
                state[y + x] ^= D[x];

        /* ρ and π steps */
        uint64_t t = state[1];
        for (int i = 0; i < 24; ++i) {
            uint64_t tmp = state[keccak_pi[i]];
            state[keccak_pi[i]] = rotl64(t, keccak_rot[i]);
            t = tmp;
        }

        /* χ step */
        for (int y = 0; y < 25; y += 5) {
            uint64_t tmp[5];
            for (int x = 0; x < 5; ++x)
                tmp[x] = state[y + x];
            for (int x = 0; x < 5; ++x)
                state[y + x] = tmp[x] ^ ((~tmp[(x + 1) % 5]) & tmp[(x + 2) % 5]);
        }

        /* ι step */
        state[0] ^= keccak_round_constants[round];
    }
}

/*
 * Generic Keccak sponge.
 *   rate_bytes: 136 for keccak-256, 72 for keccak-512
 *   out_bytes:  32 for keccak-256, 64 for keccak-512
 */
static void keccak_absorb_squeeze(
    const uint8_t* data, size_t len,
    size_t rate_bytes, uint8_t* out, size_t out_bytes)
{
    uint64_t state[25] = {};
    const size_t rate_words = rate_bytes / 8;

    /* absorb */
    while (len >= rate_bytes) {
        for (size_t i = 0; i < rate_words; ++i) {
            uint64_t w;
            std::memcpy(&w, data + i * 8, 8);
            state[i] ^= w;
        }
        keccak_f1600(state);
        data += rate_bytes;
        len  -= rate_bytes;
    }

    /* last block: pad with 0x01...0x80 (original Keccak padding) */
    uint8_t block[200] = {};
    std::memcpy(block, data, len);
    block[len] = 0x01;
    block[rate_bytes - 1] |= 0x80;

    for (size_t i = 0; i < rate_words; ++i) {
        uint64_t w;
        std::memcpy(&w, block + i * 8, 8);
        state[i] ^= w;
    }
    keccak_f1600(state);

    /* squeeze */
    std::memcpy(out, state, out_bytes);
}

hash256 keccak256(const uint8_t* data, size_t len) {
    hash256 out;
    keccak_absorb_squeeze(data, len, /*rate=*/136, out.data(), 32);
    return out;
}

hash512 keccak512(const uint8_t* data, size_t len) {
    hash512 out;
    keccak_absorb_squeeze(data, len, /*rate=*/72, out.data(), 64);
    return out;
}

hash256 keccak256(const hash256& h) {
    return keccak256(h.data(), h.size());
}

hash512 keccak512(const hash256& h) {
    return keccak512(h.data(), h.size());
}

hash512 keccak512(const hash512& h) {
    return keccak512(h.data(), h.size());
}

} // namespace kawpow
