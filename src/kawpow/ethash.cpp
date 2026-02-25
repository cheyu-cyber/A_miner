/*
 * ethash.cpp – Ethash-style DAG generation for Ravencoin / KawPow.
 */
#include "ethash.h"
#include <cstring>
#include <algorithm>

namespace kawpow {

/* ---------- prime helpers ---------- */

static bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    return true;
}

/* ---------- size tables ---------- */

/*
 * Cache and dataset sizes follow Ethash spec:
 *   initial_cache_size = 2^24  (16 MiB)
 *   cache_growth       = 2^17  (128 KiB per epoch)
 *   initial_dataset_size = 2^30  (1 GiB)
 *   dataset_growth       = 2^23  (8 MiB per epoch)
 * Size is the largest prime below the raw size that satisfies size % HASH_BYTES == 0.
 */

static constexpr uint64_t INITIAL_CACHE_SIZE   = 1ULL << 24;
static constexpr uint64_t CACHE_GROWTH         = 1ULL << 17;
static constexpr uint64_t INITIAL_DATASET_SIZE = 1ULL << 30;
static constexpr uint64_t DATASET_GROWTH       = 1ULL << 23;

uint64_t light_cache_size(uint32_t epoch) {
    uint64_t sz = INITIAL_CACHE_SIZE + CACHE_GROWTH * epoch;
    sz -= HASH_BYTES;
    while (!is_prime(sz / HASH_BYTES))
        sz -= HASH_BYTES;
    return sz;
}

uint64_t full_dataset_size(uint32_t epoch) {
    uint64_t sz = INITIAL_DATASET_SIZE + DATASET_GROWTH * epoch;
    sz -= MIX_BYTES;
    while (!is_prime(sz / MIX_BYTES))
        sz -= MIX_BYTES;
    return sz;
}

uint32_t epoch_from_block(uint64_t block_number) {
    return static_cast<uint32_t>(block_number / EPOCH_LENGTH);
}

/* ---------- seed hash ---------- */

hash256 seed_hash(uint32_t epoch) {
    hash256 seed{};
    for (uint32_t i = 0; i < epoch; ++i)
        seed = keccak256(seed);
    return seed;
}

/* ---------- FNV-1 (Ethash variant) ---------- */

static inline uint32_t fnv1(uint32_t u, uint32_t v) {
    return (u * 0x01000193) ^ v;
}

/* ---------- light cache ---------- */

LightCache make_light_cache(uint32_t epoch) {
    LightCache lc;
    lc.epoch      = epoch;
    lc.cache_size = light_cache_size(epoch);
    lc.num_items  = static_cast<uint32_t>(lc.cache_size / HASH_BYTES);

    lc.data.resize(lc.num_items);

    /* first item: keccak-512 of seed hash */
    hash256 s = seed_hash(epoch);
    lc.data[0] = keccak512(s.data(), s.size());

    /* sequential keccak-512 chain */
    for (uint32_t i = 1; i < lc.num_items; ++i)
        lc.data[i] = keccak512(lc.data[i - 1]);

    /* Sergio Demian Lerner's RandMemoHash */
    for (uint32_t round = 0; round < LIGHT_CACHE_ROUNDS; ++round) {
        for (uint32_t i = 0; i < lc.num_items; ++i) {
            uint32_t src_idx;
            std::memcpy(&src_idx, lc.data[i].data(), 4);
            src_idx %= lc.num_items;

            uint32_t prev = (i == 0) ? lc.num_items - 1 : i - 1;

            hash512 mix;
            for (size_t b = 0; b < HASH_BYTES; ++b)
                mix[b] = lc.data[prev][b] ^ lc.data[src_idx][b];

            lc.data[i] = keccak512(mix);
        }
    }

    return lc;
}

/* ---------- single DAG item ---------- */

hash512 calc_dag_item(const LightCache& cache, uint32_t index) {
    const uint32_t num_words = HASH_BYTES / 4;  /* 16 uint32s per node */

    /* initial mix = cache[index % num_items] XOR index */
    hash512 mix = cache.data[index % cache.num_items];
    uint32_t first_word;
    std::memcpy(&first_word, mix.data(), 4);
    first_word ^= index;
    std::memcpy(mix.data(), &first_word, 4);
    mix = keccak512(mix);

    /* aggregation using FNV */
    for (uint32_t p = 0; p < DATASET_PARENTS; ++p) {
        uint32_t mix_word;
        std::memcpy(&mix_word, mix.data() + (p % num_words) * 4, 4);
        uint32_t parent_idx = fnv1(index ^ p, mix_word) % cache.num_items;

        const hash512& parent = cache.data[parent_idx];
        for (uint32_t w = 0; w < num_words; ++w) {
            uint32_t mw, pw;
            std::memcpy(&mw, mix.data() + w * 4, 4);
            std::memcpy(&pw, parent.data() + w * 4, 4);
            uint32_t res = fnv1(mw, pw);
            std::memcpy(mix.data() + w * 4, &res, 4);
        }
    }

    return keccak512(mix);
}

} // namespace kawpow
