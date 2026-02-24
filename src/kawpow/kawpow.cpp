/*
 * kawpow.cpp – CPU reference implementation of KawPow (ProgPow variant).
 *
 * This implements the random-program generation and mix function that form
 * the core of Ravencoin's proof-of-work.  The GPU implementation mirrors
 * this logic inside CUDA kernels (see cuda/kawpow_kernel.cu).
 */
#include "kawpow.h"
#include <cstring>
#include <algorithm>

namespace kawpow {

/* ---------- KISS99 PRNG (used to generate the random program) ---------- */

struct Kiss99 {
    uint32_t z, w, jsr, jcong;

    explicit Kiss99(uint64_t prog_seed) {
        z     = static_cast<uint32_t>(prog_seed);
        w     = static_cast<uint32_t>(prog_seed >> 32);
        jsr   = static_cast<uint32_t>(prog_seed);
        jcong = static_cast<uint32_t>(prog_seed >> 32);
        /* warm-up */
        for (int i = 0; i < 31; ++i) next();
    }

    uint32_t next() {
        z = 36969 * (z & 0xffff) + (z >> 16);
        w = 18000 * (w & 0xffff) + (w >> 16);
        uint32_t mwc = (z << 16) + w;
        jsr ^= (jsr << 17);
        jsr ^= (jsr >> 13);
        jsr ^= (jsr << 5);
        jcong = 69069 * jcong + 1234567;
        return (mwc ^ jcong) + jsr;
    }
};

/* ---------- FNV helpers ---------- */

static inline uint32_t fnv1a(uint32_t u, uint32_t v) {
    return (u ^ v) * 0x01000193;
}

/* ---------- ROTR32 ---------- */

static inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

/* ---------- math operations for the random program ---------- */

static uint32_t kawpow_math(uint32_t a, uint32_t b, uint32_t sel) {
    switch (sel % 11) {
        case 0:  return a + b;
        case 1:  return a * b;
        case 2:  return (uint32_t)((uint64_t)a * (uint64_t)b >> 32);
        case 3:  return (a < b) ? a : b;
        case 4:  return rotr32(a, b % 32);
        case 5:  return a & b;
        case 6:  return a | b;
        case 7:  return a ^ b;
        case 8:  return __builtin_clz(a) + __builtin_clz(b);
        case 9:  return __builtin_popcount(a) + __builtin_popcount(b);
        default: return (a < b) ? a : b;
    }
}

/* ---------- merge helper ---------- */

static uint32_t kawpow_merge(uint32_t a, uint32_t b, uint32_t sel) {
    switch (sel % 4) {
        case 0: return a * 33 + b;
        case 1: return (a ^ b) * 33;
        case 2: return rotr32(a, b % 32);
        default: return ((a << (b % 32)) | (a >> (32 - (b % 32))));
    }
}

/* ---------- core KawPow loop ---------- */

/*
 * The ProgPow mix function.  For each lane we maintain a set of 32 registers
 * and perform random math and DAG lookups directed by a PRNG seeded from the
 * block number.
 */
static void kawpow_progpow_loop(
    uint64_t block_number,
    uint32_t loop,
    uint32_t mix[KAWPOW_LANES][KAWPOW_REGS],
    const LightCache& cache,
    uint32_t full_dataset_items,
    const uint32_t* l1_cache /* per-lane */)
{
    uint64_t prog_seed = block_number / KAWPOW_PERIOD;
    Kiss99 rng(prog_seed);

    /* Generate random sequences for cache & math operations */
    uint32_t dst_seq[KAWPOW_CNT_CACHE + KAWPOW_CNT_MATH];
    uint32_t src_seq[KAWPOW_CNT_CACHE + KAWPOW_CNT_MATH];
    uint32_t sel_seq[KAWPOW_CNT_CACHE + KAWPOW_CNT_MATH];

    const uint32_t total_ops = KAWPOW_CNT_CACHE + KAWPOW_CNT_MATH;
    for (uint32_t i = 0; i < total_ops; ++i) {
        dst_seq[i] = rng.next() % KAWPOW_REGS;
        src_seq[i] = rng.next() % KAWPOW_REGS;
        sel_seq[i] = rng.next();
    }

    /* DAG access sequence */
    uint32_t dag_src_reg = rng.next() % KAWPOW_REGS;

    /* Process each lane independently */
    for (uint32_t lane = 0; lane < KAWPOW_LANES; ++lane) {
        uint32_t* r = mix[lane];

        /* process interleaved cache / math / DAG ops */
        uint32_t cache_op = 0, math_op = 0;

        for (uint32_t i = 0; i < KAWPOW_CNT_DAG; ++i) {
            /* DAG access */
            uint32_t dag_idx = r[dag_src_reg] % full_dataset_items;
            hash512 dag_item = calc_dag_item(cache, dag_idx / KAWPOW_DAG_LOADS);

            uint32_t dag_words[KAWPOW_DAG_LOADS];
            for (uint32_t w = 0; w < KAWPOW_DAG_LOADS; ++w) {
                uint32_t offset = ((dag_idx % (HASH_BYTES / 4)) + w) % (HASH_BYTES / 4);
                std::memcpy(&dag_words[w], dag_item.data() + offset * 4, 4);
            }

            /* merge DAG data */
            for (uint32_t w = 0; w < KAWPOW_DAG_LOADS; ++w) {
                uint32_t dst = rng.next() % KAWPOW_REGS;
                r[dst] = kawpow_merge(r[dst], dag_words[w], rng.next());
            }

            /* cache reads */
            if (cache_op < KAWPOW_CNT_CACHE && (i % (KAWPOW_CNT_DAG / KAWPOW_CNT_CACHE)) == 0) {
                uint32_t src = r[src_seq[cache_op]] % KAWPOW_CACHE_WORDS;
                uint32_t cache_val = l1_cache[lane * KAWPOW_CACHE_WORDS + src];
                r[dst_seq[cache_op]] = kawpow_merge(r[dst_seq[cache_op]], cache_val, sel_seq[cache_op]);
                ++cache_op;
            }

            /* math operations */
            if (math_op < KAWPOW_CNT_MATH && (i % (KAWPOW_CNT_DAG / KAWPOW_CNT_MATH)) == 0) {
                uint32_t idx = KAWPOW_CNT_CACHE + math_op;
                uint32_t a = r[src_seq[idx]];
                uint32_t b = r[dst_seq[idx]];
                r[dst_seq[idx]] = kawpow_math(a, b, sel_seq[idx]);
                ++math_op;
            }
        }
    }
}

/* ---------- public interface ---------- */

KawpowResult kawpow_hash(
    const hash256& header_hash,
    uint64_t nonce,
    uint64_t block_number,
    const LightCache& cache)
{
    uint32_t epoch = epoch_from_block(block_number);
    uint64_t ds_size = full_dataset_size(epoch);
    uint32_t full_dataset_items = static_cast<uint32_t>(ds_size / MIX_BYTES);

    /* Keccak-512 of header_hash ++ nonce */
    uint8_t seed_buf[40];
    std::memcpy(seed_buf, header_hash.data(), 32);
    uint8_t nonce_le[8];
    for (int i = 0; i < 8; ++i)
        nonce_le[i] = static_cast<uint8_t>(nonce >> (i * 8));
    std::memcpy(seed_buf + 32, nonce_le, 8);

    hash512 seed = keccak512(seed_buf, 40);

    /* Initialise mix registers from seed */
    uint32_t mix[KAWPOW_LANES][KAWPOW_REGS];
    for (uint32_t lane = 0; lane < KAWPOW_LANES; ++lane) {
        uint32_t s0;
        std::memcpy(&s0, seed.data() + 0, 4);
        s0 = fnv1a(s0, lane);

        for (uint32_t reg = 0; reg < KAWPOW_REGS; ++reg) {
            uint32_t sw;
            std::memcpy(&sw, seed.data() + (reg % 16) * 4, 4);
            mix[lane][reg] = fnv1a(s0, sw);
        }
    }

    /* Build L1 cache (small random-access cache per lane) */
    std::vector<uint32_t> l1_cache(KAWPOW_LANES * KAWPOW_CACHE_WORDS, 0);
    for (uint32_t lane = 0; lane < KAWPOW_LANES; ++lane) {
        uint32_t lane_seed;
        std::memcpy(&lane_seed, seed.data(), 4);
        lane_seed = fnv1a(lane_seed, lane);
        for (uint32_t w = 0; w < KAWPOW_CACHE_WORDS; ++w) {
            uint32_t cache_idx = (lane_seed + w) % cache.num_items;
            uint32_t word_val;
            std::memcpy(&word_val, cache.data[cache_idx].data() + (w % 16) * 4, 4);
            l1_cache[lane * KAWPOW_CACHE_WORDS + w] = word_val;
        }
    }

    /* Run the ProgPow loop */
    kawpow_progpow_loop(block_number, 0, mix, cache, full_dataset_items, l1_cache.data());

    /* Reduce mix to 256 bits (8 x uint32) */
    uint32_t digest[8] = {};
    for (uint32_t lane = 0; lane < KAWPOW_LANES; ++lane) {
        for (uint32_t i = 0; i < 8; ++i) {
            uint32_t acc = 0x811c9dc5;
            for (uint32_t r = i * (KAWPOW_REGS / 8); r < (i + 1) * (KAWPOW_REGS / 8); ++r)
                acc = fnv1a(acc, mix[lane][r]);
            digest[i] = fnv1a(digest[i] == 0 ? 0x811c9dc5 : digest[i], acc);
        }
    }

    /* Build mix_hash */
    KawpowResult result;
    std::memcpy(result.mix_hash.data(), digest, 32);

    /* Final hash = keccak-256(seed ++ mix_hash) */
    uint8_t final_buf[64 + 32];
    std::memcpy(final_buf, seed.data(), 64);
    std::memcpy(final_buf + 64, result.mix_hash.data(), 32);
    result.final_hash = keccak256(final_buf, 96);

    return result;
}

bool meets_target(const hash256& hash, const hash256& target) {
    /* Big-endian comparison */
    for (int i = 0; i < 32; ++i) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true;  /* equal */
}

} // namespace kawpow
