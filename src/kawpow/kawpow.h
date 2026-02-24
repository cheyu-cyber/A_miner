/*
 * kawpow.h – KawPow (ProgPow variant) mining algorithm for Ravencoin.
 *
 * KawPow is Ravencoin's proof-of-work algorithm.  It is a ProgPow variant
 * that generates a random program per period (every KAWPOW_PERIOD blocks)
 * to resist ASIC optimisation.
 */
#pragma once

#include "ethash.h"
#include "keccak.h"
#include <cstdint>
#include <array>
#include <vector>

namespace kawpow {

/* ProgPow / KawPow tuning constants */
constexpr uint32_t KAWPOW_PERIOD   = 3;       /* new program every 3 blocks */
constexpr uint32_t KAWPOW_LANES    = 16;
constexpr uint32_t KAWPOW_REGS     = 32;
constexpr uint32_t KAWPOW_DAG_LOADS = 4;
constexpr uint32_t KAWPOW_CNT_CACHE = 11;
constexpr uint32_t KAWPOW_CNT_MATH  = 18;
constexpr uint32_t KAWPOW_CNT_DAG   = 64;     /* number of DAG accesses */
constexpr uint32_t KAWPOW_CACHE_WORDS = 4096;  /* 16 KiB L1 cache per lane */

/* Result of a KawPow hash computation */
struct KawpowResult {
    hash256 final_hash;  /* the hash compared against the target */
    hash256 mix_hash;    /* submitted alongside the nonce */
};

/*
 * Compute the KawPow hash for a given header, nonce, and block height.
 *
 *   header_hash : keccak-256 of the block header (without nonce)
 *   nonce       : 64-bit nonce
 *   block_number: current block height  (determines epoch & program seed)
 *   cache       : pre-built light cache for the epoch
 *
 * For GPU mining this function is implemented in CUDA; this CPU version
 * serves as a reference / verification implementation.
 */
KawpowResult kawpow_hash(
    const hash256& header_hash,
    uint64_t nonce,
    uint64_t block_number,
    const LightCache& cache);

/*
 * Check whether a result meets the target difficulty.
 * target is a 256-bit big-endian number.
 */
bool meets_target(const hash256& hash, const hash256& target);

} // namespace kawpow
