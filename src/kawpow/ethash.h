/*
 * ethash.h – Ethash-style DAG generation adapted for Ravencoin (KawPow).
 *
 * Ravencoin uses epoch_length = 7500 (vs Ethash's 30000).
 * The light-cache and full-dataset generation follow the original Ethash spec
 * with the Ravencoin epoch length.
 */
#pragma once

#include "keccak.h"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace kawpow {

/* Ravencoin KawPow constants */
constexpr uint32_t EPOCH_LENGTH        = 7500;
constexpr uint32_t LIGHT_CACHE_ROUNDS  = 3;
constexpr uint32_t DATASET_PARENTS     = 512;
constexpr uint32_t HASH_BYTES          = 64;   /* keccak-512 output */
constexpr uint32_t MIX_BYTES           = 256;

/* Size helpers */
uint64_t light_cache_size(uint32_t epoch);
uint64_t full_dataset_size(uint32_t epoch);
uint32_t epoch_from_block(uint64_t block_number);

/* Seed hash: keccak-256 applied `epoch` times to 32 zero bytes */
hash256 seed_hash(uint32_t epoch);

/*
 * Light cache: a vector of keccak-512 hashes produced from the seed hash.
 * Used to compute individual DAG items on-the-fly.
 */
struct LightCache {
    uint32_t epoch;
    uint64_t cache_size;     /* bytes */
    uint32_t num_items;      /* number of 64-byte nodes */
    std::vector<hash512> data;
};

LightCache make_light_cache(uint32_t epoch);

/*
 * Compute a single DAG item from the light cache.
 * This is the core function used by both CPU verification and GPU DAG build.
 */
hash512 calc_dag_item(const LightCache& cache, uint32_t index);

} // namespace kawpow
