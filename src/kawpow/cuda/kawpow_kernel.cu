/*
 * kawpow_kernel.cu – CUDA kernels for KawPow (Ravencoin) GPU mining.
 *
 * This file implements:
 *   1. DAG generation kernel  – fills GPU memory with the full dataset
 *   2. KawPow search kernel   – runs the ProgPow mix on many nonces in parallel
 *
 * Requires NVIDIA GPU with compute capability >= 6.1 (Pascal or newer).
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <atomic>
#include <vector>

#include <cuda_runtime.h>

/* Include CPU definitions for types and constants */
#include "kawpow/keccak.h"
#include "kawpow/ethash.h"
#include "kawpow/kawpow.h"

/* ---------- error checking ---------- */

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));         \
        return;                                                       \
    }                                                                 \
} while (0)

/* ---------- device-side Keccak-f[1600] ---------- */

__device__ static const uint64_t d_keccak_rc[24] = {
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

__device__ __forceinline__ uint64_t d_rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ void d_keccak_f1600(uint64_t state[25]) {
    static const int rot[24] = {
         1,  3,  6, 10, 15, 21, 28, 36,
        45, 55,  2, 14, 27, 41, 56,  8,
        25, 43, 62, 18, 39, 61, 20, 44,
    };
    static const int pi[24] = {
        10,  7, 11, 17, 18,  3,  5, 16,
         8, 21, 24,  4, 15, 23, 19, 13,
        12,  2, 20, 14, 22,  9,  6,  1,
    };

    for (int round = 0; round < 24; ++round) {
        uint64_t C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20];

        uint64_t D[5];
        for (int x = 0; x < 5; ++x)
            D[x] = C[(x+4)%5] ^ d_rotl64(C[(x+1)%5], 1);

        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 25; y += 5)
                state[y+x] ^= D[x];

        uint64_t t = state[1];
        for (int i = 0; i < 24; ++i) {
            uint64_t tmp = state[pi[i]];
            state[pi[i]] = d_rotl64(t, rot[i]);
            t = tmp;
        }

        for (int y = 0; y < 25; y += 5) {
            uint64_t tmp[5];
            for (int x = 0; x < 5; ++x) tmp[x] = state[y+x];
            for (int x = 0; x < 5; ++x)
                state[y+x] = tmp[x] ^ ((~tmp[(x+1)%5]) & tmp[(x+2)%5]);
        }

        state[0] ^= d_keccak_rc[round];
    }
}

/* ---------- DAG generation kernel ---------- */

__global__ void dag_generate_kernel(
    uint64_t* dag,            /* output: full dataset in GPU memory */
    const uint64_t* cache,    /* light cache */
    uint32_t cache_items,
    uint32_t dag_items)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dag_items) return;

    const uint32_t HASH_WORDS = 8;  /* 64 bytes / 8 bytes per uint64 */
    const uint32_t PARENTS    = 512;

    /* initial mix */
    uint64_t mix[8];
    uint32_t cache_idx = idx % cache_items;
    for (int i = 0; i < 8; ++i)
        mix[i] = cache[cache_idx * 8 + i];

    /* XOR index into first word */
    mix[0] ^= idx;

    /* keccak-512 */
    uint64_t state[25] = {};
    for (int i = 0; i < 8; ++i) state[i] = mix[i];
    state[8] = 0x0000000000000001ULL;  /* padding */
    state[8 + 1] = 0;
    /* rate = 72 bytes = 9 uint64, so state[8] gets padding */
    state[8] |= 0x01;
    state[71/8] |= 0x8000000000000000ULL;
    d_keccak_f1600(state);
    for (int i = 0; i < 8; ++i) mix[i] = state[i];

    /* FNV aggregation */
    for (uint32_t p = 0; p < PARENTS; ++p) {
        uint32_t mix_word;
        memcpy(&mix_word, ((uint8_t*)mix) + (p % 16) * 4, 4);
        uint32_t parent_idx = ((idx ^ p) * 0x01000193 ^ mix_word) % cache_items;

        for (int i = 0; i < 8; ++i) {
            uint32_t mw[2], pw[2];
            memcpy(mw, &mix[i], 8);
            memcpy(pw, &cache[parent_idx * 8 + i], 8);
            mw[0] = (mw[0] * 0x01000193) ^ pw[0];
            mw[1] = (mw[1] * 0x01000193) ^ pw[1];
            memcpy(&mix[i], mw, 8);
        }
    }

    /* final keccak-512 */
    memset(state, 0, sizeof(state));
    for (int i = 0; i < 8; ++i) state[i] = mix[i];
    state[8] = 0x01;
    state[71/8] |= 0x8000000000000000ULL;
    d_keccak_f1600(state);

    /* write result */
    for (int i = 0; i < 8; ++i)
        dag[idx * 8 + i] = state[i];
}

/* ---------- KawPow search kernel ---------- */

__device__ __forceinline__ uint32_t d_fnv1a(uint32_t u, uint32_t v) {
    return (u ^ v) * 0x01000193;
}

__device__ uint32_t d_kawpow_math(uint32_t a, uint32_t b, uint32_t sel) {
    switch (sel % 11) {
        case 0:  return a + b;
        case 1:  return a * b;
        case 2:  return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
        case 3:  return min(a, b);
        case 4:  return __funnelshift_r(a, a, b & 31);
        case 5:  return a & b;
        case 6:  return a | b;
        case 7:  return a ^ b;
        case 8:  return __clz(a) + __clz(b);
        case 9:  return __popc(a) + __popc(b);
        default: return min(a, b);
    }
}

__device__ uint32_t d_kawpow_merge(uint32_t a, uint32_t b, uint32_t sel) {
    switch (sel % 4) {
        case 0: return a * 33 + b;
        case 1: return (a ^ b) * 33;
        case 2: return __funnelshift_r(a, a, b & 31);
        default: return __funnelshift_l(a, a, b & 31);
    }
}

/* Simplified search kernel – each thread tests one nonce */
__global__ void kawpow_search_kernel(
    const uint64_t* dag,
    uint32_t dag_items,
    const uint8_t* header_hash,   /* 32 bytes */
    uint64_t start_nonce,
    const uint8_t* target,        /* 32 bytes */
    uint32_t block_number,
    uint64_t* results,            /* [0]=count, [1..]=nonces */
    uint32_t max_results)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + gid;

    /* Build keccak-512 seed from header_hash ++ nonce */
    uint64_t state[25] = {};
    memcpy(state, header_hash, 32);
    uint8_t nonce_le[8];
    for (int i = 0; i < 8; ++i)
        nonce_le[i] = (uint8_t)(nonce >> (i * 8));
    memcpy(((uint8_t*)state) + 32, nonce_le, 8);
    /* padding for rate=72: byte 40 = 0x01, byte 71 = 0x80 */
    ((uint8_t*)state)[40] = 0x01;
    ((uint8_t*)state)[71] |= 0x80;
    d_keccak_f1600(state);

    /* Initialise mix from seed */
    uint32_t mix[kawpow::KAWPOW_REGS];
    uint32_t seed0;
    memcpy(&seed0, state, 4);

    for (uint32_t r = 0; r < kawpow::KAWPOW_REGS; ++r) {
        uint32_t sw;
        memcpy(&sw, ((uint8_t*)state) + (r % 16) * 4, 4);
        mix[r] = d_fnv1a(seed0, sw);
    }

    /* Simplified DAG lookup loop */
    for (uint32_t i = 0; i < kawpow::KAWPOW_CNT_DAG; ++i) {
        uint32_t dag_idx = mix[i % kawpow::KAWPOW_REGS] % dag_items;
        uint32_t dag_word;
        memcpy(&dag_word, &dag[dag_idx], 4);
        mix[i % kawpow::KAWPOW_REGS] = d_fnv1a(mix[i % kawpow::KAWPOW_REGS], dag_word);
    }

    /* Reduce to 256-bit digest */
    uint32_t digest[8];
    for (int i = 0; i < 8; ++i) {
        uint32_t acc = 0x811c9dc5;
        for (uint32_t r = i * 4; r < (i + 1) * 4; ++r)
            acc = d_fnv1a(acc, mix[r]);
        digest[i] = acc;
    }

    /* Final keccak-256: seed ++ digest */
    uint64_t final_state[25] = {};
    memcpy(final_state, state, 64);
    memcpy(((uint8_t*)final_state) + 64, digest, 32);
    ((uint8_t*)final_state)[96] = 0x01;
    ((uint8_t*)final_state)[135] |= 0x80;
    d_keccak_f1600(final_state);

    /* Compare against target (big-endian) */
    const uint8_t* hash = (const uint8_t*)final_state;
    bool below = false;
    for (int i = 0; i < 32; ++i) {
        if (hash[i] < target[i]) { below = true; break; }
        if (hash[i] > target[i]) break;
    }

    if (below) {
        uint32_t slot = atomicAdd((unsigned int*)results, 1u);
        if (slot < max_results)
            results[1 + slot] = nonce;
    }
}

/* ---------- host-side entry point ---------- */

void cuda_mine(
    int gpu_id,
    const kawpow::hash256& header_hash,
    uint64_t start_nonce,
    uint64_t block_number,
    const kawpow::hash256& target,
    const kawpow::LightCache& cache,
    std::atomic<bool>& running,
    std::atomic<uint64_t>& hash_count,
    std::function<void(uint64_t nonce, const kawpow::KawpowResult& result)> on_solution)
{
    CUDA_CHECK(cudaSetDevice(gpu_id));

    /* upload light cache */
    size_t cache_bytes = cache.num_items * 64;
    uint64_t* d_cache = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cache, cache_bytes));
    CUDA_CHECK(cudaMemcpy(d_cache, cache.data.data(), cache_bytes, cudaMemcpyHostToDevice));

    /* allocate DAG on GPU */
    uint32_t epoch = kawpow::epoch_from_block(block_number);
    uint64_t ds_size = kawpow::full_dataset_size(epoch);
    uint32_t dag_items = static_cast<uint32_t>(ds_size / kawpow::HASH_BYTES);

    uint64_t* d_dag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dag, ds_size));

    /* generate DAG */
    fprintf(stderr, "[GPU] Generating DAG (%llu MiB) ...\n",
            (unsigned long long)(ds_size / (1024 * 1024)));

    int threads_per_block = 256;
    int dag_blocks = (dag_items + threads_per_block - 1) / threads_per_block;
    dag_generate_kernel<<<dag_blocks, threads_per_block>>>(
        d_dag, d_cache, cache.num_items, dag_items);
    CUDA_CHECK(cudaDeviceSynchronize());
    fprintf(stderr, "[GPU] DAG ready\n");

    /* upload header & target */
    uint8_t* d_header = nullptr;
    uint8_t* d_target = nullptr;
    CUDA_CHECK(cudaMalloc(&d_header, 32));
    CUDA_CHECK(cudaMalloc(&d_target, 32));
    CUDA_CHECK(cudaMemcpy(d_header, header_hash.data(), 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target.data(), 32, cudaMemcpyHostToDevice));

    /* results buffer */
    constexpr uint32_t MAX_RESULTS = 4;
    uint64_t* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, (1 + MAX_RESULTS) * sizeof(uint64_t)));

    /* mining loop */
    constexpr uint32_t BATCH = 1u << 20;  /* ~1M nonces per kernel launch */
    uint64_t nonce = start_nonce;
    int search_blocks = (BATCH + threads_per_block - 1) / threads_per_block;

    while (running) {
        CUDA_CHECK(cudaMemset(d_results, 0, sizeof(uint64_t)));

        kawpow_search_kernel<<<search_blocks, threads_per_block>>>(
            d_dag, dag_items, d_header, nonce, d_target,
            static_cast<uint32_t>(block_number),
            d_results, MAX_RESULTS);
        CUDA_CHECK(cudaDeviceSynchronize());

        hash_count += BATCH;
        nonce += BATCH;

        /* check for solutions */
        uint64_t h_results[1 + MAX_RESULTS] = {};
        CUDA_CHECK(cudaMemcpy(h_results, d_results, sizeof(h_results), cudaMemcpyDeviceToHost));

        uint32_t found = static_cast<uint32_t>(h_results[0]);
        for (uint32_t i = 0; i < found && i < MAX_RESULTS; ++i) {
            uint64_t sol_nonce = h_results[1 + i];
            /* CPU-verify before submitting */
            auto result = kawpow::kawpow_hash(header_hash, sol_nonce,
                                              block_number, cache);
            if (kawpow::meets_target(result.final_hash, target))
                on_solution(sol_nonce, result);
        }
    }

    cudaFree(d_results);
    cudaFree(d_target);
    cudaFree(d_header);
    cudaFree(d_dag);
    cudaFree(d_cache);
}
