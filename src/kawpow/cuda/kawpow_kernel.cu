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

    //const uint32_t HASH_WORDS = 8;  /* 64 bytes / 8 bytes per uint64 */
    const uint32_t PARENTS    = 512;

    /* initial mix */
    uint64_t mix[8];
    uint32_t cache_idx = idx % cache_items;
    for (int i = 0; i < 8; ++i)
        mix[i] = cache[cache_idx * 8 + i];

    /* XOR index into first word */
    mix[0] ^= idx;

    /* keccak-512 (rate = 72 bytes = 9 uint64 words)
     * Input: 64 bytes (8 words) → padding at byte 64 (0x01) and byte 71 (0x80)
     * Both land in state[8]: low byte = 0x01, high byte = 0x80 */
    uint64_t state[25] = {};
    for (int i = 0; i < 8; ++i) state[i] = mix[i];
    state[8] = 0x8000000000000001ULL;
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

    /* final keccak-512 (same padding as above) */
    memset(state, 0, sizeof(state));
    for (int i = 0; i < 8; ++i) state[i] = mix[i];
    state[8] = 0x8000000000000001ULL;
    d_keccak_f1600(state);

    /* write result */
    for (int i = 0; i < 8; ++i)
        dag[idx * 8 + i] = state[i];
}

/* ---------- KawPow search kernel ---------- */

__device__ __forceinline__ uint32_t d_fnv1a(uint32_t u, uint32_t v) {
    return (u ^ v) * 0x01000193;
}

struct d_Kiss99 {
    uint32_t z, w, jsr, jcong;

    __device__ explicit d_Kiss99(uint64_t prog_seed) {
        z     = static_cast<uint32_t>(prog_seed);
        w     = static_cast<uint32_t>(prog_seed >> 32);
        jsr   = static_cast<uint32_t>(prog_seed);
        jcong = static_cast<uint32_t>(prog_seed >> 32);
        for (int i = 0; i < 31; ++i) next();
    }

    __device__ __forceinline__ uint32_t next() {
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
        case 10: return max(a, b);
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

/*
 * KawPow search kernel – warp-lane parallel.
 * 16 threads cooperate on one nonce (one thread per ProgPow lane).
 * Lanes communicate via __shfl_sync for the final digest reduction.
 */
__global__ void kawpow_search_kernel(
    const uint64_t* dag,
    uint32_t dag_items,
    const uint64_t* cache,
    uint32_t cache_items,
    const uint8_t* header_hash,   /* 32 bytes */
    uint64_t start_nonce,
    const uint8_t* target,        /* 32 bytes */
    uint32_t block_number,
    uint64_t* results,            /* [0]=count, [1..]=nonces */
    uint32_t max_results)
{
    const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane = global_id % kawpow::KAWPOW_LANES;       /* 0..15 */
    const uint32_t nonce_idx = global_id / kawpow::KAWPOW_LANES;
    const uint64_t nonce = start_nonce + nonce_idx;

    /* Build keccak-512 seed from header_hash ++ nonce.
     * All 16 threads in a group compute the identical seed. */
    uint64_t state[25] = {};
    memcpy(state, header_hash, 32);
    uint8_t nonce_le[8];
    for (int i = 0; i < 8; ++i)
        nonce_le[i] = (uint8_t)(nonce >> (i * 8));
    memcpy(((uint8_t*)state) + 32, nonce_le, 8);
    ((uint8_t*)state)[40] = 0x01;
    ((uint8_t*)state)[71] |= 0x80;
    d_keccak_f1600(state);

    uint32_t seed0;
    memcpy(&seed0, state, 4);

    /* ProgPow schedule (identical across all 16 lanes) */
    uint64_t prog_seed = static_cast<uint64_t>(block_number / kawpow::KAWPOW_PERIOD);
    d_Kiss99 rng(prog_seed);

    uint32_t dst_seq[kawpow::KAWPOW_CNT_CACHE + kawpow::KAWPOW_CNT_MATH];
    uint32_t src_seq[kawpow::KAWPOW_CNT_CACHE + kawpow::KAWPOW_CNT_MATH];
    uint32_t sel_seq[kawpow::KAWPOW_CNT_CACHE + kawpow::KAWPOW_CNT_MATH];

    const uint32_t total_ops = kawpow::KAWPOW_CNT_CACHE + kawpow::KAWPOW_CNT_MATH;
    for (uint32_t i = 0; i < total_ops; ++i) {
        dst_seq[i] = rng.next() % kawpow::KAWPOW_REGS;
        src_seq[i] = rng.next() % kawpow::KAWPOW_REGS;
        sel_seq[i] = rng.next();
    }

    uint32_t dag_src_reg = rng.next() % kawpow::KAWPOW_REGS;

    /* Advance RNG to this lane's starting state for DAG-merge draws.
     * In the serial version each lane consumes
     * KAWPOW_CNT_DAG * KAWPOW_DAG_LOADS * 2 sequential RNG calls. */
    const uint32_t rng_per_lane =
        kawpow::KAWPOW_CNT_DAG * kawpow::KAWPOW_DAG_LOADS * 2;  /* 64*4*2 = 512 */
    for (uint32_t i = 0; i < lane * rng_per_lane; ++i)
        rng.next();

    const uint32_t* dag32  = reinterpret_cast<const uint32_t*>(dag);
    const uint32_t* cache32 = reinterpret_cast<const uint32_t*>(cache);
    uint32_t full_dataset_items = dag_items / kawpow::KAWPOW_DAG_LOADS;
    if (full_dataset_items == 0 || cache_items == 0) return;

    /* Initialise per-lane registers */
    uint32_t r[kawpow::KAWPOW_REGS];
    uint32_t lane_seed = d_fnv1a(seed0, lane);
    for (uint32_t reg = 0; reg < kawpow::KAWPOW_REGS; ++reg) {
        uint32_t sw;
        memcpy(&sw, ((uint8_t*)state) + (reg % 16) * 4, 4);
        r[reg] = d_fnv1a(lane_seed, sw);
    }

    /* Main loop – 64 DAG accesses (each lane independent) */
    uint32_t cache_op = 0;
    uint32_t math_op  = 0;

    for (uint32_t i = 0; i < kawpow::KAWPOW_CNT_DAG; ++i) {
        uint32_t dag_idx = r[dag_src_reg] % full_dataset_items;
        uint32_t dag_node = dag_idx / kawpow::KAWPOW_DAG_LOADS;
        uint32_t dag_base = dag_idx % (kawpow::HASH_BYTES / 4);

        for (uint32_t w = 0; w < kawpow::KAWPOW_DAG_LOADS; ++w) {
            uint32_t offset = (dag_base + w) % (kawpow::HASH_BYTES / 4);
            uint32_t dag_word = dag32[static_cast<uint64_t>(dag_node) * 16ULL + offset];
            uint32_t dst = rng.next() % kawpow::KAWPOW_REGS;
            r[dst] = d_kawpow_merge(r[dst], dag_word, rng.next());
        }

        if (cache_op < kawpow::KAWPOW_CNT_CACHE &&
            (i % (kawpow::KAWPOW_CNT_DAG / kawpow::KAWPOW_CNT_CACHE)) == 0)
        {
            uint32_t src = r[src_seq[cache_op]] % kawpow::KAWPOW_CACHE_WORDS;
            uint32_t cache_idx = (lane_seed + src) % cache_items;
            uint32_t cache_val = cache32[static_cast<uint64_t>(cache_idx) * 16ULL + (src % 16)];
            r[dst_seq[cache_op]] = d_kawpow_merge(r[dst_seq[cache_op]],
                                                  cache_val,
                                                  sel_seq[cache_op]);
            ++cache_op;
        }

        if (math_op < kawpow::KAWPOW_CNT_MATH &&
            (i % (kawpow::KAWPOW_CNT_DAG / kawpow::KAWPOW_CNT_MATH)) == 0)
        {
            uint32_t idx = kawpow::KAWPOW_CNT_CACHE + math_op;
            uint32_t a = r[src_seq[idx]];
            uint32_t b = r[dst_seq[idx]];
            r[dst_seq[idx]] = d_kawpow_math(a, b, sel_seq[idx]);
            ++math_op;
        }
    }

    /* -------- digest reduction -------- */
    /* Per-lane: reduce 32 regs → 8 words */
    uint32_t lane_acc[8];
    for (uint32_t i = 0; i < 8; ++i) {
        uint32_t acc = 0x811c9dc5;
        for (uint32_t reg = i * (kawpow::KAWPOW_REGS / 8);
             reg < (i + 1) * (kawpow::KAWPOW_REGS / 8);
             ++reg)
            acc = d_fnv1a(acc, r[reg]);
        lane_acc[i] = acc;
    }

    /* Cross-lane: combine 16 lanes sequentially via warp shuffle.
     * __shfl_sync with width=16 partitions each warp into two
     * independent 16-thread groups – exactly matching our lane groups. */
    uint32_t digest[8];
    for (uint32_t i = 0; i < 8; ++i) {
        uint32_t d = 0;
        for (uint32_t l = 0; l < kawpow::KAWPOW_LANES; ++l) {
            uint32_t val = __shfl_sync(0xFFFFFFFF, lane_acc[i], l,
                                       kawpow::KAWPOW_LANES);
            d = d_fnv1a(d == 0 ? 0x811c9dc5u : d, val);
        }
        digest[i] = d;
    }

    /* Only lane 0 performs the final hash and target check */
    if (lane != 0) return;

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
    auto cuda_ok = [](cudaError_t err, const char* call, int line) {
        if (err == cudaSuccess) return true;
        fprintf(stderr, "CUDA error %s:%d in %s: %s\n",
                __FILE__, line, call, cudaGetErrorString(err));
        return false;
    };

#define CUDA_CHECK_CLEAN(call) do { if (!cuda_ok((call), #call, __LINE__)) goto cleanup; } while (0)

    uint64_t* d_cache = nullptr;
    uint64_t* d_dag = nullptr;
    uint8_t* d_header = nullptr;
    uint8_t* d_target = nullptr;
    uint64_t* d_results = nullptr;
    size_t cache_bytes = 0;
    uint32_t epoch = 0;
    uint64_t ds_size = 0;
    uint32_t dag_items = 0;
    int threads_per_block = 256;
    int dag_blocks = 0;
    const uint32_t MAX_RESULTS = 4;
    const uint32_t BATCH = 1u << 20;
    uint64_t nonce = start_nonce;
    int search_blocks = 0;

    CUDA_CHECK_CLEAN(cudaSetDevice(gpu_id));

    /* upload light cache */
    cache_bytes = cache.num_items * 64;
    CUDA_CHECK_CLEAN(cudaMalloc(&d_cache, cache_bytes));
    CUDA_CHECK_CLEAN(cudaMemcpy(d_cache, cache.data.data(), cache_bytes, cudaMemcpyHostToDevice));

    /* allocate DAG on GPU */
    epoch = kawpow::epoch_from_block(block_number);
    ds_size = kawpow::full_dataset_size(epoch);
    dag_items = static_cast<uint32_t>(ds_size / kawpow::HASH_BYTES);

    CUDA_CHECK_CLEAN(cudaMalloc(&d_dag, ds_size));

    /* generate DAG */
    fprintf(stderr, "[GPU] Generating DAG (%llu MiB) ...\n",
            (unsigned long long)(ds_size / (1024 * 1024)));

    dag_blocks = (dag_items + threads_per_block - 1) / threads_per_block;
    dag_generate_kernel<<<dag_blocks, threads_per_block>>>(
        d_dag, d_cache, cache.num_items, dag_items);
    CUDA_CHECK_CLEAN(cudaDeviceSynchronize());
    fprintf(stderr, "[GPU] DAG ready\n");

    /* upload header & target */
    CUDA_CHECK_CLEAN(cudaMalloc(&d_header, 32));
    CUDA_CHECK_CLEAN(cudaMalloc(&d_target, 32));
    CUDA_CHECK_CLEAN(cudaMemcpy(d_header, header_hash.data(), 32, cudaMemcpyHostToDevice));
    CUDA_CHECK_CLEAN(cudaMemcpy(d_target, target.data(), 32, cudaMemcpyHostToDevice));

    /* results buffer */
    CUDA_CHECK_CLEAN(cudaMalloc(&d_results, (1 + MAX_RESULTS) * sizeof(uint64_t)));

    /* mining loop – 16 threads per nonce (one per KawPow lane) */
    search_blocks = (BATCH * kawpow::KAWPOW_LANES + threads_per_block - 1)
                    / threads_per_block;

    while (running) {
        CUDA_CHECK_CLEAN(cudaMemset(d_results, 0, sizeof(uint64_t)));

        kawpow_search_kernel<<<search_blocks, threads_per_block>>>(
            d_dag, dag_items, d_cache, cache.num_items,
            d_header, nonce, d_target,
            static_cast<uint32_t>(block_number),
            d_results, MAX_RESULTS);
        CUDA_CHECK_CLEAN(cudaDeviceSynchronize());

        hash_count += BATCH;
        nonce += BATCH;

        /* check for solutions */
        uint64_t h_results[1 + MAX_RESULTS] = {};
        CUDA_CHECK_CLEAN(cudaMemcpy(h_results, d_results, sizeof(h_results), cudaMemcpyDeviceToHost));

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

cleanup:
    if (d_results) cudaFree(d_results);
    if (d_target) cudaFree(d_target);
    if (d_header) cudaFree(d_header);
    if (d_dag) cudaFree(d_dag);
    if (d_cache) cudaFree(d_cache);

#undef CUDA_CHECK_CLEAN
}
