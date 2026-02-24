/*
 * kawpow_cuda.cu – KawPow (ProgPoW) CUDA kernel for Ravencoin mining
 *
 * Implements the KawPow algorithm as specified for Ravencoin:
 *   https://github.com/RavenProject/Ravencoin/blob/master/src/crypto/kawpow
 *
 * KawPow is ProgPoW with the following parameters:
 *   PROGPOW_PERIOD      =  3   (program changes every 3 blocks)
 *   PROGPOW_LANES       = 16
 *   PROGPOW_REGS        = 32
 *   PROGPOW_DAG_LOADS   =  4
 *   PROGPOW_CACHE_BYTES = 16 * 1024 (16 KB L1 scratch)
 *   PROGPOW_CNT_CACHE   = 11
 *   PROGPOW_CNT_MATH    = 18
 *   EPOCH_LENGTH        = 7500 blocks
 */

#include <stdint.h>
#include <cuda_runtime.h>

/* =========================================================================
 * Algorithm constants
 * ========================================================================= */
#define PROGPOW_PERIOD       3u
#define PROGPOW_LANES        16u
#define PROGPOW_REGS         32u
#define PROGPOW_DAG_LOADS    4u
#define PROGPOW_CACHE_BYTES  (16u * 1024u)
#define PROGPOW_CNT_CACHE    11u
#define PROGPOW_CNT_MATH     18u
#define PROGPOW_CNT_MEM      PROGPOW_DAG_LOADS

#define NODE_WORDS           16u   /* 64-byte DAG node = 16 x uint32 */
#define EPOCH_LENGTH         7500u

#define FNV_PRIME            0x01000193u
#define FNV_OFFSET           0x811c9dc5u

/* =========================================================================
 * Keccak-f[800] – the 800-bit variant used by ProgPoW
 * ========================================================================= */
__device__ __constant__ uint32_t keccak_rc[22] = {
    0x00000001u, 0x00008082u, 0x0000808Au, 0x80008000u,
    0x0000808Bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008Au, 0x00000088u, 0x80008009u, 0x8000000Au,
    0x8000808Bu, 0x0000008Bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800Au, 0x8000000Au,
    0x80008081u, 0x00008080u
};

__device__ void keccak_f800(uint32_t st[25])
{
    /* rotation offsets (Rho) */
    const uint32_t rotc[24] = {
         1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
        27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
    };
    /* Pi permutation */
    const uint32_t piln[24] = {
        10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
        15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
    };

    for (int round = 0; round < 22; round++) {
        uint32_t bc[5], tmp;

        /* Theta */
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];
        for (int i = 0; i < 5; i++) {
            tmp = bc[(i+4)%5] ^ __funnelshift_l(bc[(i+1)%5], bc[(i+1)%5], 1);
            for (int j = 0; j < 25; j += 5)
                st[j+i] ^= tmp;
        }

        /* Rho + Pi */
        tmp = st[1];
        for (int i = 0; i < 24; i++) {
            int j  = piln[i];
            bc[0]  = st[j];
            st[j]  = __funnelshift_l(tmp, tmp, rotc[i] % 32);
            tmp    = bc[0];
        }

        /* Chi */
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) bc[i] = st[j+i];
            for (int i = 0; i < 5; i++)
                st[j+i] ^= (~bc[(i+1)%5]) & bc[(i+2)%5];
        }

        /* Iota */
        st[0] ^= keccak_rc[round];
    }
}

/*
 * Short keccak – load header_hash (8 words) + nonce (2 words), run f800,
 * return first 8 words as seed.
 */
__device__ void keccak_f800_short(
    const uint32_t header[8],
    const uint64_t nonce,
    uint32_t       seed_out[8])
{
    uint32_t st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    for (int i = 0; i <  8; i++) st[i] = header[i];
    st[8]  = (uint32_t)(nonce & 0xFFFFFFFFu);
    st[9]  = (uint32_t)(nonce >> 32);
    keccak_f800(st);
    for (int i = 0; i < 8; i++) seed_out[i] = st[i];
}

/*
 * Long keccak – produce the final result hash from
 * header_hash (8w) + seed (8w) + mix_hash (8w).
 */
__device__ void keccak_f800_long(
    const uint32_t header[8],
    const uint32_t seed[8],
    const uint32_t mix_hash[8],
    uint32_t       result[8])
{
    uint32_t st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    for (int i = 0; i <  8; i++) st[i]      = header[i];
    for (int i = 0; i <  8; i++) st[8  + i] = seed[i];
    for (int i = 0; i <  8; i++) st[16 + i] = mix_hash[i];
    keccak_f800(st);
    for (int i = 0; i < 8; i++) result[i] = st[i];
}

/* =========================================================================
 * KISS99 PRNG – used to generate the per-period random program
 * ========================================================================= */
typedef struct { uint32_t z, w, jsr, jcong; } kiss99_t;

__device__ uint32_t kiss99_next(kiss99_t *k)
{
    k->z     = 36969u * (k->z    & 65535u) + (k->z    >> 16);
    k->w     = 18000u * (k->w    & 65535u) + (k->w    >> 16);
    uint32_t MWC = (k->z << 16) + (k->w & 65535u);
    k->jsr  ^= (k->jsr << 17);
    k->jsr  ^= (k->jsr >> 13);
    k->jsr  ^= (k->jsr <<  5);
    k->jcong = 69069u * k->jcong + 1234567u;
    return (MWC ^ k->jcong) + k->jsr;
}

/* =========================================================================
 * FNV helpers
 * ========================================================================= */
__device__ __forceinline__ uint32_t fnv1a(uint32_t h, uint32_t d)
{
    return (h ^ d) * FNV_PRIME;
}

/* =========================================================================
 * ProgPoW math operations (11 variants, selected by r % 11)
 * ========================================================================= */
__device__ __forceinline__ uint32_t progpow_math(uint32_t a, uint32_t b, uint32_t r)
{
    switch (r % 11) {
        case  0: return a + b;
        case  1: return a * b;
        case  2: return (uint32_t)__umulhi(a, b);
        case  3: return min(a, b);
        case  4: return __clz(a) + __clz(b);
        case  5: return __clz(a | b);
        case  6: return __popc(a) + __popc(b);
        case  7: return a & b;
        case  8: return a | b;
        case  9: return a ^ b;
        case 10: return __byte_perm(a, b, (r >> 16) & 0x7777u);
        default: return 0;
    }
}

/* Mix/merge operation (4 variants, selected by r % 4) */
__device__ __forceinline__ uint32_t progpow_merge(uint32_t a, uint32_t b, uint32_t r)
{
    uint32_t rot = ((r >> 16) % 31u) + 1u;
    switch (r % 4) {
        case 0: return a * 33u + b;
        case 1: return (a ^ b) * 33u;
        case 2: return __funnelshift_l(a, a, rot) ^ b;
        case 3: return __funnelshift_r(a, a, rot) ^ b;
        default: return 0;
    }
}

/* =========================================================================
 * Per-period random program descriptor
 * ========================================================================= */
typedef struct {
    /* PROGPOW_CNT_MEM DAG-lane read ops */
    uint32_t dag_dst[PROGPOW_CNT_MEM];   /* destination lane index */
    uint32_t dag_src[PROGPOW_CNT_MEM];   /* source register */

    /* PROGPOW_CNT_CACHE L1 cache read ops */
    uint32_t cache_src[PROGPOW_CNT_CACHE];
    uint32_t cache_dst[PROGPOW_CNT_CACHE];
    uint32_t cache_r  [PROGPOW_CNT_CACHE];

    /* PROGPOW_CNT_MATH ALU ops */
    uint32_t math_src_a[PROGPOW_CNT_MATH];
    uint32_t math_src_b[PROGPOW_CNT_MATH];
    uint32_t math_dst  [PROGPOW_CNT_MATH];
    uint32_t math_r    [PROGPOW_CNT_MATH];
} progpow_program_t;

/* Build the random program for the given period.  Called once per period from
 * the host and passed to the kernel via constant memory. */
__host__ void build_program(uint32_t period_seed, progpow_program_t *prog)
{
    kiss99_t rng;
    rng.z     = (period_seed & 0xFFFFFFFFu) ^ FNV_OFFSET;
    rng.w     = ((period_seed >> 16) ^ 0xDEADBEEFu);
    rng.jsr   = 0x13579BDFu;
    rng.jcong = 0x24681357u;

    /* Warm up */
    for (int i = 0; i < 4; i++) kiss99_next(&rng);

    for (uint32_t i = 0; i < PROGPOW_CNT_MEM; i++) {
        prog->dag_dst[i] = kiss99_next(&rng) % PROGPOW_LANES;
        prog->dag_src[i] = kiss99_next(&rng) % PROGPOW_REGS;
    }
    for (uint32_t i = 0; i < PROGPOW_CNT_CACHE; i++) {
        prog->cache_src[i] = kiss99_next(&rng) % PROGPOW_REGS;
        prog->cache_dst[i] = kiss99_next(&rng) % PROGPOW_REGS;
        prog->cache_r[i]   = kiss99_next(&rng);
    }
    for (uint32_t i = 0; i < PROGPOW_CNT_MATH; i++) {
        prog->math_src_a[i] = kiss99_next(&rng) % PROGPOW_REGS;
        prog->math_src_b[i] = kiss99_next(&rng) % PROGPOW_REGS;
        prog->math_dst[i]   = kiss99_next(&rng) % PROGPOW_REGS;
        prog->math_r[i]     = kiss99_next(&rng);
    }
}

/* =========================================================================
 * Main KawPow kernel
 *
 * Grid:  one block  per nonce batch
 * Block: PROGPOW_LANES threads (one per lane)
 *
 * Each block evaluates a single nonce:
 *   nonce = base_nonce + blockIdx.x
 *
 * Outputs
 *   found[]       – 0/1 flag per nonce tried
 *   result_nonce  – winning nonce (if any)
 *   mix_out       – 8-word mix hash for the winning nonce
 * ========================================================================= */
__global__ void kawpow_kernel(
    const uint32_t * __restrict__ header,       /* 8 x uint32 */
    const uint32_t * __restrict__ dag,          /* full DAG in device mem */
    const uint32_t               dag_num_items, /* # of 64-byte DAG nodes */
    const uint64_t               base_nonce,
    const uint32_t               target[8],     /* 256-bit target (8 x uint32, little-endian) */
    const progpow_program_t      prog,
    uint32_t       * __restrict__ result_nonce, /* output: winning nonce lo */
    uint32_t       * __restrict__ mix_out)      /* output: mix hash (8 w) */
{
    /* Each block handles one nonce; each thread is one ProgPoW lane */
    const uint64_t nonce = base_nonce + (uint64_t)blockIdx.x;
    const uint32_t lane  = threadIdx.x; /* 0 .. PROGPOW_LANES-1 */

    /* -----------------------------------------------------------------------
     * 1.  Compute seed = keccak_f800_short(header, nonce)
     * ----------------------------------------------------------------------- */
    uint32_t seed[8];
    if (lane == 0) keccak_f800_short(header, nonce, seed);
    /* Broadcast seed to all lanes via shared memory */
    __shared__ uint32_t sh_seed[8];
    if (lane < 8) sh_seed[lane] = seed[lane];
    __syncthreads();
    for (int i = 0; i < 8; i++) seed[i] = sh_seed[i];

    /* -----------------------------------------------------------------------
     * 2.  Initialise mix registers for this lane
     *       mix[r] = fnv1a( seed[r % 8], lane * PROGPOW_REGS + r )
     * ----------------------------------------------------------------------- */
    uint32_t mix[PROGPOW_REGS];
    for (uint32_t r = 0; r < PROGPOW_REGS; r++)
        mix[r] = fnv1a(seed[r % 8], lane * PROGPOW_REGS + r);

    /* -----------------------------------------------------------------------
     * 3.  L1 scratch buffer (16 KB per block), seeded from mix[0]
     * ----------------------------------------------------------------------- */
    __shared__ uint32_t l1[PROGPOW_CACHE_BYTES / sizeof(uint32_t)];
    {
        uint32_t init = fnv1a(seed[lane % 8], lane);
        uint32_t num  = PROGPOW_CACHE_BYTES / sizeof(uint32_t);
        for (uint32_t i = lane; i < num; i += PROGPOW_LANES)
            l1[i] = fnv1a(init, i);
    }
    __syncthreads();

    /* -----------------------------------------------------------------------
     * 4.  ProgPoW mixing loop – PROGPOW_CNT_MEM rounds
     *     Each round:  one DAG read  +  PROGPOW_CNT_CACHE/CNT_MEM cache reads
     *                               +  PROGPOW_CNT_MATH/CNT_MEM math ops
     * ----------------------------------------------------------------------- */
    for (uint32_t loop = 0; loop < PROGPOW_CNT_MEM; loop++) {

        /* -- DAG read --------------------------------------------------------
         * All lanes cooperate: each contributes one 4-word slice
         * from the DAG node addressed by mix[dag_src] in the target lane.
         */
        uint32_t dag_idx;
        {
            /* The lane specified by prog.dag_dst[loop] supplies the address */
            __shared__ uint32_t sh_dag_idx;
            if (lane == prog.dag_dst[loop]) {
                uint32_t addr_reg = mix[prog.dag_src[loop]];
                sh_dag_idx = (addr_reg % dag_num_items) * NODE_WORDS;
            }
            __syncthreads();
            dag_idx = sh_dag_idx;
        }

        /* Each lane reads DAG_LOADS words from the node */
        uint32_t dag_data[PROGPOW_DAG_LOADS];
        for (uint32_t d = 0; d < PROGPOW_DAG_LOADS; d++)
            dag_data[d] = dag[dag_idx + lane % NODE_WORDS + d];

        /* Merge DAG data into mix */
        for (uint32_t d = 0; d < PROGPOW_DAG_LOADS; d++)
            mix[d] = progpow_merge(mix[d], dag_data[d],
                                   seed[loop % 8] ^ dag_data[d]);

        /* -- L1 cache reads -------------------------------------------------
         * Interleaved with DAG reads; PROGPOW_CNT_CACHE / PROGPOW_CNT_MEM
         * ops per DAG loop iteration (integer division rounds).
         */
        uint32_t cache_per_loop = PROGPOW_CNT_CACHE / PROGPOW_CNT_MEM;
        uint32_t cache_base     = loop * cache_per_loop;
        for (uint32_t c = cache_base;
             c < cache_base + cache_per_loop && c < PROGPOW_CNT_CACHE; c++) {
            uint32_t src_val = mix[prog.cache_src[c]];
            uint32_t idx     = src_val % (PROGPOW_CACHE_BYTES / sizeof(uint32_t));
            mix[prog.cache_dst[c]] =
                progpow_merge(mix[prog.cache_dst[c]], l1[idx], prog.cache_r[c]);
        }

        /* -- Math ops --------------------------------------------------------
         * PROGPOW_CNT_MATH / PROGPOW_CNT_MEM ops per DAG loop iteration.
         */
        uint32_t math_per_loop = PROGPOW_CNT_MATH / PROGPOW_CNT_MEM;
        uint32_t math_base     = loop * math_per_loop;
        for (uint32_t m = math_base;
             m < math_base + math_per_loop && m < PROGPOW_CNT_MATH; m++) {
            uint32_t r =
                progpow_math(mix[prog.math_src_a[m]],
                             mix[prog.math_src_b[m]],
                             prog.math_r[m]);
            mix[prog.math_dst[m]] =
                progpow_merge(mix[prog.math_dst[m]], r, prog.math_r[m]);
        }

        __syncthreads();
    }

    /* -----------------------------------------------------------------------
     * 5.  Reduce mix to 256-bit mix_hash (one per block, not per lane)
     *     mix_hash[i] = fnv1a over all lanes for word i
     * ----------------------------------------------------------------------- */
    __shared__ uint32_t sh_mix[PROGPOW_REGS];
    for (uint32_t r = 0; r < PROGPOW_REGS; r++) sh_mix[r] = FNV_OFFSET;
    __syncthreads();

    /* Each lane atomically folds its registers into sh_mix */
    for (uint32_t r = 0; r < PROGPOW_REGS; r++)
        atomicXor(&sh_mix[r], mix[r]);
    __syncthreads();

    /* Reduce 32 words to 8 words */
    __shared__ uint32_t sh_mix_hash[8];
    if (lane < 8) {
        uint32_t h = FNV_OFFSET;
        for (uint32_t r = lane; r < PROGPOW_REGS; r += 8)
            h = fnv1a(h, sh_mix[r]);
        sh_mix_hash[lane] = h;
    }
    __syncthreads();

    /* -----------------------------------------------------------------------
     * 6.  Compute result = keccak_f800_long(header, seed, mix_hash)
     *     Only lane 0 does the final check.
     * ----------------------------------------------------------------------- */
    if (lane == 0) {
        uint32_t result[8];
        keccak_f800_long(header, seed, sh_mix_hash, result);

        /*
         * Full 256-bit comparison: result <= target
         * Both arrays are 8 x uint32 in little-endian order, so index 7 is
         * the most significant word.  Compare from MSW downward.
         */
        bool below_target = false;
        bool decided      = false;
        for (int i = 7; i >= 0 && !decided; i--) {
            if (result[i] < target[i]) { below_target = true;  decided = true; }
            if (result[i] > target[i]) { below_target = false; decided = true; }
        }
        if (!decided) below_target = true; /* equal is a valid share */

        if (below_target) {
            atomicMin(result_nonce, (uint32_t)(nonce & 0xFFFFFFFFu));
            for (int i = 0; i < 8; i++)
                mix_out[i] = sh_mix_hash[i];
        }
    }
}
