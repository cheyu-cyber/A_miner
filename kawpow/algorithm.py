"""
kawpow/algorithm.py – KawPow algorithm: DAG generation and CUDA dispatch.

KawPow is ProgPoW adapted for Ravencoin with:
    EPOCH_LENGTH  = 7500 blocks
    PROGPOW_PERIOD = 3 blocks
    PROGPOW_LANES = 16
    PROGPOW_REGS  = 32
"""

import hashlib
import logging
import os
import struct
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("A_miner.kawpow")

# ---------------------------------------------------------------------------
# Algorithm constants
# ---------------------------------------------------------------------------
EPOCH_LENGTH     = 7500
PROGPOW_PERIOD   = 3
PROGPOW_LANES    = 16
PROGPOW_REGS     = 32
PROGPOW_DAG_LOADS = 4
PROGPOW_CACHE_BYTES = 16 * 1024
PROGPOW_CNT_CACHE = 11
PROGPOW_CNT_MATH  = 18
PROGPOW_CNT_MEM   = PROGPOW_DAG_LOADS

# Ethash DAG sizing
DATASET_BYTES_INIT   = 1 << 30    # 1 GB
DATASET_BYTES_GROWTH = 1 << 23    # 8 MB per epoch
CACHE_BYTES_INIT     = 1 << 24    # 16 MB
CACHE_BYTES_GROWTH   = 1 << 17    # 128 KB per epoch
MIX_BYTES            = 128
HASH_BYTES           = 64         # 512-bit node
DATASET_PARENTS      = 256
CACHE_ROUNDS         = 3
NODE_WORDS           = HASH_BYTES // 4   # 16 uint32 per node

FNV_PRIME  = 0x01000193
FNV_OFFSET = 0x811c9dc5
UINT32_MOD = 2**32


def epoch_for_block(block_number: int) -> int:
    """Return the epoch number for a given block height."""
    return block_number // EPOCH_LENGTH


# ---------------------------------------------------------------------------
# Keccak-512 and Keccak-256 (raw, not SHA3)
# ---------------------------------------------------------------------------
def _keccak_512(data: bytes) -> bytes:
    """Return 64-byte raw Keccak-512 digest (not SHA3-512)."""
    try:
        import sha3  # pysha3
    except ImportError:
        raise ImportError(
            "pysha3 is required for correct Keccak-512 (raw, not SHA3).\n"
            "Install it with: pip install pysha3"
        )
    k = sha3.keccak_512()
    k.update(data)
    return k.digest()


def _keccak_256(data: bytes) -> bytes:
    """Return 32-byte raw Keccak-256 digest."""
    try:
        import sha3
    except ImportError:
        raise ImportError(
            "pysha3 is required for correct Keccak-256 (raw, not SHA3).\n"
            "Install it with: pip install pysha3"
        )
    k = sha3.keccak_256()
    k.update(data)
    return k.digest()


# ---------------------------------------------------------------------------
# Seed hash for epoch
# ---------------------------------------------------------------------------
def seed_hash_for_epoch(epoch: int) -> bytes:
    """Compute the 32-byte seed hash for the given epoch."""
    seed = b"\x00" * 32
    for _ in range(epoch):
        seed = _keccak_256(seed)
    return seed


# ---------------------------------------------------------------------------
# Ethash-style cache generation
# ---------------------------------------------------------------------------
def _fnv(x: int, y: int) -> int:
    return ((x * FNV_PRIME) ^ y) % UINT32_MOD


def get_cache_size(epoch: int) -> int:
    """Return the number of bytes in the cache for this epoch."""
    size = CACHE_BYTES_INIT + CACHE_BYTES_GROWTH * epoch
    # Round down to nearest prime number of items
    size -= HASH_BYTES
    while not _is_prime(size // HASH_BYTES):
        size -= 2 * HASH_BYTES
    return size


def get_full_size(epoch: int) -> int:
    """Return the number of bytes in the full DAG for this epoch."""
    size = DATASET_BYTES_INIT + DATASET_BYTES_GROWTH * epoch
    size -= MIX_BYTES
    while not _is_prime(size // MIX_BYTES):
        size -= 2 * MIX_BYTES
    return size


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def make_cache(cache_size: int, seed: bytes) -> np.ndarray:
    """
    Build the Ethash verification cache.

    Returns a uint32 numpy array of shape (n_nodes, NODE_WORDS).
    """
    n = cache_size // HASH_BYTES
    logger.debug("Building cache: %d nodes (%.1f MB)", n, n * HASH_BYTES / 1e6)

    # Seed the cache
    cache = [_keccak_512(seed)]
    for i in range(1, n):
        cache.append(_keccak_512(cache[-1]))

    # RandMemo-mix rounds
    for _ in range(CACHE_ROUNDS):
        for i in range(n):
            v = struct.unpack_from("<I", cache[i], 0)[0] % n
            data = bytes(a ^ b for a, b in zip(cache[(i - 1) % n], cache[v]))
            cache[i] = _keccak_512(data)

    # Convert to uint32 array
    arr = np.frombuffer(b"".join(cache), dtype=np.uint32).copy()
    return arr.reshape(n, NODE_WORDS)


def make_dag_item(cache: np.ndarray, i: int) -> np.ndarray:
    """Compute a single 512-bit DAG item from the cache."""
    n = len(cache)
    mix = cache[i % n].copy()
    mix[0] ^= np.uint32(i)
    mix = np.frombuffer(_keccak_512(mix.tobytes()), dtype=np.uint32)

    for j in range(DATASET_PARENTS):
        cache_idx = _fnv(i ^ j, int(mix[j % NODE_WORDS])) % n
        mix = np.array(
            [np.uint32(_fnv(int(a), int(b)))
             for a, b in zip(mix, cache[cache_idx])],
            dtype=np.uint32,
        )

    return np.frombuffer(_keccak_512(mix.tobytes()), dtype=np.uint32)


def build_dag(block_number: int) -> Tuple[np.ndarray, int]:
    """
    Build the full DAG for the epoch corresponding to *block_number*.

    Returns (dag_array, dag_num_items) where dag_array is a uint32 numpy
    array of shape (dag_num_items, NODE_WORDS).

    Building the DAG is slow (minutes) and memory-heavy (1 GB+).
    """
    epoch = epoch_for_block(block_number)
    seed  = seed_hash_for_epoch(epoch)
    cache_size = get_cache_size(epoch)
    dag_size   = get_full_size(epoch)
    n_cache    = cache_size // HASH_BYTES
    n_dag      = dag_size   // HASH_BYTES

    logger.info("Epoch %d  cache %.0f MB  DAG %.1f GB",
                epoch, n_cache * HASH_BYTES / 1e6,
                n_dag * HASH_BYTES / 1e9)

    logger.info("Building cache (%d nodes)…", n_cache)
    cache = make_cache(cache_size, seed)

    logger.info("Building DAG (%d nodes, %.1f GB) – this may take several "
                "minutes…", n_dag, n_dag * HASH_BYTES / 1e9)

    dag = np.empty((n_dag, NODE_WORDS), dtype=np.uint32)
    t0  = time.time()
    for i in range(n_dag):
        dag[i] = make_dag_item(cache, i)
        if i % 100000 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (n_dag - i)
            logger.info("  DAG progress %d/%d  ETA %.0fs", i, n_dag, eta)

    logger.info("DAG built in %.0fs", time.time() - t0)
    return dag, n_dag


# ---------------------------------------------------------------------------
# KISS99 PRNG (Python, mirrors C device code)
# ---------------------------------------------------------------------------
class Kiss99:
    def __init__(self, z: int, w: int, jsr: int, jcong: int):
        self.z     = z     & 0xFFFFFFFF
        self.w     = w     & 0xFFFFFFFF
        self.jsr   = jsr   & 0xFFFFFFFF
        self.jcong = jcong & 0xFFFFFFFF

    def next(self) -> int:
        self.z     = (36969 * (self.z    & 65535) + (self.z    >> 16)) & 0xFFFFFFFF
        self.w     = (18000 * (self.w    & 65535) + (self.w    >> 16)) & 0xFFFFFFFF
        mwc        = ((self.z << 16) + (self.w & 65535)) & 0xFFFFFFFF
        self.jsr  ^= (self.jsr << 17) & 0xFFFFFFFF
        self.jsr  ^= (self.jsr >> 13) & 0xFFFFFFFF
        self.jsr  ^= (self.jsr <<  5) & 0xFFFFFFFF
        self.jcong = (69069 * self.jcong + 1234567) & 0xFFFFFFFF
        return ((mwc ^ self.jcong) + self.jsr) & 0xFFFFFFFF


def _build_program(period_seed: int) -> dict:
    """
    Build the random per-period program descriptor (Python equivalent of
    the C host function build_program in cuda_kernel.cu).
    """
    rng = Kiss99(
        (period_seed & 0xFFFFFFFF) ^ FNV_OFFSET,
        ((period_seed >> 16) ^ 0xDEADBEEF) & 0xFFFFFFFF,
        0x13579BDF,
        0x24681357,
    )
    # Warm up
    for _ in range(4):
        rng.next()

    prog = {
        "dag_dst":    [rng.next() % PROGPOW_LANES for _ in range(PROGPOW_CNT_MEM)],
        "dag_src":    [rng.next() % PROGPOW_REGS  for _ in range(PROGPOW_CNT_MEM)],
        "cache_src":  [rng.next() % PROGPOW_REGS  for _ in range(PROGPOW_CNT_CACHE)],
        "cache_dst":  [rng.next() % PROGPOW_REGS  for _ in range(PROGPOW_CNT_CACHE)],
        "cache_r":    [rng.next()                  for _ in range(PROGPOW_CNT_CACHE)],
        "math_src_a": [rng.next() % PROGPOW_REGS  for _ in range(PROGPOW_CNT_MATH)],
        "math_src_b": [rng.next() % PROGPOW_REGS  for _ in range(PROGPOW_CNT_MATH)],
        "math_dst":   [rng.next() % PROGPOW_REGS  for _ in range(PROGPOW_CNT_MATH)],
        "math_r":     [rng.next()                  for _ in range(PROGPOW_CNT_MATH)],
    }
    return prog


# ---------------------------------------------------------------------------
# KawPow – main class
# ---------------------------------------------------------------------------
class KawPow:
    """
    High-level KawPow engine.

    Manages:
      - DAG lifecycle (builds once per epoch, cached on GPU)
      - CUDA kernel compilation and launch
      - Per-period program updates
    """

    NONCES_PER_BATCH = 2048   # Number of nonces searched per kernel launch

    def __init__(self, gpu_index: int = 0):
        self.gpu_index    = gpu_index
        self._dag_epoch   = -1
        self._dag_gpu     = None       # pycuda DeviceAllocation
        self._dag_items   = 0
        self._period      = -1
        self._prog        = None       # current program dict
        self._prog_gpu    = None       # program struct on device
        self._module      = None       # compiled CUDA module
        self._kernel      = None       # kernel function handle
        self._header_gpu  = None       # current header on device
        self._initialized = False

    # ------------------------------------------------------------------
    def _init_cuda(self) -> None:
        """Lazy CUDA initialisation – called once."""
        if self._initialized:
            return
        try:
            import pycuda.autoinit          # noqa: F401 (side-effects only)
            import pycuda.driver as cuda
            from pycuda.compiler import SourceModule
        except ImportError as exc:
            raise RuntimeError(
                "pycuda is not installed. Run: pip install pycuda"
            ) from exc

        self._cuda  = cuda
        self._SourceModule = SourceModule

        # Compile CUDA kernel
        kernel_path = os.path.join(os.path.dirname(__file__), "cuda_kernel.cu")
        with open(kernel_path, "r") as fh:
            src = fh.read()

        logger.info("Compiling CUDA kernel…")
        self._module = SourceModule(src, no_extern_c=True,
                                    options=["-O3", "--use_fast_math"])
        self._kernel = self._module.get_function("kawpow_kernel")
        self._initialized = True
        logger.info("CUDA kernel ready on GPU %d", self.gpu_index)

    # ------------------------------------------------------------------
    def update_dag(self, block_number: int) -> None:
        """Build and upload the DAG to GPU if the epoch has changed."""
        epoch = epoch_for_block(block_number)
        if epoch == self._dag_epoch:
            return

        self._init_cuda()
        cuda = self._cuda
        dag, n_items = build_dag(block_number)

        # Upload to GPU
        logger.info("Uploading DAG to GPU (%.1f GB)…", dag.nbytes / 1e9)
        if self._dag_gpu is not None:
            self._dag_gpu.free()
        self._dag_gpu   = cuda.mem_alloc(dag.nbytes)
        cuda.memcpy_htod(self._dag_gpu, dag)
        self._dag_items = n_items
        self._dag_epoch = epoch
        logger.info("DAG on GPU  epoch=%d  items=%d", epoch, n_items)

    # ------------------------------------------------------------------
    def update_header(self, header_hash: bytes) -> None:
        """Upload the current block header hash (32 bytes) to GPU."""
        self._init_cuda()
        if self._header_gpu is None:
            self._header_gpu = self._cuda.mem_alloc(32)
        self._cuda.memcpy_htod(self._header_gpu, header_hash)

    # ------------------------------------------------------------------
    def _ensure_program(self, block_number: int) -> None:
        """Rebuild the per-period program if the period has changed."""
        period = block_number // PROGPOW_PERIOD
        if period == self._period:
            return
        self._period = period
        self._prog   = _build_program(period)
        logger.debug("New ProgPoW program for period %d", period)

    # ------------------------------------------------------------------
    def _pack_program(self) -> np.ndarray:
        """Pack the program dict into a flat uint32 array for the kernel."""
        p = self._prog
        parts = (
            p["dag_dst"]    + p["dag_src"]    +
            p["cache_src"]  + p["cache_dst"]  + p["cache_r"]  +
            p["math_src_a"] + p["math_src_b"] + p["math_dst"] + p["math_r"]
        )
        return np.array(parts, dtype=np.uint32)

    # ------------------------------------------------------------------
    def mine(
        self,
        header_hash: bytes,
        block_number: int,
        target: bytes,
        base_nonce: int,
    ) -> Tuple[Optional[int], Optional[bytes]]:
        """
        Search *NONCES_PER_BATCH* nonces starting from *base_nonce*.

        Returns (winning_nonce, mix_hash_bytes) if a share is found,
        or (None, None) if no solution was found in this batch.
        """
        import pycuda.driver as cuda
        import numpy as np

        self._init_cuda()
        self.update_dag(block_number)
        self.update_header(header_hash)
        self._ensure_program(block_number)

        # Pass full 256-bit target as 8 x uint32 (little-endian)
        target_arr = np.frombuffer(target, dtype=np.uint32).copy()

        # Allocate output buffers
        NO_SOLUTION  = np.uint32(0xFFFFFFFF)
        result_nonce = cuda.mem_alloc(4)
        mix_out      = cuda.mem_alloc(32)
        # Initialise result_nonce to sentinel
        cuda.memcpy_htod(result_nonce, np.array([NO_SOLUTION], dtype=np.uint32))
        cuda.memcpy_htod(mix_out,      np.zeros(8, dtype=np.uint32))

        prog_arr = self._pack_program()

        # Grid = NONCES_PER_BATCH blocks; block = PROGPOW_LANES threads
        self._kernel(
            self._header_gpu,
            self._dag_gpu,
            np.uint32(self._dag_items),
            np.uint64(base_nonce),
            target_arr,
            prog_arr,            # passed as __constant__ struct via driver API
            result_nonce,
            mix_out,
            block=(PROGPOW_LANES, 1, 1),
            grid=(self.NONCES_PER_BATCH, 1),
        )
        cuda.Context.synchronize()

        # Read back results
        nonce_out = np.empty(1, dtype=np.uint32)
        mix_bytes = np.empty(8, dtype=np.uint32)
        cuda.memcpy_dtoh(nonce_out, result_nonce)
        cuda.memcpy_dtoh(mix_bytes, mix_out)

        if nonce_out[0] != NO_SOLUTION:
            winning_nonce = (base_nonce & ~0xFFFFFFFF) | int(nonce_out[0])
            return winning_nonce, mix_bytes.tobytes()

        return None, None
