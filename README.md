# A_miner

GPU miner for **Ravencoin (RVN)** using the **KawPow** (ProgPoW) algorithm on **NVIDIA GPUs** via CUDA.

---

## Requirements

| Requirement | Notes |
|---|---|
| NVIDIA GPU | Compute Capability 6.0+ (Pascal / GTX 10xx or newer) |
| CUDA Toolkit | 11.x or 12.x — [download](https://developer.nvidia.com/cuda-downloads) |
| Python | 3.8+ |
| Python packages | `pip install -r requirements.txt` |

```
pip install -r requirements.txt
```

---

## Usage

```bash
python miner.py \
  -o stratum+tcp://<pool_host>:<port> \
  -u <YOUR_RVN_WALLET_ADDRESS> \
  -w <worker_name>
```

### Options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--pool` | *(required)* | Pool URL, e.g. `stratum+tcp://pool.rvn.io:3636` |
| `-u`, `--wallet` | *(required)* | Ravencoin wallet address or pool username |
| `-p`, `--password` | `x` | Pool password |
| `-w`, `--worker` | `worker1` | Worker / rig name |
| `--gpu` | `0` | CUDA device index (0 = first GPU) |
| `--log-level` | `INFO` | Logging verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Example

```bash
python miner.py \
  -o stratum+tcp://rvn.2miners.com:6060 \
  -u RQZq5MZNnkpWWnHzq4AVYK6qqZdQyYKN9k \
  -w rig1 \
  --gpu 0
```

---

## Architecture

```
miner.py              CLI entry point
stratum_client.py     Stratum V1 pool protocol (subscribe / authorize / notify / submit)
kawpow/
  __init__.py         Package surface
  algorithm.py        DAG generation (Ethash-compatible) + CUDA dispatch
  cuda_kernel.cu      CUDA kernel — keccak-f800, ProgPoW mixing loop, DAG lookup
requirements.txt      Python dependencies
```

### Algorithm overview

1. **Stratum** – connects to a pool, receives `mining.notify` messages with the
   current block header hash, seed hash, and target.
2. **DAG** – an epoch-based directed-acyclic graph (~1.5 GB, growing ~8 MB per
   epoch) is built once per 7500-block epoch and uploaded to GPU memory.
3. **KawPow hash** – for each candidate nonce:
   - `seed = keccak_f800_short(header, nonce)`
   - Mix registers are initialised from the seed across 16 lanes.
   - A *random program* (changes every 3 blocks) executes 4 rounds of
     DAG reads interleaved with L1-cache reads and ALU ops.
   - `mix_hash` is reduced from the 16×32 register file to 8 words.
   - `result = keccak_f800_long(header, seed, mix_hash)`
   - If `result ≤ target`, a valid share is found and submitted.

---

## Popular Ravencoin Pools

| Pool | Stratum URL |
|---|---|
| 2Miners | `stratum+tcp://rvn.2miners.com:6060` |
| Minerpool.net | `stratum+tcp://rvn.minerpool.net:3636` |
| Flypool | `stratum+tcp://ravencoin.flypool.org:3636` |
