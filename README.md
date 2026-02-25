# A_miner

Ravencoin (RVN) GPU miner using the **KawPow** proof-of-work algorithm with NVIDIA CUDA support.

> It's a heater program XD

## Features

- **KawPow / ProgPow** algorithm – the Ravencoin proof-of-work
- **NVIDIA GPU** mining via CUDA (Pascal and newer, compute capability ≥ 6.1)
- **Stratum** pool connectivity with automatic reconnection
- **CPU** reference / verification mode (built when CUDA is not available)
- Clean C++17 codebase, minimal dependencies

## Requirements

| Component | Minimum version |
|-----------|----------------|
| CMake     | 3.18+          |
| GCC / G++ | 10+            |
| NVIDIA CUDA Toolkit | 11.0+ (for GPU mining) |
| OpenSSL   | optional       |

An NVIDIA GPU with **compute capability 6.1 or higher** is required for GPU mining:
Pascal (GTX 1060+), Turing (RTX 2060+), Ampere (RTX 3060+), Ada Lovelace (RTX 4060+).

## Building

```bash
# Clone
git clone https://github.com/cheyu-cyber/A_miner.git
cd A_miner

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

If the CUDA toolkit is installed, the build system detects it automatically and
enables GPU mining.  Without CUDA the miner builds in CPU-only mode (useful for
testing and verification).

## Usage

```bash
./a_miner -o <pool:port> -u <wallet.worker> [options]
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o` | Pool address (e.g. `stratum+tcp://rvn.2miners.com:6060`) | *required* |
| `-u` | Wallet address or `user.worker` | *required* |
| `-p` | Worker password | `x` |
| `-d` | CUDA device ID | `0` |
| `-v` | Verbose / debug logging | off |
| `-h` | Show help | — |

### Example

```bash
./a_miner -o rvn.2miners.com:6060 -u RYourfullwalletaddress.rig1
```

## Architecture

```
src/
├── main.cpp                  # CLI entry point
├── miner.h / miner.cpp       # Orchestration (stratum + mining threads)
├── kawpow/
│   ├── keccak.h / .cpp        # Keccak-256 / Keccak-512 (original, not NIST SHA-3)
│   ├── ethash.h / .cpp        # Light-cache & DAG item generation
│   ├── kawpow.h / .cpp        # KawPow / ProgPow CPU reference
│   └── cuda/
│       └── kawpow_kernel.cu   # CUDA kernels (DAG build + nonce search)
├── stratum/
│   └── stratum_client.h/.cpp  # Stratum protocol client (mining.notify/submit)
└── utils/
    ├── log.h / .cpp           # Thread-safe logger
    └── hex.h / .cpp           # Hex encoding / decoding
```

## Running tests

```bash
cd build
ctest --output-on-failure
```

## How it works

1. **Connect** to a Ravencoin stratum pool and receive work (block header hash,
   seed hash, target difficulty).
2. **Build DAG** – generate the Ethash light-cache on CPU, then compute the
   full ~1 GiB+ DAG on GPU memory.
3. **Mine** – launch thousands of CUDA threads, each testing a different nonce
   through the KawPow random program.
4. **Submit** – when a nonce produces a hash below the target, submit it to the
   pool via stratum.

## License

This project is provided as-is for educational purposes.
