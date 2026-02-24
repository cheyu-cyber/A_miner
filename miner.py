#!/usr/bin/env python3
"""
A_miner – Ravencoin (KawPow) NVIDIA GPU Miner

Usage:
    python miner.py -o stratum+tcp://pool.rvn.io:3636 \
                    -u <YOUR_WALLET_ADDRESS> \
                    -w rig1

Requirements:
    pip install -r requirements.txt
    CUDA Toolkit + NVIDIA driver must be installed.
"""

import argparse
import logging
import sys
import urllib.parse

from kawpow import KawPow
from stratum_client import StratumClient

logger = logging.getLogger("A_miner")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="miner",
        description="A_miner – Ravencoin (KawPow) NVIDIA GPU Miner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o", "--pool", required=True,
        metavar="URL",
        help="Pool URL, e.g. stratum+tcp://pool.rvn.io:3636",
    )
    parser.add_argument(
        "-u", "--wallet", required=True,
        metavar="ADDRESS",
        help="Ravencoin wallet address (or pool username)",
    )
    parser.add_argument(
        "-p", "--password", default="x",
        metavar="PASS",
        help="Pool password",
    )
    parser.add_argument(
        "-w", "--worker", default="worker1",
        metavar="NAME",
        help="Worker / rig name appended to wallet as wallet.worker",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        metavar="INDEX",
        help="CUDA device index to use (0 = first GPU)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pool URL parsing
# ---------------------------------------------------------------------------
def _parse_pool_url(url: str):
    """
    Accept formats:
      stratum+tcp://host:port
      stratum+ssl://host:port
      host:port
    Returns (host, port).
    """
    clean = url
    for scheme in ("stratum+tcp://", "stratum+ssl://", "tcp://", "ssl://"):
        if clean.startswith(scheme):
            clean = clean[len(scheme):]
            break

    parts = clean.rsplit(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse pool URL '{url}'.  "
            "Expected format: stratum+tcp://host:port"
        )
    host = parts[0]
    try:
        port = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid port in pool URL: '{parts[1]}'")
    return host, port


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Check CUDA availability early
    try:
        import pycuda.autoinit   # noqa: F401
    except ImportError:
        logger.error(
            "pycuda is not installed. Run: pip install pycuda\n"
            "CUDA Toolkit must also be installed from https://developer.nvidia.com/cuda-downloads"
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("CUDA initialisation failed: %s", exc)
        logger.error("Make sure an NVIDIA GPU and driver are present.")
        sys.exit(1)

    # Parse pool address
    try:
        host, port = _parse_pool_url(args.pool)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  A_miner – Ravencoin (KawPow) NVIDIA GPU Miner")
    logger.info("=" * 60)
    logger.info("  Pool   : %s:%d", host, port)
    logger.info("  Wallet : %s", args.wallet)
    logger.info("  Worker : %s", args.worker)
    logger.info("  GPU    : device %d", args.gpu)
    logger.info("=" * 60)

    kawpow = KawPow(gpu_index=args.gpu)

    client = StratumClient(
        host=host,
        port=port,
        wallet=args.wallet,
        password=args.password,
        worker=args.worker,
        kawpow=kawpow,
    )

    try:
        client.start()
        client.wait()          # blocks until KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down…")
        client.stop()


if __name__ == "__main__":
    main()
