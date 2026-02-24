"""
kawpow – KawPow (ProgPoW) algorithm for Ravencoin mining.

Public surface:
    KawPow          – high-level class wrapping CUDA kernel + DAG management
    build_dag        – generate Ethash-style DAG on the CPU
    epoch_for_block  – map block number → epoch number
"""

from kawpow.algorithm import KawPow, build_dag, epoch_for_block

__all__ = ["KawPow", "build_dag", "epoch_for_block"]
