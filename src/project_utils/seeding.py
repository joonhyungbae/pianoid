"""
Deterministic seeding for reproducible training.

The original paper checkpoints were produced without a fixed seed, so the
paper numbers reflect one particular stochastic initialization. Release-time
re-training showed that without a fixed seed, Stage 2 converges to an
"under-expressive" regime across multiple seeds (Δ1 ≈ 0.42 vs the paper's
0.27). To let downstream users sweep seeds consistently and lock a chosen
seed into the released config, all per-stage trainers now call `fix_seed()`
and pass `seed_worker` to their DataLoaders.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def fix_seed(seed: int, cudnn_deterministic: bool = True) -> None:
    """Fix Python / NumPy / PyTorch / CUDA RNGs and CuDNN behaviour.

    `cudnn_deterministic=True` disables autotune and forces deterministic
    kernels. It makes training ~10-15% slower but removes the main remaining
    non-determinism on GPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = not bool(cudnn_deterministic)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn that seeds Python/NumPy per worker.

    Use as `DataLoader(..., worker_init_fn=seed_worker, generator=make_generator(seed))`
    so that shuffling and random augmentations reproduce across runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Create a torch.Generator seeded deterministically."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
