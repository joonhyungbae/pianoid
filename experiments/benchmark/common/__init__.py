"""
experiments/benchmark/common - shared module used by every baseline.

Provides:
- BaseMotionModel: abstract base class.
- metrics: MPJPE, FID, Jitter, Velocity Correlation.
- data_loader: train/test data loaders.
"""

from .base_model import BaseMotionModel
from .metrics import (
    compute_mpjpe,
    compute_mpjpe_hands,
    compute_fid,
    compute_jitter,
    compute_velocity_correlation,
    compute_all_metrics,
)
from .data_loader import (
    get_train_loader,
    get_test_loader,
    BenchmarkDataset,
)

__all__ = [
    'BaseMotionModel',
    'compute_mpjpe',
    'compute_mpjpe_hands',
    'compute_fid',
    'compute_jitter',
    'compute_velocity_correlation',
    'compute_all_metrics',
    'get_train_loader',
    'get_test_loader',
    'BenchmarkDataset',
]
