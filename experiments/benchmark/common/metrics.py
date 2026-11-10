"""
experiments/benchmark/common/metrics.py

Re-exports metrics from src.evaluation (canonical source)
"""

# Re-export all metrics from canonical source
from src.evaluation import (
    compute_mpjpe,
    compute_mpjpe_hands,
    compute_jitter,
    compute_velocity_correlation,
    compute_motion_features,
    compute_fid,
    compute_fid_from_motions,
    compute_all_metrics,
    aggregate_metrics,
    format_metrics,
)

__all__ = [
    'compute_mpjpe',
    'compute_mpjpe_hands',
    'compute_jitter',
    'compute_velocity_correlation',
    'compute_motion_features',
    'compute_fid',
    'compute_fid_from_motions',
    'compute_all_metrics',
    'aggregate_metrics',
    'format_metrics',
]
