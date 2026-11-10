"""
src/evaluation - shared evaluation-metrics module.

The single source for evaluation metrics used across the project.

Usage:
    from src.evaluation import compute_mpjpe, compute_fid, compute_all_metrics
"""

from .metrics import (
    # I/O helper
    load_npz_result,
    # Core metrics
    compute_mpjpe,
    compute_mpjpe_hands,
    compute_jitter,
    compute_velocity_correlation,
    # FID
    compute_motion_features,
    compute_fid,
    compute_fid_from_motions,
    # Aggregation
    compute_all_metrics,
    aggregate_metrics,
    format_metrics,
)

__all__ = [
    'load_npz_result',
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
