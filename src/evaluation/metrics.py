"""
src/evaluation/metrics.py

Canonical source for evaluation metrics.

Metrics:
- MPJPE: Mean Per-Joint Position Error (mm)
- FID: Frechet Inception Distance on motion features
- Jitter: Mean jerk magnitude (smoothness)
- Onset Sync: synchronization between note onsets and hand-velocity peaks
- Velocity Correlation: velocity correlation between GT and generated motion
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.linalg import sqrtm

try:
    from data.constants import RWRIST_IDX, LWRIST_IDX, RHAND_IDX, LHAND_IDX
except ModuleNotFoundError:  # when imported from repo root as `src.evaluation`
    from src.data.constants import RWRIST_IDX, LWRIST_IDX, RHAND_IDX, LHAND_IDX


def load_npz_result(data, npz_file: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load (gt, pred, sequence_idx) from an evaluation .npz.

    Supports two on-disk formats:
      - Ablation: keys 'gt', 'generated'; filename 00000.npz -> idx 0.
      - Stage4:   keys 'gt_joints', 'generated'; filename sample_000.npz -> idx 0.
    """
    keys = getattr(data, 'files', list(data.keys()))
    if 'gt' in keys:
        gt = data['gt']
    elif 'gt_joints' in keys:
        gt = data['gt_joints']
    else:
        raise KeyError(f"npz must have 'gt' or 'gt_joints': {npz_file}")
    pred = data['generated']
    if pred.ndim == 4 and pred.shape[0] == 1:
        pred = pred[0]
    stem = npz_file.stem
    if stem.isdigit():
        idx = int(stem)
    elif stem.startswith('sample_') and len(stem) > 7 and stem[7:].isdigit():
        idx = int(stem[7:])
    else:
        idx = 0
    return gt, pred, idx


# ============================================================
# Core Metrics
# ============================================================

def compute_mpjpe(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Per-Joint Position Error in mm.
    
    Args:
        gt: (T, J, 3) ground truth motion (meters)
        pred: (T, J, 3) predicted motion (meters)
    
    Returns:
        mpjpe: Mean error in millimeters
    """
    min_len = min(len(gt), len(pred))
    gt, pred = gt[:min_len], pred[:min_len]
    
    error = np.sqrt(np.sum((gt - pred) ** 2, axis=-1))  # (T, J)
    return float(error.mean() * 1000)  # m -> mm


def compute_mpjpe_hands(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute MPJPE for hand joints only (wrist + fingertip)."""
    hand_indices = [LWRIST_IDX, LHAND_IDX, RWRIST_IDX, RHAND_IDX]
    
    min_len = min(len(gt), len(pred))
    gt_hands = gt[:min_len, hand_indices]
    pred_hands = pred[:min_len, hand_indices]
    
    error = np.sqrt(np.sum((gt_hands - pred_hands) ** 2, axis=-1))
    return float(error.mean() * 1000)


def compute_jitter(motion: np.ndarray) -> float:
    """Compute jitter (mean jerk magnitude).
    
    Jitter = mean |d³x/dt³|, measures smoothness (lower = smoother).
    """
    if len(motion) < 4:
        return 0.0
    
    velocity = np.diff(motion, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    
    jerk_magnitude = np.linalg.norm(jerk, axis=-1)
    return float(jerk_magnitude.mean())


def compute_velocity_correlation(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute velocity correlation between GT and predicted motion."""
    min_len = min(len(gt), len(pred))
    gt, pred = gt[:min_len], pred[:min_len]
    
    if len(gt) < 2:
        return 0.0
    
    gt_vel = np.linalg.norm(np.diff(gt, axis=0), axis=-1).flatten()
    pred_vel = np.linalg.norm(np.diff(pred, axis=0), axis=-1).flatten()
    
    if gt_vel.std() < 1e-8 or pred_vel.std() < 1e-8:
        return 0.0
    
    corr, _ = stats.pearsonr(gt_vel, pred_vel)
    return float(corr) if not np.isnan(corr) else 0.0


# ============================================================
# FID Computation
# ============================================================

def compute_motion_features(motion: np.ndarray) -> np.ndarray:
    """Extract motion features for FID computation."""
    T, J, _ = motion.shape
    
    # Normalized positions
    pos = motion.reshape(T, -1)
    pos_norm = (pos - pos.mean(axis=0, keepdims=True)) / (pos.std(axis=0, keepdims=True) + 1e-6)
    
    # Velocities
    vel = np.diff(motion, axis=0)
    vel = np.concatenate([np.zeros((1, J, 3)), vel], axis=0).reshape(T, -1)
    
    # Accelerations
    acc = np.diff(vel, axis=0)
    acc = np.concatenate([np.zeros((1, vel.shape[1])), acc], axis=0)
    
    return np.concatenate([pos_norm, vel, acc], axis=-1)


def compute_fid(gt_features: np.ndarray, pred_features: np.ndarray) -> float:
    """Compute Frechet Inception Distance between feature distributions."""
    if len(gt_features) == 0 or len(pred_features) == 0:
        return float('inf')
    
    mu_gt, mu_pred = gt_features.mean(axis=0), pred_features.mean(axis=0)
    cov_gt, cov_pred = np.cov(gt_features, rowvar=False), np.cov(pred_features, rowvar=False)
    
    # Handle scalar case
    if cov_gt.ndim == 0:
        cov_gt = np.array([[cov_gt]])
    if cov_pred.ndim == 0:
        cov_pred = np.array([[cov_pred]])
    
    # Regularization
    eps = 1e-6
    cov_gt += eps * np.eye(cov_gt.shape[0])
    cov_pred += eps * np.eye(cov_pred.shape[0])
    
    diff = mu_gt - mu_pred
    mean_diff = np.sum(diff ** 2)
    
    try:
        covmean = sqrtm(cov_gt @ cov_pred)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        trace_term = np.trace(cov_gt) + np.trace(cov_pred) - 2 * np.trace(covmean)
        fid = float(mean_diff + trace_term)
    except Exception:
        fid = float(mean_diff + np.trace(cov_gt) + np.trace(cov_pred))
    
    return max(0, fid)


def compute_fid_from_motions(
    gt_motions: List[np.ndarray],
    pred_motions: List[np.ndarray],
) -> float:
    """Compute FID from motion sequences."""
    gt_features = [compute_motion_features(m).mean(axis=0) for m in gt_motions]
    pred_features = [compute_motion_features(m).mean(axis=0) for m in pred_motions]
    
    return compute_fid(np.array(gt_features), np.array(pred_features))


# ============================================================
# Aggregation & Formatting
# ============================================================

def compute_all_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    onset: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all metrics for a single sample."""
    metrics = {
        'mpjpe': compute_mpjpe(gt, pred),
        'mpjpe_hands': compute_mpjpe_hands(gt, pred),
        'jitter_gt': compute_jitter(gt),
        'jitter_pred': compute_jitter(pred),
        'vel_corr': compute_velocity_correlation(gt, pred),
    }
    
    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]],
    gt_motions: Optional[List[np.ndarray]] = None,
    pred_motions: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """Aggregate metrics across samples."""
    if not all_metrics:
        return {}
    
    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregated[key] = float(np.mean(values))
    
    if gt_motions and pred_motions:
        aggregated['fid'] = compute_fid_from_motions(gt_motions, pred_motions)
    
    return aggregated


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for display."""
    lines = []
    if 'mpjpe' in metrics:
        lines.append(f"MPJPE: {metrics['mpjpe']:.1f} mm")
    if 'mpjpe_hands' in metrics:
        lines.append(f"Hand MPJPE: {metrics['mpjpe_hands']:.1f} mm")
    if 'fid' in metrics:
        lines.append(f"FID: {metrics['fid']:.2f}")
    if 'jitter_pred' in metrics:
        lines.append(f"Jitter: {metrics['jitter_pred']*1e4:.1f} (×10⁻⁴)")
    if 'vel_corr' in metrics:
        lines.append(f"Vel Corr: {metrics['vel_corr']:.2f}")
    return "\n".join(lines)
