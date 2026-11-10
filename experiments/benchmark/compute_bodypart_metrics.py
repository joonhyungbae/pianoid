#!/usr/bin/env python3
"""
compute_bodypart_metrics.py

Compute all evaluation metrics (MPJPE, FID, Jitter, Velocity Correlation)
per body part and per generation stage for all models.

Uses the canonical metrics from src/evaluation/metrics.py
"""

import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import canonical metrics
from src.evaluation.metrics import (
    compute_mpjpe,
    compute_jitter,
    compute_velocity_correlation,
    compute_fid_from_motions,
)

# =============================================================================
# Joint Definitions
# =============================================================================

# Individual body parts
JOINT_PARTS = {
    'Head': [0, 1, 2, 3],           # RFHD, LFHD, RBHD, LBHD
    'Neck': [4, 5],                  # C7, CLAV
    'Chest': [6, 7],                 # T10, STRN
    'Shoulder': [8, 12],             # RSHO, LSHO
    'Elbow': [9, 13],                # RELB, LELB
    'Wrist': [10, 14],               # RWRIST, LWRIST
    'Hand': [11, 15],                # RHAND, LHAND
    'Pelvis': [16, 17, 18, 19],      # RASI, LASI, RPSI, LPSI
    'Knee': [20, 23],                # RKNE, LKNE
    'Ankle': [21, 24],               # RANKLE, LANKLE
    'Toe': [22, 25],                 # RTOE, LTOE
}

# Stage-wise groupings (matching Pianoid pipeline)
STAGE_GROUPS = {
    'Hand_Stage': [10, 11, 14, 15],  # Hand MDM output
    'Backbone_Stage': [0, 1, 2, 3, 4, 5, 6, 7, 8, 12],  # Backbone MDM output
    'Elbow_Stage': [9, 13],          # Elbow MDM output
    'Lower_Stage': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],  # FiLM Lower output
}

# Semantic groupings for paper
SEMANTIC_GROUPS = {
    'Upper_Body': list(range(0, 16)),  # Head to hands
    'Lower_Body': list(range(16, 26)), # Pelvis to toes
    'Hands_Only': [10, 11, 14, 15],    # Wrist + Hand
    'Arms_Full': [8, 9, 10, 11, 12, 13, 14, 15],  # Shoulder to Hand
}

# =============================================================================
# Metric Functions (wrappers for body-part specific computation)
# =============================================================================

def compute_mpjpe_part(gt, pred, joint_indices=None):
    """Mean Per-Joint Position Error in mm for specific joints."""
    if joint_indices is not None:
        gt = gt[:, joint_indices]
        pred = pred[:, joint_indices]
    return compute_mpjpe(gt, pred)


def compute_jitter_part(motion, joint_indices=None):
    """Compute jitter for specified joints."""
    if joint_indices is not None:
        motion = motion[:, joint_indices]
    return compute_jitter(motion)


def compute_vel_corr_part(gt, pred, joint_indices=None):
    """Velocity correlation for specific joints."""
    if joint_indices is not None:
        gt = gt[:, joint_indices]
        pred = pred[:, joint_indices]
    return compute_velocity_correlation(gt, pred)


def compute_fid_part(gt_list, pred_list, joint_indices=None):
    """FID for specific joints."""
    if joint_indices is not None:
        gt_list = [g[:, joint_indices] for g in gt_list]
        pred_list = [p[:, joint_indices] for p in pred_list]
    return compute_fid_from_motions(gt_list, pred_list)


# =============================================================================
# Data Loading
# =============================================================================

def load_results(results_dir, gt_key='gt_joints', pred_key='generated'):
    """Load all npz results from a directory."""
    results_dir = Path(results_dir)
    npz_files = sorted(results_dir.glob('*.npz'))
    
    gt_list, pred_list = [], []
    for f in npz_files:
        if 'metrics' in f.name:
            continue
        data = np.load(f)
        gt = data.get(gt_key, data.get('gt'))
        pred = data.get(pred_key, data.get('pred'))
        
        if gt is None or pred is None:
            continue
        
        # Handle different shapes
        if pred.ndim == 4:
            pred = pred[0]
        if gt.ndim == 4:
            gt = gt[0]
        
        # Align lengths
        min_len = min(len(gt), len(pred))
        gt_list.append(gt[:min_len])
        pred_list.append(pred[:min_len])
    
    return gt_list, pred_list


# =============================================================================
# Main Computation
# =============================================================================

def compute_all_metrics_part(gt_list, pred_list, joint_indices=None, name=""):
    """Compute all metrics for given joint subset."""
    if not gt_list:
        return None
    
    # Per-sequence metrics
    mpjpe_list = []
    jitter_gt_list = []
    jitter_pred_list = []
    vel_corr_list = []
    
    for gt, pred in zip(gt_list, pred_list):
        mpjpe_list.append(compute_mpjpe_part(gt, pred, joint_indices))
        jitter_gt_list.append(compute_jitter_part(gt, joint_indices))
        jitter_pred_list.append(compute_jitter_part(pred, joint_indices))
        vel_corr_list.append(compute_vel_corr_part(gt, pred, joint_indices))
    
    # FID (computed across all samples)
    fid = compute_fid_part(gt_list, pred_list, joint_indices)
    
    # Jitter deviation from GT (×10^5 for readability in paper)
    jitter_gt_mean = np.mean(jitter_gt_list)
    jitter_pred_mean = np.mean(jitter_pred_list)
    jitter_delta = abs(jitter_pred_mean - jitter_gt_mean) * 1e5  # Scale for paper
    
    return {
        'mpjpe': np.mean(mpjpe_list),
        'mpjpe_std': np.std(mpjpe_list),
        'fid': fid,
        'jitter_gt': jitter_gt_mean,
        'jitter_pred': jitter_pred_mean,
        'jitter_delta': jitter_delta,  # Already scaled
        'vel_corr': np.mean(vel_corr_list),
        'n_samples': len(gt_list),
    }


def main():
    # Paper Table 4: per-body-part MPJPE for Pianoid.
    models = {
        'Pianoid': 'results/hierarchical_mdm',
    }
    
    all_results = {}
    
    for model_name, results_path in models.items():
        if not Path(results_path).exists():
            print(f"Skipping {model_name}: {results_path} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")
        
        gt_list, pred_list = load_results(results_path)
        if not gt_list:
            print(f"  No results found")
            continue
        
        print(f"  Loaded {len(gt_list)} sequences")
        
        model_results = {}
        
        # Full body
        model_results['Full_Body'] = compute_all_metrics_part(gt_list, pred_list, None)
        
        # Stage groups
        for group_name, indices in STAGE_GROUPS.items():
            model_results[group_name] = compute_all_metrics_part(gt_list, pred_list, indices)
        
        # Semantic groups
        for group_name, indices in SEMANTIC_GROUPS.items():
            model_results[group_name] = compute_all_metrics_part(gt_list, pred_list, indices)
        
        # Individual parts
        for part_name, indices in JOINT_PARTS.items():
            model_results[part_name] = compute_all_metrics_part(gt_list, pred_list, indices)
        
        all_results[model_name] = model_results
    
    # Save results
    output_path = Path('experiments/benchmark/bodypart_metrics.json')
    
    # Convert numpy types to Python types for JSON
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert_types(all_results), f, indent=2)
    
    print(f"\n\nResults saved to {output_path}")
    
    # Print summary tables
    print_summary(all_results)
    
    return all_results


def print_summary(all_results):
    """Print formatted summary tables."""
    
    print("\n" + "="*80)
    print("FULL BODY METRICS")
    print("="*80)
    print(f"{'Model':<20} {'MPJPE↓':<10} {'FID↓':<10} {'|ΔJ|↓':<10} {'Vel↑':<10}")
    print("-"*80)
    
    for model, results in all_results.items():
        r = results.get('Full_Body', {})
        if r:
            print(f"{model:<20} {r['mpjpe']:<10.1f} {r['fid']:<10.2f} {r['jitter_delta']:<10.1f} {r['vel_corr']:<10.2f}")
    
    print("\n" + "="*80)
    print("STAGE-WISE MPJPE (mm)")
    print("="*80)
    print(f"{'Model':<20} {'Hand':<10} {'Backbone':<12} {'Elbow':<10} {'Lower':<10}")
    print("-"*80)
    
    for model, results in all_results.items():
        hand = results.get('Hand_Stage', {}).get('mpjpe', 0)
        backbone = results.get('Backbone_Stage', {}).get('mpjpe', 0)
        elbow = results.get('Elbow_Stage', {}).get('mpjpe', 0)
        lower = results.get('Lower_Stage', {}).get('mpjpe', 0)
        print(f"{model:<20} {hand:<10.1f} {backbone:<12.1f} {elbow:<10.1f} {lower:<10.1f}")
    
    print("\n" + "="*80)
    print("STAGE-WISE VELOCITY CORRELATION")
    print("="*80)
    print(f"{'Model':<20} {'Hand':<10} {'Backbone':<12} {'Elbow':<10} {'Lower':<10}")
    print("-"*80)
    
    for model, results in all_results.items():
        hand = results.get('Hand_Stage', {}).get('vel_corr', 0)
        backbone = results.get('Backbone_Stage', {}).get('vel_corr', 0)
        elbow = results.get('Elbow_Stage', {}).get('vel_corr', 0)
        lower = results.get('Lower_Stage', {}).get('vel_corr', 0)
        print(f"{model:<20} {hand:<10.2f} {backbone:<12.2f} {elbow:<10.2f} {lower:<10.2f}")
    
    print("\n" + "="*80)
    print("SEMANTIC GROUPINGS - MPJPE (mm)")
    print("="*80)
    print(f"{'Model':<20} {'Upper':<10} {'Lower':<10} {'Hands':<10} {'Arms':<10}")
    print("-"*80)
    
    for model, results in all_results.items():
        upper = results.get('Upper_Body', {}).get('mpjpe', 0)
        lower = results.get('Lower_Body', {}).get('mpjpe', 0)
        hands = results.get('Hands_Only', {}).get('mpjpe', 0)
        arms = results.get('Arms_Full', {}).get('mpjpe', 0)
        print(f"{model:<20} {upper:<10.1f} {lower:<10.1f} {hands:<10.1f} {arms:<10.1f}")


if __name__ == '__main__':
    main()
