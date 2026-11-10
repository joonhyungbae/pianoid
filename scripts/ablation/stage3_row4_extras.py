#!/usr/bin/env python3
"""Compute Z-rng ratio and ER-jerk for Stage 3 Row 4 (functional + predicted-train).
Reuses s32_evaluate.py infrastructure: loads ckpt, runs windowed inference on test,
then computes elbow-only Z-range ratio and jerk ratio (mean |pred| / mean |gt|).
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from stage3.s32_evaluate import (
    load_test_data, windowed_regression, windowed_mdm, ARM_INDICES, ARM_NAMES,
)
from models.model_factory import build_stage3_arm_model

CKPT_PATTERN = str(REPO / 'checkpoints/stage3_ablation/row4_functional_predicted_train/*/best.pt')
CACHE_DIR = str(REPO / 'MOSA_dataset/training_cache_W10')
HAND_DIR = str(REPO / 'MOSA_dataset/phase0_hand_predictions')
TORSO_DIR = str(REPO / 'MOSA_dataset/phase1_torso_predictions')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compute_elbow_metrics(pred_12d, gt_12d):
    """pred/gt: (T, 12) with [RSHO, RELB, LSHO, LELB] × 3 axes.
    Returns elbow-axis-wise range ratio and |jerk| ratio (pred mean / gt mean)."""
    # Elbows at indices 3:6 (RELB) and 9:12 (LELB)
    pe = np.concatenate([pred_12d[:, 3:6], pred_12d[:, 9:12]], axis=-1)  # (T, 6): RELB xyz, LELB xyz
    ge = np.concatenate([gt_12d[:, 3:6], gt_12d[:, 9:12]], axis=-1)
    # Z axis: index 2 and 5 (of 6-dim elbow block)
    pe_z = pe[:, [2, 5]]  # (T, 2)
    ge_z = ge[:, [2, 5]]

    # Z range ratio (per-joint)
    pe_rng = pe_z.max(0) - pe_z.min(0)  # (2,)
    ge_rng = ge_z.max(0) - ge_z.min(0)
    z_rng_ratios = np.where(ge_rng > 1e-6, pe_rng / ge_rng, 0)  # (2,)

    # Jerk (3rd derivative, per-joint all axes averaged)
    def mean_abs_jerk(x):
        if len(x) < 4:
            return np.full(x.shape[1], np.nan)
        j = np.diff(x, n=3, axis=0)
        return np.abs(j).mean(axis=0)
    pe_jerk = mean_abs_jerk(pe)  # (6,)
    ge_jerk = mean_abs_jerk(ge)
    jerk_ratios = np.where(ge_jerk > 1e-9, pe_jerk / ge_jerk, np.nan)

    return {
        'z_rng_RELB': float(z_rng_ratios[0]),
        'z_rng_LELB': float(z_rng_ratios[1]),
        'z_rng_mean': float(np.nanmean(z_rng_ratios)),
        'jerk_RELB_mean': float(np.nanmean(jerk_ratios[:3])),
        'jerk_LELB_mean': float(np.nanmean(jerk_ratios[3:])),
        'jerk_mean': float(np.nanmean(jerk_ratios)),
    }


# Load checkpoint
from glob import glob
ckpt_files = glob(CKPT_PATTERN)
assert len(ckpt_files) >= 1, f'no ckpt found at {CKPT_PATTERN}'
ckpt_path = ckpt_files[0]
print(f'Loading: {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
config = ckpt['config']
print(f'  arch={config.get("arch")}, functional={config.get("functional_grouping")}, '
      f'gt_inputs={config.get("use_gt_inputs")}, relative={config.get("relative_coords")}')

model = build_stage3_arm_model(config, DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

test_data = load_test_data(CACHE_DIR, HAND_DIR, TORSO_DIR, config)
print(f'Loaded {len(test_data)} test samples')

per_seq = []
all_errors = []
for arm_gt, spatial, T, aria, od in tqdm(test_data):
    # arm_gt is (T, 12): [RSHO, RELB, LSHO, LELB] axes
    pred = windowed_regression(model, spatial, T, DEVICE, aria=aria, od_np=od,
                                use_aria=config.get('use_aria', False))
    # In functional grouping pred is (T, 12) directly (no relative reconstruction)
    relative = config.get('relative_coords', False)
    if relative:
        # anatomical: pred is already [rsho, rsho+relb, lsho, lsho+lelb]
        # (not our case for row4)
        pass
    gt = arm_gt.reshape(T, 12)
    m = compute_elbow_metrics(pred, gt)
    m['MPJPE_elbows'] = float(np.mean(np.sqrt(
        ((pred.reshape(T,4,3)[:,[1,3]] - gt.reshape(T,4,3)[:,[1,3]])**2).sum(-1))) * 1000)
    m['MPJPE_RELB'] = float(np.mean(np.sqrt(
        ((pred.reshape(T,4,3)[:,1] - gt.reshape(T,4,3)[:,1])**2).sum(-1))) * 1000)
    m['MPJPE_LELB'] = float(np.mean(np.sqrt(
        ((pred.reshape(T,4,3)[:,3] - gt.reshape(T,4,3)[:,3])**2).sum(-1))) * 1000)
    per_seq.append(m)

# Aggregate
def agg(key):
    vals = [s[key] for s in per_seq if not np.isnan(s[key])]
    return float(np.mean(vals))

summary = {
    'n_seq': len(per_seq),
    'MPJPE_elbows': agg('MPJPE_elbows'),
    'MPJPE_RELB': agg('MPJPE_RELB'),
    'MPJPE_LELB': agg('MPJPE_LELB'),
    'LR_asym_elbows': abs(agg('MPJPE_RELB') - agg('MPJPE_LELB')),
    'Z_rng_RELB': agg('z_rng_RELB'),
    'Z_rng_LELB': agg('z_rng_LELB'),
    'Z_rng_mean': agg('z_rng_mean'),
    'Jerk_ratio_mean': agg('jerk_mean'),
}
print('\n=== Row 4 (Functional + Predicted-train) ===')
for k, v in summary.items():
    print(f'  {k:<20} {v:.3f}')

out = REPO / 'checkpoints/stage3_ablation/row4_extras.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2))
print(f'\nSaved: {out}')
