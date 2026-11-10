#!/usr/bin/env python3
"""
Compute ER metrics (range, speed, jerk) + MPJPE for single-stage MDM zscore results.

Usage:
    python experiments/benchmark/single_stage_mdm_zscore/compute_er_metrics.py
    python experiments/benchmark/single_stage_mdm_zscore/compute_er_metrics.py \
        --results_dir experiments/benchmark/single_stage_mdm_zscore/checkpoints/results
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from data.constants import (
    RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX,
    C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
    RSHO_IDX, LSHO_IDX,
)

BACKBONE_INDICES = [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX,
                    C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
                    RSHO_IDX, LSHO_IDX]


def compute_er(pred_joints, gt_joints, joint_indices):
    """Compute ER-range, ER-speed, ER-jerk for given joint indices."""
    pred = pred_joints[:, joint_indices, :]  # (T, J, 3)
    gt = gt_joints[:, joint_indices, :]

    J, A = pred.shape[1], pred.shape[2]
    er_range, er_speed, er_jerk = [], [], []

    for j in range(J):
        for a in range(A):
            p, g = pred[:, j, a], gt[:, j, a]
            # range
            pr = p.max() - p.min()
            gr = g.max() - g.min()
            if gr > 1e-6:
                er_range.append(pr / gr)
            # speed
            pv = np.abs(np.diff(p))
            gv = np.abs(np.diff(g))
            if gv.mean() > 1e-6:
                er_speed.append(pv.mean() / gv.mean())
            # jerk
            pa = np.diff(p, n=3)
            ga = np.diff(g, n=3)
            if np.abs(ga).mean() > 1e-6:
                er_jerk.append(np.abs(pa).mean() / np.abs(ga).mean())

    return np.mean(er_range), np.mean(er_speed), np.mean(er_jerk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',
                        default='experiments/benchmark/single_stage_mdm_zscore/checkpoints/results')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob('sample_*.npz'))
    print(f"Found {len(files)} result files in {results_dir}")

    mpjpes, er_ranges, er_speeds, er_jerks = [], [], [], []

    for f in tqdm(files):
        d = np.load(f)
        gen = d['generated']  # (T, 26, 3)
        gt = d['gt']          # (T, 26, 3)
        if gen.ndim == 2:
            gen = gen.reshape(-1, 26, 3)
        if gt.ndim == 2:
            gt = gt.reshape(-1, 26, 3)

        # MPJPE (all joints)
        mpjpe = np.sqrt(((gen - gt) ** 2).sum(axis=-1)).mean() * 1000  # mm
        mpjpes.append(mpjpe)

        # ER on backbone joints
        er_r, er_s, er_j = compute_er(gen, gt, BACKBONE_INDICES)
        er_ranges.append(er_r)
        er_speeds.append(er_s)
        er_jerks.append(er_j)

    results = {
        'MPJPE_mm': float(np.mean(mpjpes)),
        'ER_range': float(np.mean(er_ranges)),
        'ER_speed': float(np.mean(er_speeds)),
        'ER_jerk': float(np.mean(er_jerks)),
        'delta_1': float(np.mean([
            abs(np.log(max(np.mean(er_ranges), 1e-6))),
            abs(np.log(max(np.mean(er_speeds), 1e-6))),
            abs(np.log(max(np.mean(er_jerks), 1e-6))),
        ])),
        'delta_1_definition': 'mean|log ER| (symmetric penalty)',
        'n_samples': len(files),
    }

    print(f"\n{'='*50}")
    print(f"Single-Stage MDM + zscore Results ({len(files)} samples)")
    print(f"{'='*50}")
    print(f"  MPJPE:    {results['MPJPE_mm']:.1f} mm")
    print(f"  ER-range: {results['ER_range']:.2f}")
    print(f"  ER-speed: {results['ER_speed']:.2f}")
    print(f"  ER-jerk:  {results['ER_jerk']:.2f}")
    print(f"  Δ₁:      {results['delta_1']:.2f}")

    out_path = results_dir / 'metrics.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
