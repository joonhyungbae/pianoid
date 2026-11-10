#!/usr/bin/env python3
"""Compute Δ₁ axis-correlation and per-performer ER variance.

Two questions:
  (1) R5: Are log-ER-range, log-ER-speed, log-ER-jerk correlated across the
      80 test sequences? If yes, Δ₁ aggregation has effective d.o.f. < 3.
  (2) R9: Does Pianoid produce different motion for yp07 vs yp08 (no identity
      conditioning)? Per-performer ER mean and within-performer variance.

Backbone joints (Stage 2 output): RFHD, LFHD, RBHD, LBHD, C7, CLAV, T10, STRN, RSHO, LSHO.
ER ratios computed per-axis per-joint, then averaged.

Outputs:
  - print summary (suitable for paper footnote)
  - JSON file with full numbers
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from data.constants import (
    RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX,
    C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
    RSHO_IDX, LSHO_IDX,
)

BACKBONE_INDICES = [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX,
                    C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
                    RSHO_IDX, LSHO_IDX]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = PROJECT_ROOT / 'MOSA_dataset/full_body_predictions/test'
META = PROJECT_ROOT / 'MOSA_dataset/training_cache_W10/metadata.json'
FPS = 60


def er_ratios_for_seq(gt, pred):
    """Return per-axis ER (range, speed, jerk), averaged across backbone joints.

    gt, pred: (T, 26, 3) absolute positions (meters).
    Returns dict with er_range, er_speed, er_jerk (each scalar).
    """
    gt_b = gt[:, BACKBONE_INDICES, :]    # (T, 10, 3)
    pred_b = pred[:, BACKBONE_INDICES, :]  # (T, 10, 3)

    # Range per-axis per-joint
    gt_rng = gt_b.max(axis=0) - gt_b.min(axis=0)    # (10, 3)
    pred_rng = pred_b.max(axis=0) - pred_b.min(axis=0)
    er_range = (pred_rng / np.maximum(gt_rng, 1e-6)).mean()

    # Speed = mean |first diff|
    gt_speed_pj = np.abs(np.diff(gt_b, axis=0)).mean(axis=0)    # (10, 3)
    pred_speed_pj = np.abs(np.diff(pred_b, axis=0)).mean(axis=0)
    er_speed = (pred_speed_pj / np.maximum(gt_speed_pj, 1e-9)).mean()

    # Jerk = mean |3rd diff|
    if len(gt_b) > 4:
        gt_jerk_pj = np.abs(np.diff(gt_b, n=3, axis=0)).mean(axis=0)
        pred_jerk_pj = np.abs(np.diff(pred_b, n=3, axis=0)).mean(axis=0)
        er_jerk = (pred_jerk_pj / np.maximum(gt_jerk_pj, 1e-9)).mean()
    else:
        er_jerk = float('nan')

    return {
        'er_range': float(er_range),
        'er_speed': float(er_speed),
        'er_jerk': float(er_jerk),
    }


def main():
    meta = json.loads(META.read_text())
    test = {s['idx']: s for s in meta['splits']['test']['samples']}

    per_seq = []
    for i in range(80):
        d = PRED_DIR / f'{i:05d}'
        gt_path = d / 'gt_motion.npy'
        gen_path = d / 'generated_motion.npy'
        if not (gt_path.exists() and gen_path.exists()):
            continue
        gt = np.load(gt_path).astype(np.float32)
        pred = np.load(gen_path).astype(np.float32)
        if gt.shape != pred.shape:
            print(f'  shape mismatch idx={i}: gt {gt.shape} vs pred {pred.shape}, truncating')
            T = min(len(gt), len(pred))
            gt, pred = gt[:T], pred[:T]
        ers = er_ratios_for_seq(gt, pred)
        ers['idx'] = i
        ers['performer'] = test.get(i, {}).get('performer', '?')
        ers['piece'] = test.get(i, {}).get('piece', '?')
        per_seq.append(ers)

    print(f'Loaded {len(per_seq)} test sequences')

    # === R5: axis correlation ===
    log_rng = np.log(np.maximum([s['er_range'] for s in per_seq], 1e-6))
    log_spd = np.log(np.maximum([s['er_speed'] for s in per_seq], 1e-6))
    log_jrk = np.log(np.maximum([s['er_jerk']  for s in per_seq], 1e-6))

    abs_log_rng = np.abs(log_rng)
    abs_log_spd = np.abs(log_spd)
    abs_log_jrk = np.abs(log_jrk)

    corr_log = np.corrcoef([log_rng, log_spd, log_jrk])  # 3x3
    corr_abs = np.corrcoef([abs_log_rng, abs_log_spd, abs_log_jrk])

    print()
    print('=== R5: Δ₁ axis correlation across 80 test sequences ===')
    print(f'Pearson r on log(ER):')
    print(f'  log-range vs log-speed:  {corr_log[0,1]:+.3f}')
    print(f'  log-range vs log-jerk:   {corr_log[0,2]:+.3f}')
    print(f'  log-speed vs log-jerk:   {corr_log[1,2]:+.3f}')
    print(f'Pearson r on |log(ER)| (Δ₁ contributors):')
    print(f'  |log-range| vs |log-speed|:  {corr_abs[0,1]:+.3f}')
    print(f'  |log-range| vs |log-jerk|:   {corr_abs[0,2]:+.3f}')
    print(f'  |log-speed| vs |log-jerk|:   {corr_abs[1,2]:+.3f}')

    # Per-axis Δ₁ mean
    print(f'\nPer-axis |log(ER)| means (Δ₁ contributions):')
    print(f'  |log ER-range|: {abs_log_rng.mean():.3f}')
    print(f'  |log ER-speed|: {abs_log_spd.mean():.3f}')
    print(f'  |log ER-jerk|:  {abs_log_jrk.mean():.3f}')
    print(f'  Δ₁ (mean of 3): {(abs_log_rng.mean() + abs_log_spd.mean() + abs_log_jrk.mean())/3:.3f}')

    # === R9: per-performer ER variance ===
    print()
    print('=== R9: Per-performer ER (no identity conditioning) ===')
    by_perf = {}
    for s in per_seq:
        p = s['performer']
        by_perf.setdefault(p, []).append(s)
    for p in sorted(by_perf):
        seqs = by_perf[p]
        ranges = [s['er_range'] for s in seqs]
        speeds = [s['er_speed'] for s in seqs]
        jerks  = [s['er_jerk'] for s in seqs]
        print(f'  {p} (n={len(seqs)}): '
              f'ER-range {np.mean(ranges):.3f}±{np.std(ranges):.3f}, '
              f'ER-speed {np.mean(speeds):.3f}±{np.std(speeds):.3f}, '
              f'ER-jerk {np.mean(jerks):.3f}±{np.std(jerks):.3f}')

    # Save
    out = {
        'n_seq': len(per_seq),
        'corr_log_ER': corr_log.tolist(),
        'corr_abs_log_ER': corr_abs.tolist(),
        'per_axis_abs_log_ER_mean': {
            'range': float(abs_log_rng.mean()),
            'speed': float(abs_log_spd.mean()),
            'jerk':  float(abs_log_jrk.mean()),
        },
        'delta1_aggregate': float((abs_log_rng.mean() + abs_log_spd.mean() + abs_log_jrk.mean())/3),
        'per_performer': {
            p: {
                'n': len(seqs),
                'er_range_mean': float(np.mean([s['er_range'] for s in seqs])),
                'er_range_std':  float(np.std([s['er_range'] for s in seqs])),
                'er_speed_mean': float(np.mean([s['er_speed'] for s in seqs])),
                'er_speed_std':  float(np.std([s['er_speed'] for s in seqs])),
                'er_jerk_mean':  float(np.mean([s['er_jerk'] for s in seqs])),
                'er_jerk_std':   float(np.std([s['er_jerk'] for s in seqs])),
            }
            for p, seqs in by_perf.items()
        },
        'per_seq': per_seq,
    }
    out_path = PROJECT_ROOT / 'results/delta1_axis_correlation_and_perf.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
