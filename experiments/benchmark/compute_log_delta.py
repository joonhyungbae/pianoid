#!/usr/bin/env python3
"""Recompute symmetric Δ_log = mean|log ER| for all benchmarks, with bootstrap CI.

Matches the aggregate-level definition style of existing Δ_1 = mean|1-mean(ER)|:
  Δ_log = mean[|log mean(ER_r)|, |log mean(ER_s)|, |log mean(ER_j)|]
Bootstrap: resample sequences with replacement, recompute per-metric mean, then Δ_log.
"""
import sys
import math
import json
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


def compute_er_per_seq(pred, gt, joint_indices):
    """Return (er_range, er_speed, er_jerk) — one value per sequence, averaged over joint-axes."""
    pred = pred[:, joint_indices, :]
    gt = gt[:, joint_indices, :]
    J, A = pred.shape[1], pred.shape[2]
    r_list, s_list, j_list = [], [], []
    for j in range(J):
        for a in range(A):
            p, g = pred[:, j, a], gt[:, j, a]
            pr = p.max() - p.min()
            gr = g.max() - g.min()
            if gr > 1e-6:
                r_list.append(pr / gr)
            pv = np.abs(np.diff(p))
            gv = np.abs(np.diff(g))
            if gv.mean() > 1e-6:
                s_list.append(pv.mean() / gv.mean())
            pa = np.diff(p, n=3)
            ga = np.diff(g, n=3)
            if np.abs(ga).mean() > 1e-6:
                j_list.append(np.abs(pa).mean() / np.abs(ga).mean())
    return np.mean(r_list), np.mean(s_list), np.mean(j_list)


def load_results(results_dir, backbone=BACKBONE_INDICES):
    """Load all samples from a results dir, return arrays of per-seq ER values."""
    files = sorted(Path(results_dir).glob('sample_*.npz'))
    ers = []
    for f in files:
        d = np.load(f)
        gen_key = 'generated' if 'generated' in d.files else ('pred' if 'pred' in d.files else None)
        gt_key = 'gt' if 'gt' in d.files else ('gt_joints' if 'gt_joints' in d.files else None)
        if gen_key is None or gt_key is None:
            raise RuntimeError(f'unknown keys in {f}: {d.files}')
        gen, gt = d[gen_key], d[gt_key]
        if gen.ndim == 2:
            gen = gen.reshape(-1, 26, 3)
        if gt.ndim == 2:
            gt = gt.reshape(-1, 26, 3)
        ers.append(compute_er_per_seq(gen, gt, backbone))
    return np.array(ers)  # (N, 3)


def deltas(er_array, n_boot=2000, seed=42):
    """Compute aggregate-level Δ_1 = mean|1-mean(ER)| and Δ_log = mean|log mean(ER)|.
    Bootstrap CIs by resampling sequences."""
    er_mean = er_array.mean(axis=0)  # (3,)
    d1 = np.mean(np.abs(1 - er_mean))
    dlog = np.mean(np.abs(np.log(np.clip(er_mean, 1e-6, None))))
    # Bootstrap
    rng = np.random.RandomState(seed)
    n = len(er_array)
    d1_boots, dlog_boots = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        m = er_array[idx].mean(axis=0)
        d1_boots.append(np.mean(np.abs(1 - m)))
        dlog_boots.append(np.mean(np.abs(np.log(np.clip(m, 1e-6, None)))))
    d1_lo, d1_hi = np.percentile(d1_boots, [2.5, 97.5])
    dlog_lo, dlog_hi = np.percentile(dlog_boots, [2.5, 97.5])
    return {
        'er_range': float(er_mean[0]),
        'er_speed': float(er_mean[1]),
        'er_jerk':  float(er_mean[2]),
        'delta_1':  float(d1),
        'delta_1_ci': [float(d1_lo), float(d1_hi)],
        'delta_log': float(dlog),
        'delta_log_ci': [float(dlog_lo), float(dlog_hi)],
        'n_samples': n,
    }


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH = PROJECT_ROOT / 'experiments' / 'benchmark'
METHODS = {
    'pianoid (ours)':        str(BENCH / 'pianoid/results_final'),
    'mdm_adapted':           str(BENCH / 'mdm_adapted/checkpoints/results'),
    'edge_adapted':          str(BENCH / 'edge_adapted/checkpoints/results'),
    'single_stage_mdm+zs':   str(BENCH / 'single_stage_mdm_zscore/checkpoints/results'),
    'bilstm_backbone':       str(BENCH / 'bilstm_backbone/checkpoints/results'),
    'li2018':                str(BENCH / 'li2018/results'),
}

print(f'{"Method":<22} {"ER-r":>6} {"ER-s":>6} {"ER-j":>6} {"Δ1":>7}  {"Δ1 CI":<15} {"Δlog":>7}  {"Δlog CI":<15}  n')
print('-' * 115)
results = {}
for name, path in METHODS.items():
    try:
        er = load_results(path)
        r = deltas(er)
        results[name] = r
        d1_ci = f'[{r["delta_1_ci"][0]:.2f},{r["delta_1_ci"][1]:.2f}]'
        dlog_ci = f'[{r["delta_log_ci"][0]:.2f},{r["delta_log_ci"][1]:.2f}]'
        print(f'{name:<22} {r["er_range"]:>6.3f} {r["er_speed"]:>6.3f} {r["er_jerk"]:>6.3f} '
              f'{r["delta_1"]:>7.3f}  {d1_ci:<15} {r["delta_log"]:>7.3f}  {dlog_ci:<15}  {r["n_samples"]}')
    except Exception as e:
        print(f'{name:<22} ERROR: {e}')
        results[name] = {'error': str(e)}

Path('/tmp/log_delta_results.json').write_text(json.dumps(results, indent=2))
print(f'\nSaved: /tmp/log_delta_results.json')
