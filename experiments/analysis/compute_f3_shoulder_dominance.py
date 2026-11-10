#!/usr/bin/env python3
"""F3: shoulder-torso vs shoulder-wrist correlation — per-piece breakdown."""
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr, binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from data.constants import (LSHO_IDX, RSHO_IDX, C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
                             RWRIST_IDX, LWRIST_IDX)

CACHE = PROJECT_ROOT / 'MOSA_dataset/training_cache_W05'
TORSO_IDXS = [C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX]


def compute_corr(joints, idx_a, idx_b, axis=0):
    """Mean abs correlation between joint-a axis and joint-b axis across joints."""
    a = joints[:, idx_a, axis] if isinstance(idx_a, int) else np.mean(joints[:, idx_a, axis], axis=-1)
    b = joints[:, idx_b, axis] if isinstance(idx_b, int) else np.mean(joints[:, idx_b, axis], axis=-1)
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return None
    return pearsonr(a, b)[0]


def load_split(split):
    meta = json.loads((CACHE / 'metadata.json').read_text())
    return meta['splits'][split]['samples']


def analyze(split_name):
    samples = load_split(split_name)
    # group by (piece) — paper says 14 pieces
    # or by (piece, performer) — but paper says 14 pieces
    by_piece = defaultdict(list)
    for s in samples:
        npz = np.load(CACHE / split_name / f'{s["idx"]:05d}.npz')
        by_piece[s['piece']].append(npz['joints'])  # (T, 26, 3)

    print(f'\n=== {split_name}: {len(samples)} sequences, {len(by_piece)} pieces ===')

    # For each piece: pool all performer trajectories, compute mean abs corr
    results = {}
    for piece, arrs in sorted(by_piece.items()):
        # concat along time (all performers together)
        J = np.concatenate(arrs, axis=0)  # (sumT, 26, 3)
        # shoulder-torso: mean corr over {L,R shoulder} × {C7, CLAV, T10, STRN} × 3 axes
        r_st = []
        for sh in [LSHO_IDX, RSHO_IDX]:
            for tr in TORSO_IDXS:
                for ax in range(3):
                    c = compute_corr(J, sh, tr, axis=ax)
                    if c is not None:
                        r_st.append(abs(c))
        # shoulder-wrist (ipsilateral): LSHO-LWRIST, RSHO-RWRIST × 3 axes
        r_sw = []
        for sh, wr in [(LSHO_IDX, LWRIST_IDX), (RSHO_IDX, RWRIST_IDX)]:
            for ax in range(3):
                c = compute_corr(J, sh, wr, axis=ax)
                if c is not None:
                    r_sw.append(abs(c))
        results[piece] = {
            'shoulder_torso_mean': float(np.mean(r_st)),
            'shoulder_torso_min': float(np.min(r_st)),
            'shoulder_torso_max': float(np.max(r_st)),
            'shoulder_wrist_mean': float(np.mean(r_sw)),
            'shoulder_wrist_min': float(np.min(r_sw)),
            'shoulder_wrist_max': float(np.max(r_sw)),
            'dominance': float(np.mean(r_st) > np.mean(r_sw)),
        }
        print(f'  {piece:<8} shldr-torso r={results[piece]["shoulder_torso_mean"]:.3f} '
              f'[{results[piece]["shoulder_torso_min"]:.2f},{results[piece]["shoulder_torso_max"]:.2f}]  '
              f'vs shldr-wrist r={results[piece]["shoulder_wrist_mean"]:.3f} '
              f'[{results[piece]["shoulder_wrist_min"]:.2f},{results[piece]["shoulder_wrist_max"]:.2f}]  '
              f'dominance={int(results[piece]["dominance"])}')

    # aggregate
    all_st = np.mean([r['shoulder_torso_mean'] for r in results.values()])
    all_sw = np.mean([r['shoulder_wrist_mean'] for r in results.values()])
    n_dom = sum(1 for r in results.values() if r['dominance'])
    n_total = len(results)
    p = binomtest(n_dom, n_total, 0.5, alternative='greater').pvalue

    # Range summary (over all pieces)
    all_st_mins = [r['shoulder_torso_mean'] for r in results.values()]
    all_sw_means = [r['shoulder_wrist_mean'] for r in results.values()]
    print(f'\n  OVERALL: shldr-torso={np.mean(all_st_mins):.3f} '
          f'[{np.min(all_st_mins):.2f}-{np.max(all_st_mins):.2f}]  vs  '
          f'shldr-wrist={np.mean(all_sw_means):.3f} '
          f'[{np.min(all_sw_means):.2f}-{np.max(all_sw_means):.2f}]')
    print(f'  Dominance: {n_dom}/{n_total} pieces, binomial p={p:.2e}')
    return results


print('─── TRAIN-ONLY ANALYSIS (F3: paper claims 0.73-0.82 vs 0.39-0.45, 14/14 pieces) ───')
analyze('train')
print('\n─── ALL SPLITS (for comparison; paper reported on all 272) ───')
# combine
samples_all = []
for split in ['train', 'val', 'test']:
    for s in load_split(split):
        s['_split'] = split
        samples_all.append(s)
by_piece_all = defaultdict(list)
for s in samples_all:
    npz = np.load(CACHE / s['_split'] / f'{s["idx"]:05d}.npz')
    by_piece_all[s['piece']].append(npz['joints'])
print(f'Total sequences across all splits: {len(samples_all)}, {len(by_piece_all)} pieces')

# Full analysis for comparison
print('\n─── FULL (all 272 seq, 14 pieces) ───')
by_piece_all = defaultdict(list)
for s in samples_all:
    npz = np.load(CACHE / s['_split'] / f'{s["idx"]:05d}.npz')
    by_piece_all[s['piece']].append(npz['joints'])

results_all = {}
for piece, arrs in sorted(by_piece_all.items()):
    J = np.concatenate(arrs, axis=0)
    r_st = []
    for sh in [LSHO_IDX, RSHO_IDX]:
        for tr in TORSO_IDXS:
            for ax in range(3):
                c = compute_corr(J, sh, tr, axis=ax)
                if c is not None: r_st.append(abs(c))
    r_sw = []
    for sh, wr in [(LSHO_IDX, LWRIST_IDX), (RSHO_IDX, RWRIST_IDX)]:
        for ax in range(3):
            c = compute_corr(J, sh, wr, axis=ax)
            if c is not None: r_sw.append(abs(c))
    results_all[piece] = {
        'st': float(np.mean(r_st)), 'sw': float(np.mean(r_sw)),
    }
    dom = 'DOM' if results_all[piece]['st'] > results_all[piece]['sw'] else 'no'
    print(f'  {piece:<8} shldr-torso={results_all[piece]["st"]:.3f}  shldr-wrist={results_all[piece]["sw"]:.3f}  [{dom}]')

all_st = [r['st'] for r in results_all.values()]
all_sw = [r['sw'] for r in results_all.values()]
n_dom = sum(1 for r in results_all.values() if r['st'] > r['sw'])
print(f'\n  OVERALL: shldr-torso={np.mean(all_st):.3f} [{np.min(all_st):.2f}-{np.max(all_st):.2f}]  '
      f'vs  shldr-wrist={np.mean(all_sw):.3f} [{np.min(all_sw):.2f}-{np.max(all_sw):.2f}]')
from scipy.stats import binomtest
print(f'  Dominance: {n_dom}/{len(results_all)}, binomial p={binomtest(n_dom, len(results_all), 0.5, alternative="greater").pvalue:.2e}')
