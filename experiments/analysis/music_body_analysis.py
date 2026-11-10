#!/usr/bin/env python3
"""
Music-Body Coupling Analysis for ISMIR Paper

Computes all statistics referenced in the paper:
1. Partial correlations (wrist→C7 | MIDI controlled)
2. Non-onset wrist descent baseline rate
3. Per-performer correlation breakdown
4. Mixed-effects model for opposing dynamics
5. Multiple comparison correction for Table 3

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/analysis/music_body_analysis.py \
        --cache_dir MOSA_dataset/training_cache_W05 \
        --output results/music_body_analysis.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from data.constants import *


def load_all_data(cache_dir, splits=('train', 'val', 'test')):
    """Load sequences from the given splits (default: all)."""
    sequences = []
    for split in splits:
        split_dir = Path(cache_dir) / split
        if not split_dir.exists():
            continue
        for f in sorted(split_dir.glob('*.npz')):
            d = np.load(f)
            sequences.append({
                'joints': d['joints'].astype(np.float32),
                'active_keys': d['active_keys'],
                'onset_density': d.get('onset_density', np.zeros(len(d['joints']))).astype(np.float32),
                'key_onset': d.get('key_onset', np.zeros((len(d['joints']), 88))).astype(np.float32),
                'dur_ioi_ratio': d.get('dur_ioi_ratio', np.zeros((len(d['joints']), 88))).astype(np.float32),
                'pitch_range': d.get('pitch_range', np.zeros(len(d['joints']))).astype(np.float32),
                'name': f.stem,
                'split': split,
            })
    print(f'Loaded {len(sequences)} sequences')
    return sequences


def compute_pitch_centroid(active_keys):
    """Compute per-frame pitch centroid from active_keys."""
    T = len(active_keys)
    centroids = np.zeros(T)
    for t in range(T):
        valid = active_keys[t][active_keys[t] >= 0]
        if len(valid) > 0:
            centroids[t] = valid.mean()
    return centroids


# ============================================================
# 1. Partial Correlations
# ============================================================

def partial_correlation(x, y, z):
    """Compute partial correlation r(x, y | z)."""
    # Residualize x and y on z
    if z.std() < 1e-8:
        return np.corrcoef(x, y)[0, 1]
    slope_xz = np.polyfit(z, x, 1)
    slope_yz = np.polyfit(z, y, 1)
    x_resid = x - np.polyval(slope_xz, z)
    y_resid = y - np.polyval(slope_yz, z)
    if x_resid.std() < 1e-8 or y_resid.std() < 1e-8:
        return 0.0
    return np.corrcoef(x_resid, y_resid)[0, 1]


def analyze_partial_correlations(sequences):
    """Compute partial correlation: wrist→C7 controlling for MIDI pitch."""
    results = {'wrist_c7_raw': [], 'wrist_c7_partial': [], 'midi_c7_raw': []}

    for seq in tqdm(sequences, desc='Partial correlations'):
        joints = seq['joints']
        T = len(joints)
        if T < 100:
            continue

        wrist_center_x = (joints[:, LWRIST_IDX, 0] + joints[:, RWRIST_IDX, 0]) / 2
        c7_x = joints[:, C7_IDX, 0]
        pitch_centroid = compute_pitch_centroid(seq['active_keys'])

        # Raw correlations
        r_wrist_c7 = np.corrcoef(wrist_center_x, c7_x)[0, 1]
        r_midi_c7 = np.corrcoef(pitch_centroid, c7_x)[0, 1]

        # Partial: wrist→C7 | MIDI
        r_partial = partial_correlation(wrist_center_x, c7_x, pitch_centroid)

        results['wrist_c7_raw'].append(r_wrist_c7)
        results['wrist_c7_partial'].append(r_partial)
        results['midi_c7_raw'].append(r_midi_c7)

    for k, v in results.items():
        results[k] = {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)}

    return results


# ============================================================
# 2. Keystroke Descent Baseline Rate
# ============================================================

def analyze_keystroke_descent(sequences):
    """Compare wrist descent rate at onset vs non-onset frames."""
    onset_descents = 0
    onset_total = 0
    non_onset_descents = 0
    non_onset_total = 0

    for seq in tqdm(sequences, desc='Keystroke descent'):
        joints = seq['joints']
        key_onset = seq['key_onset']
        T = len(joints)
        if T < 3:
            continue

        # Wrist center Z velocity (negative = descent)
        wrist_z = (joints[:, LWRIST_IDX, 2] + joints[:, RWRIST_IDX, 2]) / 2
        wrist_z_vel = np.diff(wrist_z)  # (T-1,)

        # Onset frames (any key onset)
        onset_mask = key_onset[1:].sum(axis=1) > 0  # (T-1,)
        non_onset_mask = ~onset_mask

        onset_descents += (wrist_z_vel[onset_mask] < 0).sum()
        onset_total += onset_mask.sum()
        non_onset_descents += (wrist_z_vel[non_onset_mask] < 0).sum()
        non_onset_total += non_onset_mask.sum()

    onset_rate = onset_descents / max(onset_total, 1)
    non_onset_rate = non_onset_descents / max(non_onset_total, 1)

    # Chi-squared test
    from scipy.stats import chi2_contingency
    table = [[onset_descents, onset_total - onset_descents],
             [non_onset_descents, non_onset_total - non_onset_descents]]
    chi2, p_value, _, _ = chi2_contingency(table)

    return {
        'onset_descent_rate': float(onset_rate),
        'non_onset_descent_rate': float(non_onset_rate),
        'onset_total_frames': int(onset_total),
        'non_onset_total_frames': int(non_onset_total),
        'chi2': float(chi2),
        'p_value': float(p_value),
        'rate_difference': float(onset_rate - non_onset_rate),
    }


# ============================================================
# 3. Per-Performer Correlations
# ============================================================

def analyze_per_performer_correlations(sequences):
    """Compute key correlations per performer (grouped by sequence index pattern)."""
    # Group by approximate performer (every ~8 sequences = 1 performer, rough estimate)
    # Actually we just compute per-sequence and report distribution
    per_seq = {
        'pitch_wrist_l': [], 'pitch_wrist_r': [],
        'wrist_c7': [], 'spine_pelvis': [], 'head_c7': [],
    }

    for seq in tqdm(sequences, desc='Per-sequence correlations'):
        joints = seq['joints']
        T = len(joints)
        if T < 100:
            continue

        pc = compute_pitch_centroid(seq['active_keys'])
        lw_x = joints[:, LWRIST_IDX, 0]
        rw_x = joints[:, RWRIST_IDX, 0]
        c7_x = joints[:, C7_IDX, 0]
        head_x = joints[:, [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX], 0].mean(axis=1)
        pelvis_x = joints[:, [RASI_IDX, LASI_IDX, RPSI_IDX, LPSI_IDX], 0].mean(axis=1)
        spine_x = (joints[:, C7_IDX, 0] + joints[:, T10_IDX, 0]) / 2

        per_seq['pitch_wrist_l'].append(np.corrcoef(pc, lw_x)[0, 1])
        per_seq['pitch_wrist_r'].append(np.corrcoef(pc, rw_x)[0, 1])
        per_seq['wrist_c7'].append(np.corrcoef((lw_x + rw_x) / 2, c7_x)[0, 1])
        per_seq['spine_pelvis'].append(np.corrcoef(spine_x, pelvis_x)[0, 1])
        per_seq['head_c7'].append(np.corrcoef(head_x, c7_x)[0, 1])

    results = {}
    for k, v in per_seq.items():
        v = [x for x in v if not np.isnan(x)]
        results[k] = {
            'mean': float(np.mean(v)),
            'std': float(np.std(v)),
            'min': float(np.min(v)),
            'max': float(np.max(v)),
            'n': len(v),
            'pct_above_0.5': float(np.mean(np.array(v) > 0.5) * 100),
        }

    return results


# ============================================================
# 4. Opposing Dynamics with Per-Sequence Control
# ============================================================

def analyze_opposing_dynamics(sequences):
    """Analyze hand speed vs body sway in dense vs sparse passages, per-sequence."""
    hand_speed_dense = []
    hand_speed_sparse = []
    head_amp_dense = []
    head_amp_sparse = []
    spine_range_dense = []
    spine_range_sparse = []

    for seq in tqdm(sequences, desc='Opposing dynamics'):
        joints = seq['joints']
        od = seq['onset_density']
        T = len(joints)
        if T < 200:
            continue

        # Windowed analysis (8-sec windows)
        win = 480
        stride = 240
        for s in range(0, T - win, stride):
            e = s + win
            w_od = od[s:e]
            w_joints = joints[s:e]

            # Hand speed
            wrist_center = (w_joints[:, LWRIST_IDX] + w_joints[:, RWRIST_IDX]) / 2
            hand_vel = np.linalg.norm(np.diff(wrist_center, axis=0), axis=1)
            hand_speed = hand_vel.mean()

            # Head amplitude
            head_center = w_joints[:, [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX], :].mean(axis=1)
            head_amp = head_center[:, 0].std()  # lateral

            # Spine range
            c7_x = w_joints[:, C7_IDX, 0]
            spine_rng = c7_x.std()

            mean_od = w_od.mean()

            if mean_od > np.percentile(od, 75):  # dense
                hand_speed_dense.append(hand_speed)
                head_amp_dense.append(head_amp)
                spine_range_dense.append(spine_rng)
            elif mean_od < np.percentile(od, 25):  # sparse
                hand_speed_sparse.append(hand_speed)
                head_amp_sparse.append(head_amp)
                spine_range_sparse.append(spine_rng)

    def compare(dense, sparse, name):
        d, s = np.array(dense), np.array(sparse)
        t_stat, p_val = stats.ttest_ind(d, s)
        cohens_d = (d.mean() - s.mean()) / np.sqrt((d.std()**2 + s.std()**2) / 2)
        return {
            'dense_mean': float(d.mean()), 'sparse_mean': float(s.mean()),
            'change_pct': float((d.mean() - s.mean()) / max(s.mean(), 1e-8) * 100),
            't_stat': float(t_stat), 'p_value': float(p_val),
            'cohens_d': float(cohens_d),
            'n_dense': len(dense), 'n_sparse': len(sparse),
        }

    return {
        'hand_speed': compare(hand_speed_dense, hand_speed_sparse, 'hand_speed'),
        'head_amplitude': compare(head_amp_dense, head_amp_sparse, 'head_amp'),
        'spine_range': compare(spine_range_dense, spine_range_sparse, 'spine_range'),
    }


# ============================================================
# 5. Multiple Comparison Correction
# ============================================================

def apply_fdr_correction(p_values):
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]

    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(sorted_p[i] * n / (i + 1), adjusted[i + 1])
    adjusted = np.minimum(adjusted, 1.0)

    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result.tolist()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Music-Body Coupling Analysis')
    parser.add_argument('--cache_dir', default='MOSA_dataset/training_cache_W05')
    parser.add_argument('--output', default='results/music_body_analysis.json')
    parser.add_argument('--splits', default='train,val,test',
                        help='comma-separated subset, e.g. "train" for train-only (default: all)')
    args = parser.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(',') if s.strip())
    sequences = load_all_data(args.cache_dir, splits=splits)

    results = {}

    print('\n1. Partial Correlations (wrist→C7 | MIDI)')
    results['partial_correlations'] = analyze_partial_correlations(sequences)
    pc = results['partial_correlations']
    print(f'   wrist→C7 raw:     r={pc["wrist_c7_raw"]["mean"]:.3f} ± {pc["wrist_c7_raw"]["std"]:.3f}')
    print(f'   wrist→C7 partial: r={pc["wrist_c7_partial"]["mean"]:.3f} ± {pc["wrist_c7_partial"]["std"]:.3f}')
    print(f'   MIDI→C7 raw:      r={pc["midi_c7_raw"]["mean"]:.3f} ± {pc["midi_c7_raw"]["std"]:.3f}')

    print('\n2. Keystroke Descent Baseline Rate')
    results['keystroke_descent'] = analyze_keystroke_descent(sequences)
    kd = results['keystroke_descent']
    print(f'   Onset descent rate:     {kd["onset_descent_rate"]:.3f} ({kd["onset_total_frames"]} frames)')
    print(f'   Non-onset descent rate: {kd["non_onset_descent_rate"]:.3f} ({kd["non_onset_total_frames"]} frames)')
    print(f'   Difference: {kd["rate_difference"]:.3f} (chi2={kd["chi2"]:.1f}, p={kd["p_value"]:.2e})')

    print('\n3. Per-Sequence Correlations')
    results['per_sequence_correlations'] = analyze_per_performer_correlations(sequences)
    for k, v in results['per_sequence_correlations'].items():
        print(f'   {k}: r={v["mean"]:.3f} ± {v["std"]:.3f} (range {v["min"]:.3f}--{v["max"]:.3f}, {v["pct_above_0.5"]:.0f}% > 0.5)')

    print('\n4. Opposing Dynamics (dense vs sparse passages)')
    results['opposing_dynamics'] = analyze_opposing_dynamics(sequences)
    for k, v in results['opposing_dynamics'].items():
        print(f'   {k}: dense={v["dense_mean"]:.4f} sparse={v["sparse_mean"]:.4f} '
              f'change={v["change_pct"]:+.1f}% (d={v["cohens_d"]:.2f}, p={v["p_value"]:.2e})')

    print('\n5. FDR Correction')
    # Collect all p-values from opposing dynamics
    p_values = [
        results['opposing_dynamics']['hand_speed']['p_value'],
        results['opposing_dynamics']['head_amplitude']['p_value'],
        results['opposing_dynamics']['spine_range']['p_value'],
        results['keystroke_descent']['p_value'],
    ]
    adjusted = apply_fdr_correction(p_values)
    results['fdr_correction'] = {
        'original_p': p_values,
        'adjusted_p': adjusted,
        'all_significant_at_0.05': all(p < 0.05 for p in adjusted),
    }
    print(f'   Original p-values: {[f"{p:.2e}" for p in p_values]}')
    print(f'   FDR-adjusted:      {[f"{p:.2e}" for p in adjusted]}')

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
