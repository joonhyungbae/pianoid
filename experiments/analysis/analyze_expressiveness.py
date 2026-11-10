#!/usr/bin/env python3
"""
Analyze expressiveness gap between GT and generated motion.

Compares motion amplitude, speed, range of motion per body part.

Usage:
    python experiments/analysis/analyze_expressiveness.py
"""

import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from data.constants import *

PRED_DIR = Path('MOSA_dataset/full_body_predictions/test')

BODY_GROUPS = {
    'Wrists': [LWRIST_IDX, RWRIST_IDX],
    'Head': [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX],
    'Torso (neck)': [C7_IDX, CLAV_IDX],
    'Torso (chest)': [T10_IDX, STRN_IDX],
    'Shoulders': [RSHO_IDX, LSHO_IDX],
    'Elbows': [RELB_IDX, LELB_IDX],
    'Pelvis': [RASI_IDX, LASI_IDX, RPSI_IDX, LPSI_IDX],
    'Legs': [RKNE_IDX, RANKLE_IDX, RTOE_IDX, LKNE_IDX, LANKLE_IDX, LTOE_IDX],
}


def analyze_motion(motion):
    """Compute motion statistics: std, speed, range per body group."""
    T = len(motion)
    results = {}

    for name, indices in BODY_GROUPS.items():
        joints = motion[:, indices, :]  # (T, N, 3)

        # Position std (mm) - how much does it move overall
        pos_std = joints.reshape(T, -1).std(axis=0).mean() * 1000

        # Speed (mm/frame) - how fast does it move
        vel = np.diff(joints, axis=0)  # (T-1, N, 3)
        speed = np.linalg.norm(vel, axis=-1).mean() * 1000  # avg speed in mm/frame

        # Range of motion (mm) - max extent
        rom = (joints.max(axis=0) - joints.min(axis=0)).mean() * 1000

        # Lateral range specifically (X axis)
        lateral_range = (joints[:, :, 0].max(axis=0) - joints[:, :, 0].min(axis=0)).mean() * 1000

        # Vertical range (Z in MOSA = height)
        vertical_range = (joints[:, :, 2].max(axis=0) - joints[:, :, 2].min(axis=0)).mean() * 1000

        results[name] = {
            'pos_std': pos_std,
            'speed': speed,
            'rom': rom,
            'lateral_range': lateral_range,
            'vertical_range': vertical_range,
        }

    return results


def main():
    gt_stats_all = {name: {k: [] for k in ['pos_std', 'speed', 'rom', 'lateral_range', 'vertical_range']}
                    for name in BODY_GROUPS}
    gen_stats_all = {name: {k: [] for k in ['pos_std', 'speed', 'rom', 'lateral_range', 'vertical_range']}
                     for name in BODY_GROUPS}

    n_samples = 0
    for i in range(80):
        d = PRED_DIR / f'{i:05d}'
        if not d.exists():
            continue

        gt = np.load(d / 'gt_motion.npy')
        gen = np.load(d / 'generated_motion.npy')
        T = min(len(gt), len(gen))
        gt, gen = gt[:T], gen[:T]

        gt_stats = analyze_motion(gt)
        gen_stats = analyze_motion(gen)

        for name in BODY_GROUPS:
            for k in gt_stats[name]:
                gt_stats_all[name][k].append(gt_stats[name][k])
                gen_stats_all[name][k].append(gen_stats[name][k])

        n_samples += 1

    # Print comparison
    print(f'Expressiveness Analysis: GT vs Generated ({n_samples} test sequences)')
    print('=' * 100)

    print(f'\n{"Body Part":<18} {"Metric":<18} {"GT":<12} {"Gen":<12} {"Ratio":<10} {"Gap":<10}')
    print('-' * 100)

    all_results = {}

    for name in BODY_GROUPS:
        for metric in ['pos_std', 'speed', 'rom', 'lateral_range', 'vertical_range']:
            gt_val = np.mean(gt_stats_all[name][metric])
            gen_val = np.mean(gen_stats_all[name][metric])
            ratio = gen_val / gt_val if gt_val > 0 else 0
            gap = gen_val - gt_val

            metric_label = {
                'pos_std': 'Pos Std (mm)',
                'speed': 'Speed (mm/f)',
                'rom': 'Range (mm)',
                'lateral_range': 'Lateral (mm)',
                'vertical_range': 'Vertical (mm)',
            }[metric]

            # Highlight concerning ratios
            flag = ' ⚠️' if ratio < 0.7 else ''
            print(f'{name:<18} {metric_label:<18} {gt_val:<12.2f} {gen_val:<12.2f} {ratio:<10.2f} {gap:<+10.2f}{flag}')

            if name not in all_results:
                all_results[name] = {}
            all_results[name][metric] = {
                'gt': round(gt_val, 2),
                'gen': round(gen_val, 2),
                'ratio': round(ratio, 2),
            }

        print()

    # Summary: which parts are most under-expressive
    print('\n' + '=' * 100)
    print('SUMMARY: Gen/GT ratio by body part (< 0.7 = significantly under-expressive)')
    print('=' * 100)
    for name in BODY_GROUPS:
        speed_ratio = all_results[name]['speed']['ratio']
        rom_ratio = all_results[name]['rom']['ratio']
        lat_ratio = all_results[name]['lateral_range']['ratio']
        flag = ' ⚠️ UNDER-EXPRESSIVE' if min(speed_ratio, rom_ratio) < 0.7 else ''
        print(f'  {name:<18} speed={speed_ratio:.2f}x  range={rom_ratio:.2f}x  lateral={lat_ratio:.2f}x{flag}')

    # Save
    Path('results').mkdir(exist_ok=True)
    with open('results/expressiveness_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved to results/expressiveness_analysis.json')


if __name__ == '__main__':
    main()
