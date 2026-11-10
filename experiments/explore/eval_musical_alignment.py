#!/usr/bin/env python3
"""
Evaluate musical structure alignment in generated motion.
No retraining needed — runs on existing generated outputs + MOSA annotations.

Metrics:
1. Beat Alignment Score (BAS): body velocity peaks vs GT beat positions
2. Downbeat Emphasis Ratio: motion amplitude at downbeats vs regular beats
3. Phrase Boundary Reset: body displacement spike at phrase boundaries
4. Anticipatory Lag Preservation: 65ms body-before-hand lag in generated motion
5. Per-body-part MPJPE breakdown
"""

import json
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import signal, stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data.constants import (
    FPS, RWRIST_IDX, LWRIST_IDX, RHAND_IDX, LHAND_IDX,
)

MOTION_FPS = 60  # generated/preprocessed motion is 60fps
DATASET_ROOT = ROOT / "MOSA_dataset"

# Joint group indices in the 26-joint skeleton
JOINT_GROUPS = {
    'head': [0, 1, 2, 3],           # RFHD, LFHD, RBHD, LBHD
    'neck': [4, 5],                  # C7, CLAV
    'chest': [6, 7],                 # T10, STRN
    'shoulders': [8, 12],            # RSHO, LSHO
    'elbows': [9, 13],               # RELB, LELB
    'wrists': [10, 14],              # RWRIST, LWRIST
    'hands': [11, 15],               # RHAND, LHAND
    'pelvis': [16, 17, 18, 19],      # RASI, LASI, RPSI, LPSI
    'legs': [20, 21, 22, 23, 24, 25],# knees, ankles, toes
}

SPINE_JOINTS = [4, 5, 6, 7]  # C7, CLAV, T10, STRN
HEAD_JOINTS = [0, 1, 2, 3]
HAND_JOINTS = [10, 11, 14, 15]  # wrists + hands


def load_splits():
    with open(DATASET_ROOT / "splits.json") as f:
        return json.load(f)


def build_score_to_perf_map(align_df):
    df = align_df.sort_values('score_position')
    sp = df['score_position'].values
    pt = df['onset'].values
    _, idx = np.unique(sp, return_index=True)
    sp, pt = sp[idx], pt[idx]
    return lambda x: np.interp(x, sp, pt)


def compute_velocity(motion, fps=MOTION_FPS):
    """Compute per-frame velocity magnitude for each joint. (T-1, J)"""
    diff = np.diff(motion, axis=0)  # (T-1, J, 3)
    return np.linalg.norm(diff, axis=-1) * fps  # mm/s


def compute_group_velocity(motion, joint_indices, fps=MOTION_FPS):
    """Compute mean velocity of a joint group. Returns (T-1,)"""
    vel = compute_velocity(motion, fps)
    return vel[:, joint_indices].mean(axis=1)


def find_velocity_peaks(velocity, distance=None, prominence=None):
    """Find peaks in velocity signal."""
    if distance is None:
        distance = int(0.2 * MOTION_FPS)  # min 200ms between peaks
    peaks, props = signal.find_peaks(velocity, distance=distance, prominence=prominence)
    return peaks


# ============================================================
# Metric 1: Beat Alignment Score (BAS)
# ============================================================

def compute_beat_alignment_score(motion, beat_times, fps=MOTION_FPS, tolerance_ms=100):
    """
    Beat Alignment Score: fraction of beats that have a body velocity peak nearby.
    Uses spine + head velocity as the "body kinematic beat."
    """
    body_vel = compute_group_velocity(motion, SPINE_JOINTS + HEAD_JOINTS, fps)

    # Smooth slightly to avoid noise peaks
    kernel_size = max(3, int(0.05 * fps))
    if kernel_size % 2 == 0:
        kernel_size += 1
    body_vel_smooth = np.convolve(body_vel, np.ones(kernel_size)/kernel_size, mode='same')

    peaks = find_velocity_peaks(body_vel_smooth, distance=int(0.15 * fps))
    peak_times = peaks / fps

    tol = tolerance_ms / 1000.0
    aligned = 0
    for bt in beat_times:
        if len(peak_times) == 0:
            break
        min_dist = np.min(np.abs(peak_times - bt))
        if min_dist <= tol:
            aligned += 1

    if len(beat_times) == 0:
        return np.nan
    return aligned / len(beat_times)


# ============================================================
# Metric 2: Downbeat Emphasis Ratio
# ============================================================

def compute_downbeat_emphasis(motion, beat_times, downbeat_times, fps=MOTION_FPS):
    """
    Ratio of mean body velocity at downbeats vs regular beats.
    > 1.0 means the model emphasizes downbeats (musically correct).
    """
    body_vel = compute_group_velocity(motion, SPINE_JOINTS + HEAD_JOINTS, fps)

    context = int(0.1 * fps)  # 100ms window around each beat
    downbeat_set = set(np.round(downbeat_times, 3).tolist())

    downbeat_vels = []
    beat_vels = []

    for bt in beat_times:
        frame = int(bt * fps)
        start = max(0, frame - context)
        end = min(len(body_vel), frame + context)
        if end <= start:
            continue

        mean_vel = np.mean(body_vel[start:end])
        if round(bt, 3) in downbeat_set:
            downbeat_vels.append(mean_vel)
        else:
            beat_vels.append(mean_vel)

    if not downbeat_vels or not beat_vels:
        return np.nan
    return np.mean(downbeat_vels) / (np.mean(beat_vels) + 1e-8)


# ============================================================
# Metric 3: Phrase Boundary Reset
# ============================================================

def compute_phrase_boundary_displacement(motion, phrase_boundaries, fps=MOTION_FPS, window_sec=1.0):
    """
    Measure body displacement around phrase boundaries vs mid-phrase.
    Higher displacement at boundaries indicates the model learned phrase-level structure.
    """
    spine_pos = motion[:, SPINE_JOINTS, :].mean(axis=1)  # (T, 3)
    spine_disp = np.linalg.norm(np.diff(spine_pos, axis=0), axis=1)  # (T-1,)

    window = int(window_sec * fps)

    boundary_disps = []
    for pb in phrase_boundaries:
        frame = int(pb * fps)
        start = max(0, frame - window // 2)
        end = min(len(spine_disp), frame + window // 2)
        if end - start < window // 4:
            continue
        boundary_disps.append(np.mean(spine_disp[start:end]))

    # Mid-phrase displacement (sample random mid-phrase points)
    total_frames = len(spine_disp)
    mid_disps = []
    boundary_frames = set(int(pb * fps) for pb in phrase_boundaries)

    np.random.seed(42)
    for _ in range(len(phrase_boundaries) * 3):
        frame = np.random.randint(window, total_frames - window)
        if any(abs(frame - bf) < window for bf in boundary_frames):
            continue
        start = max(0, frame - window // 2)
        end = min(total_frames, frame + window // 2)
        mid_disps.append(np.mean(spine_disp[start:end]))

    if not boundary_disps or not mid_disps:
        return np.nan, np.nan, np.nan

    ratio = np.mean(boundary_disps) / (np.mean(mid_disps) + 1e-8)
    t_stat, p_val = stats.ttest_ind(boundary_disps, mid_disps) if len(boundary_disps) > 1 and len(mid_disps) > 1 else (0, 1)
    return ratio, t_stat, p_val


# ============================================================
# Metric 4: Anticipatory Lag Preservation
# ============================================================

def compute_anticipatory_lag(motion, fps=MOTION_FPS, max_lag_ms=200):
    """
    Cross-correlation between spine lateral (X) velocity and hand lateral velocity.
    Negative lag = body leads hand (anticipatory motion).
    Returns peak lag in ms.
    """
    hand_x = motion[:, HAND_JOINTS, 0].mean(axis=1)  # (T,)
    spine_x = motion[:, SPINE_JOINTS, 0].mean(axis=1)

    hand_vel = np.diff(hand_x)
    spine_vel = np.diff(spine_x)

    # Normalize
    hand_vel = (hand_vel - hand_vel.mean()) / (hand_vel.std() + 1e-8)
    spine_vel = (spine_vel - spine_vel.mean()) / (spine_vel.std() + 1e-8)

    max_lag_frames = int(max_lag_ms / 1000 * fps)
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)

    corrs = np.correlate(hand_vel, spine_vel, mode='full')
    center = len(corrs) // 2
    corr_window = corrs[center - max_lag_frames:center + max_lag_frames + 1]

    if len(corr_window) != len(lags):
        return np.nan, np.nan

    corr_window = corr_window / len(hand_vel)
    peak_idx = np.argmax(corr_window)
    peak_lag_frames = lags[peak_idx]
    peak_lag_ms = peak_lag_frames / fps * 1000
    peak_corr = corr_window[peak_idx]

    return peak_lag_ms, peak_corr


# ============================================================
# Metric 5: Per-body-part MPJPE
# ============================================================

def compute_bodypart_mpjpe(gt, pred):
    """Per-body-group MPJPE in mm."""
    T = min(len(gt), len(pred))
    gt, pred = gt[:T], pred[:T]

    results = {}
    for group, indices in JOINT_GROUPS.items():
        err = np.linalg.norm(gt[:, indices] - pred[:, indices], axis=-1)  # (T, n_joints)
        results[group] = float(np.mean(err) * 1000)  # convert to mm
    results['overall'] = float(np.mean(np.linalg.norm(gt - pred, axis=-1)) * 1000)
    return results


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_sample(sample, gen_root, data_root):
    """Evaluate a single test sample on all musical alignment metrics."""
    piece = sample['piece']
    performer = sample['performer']
    trial = sample['trial']
    sample_path = f"yp/{piece}/{performer}/{trial}"

    # Load generated motion
    gen_path = gen_root / sample_path / "generated_motion.npy"
    if not gen_path.exists():
        return None
    gen_motion = np.load(gen_path)  # (T, 26, 3)

    # Load GT motion (preprocessed 26j)
    gt_path = data_root / "preprocessed" / sample_path / "motion_26j.npy"
    if not gt_path.exists():
        return None
    gt_motion = np.load(gt_path)  # (T, 26, 3)

    # Load annotations
    ann_dir = data_root / sample_path / "annotation" / "annotations"
    align_file = ann_dir / f"{piece}_{performer}_{trial}_align_notetime.csv"
    beat_file = ann_dir / f"{piece}_{performer}_{trial}_beat.csv"
    downbeat_file = ann_dir / f"{piece}_{performer}_{trial}_downbeat.csv"
    phrase_file = ann_dir / f"{piece}_{performer}_{trial}_phrase_section.csv"

    if not all(f.exists() for f in [align_file, beat_file, downbeat_file]):
        return None

    align_df = pd.read_csv(align_file)
    score_to_perf = build_score_to_perf_map(align_df)

    # Convert score-position beats to performance time
    beats_score = pd.read_csv(beat_file).iloc[:, 0].values
    downbeats_score = pd.read_csv(downbeat_file).iloc[:, 0].values
    beat_times = score_to_perf(beats_score)
    downbeat_times = score_to_perf(downbeats_score)

    # Phrase boundaries
    phrase_boundaries = []
    if phrase_file.exists():
        phrase_df = pd.read_csv(phrase_file)
        if len(phrase_df) > 0:
            for _, row in phrase_df.iterrows():
                phrase_boundaries.append(score_to_perf(row['onset']))

    result = {'sample': f"{piece}/{performer}/{trial}"}

    # 1. Beat Alignment Score
    result['bas_gen'] = compute_beat_alignment_score(gen_motion, beat_times)
    result['bas_gt'] = compute_beat_alignment_score(gt_motion, beat_times)

    # 2. Downbeat Emphasis
    result['downbeat_emphasis_gen'] = compute_downbeat_emphasis(gen_motion, beat_times, downbeat_times)
    result['downbeat_emphasis_gt'] = compute_downbeat_emphasis(gt_motion, beat_times, downbeat_times)

    # 3. Phrase Boundary Reset
    if phrase_boundaries:
        ratio_gen, _, _ = compute_phrase_boundary_displacement(gen_motion, phrase_boundaries)
        ratio_gt, _, _ = compute_phrase_boundary_displacement(gt_motion, phrase_boundaries)
        result['phrase_reset_gen'] = ratio_gen
        result['phrase_reset_gt'] = ratio_gt
    else:
        result['phrase_reset_gen'] = np.nan
        result['phrase_reset_gt'] = np.nan

    # 4. Anticipatory Lag
    lag_gen, corr_gen = compute_anticipatory_lag(gen_motion)
    lag_gt, corr_gt = compute_anticipatory_lag(gt_motion)
    result['antic_lag_gen_ms'] = lag_gen
    result['antic_lag_gt_ms'] = lag_gt
    result['antic_corr_gen'] = corr_gen
    result['antic_corr_gt'] = corr_gt

    # 5. Per-body-part MPJPE
    bp_mpjpe = compute_bodypart_mpjpe(gt_motion, gen_motion)
    for group, val in bp_mpjpe.items():
        result[f'mpjpe_{group}'] = val

    return result


def main():
    print("=" * 80)
    print("Musical Structure Alignment Evaluation")
    print("=" * 80)

    splits = load_splits()
    test_samples = splits['samples']['test']
    gen_root = DATASET_ROOT / "generated"

    print(f"Test samples: {len(test_samples)}")

    all_results = []
    for sample in test_samples:
        r = evaluate_sample(sample, gen_root, DATASET_ROOT)
        if r is not None:
            all_results.append(r)

    print(f"Successfully evaluated: {len(all_results)}/{len(test_samples)}")

    df = pd.DataFrame(all_results)

    # ====== Report ======

    # 1. Beat Alignment Score
    print("\n" + "=" * 60)
    print("1. Beat Alignment Score (BAS)")
    print("   Fraction of beats with nearby body velocity peak (±100ms)")
    print("=" * 60)
    bas_gen = df['bas_gen'].dropna()
    bas_gt = df['bas_gt'].dropna()
    print(f"  Generated:    {bas_gen.mean():.3f} ± {bas_gen.std():.3f}")
    print(f"  Ground Truth: {bas_gt.mean():.3f} ± {bas_gt.std():.3f}")
    print(f"  Ratio (Gen/GT): {bas_gen.mean() / (bas_gt.mean() + 1e-8):.3f}")

    # 2. Downbeat Emphasis
    print("\n" + "=" * 60)
    print("2. Downbeat Emphasis Ratio")
    print("   Body velocity at downbeats / regular beats (>1 = correct)")
    print("=" * 60)
    de_gen = df['downbeat_emphasis_gen'].dropna()
    de_gt = df['downbeat_emphasis_gt'].dropna()
    print(f"  Generated:    {de_gen.mean():.3f} ± {de_gen.std():.3f}")
    print(f"  Ground Truth: {de_gt.mean():.3f} ± {de_gt.std():.3f}")

    # 3. Phrase Boundary Reset
    print("\n" + "=" * 60)
    print("3. Phrase Boundary Body Displacement")
    print("   Spine displacement at phrase boundaries / mid-phrase")
    print("=" * 60)
    pr_gen = df['phrase_reset_gen'].dropna()
    pr_gt = df['phrase_reset_gt'].dropna()
    print(f"  Generated:    {pr_gen.mean():.3f} ± {pr_gen.std():.3f}")
    print(f"  Ground Truth: {pr_gt.mean():.3f} ± {pr_gt.std():.3f}")
    print(f"  (>1.0 = more displacement at phrase boundaries)")

    # 4. Anticipatory Lag
    print("\n" + "=" * 60)
    print("4. Anticipatory Body-Hand Lag (Cross-Correlation)")
    print("   Negative = body leads hand (expected ~-65ms)")
    print("=" * 60)
    lag_gen = df['antic_lag_gen_ms'].dropna()
    lag_gt = df['antic_lag_gt_ms'].dropna()
    corr_gen = df['antic_corr_gen'].dropna()
    corr_gt = df['antic_corr_gt'].dropna()
    print(f"  Generated:    lag = {lag_gen.mean():+.1f} ± {lag_gen.std():.1f} ms,  r = {corr_gen.mean():.3f}")
    print(f"  Ground Truth: lag = {lag_gt.mean():+.1f} ± {lag_gt.std():.1f} ms,  r = {corr_gt.mean():.3f}")

    # 5. Per-body-part MPJPE
    print("\n" + "=" * 60)
    print("5. Per-Body-Part MPJPE (mm)")
    print("=" * 60)
    bp_cols = [c for c in df.columns if c.startswith('mpjpe_')]
    print(f"  {'Body Group':<15} {'Mean':>10} {'Std':>10}")
    print("  " + "-" * 38)
    for col in sorted(bp_cols):
        group = col.replace('mpjpe_', '')
        vals = df[col].dropna()
        print(f"  {group:<15} {vals.mean():>10.1f} {vals.std():>10.1f}")

    # Save results
    output_path = ROOT / "results" / "musical_alignment_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'beat_alignment_score': {'gen': float(bas_gen.mean()), 'gt': float(bas_gt.mean())},
        'downbeat_emphasis': {'gen': float(de_gen.mean()), 'gt': float(de_gt.mean())},
        'phrase_boundary_reset': {'gen': float(pr_gen.mean()), 'gt': float(pr_gt.mean())},
        'anticipatory_lag_ms': {'gen': float(lag_gen.mean()), 'gt': float(lag_gt.mean())},
        'per_bodypart_mpjpe': {
            col.replace('mpjpe_', ''): float(df[col].dropna().mean())
            for col in bp_cols
        },
        'n_samples': len(all_results),
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
