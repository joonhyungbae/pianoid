"""
MIDI-derived feature × Motion analysis for ISMIR paper.
All music features are computed purely from MIDI (align_notetime).
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Constants
# =============================================================================
MOTION_FPS = 120
DATASET_ROOT = "MOSA_dataset/yp"

# Body group definitions (column name prefixes in motion_norm.csv)
# Wrist = average of WRA and WRB per the paper
BODY_GROUPS = {
    'head': ['RFHD', 'LFHD', 'RBHD', 'LBHD'],
    'spine': ['C7', 'T10', 'CLAV', 'STRN'],
    'shoulders': ['RSHO', 'LSHO'],
    'r_wrist': ['RWRA', 'RWRB'],  # will be averaged
    'l_wrist': ['LWRA', 'LWRB'],  # will be averaged
    'r_hand': ['RFIN'],
    'l_hand': ['LFIN'],
    'elbows': ['RELB', 'LELB'],
    'pelvis': ['RASI', 'LASI', 'RPSI', 'LPSI'],
}

# =============================================================================
# Data Loading
# =============================================================================

def find_all_recordings():
    """Find all recording directories with both motion and align_notetime data."""
    recordings = []
    for motion_file in glob.glob(os.path.join(DATASET_ROOT, "**/", "*_motion_norm.csv"), recursive=True):
        rec_dir = os.path.dirname(motion_file)
        # Find matching align_notetime
        ann_dir = os.path.join(rec_dir, "annotation", "annotations")
        align_files = glob.glob(os.path.join(ann_dir, "*_align_notetime.csv"))
        if align_files:
            basename = os.path.basename(motion_file).replace("_motion_norm.csv", "")
            recordings.append({
                'id': basename,
                'motion_file': motion_file,
                'align_file': align_files[0],
                'ann_dir': ann_dir,
            })
    return recordings


def load_motion(filepath):
    """Load motion capture data. Returns DataFrame at 120fps."""
    df = pd.read_csv(filepath)
    return df


def load_align_notetime(filepath):
    """Load the score-to-performance time alignment with MIDI info."""
    df = pd.read_csv(filepath)
    # Columns: note_id, onset(sec), offset(sec), score_position, score_position_offset,
    #          note, midi_number, note_duration, bar_id, bar_position
    return df


def get_joint_positions(motion_df, joint_prefix):
    """Extract XYZ positions for a joint prefix. Returns (N, 3) array."""
    cols = [f"{joint_prefix}_{ax}" for ax in ['X', 'Y', 'Z']]
    return motion_df[cols].values


def get_group_positions(motion_df, group_name):
    """Get averaged positions for a body group. Returns (N, 3) array."""
    joints = BODY_GROUPS[group_name]
    positions = np.stack([get_joint_positions(motion_df, j) for j in joints], axis=0)
    return positions.mean(axis=0)  # (N, 3)


def time_to_frame(t_sec, fps=MOTION_FPS):
    """Convert time in seconds to frame index."""
    return int(round(t_sec * fps))


def compute_motion_amplitude(positions, axis=None):
    """Compute std of positions. axis: 0=X(lateral), 1=Y(front-back), 2=Z(vertical), None=all."""
    if axis is not None:
        return np.std(positions[:, axis])
    return np.mean([np.std(positions[:, i]) for i in range(3)])


def compute_speed(positions, fps=MOTION_FPS):
    """Compute per-frame speed (mm/frame) as Euclidean norm of velocity."""
    vel = np.diff(positions, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    return speed


# =============================================================================
# MIDI Feature Extraction (purely from align_notetime)
# =============================================================================

def compute_frame_velocity(align_df, n_frames, fps=MOTION_FPS):
    """Compute per-frame average MIDI velocity proxy.
    Since align_notetime doesn't have velocity, we use note density as a proxy.
    But we can also check if there's velocity info in the MIDI.
    For now, we use the number of concurrent notes weighted by inverse duration
    (shorter notes = more percussive = higher dynamic tendency).

    Actually, re-reading the data: align_notetime has onset/offset in seconds.
    We don't have MIDI velocity directly. We'll compute note density instead.
    """
    pass  # Will use note density approach


def compute_frame_note_density(align_df, n_frames, fps=MOTION_FPS):
    """Compute per-frame note density (number of active notes at each frame)."""
    density = np.zeros(n_frames)
    for _, row in align_df.iterrows():
        f_on = time_to_frame(row['onset'], fps)
        f_off = time_to_frame(row['offset'], fps)
        f_on = max(0, min(f_on, n_frames - 1))
        f_off = max(0, min(f_off, n_frames - 1))
        density[f_on:f_off + 1] += 1
    return density


def compute_frame_onset_density(align_df, n_frames, fps=MOTION_FPS, window_sec=2.0):
    """Compute onset density: number of onsets within a sliding window."""
    window_frames = int(window_sec * fps)
    onset_frames = np.zeros(n_frames)
    for _, row in align_df.iterrows():
        f = time_to_frame(row['onset'], fps)
        if 0 <= f < n_frames:
            onset_frames[f] += 1
    # Sliding window sum
    density = np.convolve(onset_frames, np.ones(window_frames), mode='same') / window_sec
    return density  # onsets per second


def compute_frame_pitch_range(align_df, n_frames, fps=MOTION_FPS):
    """Compute per-frame pitch range (max MIDI number - min MIDI number of active notes)."""
    pitch_min = np.full(n_frames, np.nan)
    pitch_max = np.full(n_frames, np.nan)

    for _, row in align_df.iterrows():
        f_on = time_to_frame(row['onset'], fps)
        f_off = time_to_frame(row['offset'], fps)
        f_on = max(0, min(f_on, n_frames - 1))
        f_off = max(0, min(f_off, n_frames - 1))
        midi_num = row['midi_number']
        for f in range(f_on, f_off + 1):
            if np.isnan(pitch_min[f]) or midi_num < pitch_min[f]:
                pitch_min[f] = midi_num
            if np.isnan(pitch_max[f]) or midi_num > pitch_max[f]:
                pitch_max[f] = midi_num

    pitch_range = pitch_max - pitch_min
    pitch_range[np.isnan(pitch_range)] = 0
    return pitch_range


def compute_frame_pitch_centroid(align_df, n_frames, fps=MOTION_FPS):
    """Compute per-frame pitch centroid (mean MIDI number of active notes)."""
    pitch_sum = np.zeros(n_frames)
    pitch_count = np.zeros(n_frames)

    for _, row in align_df.iterrows():
        f_on = time_to_frame(row['onset'], fps)
        f_off = time_to_frame(row['offset'], fps)
        f_on = max(0, min(f_on, n_frames - 1))
        f_off = max(0, min(f_off, n_frames - 1))
        pitch_sum[f_on:f_off + 1] += row['midi_number']
        pitch_count[f_on:f_off + 1] += 1

    centroid = np.where(pitch_count > 0, pitch_sum / pitch_count, np.nan)
    return centroid


def compute_note_ioi(align_df):
    """Compute inter-onset intervals from sorted onsets."""
    onsets = align_df['onset'].sort_values().values
    ioi = np.diff(onsets)
    return onsets[1:], ioi  # return onset times and their IOIs


def compute_duration_ioi_ratio(align_df):
    """Compute duration/IOI ratio per note (articulation proxy).
    Returns DataFrame with onset, ratio, midi_number."""
    df = align_df.sort_values('onset').copy()
    df['performed_duration'] = df['offset'] - df['onset']
    df['ioi'] = df['onset'].diff()
    # Only compute ratio where IOI > 0
    mask = df['ioi'] > 0.01  # minimum 10ms
    df.loc[mask, 'dur_ioi_ratio'] = df.loc[mask, 'performed_duration'] / df.loc[mask, 'ioi']
    return df[['onset', 'offset', 'midi_number', 'performed_duration', 'ioi', 'dur_ioi_ratio']].dropna()


def compute_local_tempo_variation(align_df, window_notes=8):
    """Compute local tempo variation from IOI.
    Returns onset times and local IOI coefficient of variation."""
    onsets = align_df['onset'].sort_values().values
    ioi = np.diff(onsets)

    results_time = []
    results_cv = []
    results_ioi_deriv = []

    half_w = window_notes // 2
    for i in range(half_w, len(ioi) - half_w):
        local_ioi = ioi[i - half_w:i + half_w]
        cv = np.std(local_ioi) / (np.mean(local_ioi) + 1e-8)
        # IOI derivative (acceleration/deceleration)
        ioi_deriv = (ioi[min(i+1, len(ioi)-1)] - ioi[max(i-1, 0)]) / 2
        results_time.append(onsets[i + 1])
        results_cv.append(cv)
        results_ioi_deriv.append(ioi_deriv)

    return np.array(results_time), np.array(results_cv), np.array(results_ioi_deriv)


# =============================================================================
# Analysis Functions
# =============================================================================

def analysis1_velocity_motion(recordings):
    """Analysis 1: MIDI Velocity (onset density as proxy) × Body motion amplitude.
    Since we don't have MIDI velocity directly, we use onset density as a
    dynamic intensity proxy (higher onset density = more active playing)."""

    print("=" * 70)
    print("ANALYSIS 1: Note Onset Density (dynamic proxy) × Body Motion Amplitude")
    print("=" * 70)

    # Collect per-window statistics across all recordings
    window_sec = 4.0
    results = defaultdict(list)

    for rec in recordings:
        motion_df = load_motion(rec['motion_file'])
        align_df = load_align_notetime(rec['align_file'])
        n_frames = len(motion_df)

        onset_density = compute_frame_onset_density(align_df, n_frames, MOTION_FPS, window_sec=2.0)

        # Divide into windows
        window_frames = int(window_sec * MOTION_FPS)
        for start in range(0, n_frames - window_frames, window_frames // 2):
            end = start + window_frames
            win_density = np.mean(onset_density[start:end])

            for group_name in ['head', 'spine', 'shoulders', 'r_wrist', 'l_wrist', 'pelvis']:
                pos = get_group_positions(motion_df.iloc[start:end], group_name)
                amp = compute_motion_amplitude(pos)
                results[group_name].append((win_density, amp))

    # Compute correlations
    print(f"\nTotal windows analyzed: {len(results['spine'])}")
    print(f"\n{'Body Group':<15} {'Pearson r':>10} {'p-value':>12} {'Spearman ρ':>12}")
    print("-" * 55)

    for group_name in ['r_wrist', 'l_wrist', 'head', 'spine', 'shoulders', 'pelvis']:
        data = np.array(results[group_name])
        density_vals = data[:, 0]
        amp_vals = data[:, 1]
        r, p = stats.pearsonr(density_vals, amp_vals)
        rho, p_s = stats.spearmanr(density_vals, amp_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{group_name:<15} {r:>10.3f} {p:>10.2e} {sig:>2} {rho:>10.3f}")

    # Quantile analysis
    print("\n--- Motion amplitude by onset density quartile (mm) ---")
    spine_data = np.array(results['spine'])
    wrist_data = np.array(results['r_wrist'])
    head_data = np.array(results['head'])

    quartiles = np.percentile(spine_data[:, 0], [25, 50, 75])
    labels = ['Q1 (sparse)', 'Q2', 'Q3', 'Q4 (dense)']

    print(f"\n{'Quartile':<15} {'Density range':>18} {'Spine':>10} {'R_Wrist':>10} {'Head':>10}")
    print("-" * 70)

    bounds = [0] + list(quartiles) + [np.inf]
    for i in range(4):
        mask = (spine_data[:, 0] >= bounds[i]) & (spine_data[:, 0] < bounds[i + 1])
        spine_amp = spine_data[mask, 1].mean()
        wrist_amp = wrist_data[mask, 1].mean()
        head_amp = head_data[mask, 1].mean()
        d_lo, d_hi = bounds[i], bounds[i+1]
        d_range = f"{d_lo:.1f}-{min(d_hi, 99):.1f}"
        print(f"{labels[i]:<15} {d_range:>18} {spine_amp:>10.2f} {wrist_amp:>10.2f} {head_amp:>10.2f}")


def analysis2_note_density_motion(recordings):
    """Analysis 2: Note Density (polyphony) × Motion characteristics."""

    print("\n" + "=" * 70)
    print("ANALYSIS 2: Note Density (polyphony) × Motion Characteristics")
    print("=" * 70)

    window_sec = 2.0
    results = defaultdict(list)

    for rec in recordings:
        motion_df = load_motion(rec['motion_file'])
        align_df = load_align_notetime(rec['align_file'])
        n_frames = len(motion_df)

        note_density = compute_frame_note_density(align_df, n_frames, MOTION_FPS)

        window_frames = int(window_sec * MOTION_FPS)
        for start in range(0, n_frames - window_frames, window_frames // 2):
            end = start + window_frames
            win_density = np.mean(note_density[start:end])
            if win_density < 0.1:
                continue  # skip silent sections

            # Wrist Z-axis amplitude (vertical keystroke motion)
            rw_pos = get_group_positions(motion_df.iloc[start:end], 'r_wrist')
            lw_pos = get_group_positions(motion_df.iloc[start:end], 'l_wrist')
            wrist_z_amp = (np.std(rw_pos[:, 2]) + np.std(lw_pos[:, 2])) / 2

            # Wrist speed
            rw_speed = compute_speed(rw_pos).mean()
            lw_speed = compute_speed(lw_pos).mean()
            wrist_speed = (rw_speed + lw_speed) / 2

            # Spine lateral (X) amplitude
            spine_pos = get_group_positions(motion_df.iloc[start:end], 'spine')
            spine_x_amp = np.std(spine_pos[:, 0])

            # Head amplitude
            head_pos = get_group_positions(motion_df.iloc[start:end], 'head')
            head_amp = compute_motion_amplitude(head_pos)

            results['density'].append(win_density)
            results['wrist_z_amp'].append(wrist_z_amp)
            results['wrist_speed'].append(wrist_speed)
            results['spine_x_amp'].append(spine_x_amp)
            results['head_amp'].append(head_amp)

    df = pd.DataFrame(results)

    print(f"\nTotal windows: {len(df)}")
    print(f"\n{'Motion Feature':<20} {'Pearson r':>10} {'p-value':>12} {'Spearman ρ':>12}")
    print("-" * 60)

    for feat in ['wrist_z_amp', 'wrist_speed', 'spine_x_amp', 'head_amp']:
        r, p = stats.pearsonr(df['density'], df[feat])
        rho, _ = stats.spearmanr(df['density'], df[feat])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{feat:<20} {r:>10.3f} {p:>10.2e} {sig:>2} {rho:>10.3f}")

    # Quartile analysis
    print("\n--- Motion features by note density quartile ---")
    quartiles = np.percentile(df['density'], [25, 50, 75])
    labels = ['Q1 (sparse)', 'Q2', 'Q3', 'Q4 (dense)']
    bounds = [0] + list(quartiles) + [np.inf]

    print(f"\n{'Quartile':<15} {'Density':>10} {'Wrist Z':>10} {'Wrist Spd':>10} {'Spine X':>10} {'Head':>10}")
    print("-" * 70)
    for i in range(4):
        mask = (df['density'] >= bounds[i]) & (df['density'] < bounds[i + 1])
        sub = df[mask]
        print(f"{labels[i]:<15} {sub['density'].mean():>10.1f} {sub['wrist_z_amp'].mean():>10.2f} "
              f"{sub['wrist_speed'].mean():>10.2f} {sub['spine_x_amp'].mean():>10.2f} {sub['head_amp'].mean():>10.2f}")


def analysis3_articulation_hand_motion(recordings):
    """Analysis 3: Duration/IOI Ratio (articulation proxy) × Hand motion profile."""

    print("\n" + "=" * 70)
    print("ANALYSIS 3: Duration/IOI Ratio (Articulation) × Hand Motion Profile")
    print("=" * 70)

    all_legato = defaultdict(list)
    all_staccato = defaultdict(list)
    all_normal = defaultdict(list)

    context_frames = int(0.15 * MOTION_FPS)  # 150ms around onset

    for rec in recordings:
        motion_df = load_motion(rec['motion_file'])
        align_df = load_align_notetime(rec['align_file'])
        n_frames = len(motion_df)

        dur_ioi = compute_duration_ioi_ratio(align_df)

        for _, row in dur_ioi.iterrows():
            onset_frame = time_to_frame(row['onset'], MOTION_FPS)
            if onset_frame < context_frames or onset_frame >= n_frames - context_frames:
                continue

            ratio = row['dur_ioi_ratio']
            if ratio > 10:  # outlier
                continue

            # Get wrist motion around this onset
            start = onset_frame - context_frames
            end = onset_frame + context_frames

            # Use right wrist for higher pitches, left for lower
            if row['midi_number'] >= 60:
                wrist_pos = get_group_positions(motion_df.iloc[start:end], 'r_wrist')
            else:
                wrist_pos = get_group_positions(motion_df.iloc[start:end], 'l_wrist')

            # Metrics
            z_range = np.ptp(wrist_pos[:, 2])  # vertical range
            z_descent = wrist_pos[0, 2] - np.min(wrist_pos[:, 2])  # descent depth
            speed = compute_speed(wrist_pos)
            max_speed = np.max(speed)
            mean_speed = np.mean(speed)

            entry = {
                'z_range': z_range,
                'z_descent': z_descent,
                'max_speed': max_speed,
                'mean_speed': mean_speed,
                'ratio': ratio,
            }

            if ratio < 0.5:
                for k, v in entry.items():
                    all_staccato[k].append(v)
            elif ratio > 1.2:
                for k, v in entry.items():
                    all_legato[k].append(v)
            else:
                for k, v in entry.items():
                    all_normal[k].append(v)

    print(f"\nNote counts - Staccato (ratio<0.5): {len(all_staccato['z_range'])}, "
          f"Normal (0.5-1.2): {len(all_normal['z_range'])}, "
          f"Legato (ratio>1.2): {len(all_legato['z_range'])}")

    print(f"\n{'Metric':<20} {'Staccato':>12} {'Normal':>12} {'Legato':>12} {'S vs L (t-test)':>18}")
    print("-" * 80)

    for metric in ['z_range', 'z_descent', 'max_speed', 'mean_speed']:
        s_vals = np.array(all_staccato[metric])
        n_vals = np.array(all_normal[metric])
        l_vals = np.array(all_legato[metric])

        t_stat, p_val = stats.ttest_ind(s_vals, l_vals)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        print(f"{metric:<20} {np.mean(s_vals):>12.3f} {np.mean(n_vals):>12.3f} "
              f"{np.mean(l_vals):>12.3f} {f't={t_stat:.2f} {sig}':>18}")

    # Correlation between ratio and motion features
    print("\n--- Continuous correlation: dur/IOI ratio vs motion ---")
    all_ratios = all_staccato['ratio'] + all_normal['ratio'] + all_legato['ratio']
    for metric in ['z_range', 'max_speed']:
        all_metric = all_staccato[metric] + all_normal[metric] + all_legato[metric]
        r, p = stats.pearsonr(all_ratios, all_metric)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  ratio vs {metric:<15}: r={r:.3f}, p={p:.2e} {sig}")


def analysis4_pitch_range_sway(recordings):
    """Analysis 4: Pitch Range × Upper body sway."""

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Pitch Range × Upper Body Lateral Sway")
    print("=" * 70)

    window_sec = 2.0
    results = defaultdict(list)

    for rec in recordings:
        motion_df = load_motion(rec['motion_file'])
        align_df = load_align_notetime(rec['align_file'])
        n_frames = len(motion_df)

        pitch_range = compute_frame_pitch_range(align_df, n_frames, MOTION_FPS)
        pitch_centroid = compute_frame_pitch_centroid(align_df, n_frames, MOTION_FPS)

        window_frames = int(window_sec * MOTION_FPS)
        for start in range(0, n_frames - window_frames, window_frames // 2):
            end = start + window_frames
            win_pr = np.nanmean(pitch_range[start:end])
            win_pc = np.nanmean(pitch_centroid[start:end])

            if np.isnan(win_pr) or win_pr < 1:
                continue

            # Spine C7 lateral (X axis)
            c7_pos = get_joint_positions(motion_df.iloc[start:end], 'C7')
            c7_x_std = np.std(c7_pos[:, 0])
            c7_x_range = np.ptp(c7_pos[:, 0])

            # Spine T10 lateral
            t10_pos = get_joint_positions(motion_df.iloc[start:end], 'T10')
            t10_x_std = np.std(t10_pos[:, 0])

            # Head lateral
            head_pos = get_group_positions(motion_df.iloc[start:end], 'head')
            head_x_std = np.std(head_pos[:, 0])

            results['pitch_range'].append(win_pr)
            results['pitch_centroid'].append(win_pc)
            results['c7_x_std'].append(c7_x_std)
            results['c7_x_range'].append(c7_x_range)
            results['t10_x_std'].append(t10_x_std)
            results['head_x_std'].append(head_x_std)

    df = pd.DataFrame(results)

    print(f"\nTotal windows: {len(df)}")
    print(f"\n{'Motion Feature':<20} {'vs Pitch Range':>18} {'vs Pitch Centroid':>20}")
    print(f"{'':20} {'r (p)':>18} {'r (p)':>20}")
    print("-" * 62)

    for feat in ['c7_x_std', 'c7_x_range', 't10_x_std', 'head_x_std']:
        r1, p1 = stats.pearsonr(df['pitch_range'], df[feat])
        r2, p2 = stats.pearsonr(df['pitch_centroid'], df[feat])
        sig1 = "***" if p1 < 0.001 else "**" if p1 < 0.01 else "*" if p1 < 0.05 else ""
        sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else ""
        print(f"{feat:<20} {r1:>7.3f} ({p1:.1e}){sig1:>3} {r2:>9.3f} ({p2:.1e}){sig2:>3}")

    # Quartile breakdown
    print("\n--- C7 lateral sway by pitch range quartile ---")
    quartiles = np.percentile(df['pitch_range'], [25, 50, 75])
    labels = ['Q1 (narrow)', 'Q2', 'Q3', 'Q4 (wide)']
    bounds = [0] + list(quartiles) + [np.inf]

    print(f"\n{'Quartile':<15} {'Pitch Range':>12} {'C7 X std':>10} {'C7 X range':>12} {'Head X std':>12}")
    print("-" * 65)
    for i in range(4):
        mask = (df['pitch_range'] >= bounds[i]) & (df['pitch_range'] < bounds[i + 1])
        sub = df[mask]
        print(f"{labels[i]:<15} {sub['pitch_range'].mean():>12.1f} {sub['c7_x_std'].mean():>10.2f} "
              f"{sub['c7_x_range'].mean():>12.2f} {sub['head_x_std'].mean():>12.2f}")


def analysis5_ioi_variation_anticipation(recordings):
    """Analysis 5: IOI variability (local tempo) × Body anticipatory motion."""

    print("\n" + "=" * 70)
    print("ANALYSIS 5: IOI Variability (Tempo Flexibility) × Body Motion")
    print("=" * 70)

    results = defaultdict(list)

    for rec in recordings:
        motion_df = load_motion(rec['motion_file'])
        align_df = load_align_notetime(rec['align_file'])
        n_frames = len(motion_df)

        times, cv_vals, ioi_deriv = compute_local_tempo_variation(align_df, window_notes=8)

        context_frames = int(0.5 * MOTION_FPS)  # 500ms window around each point

        for t, cv, deriv in zip(times, cv_vals, ioi_deriv):
            center_frame = time_to_frame(t, MOTION_FPS)
            if center_frame < context_frames or center_frame >= n_frames - context_frames:
                continue

            start = center_frame - context_frames
            end = center_frame + context_frames

            # Spine Y-axis (front-back tilt)
            c7_pos = get_joint_positions(motion_df.iloc[start:end], 'C7')
            c7_y_std = np.std(c7_pos[:, 1])
            c7_y_vel = np.mean(np.diff(c7_pos[:, 1]))  # average forward/backward velocity

            # Spine X (lateral)
            c7_x_std = np.std(c7_pos[:, 0])

            # Head movement
            head_pos = get_group_positions(motion_df.iloc[start:end], 'head')
            head_amp = compute_motion_amplitude(head_pos)

            # Overall body speed
            spine_pos = get_group_positions(motion_df.iloc[start:end], 'spine')
            spine_speed = compute_speed(spine_pos).mean()

            results['ioi_cv'].append(cv)
            results['ioi_deriv'].append(deriv)
            results['c7_y_std'].append(c7_y_std)
            results['c7_y_vel'].append(c7_y_vel)
            results['c7_x_std'].append(c7_x_std)
            results['head_amp'].append(head_amp)
            results['spine_speed'].append(spine_speed)

    df = pd.DataFrame(results)

    print(f"\nTotal data points: {len(df)}")

    # IOI CV (tempo variability) correlations
    print(f"\n--- IOI Coefficient of Variation (rubato) vs Motion ---")
    print(f"{'Motion Feature':<20} {'Pearson r':>10} {'p-value':>12} {'Spearman ρ':>12}")
    print("-" * 60)

    for feat in ['c7_y_std', 'c7_x_std', 'head_amp', 'spine_speed']:
        r, p = stats.pearsonr(df['ioi_cv'], df[feat])
        rho, _ = stats.spearmanr(df['ioi_cv'], df[feat])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{feat:<20} {r:>10.3f} {p:>10.2e} {sig:>2} {rho:>10.3f}")

    # IOI derivative (accel/decel) correlations
    print(f"\n--- IOI Derivative (accel < 0 / decel > 0) vs Body forward/back ---")
    print(f"{'Motion Feature':<20} {'Pearson r':>10} {'p-value':>12}")
    print("-" * 48)

    for feat in ['c7_y_vel', 'c7_y_std', 'spine_speed']:
        r, p = stats.pearsonr(df['ioi_deriv'], df[feat])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{feat:<20} {r:>10.3f} {p:>10.2e} {sig:>2}")

    # Quartile breakdown for IOI CV
    print("\n--- Body motion by IOI variability quartile ---")
    quartiles = np.percentile(df['ioi_cv'], [25, 50, 75])
    labels = ['Q1 (steady)', 'Q2', 'Q3', 'Q4 (rubato)']
    bounds = [0] + list(quartiles) + [np.inf]

    print(f"\n{'Quartile':<15} {'IOI CV':>10} {'C7 Y std':>10} {'C7 X std':>10} {'Head Amp':>10} {'Spine Spd':>10}")
    print("-" * 70)
    for i in range(4):
        mask = (df['ioi_cv'] >= bounds[i]) & (df['ioi_cv'] < bounds[i + 1])
        sub = df[mask]
        print(f"{labels[i]:<15} {sub['ioi_cv'].mean():>10.3f} {sub['c7_y_std'].mean():>10.2f} "
              f"{sub['c7_x_std'].mean():>10.2f} {sub['head_amp'].mean():>10.2f} {sub['spine_speed'].mean():>10.3f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Loading recordings...")
    recordings = find_all_recordings()
    print(f"Found {len(recordings)} recordings with align_notetime data\n")

    # Sample a subset for faster iteration (or use all)
    # recordings = recordings[:20]  # uncomment for quick test

    analysis1_velocity_motion(recordings)
    analysis2_note_density_motion(recordings)
    analysis3_articulation_hand_motion(recordings)
    analysis4_pitch_range_sway(recordings)
    analysis5_ioi_variation_anticipation(recordings)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
