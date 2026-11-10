"""
Compare MIDI-derived proxy features vs GT annotations.
Goal: Quantify how well MIDI-only features approximate ground-truth musical labels,
since the model can only use MIDI at inference time.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

MOTION_FPS = 120
DATASET_ROOT = "MOSA_dataset/yp"

BODY_GROUPS = {
    'head': ['RFHD', 'LFHD', 'RBHD', 'LBHD'],
    'spine': ['C7', 'T10', 'CLAV', 'STRN'],
    'shoulders': ['RSHO', 'LSHO'],
    'r_wrist': ['RWRA', 'RWRB'],
    'l_wrist': ['LWRA', 'LWRB'],
    'pelvis': ['RASI', 'LASI', 'RPSI', 'LPSI'],
}

# Dynamic levels ordered by intensity
DYNAMIC_ORDER = {'ppp': 0, 'pp': 1, 'p': 2, 'mp': 3, 'mf': 4, 'f': 5, 'ff': 6}
DYNAMIC_LEVELS = set(DYNAMIC_ORDER.keys())
ARTICULATION_TYPES = {'legato', 'staccato'}
TEMPO_TYPES = {'rit', 'accel'}

# ============================================================
# Data Loading
# ============================================================

def find_all_recordings():
    recordings = []
    for motion_file in glob.glob(os.path.join(DATASET_ROOT, "**", "*_motion_norm.csv"), recursive=True):
        rec_dir = os.path.dirname(motion_file)
        ann_dir = os.path.join(rec_dir, "annotation", "annotations")
        align_files = glob.glob(os.path.join(ann_dir, "*_align_notetime.csv"))
        expr_files = glob.glob(os.path.join(ann_dir, "*_expression.csv"))
        beat_files = glob.glob(os.path.join(ann_dir, "*_beat.csv"))
        downbeat_files = glob.glob(os.path.join(ann_dir, "*_downbeat.csv"))
        if align_files and expr_files:
            basename = os.path.basename(motion_file).replace("_motion_norm.csv", "")
            recordings.append({
                'id': basename,
                'motion_file': motion_file,
                'align_file': align_files[0],
                'expr_file': expr_files[0],
                'beat_file': beat_files[0] if beat_files else None,
                'downbeat_file': downbeat_files[0] if downbeat_files else None,
                'ann_dir': ann_dir,
            })
    return recordings


def load_motion(filepath):
    return pd.read_csv(filepath)


def load_align_notetime(filepath):
    return pd.read_csv(filepath)


def load_expression(filepath):
    df = pd.read_csv(filepath)
    if len(df) == 0:
        return pd.DataFrame(columns=['onset', 'offset', 'expression'])
    return df


def load_beats(filepath):
    if filepath is None:
        return np.array([])
    df = pd.read_csv(filepath)
    return df.iloc[:, 0].values


def load_downbeats(filepath):
    if filepath is None:
        return np.array([])
    df = pd.read_csv(filepath)
    return df.iloc[:, 0].values


def get_joint_positions(motion_df, joint_prefix):
    cols = [f"{joint_prefix}_{ax}" for ax in ['X', 'Y', 'Z']]
    return motion_df[cols].values


def get_group_positions(motion_df, group_name):
    joints = BODY_GROUPS[group_name]
    positions = np.stack([get_joint_positions(motion_df, j) for j in joints], axis=0)
    return positions.mean(axis=0)


def time_to_frame(t_sec, fps=MOTION_FPS):
    return int(round(t_sec * fps))


def compute_speed(positions):
    vel = np.diff(positions, axis=0)
    return np.linalg.norm(vel, axis=1)


# ============================================================
# Score position -> Performance time mapping
# ============================================================

def build_score_to_perf_map(align_df):
    """Build interpolation from score position to performance time (seconds)."""
    df = align_df.sort_values('score_position')
    score_pos = df['score_position'].values
    perf_time = df['onset'].values

    # Remove duplicates (keep first)
    _, unique_idx = np.unique(score_pos, return_index=True)
    score_pos = score_pos[unique_idx]
    perf_time = perf_time[unique_idx]

    def map_fn(sp):
        return np.interp(sp, score_pos, perf_time)

    return map_fn


# ============================================================
# MIDI-derived feature computation
# ============================================================

def compute_onset_density_at_window(align_df, perf_start, perf_end):
    """Count onsets per second in a performance time window."""
    mask = (align_df['onset'] >= perf_start) & (align_df['onset'] < perf_end)
    n_onsets = mask.sum()
    duration = perf_end - perf_start
    return n_onsets / max(duration, 0.01)


def compute_mean_velocity_proxy(align_df, perf_start, perf_end):
    """Compute average note density (polyphony) as velocity/intensity proxy."""
    n_frames = len(align_df)
    active_count = 0
    total_frames = 0

    # Sample at 10ms resolution
    for t in np.arange(perf_start, perf_end, 0.01):
        mask = (align_df['onset'] <= t) & (align_df['offset'] > t)
        active_count += mask.sum()
        total_frames += 1

    return active_count / max(total_frames, 1)


def compute_dur_ioi_ratio_in_window(align_df, perf_start, perf_end):
    """Compute mean duration/IOI ratio for notes starting in window."""
    mask = (align_df['onset'] >= perf_start) & (align_df['onset'] < perf_end)
    notes = align_df[mask].sort_values('onset')
    if len(notes) < 2:
        return np.nan

    durations = (notes['offset'] - notes['onset']).values
    iois = np.diff(notes['onset'].values)
    # Match: each note's duration vs IOI to next note
    valid = iois > 0.01
    if valid.sum() == 0:
        return np.nan
    ratios = durations[:-1][valid] / iois[valid]
    return np.median(ratios)


def compute_ioi_cv_in_window(align_df, perf_start, perf_end):
    """Compute IOI coefficient of variation in window."""
    mask = (align_df['onset'] >= perf_start) & (align_df['onset'] < perf_end)
    onsets = align_df[mask]['onset'].sort_values().values
    if len(onsets) < 3:
        return np.nan
    ioi = np.diff(onsets)
    ioi = ioi[ioi > 0.01]
    if len(ioi) < 2:
        return np.nan
    return np.std(ioi) / (np.mean(ioi) + 1e-8)


# ============================================================
# Comparison 1: MIDI onset density vs GT dynamics
# ============================================================

def comparison1_dynamics(recordings):
    """Compare MIDI onset density / polyphony vs GT dynamic labels (p, mp, mf, f, ff)."""
    print("=" * 80)
    print("COMPARISON 1: MIDI Onset Density vs GT Dynamic Markings")
    print("=" * 80)

    results = defaultdict(list)  # dynamic_label -> list of (midi_density, motion_amp)

    n_used = 0
    for rec in recordings:
        motion_df = load_motion(rec['motion_file'])
        align_df = load_align_notetime(rec['align_file'])
        expr_df = load_expression(rec['expr_file'])
        n_frames = len(motion_df)

        if len(expr_df) == 0:
            continue

        score_to_perf = build_score_to_perf_map(align_df)

        # Extract dynamic markings
        dynamic_rows = expr_df[expr_df['expression'].isin(DYNAMIC_LEVELS)]
        if len(dynamic_rows) == 0:
            continue

        n_used += 1
        for _, row in dynamic_rows.iterrows():
            label = row['expression']
            perf_start = score_to_perf(row['onset'])
            perf_end = score_to_perf(row['offset'])

            if perf_end - perf_start < 0.5:
                continue

            f_start = max(0, time_to_frame(perf_start))
            f_end = min(n_frames, time_to_frame(perf_end))
            if f_end - f_start < 60:
                continue

            # MIDI features
            onset_density = compute_onset_density_at_window(align_df, perf_start, perf_end)

            # Motion features
            spine_pos = get_group_positions(motion_df.iloc[f_start:f_end], 'spine')
            rw_pos = get_group_positions(motion_df.iloc[f_start:f_end], 'r_wrist')
            lw_pos = get_group_positions(motion_df.iloc[f_start:f_end], 'l_wrist')
            head_pos = get_group_positions(motion_df.iloc[f_start:f_end], 'head')

            spine_amp = np.mean([np.std(spine_pos[:, i]) for i in range(3)])
            wrist_speed = (compute_speed(rw_pos).mean() + compute_speed(lw_pos).mean()) / 2
            head_amp = np.mean([np.std(head_pos[:, i]) for i in range(3)])
            wrist_z_amp = (np.std(rw_pos[:, 2]) + np.std(lw_pos[:, 2])) / 2

            results[label].append({
                'onset_density': onset_density,
                'spine_amp': spine_amp,
                'wrist_speed': wrist_speed,
                'head_amp': head_amp,
                'wrist_z_amp': wrist_z_amp,
                'dynamic_level': DYNAMIC_ORDER[label],
            })

    print(f"Recordings with dynamics annotations: {n_used}")

    # Print per-dynamic stats
    ordered_labels = sorted([l for l in results.keys() if len(results[l]) > 0],
                            key=lambda x: DYNAMIC_ORDER[x])

    print(f"\n{'Dynamic':<8} {'N':>5} {'Onset Dens':>12} {'Spine Amp':>12} {'Wrist Spd':>12} {'Head Amp':>12} {'Wrist Z':>12}")
    print("-" * 80)
    for label in ordered_labels:
        data = results[label]
        df = pd.DataFrame(data)
        print(f"{label:<8} {len(df):>5} {df['onset_density'].mean():>12.2f} "
              f"{df['spine_amp'].mean():>12.2f} {df['wrist_speed'].mean():>12.3f} "
              f"{df['head_amp'].mean():>12.2f} {df['wrist_z_amp'].mean():>12.2f}")

    # Correlation: dynamic level (ordinal) vs MIDI density and vs motion
    all_data = []
    for label, entries in results.items():
        all_data.extend(entries)
    all_df = pd.DataFrame(all_data)

    print(f"\n--- Correlation with GT dynamic level (ordinal: ppp=0 ... ff=6) ---")
    print(f"{'Feature':<20} {'Pearson r':>10} {'p-value':>12} {'Spearman ρ':>12}")
    print("-" * 60)
    for feat in ['onset_density', 'spine_amp', 'wrist_speed', 'head_amp', 'wrist_z_amp']:
        r, p = stats.pearsonr(all_df['dynamic_level'], all_df[feat])
        rho, p_s = stats.spearmanr(all_df['dynamic_level'], all_df[feat])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{feat:<20} {r:>10.3f} {p:>10.2e} {sig:>2} {rho:>10.3f}")

    # Key question: Does MIDI density predict dynamics?
    print(f"\n--- Can MIDI onset density predict GT dynamic level? ---")
    r, p = stats.pearsonr(all_df['onset_density'], all_df['dynamic_level'])
    rho, _ = stats.spearmanr(all_df['onset_density'], all_df['dynamic_level'])
    print(f"  onset_density vs dynamic_level: r={r:.3f}, ρ={rho:.3f}, p={p:.2e}")

    # Compare: MIDI density -> motion vs GT dynamics -> motion
    print(f"\n--- Predictive power: MIDI proxy vs GT label -> Motion ---")
    print(f"{'Motion Target':<20} {'r(MIDI dens→Mot)':>20} {'r(GT dyn→Mot)':>18} {'Gap':>8}")
    print("-" * 70)
    for feat in ['spine_amp', 'wrist_speed', 'head_amp', 'wrist_z_amp']:
        r_midi, _ = stats.pearsonr(all_df['onset_density'], all_df[feat])
        r_gt, _ = stats.pearsonr(all_df['dynamic_level'], all_df[feat])
        gap = abs(r_gt) - abs(r_midi)
        print(f"{feat:<20} {r_midi:>20.3f} {r_gt:>18.3f} {gap:>+8.3f}")


# ============================================================
# Comparison 2: Duration/IOI ratio vs GT articulation
# ============================================================

def comparison2_articulation(recordings):
    """Compare MIDI dur/IOI ratio vs GT articulation labels (legato/staccato)."""
    print("\n" + "=" * 80)
    print("COMPARISON 2: MIDI Duration/IOI Ratio vs GT Articulation Labels")
    print("=" * 80)

    gt_legato_ratios = []
    gt_staccato_ratios = []
    gt_none_ratios = []

    gt_legato_motion = defaultdict(list)
    gt_staccato_motion = defaultdict(list)
    gt_none_motion = defaultdict(list)

    midi_legato_motion = defaultdict(list)
    midi_staccato_motion = defaultdict(list)

    context_frames = int(0.15 * MOTION_FPS)
    n_used = 0

    for rec in recordings:
        motion_df = load_motion(rec['motion_file'])
        align_df = load_align_notetime(rec['align_file'])
        expr_df = load_expression(rec['expr_file'])
        n_frames = len(motion_df)

        if len(expr_df) == 0:
            continue

        score_to_perf = build_score_to_perf_map(align_df)

        # Get GT articulation regions (in score position)
        artic_rows = expr_df[expr_df['expression'].isin(ARTICULATION_TYPES)]
        if len(artic_rows) == 0:
            continue
        n_used += 1

        # For each note, determine: (a) GT label, (b) MIDI dur/IOI ratio
        notes_sorted = align_df.sort_values('onset').copy()
        notes_sorted['perf_duration'] = notes_sorted['offset'] - notes_sorted['onset']
        notes_sorted['ioi'] = notes_sorted['onset'].diff()

        for idx in range(1, len(notes_sorted)):
            row = notes_sorted.iloc[idx]
            ioi = row['ioi']
            if ioi < 0.01:
                continue

            dur_ioi = row['perf_duration'] / ioi
            if dur_ioi > 10:
                continue

            score_pos = row['score_position']
            onset_frame = time_to_frame(row['onset'])

            if onset_frame < context_frames or onset_frame >= n_frames - context_frames:
                continue

            # Determine GT label at this score position
            gt_label = 'none'
            for _, ar in artic_rows.iterrows():
                if ar['onset'] <= score_pos < ar['offset']:
                    gt_label = ar['expression']
                    break

            # Determine MIDI-derived label
            midi_label = 'normal'
            if dur_ioi < 0.5:
                midi_label = 'staccato'
            elif dur_ioi > 1.2:
                midi_label = 'legato'

            # Get hand motion
            start = onset_frame - context_frames
            end = onset_frame + context_frames
            if row['midi_number'] >= 60:
                wrist_pos = get_group_positions(motion_df.iloc[start:end], 'r_wrist')
            else:
                wrist_pos = get_group_positions(motion_df.iloc[start:end], 'l_wrist')

            z_range = np.ptp(wrist_pos[:, 2])
            max_speed = np.max(compute_speed(wrist_pos))
            mean_speed = np.mean(compute_speed(wrist_pos))
            motion_entry = {'z_range': z_range, 'max_speed': max_speed, 'mean_speed': mean_speed}

            # Store GT classification
            if gt_label == 'legato':
                gt_legato_ratios.append(dur_ioi)
                for k, v in motion_entry.items():
                    gt_legato_motion[k].append(v)
            elif gt_label == 'staccato':
                gt_staccato_ratios.append(dur_ioi)
                for k, v in motion_entry.items():
                    gt_staccato_motion[k].append(v)
            else:
                gt_none_ratios.append(dur_ioi)
                for k, v in motion_entry.items():
                    gt_none_motion[k].append(v)

            # Store MIDI classification
            if midi_label == 'staccato':
                for k, v in motion_entry.items():
                    midi_staccato_motion[k].append(v)
            elif midi_label == 'legato':
                for k, v in motion_entry.items():
                    midi_legato_motion[k].append(v)

    print(f"Recordings with articulation annotations: {n_used}")

    # Q1: Does MIDI dur/IOI ratio agree with GT labels?
    print(f"\n--- MIDI dur/IOI ratio distribution by GT label ---")
    print(f"{'GT Label':<12} {'N':>8} {'Mean ratio':>12} {'Median':>10} {'Std':>10}")
    print("-" * 55)
    for label, ratios in [('staccato', gt_staccato_ratios), ('none', gt_none_ratios), ('legato', gt_legato_ratios)]:
        if len(ratios) > 0:
            arr = np.array(ratios)
            print(f"{label:<12} {len(arr):>8} {np.mean(arr):>12.3f} {np.median(arr):>10.3f} {np.std(arr):>10.3f}")

    # Classification agreement
    print(f"\n--- Classification agreement (MIDI threshold: staccato<0.5, legato>1.2) ---")
    gt_s = np.array(gt_staccato_ratios)
    gt_l = np.array(gt_legato_ratios)

    if len(gt_s) > 0:
        midi_agrees_s = (gt_s < 0.5).mean() * 100
        print(f"  GT staccato notes with MIDI ratio < 0.5: {midi_agrees_s:.1f}%")
        print(f"  GT staccato notes with MIDI ratio < 0.8: {(gt_s < 0.8).mean()*100:.1f}%")
    if len(gt_l) > 0:
        midi_agrees_l = (gt_l > 1.2).mean() * 100
        print(f"  GT legato notes with MIDI ratio > 1.2:   {midi_agrees_l:.1f}%")
        print(f"  GT legato notes with MIDI ratio > 0.8:   {(gt_l > 0.8).mean()*100:.1f}%")

    # Q2: Motion differences - GT labels vs MIDI labels
    print(f"\n--- Motion by GT articulation label ---")
    print(f"{'Metric':<15} {'GT Staccato':>14} {'GT None':>14} {'GT Legato':>14} {'S vs L t-test':>18}")
    print("-" * 80)
    for metric in ['z_range', 'max_speed', 'mean_speed']:
        s = np.array(gt_staccato_motion[metric]) if gt_staccato_motion[metric] else np.array([0])
        n = np.array(gt_none_motion[metric]) if gt_none_motion[metric] else np.array([0])
        l = np.array(gt_legato_motion[metric]) if gt_legato_motion[metric] else np.array([0])
        if len(s) > 1 and len(l) > 1:
            t_stat, p_val = stats.ttest_ind(s, l)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        else:
            t_stat, sig = 0, "n/a"
        print(f"{metric:<15} {np.mean(s):>14.3f} {np.mean(n):>14.3f} {np.mean(l):>14.3f} {f't={t_stat:.1f} {sig}':>18}")

    print(f"\n--- Motion by MIDI-derived articulation label ---")
    print(f"{'Metric':<15} {'MIDI Staccato':>14} {'MIDI Legato':>14} {'S vs L t-test':>18}")
    print("-" * 65)
    for metric in ['z_range', 'max_speed', 'mean_speed']:
        s = np.array(midi_staccato_motion[metric]) if midi_staccato_motion[metric] else np.array([0])
        l = np.array(midi_legato_motion[metric]) if midi_legato_motion[metric] else np.array([0])
        if len(s) > 1 and len(l) > 1:
            t_stat, p_val = stats.ttest_ind(s, l)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        else:
            t_stat, sig = 0, "n/a"
        print(f"{metric:<15} {np.mean(s):>14.3f} {np.mean(l):>14.3f} {f't={t_stat:.1f} {sig}':>18}")

    # Effect size comparison
    print(f"\n--- Effect size comparison (Cohen's d): GT labels vs MIDI labels ---")
    print(f"{'Metric':<15} {'d(GT S vs L)':>14} {'d(MIDI S vs L)':>16} {'Ratio':>8}")
    print("-" * 58)
    for metric in ['z_range', 'max_speed', 'mean_speed']:
        gt_s_arr = np.array(gt_staccato_motion[metric])
        gt_l_arr = np.array(gt_legato_motion[metric])
        midi_s_arr = np.array(midi_staccato_motion[metric])
        midi_l_arr = np.array(midi_legato_motion[metric])

        if len(gt_s_arr) > 1 and len(gt_l_arr) > 1 and len(midi_s_arr) > 1 and len(midi_l_arr) > 1:
            pooled_gt = np.sqrt((np.var(gt_s_arr) + np.var(gt_l_arr)) / 2)
            d_gt = (np.mean(gt_s_arr) - np.mean(gt_l_arr)) / (pooled_gt + 1e-8)
            pooled_midi = np.sqrt((np.var(midi_s_arr) + np.var(midi_l_arr)) / 2)
            d_midi = (np.mean(midi_s_arr) - np.mean(midi_l_arr)) / (pooled_midi + 1e-8)
            ratio = abs(d_midi) / (abs(d_gt) + 1e-8)
            print(f"{metric:<15} {d_gt:>14.3f} {d_midi:>16.3f} {ratio:>8.2f}")


# ============================================================
# Comparison 3: IOI variation vs GT tempo markings (rit/accel)
# ============================================================

def comparison3_tempo(recordings):
    """Compare MIDI IOI variation vs GT tempo markings (rit, accel)."""
    print("\n" + "=" * 80)
    print("COMPARISON 3: MIDI IOI Variation vs GT Tempo Markings (rit/accel)")
    print("=" * 80)

    results = defaultdict(list)  # 'rit', 'accel', 'none' -> list of dicts
    n_used = 0

    for rec in recordings:
        motion_df = load_motion(rec['motion_file'])
        align_df = load_align_notetime(rec['align_file'])
        expr_df = load_expression(rec['expr_file'])
        n_frames = len(motion_df)

        if len(expr_df) == 0:
            continue

        score_to_perf = build_score_to_perf_map(align_df)

        tempo_rows = expr_df[expr_df['expression'].isin(TEMPO_TYPES)]
        # Also need non-tempo regions for comparison
        has_tempo = len(tempo_rows) > 0

        # Divide into windows (2s)
        notes_sorted = align_df.sort_values('onset')
        if len(notes_sorted) < 10:
            continue

        min_t = notes_sorted['onset'].min()
        max_t = notes_sorted['onset'].max()
        window_sec = 2.0

        n_used += 1
        for t_start in np.arange(min_t, max_t - window_sec, window_sec / 2):
            t_end = t_start + window_sec

            # Compute MIDI IOI CV
            ioi_cv = compute_ioi_cv_in_window(align_df, t_start, t_end)
            if np.isnan(ioi_cv):
                continue

            # Map performance time back to approximate score position for GT lookup
            # Use inverse: find notes in this time range, get their score positions
            mask = (align_df['onset'] >= t_start) & (align_df['onset'] < t_end)
            if mask.sum() < 2:
                continue
            score_range = (align_df[mask]['score_position'].min(), align_df[mask]['score_position'].max())

            # Determine GT label
            gt_label = 'none'
            if has_tempo:
                for _, tr in tempo_rows.iterrows():
                    if tr['onset'] < score_range[1] and tr['offset'] > score_range[0]:
                        gt_label = tr['expression']
                        break

            # Motion features
            f_start = max(0, time_to_frame(t_start))
            f_end = min(n_frames, time_to_frame(t_end))
            if f_end - f_start < 60:
                continue

            spine_pos = get_group_positions(motion_df.iloc[f_start:f_end], 'spine')
            spine_speed = compute_speed(spine_pos).mean()
            c7_pos = get_joint_positions(motion_df.iloc[f_start:f_end], 'C7')
            c7_y_std = np.std(c7_pos[:, 1])

            results[gt_label].append({
                'ioi_cv': ioi_cv,
                'spine_speed': spine_speed,
                'c7_y_std': c7_y_std,
            })

    print(f"Recordings used: {n_used}")

    print(f"\n--- MIDI IOI CV distribution by GT tempo label ---")
    print(f"{'GT Label':<10} {'N':>8} {'Mean IOI CV':>14} {'Median':>10} {'Std':>10}")
    print("-" * 58)
    for label in ['none', 'accel', 'rit']:
        if results[label]:
            df = pd.DataFrame(results[label])
            print(f"{label:<10} {len(df):>8} {df['ioi_cv'].mean():>14.3f} "
                  f"{df['ioi_cv'].median():>10.3f} {df['ioi_cv'].std():>10.3f}")

    # Statistical test
    if results['rit'] and results['none']:
        rit_cv = [d['ioi_cv'] for d in results['rit']]
        none_cv = [d['ioi_cv'] for d in results['none']]
        t, p = stats.ttest_ind(rit_cv, none_cv)
        d = (np.mean(rit_cv) - np.mean(none_cv)) / np.sqrt((np.var(rit_cv) + np.var(none_cv))/2)
        print(f"\n  rit vs none IOI CV: t={t:.2f}, p={p:.2e}, Cohen's d={d:.3f}")

    if results['accel'] and results['none']:
        accel_cv = [d['ioi_cv'] for d in results['accel']]
        none_cv = [d['ioi_cv'] for d in results['none']]
        t, p = stats.ttest_ind(accel_cv, none_cv)
        d = (np.mean(accel_cv) - np.mean(none_cv)) / np.sqrt((np.var(accel_cv) + np.var(none_cv))/2)
        print(f"  accel vs none IOI CV: t={t:.2f}, p={p:.2e}, Cohen's d={d:.3f}")

    # Motion comparison: GT label vs MIDI IOI CV
    print(f"\n--- Motion by GT tempo label ---")
    print(f"{'GT Label':<10} {'Spine Speed':>14} {'C7 Y std':>14}")
    print("-" * 42)
    for label in ['none', 'accel', 'rit']:
        if results[label]:
            df = pd.DataFrame(results[label])
            print(f"{label:<10} {df['spine_speed'].mean():>14.4f} {df['c7_y_std'].mean():>14.3f}")

    # Compare predictive power
    print(f"\n--- Predictive power: MIDI IOI CV vs GT label -> Spine Speed ---")
    all_data = []
    for label, entries in results.items():
        for e in entries:
            e['is_rit'] = 1 if label == 'rit' else 0
            e['is_tempo_change'] = 1 if label in ('rit', 'accel') else 0
            all_data.append(e)
    all_df = pd.DataFrame(all_data)

    r_midi, p_midi = stats.pearsonr(all_df['ioi_cv'], all_df['spine_speed'])
    r_gt, p_gt = stats.pearsonr(all_df['is_tempo_change'], all_df['spine_speed'])
    print(f"  r(MIDI IOI CV → spine_speed) = {r_midi:.3f} (p={p_midi:.2e})")
    print(f"  r(GT tempo label → spine_speed) = {r_gt:.3f} (p={p_gt:.2e})")
    print(f"  MIDI / GT ratio: {abs(r_midi)/max(abs(r_gt), 1e-8):.2f}x")


# ============================================================
# Comparison 4: Beat-level analysis (MIDI onset pattern vs GT beat)
# ============================================================

def comparison4_beat_structure(recordings):
    """Compare motion at GT beats/downbeats vs MIDI-inferred metric positions."""
    print("\n" + "=" * 80)
    print("COMPARISON 4: Motion at GT Beats/Downbeats vs MIDI Onset Patterns")
    print("=" * 80)

    gt_beat_motion = defaultdict(list)
    gt_downbeat_motion = defaultdict(list)
    gt_offbeat_motion = defaultdict(list)

    # MIDI-derived: high onset density frames vs low
    midi_high_density_motion = defaultdict(list)
    midi_low_density_motion = defaultdict(list)

    n_used = 0
    for rec in recordings:
        if rec['beat_file'] is None:
            continue

        motion_df = load_motion(rec['motion_file'])
        align_df = load_align_notetime(rec['align_file'])
        n_frames = len(motion_df)
        beats = load_beats(rec['beat_file'])
        downbeats = load_downbeats(rec['downbeat_file']) if rec['downbeat_file'] else np.array([])

        if len(beats) < 4 or len(align_df) < 10:
            continue

        score_to_perf = build_score_to_perf_map(align_df)
        n_used += 1

        downbeat_set = set(downbeats.tolist())
        context_frames = int(0.1 * MOTION_FPS)  # 100ms

        # GT beat analysis
        for beat_pos in beats:
            perf_time = score_to_perf(beat_pos)
            center = time_to_frame(perf_time)
            if center < context_frames or center >= n_frames - context_frames:
                continue

            start = center - context_frames
            end = center + context_frames

            spine_pos = get_group_positions(motion_df.iloc[start:end], 'spine')
            head_pos = get_group_positions(motion_df.iloc[start:end], 'head')

            spine_z_vel = np.mean(np.abs(np.diff(spine_pos[:, 2])))  # vertical acceleration
            head_z_vel = np.mean(np.abs(np.diff(head_pos[:, 2])))

            entry = {'spine_z_vel': spine_z_vel, 'head_z_vel': head_z_vel}

            if beat_pos in downbeat_set:
                for k, v in entry.items():
                    gt_downbeat_motion[k].append(v)
            else:
                for k, v in entry.items():
                    gt_beat_motion[k].append(v)

        # Off-beat: halfway between consecutive beats
        for i in range(len(beats) - 1):
            offbeat_pos = (beats[i] + beats[i+1]) / 2
            perf_time = score_to_perf(offbeat_pos)
            center = time_to_frame(perf_time)
            if center < context_frames or center >= n_frames - context_frames:
                continue

            start = center - context_frames
            end = center + context_frames

            spine_pos = get_group_positions(motion_df.iloc[start:end], 'spine')
            head_pos = get_group_positions(motion_df.iloc[start:end], 'head')

            spine_z_vel = np.mean(np.abs(np.diff(spine_pos[:, 2])))
            head_z_vel = np.mean(np.abs(np.diff(head_pos[:, 2])))

            for k, v in {'spine_z_vel': spine_z_vel, 'head_z_vel': head_z_vel}.items():
                gt_offbeat_motion[k].append(v)

    print(f"Recordings used: {n_used}")

    print(f"\n--- Vertical motion velocity at metric positions ---")
    print(f"{'Position':<15} {'N':>8} {'Spine Z vel':>14} {'Head Z vel':>14}")
    print("-" * 55)
    for label, data in [('Downbeat', gt_downbeat_motion), ('Beat', gt_beat_motion), ('Off-beat', gt_offbeat_motion)]:
        if data['spine_z_vel']:
            print(f"{label:<15} {len(data['spine_z_vel']):>8} {np.mean(data['spine_z_vel']):>14.4f} "
                  f"{np.mean(data['head_z_vel']):>14.4f}")

    # Statistical tests
    if gt_downbeat_motion['spine_z_vel'] and gt_offbeat_motion['spine_z_vel']:
        t, p = stats.ttest_ind(gt_downbeat_motion['spine_z_vel'], gt_offbeat_motion['spine_z_vel'])
        d = (np.mean(gt_downbeat_motion['spine_z_vel']) - np.mean(gt_offbeat_motion['spine_z_vel'])) / \
            np.sqrt((np.var(gt_downbeat_motion['spine_z_vel']) + np.var(gt_offbeat_motion['spine_z_vel']))/2)
        print(f"\n  Downbeat vs Off-beat (spine Z vel): t={t:.2f}, p={p:.2e}, d={d:.3f}")

    if gt_downbeat_motion['spine_z_vel'] and gt_beat_motion['spine_z_vel']:
        t, p = stats.ttest_ind(gt_downbeat_motion['spine_z_vel'], gt_beat_motion['spine_z_vel'])
        d = (np.mean(gt_downbeat_motion['spine_z_vel']) - np.mean(gt_beat_motion['spine_z_vel'])) / \
            np.sqrt((np.var(gt_downbeat_motion['spine_z_vel']) + np.var(gt_beat_motion['spine_z_vel']))/2)
        print(f"  Downbeat vs Beat (spine Z vel): t={t:.2f}, p={p:.2e}, d={d:.3f}")

    # Key insight: can MIDI onset pattern capture beat structure?
    print(f"\n--- Key question: MIDI onsets at GT beats vs off-beats ---")
    # Check what fraction of GT beats have MIDI onsets nearby
    n_beats_with_onset = 0
    n_beats_total = 0
    n_downbeats_with_onset = 0
    n_downbeats_total = 0

    for rec in recordings:
        if rec['beat_file'] is None:
            continue
        align_df = load_align_notetime(rec['align_file'])
        beats = load_beats(rec['beat_file'])
        downbeats = load_downbeats(rec['downbeat_file']) if rec['downbeat_file'] else np.array([])
        downbeat_set = set(downbeats.tolist())

        onset_positions = set(align_df['score_position'].values)

        for b in beats:
            # Check if any note onset within 0.125 beats of this beat
            has_onset = any(abs(b - op) < 0.125 for op in onset_positions)
            n_beats_total += 1
            if has_onset:
                n_beats_with_onset += 1
            if b in downbeat_set:
                n_downbeats_total += 1
                if has_onset:
                    n_downbeats_with_onset += 1

    print(f"  GT beats with MIDI onset (±0.125 beats): {n_beats_with_onset}/{n_beats_total} = {n_beats_with_onset/max(n_beats_total,1)*100:.1f}%")
    print(f"  GT downbeats with MIDI onset: {n_downbeats_with_onset}/{n_downbeats_total} = {n_downbeats_with_onset/max(n_downbeats_total,1)*100:.1f}%")


# ============================================================
# Summary: Information gap analysis
# ============================================================

def summary_information_gap():
    print("\n" + "=" * 80)
    print("SUMMARY: MIDI vs GT Information Gap for Model Design")
    print("=" * 80)
    print("""
┌─────────────────┬──────────────────────────┬───────────────────────────┬──────────────┐
│ Musical Feature │ MIDI Proxy               │ GT Annotation             │ Gap          │
├─────────────────┼──────────────────────────┼───────────────────────────┼──────────────┤
│ Dynamics        │ Onset density,           │ p/mp/mf/f/ff labels       │ See Comp. 1  │
│                 │ polyphony count           │                           │              │
│ Articulation    │ Duration/IOI ratio        │ legato/staccato labels    │ See Comp. 2  │
│ Tempo changes   │ IOI coefficient of var.   │ rit/accel markings        │ See Comp. 3  │
│ Metric structure│ Onset pattern regularity  │ Beat/downbeat positions   │ See Comp. 4  │
└─────────────────┴──────────────────────────┴───────────────────────────┴──────────────┘

Key implications for model design:
- If MIDI proxy ≈ GT in predicting motion → current MIDI-only input is sufficient
- If GT >> MIDI proxy → information loss; model may need architectural compensation
  (e.g., ARIA embeddings, attention to longer context)
- If MIDI proxy > GT → MIDI captures continuous variation better than discrete labels
""")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Loading recordings...")
    recordings = find_all_recordings()
    print(f"Found {len(recordings)} recordings\n")

    comparison1_dynamics(recordings)
    comparison2_articulation(recordings)
    comparison3_tempo(recordings)
    comparison4_beat_structure(recordings)
    summary_information_gap()

    print("\nALL COMPARISONS COMPLETE")
