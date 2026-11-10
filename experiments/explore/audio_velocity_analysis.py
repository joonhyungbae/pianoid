"""
Explore using audio WAV to recover per-note velocity for MIDI enrichment.

Key question: Can we extract per-note dynamics from audio and add it to MIDI,
recovering the missing velocity information in MOSA dataset?

Approach:
1. Compute frame-level RMS energy from audio
2. Extract per-note onset RMS as velocity proxy
3. Validate against GT dynamic annotations
4. Compare: audio-derived velocity vs onset density for motion prediction
"""

import os
import glob
import wave
import struct
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

MOTION_FPS = 120
AUDIO_SR = 44100
DATASET_ROOT = "MOSA_dataset/yp"

BODY_GROUPS = {
    'head': ['RFHD', 'LFHD', 'RBHD', 'LBHD'],
    'spine': ['C7', 'T10', 'CLAV', 'STRN'],
    'r_wrist': ['RWRA', 'RWRB'],
    'l_wrist': ['LWRA', 'LWRB'],
    'pelvis': ['RASI', 'LASI', 'RPSI', 'LPSI'],
}

DYNAMIC_ORDER = {'ppp': 0, 'pp': 1, 'p': 2, 'mp': 3, 'mf': 4, 'f': 5, 'ff': 6}

# ============================================================
# Data Loading
# ============================================================

def find_all_recordings():
    recordings = []
    for motion_file in glob.glob(os.path.join(DATASET_ROOT, "**", "*_motion_norm.csv"), recursive=True):
        rec_dir = os.path.dirname(motion_file)
        ann_dir = os.path.join(rec_dir, "annotation", "annotations")
        align_files = glob.glob(os.path.join(ann_dir, "*_align_notetime.csv"))
        audio_files = glob.glob(os.path.join(rec_dir, "*_audio.wav"))
        expr_files = glob.glob(os.path.join(ann_dir, "*_expression.csv"))
        if align_files and audio_files:
            basename = os.path.basename(motion_file).replace("_motion_norm.csv", "")
            recordings.append({
                'id': basename,
                'motion_file': motion_file,
                'align_file': align_files[0],
                'audio_file': audio_files[0],
                'expr_file': expr_files[0] if expr_files else None,
                'ann_dir': ann_dir,
            })
    return recordings


def load_audio_rms(filepath, hop_sec=1.0/60, window_sec=0.05):
    """Load audio and compute RMS energy at regular intervals.

    Returns RMS at ~60fps (matching motion fps after 2x downsample).
    Also returns RMS at finer resolution for per-note extraction.
    """
    with wave.open(filepath, 'r') as f:
        n_channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        framerate = f.getframerate()
        n_frames = f.getnframes()
        raw = f.readframes(n_frames)

    # Convert to float array
    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
    else:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)

    if n_channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)

    # Normalize
    samples = samples / (np.max(np.abs(samples)) + 1e-8)

    # Compute RMS at desired hop rate
    hop_samples = int(hop_sec * framerate)
    window_samples = int(window_sec * framerate)

    n_hops = len(samples) // hop_samples
    rms = np.zeros(n_hops)

    for i in range(n_hops):
        center = i * hop_samples
        start = max(0, center - window_samples // 2)
        end = min(len(samples), center + window_samples // 2)
        segment = samples[start:end]
        rms[i] = np.sqrt(np.mean(segment ** 2))

    return rms, hop_sec  # rms at ~60fps


def get_note_onset_rms(audio_rms, rms_hop_sec, onset_sec, window_sec=0.05):
    """Get RMS around a note onset."""
    center_idx = int(onset_sec / rms_hop_sec)
    half_w = max(1, int(window_sec / rms_hop_sec / 2))

    start = max(0, center_idx - half_w)
    end = min(len(audio_rms), center_idx + half_w + 1)

    if end <= start:
        return 0.0

    # Peak RMS around onset (attack)
    return np.max(audio_rms[start:end])


def load_motion(filepath):
    return pd.read_csv(filepath)


def load_align_notetime(filepath):
    return pd.read_csv(filepath)


def load_expression(filepath):
    if filepath is None:
        return pd.DataFrame(columns=['onset', 'offset', 'expression'])
    df = pd.read_csv(filepath)
    return df if len(df) > 0 else pd.DataFrame(columns=['onset', 'offset', 'expression'])


def build_score_to_perf_map(align_df):
    df = align_df.sort_values('score_position')
    score_pos = df['score_position'].values
    perf_time = df['onset'].values
    _, unique_idx = np.unique(score_pos, return_index=True)
    score_pos = score_pos[unique_idx]
    perf_time = perf_time[unique_idx]
    def map_fn(sp):
        return np.interp(sp, score_pos, perf_time)
    return map_fn


def get_group_positions(motion_df, group_name):
    joints = BODY_GROUPS[group_name]
    positions = []
    for j in joints:
        cols = [f"{j}_{ax}" for ax in ['X', 'Y', 'Z']]
        positions.append(motion_df[cols].values)
    return np.stack(positions, axis=0).mean(axis=0)


def time_to_frame(t_sec, fps=MOTION_FPS):
    return int(round(t_sec * fps))


def compute_speed(positions):
    return np.linalg.norm(np.diff(positions, axis=0), axis=1)


# ============================================================
# Part 1: Validate audio RMS against GT dynamics
# ============================================================

def part1_audio_vs_gt_dynamics(recordings):
    print("=" * 80)
    print("PART 1: Audio RMS vs GT Dynamic Markings")
    print("=" * 80)

    results = defaultdict(list)
    n_used = 0

    for rec in recordings:
        expr_df = load_expression(rec['expr_file'])
        if len(expr_df) == 0:
            continue

        align_df = load_align_notetime(rec['align_file'])
        dynamic_labels = set(DYNAMIC_ORDER.keys())
        dyn_rows = expr_df[expr_df['expression'].isin(dynamic_labels)]
        if len(dyn_rows) == 0:
            continue

        audio_rms, rms_hop = load_audio_rms(rec['audio_file'])
        score_to_perf = build_score_to_perf_map(align_df)
        n_used += 1

        for _, row in dyn_rows.iterrows():
            label = row['expression']
            perf_start = score_to_perf(row['onset'])
            perf_end = score_to_perf(row['offset'])
            if perf_end - perf_start < 0.5:
                continue

            # Get mean RMS in this region
            idx_start = int(perf_start / rms_hop)
            idx_end = int(perf_end / rms_hop)
            idx_start = max(0, min(idx_start, len(audio_rms) - 1))
            idx_end = max(idx_start + 1, min(idx_end, len(audio_rms)))

            mean_rms = np.mean(audio_rms[idx_start:idx_end])
            peak_rms = np.max(audio_rms[idx_start:idx_end])

            results[label].append({
                'mean_rms': mean_rms,
                'peak_rms': peak_rms,
                'dynamic_level': DYNAMIC_ORDER[label],
            })

    print(f"Recordings with dynamics: {n_used}")

    ordered = sorted([l for l in results if results[l]], key=lambda x: DYNAMIC_ORDER[x])
    print(f"\n{'Dynamic':<8} {'N':>6} {'Mean RMS':>12} {'Peak RMS':>12}")
    print("-" * 42)
    for label in ordered:
        data = results[label]
        df = pd.DataFrame(data)
        print(f"{label:<8} {len(df):>6} {df['mean_rms'].mean():>12.4f} {df['peak_rms'].mean():>12.4f}")

    # Correlation
    all_data = []
    for entries in results.values():
        all_data.extend(entries)
    all_df = pd.DataFrame(all_data)

    r_mean, p_mean = stats.pearsonr(all_df['dynamic_level'], all_df['mean_rms'])
    r_peak, p_peak = stats.pearsonr(all_df['dynamic_level'], all_df['peak_rms'])
    print(f"\n  dynamic_level vs mean_rms: r={r_mean:.3f} (p={p_mean:.2e})")
    print(f"  dynamic_level vs peak_rms: r={r_peak:.3f} (p={p_peak:.2e})")

    return all_df


# ============================================================
# Part 2: Per-note audio velocity extraction
# ============================================================

def part2_per_note_velocity(recordings):
    print("\n" + "=" * 80)
    print("PART 2: Per-Note Audio-Derived Velocity")
    print("=" * 80)

    all_notes = []
    n_processed = 0

    for rec in recordings[:50]:  # subset for speed
        align_df = load_align_notetime(rec['align_file'])
        audio_rms, rms_hop = load_audio_rms(rec['audio_file'])
        n_processed += 1

        for _, row in align_df.iterrows():
            onset_rms = get_note_onset_rms(audio_rms, rms_hop, row['onset'])
            all_notes.append({
                'onset': row['onset'],
                'midi_number': row['midi_number'],
                'onset_rms': onset_rms,
                'rec_id': rec['id'],
            })

    notes_df = pd.DataFrame(all_notes)

    print(f"Processed {n_processed} recordings, {len(notes_df)} notes")
    print(f"\nOnset RMS stats: mean={notes_df['onset_rms'].mean():.4f}, "
          f"std={notes_df['onset_rms'].std():.4f}, "
          f"min={notes_df['onset_rms'].min():.4f}, "
          f"max={notes_df['onset_rms'].max():.4f}")

    # Distribution of audio-derived velocity
    print(f"\nOnset RMS percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(notes_df['onset_rms'], p):.4f}")

    # Variation within a recording
    print(f"\nPer-recording onset RMS coefficient of variation:")
    cvs = []
    for rec_id, group in notes_df.groupby('rec_id'):
        cv = group['onset_rms'].std() / (group['onset_rms'].mean() + 1e-8)
        cvs.append(cv)
    print(f"  Mean CV: {np.mean(cvs):.3f}, range: [{np.min(cvs):.3f}, {np.max(cvs):.3f}]")

    return notes_df


# ============================================================
# Part 3: Audio velocity vs motion (compared to onset density)
# ============================================================

def part3_audio_vel_vs_motion(recordings):
    print("\n" + "=" * 80)
    print("PART 3: Audio-Derived Velocity vs Onset Density → Motion Prediction")
    print("=" * 80)

    window_sec = 4.0
    results = defaultdict(list)

    for rec in recordings[:80]:  # subset
        motion_df = load_motion(rec['motion_file'])
        align_df = load_align_notetime(rec['align_file'])
        audio_rms, rms_hop = load_audio_rms(rec['audio_file'])
        n_frames = len(motion_df)

        window_frames = int(window_sec * MOTION_FPS)
        for start in range(0, n_frames - window_frames, window_frames // 2):
            end = start + window_frames
            t_start = start / MOTION_FPS
            t_end = end / MOTION_FPS

            # Audio-derived: mean RMS in window
            rms_start = int(t_start / rms_hop)
            rms_end = int(t_end / rms_hop)
            rms_start = max(0, min(rms_start, len(audio_rms) - 1))
            rms_end = max(rms_start + 1, min(rms_end, len(audio_rms)))
            win_rms = np.mean(audio_rms[rms_start:rms_end])

            # Per-note onset RMS in window
            mask = (align_df['onset'] >= t_start) & (align_df['onset'] < t_end)
            win_notes = align_df[mask]
            if len(win_notes) == 0:
                continue

            note_rms_vals = []
            for _, row in win_notes.iterrows():
                note_rms_vals.append(get_note_onset_rms(audio_rms, rms_hop, row['onset']))
            mean_note_rms = np.mean(note_rms_vals)

            # Onset density
            onset_density = len(win_notes) / window_sec

            # Motion features
            spine_pos = get_group_positions(motion_df.iloc[start:end], 'spine')
            spine_amp = np.mean([np.std(spine_pos[:, i]) for i in range(3)])

            rw_pos = get_group_positions(motion_df.iloc[start:end], 'r_wrist')
            lw_pos = get_group_positions(motion_df.iloc[start:end], 'l_wrist')
            wrist_speed = (compute_speed(rw_pos).mean() + compute_speed(lw_pos).mean()) / 2
            wrist_z_amp = (np.std(rw_pos[:, 2]) + np.std(lw_pos[:, 2])) / 2

            head_pos = get_group_positions(motion_df.iloc[start:end], 'head')
            head_amp = np.mean([np.std(head_pos[:, i]) for i in range(3)])

            results['win_rms'].append(win_rms)
            results['mean_note_rms'].append(mean_note_rms)
            results['onset_density'].append(onset_density)
            results['spine_amp'].append(spine_amp)
            results['wrist_speed'].append(wrist_speed)
            results['wrist_z_amp'].append(wrist_z_amp)
            results['head_amp'].append(head_amp)

    df = pd.DataFrame(results)
    print(f"Total windows: {len(df)}")

    # Compare predictive power: audio RMS vs onset density → motion
    print(f"\n--- Predictive power comparison: Audio RMS vs Onset Density → Motion ---")
    print(f"{'Motion Target':<18} {'r(AudioRMS)':>14} {'r(NoteRMS)':>14} {'r(OnsetDens)':>14} {'Best':>8}")
    print("-" * 75)

    for target in ['spine_amp', 'wrist_speed', 'wrist_z_amp', 'head_amp']:
        r_rms, _ = stats.pearsonr(df['win_rms'], df[target])
        r_note, _ = stats.pearsonr(df['mean_note_rms'], df[target])
        r_dens, _ = stats.pearsonr(df['onset_density'], df[target])

        best = 'Audio' if abs(r_rms) >= abs(r_dens) else 'Density'
        if abs(r_note) > max(abs(r_rms), abs(r_dens)):
            best = 'NoteRMS'

        print(f"{target:<18} {r_rms:>14.3f} {r_note:>14.3f} {r_dens:>14.3f} {best:>8}")

    # Audio RMS vs onset density correlation
    r_rms_dens, _ = stats.pearsonr(df['win_rms'], df['onset_density'])
    print(f"\n  Audio RMS ↔ Onset Density correlation: r={r_rms_dens:.3f}")

    # Combined feature (RMS + density) via partial correlation
    print(f"\n--- Partial correlations (unique contribution) ---")
    from numpy.linalg import lstsq

    for target in ['spine_amp', 'wrist_speed', 'head_amp']:
        y = df[target].values
        x_rms = df['win_rms'].values
        x_dens = df['onset_density'].values

        # Residualize RMS from density
        A = np.column_stack([x_dens, np.ones(len(x_dens))])
        coef, _, _, _ = lstsq(A, x_rms, rcond=None)
        rms_resid = x_rms - A @ coef

        # Residualize target from density
        coef2, _, _, _ = lstsq(A, y, rcond=None)
        y_resid = y - A @ coef2

        r_partial, p_partial = stats.pearsonr(rms_resid, y_resid)
        sig = "***" if p_partial < 0.001 else "**" if p_partial < 0.01 else "*" if p_partial < 0.05 else ""
        print(f"  Audio RMS → {target:<15} (controlling for density): r_partial={r_partial:.3f} {sig}")


# ============================================================
# Part 4: Feasibility of audio → MIDI velocity enrichment
# ============================================================

def part4_enrichment_feasibility(recordings):
    print("\n" + "=" * 80)
    print("PART 4: Audio → MIDI Velocity Enrichment Feasibility")
    print("=" * 80)

    # Validate against GT dynamics using per-note RMS
    gt_results = defaultdict(list)
    n_used = 0

    for rec in recordings[:100]:
        expr_df = load_expression(rec['expr_file'])
        if len(expr_df) == 0:
            continue

        align_df = load_align_notetime(rec['align_file'])
        dynamic_labels = set(DYNAMIC_ORDER.keys())
        dyn_rows = expr_df[expr_df['expression'].isin(dynamic_labels)]
        if len(dyn_rows) == 0:
            continue

        audio_rms, rms_hop = load_audio_rms(rec['audio_file'])
        score_to_perf = build_score_to_perf_map(align_df)
        n_used += 1

        for _, note_row in align_df.iterrows():
            onset_rms = get_note_onset_rms(audio_rms, rms_hop, note_row['onset'])
            score_pos = note_row['score_position']

            # Find GT dynamic at this score position
            gt_label = None
            for _, dr in dyn_rows.iterrows():
                if dr['onset'] <= score_pos < dr['offset']:
                    gt_label = dr['expression']
                    break

            if gt_label is not None:
                gt_results[gt_label].append(onset_rms)

    print(f"Recordings used: {n_used}")

    ordered = sorted([l for l in gt_results if gt_results[l]], key=lambda x: DYNAMIC_ORDER[x])
    print(f"\n{'Dynamic':<8} {'N':>8} {'Mean NoteRMS':>14} {'Std':>10}")
    print("-" * 44)
    for label in ordered:
        vals = np.array(gt_results[label])
        print(f"{label:<8} {len(vals):>8} {np.mean(vals):>14.4f} {np.std(vals):>10.4f}")

    # Can we separate dynamics by note-level RMS?
    if 'p' in gt_results and 'f' in gt_results:
        p_vals = np.array(gt_results['p'])
        f_vals = np.array(gt_results['f'])
        t, p_val = stats.ttest_ind(p_vals, f_vals)
        d = (np.mean(f_vals) - np.mean(p_vals)) / np.sqrt((np.var(p_vals) + np.var(f_vals)) / 2)
        print(f"\n  p vs f (note-level RMS): t={t:.2f}, p={p_val:.2e}, Cohen's d={d:.3f}")

    if 'pp' in gt_results and 'ff' in gt_results:
        pp_vals = np.array(gt_results['pp'])
        ff_vals = np.array(gt_results['ff'])
        t, p_val = stats.ttest_ind(pp_vals, ff_vals)
        d = (np.mean(ff_vals) - np.mean(pp_vals)) / np.sqrt((np.var(pp_vals) + np.var(ff_vals)) / 2)
        print(f"  pp vs ff (note-level RMS): t={t:.2f}, p={p_val:.2e}, Cohen's d={d:.3f}")

    # Summary
    print(f"\n--- Summary ---")
    all_vals = []
    all_levels = []
    for label, vals in gt_results.items():
        all_vals.extend(vals)
        all_levels.extend([DYNAMIC_ORDER[label]] * len(vals))

    r, p = stats.pearsonr(all_levels, all_vals)
    rho, _ = stats.spearmanr(all_levels, all_vals)
    print(f"  GT dynamic level vs note-onset RMS: r={r:.3f}, ρ={rho:.3f}")
    print(f"  → Audio RMS is a {'strong' if abs(r) > 0.3 else 'moderate' if abs(r) > 0.15 else 'weak'} predictor of intended dynamics")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Loading recordings...")
    recordings = find_all_recordings()
    print(f"Found {len(recordings)} recordings\n")

    dyn_df = part1_audio_vs_gt_dynamics(recordings)
    notes_df = part2_per_note_velocity(recordings)
    part3_audio_vel_vs_motion(recordings)
    part4_enrichment_feasibility(recordings)

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
If audio RMS strongly correlates with GT dynamics AND predicts motion
better than onset density alone, then:

  1. TRAINING: Extract per-note RMS from audio → use as velocity in onset_density_per_key
  2. INFERENCE: User provides MIDI with velocity (most MIDI files have it)
     OR audio-based velocity extraction as preprocessing

This would recover the missing velocity channel in MOSA and improve
the model's ability to capture dynamics-driven motion.
""")
