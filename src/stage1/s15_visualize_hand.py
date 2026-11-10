#!/usr/bin/env python3
"""
Stage 1 Hand Prediction Visualization

GT vs Predicted hand joint comparison video generation.
2 joints: LWrist, RWrist

Usage:
    python src/stage1/s15_visualize_hand.py --sample_idx 0 --split test
    python src/stage1/s15_visualize_hand.py --sample_idx 0 --split test --frame 500
    python src/stage1/s15_visualize_hand.py --sample_idx 0 --split test --max_frames 600
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Joint layout in the 6-dim wrist vector
JOINT_NAMES = ['LWrist', 'RWrist']
JOINT_NAMES_SHORT = ['LW', 'RW']
JOINT_COLORS = ['#2196F3', '#F44336']  # blue for left, red for right
JOINT_SLICES = [slice(0, 3), slice(3, 6)]

# Piano constants
FPS = 60
NUM_KEYS = 88
PIANO_KEY_MIN = 21  # A0 MIDI number


def load_sample(split: str, sample_idx: int):
    """Load prediction + training cache for a sample."""
    pred_dir = PROJECT_ROOT / 'MOSA_dataset' / 'phase0_hand_predictions' / split
    cache_dir = PROJECT_ROOT / 'MOSA_dataset' / 'training_cache_B' / split

    pred_path = pred_dir / f'{sample_idx:05d}.npz'
    cache_path = cache_dir / f'{sample_idx:05d}.npz'

    pred_data = np.load(pred_path, allow_pickle=True)
    hand_pred = pred_data['hand_pred']  # (T, 6)
    hand_gt = pred_data['hand_gt']      # (T, 6)
    sample_info = json.loads(str(pred_data['sample_info']))

    # Load MIDI data from training cache
    active_keys = None
    key_onset = None
    if cache_path.exists():
        cache_data = np.load(cache_path)
        active_keys = cache_data['active_keys']        # (T, 10) active MIDI key indices
        key_onset = cache_data['key_onset']             # (T, 88) binary onset mask

    return hand_pred, hand_gt, sample_info, active_keys, key_onset


def reshape_joints(hand_vec: np.ndarray) -> np.ndarray:
    """Reshape (T, 6) or (6,) to (T, 2, 3) or (2, 3)."""
    if hand_vec.ndim == 1:
        return hand_vec.reshape(2, 3)
    return hand_vec.reshape(-1, 2, 3)


def compute_errors(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Per-joint error in mm.
    gt, pred: (T, 2, 3)
    Returns: err (T, 4) in mm, stats dict
    """
    err = np.sqrt(((gt - pred) ** 2).sum(axis=-1)) * 1000  # mm
    stats = {
        'mean': err.mean(),
        'max': err.max(),
        'std': err.std(),
        'per_joint': {JOINT_NAMES[i]: err[:, i].mean() for i in range(2)},
    }
    return err, stats


def draw_piano_roll_bar(ax, active_keys: Optional[np.ndarray], frame: int,
                        y_range: Tuple[float, float]):
    """Draw a vertical piano roll bar showing currently active keys."""
    if active_keys is None:
        return
    keys = active_keys[frame]
    valid = keys[keys >= 0]
    if len(valid) == 0:
        return
    y_min, y_max = y_range
    for k in valid:
        # Map key index (0-87) to y coordinate
        y = y_min + (k / (NUM_KEYS - 1)) * (y_max - y_min)
        ax.axhline(y=y, color='#00E5FF', alpha=0.3, linewidth=4, zorder=1)


def visualize_single_frame(
    hand_pred: np.ndarray,
    hand_gt: np.ndarray,
    sample_info: dict,
    active_keys: Optional[np.ndarray] = None,
    key_onset: Optional[np.ndarray] = None,
    frame_idx: int = 0,
    output_path: str = 'frame.png',
    title: str = 'GT vs Predicted',
):
    """Single frame visualization for debugging."""
    gt = reshape_joints(hand_gt)    # (T, 2, 3)
    pred = reshape_joints(hand_pred)

    gt_f = gt[frame_idx]    # (2, 3)
    pred_f = pred[frame_idx]

    err_f = np.sqrt(((gt_f - pred_f) ** 2).sum(axis=-1)) * 1000  # mm per joint

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    piece = sample_info.get('piece', '?')
    performer = sample_info.get('performer', '?')
    fig.suptitle(f'{title} | {piece}/{performer} | Frame {frame_idx} | Mean err: {err_f.mean():.1f}mm',
                 fontsize=14, fontweight='bold')

    # --- Top view (X-Y plane, bird's eye) ---
    ax = axes[0]
    ax.set_title('Top View (X-Y)', fontsize=12)
    ax.set_xlabel('X (left-right, m)')
    ax.set_ylabel('Y (depth, m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # more negative Y = further toward piano

    for i in range(2):
        ax.scatter(gt_f[i, 0], gt_f[i, 1], c=JOINT_COLORS[i], s=150, marker='o',
                   edgecolors='black', linewidths=2, zorder=10)
        ax.scatter(pred_f[i, 0], pred_f[i, 1], s=150, marker='o',
                   facecolors='white', edgecolors=JOINT_COLORS[i], linewidths=3, zorder=10)
        ax.plot([gt_f[i, 0], pred_f[i, 0]], [gt_f[i, 1], pred_f[i, 1]],
                c=JOINT_COLORS[i], alpha=0.5, linewidth=2, zorder=9)
        ax.annotate(JOINT_NAMES_SHORT[i], (pred_f[i, 0], pred_f[i, 1]),
                    fontsize=10, ha='center', va='bottom', alpha=0.8)

    ax.scatter([], [], c='gray', s=80, marker='o', edgecolors='black', label='GT')
    ax.scatter([], [], s=80, marker='o', facecolors='white', edgecolors='gray', linewidths=2, label='Pred')
    ax.legend(loc='upper right', fontsize=10)

    # --- Side view (Y-Z, from the side) ---
    ax = axes[1]
    ax.set_title('Side View (Y-Z)', fontsize=12)
    ax.set_xlabel('Y (depth, m)')
    ax.set_ylabel('Z (height, m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    for i in range(2):
        ax.scatter(gt_f[i, 1], gt_f[i, 2], c=JOINT_COLORS[i], s=150, marker='o',
                   edgecolors='black', linewidths=2, zorder=10)
        ax.scatter(pred_f[i, 1], pred_f[i, 2], s=150, marker='o',
                   facecolors='white', edgecolors=JOINT_COLORS[i], linewidths=3, zorder=10)
        ax.plot([gt_f[i, 1], pred_f[i, 1]], [gt_f[i, 2], pred_f[i, 2]],
                c=JOINT_COLORS[i], alpha=0.5, linewidth=2, zorder=9)
        ax.annotate(JOINT_NAMES_SHORT[i], (pred_f[i, 1], pred_f[i, 2] + 0.003),
                    fontsize=10, ha='center', alpha=0.8)

    # --- Front view (X-Z) ---
    ax = axes[2]
    ax.set_title('Front View (X-Z)', fontsize=12)
    ax.set_xlabel('X (left-right, m)')
    ax.set_ylabel('Z (height, m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    for i in range(2):
        ax.scatter(gt_f[i, 0], gt_f[i, 2], c=JOINT_COLORS[i], s=150, marker='o',
                   edgecolors='black', linewidths=2, zorder=10)
        ax.scatter(pred_f[i, 0], pred_f[i, 2], s=150, marker='o',
                   facecolors='white', edgecolors=JOINT_COLORS[i], linewidths=3, zorder=10)
        ax.plot([gt_f[i, 0], pred_f[i, 0]], [gt_f[i, 2], pred_f[i, 2]],
                c=JOINT_COLORS[i], alpha=0.5, linewidth=2, zorder=9)
        ax.annotate(JOINT_NAMES_SHORT[i], (pred_f[i, 0], pred_f[i, 2] + 0.003),
                    fontsize=10, ha='center', alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Saved: {output_path}')
    print(f'Mean error: {err_f.mean():.1f}mm')
    for i in range(2):
        print(f'  {JOINT_NAMES[i]}: {err_f[i]:.1f}mm')


def visualize_hand_comparison(
    hand_pred: np.ndarray,
    hand_gt: np.ndarray,
    sample_info: dict,
    active_keys: Optional[np.ndarray] = None,
    key_onset: Optional[np.ndarray] = None,
    output_path: str = 'hand_comparison.mp4',
    title: str = 'Stage1 Hand',
    fps: int = FPS,
    max_frames: Optional[int] = None,
    figsize: Tuple[int, int] = (20, 10),
):
    """
    Generate GT vs Predicted hand joint comparison video.

    Layout: 3 columns
      [Top View (X-Y)] [Side View (Y-Z)] [Piano Roll + Error]
    """
    gt = reshape_joints(hand_gt)      # (T, 2, 3)
    pred = reshape_joints(hand_pred)

    T = min(len(gt), len(pred))
    if max_frames is not None:
        T = min(T, max_frames)
    gt = gt[:T]
    pred = pred[:T]
    if active_keys is not None:
        active_keys = active_keys[:T]
    if key_onset is not None:
        key_onset = key_onset[:T]

    err, stats = compute_errors(gt, pred)

    piece = sample_info.get('piece', '?')
    performer = sample_info.get('performer', '?')
    henle = sample_info.get('henle_difficulty', '?')

    print(f'Rendering: {piece}/{performer} | Henle {henle} | {T} frames')
    print(f'Mean error: {stats["mean"]:.1f}mm')
    for name, val in stats['per_joint'].items():
        print(f'  {name}: {val:.1f}mm')

    # Compute axis ranges from data (with padding)
    all_pts = np.concatenate([gt.reshape(-1, 3), pred.reshape(-1, 3)], axis=0)
    pad = 0.03
    x_min, x_max = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    y_min, y_max = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    z_min, z_max = all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad

    # Figure setup
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.25, wspace=0.3)

    ax_top = fig.add_subplot(gs[0, 0])    # Top view (X-Y)
    ax_side = fig.add_subplot(gs[0, 1])   # Side view (Y-Z)
    ax_front = fig.add_subplot(gs[0, 2])  # Front view (X-Z)
    ax_roll = fig.add_subplot(gs[1, :])   # Piano roll + error timeline

    # --- Configure static axes ---
    for ax, xlabel, ylabel, xlim, ylim, ttl in [
        (ax_top,   'X (left-right)', 'Y (depth)', (x_min, x_max), (y_max, y_min), 'Top View (X-Y)'),
        (ax_side,  'Y (depth)',      'Z (height)', (y_min, y_max), (z_min, z_max), 'Side View (Y-Z)'),
        (ax_front, 'X (left-right)', 'Z (height)', (x_min, x_max), (z_min, z_max), 'Front View (X-Z)'),
    ]:
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(ttl, fontsize=11)

    # Legend on top view
    ax_top.scatter([], [], c='gray', s=80, marker='o', edgecolors='black', label='GT')
    ax_top.scatter([], [], s=80, marker='o', facecolors='white', edgecolors='gray', linewidths=2, label='Pred')
    ax_top.legend(loc='upper right', fontsize=9)

    # --- Piano roll axis ---
    ax_roll.set_xlabel('Time (s)', fontsize=10)
    ax_roll.set_ylabel('MIDI Key', fontsize=10)
    ax_roll.set_xlim(0, T / fps)
    ax_roll.set_ylim(0, NUM_KEYS)
    ax_roll.set_title('Piano Roll + Error', fontsize=11)

    # Pre-draw piano roll background
    if key_onset is not None:
        onset_times, onset_keys = np.where(key_onset > 0)
        if len(onset_times) > 0:
            ax_roll.scatter(onset_times / fps, onset_keys, c='#90CAF9', s=2, alpha=0.5, zorder=1)

    # Active keys background (draw all at once for efficiency)
    if active_keys is not None:
        for t in range(0, T, 3):  # subsample for speed
            keys = active_keys[t]
            valid = keys[keys >= 0]
            for k in valid:
                ax_roll.plot([t / fps, (t + 3) / fps], [k, k], c='#2196F3', alpha=0.3, linewidth=1, zorder=2)

    # --- Create scatter/line artists for animation ---
    # Top view (X-Y)
    gt_sc_top = [ax_top.scatter([], [], c=c, s=150, marker='o', edgecolors='black',
                                linewidths=2, zorder=10) for c in JOINT_COLORS]
    pred_sc_top = [ax_top.scatter([], [], s=150, marker='o', facecolors='white',
                                  edgecolors=c, linewidths=3, zorder=10) for c in JOINT_COLORS]
    lines_top = [ax_top.plot([], [], c=c, alpha=0.5, linewidth=2, zorder=9)[0] for c in JOINT_COLORS]
    # (No wrist-hand connection lines needed for 2-joint model)
    annots_top = [ax_top.annotate('', xy=(0, 0), fontsize=9, ha='center', va='bottom', alpha=0.8)
                  for _ in range(2)]

    # Side view (Y-Z)
    gt_sc_side = [ax_side.scatter([], [], c=c, s=150, marker='o', edgecolors='black',
                                  linewidths=2, zorder=10) for c in JOINT_COLORS]
    pred_sc_side = [ax_side.scatter([], [], s=150, marker='o', facecolors='white',
                                    edgecolors=c, linewidths=3, zorder=10) for c in JOINT_COLORS]
    lines_side = [ax_side.plot([], [], c=c, alpha=0.5, linewidth=2, zorder=9)[0] for c in JOINT_COLORS]
    annots_side = [ax_side.annotate('', xy=(0, 0), fontsize=9, ha='center', alpha=0.8)
                   for _ in range(2)]

    # Front view (X-Z)
    gt_sc_front = [ax_front.scatter([], [], c=c, s=150, marker='o', edgecolors='black',
                                    linewidths=2, zorder=10) for c in JOINT_COLORS]
    pred_sc_front = [ax_front.scatter([], [], s=150, marker='o', facecolors='white',
                                      edgecolors=c, linewidths=3, zorder=10) for c in JOINT_COLORS]
    lines_front = [ax_front.plot([], [], c=c, alpha=0.5, linewidth=2, zorder=9)[0] for c in JOINT_COLORS]

    # Timeline cursor
    cursor_line = ax_roll.axvline(x=0, color='red', linewidth=1.5, zorder=10)

    # Title text
    main_title = fig.suptitle('', fontsize=13, fontweight='bold')

    # Stats text
    stats_text = ax_top.text(0.02, 0.02, '', transform=ax_top.transAxes, fontsize=9,
                             verticalalignment='bottom', fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def update(frame):
        gt_f = gt[frame]      # (2, 3)
        pred_f = pred[frame]
        err_f = err[frame]    # (4,)

        # Top view (X-Y)
        for i in range(2):
            gt_sc_top[i].set_offsets([[gt_f[i, 0], gt_f[i, 1]]])
            pred_sc_top[i].set_offsets([[pred_f[i, 0], pred_f[i, 1]]])
            lines_top[i].set_data([gt_f[i, 0], pred_f[i, 0]], [gt_f[i, 1], pred_f[i, 1]])
            annots_top[i].set_position((pred_f[i, 0], pred_f[i, 1]))
            annots_top[i].set_text(JOINT_NAMES_SHORT[i])

        # Side view (Y-Z)
        for i in range(2):
            gt_sc_side[i].set_offsets([[gt_f[i, 1], gt_f[i, 2]]])
            pred_sc_side[i].set_offsets([[pred_f[i, 1], pred_f[i, 2]]])
            lines_side[i].set_data([gt_f[i, 1], pred_f[i, 1]], [gt_f[i, 2], pred_f[i, 2]])
            annots_side[i].set_position((pred_f[i, 1], pred_f[i, 2] + 0.003))
            annots_side[i].set_text(JOINT_NAMES_SHORT[i])

        # Front view (X-Z)
        for i in range(2):
            gt_sc_front[i].set_offsets([[gt_f[i, 0], gt_f[i, 2]]])
            pred_sc_front[i].set_offsets([[pred_f[i, 0], pred_f[i, 2]]])
            lines_front[i].set_data([gt_f[i, 0], pred_f[i, 0]], [gt_f[i, 2], pred_f[i, 2]])

        # Timeline cursor
        cursor_line.set_xdata([frame / fps])

        # Title
        time_s = frame / fps
        main_title.set_text(
            f'{title} | {piece}/{performer} (Henle {henle}) | '
            f'Frame {frame}/{T-1} ({time_s:.1f}s) | Err: {err_f.mean():.0f}mm'
        )

        # Stats
        stats_text.set_text(
            f'Frame: {err_f.mean():.1f}mm\n'
            f'Total: {stats["mean"]:.1f}mm\n'
            + '\n'.join(f'{JOINT_NAMES_SHORT[i]}: {err_f[i]:.0f}mm' for i in range(2))
        )

        artists = (gt_sc_top + pred_sc_top + lines_top +
                   gt_sc_side + pred_sc_side + lines_side +
                   gt_sc_front + pred_sc_front + lines_front +
                   annots_top + annots_side +
                   [cursor_line, main_title, stats_text])
        return artists

    from tqdm import tqdm

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=True)

    writer = FFMpegWriter(
        fps=fps,
        bitrate=2000,
        extra_args=['-preset', 'ultrafast', '-tune', 'animation']
    )

    pbar = tqdm(total=T, desc='Rendering', unit='frame')
    def progress_callback(i, n):
        pbar.update(1)

    ani.save(output_path, writer=writer, dpi=80, progress_callback=progress_callback)
    pbar.close()
    plt.close()

    print(f'Saved: {output_path}')
    return stats


def main():
    parser = argparse.ArgumentParser(description='Stage1 hand prediction visualization')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (auto-generated if not specified)')
    parser.add_argument('--title', type=str, default='Stage1 Hand')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames to render (None=all)')
    parser.add_argument('--fps', type=int, default=FPS)
    parser.add_argument('--frame', type=int, default=None,
                        help='Single frame mode: render one frame as PNG')

    args = parser.parse_args()

    hand_pred, hand_gt, sample_info, active_keys, key_onset = load_sample(args.split, args.sample_idx)

    piece = sample_info.get('piece', '?')
    performer = sample_info.get('performer', '?')

    if args.frame is not None:
        output = args.output or f'vis_hand_{piece}_{performer}_f{args.frame}.png'
        visualize_single_frame(
            hand_pred, hand_gt, sample_info,
            active_keys=active_keys, key_onset=key_onset,
            frame_idx=args.frame, output_path=output, title=args.title,
        )
    else:
        output = args.output or f'vis_hand_{piece}_{performer}.mp4'
        visualize_hand_comparison(
            hand_pred, hand_gt, sample_info,
            active_keys=active_keys, key_onset=key_onset,
            output_path=output, title=args.title,
            fps=args.fps, max_frames=args.max_frames,
        )


if __name__ == '__main__':
    main()
