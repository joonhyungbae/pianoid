#!/usr/bin/env python3
"""
Visualize Stage 2 torso predictions: GT vs Generated comparison.

Shows 6 joints (C7, CLAV, T10, STRN + LWRIST, RWRIST from Stage 1)
with skeleton edges and per-frame MPJPE.

Usage:
    # Single sample video
    python experiments/vis/visualize_stage2.py --sample_idx 0

    # Short clip (first 600 frames = 10s)
    python experiments/vis/visualize_stage2.py --sample_idx 0 --max_frames 600

    # Single frame PNG
    python experiments/vis/visualize_stage2.py --sample_idx 0 --frame 300

    # Batch (first 5 samples)
    python experiments/vis/visualize_stage2.py --batch 5

    # Custom checkpoint's predictions
    python experiments/vis/visualize_stage2.py --torso_pred_dir MOSA_dataset/phase1_torso_predictions
"""

import argparse
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data.constants import (
    FPS, TORSO_INDICES, LWRIST_IDX, RWRIST_IDX,
    C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
)

# ── Local joint layout (8 joints) ──────────────────────────────────
# 0:LWRIST 1:RWRIST 2:C7 3:CLAV 4:T10 5:STRN (+ optional 6:LHAND 7:RHAND)
# We visualize hand(2) + torso(4) = 6 joints
LOCAL_JOINT_NAMES = ['LWRIST', 'RWRIST', 'C7', 'CLAV', 'T10', 'STRN']
LOCAL_EDGES = [
    # Torso spine
    (2, 4),  # C7 -> T10
    (3, 5),  # CLAV -> STRN
    # C7/CLAV cross
    (2, 3),  # C7 -> CLAV
    (4, 5),  # T10 -> STRN
    # Shoulders (C7/CLAV -> wrist, simplified — no elbow)
    (2, 0),  # C7 -> LWRIST (left side approx)
    (2, 1),  # C7 -> RWRIST (right side approx)
    (3, 0),  # CLAV -> LWRIST
    (3, 1),  # CLAV -> RWRIST
]

# Colors: wrists=red (Stage 1), torso=green (Stage 2)
LOCAL_COLORS = {
    0: '#e74c3c', 1: '#e74c3c',  # wrists
    2: '#2ecc71', 3: '#2ecc71',  # neck
    4: '#27ae60', 5: '#27ae60',  # chest
}
LOCAL_SIZES = {
    0: 50, 1: 50,    # wrists
    2: 60, 3: 60,    # neck
    4: 60, 5: 60,    # chest
}


def draw_skeleton(ax, joints, title='', alpha=1.0):
    """Draw 6-joint skeleton on 3D axes. joints: (6, 3) in mm."""
    ax.clear()
    for i in range(len(joints)):
        ax.scatter(joints[i, 0], joints[i, 1], joints[i, 2],
                   c=LOCAL_COLORS.get(i, '#999'), s=LOCAL_SIZES.get(i, 40),
                   alpha=alpha, zorder=5)
    for i, j in LOCAL_EDGES:
        if i < len(joints) and j < len(joints):
            c = LOCAL_COLORS.get(i, '#aaa')
            ax.plot([joints[i, 0], joints[j, 0]],
                    [joints[i, 1], joints[j, 1]],
                    [joints[i, 2], joints[j, 2]],
                    c=c, linewidth=2.0, alpha=alpha * 0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')


def load_sample(cache_dir, hand_pred_dir, torso_pred_dir, idx):
    """Load GT + predictions for one sample. Returns (gt_6j, pred_6j, T)."""
    cache_path = Path(cache_dir) / 'test' / f'{idx:05d}.npz'
    hand_path = Path(hand_pred_dir) / 'test' / f'{idx:05d}.npz'
    torso_path = Path(torso_pred_dir) / 'test' / f'{idx:05d}.npz'

    # GT: from cache
    cache_data = np.load(cache_path)
    joints = cache_data['joints'].astype(np.float32)  # (T, 26, 3)
    T = len(joints)

    gt_hand = np.concatenate([joints[:, LWRIST_IDX], joints[:, RWRIST_IDX]], axis=-1)  # (T, 6)
    gt_torso = joints[:, TORSO_INDICES].reshape(T, -1)  # (T, 12)
    gt_6j = np.concatenate([gt_hand, gt_torso], axis=-1)  # (T, 18) = 6 joints × 3

    # Predictions
    hand_data = np.load(hand_path)
    hand_pred = hand_data.get('hand_pred', hand_data.get('pred')).astype(np.float32)  # (T, 6)

    torso_data = np.load(torso_path)
    torso_pred = torso_data['torso_pred'].astype(np.float32)  # (T, 12)

    pred_6j = np.concatenate([hand_pred[:T], torso_pred[:T]], axis=-1)  # (T, 18)

    return gt_6j, pred_6j, T


def render_frame(gt, pred, frame_idx, output_path, title=''):
    """Render single frame comparison PNG."""
    fig = plt.figure(figsize=(16, 7))
    gt_f = gt[frame_idx].reshape(6, 3) * 1000
    pred_f = pred[frame_idx].reshape(6, 3) * 1000

    ax1 = fig.add_subplot(121, projection='3d')
    draw_skeleton(ax1, gt_f, title='GT')
    ax2 = fig.add_subplot(122, projection='3d')
    draw_skeleton(ax2, pred_f, title='Generated')

    # Shared bounds
    all_pts = np.concatenate([gt_f, pred_f])
    mid = all_pts.mean(0)
    r = (all_pts.max(0) - all_pts.min(0)).max() / 2 * 1.3
    for ax in [ax1, ax2]:
        ax.set_xlim(mid[0]-r, mid[0]+r); ax.set_ylim(mid[1]-r, mid[1]+r); ax.set_zlim(mid[2]-r, mid[2]+r)

    # Torso-only MPJPE (joints 2-5)
    err = np.sqrt(((gt_f[2:] - pred_f[2:]) ** 2).sum(-1)).mean()
    fig.suptitle(f'{title} | Frame {frame_idx} | Torso MPJPE: {err:.0f}mm', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def render_video(gt, pred, output_path, title='', max_frames=None, fps=30):
    """Render GT vs Generated comparison video (MP4)."""
    T = min(len(gt), len(pred))
    if max_frames:
        T = min(T, max_frames)
    gt = gt[:T]; pred = pred[:T]

    # Per-frame torso MPJPE (joints 2-5 in local layout)
    gt_t = gt.reshape(T, 6, 3)[:, 2:] * 1000
    pred_t = pred.reshape(T, 6, 3)[:, 2:] * 1000
    err = np.sqrt(((gt_t - pred_t) ** 2).sum(-1)).mean(1)  # (T,)

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Global bounds (from all frames)
    all_gt = gt.reshape(-1, 3) * 1000
    all_pred = pred.reshape(-1, 3) * 1000
    all_pts = np.concatenate([all_gt, all_pred])
    mid = all_pts.mean(0)
    r = (all_pts.max(0) - all_pts.min(0)).max() / 2 * 1.1

    def update(frame):
        gt_f = gt[frame].reshape(6, 3) * 1000
        pred_f = pred[frame].reshape(6, 3) * 1000

        draw_skeleton(ax1, gt_f, title='GT')
        draw_skeleton(ax2, pred_f, title='Generated')

        for ax in [ax1, ax2]:
            ax.set_xlim(mid[0]-r, mid[0]+r)
            ax.set_ylim(mid[1]-r, mid[1]+r)
            ax.set_zlim(mid[2]-r, mid[2]+r)
            ax.view_init(elev=15, azim=-60)

        time_s = frame / FPS
        fig.suptitle(f'{title} | {time_s:.1f}s (frame {frame}/{T}) | Torso MPJPE: {err[frame]:.0f}mm',
                     fontsize=12)

    ani = FuncAnimation(fig, update, frames=T, interval=1000/fps)
    writer = FFMpegWriter(fps=fps, bitrate=2000,
                          extra_args=['-preset', 'ultrafast', '-tune', 'animation'])

    pbar = tqdm(total=T, desc='Rendering', unit='frame')
    ani.save(str(output_path), writer=writer, dpi=100,
             progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.close()
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize Stage 2 torso predictions')
    parser.add_argument('--cache_dir', default='MOSA_dataset/training_cache_W05')
    parser.add_argument('--hand_pred_dir', default='MOSA_dataset/phase0_hand_predictions')
    parser.add_argument('--torso_pred_dir', default='MOSA_dataset/phase1_torso_predictions')
    parser.add_argument('--output_dir', default='results/vis/stage2')
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--frame', type=int, default=None, help='Single frame PNG')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None, help='Render first N samples')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = range(args.batch) if args.batch else [args.sample_idx]

    for idx in indices:
        try:
            gt, pred, T = load_sample(args.cache_dir, args.hand_pred_dir,
                                       args.torso_pred_dir, idx)
        except Exception as e:
            print(f'Skip sample {idx}: {e}')
            continue

        gt_r = gt.reshape(T, 6, 3); pred_r = pred.reshape(T, 6, 3)
        torso_err = np.sqrt(((gt_r[:, 2:] - pred_r[:, 2:]) ** 2).sum(-1)).mean() * 1000
        hand_err = np.sqrt(((gt_r[:, :2] - pred_r[:, :2]) ** 2).sum(-1)).mean() * 1000
        print(f'\nSample {idx}: T={T} | Hand MPJPE: {hand_err:.1f}mm | Torso MPJPE: {torso_err:.1f}mm')

        title = f'Sample {idx}'
        if args.frame is not None:
            render_frame(gt, pred, args.frame,
                         output_dir / f's2_sample{idx:05d}_frame{args.frame}.png', title)
        else:
            render_video(gt, pred,
                         output_dir / f's2_sample{idx:05d}.mp4', title,
                         max_frames=args.max_frames, fps=args.fps)


if __name__ == '__main__':
    main()
