#!/usr/bin/env python3
"""
Visualize full pipeline results (26-joint GT vs Generated comparison).

Usage:
    # Single sample video
    python experiments/vis/visualize_full_pipeline.py --sample_idx 0

    # Single frame image
    python experiments/vis/visualize_full_pipeline.py --sample_idx 0 --frame 500

    # Short clip
    python experiments/vis/visualize_full_pipeline.py --sample_idx 0 --max_frames 600

    # Batch (first 5 samples)
    python experiments/vis/visualize_full_pipeline.py --batch 5
"""

import argparse
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data.constants import FULL_BODY_EDGES, FULL_BODY_JOINTS, FPS

# Stage-based colors for joints
STAGE_JOINT_COLORS = {}
for i in [14, 10]:  # LWRIST, RWRIST
    STAGE_JOINT_COLORS[i] = '#e74c3c'  # Red - Stage 1 (wrist)
for i in [4, 5, 6, 7]:  # C7, CLAV, T10, STRN
    STAGE_JOINT_COLORS[i] = '#2ecc71'  # Green - Stage 2a (torso)
for i in [0, 1, 2, 3]:  # Head markers
    STAGE_JOINT_COLORS[i] = '#27ae60'  # Dark Green - Stage 2b (head)
for i in [8, 9, 12, 13]:  # RSHO, RELB, LSHO, LELB
    STAGE_JOINT_COLORS[i] = '#f39c12'  # Orange - Stage 3 (arm)
for i in [11, 15]:  # RHAND, LHAND (not predicted, placeholder)
    STAGE_JOINT_COLORS[i] = '#95a5a6'  # Gray
for i in range(16, 26):  # Lower body
    STAGE_JOINT_COLORS[i] = '#3498db'  # Blue - Stage 4 (lower)


def draw_skeleton_3d(ax, joints, title='', alpha=1.0, color_by_stage=True):
    """Draw 26-joint skeleton on a 3D axis."""
    ax.clear()

    for i in range(len(joints)):
        c = STAGE_JOINT_COLORS.get(i, '#95a5a6') if color_by_stage else '#333333'
        ax.scatter(joints[i, 0], joints[i, 1], joints[i, 2],
                   c=c, s=30, alpha=alpha, zorder=5)

    for i, j in FULL_BODY_EDGES:
        if i < len(joints) and j < len(joints):
            c = STAGE_JOINT_COLORS.get(i, '#aaaaaa') if color_by_stage else '#666666'
            ax.plot([joints[i, 0], joints[j, 0]],
                    [joints[i, 1], joints[j, 1]],
                    [joints[i, 2], joints[j, 2]],
                    c=c, linewidth=1.5, alpha=alpha * 0.8)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def set_axes_equal(ax, joints_list):
    """Set equal aspect ratio for 3D plot."""
    all_pts = np.concatenate(joints_list, axis=0).reshape(-1, 3)
    mid = all_pts.mean(axis=0)
    max_range = float((all_pts.max(axis=0) - all_pts.min(axis=0)).max()) / 2 * 1.2
    ax.set_xlim(float(mid[0]) - max_range, float(mid[0]) + max_range)
    ax.set_ylim(float(mid[1]) - max_range, float(mid[1]) + max_range)
    ax.set_zlim(float(mid[2]) - max_range, float(mid[2]) + max_range)


def render_frame(gt, pred, frame_idx, output_path, title=''):
    """Render a single frame comparison (GT left, Pred right)."""
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    draw_skeleton_3d(ax1, gt[frame_idx] * 1000, title=f'GT - Frame {frame_idx}')
    set_axes_equal(ax1, [gt[frame_idx:frame_idx+1] * 1000])

    ax2 = fig.add_subplot(122, projection='3d')
    draw_skeleton_3d(ax2, pred[frame_idx] * 1000, title=f'Generated - Frame {frame_idx}')
    set_axes_equal(ax2, [pred[frame_idx:frame_idx+1] * 1000])

    # MPJPE for this frame
    err = np.sqrt(((gt[frame_idx] - pred[frame_idx]) ** 2).sum(axis=-1)).mean() * 1000
    fig.suptitle(f'{title} | Frame {frame_idx} | MPJPE: {err:.1f}mm', fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def render_video(gt, pred, output_path, title='', max_frames=None, fps=30):
    """Render GT vs Generated comparison video."""
    T = min(len(gt), len(pred))
    if max_frames:
        T = min(T, max_frames)
    gt = gt[:T]
    pred = pred[:T]

    # Compute per-frame MPJPE
    err = np.sqrt(((gt - pred) ** 2).sum(axis=-1)).mean(axis=1) * 1000  # (T,)

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Compute global bounds
    all_pts = np.concatenate([gt.reshape(-1, 3), pred.reshape(-1, 3)]) * 1000
    mid = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.1

    def update(frame):
        draw_skeleton_3d(ax1, gt[frame] * 1000, title=f'GT')
        ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)

        draw_skeleton_3d(ax2, pred[frame] * 1000, title=f'Generated')
        ax2.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax2.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax2.set_zlim(mid[2] - max_range, mid[2] + max_range)

        time_s = frame / FPS
        fig.suptitle(f'{title} | Frame {frame}/{T} ({time_s:.1f}s) | MPJPE: {err[frame]:.0f}mm',
                     fontsize=12)

    ani = FuncAnimation(fig, update, frames=T, interval=1000/fps)
    writer = FFMpegWriter(fps=fps, bitrate=2000,
                          extra_args=['-preset', 'ultrafast', '-tune', 'animation'])

    pbar = tqdm(total=T, desc='Rendering', unit='frame')
    ani.save(output_path, writer=writer, dpi=100,
             progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.close()
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize full pipeline results')
    parser.add_argument('--pred_dir', default='MOSA_dataset/full_body_predictions/test')
    parser.add_argument('--output_dir', default='results/vis')
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--frame', type=int, default=None, help='Single frame PNG')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None, help='Render first N samples')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        indices = range(args.batch)
    else:
        indices = [args.sample_idx]

    for idx in indices:
        sample_dir = Path(args.pred_dir) / f'{idx:05d}'
        if not sample_dir.exists():
            print(f'Skip {idx}: {sample_dir} not found')
            continue

        gt = np.load(sample_dir / 'gt_motion.npy')
        pred = np.load(sample_dir / 'generated_motion.npy')
        print(f'\nSample {idx}: GT {gt.shape}, Pred {pred.shape}')

        mpjpe = np.sqrt(((gt - pred) ** 2).sum(axis=-1)).mean() * 1000
        print(f'  MPJPE: {mpjpe:.1f}mm')

        title = f'Sample {idx}'

        if args.frame is not None:
            render_frame(gt, pred, args.frame,
                         output_dir / f'sample{idx:05d}_frame{args.frame}.png', title)
        else:
            render_video(gt, pred,
                         output_dir / f'sample{idx:05d}.mp4', title,
                         max_frames=args.max_frames, fps=args.fps)


if __name__ == '__main__':
    main()
