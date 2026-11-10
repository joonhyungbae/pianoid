#!/usr/bin/env python3
"""
render_motion_image.py: save a single GT motion frame as an image.

Usage:
    # Default (first frame)
    python experiments/vis/render_motion_image.py --sample_id yp/ba1/yp07/t1

    # Specific frame
    python experiments/vis/render_motion_image.py --sample_id yp/ba1/yp07/t1 --frame_idx 100

    # Multiple views
    python experiments/vis/render_motion_image.py --sample_id yp/ba1/yp07/t1 --multi_view

    # Stage colors
    python experiments/vis/render_motion_image.py --sample_id yp/ba1/yp07/t1 --stage_colors

    # Pick by sample_idx (test split)
    python experiments/vis/render_motion_image.py --sample_idx 0 --split test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Project root
proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))
sys.path.insert(0, str(proj_root / 'src'))

from data.constants import FPS, FULL_BODY_EDGES
from experiments.vis.create_video import (
    load_ground_truth,
    get_sample_id_from_splits,
    get_joint_color,
    get_joint_color_by_stage,
    get_joint_size,
    get_edge_color,
    calculate_bounds,
    setup_axes,
    draw_skeleton,
    STAGE_COLORS,
)


def render_single_image(
    frame: np.ndarray,
    output_path: Path,
    title: str = "Motion",
    view_angle: tuple = (20, 60),
    figsize: tuple = (10, 10),
    dpi: int = 150,
    use_stage_colors: bool = False,
    bounds: tuple = None,
):
    """
    Save a single frame as an image.

    Args:
        frame: (26, 3) joint positions
        output_path: file path to save to
        title: image title
        view_angle: (elevation, azimuth)
        figsize: figure size
        dpi: resolution
        use_stage_colors: whether to apply per-stage colors
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max), or None for auto.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate bounds if not provided
    if bounds is None:
        frame_expanded = frame[np.newaxis, :, :]  # (1, 26, 3)
        bounds = calculate_bounds(frame_expanded)
    
    setup_axes(ax, bounds, title=title, view_angle=view_angle)
    draw_skeleton(ax, frame, use_stage_colors=use_stage_colors)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"Saved: {output_path}")


def render_multi_view(
    frame: np.ndarray,
    output_path: Path,
    title: str = "Motion",
    figsize: tuple = (20, 10),
    dpi: int = 150,
    use_stage_colors: bool = False,
    bounds: tuple = None,
):
    """
    Save multiple views (front, side, top) into a single image.

    Args:
        frame: (26, 3) joint positions
        output_path: file path to save to
        title: image title
        figsize: figure size
        dpi: resolution
        use_stage_colors: whether to apply per-stage colors
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max), or None for auto.
    """
    views = [
        ("Front", (20, 90)),
        ("Side", (20, 0)),
        ("Top", (80, 90)),
    ]
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Calculate bounds if not provided
    if bounds is None:
        frame_expanded = frame[np.newaxis, :, :]
        bounds = calculate_bounds(frame_expanded)
    
    for i, (view_name, view_angle) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        setup_axes(ax, bounds, title=view_name, view_angle=view_angle)
        draw_skeleton(ax, frame, use_stage_colors=use_stage_colors)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"Saved: {output_path}")


def render_comparison_image(
    gt_frame: np.ndarray,
    gen_frame: np.ndarray,
    output_path: Path,
    view_angle: tuple = (20, 60),
    figsize: tuple = (16, 8),
    dpi: int = 150,
    use_stage_colors: bool = False,
):
    """
    Save a GT vs Generated comparison image.

    Args:
        gt_frame: (26, 3) GT joint positions
        gen_frame: (26, 3) Generated joint positions
        output_path: file path to save to
        view_angle: (elevation, azimuth)
        figsize: figure size
        dpi: resolution
        use_stage_colors: whether to apply per-stage colors to the generated side.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Calculate combined bounds for consistent view
    combined = np.stack([gt_frame, gen_frame], axis=0)
    bounds = calculate_bounds(combined)
    
    # GT on LEFT
    ax1 = fig.add_subplot(121, projection='3d')
    setup_axes(ax1, bounds, title='Ground Truth', view_angle=view_angle)
    draw_skeleton(ax1, gt_frame, use_stage_colors=False)
    
    # Generated on RIGHT
    ax2 = fig.add_subplot(122, projection='3d')
    gen_title = 'Generated (Stage Colors)' if use_stage_colors else 'Generated'
    setup_axes(ax2, bounds, title=gen_title, view_angle=view_angle)
    draw_skeleton(ax2, gen_frame, use_stage_colors=use_stage_colors)
    
    if use_stage_colors:
        fig.suptitle('GT vs Generated\n'
                     'Hand (Stage 2) | Backbone (Stage 3-1) | Elbow (Stage 3-2) | Lower (Stage 3-3)',
                     fontsize=12, fontweight='bold')
    else:
        fig.suptitle('GT vs Generated', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Save GT motion frame(s) as image(s).'
    )
    
    # Sample selection
    parser.add_argument('--sample_id', type=str, default=None,
                        help='Sample path (e.g., yp/ba1/yp07/t1)')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Sample index in split')
    parser.add_argument('--split', type=str, default='test',
                        help='Split name (default: test)')
    parser.add_argument('--data_root', type=str, default='MOSA_dataset',
                        help='Dataset root path')
    
    # Frame selection
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='Frame index to render (default: 0)')
    parser.add_argument('--time', type=float, default=None,
                        help='Time in seconds (overrides frame_idx)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: experiments/vis/outputs/images/)')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Output filename (without extension)')
    
    # Rendering options
    parser.add_argument('--multi_view', action='store_true',
                        help='Render multiple views (front, side, top)')
    parser.add_argument('--stage_colors', action='store_true', default=True,
                        help='Use stage-based colors (Pianoid pipeline) [default]')
    parser.add_argument('--no_stage_colors', action='store_false', dest='stage_colors',
                        help='Use ParCo colors instead of stage colors')
    parser.add_argument('--view_elev', type=float, default=20,
                        help='View elevation angle')
    parser.add_argument('--view_azim', type=float, default=60,
                        help='View azimuth angle')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Image DPI (default: 150)')
    parser.add_argument('--figsize', type=str, default='10,10',
                        help='Figure size as "width,height" (default: 10,10)')
    
    args = parser.parse_args()
    
    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # Resolve sample_id
    data_root = Path(args.data_root)
    
    if args.sample_id:
        sample_id = args.sample_id
    elif args.sample_idx is not None:
        sample_id = get_sample_id_from_splits(data_root, args.sample_idx, args.split)
    else:
        # Default: first test sample
        sample_id = get_sample_id_from_splits(data_root, 0, 'test')
    
    print('=' * 70)
    print('GT Motion Image Renderer')
    print('=' * 70)
    print(f'Sample: {sample_id}')
    
    # Load GT motion
    gt_motion = load_ground_truth(data_root, sample_id)
    
    if gt_motion is None:
        print(f'Error: Ground truth not found for {sample_id}')
        print('Run preprocessing first: bash scripts/train/stage1.sh')
        return
    
    print(f'Motion shape: {gt_motion.shape}')
    
    # Resolve frame index
    if args.time is not None:
        frame_idx = int(args.time * FPS)
    else:
        frame_idx = args.frame_idx
    
    # Clamp frame index
    frame_idx = max(0, min(frame_idx, len(gt_motion) - 1))
    time_sec = frame_idx / FPS
    print(f'Frame: {frame_idx} (t={time_sec:.2f}s)')
    print(f'Stage colors: {args.stage_colors}')
    print('=' * 70)
    
    # Get frame
    frame = gt_motion[frame_idx]
    
    # Output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / 'outputs' / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename
    sample_name = sample_id.replace('/', '_')
    if args.output_name:
        output_name = args.output_name
    else:
        # stage_colors is the default, so add a suffix only when it is disabled.
        suffix = '_parco_colors' if not args.stage_colors else ''
        if args.multi_view:
            output_name = f'{sample_name}_frame{frame_idx:04d}_multiview{suffix}'
        else:
            output_name = f'{sample_name}_frame{frame_idx:04d}{suffix}'
    
    output_path = output_dir / f'{output_name}.png'
    
    # Render
    if args.multi_view:
        render_multi_view(
            frame=frame,
            output_path=output_path,
            title=f'{sample_id} (Frame {frame_idx}, t={time_sec:.2f}s)',
            dpi=args.dpi,
            use_stage_colors=args.stage_colors,
        )
    else:
        render_single_image(
            frame=frame,
            output_path=output_path,
            title=f'{sample_id}\nFrame {frame_idx} (t={time_sec:.2f}s)',
            view_angle=(args.view_elev, args.view_azim),
            figsize=figsize,
            dpi=args.dpi,
            use_stage_colors=args.stage_colors,
        )
    
    print()
    print('=' * 70)
    print('Done!')
    print(f'Output: {output_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()
