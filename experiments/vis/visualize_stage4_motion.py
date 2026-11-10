#!/usr/bin/env python3
"""
visualize_stage4_motion.py: visualize only motions generated from musical features.

Renders only the output of scripts/train/stage4.sh.
- Input:  results/hierarchical_mdm/sample_XXX.npz (Hand/Backbone/Lower MDM trained on MIDI+ARIA).
- Output: 3D skeleton video (generated motion only, or GT vs generated comparison).

Other sources (data_root/generated/, gt_only, batches, ...) are not used.

Usage:
    # No args -> all samples, GT vs generated comparison.
    python experiments/vis/visualize_stage4_motion.py

    # Single sample.
    python experiments/vis/visualize_stage4_motion.py --sample_idx 1

    # Disable comparison.
    python experiments/vis/visualize_stage4_motion.py --no_compare_gt
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

import numpy as np

# Project root and normal import so multiprocessing Pool can pickle experiments.vis.create_video functions
proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))
sys.path.insert(0, str(proj_root / 'src'))
from data.constants import FPS
from experiments.vis.create_video import (
    PianoMotionVideoCreator,
    load_ground_truth,
    load_hierarchical_mdm_result,
    get_sample_id_from_splits,
    add_audio_to_video,
)


def visualize_single_sample(args_tuple) -> dict:
    """
    Visualize a single sample. Designed for parallel execution.
    args_tuple: (sample_idx, hierarchical_mdm_dir, data_root, split, output_base,
                 creator_config, compare_gt, with_audio, max_frames, use_stage_colors)
    Returns {'status': 'success'|'skip'|'error', ...}
    """
    (sample_idx, hierarchical_mdm_dir, data_root, split, output_base,
     creator_config, compare_gt, with_audio, max_frames, use_stage_colors) = args_tuple
    
    # Convert back to Path
    hierarchical_mdm_dir = Path(hierarchical_mdm_dir)
    data_root = Path(data_root)
    output_base = Path(output_base)
    
    try:
        # Check if npz exists
        npz_path = hierarchical_mdm_dir / f"sample_{sample_idx:03d}.npz"
        if not npz_path.exists():
            return {'status': 'skip', 'sample_idx': sample_idx, 'reason': 'npz not found'}
        
        # Get sample_id
        sample_id = get_sample_id_from_splits(data_root, sample_idx, split)
        
        # Load generated motion
        generated = load_hierarchical_mdm_result(hierarchical_mdm_dir, sample_idx)
        if max_frames:
            generated = generated[:max_frames]
        
        # Output directory
        output_dir = output_base / sample_id.replace('/', '_')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create creator inside worker (avoids pickle issues)
        creator = PianoMotionVideoCreator(**creator_config)
        
        if compare_gt:
            ground_truth = load_ground_truth(data_root, sample_id)
            if ground_truth is not None:
                T = min(len(ground_truth), len(generated))
                ground_truth = ground_truth[:T]
                generated = generated[:T]
                # Determine output filename based on stage colors
                # stage_colors is the default, so use the default filename
                out_path = output_dir / 'comparison_gt_vs_generated.mp4'
                creator.create_comparison_video(
                    ground_truth=ground_truth,
                    generated=generated,
                    output_path=out_path,
                    use_stage_colors=use_stage_colors,
                )
            else:
                out_path = output_dir / 'motion_generated.mp4'
                creator.create_video(
                    joints=generated,
                    output_path=out_path,
                    title=f'Generated - {sample_id}',
                )
        else:
            out_path = output_dir / 'motion_generated.mp4'
            creator.create_video(
                joints=generated,
                output_path=out_path,
                title=f'Generated - {sample_id}',
            )
        
        if with_audio:
            parts = sample_id.split('/')
            if len(parts) >= 4:
                piece, performer, trial = parts[1], parts[2], parts[3]
                audio_path = data_root / sample_id / f'{piece}_{performer}_{trial}_audio.wav'
                if audio_path.exists():
                    add_audio_to_video(
                        video_path=out_path,
                        audio_path=audio_path,
                        output_path=output_dir / (out_path.stem + '_with_audio.mp4'),
                    )
        
        return {'status': 'success', 'sample_idx': sample_idx, 'sample_id': sample_id, 'path': str(out_path)}
    
    except Exception as e:
        return {'status': 'error', 'sample_idx': sample_idx, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Visualize only Stage 4 musical-feature motion (output of scripts/train/stage4.sh)'
    )
    parser.add_argument('--hierarchical_mdm_dir', type=str, default=None,
                        help='Stage 4 output directory (default: <project_root>/results/hierarchical_mdm)')
    parser.add_argument('--sample_idx', type=int, default=-1,
                        help='Sample index (-1 = process all samples in parallel; default: -1)')
    parser.add_argument('--data_root', type=str, default='MOSA_dataset',
                        help='GT / split path (used when compare_gt or with_audio is set)')
    parser.add_argument('--split', type=str, default='test',
                        help='Split used for sample_idx mapping (default: test)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Video output directory (default: experiments/vis/outputs/)')
    parser.add_argument('--compare_gt', action='store_true', default=True,
                        help='GT vs generated comparison video (on by default)')
    parser.add_argument('--no_compare_gt', action='store_false', dest='compare_gt',
                        help='Generated motion only, no comparison')
    parser.add_argument('--stage_colors', action='store_true', default=True,
                        help='Color generated motion by stage (Hand / Backbone / Elbow / Lower) [default]')
    parser.add_argument('--no_stage_colors', action='store_false', dest='stage_colors',
                        help='Disable per-stage colors (use ParCo colors instead)')
    parser.add_argument('--with_audio', action='store_true', help='Attach audio to the video')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--video_fps', type=int, default=30)
    parser.add_argument('--render_fps', type=int, default=30)
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--view_elev', type=float, default=20)
    parser.add_argument('--view_azim', type=float, default=60)
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: half of available CPU cores)')
    args = parser.parse_args()

    proj_root = Path(__file__).parent.parent
    data_root = Path(args.data_root)
    hierarchical_mdm_dir = Path(args.hierarchical_mdm_dir) if args.hierarchical_mdm_dir else proj_root / 'results' / 'hierarchical_mdm'
    output_base = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'outputs'
    
    # Handle duration -> max_frames
    max_frames = args.max_frames
    if args.duration and FPS:
        max_frames = int(args.duration * FPS)
    
    # Number of parallel workers (default: half of CPU cores)
    num_workers = args.workers if args.workers else max(1, os.cpu_count() // 2)
    
    # Creator config (passed to workers, created inside each worker)
    creator_config = {
        'video_fps': args.video_fps,
        'render_fps': args.render_fps,
        'dpi': args.dpi,
        'view_angle': (args.view_elev, args.view_azim),
        'n_workers': 1,  # Each sample uses 1 worker internally
    }
    
    # Find all sample npz files
    all_npz = sorted(hierarchical_mdm_dir.glob('sample_*.npz'))
    if not all_npz:
        print(f'No sample_*.npz found in {hierarchical_mdm_dir}')
        print('Run stage4 first: bash scripts/train/stage4.sh')
        return
    
    # Determine which samples to process
    if args.sample_idx == -1:
        # All samples mode
        sample_indices = [int(p.stem.split('_')[1]) for p in all_npz]
    else:
        # Single sample mode (default: sample 0)
        sample_indices = [args.sample_idx]
    
    print('=' * 70)
    print('Stage 4 musical-feature motion visualization')
    print('=' * 70)
    print(f'Source: {hierarchical_mdm_dir}')
    print(f'Samples: {len(sample_indices)} ({sample_indices[0]} ~ {sample_indices[-1]})')
    print(f'Compare GT: {args.compare_gt}')
    print(f'Stage Colors: {args.stage_colors}')
    if args.stage_colors:
        print('  Hand (Stage 2) | Backbone (Stage 3-1) | Elbow (Stage 3-2) | Lower (Stage 3-3)')
    print(f'Workers: {num_workers} (parallel)')
    print(f'Output: {output_base}')
    print('=' * 70)
    
    # Prepare task arguments
    tasks = [
        (
            sample_idx,
            str(hierarchical_mdm_dir),  # Convert to str for pickle
            str(data_root),
            args.split,
            str(output_base),
            creator_config,
            args.compare_gt,
            args.with_audio,
            max_frames,
            args.stage_colors,
        )
        for sample_idx in sample_indices
    ]
    
    # Process samples in parallel
    results = {'success': 0, 'skip': 0, 'error': 0}
    
    if len(sample_indices) == 1:
        # Single sample: run directly (verbose)
        result = visualize_single_sample(tasks[0])
        results[result['status']] += 1
        if result['status'] == 'success':
            print(f"\nSaved: {result['path']}")
        elif result['status'] == 'error':
            print(f"\nError: {result.get('error', 'unknown')}")
    else:
        # Multiple samples: parallel execution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(visualize_single_sample, task): task[0] for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc='Visualizing'):
                result = future.result()
                results[result['status']] += 1
                if result['status'] == 'error':
                    tqdm.write(f"  Error [{result['sample_idx']}]: {result.get('error', 'unknown')}")
    
    print()
    print('=' * 70)
    print('Done')
    print(f"Success: {results['success']}, Skip: {results['skip']}, Error: {results['error']}")
    print(f'Output: {output_base}')
    print('=' * 70)


if __name__ == '__main__':
    main()
