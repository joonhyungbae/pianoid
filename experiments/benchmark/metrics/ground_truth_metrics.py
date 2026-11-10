#!/usr/bin/env python3
"""
ground_truth_metrics.py: Compute Ground Truth reference metrics

This script computes reference metrics from the ground truth data to establish
baselines and upper bounds for evaluation. These values help interpret model
performance in context.

Metrics computed:
    1. FID (train vs test): Distribution gap between train and test sets
       - This is the baseline FID; a perfect generative model would achieve this
    2. Jitter: Natural jitter level in GT pianist motion
    3. Onset Sync: How well GT hand motion correlates with note onsets
    4. Velocity Corr: Correlation between MIDI velocity and hand motion in GT
    5. Inter-performer MPJPE: Natural variation between different performers
       playing the same piece (establishes "natural" variation range)

Usage:
    python evaluation/metrics/ground_truth_metrics.py
    python evaluation/metrics/ground_truth_metrics.py --cache_dir=path/to/cache
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import (
    FPS, NUM_JOINTS,
    RWRIST_IDX, LWRIST_IDX, RHAND_IDX, LHAND_IDX,
)

from src.evaluation import (
    compute_jitter, compute_motion_features, compute_fid,
    compute_mpjpe,
)


def compute_gt_velocity_corr(
    motion: np.ndarray,
    velocity: np.ndarray,
    onset: np.ndarray,
) -> float:
    """Compute correlation between MIDI velocity and hand motion amplitude in GT.
    
    For each note onset, measures the hand motion amplitude in a window around
    the onset and correlates it with the MIDI velocity.
    
    Args:
        motion: (T, J, 3) ground truth motion
        velocity: (T, 88) MIDI velocity per key
        onset: (T, 88) onset mask
    
    Returns:
        corr: Pearson correlation coefficient
    """
    # Get hand Z-axis (vertical motion, most relevant for keypresses)
    hand_z = np.stack([
        motion[:, LHAND_IDX, 2],
        motion[:, RHAND_IDX, 2],
    ], axis=-1)  # (T, 2)
    
    # Hand velocity magnitude
    hand_vel = np.abs(np.diff(hand_z, axis=0))  # (T-1, 2)
    
    # Collect velocity-amplitude pairs at onset frames
    midi_velocities = []
    motion_amplitudes = []
    
    window = 3  # frames around onset
    T = len(hand_vel)
    
    for t in range(1, len(onset) - window):
        frame_onset = onset[t].sum()
        if frame_onset > 0:
            # Get max MIDI velocity for this onset
            midi_vel = velocity[t].max()
            if midi_vel > 0:
                # Get max hand velocity in window around onset
                start = max(0, t - window)
                end = min(T, t + window)
                motion_amp = hand_vel[start:end].max()
                
                midi_velocities.append(midi_vel)
                motion_amplitudes.append(motion_amp)
    
    if len(midi_velocities) < 10:
        return 0.0
    
    corr, _ = stats.pearsonr(midi_velocities, motion_amplitudes)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_inter_performer_mpjpe(
    sequences_by_piece: Dict[str, List[np.ndarray]],
) -> Dict[str, float]:
    """Compute MPJPE between different performers playing the same piece.
    
    This establishes the "natural" variation range - even GT performers differ
    when playing the same piece.
    
    Args:
        sequences_by_piece: Dict mapping piece_id to list of motion sequences
                           from different performers
    
    Returns:
        Dict with mean, std, min, max of inter-performer MPJPE
    """
    all_mpjpes = []
    
    for piece_id, sequences in sequences_by_piece.items():
        if len(sequences) < 2:
            continue
        
        # Compare all pairs of performers for this piece
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                seq1, seq2 = sequences[i], sequences[j]
                
                # Align sequences (take minimum length)
                min_len = min(len(seq1), len(seq2))
                if min_len < 60:  # Skip very short sequences
                    continue
                
                mpjpe = compute_mpjpe(seq1[:min_len], seq2[:min_len])
                all_mpjpes.append(mpjpe)
    
    if not all_mpjpes:
        return {
            'inter_performer_mpjpe_mean': 0.0,
            'inter_performer_mpjpe_std': 0.0,
            'inter_performer_mpjpe_min': 0.0,
            'inter_performer_mpjpe_max': 0.0,
        }
    
    return {
        'inter_performer_mpjpe_mean': float(np.mean(all_mpjpes)),
        'inter_performer_mpjpe_std': float(np.std(all_mpjpes)),
        'inter_performer_mpjpe_min': float(np.min(all_mpjpes)),
        'inter_performer_mpjpe_max': float(np.max(all_mpjpes)),
    }


def load_sequences_with_metadata(cache_dir: Path, split: str = 'train'):
    """Load motion sequences with their metadata.
    
    Returns:
        sequences: List of (motion, onset, velocity, metadata) tuples
    """
    split_dir = cache_dir / split
    if not split_dir.exists():
        return []
    
    sequences = []
    npz_files = sorted(split_dir.glob('*.npz'))
    
    for npz_file in tqdm(npz_files, desc=f"Loading {split} sequences"):
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            # Motion data: 'joints' key (T, J, 3)
            motion = data.get('joints', data.get('motion', None))
            if motion is None:
                print(f"No motion data in {npz_file}")
                continue
            
            if motion.ndim == 2:
                motion = motion.reshape(motion.shape[0], -1, 3)
            
            # Get MIDI data - use 'key_onset' and 'onset_density_per_key' keys
            onset = data.get('key_onset', data.get('onset', np.zeros((len(motion), 88))))
            velocity = data.get('onset_density_per_key', data.get('velocity', np.zeros((len(motion), 88))))
            
            # Ensure correct dtype
            onset = onset.astype(np.float32)
            velocity = velocity.astype(np.float32)
            
            # Metadata
            metadata = {
                'file': str(npz_file),
                'piece_id': str(data.get('piece_id', npz_file.stem)),
                'performer_id': str(data.get('performer_id', '')),
            }
            
            sequences.append((motion, onset, velocity, metadata))
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue
    
    return sequences


def compute_all_gt_metrics(cache_dir: Path) -> Dict[str, float]:
    """Compute all ground truth reference metrics.
    
    Args:
        cache_dir: Path to training cache directory
    
    Returns:
        metrics: Dictionary of all GT reference metrics
    """
    print("Loading train and test sequences...")
    train_sequences = load_sequences_with_metadata(cache_dir, 'train')
    test_sequences = load_sequences_with_metadata(cache_dir, 'test')
    
    if not train_sequences or not test_sequences:
        print("Warning: Could not load sequences")
        return {}
    
    print(f"Loaded {len(train_sequences)} train, {len(test_sequences)} test sequences")
    
    metrics = {}
    
    # 1. FID (train vs test)
    print("\n1. Computing FID (train vs test)...")
    train_features = []
    test_features = []
    
    for motion, _, _, _ in tqdm(train_sequences, desc="Train features"):
        feat = compute_motion_features(motion)
        train_features.append(feat.mean(axis=0))
    
    for motion, _, _, _ in tqdm(test_sequences, desc="Test features"):
        feat = compute_motion_features(motion)
        test_features.append(feat.mean(axis=0))
    
    train_features = np.stack(train_features)
    test_features = np.stack(test_features)
    
    fid_train_test = compute_fid(train_features, test_features)
    metrics['fid_train_test'] = fid_train_test
    print(f"   FID (train vs test): {fid_train_test:.2f}")
    
    # 2. Jitter (GT motion smoothness)
    print("\n2. Computing Jitter...")
    jitters_train = [compute_jitter(m) for m, _, _, _ in tqdm(train_sequences, desc="Train jitter")]
    jitters_test = [compute_jitter(m) for m, _, _, _ in tqdm(test_sequences, desc="Test jitter")]
    
    metrics['jitter_train_mean'] = float(np.mean(jitters_train))
    metrics['jitter_train_std'] = float(np.std(jitters_train))
    metrics['jitter_test_mean'] = float(np.mean(jitters_test))
    metrics['jitter_test_std'] = float(np.std(jitters_test))
    metrics['jitter_combined_mean'] = float(np.mean(jitters_train + jitters_test))
    print(f"   Jitter (train): {metrics['jitter_train_mean']:.4f} ± {metrics['jitter_train_std']:.4f}")
    print(f"   Jitter (test): {metrics['jitter_test_mean']:.4f} ± {metrics['jitter_test_std']:.4f}")
    
    # 3. Velocity Correlation (GT motion vs MIDI velocity)
    print("\n3. Computing Velocity Correlation...")
    velocity_corrs = []
    
    for motion, onset, velocity, _ in tqdm(test_sequences, desc="GT velocity corr"):
        if velocity.sum() > 0:
            vel_corr = compute_gt_velocity_corr(motion, velocity, onset)
            velocity_corrs.append(vel_corr)
    
    metrics['velocity_corr_gt_mean'] = float(np.mean(velocity_corrs)) if velocity_corrs else 0.0
    metrics['velocity_corr_gt_std'] = float(np.std(velocity_corrs)) if velocity_corrs else 0.0
    
    print(f"   Velocity Corr (GT): {metrics['velocity_corr_gt_mean']:.3f} ± {metrics['velocity_corr_gt_std']:.3f}")
    
    # 4. Inter-performer MPJPE
    print("\n4. Computing Inter-performer MPJPE...")
    # Group sequences by piece_id
    sequences_by_piece: Dict[str, List[np.ndarray]] = {}
    for motion, _, _, meta in train_sequences + test_sequences:
        piece_id = meta.get('piece_id', 'unknown')
        if piece_id not in sequences_by_piece:
            sequences_by_piece[piece_id] = []
        sequences_by_piece[piece_id].append(motion)
    
    inter_performer_metrics = compute_inter_performer_mpjpe(sequences_by_piece)
    metrics.update(inter_performer_metrics)
    
    if inter_performer_metrics['inter_performer_mpjpe_mean'] > 0:
        print(f"   Inter-performer MPJPE: {inter_performer_metrics['inter_performer_mpjpe_mean']:.1f} ± "
              f"{inter_performer_metrics['inter_performer_mpjpe_std']:.1f} mm")
        print(f"   Range: [{inter_performer_metrics['inter_performer_mpjpe_min']:.1f}, "
              f"{inter_performer_metrics['inter_performer_mpjpe_max']:.1f}] mm")
    else:
        print("   (Not enough multi-performer data for inter-performer MPJPE)")
    
    # 5. Per-sequence statistics
    print("\n5. Computing per-sequence statistics...")
    metrics['num_train_sequences'] = len(train_sequences)
    metrics['num_test_sequences'] = len(test_sequences)
    
    # Motion statistics
    all_positions = np.concatenate([m.reshape(-1, 3) for m, _, _, _ in train_sequences])
    metrics['motion_mean_x'] = float(all_positions[:, 0].mean())
    metrics['motion_mean_y'] = float(all_positions[:, 1].mean())
    metrics['motion_mean_z'] = float(all_positions[:, 2].mean())
    metrics['motion_std_x'] = float(all_positions[:, 0].std())
    metrics['motion_std_y'] = float(all_positions[:, 1].std())
    metrics['motion_std_z'] = float(all_positions[:, 2].std())
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compute Ground Truth metrics')
    parser.add_argument('--cache_dir', type=str, default='MOSA_dataset/training_cache',
                        help='Path to training cache directory')
    parser.add_argument('--output', type=str, default='evaluation/metrics/gt_metrics.json',
                        help='Output JSON file')
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        print("Please run preprocessing first.")
        return
    
    print("="*60)
    print("Computing Ground Truth Reference Metrics")
    print("="*60)
    
    metrics = compute_all_gt_metrics(cache_dir)
    
    if not metrics:
        print("Failed to compute metrics")
        return
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*60)
    print("Ground Truth Metrics Summary")
    print("="*60)
    print(f"\nFID (train vs test): {metrics['fid_train_test']:.2f}")
    print(f"  - This is the baseline FID; a perfect model would achieve this.")
    print(f"\nJitter (GT): {metrics['jitter_test_mean']:.4f}")
    print(f"  - Natural smoothness level of pianist motion.")
    print(f"\nVelocity Corr (GT): {metrics['velocity_corr_gt_mean']:.3f}")
    print(f"  - Correlation between MIDI velocity and motion amplitude in GT.")
    
    if metrics.get('inter_performer_mpjpe_mean', 0) > 0:
        print(f"\nInter-performer MPJPE: {metrics['inter_performer_mpjpe_mean']:.1f} mm")
        print(f"  - Natural variation between performers playing same piece.")
    
    print(f"\nResults saved to: {output_path}")
    print("\nUse these values as reference in your evaluation tables.")


if __name__ == '__main__':
    main()
