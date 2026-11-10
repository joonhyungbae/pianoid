#!/usr/bin/env python3
"""
s12_precompute_training_data.py: Precompute Training Data for Fast Loading

Converts all training data to efficient .npz format for fast loading during training.
This eliminates CSV parsing overhead and significantly speeds up training.

Output structure:
    MOSA_dataset/training_cache/
    ├── train/
    │   ├── 0000.npz  (joints, active_keys, aria_embed)
    │   ├── 0001.npz
    │   └── ...
    ├── val/
    │   └── ...
    └── metadata.json  (sample info mapping)

Usage:
    python src/stage1/s12_precompute_training_data.py
    python src/stage1/s12_precompute_training_data.py --force
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import FPS, RWRIST_IDX, LWRIST_IDX, RFIN_IDX, LFIN_IDX, PIANO_KEY_MIN
from data.dataset import load_notes_csv


def load_notes_to_arrays(notes_path: Path, num_frames: int, fps: int = FPS,
                         midi_path: Path = None,
                         onset_window_sec: float = 2.0) -> Dict[str, np.ndarray]:
    """
    Convert notes CSV to efficient numpy arrays with velocity, onset, and
    MIDI-derived musical features.

    Returns:
        - active_keys: (T, max_concurrent) - active MIDI keys per frame (0-87)
        - onset_density_per_key: (T, 88) - onset density per active key (0-1 normalized)
        - key_onset: (T, 88) - onset mask (1 at onset frame, 0 otherwise)
        - onset_density: (T,) - frame-level onset density (onsets/sec, normalized)
        - dur_ioi_ratio: (T, 88) - duration/IOI ratio per onset (articulation proxy)
        - pitch_range: (T,) - pitch range of active notes (semitones, normalized)
    """
    df = load_notes_csv(notes_path)
    max_concurrent = 10

    active_keys = np.full((num_frames, max_concurrent), -1, dtype=np.int32)
    onset_density_per_key = np.zeros((num_frames, 88), dtype=np.float32)
    key_onset = np.zeros((num_frames, 88), dtype=np.float32)
    dur_ioi_ratio = np.zeros((num_frames, 88), dtype=np.float32)
    pitch_range = np.zeros(num_frames, dtype=np.float32)

    # Compute onset frames first (needed for density and dur/IOI)
    onset_frames_list = []  # (frame, key_idx, duration_sec)
    for _, row in df.iterrows():
        onset_sec = row['Onset']
        offset_sec = row['Offset']
        midi_num = int(row['MIDI number'])
        key_idx = midi_num - PIANO_KEY_MIN

        if 0 <= key_idx < 88:
            onset_frame = int(onset_sec * fps)
            if 0 <= onset_frame < num_frames:
                key_onset[onset_frame, key_idx] = 1.0
                onset_frames_list.append((onset_frame, key_idx, offset_sec - onset_sec))

    # --- Onset density (onsets/sec in a 2-second window) ---
    raw_onset_count = np.zeros(num_frames, dtype=np.float32)
    for frame, _, _ in onset_frames_list:
        raw_onset_count[frame] += 1.0

    window_frames = int(onset_window_sec * fps)
    kernel = np.ones(window_frames, dtype=np.float32)
    onset_density_raw = np.convolve(raw_onset_count, kernel, mode='same') / onset_window_sec  # onsets/sec
    # Normalize to 0-1 (clip at 20 onsets/sec as practical max)
    onset_density = np.clip(onset_density_raw / 20.0, 0.0, 1.0)

    # --- Duration/IOI ratio per onset (articulation proxy) ---
    # Sort onsets by frame, then compute IOI
    onset_frames_sorted = sorted(onset_frames_list, key=lambda x: x[0])
    for i in range(1, len(onset_frames_sorted)):
        frame, key_idx, duration = onset_frames_sorted[i]
        prev_frame = onset_frames_sorted[i - 1][0]
        ioi_frames = frame - prev_frame
        if ioi_frames > 0:
            ioi_sec = ioi_frames / fps
            ratio = duration / ioi_sec
            # Normalize: clip to [0, 3] then scale to [0, 1]
            dur_ioi_ratio[frame, key_idx] = min(ratio / 3.0, 1.0)

    # --- Fill active keys and velocity (now onset density instead of constant) ---
    for t in range(num_frames):
        time_sec = t / fps
        active = df[(df['Onset'] <= time_sec) & (df['Offset'] > time_sec)]
        midi_nums = active['MIDI number'].values[:max_concurrent]
        active_keys[t, :len(midi_nums)] = midi_nums - PIANO_KEY_MIN

        # Use onset density as velocity proxy for active keys
        for midi_num in active['MIDI number'].values:
            key_idx = int(midi_num) - PIANO_KEY_MIN
            if 0 <= key_idx < 88:
                onset_density_per_key[t, key_idx] = onset_density[t]

        # --- Pitch range ---
        if len(midi_nums) > 0:
            # Normalize: 88 semitones max range -> [0, 1]
            pitch_range[t] = (midi_nums.max() - midi_nums.min()) / 88.0

    return {
        'active_keys': active_keys,
        'onset_density_per_key': onset_density_per_key,
        'key_onset': key_onset,
        'onset_density': onset_density,
        'dur_ioi_ratio': dur_ioi_ratio,
        'pitch_range': pitch_range,
    }


def precompute_sample(
    sample: Dict,
    data_root: Path,
    output_dir: Path,
    idx: int,
    onset_window_sec: float = 2.0,
) -> Dict:
    """Precompute a single sample to .npz format"""
    piece = sample['piece']
    performer = sample['performer']
    trial = sample['trial']
    
    # Load preprocessed motion
    prep_path = data_root / 'preprocessed' / sample['path'] / 'motion_26j.npy'
    if not prep_path.exists():
        return {'status': 'error', 'reason': 'preprocessed motion not found'}
    
    joints = np.load(prep_path).astype(np.float32)
    num_frames = len(joints)
    
    # Load notes
    sample_path = data_root / sample['path']
    notes_path = sample_path / 'annotation' / 'annotations' / f"{piece}_{performer}_{trial}_note.csv"
    if not notes_path.exists():
        return {'status': 'error', 'reason': 'notes not found'}
    
    # MIDI path for velocity extraction
    midi_path = data_root / 'preprocessed' / sample['path'] / 'performance.mid'
    
    notes_data = load_notes_to_arrays(notes_path, num_frames, midi_path=midi_path,
                                         onset_window_sec=onset_window_sec)
    
    # Load ARIA embedding (OPTIONAL — only needed to reproduce the
    # Table 3(D) Stage-4 ARIA ablation row; the main pipeline does not use it).
    aria_temporal_path = data_root / 'preprocessed' / sample['path'] / 'aria_embedding_temporal.npy'
    aria_global_path = data_root / 'preprocessed' / sample['path'] / 'aria_embedding.npy'

    if aria_temporal_path.exists():
        aria_embed = np.load(aria_temporal_path).astype(np.float32)  # (num_windows, 512)
    elif aria_global_path.exists():
        aria_embed = np.load(aria_global_path).astype(np.float32)    # (512,)
    else:
        # Fallback placeholder; downstream code only consumes it when use_aria=True.
        aria_embed = np.zeros((1, 512), dtype=np.float32)
    
    # Ensure lengths match
    for key in ['active_keys', 'onset_density_per_key', 'key_onset', 'dur_ioi_ratio']:
        notes_data[key] = notes_data[key][:num_frames]
    notes_data['onset_density'] = notes_data['onset_density'][:num_frames]
    notes_data['pitch_range'] = notes_data['pitch_range'][:num_frames]

    # Save as npz
    output_path = output_dir / f'{idx:05d}.npz'
    np.savez_compressed(
        output_path,
        joints=joints,                              # (T, 26, 3)
        active_keys=notes_data['active_keys'],      # (T, 10)
        onset_density_per_key=notes_data['onset_density_per_key'],    # (T, 88) - onset density per key
        key_onset=notes_data['key_onset'],          # (T, 88) - onset mask
        onset_density=notes_data['onset_density'],  # (T,) - frame-level onset density
        dur_ioi_ratio=notes_data['dur_ioi_ratio'],  # (T, 88) - articulation proxy
        pitch_range=notes_data['pitch_range'],      # (T,) - normalized pitch range
        aria_embed=aria_embed,                      # (num_windows, 512) or (512,)
    )
    
    return {
        'status': 'success',
        'idx': idx,
        'path': str(output_path),
        'num_frames': num_frames,
        'sample_info': sample
    }


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial
    
    parser = argparse.ArgumentParser(description='s12: Precompute Training Data')
    parser.add_argument('--data_root', type=str, default='MOSA_dataset')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers (default: half of CPU cores)')
    parser.add_argument('--onset_window', type=float, default=2.0,
                       help='Onset density window in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / 'training_cache'
    num_workers = args.num_workers or max(1, os.cpu_count() // 2)
    
    print("=" * 70)
    print("s12: Precompute Training Data")
    print("=" * 70)
    print(f"Onset density window: {args.onset_window}s")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Workers: {num_workers}")
    
    # Check if already exists
    metadata_path = output_dir / 'metadata.json'
    if metadata_path.exists() and not args.force:
        print(f"\nCache already exists at {output_dir}")
        print("Use --force to regenerate")
        return
    
    # Load splits
    splits_path = data_root / 'splits.json'
    with open(splits_path) as f:
        splits_data = json.load(f)
    
    metadata = {
        'data_root': str(data_root),
        'fps': FPS,
        'joint_indices': {
            'rwrist': RWRIST_IDX,
            'lwrist': LWRIST_IDX,
            'rfin': RFIN_IDX,
            'lfin': LFIN_IDX,
        },
        'splits': {}
    }
    
    # Process each split
    for split in ['train', 'val', 'test']:
        samples = splits_data['samples'].get(split, [])
        print(f"\nProcessing {split}: {len(samples)} samples")
        
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tasks with indices
        tasks = [(sample, data_root, split_dir, idx, args.onset_window) for idx, sample in enumerate(samples)]
        
        split_metadata = []
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(precompute_sample, *task): task[3] for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=split):
                idx = futures[future]
                sample = samples[idx]
                result = future.result()
                
                if result['status'] == 'success':
                    split_metadata.append({
                        'idx': idx,
                        'num_frames': result['num_frames'],
                        'piece': sample['piece'],
                        'performer': sample['performer'],
                        'trial': sample['trial'],
                        'henle_difficulty': sample.get('henle_difficulty', 5),
                    })
                else:
                    tqdm.write(f"  Error: {sample['path']}: {result.get('reason', 'unknown')}")
        
        # Sort by index
        split_metadata.sort(key=lambda x: x['idx'])
        
        metadata['splits'][split] = {
            'num_samples': len(split_metadata),
            'total_frames': sum(s['num_frames'] for s in split_metadata),
            'samples': split_metadata,
        }
        
        print(f"  ✓ {len(split_metadata)} samples, {metadata['splits'][split]['total_frames']:,} frames")
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved: {metadata_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Precomputation Complete!")
    print("=" * 70)
    for split, info in metadata['splits'].items():
        print(f"  {split}: {info['num_samples']} samples, {info['total_frames']:,} frames")
    
    print(f"\nNext step:")
    print(f"  bash scripts/download_checkpoints.sh && bash inference.sh   # reproduce paper")
    print(f"  bash scripts/train/stage1.sh                                # or train Stage 1 from scratch")


if __name__ == '__main__':
    main()
