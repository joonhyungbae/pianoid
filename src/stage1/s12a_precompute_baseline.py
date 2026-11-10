#!/usr/bin/env python3
"""
s12a_precompute_baseline.py: Generate (A) baseline training cache
with constant velocity=0.6 (raw MIDI, no insight-based features).

This produces the same format as s12_precompute_training_data.py but with:
- onset_density_per_key: constant 0.6 for active keys (original behavior)
- onset_density: zeros (placeholder)
- dur_ioi_ratio: zeros (placeholder)
- pitch_range: zeros (placeholder)

Usage:
    python src/stage1/s12a_precompute_baseline.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import FPS, RWRIST_IDX, LWRIST_IDX, RFIN_IDX, LFIN_IDX, PIANO_KEY_MIN
from data.dataset import load_notes_csv


def load_notes_to_arrays_baseline(notes_path: Path, num_frames: int, fps: int = FPS) -> Dict[str, np.ndarray]:
    """Original preprocessing: constant velocity=0.6 for active keys."""
    df = load_notes_csv(notes_path)
    max_concurrent = 10

    active_keys = np.full((num_frames, max_concurrent), -1, dtype=np.int32)
    onset_density_per_key = np.zeros((num_frames, 88), dtype=np.float32)
    key_onset = np.zeros((num_frames, 88), dtype=np.float32)

    for t in range(num_frames):
        time_sec = t / fps
        active = df[(df['Onset'] <= time_sec) & (df['Offset'] > time_sec)]
        midi_nums = active['MIDI number'].values[:max_concurrent]
        active_keys[t, :len(midi_nums)] = midi_nums - PIANO_KEY_MIN
        for midi_num in active['MIDI number'].values:
            key_idx = int(midi_num) - PIANO_KEY_MIN
            if 0 <= key_idx < 88:
                onset_density_per_key[t, key_idx] = 0.6  # constant baseline

    for _, row in df.iterrows():
        onset_sec = row['Onset']
        midi_num = int(row['MIDI number'])
        key_idx = midi_num - PIANO_KEY_MIN
        if 0 <= key_idx < 88:
            onset_frame = int(onset_sec * fps)
            if 0 <= onset_frame < num_frames:
                key_onset[onset_frame, key_idx] = 1.0

    return {
        'active_keys': active_keys,
        'onset_density_per_key': onset_density_per_key,
        'key_onset': key_onset,
        'onset_density': np.zeros(num_frames, dtype=np.float32),
        'dur_ioi_ratio': np.zeros((num_frames, 88), dtype=np.float32),
        'pitch_range': np.zeros(num_frames, dtype=np.float32),
    }


def precompute_sample(sample, data_root, output_dir, idx):
    piece = sample['piece']
    performer = sample['performer']
    trial = sample['trial']

    prep_path = data_root / 'preprocessed' / sample['path'] / 'motion_26j.npy'
    if not prep_path.exists():
        return {'status': 'error', 'reason': 'preprocessed motion not found'}

    joints = np.load(prep_path).astype(np.float32)
    num_frames = len(joints)

    sample_path = data_root / sample['path']
    notes_path = sample_path / 'annotation' / 'annotations' / f"{piece}_{performer}_{trial}_note.csv"
    if not notes_path.exists():
        return {'status': 'error', 'reason': 'notes not found'}

    notes_data = load_notes_to_arrays_baseline(notes_path, num_frames)

    aria_temporal_path = data_root / 'preprocessed' / sample['path'] / 'aria_embedding_temporal.npy'
    aria_global_path = data_root / 'preprocessed' / sample['path'] / 'aria_embedding.npy'
    if aria_temporal_path.exists():
        aria_embed = np.load(aria_temporal_path).astype(np.float32)
    elif aria_global_path.exists():
        aria_embed = np.load(aria_global_path).astype(np.float32)
    else:
        return {'status': 'error', 'reason': 'ARIA embedding not found'}

    for key in ['active_keys', 'onset_density_per_key', 'key_onset', 'dur_ioi_ratio']:
        notes_data[key] = notes_data[key][:num_frames]
    notes_data['onset_density'] = notes_data['onset_density'][:num_frames]
    notes_data['pitch_range'] = notes_data['pitch_range'][:num_frames]

    output_path = output_dir / f'{idx:05d}.npz'
    np.savez_compressed(
        output_path,
        joints=joints,
        active_keys=notes_data['active_keys'],
        onset_density_per_key=notes_data['onset_density_per_key'],
        key_onset=notes_data['key_onset'],
        onset_density=notes_data['onset_density'],
        dur_ioi_ratio=notes_data['dur_ioi_ratio'],
        pitch_range=notes_data['pitch_range'],
        aria_embed=aria_embed,
    )

    return {'status': 'success', 'idx': idx, 'path': str(output_path), 'num_frames': num_frames, 'sample_info': sample}


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description='Generate (A) baseline training cache')
    parser.add_argument('--data_root', type=str, default='MOSA_dataset')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / 'training_cache_A'
    num_workers = args.num_workers or max(1, os.cpu_count() // 2)

    print("=" * 70)
    print("s12a: Generate (A) Baseline Training Cache")
    print(f"  velocity = constant 0.6, no insight features")
    print("=" * 70)

    splits_path = data_root / 'splits.json'
    with open(splits_path) as f:
        splits_data = json.load(f)

    metadata = {
        'data_root': str(data_root), 'fps': FPS,
        'joint_indices': {'rwrist': RWRIST_IDX, 'lwrist': LWRIST_IDX, 'rfin': RFIN_IDX, 'lfin': LFIN_IDX},
        'splits': {}, 'variant': 'A_baseline',
    }

    for split in ['train', 'val', 'test']:
        samples = splits_data['samples'].get(split, [])
        print(f"\nProcessing {split}: {len(samples)} samples")
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        tasks = [(sample, data_root, split_dir, idx) for idx, sample in enumerate(samples)]
        split_metadata = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(precompute_sample, *task): task[3] for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc=split):
                idx = futures[future]
                result = future.result()
                if result['status'] == 'success':
                    split_metadata.append({
                        'idx': idx, 'num_frames': result['num_frames'],
                        'piece': samples[idx]['piece'], 'performer': samples[idx]['performer'],
                        'trial': samples[idx]['trial'],
                        'henle_difficulty': samples[idx].get('henle_difficulty', 5),
                    })

        split_metadata.sort(key=lambda x: x['idx'])
        metadata['splits'][split] = {
            'num_samples': len(split_metadata),
            'total_frames': sum(s['num_frames'] for s in split_metadata),
            'samples': split_metadata,
        }
        print(f"  {len(split_metadata)} samples, {metadata['splits'][split]['total_frames']:,} frames")

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone: {output_dir}")


if __name__ == '__main__':
    main()
