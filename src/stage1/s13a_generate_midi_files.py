#!/usr/bin/env python3
"""
s13a_generate_midi_files.py: Pre-generate MIDI files from note annotations

Converts note.csv + align_notetime.csv to proper MIDI files for efficient
ARIA embedding extraction.

Output:
    MOSA_dataset/preprocessed/{sample_path}/performance.mid

The MIDI files use real performance time (seconds) from align_notetime.csv,
not score position (beats) from note.csv.

Usage:
    python src/stage1/s13a_generate_midi_files.py
    python src/stage1/s13a_generate_midi_files.py --force
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.dataset import load_notes_csv


def notes_to_midi(notes_df, output_path: Path, bpm: float = 120.0):
    """
    Convert notes DataFrame to MIDI file.
    
    Args:
        notes_df: DataFrame with 'Onset', 'Offset' in seconds, 'MIDI number'
        output_path: Path to save MIDI file
        bpm: Beats per minute (default 120)
    
    Returns:
        True if successful, False otherwise
    """
    import mido
    
    if len(notes_df) == 0:
        return False
    
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    
    # Set tempo (microseconds per beat)
    tempo = int(60_000_000 / bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    
    # Time conversion: seconds to ticks
    ticks_per_beat = 480
    ticks_per_sec = ticks_per_beat * (bpm / 60.0)
    
    # Convert notes to MIDI events
    events = []
    for _, row in notes_df.iterrows():
        onset_sec = row['Onset']
        offset_sec = row['Offset']
        
        # Skip invalid times
        if np.isnan(onset_sec) or np.isnan(offset_sec):
            continue
        if onset_sec < 0:
            onset_sec = 0
        if offset_sec <= onset_sec:
            continue
        
        onset_tick = int(onset_sec * ticks_per_sec)
        offset_tick = int(offset_sec * ticks_per_sec)
        note = int(row['MIDI number'])
        velocity = int(row.get('Velocity', 80))
        
        events.append((onset_tick, 'note_on', note, velocity))
        events.append((offset_tick, 'note_off', note, 0))
    
    if len(events) == 0:
        return False
    
    # Sort by time, then by event type (note_off before note_on at same time)
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'note_off' else 1))
    
    # Add events to track with delta time
    prev_tick = 0
    for tick, msg_type, note, velocity in events:
        delta = max(0, tick - prev_tick)
        if msg_type == 'note_on':
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta))
        else:
            track.append(mido.Message('note_off', note=note, velocity=0, time=delta))
        prev_tick = tick
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(output_path))
    return True


def process_sample(sample: dict, data_root: Path, output_dir: Path, force: bool = False) -> dict:
    """Process a single sample."""
    piece = sample['piece']
    performer = sample['performer']
    trial = sample['trial']
    sample_path = sample['path']
    
    # Output path
    output_path = output_dir / sample_path / 'performance.mid'
    
    if output_path.exists() and not force:
        return {'status': 'skip', 'path': sample_path}
    
    # Find notes file
    notes_path = data_root / sample_path / 'annotation' / 'annotations' / f"{piece}_{performer}_{trial}_note.csv"
    
    if not notes_path.exists():
        return {'status': 'error', 'path': sample_path, 'reason': 'notes not found'}
    
    try:
        # Load notes with real performance time
        notes_df = load_notes_csv(notes_path, warn_on_missing=False)
        
        # Check if we have real time (should be > 1.0 for first onset typically)
        first_onset = notes_df['Onset'].min()
        if first_onset < 0.5:
            # Might still be in beat units, check align_notetime.csv
            align_path = notes_path.parent / notes_path.name.replace('_note.csv', '_align_notetime.csv')
            if not align_path.exists():
                return {'status': 'error', 'path': sample_path, 'reason': 'align_notetime.csv not found'}
        
        # Generate MIDI
        success = notes_to_midi(notes_df, output_path)
        
        if success:
            duration = notes_df['Offset'].max()
            num_notes = len(notes_df)
            return {
                'status': 'success', 
                'path': sample_path, 
                'num_notes': num_notes,
                'duration_sec': round(duration, 2)
            }
        else:
            return {'status': 'error', 'path': sample_path, 'reason': 'no valid notes'}
    
    except Exception as e:
        return {'status': 'error', 'path': sample_path, 'reason': str(e)}


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial
    
    parser = argparse.ArgumentParser(description='s13a: Generate MIDI files from note annotations')
    parser.add_argument('--data_root', type=str, default='MOSA_dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: data_root/preprocessed)')
    parser.add_argument('--split', type=str, default='all',
                       choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers (default: half of CPU cores)')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / 'preprocessed'
    num_workers = args.num_workers or max(1, os.cpu_count() // 2)
    
    print("=" * 60)
    print("s13a: Generate MIDI Files from Note Annotations")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Split: {args.split}")
    print(f"Workers: {num_workers}")
    print()
    
    # Load dataset info
    splits_json = data_root / 'splits.json'
    if not splits_json.exists():
        print(f"Error: {splits_json} not found")
        return
    
    with open(splits_json) as f:
        splits_data = json.load(f)
    
    # Get samples
    if args.split == 'all':
        samples = splits_data['samples'].get('train', []) + \
                  splits_data['samples'].get('val', []) + \
                  splits_data['samples'].get('test', [])
    else:
        samples = splits_data['samples'].get(args.split, [])
    
    print(f"Processing {len(samples)} samples...")
    print()
    
    # Process in parallel
    stats = {'success': 0, 'skip': 0, 'error': 0}
    total_notes = 0
    
    process_fn = partial(process_sample, data_root=data_root, output_dir=output_dir, force=args.force)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_fn, sample): sample for sample in samples}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating MIDI"):
            result = future.result()
            stats[result['status']] += 1
            
            if result['status'] == 'success':
                total_notes += result.get('num_notes', 0)
            elif result['status'] == 'error':
                tqdm.write(f"  Error: {result['path']} - {result.get('reason', 'unknown')}")
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Success: {stats['success']}")
    print(f"Skipped: {stats['skip']}")
    print(f"Errors:  {stats['error']}")
    print(f"Total notes: {total_notes}")
    print(f"Output: {output_dir}/*/performance.mid")


if __name__ == '__main__':
    main()
