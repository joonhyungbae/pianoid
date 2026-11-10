#!/usr/bin/env python3
"""
s13_extract_aria_embedding.py: Extract ARIA MIDI Embeddings

Extracts 512-dimensional MIDI embeddings using Aria model.
Based on tipiano's implementation.

Now extracts TEMPORAL embeddings: one embedding per time window,
so that each training window gets the correct music context.

Usage:
    python src/stage1/s13_extract_aria_embedding.py --data_root ./MOSA_dataset
    python src/stage1/s13_extract_aria_embedding.py --data_root ./MOSA_dataset --force
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add aria package path
ARIA_PATH = str(Path(__file__).parent.parent.parent / 'vendor' / 'aria')
if ARIA_PATH not in sys.path:
    sys.path.insert(0, ARIA_PATH)

# Default settings
EMBEDDING_DIM = 512
DEFAULT_MODEL_CONFIG = 'medium-emb'
DEFAULT_FPS = 60  # Training frame rate (downsampled from 120fps)
DEFAULT_WINDOW_SEC = 8.0  # 8 seconds = 480 frames at 60fps


def check_aria_installed():
    """Check if aria package is installed"""
    try:
        from aria.model import TransformerEMB, ModelConfig
        from aria.config import load_model_config
        from aria.embedding import get_global_embedding_from_midi
        from ariautils.tokenizer import AbsTokenizer
        return True
    except ImportError as e:
        print(f"Error: aria package is not installed: {e}")
        print("\nInstallation:")
        print("  cd vendor/aria")
        print('  pip install -e ".[all]"')
        return False


def load_aria_model(checkpoint_path: str, model_config_name: str = DEFAULT_MODEL_CONFIG, device: str = "cuda"):
    """Load Aria embedding model."""
    from aria.model import TransformerEMB, ModelConfig
    from aria.config import load_model_config
    from ariautils.tokenizer import AbsTokenizer
    from safetensors.torch import load_file
    
    print(f"Loading Aria model: {model_config_name}")
    print(f"Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("\nDownload:")
        print("  mkdir -p models")
        print("  wget https://huggingface.co/loubb/aria-medium-embedding/resolve/main/model.safetensors \\")
        print("       -O models/aria-medium-embedding.safetensors")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    config_dict = load_model_config(name=model_config_name)
    model_config = ModelConfig(**config_dict)
    model_config.set_vocab_size(AbsTokenizer().vocab_size)
    
    model = TransformerEMB(model_config)
    state_dict = load_file(filename=checkpoint_path)
    model.load_state_dict(state_dict=state_dict, strict=True)
    model.eval()
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)
        print("Model loaded on CUDA")
    else:
        device = "cpu"
        model = model.to(device)
        print("Model loaded on CPU")
    
    return model, device


def notes_to_midi_file(notes_df, temp_path: str, bpm: float = 120.0):
    """Convert notes DataFrame to a temporary MIDI file.
    
    Args:
        notes_df: DataFrame with 'Onset', 'Offset' in seconds, 'MIDI number'
        temp_path: Path to save temporary MIDI file
        bpm: Beats per minute (default 120)
    """
    import mido
    
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    
    # Time conversion: seconds to ticks
    # 480 ticks per beat, bpm beats per minute
    ticks_per_beat = 480
    ticks_per_sec = ticks_per_beat * (bpm / 60.0)  # 960 ticks/sec at 120 BPM
    
    # Convert notes to MIDI events
    events = []
    for _, row in notes_df.iterrows():
        onset_tick = int(row['Onset'] * ticks_per_sec)  # seconds to ticks
        offset_tick = int(row['Offset'] * ticks_per_sec)
        note = int(row['MIDI number'])
        velocity = int(row.get('Velocity', 80))
        
        events.append((onset_tick, 'note_on', note, velocity))
        events.append((offset_tick, 'note_off', note, 0))
    
    # Sort by time
    events.sort(key=lambda x: x[0])
    
    # Add events to track
    prev_tick = 0
    for tick, msg_type, note, velocity in events:
        delta = tick - prev_tick
        if msg_type == 'note_on':
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta))
        else:
            track.append(mido.Message('note_off', note=note, velocity=0, time=delta))
        prev_tick = tick
    
    midi.save(temp_path)
    return len(events) > 0


def extract_embedding_from_notes(model, notes_df, device: str = "cuda") -> np.ndarray:
    """
    Extract embedding from notes DataFrame (single global embedding).
    """
    from aria.embedding import get_global_embedding_from_midi
    import tempfile
    
    if len(notes_df) == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        temp_path = f.name
    
    try:
        has_notes = notes_to_midi_file(notes_df, temp_path)
        if not has_notes:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        embedding = get_global_embedding_from_midi(
            model=model,
            midi_path=temp_path,
            notes_per_chunk=300,
            device=device
        )
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"  Warning: Embedding extraction failed: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def extract_temporal_embeddings(
    model, 
    notes_df, 
    total_frames: int,
    fps: int = DEFAULT_FPS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = None,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extract temporal ARIA embeddings - one embedding per time window.
    
    Args:
        model: ARIA model
        notes_df: DataFrame with notes (Onset, Offset, MIDI number, Velocity)
        total_frames: Total number of motion frames
        fps: Frames per second (default: 120)
        window_sec: Window size in seconds (default: 8.0)
        stride_sec: Stride in seconds (default: window_sec / 2)
        device: Device to run model on
    
    Returns:
        embeddings: (num_windows, 512) array
        
    The window indices align with motion windows:
        - Window 0: frames 0 to window_size
        - Window 1: frames stride to stride + window_size
        - etc.
    """
    from aria.embedding import get_global_embedding_from_midi
    import tempfile
    
    if stride_sec is None:
        stride_sec = window_sec / 2
    
    total_duration = total_frames / fps
    window_frames = int(window_sec * fps)
    stride_frames = int(stride_sec * fps)
    
    # Calculate number of windows (same logic as training dataset)
    num_windows = max(1, (total_frames - window_frames) // stride_frames + 1)
    
    embeddings = []
    
    for i in range(num_windows):
        start_sec = (i * stride_frames) / fps
        end_sec = start_sec + window_sec
        
        # Filter notes in this time window (with some context)
        # Include notes that overlap with the window
        window_notes = notes_df[
            (notes_df['Offset'] > start_sec) & 
            (notes_df['Onset'] < end_sec)
        ].copy()
        
        if len(window_notes) == 0:
            # No notes in this window, use zero embedding
            embeddings.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
            continue
        
        # Shift note times to be relative to window start
        window_notes['Onset'] = (window_notes['Onset'] - start_sec).clip(lower=0)
        window_notes['Offset'] = (window_notes['Offset'] - start_sec).clip(upper=window_sec)
        
        # Extract embedding for this window
        emb = extract_embedding_from_notes(model, window_notes, device)
        embeddings.append(emb)
    
    return np.stack(embeddings, axis=0)  # (num_windows, 512)


def process_samples(
    data_root: Path, 
    checkpoint_path: str, 
    force: bool = False,
    window_sec: float = DEFAULT_WINDOW_SEC,
    fps: int = DEFAULT_FPS,
):
    """Process all samples and extract temporal ARIA embeddings."""
    
    # Load splits
    splits_path = data_root / 'splits.json'
    if not splits_path.exists():
        print(f"Error: splits.json not found at {splits_path}")
        return
    
    with open(splits_path) as f:
        splits_data = json.load(f)
    
    # Load model
    model, device = load_aria_model(checkpoint_path)
    
    # Process all samples
    all_samples = []
    for split in ['train', 'val', 'test']:
        all_samples.extend(splits_data['samples'].get(split, []))
    
    print(f"\nProcessing {len(all_samples)} samples...")
    print(f"Window: {window_sec}s, FPS: {fps}")
    
    success = 0
    skip = 0
    fail = 0
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
    from data.dataset import load_notes_csv
    
    with torch.no_grad():
        for sample in tqdm(all_samples, desc="Extracting temporal ARIA embeddings"):
            sample_path = data_root / sample['path']
            output_path = data_root / 'preprocessed' / sample['path'] / 'aria_embedding_temporal.npy'
            
            # Skip if exists
            if output_path.exists() and not force:
                skip += 1
                continue
            
            # Find notes file
            parts = sample['path'].split('/')
            piece, performer, trial = parts[1], parts[2], parts[3]
            notes_path = sample_path / 'annotation' / 'annotations' / f"{piece}_{performer}_{trial}_note.csv"
            
            if not notes_path.exists():
                fail += 1
                continue
            
            # Get motion length to determine number of windows
            motion_path = data_root / 'preprocessed' / sample['path'] / 'motion_26j.npy'
            if not motion_path.exists():
                motion_path = data_root / 'preprocessed' / sample['path'] / 'full_body_normalized.npy'
            
            if not motion_path.exists():
                fail += 1
                continue
            
            try:
                # Load notes
                notes_df = load_notes_csv(notes_path)
                
                # Get total frames from motion data
                motion = np.load(motion_path)
                total_frames = motion.shape[0]
                
                # Extract temporal embeddings
                embeddings = extract_temporal_embeddings(
                    model=model,
                    notes_df=notes_df,
                    total_frames=total_frames,
                    fps=fps,
                    window_sec=window_sec,
                    stride_sec=window_sec / 2,  # 50% overlap
                    device=device
                )
                
                # Save
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, embeddings)
                success += 1
                
            except Exception as e:
                print(f"  Error processing {sample['path']}: {e}")
                import traceback
                traceback.print_exc()
                fail += 1
    
    print(f"\n{'='*60}")
    print(f"Temporal ARIA Embedding Extraction Complete")
    print(f"{'='*60}")
    print(f"Output: aria_embedding_temporal.npy (num_windows, 512)")
    print(f"Success: {success}")
    print(f"Skipped: {skip}")
    print(f"Failed: {fail}")


def main():
    parser = argparse.ArgumentParser(description='s13: Extract Temporal ARIA MIDI Embeddings')
    parser.add_argument('--data_root', type=str, default='MOSA_dataset')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='models/aria-medium-embedding.safetensors')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    parser.add_argument('--window_sec', type=float, default=DEFAULT_WINDOW_SEC,
                        help=f'Window size in seconds (default: {DEFAULT_WINDOW_SEC})')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS,
                        help=f'Frames per second (default: {DEFAULT_FPS})')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("s13: Extract Temporal ARIA MIDI Embeddings")
    print("=" * 60)
    print(f"Window: {args.window_sec}s, Stride: {args.window_sec/2}s, FPS: {args.fps}")
    
    if not check_aria_installed():
        return
    
    data_root = Path(args.data_root)
    process_samples(
        data_root, 
        args.checkpoint_path, 
        args.force,
        window_sec=args.window_sec,
        fps=args.fps,
    )


if __name__ == '__main__':
    main()
