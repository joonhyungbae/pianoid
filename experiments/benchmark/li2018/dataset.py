"""
experiments/benchmark/li2018/dataset.py

Dataset for Li2018 model with beat/metric annotations.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.constants import FPS


class Li2018Dataset(Dataset):
    """
    Dataset for Li2018 model that includes metric structure.
    
    Loads from preprocessed cache + extracts beat information from annotations.
    
    Metric matrix encoding (3 channels):
    - Channel 0: Downbeat (beat 0 in measure)
    - Channel 1: Strong beat (beat 2 in 4/4, or pickup)
    - Channel 2: Other beats
    """
    
    def __init__(
        self,
        cache_dir: str = 'MOSA_dataset/training_cache',
        annotation_dir: str = 'MOSA-Music-mOtion-and-Semantic-Annotation-dataset/sample_data',
        split: str = 'train',
        window_size: int = 480,
        stride: Optional[int] = None,
        fps: int = FPS,
        beats_per_measure: int = 4,
    ):
        self.cache_dir = Path(cache_dir)
        self.annotation_dir = Path(annotation_dir)
        self.split = split
        self.window_size = window_size
        self.stride = stride or window_size // 2
        self.fps = fps
        self.beats_per_measure = beats_per_measure
        
        # Load npz files
        split_dir = self.cache_dir / split
        self.npz_files = sorted(split_dir.glob('*.npz'))
        
        if not self.npz_files:
            raise ValueError(f"No npz files found in {split_dir}")
        
        # Load metadata to map npz files to annotations
        metadata_path = self.cache_dir / 'metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Build windows
        self.windows = self._build_windows()
        print(f"Li2018Dataset: {len(self.windows)} windows from {len(self.npz_files)} sequences ({split})")
    
    def _build_windows(self) -> List[Tuple[int, int, int]]:
        """Build (file_idx, start_frame, end_frame) tuples."""
        windows = []
        
        for file_idx, npz_file in enumerate(self.npz_files):
            data = np.load(npz_file)
            T = len(data['joints'])
            
            if self.split == 'train':
                # Sliding windows for training
                for start in range(0, T - self.window_size + 1, self.stride):
                    windows.append((file_idx, start, start + self.window_size))
            else:
                # Full sequence for val/test (or single window if too long)
                if T <= self.window_size * 2:
                    windows.append((file_idx, 0, T))
                else:
                    # Multiple non-overlapping windows
                    for start in range(0, T - self.window_size + 1, self.window_size):
                        windows.append((file_idx, start, start + self.window_size))
        
        return windows
    
    def _get_annotation_path(self, npz_file: Path) -> Optional[Path]:
        """Find annotation path for a given npz file."""
        # Extract sample name from npz filename or metadata
        sample_name = npz_file.stem  # e.g., "00004"
        
        # Try to find in metadata
        if 'file_mapping' in self.metadata:
            mapping = self.metadata.get('file_mapping', {})
            if sample_name in mapping:
                rel_path = mapping[sample_name].get('annotation_dir', '')
                if rel_path:
                    return self.annotation_dir / rel_path
        
        # Fallback: search for matching annotation
        # This is slower but works without metadata
        return None
    
    def _load_metric_from_notes(self, note_df: pd.DataFrame, num_frames: int) -> np.ndarray:
        """
        Compute metric matrix from note annotations.
        
        Args:
            note_df: DataFrame with 'Onset' (beat units), 'Measure', 'Beat in Measure'
            num_frames: Number of output frames
        
        Returns:
            metric_matrix: [num_frames, 3] - downbeat, strong beat, other
        """
        metric = np.zeros((num_frames, 3), dtype=np.float32)
        
        if note_df is None or len(note_df) == 0:
            return metric
        
        # Check if required columns exist
        if 'Beat in Measure' not in note_df.columns:
            return metric
        
        # Get tempo estimation from note timings
        # Assume Onset is in beat units and we need to convert to frames
        if 'Onset' not in note_df.columns:
            return metric
        
        max_beat = note_df['Onset'].max()
        if max_beat <= 0:
            return metric
        
        # Estimate frames per beat
        frames_per_beat = num_frames / (max_beat + 1)
        
        # Mark beat positions
        for _, row in note_df.iterrows():
            onset_beat = row['Onset']
            beat_in_measure = row.get('Beat in Measure', 0)
            
            # Convert to frame
            frame = int(onset_beat * frames_per_beat)
            if frame >= num_frames:
                continue
            
            # Encode metric position
            if beat_in_measure == 0:
                # Downbeat
                metric[frame, 0] = 1.0
            elif beat_in_measure == 2 and self.beats_per_measure == 4:
                # Strong beat (beat 2 in 4/4)
                metric[frame, 1] = 1.0
            else:
                # Other beat
                metric[frame, 2] = 1.0
        
        return metric
    
    def _compute_metric_from_onset(self, onset: np.ndarray) -> np.ndarray:
        """
        Fallback: Estimate metric structure from onset patterns.
        Uses onset detection to guess beat positions.
        
        Args:
            onset: [T, 88] onset matrix
        
        Returns:
            metric_matrix: [T, 3]
        """
        T = len(onset)
        metric = np.zeros((T, 3), dtype=np.float32)
        
        # Sum onsets across keys
        onset_sum = onset.sum(axis=1)  # [T]
        
        # Find strong onsets (potential beats)
        threshold = np.percentile(onset_sum[onset_sum > 0], 50) if onset_sum.max() > 0 else 0
        beat_frames = np.where(onset_sum > threshold)[0]
        
        if len(beat_frames) < 2:
            return metric
        
        # Estimate beat period
        beat_diffs = np.diff(beat_frames)
        if len(beat_diffs) > 0:
            median_period = np.median(beat_diffs)
            
            # Mark beats
            for i, frame in enumerate(beat_frames):
                if i % self.beats_per_measure == 0:
                    metric[frame, 0] = 1.0  # Downbeat
                elif i % self.beats_per_measure == 2:
                    metric[frame, 1] = 1.0  # Strong beat
                else:
                    metric[frame, 2] = 1.0  # Other
        
        return metric
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start, end = self.windows[idx]
        npz_file = self.npz_files[file_idx]
        
        # Load data
        data = np.load(npz_file)
        
        joints = data['joints'][start:end]  # [T, 26, 3]
        onset_density_per_key = data['onset_density_per_key'][start:end]  # [T, 88]
        key_onset = data['key_onset'][start:end]  # [T, 88]
        
        # Create piano roll (binary key activation)
        piano_roll = (onset_density_per_key > 0).astype(np.float32)
        
        # Try to load metric from annotation
        annotation_path = self._get_annotation_path(npz_file)
        if annotation_path is not None:
            note_csv = annotation_path / 'annotations' / f'{annotation_path.name}_note.csv'
            if note_csv.exists():
                note_df = pd.read_csv(note_csv)
                metric = self._load_metric_from_notes(note_df, len(joints))
            else:
                metric = self._compute_metric_from_onset(key_onset)
        else:
            # Fallback: estimate from onset
            metric = self._compute_metric_from_onset(key_onset)
        
        # Flatten joints for output
        joints_flat = joints.reshape(len(joints), -1)  # [T, 78]
        
        return {
            'joints': torch.from_numpy(joints_flat).float(),
            'joints_3d': torch.from_numpy(joints).float(),
            'midi_features': {
                'piano_roll': torch.from_numpy(piano_roll).float(),
                'velocity': torch.from_numpy(onset_density_per_key).float(),
                'onset': torch.from_numpy(key_onset).float(),
            },
            'metric_matrix': torch.from_numpy(metric).float(),
            'sample_idx': torch.tensor(file_idx),
            'window_start': torch.tensor(start),
        }


class Li2018FullSequenceDataset(Dataset):
    """Full sequence dataset for evaluation."""
    
    def __init__(
        self,
        cache_dir: str = 'MOSA_dataset/training_cache',
        split: str = 'test',
    ):
        self.cache_dir = Path(cache_dir)
        split_dir = self.cache_dir / split
        self.npz_files = sorted(split_dir.glob('*.npz'))
        
        if not self.npz_files:
            raise ValueError(f"No npz files found in {split_dir}")
        
        print(f"Li2018FullSequenceDataset: {len(self.npz_files)} sequences ({split})")
    
    def __len__(self) -> int:
        return len(self.npz_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        npz_file = self.npz_files[idx]
        data = np.load(npz_file)
        
        joints = data['joints']  # [T, 26, 3]
        onset_density_per_key = data['onset_density_per_key']  # [T, 88]
        key_onset = data['key_onset']  # [T, 88]
        
        piano_roll = (onset_density_per_key > 0).astype(np.float32)
        joints_flat = joints.reshape(len(joints), -1)
        
        # Simple metric estimation from onset
        T = len(joints)
        onset_sum = key_onset.sum(axis=1)
        metric = np.zeros((T, 3), dtype=np.float32)
        
        # Mark frames with onsets
        onset_frames = np.where(onset_sum > 0)[0]
        for i, frame in enumerate(onset_frames):
            if i % 4 == 0:
                metric[frame, 0] = 1.0
            elif i % 4 == 2:
                metric[frame, 1] = 1.0
            else:
                metric[frame, 2] = 1.0
        
        return {
            'joints': torch.from_numpy(joints_flat).float(),
            'joints_3d': torch.from_numpy(joints).float(),
            'midi_features': {
                'piano_roll': torch.from_numpy(piano_roll).float(),
                'velocity': torch.from_numpy(onset_density_per_key).float(),
                'onset': torch.from_numpy(key_onset).float(),
            },
            'metric_matrix': torch.from_numpy(metric).float(),
            'sample_idx': torch.tensor(idx),
        }


def get_li2018_train_loader(
    cache_dir: str = 'MOSA_dataset/training_cache',
    batch_size: int = 32,
    window_size: int = 480,
    num_workers: int = 4,
) -> DataLoader:
    dataset = Li2018Dataset(cache_dir, split='train', window_size=window_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_li2018_val_loader(
    cache_dir: str = 'MOSA_dataset/training_cache',
    batch_size: int = 32,
    window_size: int = 480,
    num_workers: int = 4,
) -> DataLoader:
    dataset = Li2018Dataset(cache_dir, split='val', window_size=window_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_li2018_test_loader(
    cache_dir: str = 'MOSA_dataset/training_cache',
    batch_size: int = 1,
) -> DataLoader:
    dataset = Li2018FullSequenceDataset(cache_dir, split='test')
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Full sequences need sequential loading
    )


if __name__ == '__main__':
    # Test
    print("Testing Li2018Dataset...")
    
    try:
        dataset = Li2018Dataset(
            cache_dir='MOSA_dataset/training_cache',
            split='train',
            window_size=240,
        )
        
        sample = dataset[0]
        print(f"joints: {sample['joints'].shape}")
        print(f"piano_roll: {sample['midi_features']['piano_roll'].shape}")
        print(f"metric_matrix: {sample['metric_matrix'].shape}")
        print(f"Metric sum per channel: {sample['metric_matrix'].sum(dim=0)}")
        
    except Exception as e:
        print(f"Error: {e}")
