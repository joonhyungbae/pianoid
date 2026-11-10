"""
experiments/benchmark/common/data_loader.py

Unified data-loader interface.
Reads the existing Pianoid preprocessing output (training_cache).

Cache file layout (*.npz):
- joints: (T, 26, 3) - 26-joint skeleton (meters)
- onset_density_per_key: (T, 88) - MIDI velocity
- key_onset: (T, 88) - onset events
- aria_embed: (N, 512) - ARIA embeddings
- active_keys: (T, 10) - active MIDI keys
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import sys
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.constants import FPS, NUM_JOINTS


class BenchmarkDataset(Dataset):
    """Unified dataset for benchmarking.

    Uses the existing preprocessed cache to feed every baseline the same data.

    Args:
        cache_dir: path to the training_cache directory.
        split: 'train', 'val', or 'test'.
        window_size: window size in frames. None returns the full sequence.
        stride: window stride. Defaults to window_size // 2.
        return_aria: whether to return ARIA embeddings.
        preload: whether to preload into memory (recommended; large speedup).
    """
    
    def __init__(
        self,
        cache_dir: str = 'MOSA_dataset/training_cache',
        split: Literal['train', 'val', 'test'] = 'train',
        window_size: Optional[int] = 480,  # 8 sec at 60fps
        stride: Optional[int] = None,
        return_aria: bool = True,
        preload: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.window_size = window_size
        self.stride = stride or (window_size // 2 if window_size else None)
        self.return_aria = return_aria
        self.preload = preload
        
        # Load split files
        split_dir = self.cache_dir / split
        self.npz_files = sorted(split_dir.glob('*.npz'))
        
        if not self.npz_files:
            raise ValueError(f"No npz files found in {split_dir}")
        
        # Memory cache for preloaded data
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Preload all files into memory
        if self.preload:
            self._preload_data()
        
        # Build index
        self.samples = []
        self._build_index()
        
        print(f"BenchmarkDataset: {split} split, {len(self.samples)} samples")
    
    def _preload_data(self):
        """Preload every npz file into memory."""
        print(f"Preloading {len(self.npz_files)} files into memory...")
        for npz_path in tqdm(self.npz_files, desc=f"Loading {self.split}"):
            data = np.load(npz_path)
            # Store as dict of arrays (avoid keeping npz file handles open)
            self._cache[str(npz_path)] = {
                'joints': data['joints'].copy(),
                'onset_density_per_key': data['onset_density_per_key'].copy(),
                'key_onset': data['key_onset'].copy(),
            }
            if self.return_aria and 'aria_embed' in data:
                self._cache[str(npz_path)]['aria_embed'] = data['aria_embed'].copy()
        print(f"Preloaded {len(self._cache)} files")
    
    def _get_data(self, npz_path: Path) -> Dict[str, np.ndarray]:
        """Get data from the cache (load from disk if missing)."""
        key = str(npz_path)
        if key in self._cache:
            return self._cache[key]
        else:
            # Fallback to loading from disk
            data = np.load(npz_path)
            return {k: data[k] for k in data.files}
    
    def _build_index(self):
        """Build the window index."""
        for npz_path in self.npz_files:
            data = self._get_data(npz_path)
            T = data['joints'].shape[0]

            if self.window_size is None:
                # Full sequence
                self.samples.append({
                    'path': npz_path,
                    'start': 0,
                    'end': T,
                })
            else:
                # Split into windows
                for start in range(0, T - self.window_size + 1, self.stride):
                    self.samples.append({
                        'path': npz_path,
                        'start': start,
                        'end': start + self.window_size,
                    })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        data = self._get_data(sample['path'])
        
        start, end = sample['start'], sample['end']
        
        # Motion
        joints = data['joints'][start:end]  # (T, 26, 3)
        
        # MIDI features
        onset_density_per_key = data['onset_density_per_key'][start:end]  # (T, 88)
        key_onset = data['key_onset'][start:end]        # (T, 88)
        
        # Piano roll (binary from velocity)
        piano_roll = (onset_density_per_key > 0).astype(np.float32)
        
        result = {
            'joints': torch.from_numpy(joints.copy()).float(),
            'midi_features': {
                'piano_roll': torch.from_numpy(piano_roll).float(),
                'velocity': torch.from_numpy(onset_density_per_key / 127.0).float(),
                'onset': torch.from_numpy(key_onset.copy()).float(),
            },
            'sample_idx': int(sample['path'].stem),
        }
        
        # ARIA embedding (if available and requested)
        if self.return_aria and 'aria_embed' in data:
            aria = data['aria_embed']  # (N_chunks, 512)
            # Expand to frame-level by repeating
            T = end - start
            chunk_size = T // len(aria) if len(aria) > 0 else T
            if len(aria) > 0:
                aria_expanded = np.repeat(aria, chunk_size, axis=0)[:T]
                # Pad if needed
                if len(aria_expanded) < T:
                    pad = np.tile(aria[-1:], (T - len(aria_expanded), 1))
                    aria_expanded = np.concatenate([aria_expanded, pad], axis=0)
            else:
                aria_expanded = np.zeros((T, 512), dtype=np.float32)
            
            result['aria_embed'] = torch.from_numpy(aria_expanded).float()
        
        return result


class FullSequenceDataset(Dataset):
    """Dataset that returns full sequences (for evaluation).

    Returns the full sequence without windowing.
    """
    
    def __init__(
        self,
        cache_dir: str = 'MOSA_dataset/training_cache',
        split: Literal['train', 'val', 'test'] = 'test',
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split
        
        split_dir = self.cache_dir / split
        self.npz_files = sorted(split_dir.glob('*.npz'))
        
        if not self.npz_files:
            raise ValueError(f"No npz files found in {split_dir}")
        
        print(f"FullSequenceDataset: {split} split, {len(self.npz_files)} sequences")
    
    def __len__(self) -> int:
        return len(self.npz_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.npz_files[idx])
        
        joints = data['joints']
        onset_density_per_key = data['onset_density_per_key']
        key_onset = data['key_onset']
        piano_roll = (onset_density_per_key > 0).astype(np.float32)
        
        result = {
            'joints': torch.from_numpy(joints).float(),
            'midi_features': {
                'piano_roll': torch.from_numpy(piano_roll).float(),
                'velocity': torch.from_numpy(onset_density_per_key / 127.0).float(),
                'onset': torch.from_numpy(key_onset).float(),
            },
            'sample_idx': int(self.npz_files[idx].stem),
            'path': str(self.npz_files[idx]),
        }
        
        if 'aria_embed' in data:
            result['aria_embed'] = torch.from_numpy(data['aria_embed']).float()
        
        return result


def collate_variable_length(batch: List[Dict]) -> Dict:
    """Variable-length batching: zero-pad up to the longest sequence."""
    max_len = max(item['joints'].shape[0] for item in batch)
    
    batch_joints = []
    batch_piano_roll = []
    batch_velocity = []
    batch_onset = []
    batch_mask = []
    
    for item in batch:
        T = item['joints'].shape[0]
        pad_len = max_len - T
        
        # Pad joints
        joints_padded = torch.nn.functional.pad(
            item['joints'], (0, 0, 0, 0, 0, pad_len)
        )
        batch_joints.append(joints_padded)
        
        # Pad MIDI features
        piano_roll_padded = torch.nn.functional.pad(
            item['midi_features']['piano_roll'], (0, 0, 0, pad_len)
        )
        velocity_padded = torch.nn.functional.pad(
            item['midi_features']['velocity'], (0, 0, 0, pad_len)
        )
        onset_padded = torch.nn.functional.pad(
            item['midi_features']['onset'], (0, 0, 0, pad_len)
        )
        
        batch_piano_roll.append(piano_roll_padded)
        batch_velocity.append(velocity_padded)
        batch_onset.append(onset_padded)
        
        # Mask
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:T] = True
        batch_mask.append(mask)
    
    return {
        'joints': torch.stack(batch_joints),
        'midi_features': {
            'piano_roll': torch.stack(batch_piano_roll),
            'velocity': torch.stack(batch_velocity),
            'onset': torch.stack(batch_onset),
        },
        'mask': torch.stack(batch_mask),
        'sample_idx': [item['sample_idx'] for item in batch],
    }


def get_train_loader(
    cache_dir: str = 'MOSA_dataset/training_cache',
    batch_size: int = 32,
    window_size: int = 480,
    stride: Optional[int] = None,
    num_workers: int = 4,
    preload: bool = True,
    **kwargs
) -> DataLoader:
    """Return the training DataLoader.

    Args:
        cache_dir: path to training_cache.
        batch_size: batch size.
        window_size: window size in frames (default 480 = 8 seconds).
        stride: stride (default window_size // 2).
        num_workers: DataLoader worker count.
        preload: preload into memory (recommended; large speedup).

    Returns:
        DataLoader for training.
    """
    dataset = BenchmarkDataset(
        cache_dir=cache_dir,
        split='train',
        window_size=window_size,
        stride=stride,
        preload=preload,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def get_val_loader(
    cache_dir: str = 'MOSA_dataset/training_cache',
    batch_size: int = 32,
    window_size: int = 480,
    num_workers: int = 4,
    preload: bool = True,
    **kwargs
) -> DataLoader:
    """Return the validation DataLoader."""
    dataset = BenchmarkDataset(
        cache_dir=cache_dir,
        split='val',
        window_size=window_size,
        preload=preload,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def get_test_loader(
    cache_dir: str = 'MOSA_dataset/training_cache',
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    """Return the test DataLoader (full sequences).

    Returns full sequences for evaluation.
    batch_size=1 is recommended (sequences are variable-length).

    Returns:
        DataLoader with 80 test sequences.
    """
    dataset = FullSequenceDataset(
        cache_dir=cache_dir,
        split='test',
    )
    
    if batch_size > 1:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_variable_length,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
        )


# ============================================================
# Utility functions
# ============================================================

def get_split_sizes(cache_dir: str = 'MOSA_dataset/training_cache') -> Dict[str, int]:
    """Return the number of samples in each split."""
    cache_path = Path(cache_dir)
    sizes = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = cache_path / split
        if split_dir.exists():
            sizes[split] = len(list(split_dir.glob('*.npz')))
        else:
            sizes[split] = 0
    
    return sizes


if __name__ == '__main__':
    # Test
    print("Testing data loaders...")
    
    sizes = get_split_sizes()
    print(f"Split sizes: {sizes}")
    
    # Train loader
    train_loader = get_train_loader(batch_size=4, window_size=240)
    batch = next(iter(train_loader))
    print(f"\nTrain batch:")
    print(f"  joints: {batch['joints'].shape}")
    print(f"  piano_roll: {batch['midi_features']['piano_roll'].shape}")
    
    # Test loader
    test_loader = get_test_loader()
    batch = next(iter(test_loader))
    print(f"\nTest batch:")
    print(f"  joints: {batch['joints'].shape}")
    print(f"  sample_idx: {batch['sample_idx']}")
