"""
MOSA Piano Dataset Loader

Supports 26-joint simplified skeleton:
- Head (4): RFHD, LFHD, RBHD, LBHD
- Neck (2): C7, CLAV
- Chest (2): T10, STRN
- R Arm (4): RSHO, RELB, RWRIST, RHAND
- L Arm (4): LSHO, LELB, LWRIST, LHAND
- Pelvis (4): RASI, LASI, RPSI, LPSI
- R Leg (3): RKNE, RANKLE, RTOE
- L Leg (3): LKNE, LANKLE, LTOE
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from .constants import (
    MARKER_NAMES, UPPER_BODY_MARKERS, 
    FULL_BODY_JOINTS, NUM_FULL_BODY_JOINTS, JOINT_MAPPING,
    RIGHT_WRIST_MARKERS, LEFT_WRIST_MARKERS, PIANO_KEY_MIN, FPS,
    RWRIST_IDX, LWRIST_IDX, RHAND_IDX, LHAND_IDX
)

# Aliases for compatibility
JOINT_NAMES = FULL_BODY_JOINTS
NUM_JOINTS = NUM_FULL_BODY_JOINTS
RFIN_IDX = RHAND_IDX
LFIN_IDX = LHAND_IDX


@dataclass
class MOSASample:
    """Single MOSA sample data"""
    piece: str
    performer: str
    trial: str
    path: Path
    motion: np.ndarray        # (T, num_joints, 3)
    notes: pd.DataFrame       # Note annotations
    beats: Optional[pd.DataFrame] = None
    downbeats: Optional[pd.DataFrame] = None
    henle_difficulty: int = 5
    duration_sec: float = 0.0
    

def load_motion_csv(filepath: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load motion data from CSV file
    
    Returns:
        motion: (T, num_markers, 3) array
        marker_names: list of marker names
    """
    df = pd.read_csv(filepath)
    
    # Parse marker names from columns
    columns = df.columns.tolist()
    marker_names = []
    seen = set()
    for col in columns:
        # Remove _X, _Y, _Z suffix to get marker name
        marker = col.rsplit('_', 1)[0]
        if marker not in seen:
            marker_names.append(marker)
            seen.add(marker)
    
    num_markers = len(marker_names)
    num_frames = len(df)
    
    # Reshape to (T, num_markers, 3)
    motion = df.values.reshape(num_frames, num_markers, 3)
    
    return motion, marker_names


def extract_full_body_joints(motion: np.ndarray, marker_names: List[str]) -> np.ndarray:
    """
    Extract 26-joint skeleton from raw motion data (CSV format).
    Uses JOINT_MAPPING to handle single markers and averaged joints (wrist).
    
    Args:
        motion: (T, num_markers, 3)
        marker_names: list of marker names
        
    Returns:
        joints: (T, NUM_JOINTS, 3) - 26-joint skeleton
    """
    T = motion.shape[0]
    joints = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)
    
    marker_idx = {name: i for i, name in enumerate(marker_names)}
    
    for i, joint_name in enumerate(JOINT_NAMES):
        mapping = JOINT_MAPPING[joint_name]
        
        if mapping['type'] == 'single':
            # Single marker
            marker = mapping['marker']
            if marker in marker_idx:
                joints[:, i] = motion[:, marker_idx[marker]]
        
        elif mapping['type'] == 'average':
            # Average of multiple markers (only wrist uses this)
            markers = mapping['markers']
            valid_markers = [m for m in markers if m in marker_idx]
            if valid_markers:
                marker_data = np.stack([motion[:, marker_idx[m]] for m in valid_markers], axis=1)
                joints[:, i] = marker_data.mean(axis=1)
    
    return joints


# Alias for compatibility
def extract_upper_body_joints(motion: np.ndarray, marker_names: List[str]) -> np.ndarray:
    """Legacy alias - redirects to extract_full_body_joints"""
    return extract_full_body_joints(motion, marker_names)


def load_preprocessed_motion(filepath: Path) -> np.ndarray:
    """
    Load preprocessed motion data (piano-normalized, 26-joint skeleton).
    
    Args:
        filepath: Path to motion_26j.npy
        
    Returns:
        joints: (T, NUM_JOINTS, 3) - preprocessed 26-joint skeleton
    """
    return np.load(filepath)


def load_notes_csv(filepath: Path, warn_on_missing: bool = True) -> pd.DataFrame:
    """
    Load note annotations from align_notetime.csv.
    
    Uses align_notetime.csv directly which contains:
    - Real performance time (onset/offset in seconds)
    - MIDI number
    
    Hand assignment (left/right) is determined by MIDI number heuristic:
    - MIDI < 60 (below C4) -> left hand
    - MIDI >= 60 (C4 and above) -> right hand
    
    Args:
        filepath: Path to note.csv file (will look for align_notetime.csv)
        warn_on_missing: Print warning if align_notetime.csv is missing
    
    Returns:
        DataFrame with columns: Onset, Offset, MIDI number (in seconds)
    """
    # Use align_notetime.csv directly
    align_path = filepath.parent / filepath.name.replace('_note.csv', '_align_notetime.csv')
    
    if align_path.exists():
        align_df = pd.read_csv(align_path)
        
        # Rename columns to match expected format
        result = pd.DataFrame({
            'Onset': align_df['onset'],
            'Offset': align_df['offset'],
            'MIDI number': align_df['midi_number'],
        })
        
        return result
    else:
        if warn_on_missing:
            print(f"Warning: align_notetime.csv not found at {align_path}")
        # Fallback to note.csv (beat units, not seconds)
        note_df = pd.read_csv(filepath)
        return note_df[['Onset', 'Offset', 'MIDI number']]


def load_beats_csv(filepath: Path) -> pd.DataFrame:
    """Load beat annotations from CSV"""
    if filepath.exists():
        return pd.read_csv(filepath)
    return None


class MOSADataset(Dataset):
    """
    MOSA Piano Motion Dataset
    
    Loads motion capture and MIDI annotation data for piano performances.
    """
    
    def __init__(
        self,
        data_root: str,
        split: Literal['train', 'val', 'test'] = 'train',
        metadata_path: Optional[str] = None,
        splits_path: Optional[str] = None,
        normalize: bool = True,
        window_size: Optional[int] = None,  # If None, return full sequence
        stride: Optional[int] = None,
        fps: int = FPS,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.normalize = normalize
        self.window_size = window_size
        self.stride = stride or (window_size // 2 if window_size else None)
        self.fps = fps
        
        # Load metadata
        metadata_path = metadata_path or self.data_root / 'metadata.json'
        if Path(metadata_path).exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Load splits
        splits_path = splits_path or self.data_root / 'splits.json'
        if Path(splits_path).exists():
            with open(splits_path) as f:
                splits_data = json.load(f)
                self.samples_info = splits_data['samples'].get(split, [])
        else:
            self.samples_info = self._discover_samples()
        
        # Pre-load or index samples
        self.samples = []
        self._prepare_samples()
        
        # Compute normalization stats if needed
        if normalize and split == 'train':
            self._compute_normalization_stats()
    
    def _discover_samples(self) -> List[Dict]:
        """Discover samples from directory structure"""
        samples = []
        yp_dir = self.data_root / 'yp'
        
        if not yp_dir.exists():
            return samples
        
        for piece_dir in sorted(yp_dir.iterdir()):
            if not piece_dir.is_dir():
                continue
            piece = piece_dir.name
            
            for performer_dir in sorted(piece_dir.iterdir()):
                if not performer_dir.is_dir():
                    continue
                performer = performer_dir.name
                
                for trial_dir in sorted(performer_dir.iterdir()):
                    if not trial_dir.is_dir():
                        continue
                    trial = trial_dir.name
                    
                    samples.append({
                        'piece': piece,
                        'performer': performer,
                        'trial': trial,
                        'path': f'yp/{piece}/{performer}/{trial}'
                    })
        
        return samples
    
    def _prepare_samples(self):
        """Prepare sample list with optional windowing"""
        for info in self.samples_info:
            sample_path = self.data_root / info['path']
            
            # Check required files exist
            motion_path = sample_path / f"{info['piece']}_{info['performer']}_{info['trial']}_motion.csv"
            notes_path = sample_path / 'annotation' / 'annotations' / f"{info['piece']}_{info['performer']}_{info['trial']}_note.csv"
            
            if not motion_path.exists():
                # Try motion_norm.csv
                motion_path = sample_path / f"{info['piece']}_{info['performer']}_{info['trial']}_motion_norm.csv"
            
            if not motion_path.exists() or not notes_path.exists():
                continue
            
            if self.window_size is not None:
                # Load motion to get length
                motion, _ = load_motion_csv(motion_path)
                num_frames = len(motion)
                
                # Create windows
                for start in range(0, num_frames - self.window_size + 1, self.stride):
                    self.samples.append({
                        **info,
                        'motion_path': motion_path,
                        'notes_path': notes_path,
                        'start_frame': start,
                        'end_frame': start + self.window_size,
                    })
            else:
                self.samples.append({
                    **info,
                    'motion_path': motion_path,
                    'notes_path': notes_path,
                    'start_frame': None,
                    'end_frame': None,
                })
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization"""
        all_joints = []
        
        for sample in self.samples[:min(100, len(self.samples))]:  # Sample subset for speed
            motion, marker_names = load_motion_csv(sample['motion_path'])
            joints = extract_upper_body_joints(motion, marker_names)
            all_joints.append(joints)
        
        if all_joints:
            all_joints = np.concatenate(all_joints, axis=0)
            self.mean = all_joints.mean(axis=(0, 1))
            self.std = all_joints.std(axis=(0, 1))
            self.std[self.std < 1e-6] = 1.0
        else:
            self.mean = np.zeros(3)
            self.std = np.ones(3)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict:
        sample = self.samples[idx]
        
        # Load motion
        motion, marker_names = load_motion_csv(sample['motion_path'])
        joints = extract_upper_body_joints(motion, marker_names)
        
        # Window if specified
        if sample['start_frame'] is not None:
            joints = joints[sample['start_frame']:sample['end_frame']]
        
        # Normalize
        if self.normalize and hasattr(self, 'mean'):
            joints = (joints - self.mean) / self.std
        
        # Load notes
        notes_df = load_notes_csv(sample['notes_path'])
        
        # Filter notes to window
        if sample['start_frame'] is not None:
            start_time = sample['start_frame'] / self.fps
            end_time = sample['end_frame'] / self.fps
            notes_df = notes_df[
                (notes_df['Onset'] >= start_time) & (notes_df['Onset'] < end_time)
            ].copy()
            notes_df['Onset'] = notes_df['Onset'] - start_time
            notes_df['Offset'] = notes_df['Offset'] - start_time
        
        # Extract wrist positions
        right_wrist = joints[:, RWRIST_IDX]
        left_wrist = joints[:, LWRIST_IDX]
        
        # Get active keys per frame
        T = len(joints)
        active_keys = self._get_active_keys_per_frame(notes_df, T)
        
        return {
            'joints': torch.from_numpy(joints).float(),
            'right_wrist': torch.from_numpy(right_wrist).float(),
            'left_wrist': torch.from_numpy(left_wrist).float(),
            'notes': notes_df,
            'active_keys': torch.from_numpy(active_keys).long(),
            'piece': sample['piece'],
            'performer': sample['performer'],
            'trial': sample['trial'],
            'henle_difficulty': sample.get('henle_difficulty', 5),
        }
    
    def _get_active_keys_per_frame(self, notes_df: pd.DataFrame, num_frames: int) -> np.ndarray:
        """
        Get active MIDI keys for each frame
        
        Returns:
            active_keys: (T, max_concurrent_notes) - padded with -1
        """
        max_concurrent = 10  # Maximum concurrent notes to track
        active_keys = np.full((num_frames, max_concurrent), -1, dtype=np.int32)
        
        for t in range(num_frames):
            time_sec = t / self.fps
            
            # Find notes active at this time
            active = notes_df[
                (notes_df['Onset'] <= time_sec) & (notes_df['Offset'] > time_sec)
            ]
            
            midi_nums = active['MIDI number'].values[:max_concurrent]
            active_keys[t, :len(midi_nums)] = midi_nums
        
        return active_keys


def get_dataloader(
    data_root: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> torch.utils.data.DataLoader:
    """Create dataloader for MOSA dataset"""
    
    dataset = MOSADataset(data_root, split, **kwargs)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
    )
