#!/usr/bin/env python3
"""
s05_preprocess_motion.py: Motion Data Preprocessing

Preprocesses raw MOSA motion data to simplified 26-joint skeleton:
1. Piano coordinate normalization using piano markers (31-34)
2. Joint computation (only wrist uses averaging - anatomically recoverable)
3. Preserves all non-recoverable information (head, neck, chest, pelvis)

Simplified Skeleton (26 joints):
- Head: RFHD, LFHD, RBHD, LBHD (4) - individual markers
- Neck: C7, CLAV (2) - preserves front/back
- Chest: T10, STRN (2) - preserves front/back
- R Arm: RSHO, RELB, RWRIST (avg), RHAND (4)
- L Arm: LSHO, LELB, LWRIST (avg), LHAND (4)
- Pelvis: RASI, LASI, RPSI, LPSI (4) - preserves rotation
- R Leg: RKNE, RANKLE, RTOE (3)
- L Leg: LKNE, LANKLE, LTOE (3)

Output:
- MOSA_dataset/preprocessed/{sample_path}/motion_26j.npy (26 joints)
- MOSA_dataset/preprocessed/{sample_path}/piano_transform.json

Usage:
    python src/stage1/s05_preprocess_motion.py
    python src/stage1/s05_preprocess_motion.py --force
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import FULL_BODY_JOINTS, JOINT_MAPPING, NUM_FULL_BODY_JOINTS

# Piano key constants
PIANO_KEY_MIN = 21   # A0
PIANO_KEY_MAX = 108  # C8
NUM_KEYS = 88

# Standard piano dimensions (mm)
PIANO_WHITE_KEY_WIDTH = 23.5  # mm
PIANO_OCTAVE_WIDTH = 165.0    # mm (7 white keys)


class PianoCoordinateSystem:
    """
    Piano-based coordinate system for motion normalization.
    
    Uses 4 piano markers to define a local coordinate system:
    - Origin: Center of piano keyboard
    - X-axis: Along keyboard (left to right, low to high keys)
    - Y-axis: Perpendicular to keyboard (towards performer)
    - Z-axis: Up
    """
    
    def __init__(self, rightant: np.ndarray, rightpost: np.ndarray,
                 leftant: np.ndarray, leftpost: np.ndarray):
        """
        Initialize from 4 piano markers.
        
        Args:
            rightant: Right front marker (31)
            rightpost: Right back marker (32)
            leftant: Left front marker (33)
            leftpost: Left back marker (34)
        """
        self.rightant = rightant
        self.rightpost = rightpost
        self.leftant = leftant
        self.leftpost = leftpost
        
        self._compute_transform()
    
    def _compute_transform(self):
        """Compute transformation matrix from world to piano coordinates."""
        # Piano front edge (keyboard side)
        front_center = (self.rightant + self.leftant) / 2
        back_center = (self.rightpost + self.leftpost) / 2
        
        # Origin at front center of piano
        self.origin = front_center
        
        # X-axis: along keyboard (left to right)
        x_axis = self.rightant - self.leftant
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y-axis: perpendicular to keyboard (towards performer)
        y_axis = front_center - back_center
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Ensure orthogonality
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Make sure Z is pointing up
        if z_axis[2] < 0:
            z_axis = -z_axis
        
        # Recompute Y to ensure orthogonality
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Rotation matrix (world to piano)
        self.rotation = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3, 3)
        
        # Keyboard width
        self.keyboard_width = np.linalg.norm(self.rightant - self.leftant)
        
        # Key positions along X-axis
        self._compute_key_positions()
    
    def _compute_key_positions(self):
        """
        Compute 3D positions for each piano key.
        
        Standard piano: 88 keys (A0 to C8)
        Key layout follows standard piano with white and black keys.
        """
        # White key indices in an octave (0-6)
        # C, D, E, F, G, A, B
        white_keys_per_octave = [0, 2, 4, 5, 7, 9, 11]  # Semitone offsets
        
        # Piano starts at A0 (MIDI 21)
        # A0 is the 1st key, which is a white key
        
        # Compute X position for each key
        self.key_positions = {}
        
        # A0 starts at the left edge
        # The keyboard width spans from leftant to rightant
        half_width = self.keyboard_width / 2
        
        # Approximate: 88 keys span the keyboard width
        key_spacing = self.keyboard_width / NUM_KEYS
        
        for midi_num in range(PIANO_KEY_MIN, PIANO_KEY_MAX + 1):
            key_idx = midi_num - PIANO_KEY_MIN  # 0-87
            
            # X position: from left (-half_width) to right (+half_width)
            x = -half_width + key_idx * key_spacing + key_spacing / 2
            
            # Y position: at keyboard front edge (0)
            y = 0
            
            # Z position: keyboard height (assume same as origin)
            z = 0
            
            self.key_positions[midi_num] = np.array([x, y, z])
    
    def world_to_piano(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from world coordinates to piano coordinates.
        
        Args:
            points: (..., 3) world coordinates
            
        Returns:
            piano_coords: (..., 3) piano-local coordinates
        """
        # Translate to origin
        translated = points - self.origin
        
        # Rotate to piano frame
        # For batched operation
        original_shape = translated.shape
        flat = translated.reshape(-1, 3)
        rotated = (self.rotation @ flat.T).T
        
        return rotated.reshape(original_shape)
    
    def piano_to_world(self, points: np.ndarray) -> np.ndarray:
        """Transform from piano to world coordinates."""
        original_shape = points.shape
        flat = points.reshape(-1, 3)
        
        # Inverse rotation (R^T for orthogonal matrix)
        rotated = (self.rotation.T @ flat.T).T
        
        # Translate back
        translated = rotated + self.origin
        
        return translated.reshape(original_shape)
    
    def get_key_position(self, midi_num: int) -> np.ndarray:
        """Get 3D position of a piano key in piano coordinates."""
        if midi_num in self.key_positions:
            return self.key_positions[midi_num]
        return np.zeros(3)
    
    def to_dict(self) -> Dict:
        """Export to dictionary for saving."""
        return {
            'origin': self.origin.tolist(),
            'rotation': self.rotation.tolist(),
            'keyboard_width': float(self.keyboard_width),
            'rightant': self.rightant.tolist(),
            'rightpost': self.rightpost.tolist(),
            'leftant': self.leftant.tolist(),
            'leftpost': self.leftpost.tolist(),
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PianoCoordinateSystem':
        """Load from dictionary."""
        obj = cls.__new__(cls)
        obj.origin = np.array(d['origin'])
        obj.rotation = np.array(d['rotation'])
        obj.keyboard_width = d['keyboard_width']
        obj.rightant = np.array(d['rightant'])
        obj.rightpost = np.array(d['rightpost'])
        obj.leftant = np.array(d['leftant'])
        obj.leftpost = np.array(d['leftpost'])
        obj._compute_key_positions()
        return obj


def load_raw_motion(filepath: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load raw motion CSV file.
    
    Returns:
        motion: (T, num_markers, 3) array
        marker_names: list of marker names
    """
    df = pd.read_csv(filepath)
    columns = df.columns.tolist()
    
    # Extract marker names
    marker_names = []
    seen = set()
    for col in columns:
        marker = col.rsplit('_', 1)[0]
        if marker not in seen:
            marker_names.append(marker)
            seen.add(marker)
    
    num_markers = len(marker_names)
    num_frames = len(df)
    
    motion = df.values.reshape(num_frames, num_markers, 3)
    
    return motion, marker_names


def compute_wrist_centers(motion: np.ndarray, marker_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Compute wrist center positions from RWRA/RWRB and LWRA/LWRB.
    
    Args:
        motion: (T, num_markers, 3)
        marker_names: list of marker names
        
    Returns:
        dict with 'RWRIST_CENTER' and 'LWRIST_CENTER' arrays (T, 3)
    """
    marker_idx = {name: i for i, name in enumerate(marker_names)}
    
    result = {}
    
    # Right wrist center
    if 'RWRA' in marker_idx and 'RWRB' in marker_idx:
        rwra = motion[:, marker_idx['RWRA']]
        rwrb = motion[:, marker_idx['RWRB']]
        result['RWRIST_CENTER'] = (rwra + rwrb) / 2
    
    # Left wrist center
    if 'LWRA' in marker_idx and 'LWRB' in marker_idx:
        lwra = motion[:, marker_idx['LWRA']]
        lwrb = motion[:, marker_idx['LWRB']]
        result['LWRIST_CENTER'] = (lwra + lwrb) / 2
    
    return result


def extract_piano_markers(motion: np.ndarray, marker_names: List[str]) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract piano marker positions (average across all frames for stability).
    
    Returns:
        dict with 'Rightant', 'Rightpost', 'Leftant', 'Leftpost'
        or None if markers not found
    """
    marker_idx = {name: i for i, name in enumerate(marker_names)}
    
    piano_markers = ['Rightant', 'Rightpost', 'Leftant', 'Leftpost']
    
    # Check all markers exist
    for name in piano_markers:
        if name not in marker_idx:
            return None
    
    result = {}
    for name in piano_markers:
        # Use median for robustness against outliers
        result[name] = np.median(motion[:, marker_idx[name]], axis=0)
    
    return result


def preprocess_sample(
    motion: np.ndarray,
    marker_names: List[str],
    piano_system: Optional[PianoCoordinateSystem] = None,
) -> Dict:
    """
    Preprocess a single motion sample (legacy 12-joint version).
    
    Args:
        motion: (T, num_markers, 3) raw motion
        marker_names: list of marker names
        piano_system: optional pre-computed piano coordinate system
        
    Returns:
        dict with preprocessed data
    """
    marker_idx = {name: i for i, name in enumerate(marker_names)}
    T = motion.shape[0]
    
    # 1. Extract piano markers and create coordinate system
    if piano_system is None:
        piano_markers = extract_piano_markers(motion, marker_names)
        if piano_markers is None:
            print("Warning: Piano markers not found")
            piano_system = None
        else:
            piano_system = PianoCoordinateSystem(
                piano_markers['Rightant'],
                piano_markers['Rightpost'],
                piano_markers['Leftant'],
                piano_markers['Leftpost'],
            )
    
    # 2. Compute wrist centers
    wrist_centers = compute_wrist_centers(motion, marker_names)
    
    # 3. Build preprocessed joint array
    # New joint order for model:
    PREPROCESSED_JOINTS = [
        'CLAV',           # 0: Torso root
        'C7',             # 1: Neck
        'T10',            # 2: Mid back
        'STRN',           # 3: Sternum
        'RSHO',           # 4: Right shoulder
        'RELB',           # 5: Right elbow
        'RWRIST_CENTER',  # 6: Right wrist center (computed)
        'RFIN',           # 7: Right finger
        'LSHO',           # 8: Left shoulder
        'LELB',           # 9: Left elbow
        'LWRIST_CENTER',  # 10: Left wrist center (computed)
        'LFIN',           # 11: Left finger
    ]
    
    num_joints = len(PREPROCESSED_JOINTS)
    joints = np.zeros((T, num_joints, 3), dtype=np.float32)
    
    for i, joint_name in enumerate(PREPROCESSED_JOINTS):
        if joint_name == 'RWRIST_CENTER':
            joints[:, i] = wrist_centers.get('RWRIST_CENTER', np.zeros((T, 3)))
        elif joint_name == 'LWRIST_CENTER':
            joints[:, i] = wrist_centers.get('LWRIST_CENTER', np.zeros((T, 3)))
        elif joint_name in marker_idx:
            joints[:, i] = motion[:, marker_idx[joint_name]]
    
    # 4. Transform to piano coordinates
    if piano_system is not None:
        joints_normalized = piano_system.world_to_piano(joints)
    else:
        joints_normalized = joints
    
    return {
        'joints': joints,                    # Original coordinates
        'joints_normalized': joints_normalized,  # Piano-normalized
        'joint_names': PREPROCESSED_JOINTS,
        'piano_system': piano_system,
        'wrist_centers': wrist_centers,
    }


def preprocess_simplified_body(
    motion: np.ndarray,
    marker_names: List[str],
    piano_system: Optional[PianoCoordinateSystem] = None,
) -> Dict:
    """
    Preprocess motion data to simplified 26-joint skeleton.
    
    26-joint skeleton preserving non-recoverable information:
    - Head (4): RFHD, LFHD, RBHD, LBHD - individual markers
    - Neck (2): C7, CLAV - preserves front/back
    - Chest (2): T10, STRN - preserves front/back
    - R Arm (4): RSHO, RELB, RWRIST (avg), RHAND
    - L Arm (4): LSHO, LELB, LWRIST (avg), LHAND
    - Pelvis (4): RASI, LASI, RPSI, LPSI - preserves rotation
    - R Leg (3): RKNE, RANKLE, RTOE
    - L Leg (3): LKNE, LANKLE, LTOE
    
    Args:
        motion: (T, num_markers, 3) raw motion
        marker_names: list of marker names
        piano_system: optional pre-computed piano coordinate system
        
    Returns:
        dict with preprocessed simplified body data
    """
    marker_idx = {name: i for i, name in enumerate(marker_names)}
    T = motion.shape[0]
    
    # 1. Extract piano markers and create coordinate system
    if piano_system is None:
        piano_markers = extract_piano_markers(motion, marker_names)
        if piano_markers is not None:
            piano_system = PianoCoordinateSystem(
                piano_markers['Rightant'],
                piano_markers['Rightpost'],
                piano_markers['Leftant'],
                piano_markers['Leftpost'],
            )
    
    # 2. Build simplified joint array (26 joints)
    joints = np.zeros((T, NUM_FULL_BODY_JOINTS, 3), dtype=np.float32)
    
    for i, joint_name in enumerate(FULL_BODY_JOINTS):
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
    
    # 3. Transform to piano coordinates
    if piano_system is not None:
        joints_normalized = piano_system.world_to_piano(joints)
    else:
        joints_normalized = joints
    
    # 4. Split into body parts for convenience (26 joints)
    # 0-3: head (RFHD, LFHD, RBHD, LBHD)
    # 4-5: neck (C7, CLAV)
    # 6-7: chest (T10, STRN)
    # 8-11: right_arm (RSHO, RELB, RWRIST, RHAND)
    # 12-15: left_arm (LSHO, LELB, LWRIST, LHAND)
    # 16-19: pelvis (RASI, LASI, RPSI, LPSI)
    # 20-22: right_leg (RKNE, RANKLE, RTOE)
    # 23-25: left_leg (LKNE, LANKLE, LTOE)
    parts = {
        'head': joints_normalized[:, 0:4, :],           # (T, 4, 3)
        'neck': joints_normalized[:, 4:6, :],           # (T, 2, 3)
        'chest': joints_normalized[:, 6:8, :],          # (T, 2, 3)
        'right_arm': joints_normalized[:, 8:12, :],     # (T, 4, 3)
        'left_arm': joints_normalized[:, 12:16, :],     # (T, 4, 3)
        'pelvis': joints_normalized[:, 16:20, :],       # (T, 4, 3)
        'right_leg': joints_normalized[:, 20:23, :],    # (T, 3, 3)
        'left_leg': joints_normalized[:, 23:26, :],     # (T, 3, 3)
    }
    
    # Combined parts for model input
    combined_parts = {
        'backbone': joints_normalized[:, 0:8, :],       # (T, 8, 3) head + neck + chest
        'upper_body': joints_normalized[:, 0:16, :],    # (T, 16, 3) head + torso + arms
        'arms': np.concatenate([
            joints_normalized[:, 8:12, :],              # Right arm
            joints_normalized[:, 12:16, :],             # Left arm
        ], axis=1),                                     # (T, 8, 3)
        'root': joints_normalized[:, 16:20, :],         # (T, 4, 3) pelvis
        'lower_body': joints_normalized[:, 16:26, :],   # (T, 10, 3) pelvis + legs
    }
    
    return {
        'full_body': joints_normalized,
        'full_body_world': joints,
        'joint_names': FULL_BODY_JOINTS,
        'num_joints': NUM_FULL_BODY_JOINTS,
        'parts': parts,
        'combined_parts': combined_parts,
        'piano_system': piano_system,
    }


# Legacy function name for compatibility
def preprocess_full_body(
    motion: np.ndarray,
    marker_names: List[str],
    piano_system: Optional[PianoCoordinateSystem] = None,
) -> Dict:
    """Legacy wrapper - redirects to preprocess_simplified_body"""
    return preprocess_simplified_body(motion, marker_names, piano_system)


def _process_single_sample(sample, data_root, output_dir, force):
    """Process a single sample - returns dict with status and keyboard_width.
    
    This is a module-level function to allow pickling for multiprocessing.
    """
    from pathlib import Path
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    sample_path = data_root / sample['path']
    piece = sample['piece']
    performer = sample['performer']
    trial = sample['trial']
    
    # Output path
    sample_output_dir = output_dir / sample['path']
    full_body_path = sample_output_dir / 'motion_26j.npy'
    
    # Check if already processed
    if full_body_path.exists() and not force:
        return {'status': 'skipped', 'keyboard_width': None}
    
    # Find motion file
    motion_path = sample_path / f"{piece}_{performer}_{trial}_motion.csv"
    if not motion_path.exists():
        motion_path = sample_path / f"{piece}_{performer}_{trial}_motion_norm.csv"
    
    if not motion_path.exists():
        return {'status': 'error', 'keyboard_width': None, 'reason': 'motion not found'}
    
    try:
        # Load and preprocess
        motion, marker_names = load_raw_motion(motion_path)
        
        # Use simplified body (26 joints)
        fb_result = preprocess_simplified_body(motion, marker_names)
        
        # Downsample from 120fps to 60fps
        full_body = fb_result['full_body']
        full_body_60fps = full_body[::2]  # Take every 2nd frame
        
        # Convert mm -> m (all data stored in meters)
        full_body_60fps = full_body_60fps / 1000.0
        
        # Save
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save simplified 26-joint skeleton (60fps, meters)
        np.save(full_body_path, full_body_60fps)
        
        # Save piano transform
        keyboard_width = None
        if fb_result['piano_system'] is not None:
            transform_path = sample_output_dir / 'piano_transform.json'
            with open(transform_path, 'w') as f:
                json.dump(fb_result['piano_system'].to_dict(), f, indent=2)
            keyboard_width = fb_result['piano_system'].keyboard_width
        
        return {'status': 'processed', 'keyboard_width': keyboard_width}
        
    except Exception as e:
        return {'status': 'error', 'keyboard_width': None, 'reason': str(e)}


def process_dataset(data_root: Path, output_dir: Path, force: bool = False):
    """
    Process entire dataset to simplified 26-joint skeleton.
    
    Args:
        data_root: Path to MOSA dataset
        output_dir: Output directory for preprocessed data
        force: Overwrite existing files
    """
    
    # Load splits
    splits_path = data_root / 'splits.json'
    if not splits_path.exists():
        print(f"Error: splits.json not found at {splits_path}")
        return
    
    with open(splits_path) as f:
        splits_data = json.load(f)
    
    # Collect all samples
    all_samples = []
    for split in ['train', 'val', 'test']:
        all_samples.extend(splits_data['samples'].get(split, []))
    
    print(f"Processing {len(all_samples)} samples...")
    
    # Statistics
    stats = {
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'keyboard_widths': [],
    }
    
    # Use simplified 26-joint skeleton from constants
    from data.constants import FULL_BODY_JOINTS, NUM_FULL_BODY_JOINTS
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial
    
    num_workers = max(1, os.cpu_count() // 2)
    print(f"Using {num_workers} workers")
    
    # Process in parallel using module-level function
    process_fn = partial(_process_single_sample, data_root=str(data_root), output_dir=str(output_dir), force=force)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_fn, sample): sample for sample in all_samples}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Preprocessing'):
            result = future.result()
            
            if result['status'] == 'processed':
                stats['processed'] += 1
                if result['keyboard_width'] is not None:
                    stats['keyboard_widths'].append(result['keyboard_width'])
            elif result['status'] == 'skipped':
                stats['skipped'] += 1
            else:
                stats['errors'] += 1
    
    # Save joint names (26-joint skeleton)
    joint_names_path = output_dir / 'joint_names.json'
    with open(joint_names_path, 'w') as f:
        json.dump(list(FULL_BODY_JOINTS), f, indent=2)
    
    # Save piano key positions (using average keyboard width)
    if stats['keyboard_widths']:
        avg_width = np.mean(stats['keyboard_widths'])
        key_positions = {}
        half_width = avg_width / 2
        key_spacing = avg_width / NUM_KEYS
        
        for midi_num in range(PIANO_KEY_MIN, PIANO_KEY_MAX + 1):
            key_idx = midi_num - PIANO_KEY_MIN
            x = -half_width + key_idx * key_spacing + key_spacing / 2
            key_positions[str(midi_num)] = [float(x), 0.0, 0.0]
        
        piano_geometry = {
            'key_positions': key_positions,
            'keyboard_width_mm': float(avg_width),
            'key_spacing_mm': float(key_spacing),
            'num_keys': NUM_KEYS,
            'midi_min': PIANO_KEY_MIN,
            'midi_max': PIANO_KEY_MAX,
        }
        
        geometry_path = output_dir / 'piano_geometry.json'
        with open(geometry_path, 'w') as f:
            json.dump(piano_geometry, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped:   {stats['skipped']}")
    print(f"Errors:    {stats['errors']}")
    print(f"Joints:    {NUM_FULL_BODY_JOINTS}")
    if stats['keyboard_widths']:
        print(f"\nPiano keyboard width: {np.mean(stats['keyboard_widths']):.1f} ± {np.std(stats['keyboard_widths']):.1f} mm")
    print(f"\nOutput: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='s05: Preprocess Motion Data')
    parser.add_argument('--data_root', type=str, default='MOSA_dataset')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / 'preprocessed'
    
    print("=" * 70)
    print("s05: Motion Data Preprocessing (26-joint Skeleton)")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print("\n26-joint skeleton (preserves non-recoverable info):")
    print("  - Head (4): RFHD, LFHD, RBHD, LBHD")
    print("  - Neck (2): C7, CLAV (front/back)")
    print("  - Chest (2): T10, STRN (front/back)")
    print("  - R Arm (4): RSHO, RELB, RWRIST (avg), RHAND")
    print("  - L Arm (4): LSHO, LELB, LWRIST (avg), LHAND")
    print("  - Pelvis (4): RASI, LASI, RPSI, LPSI (rotation)")
    print("  - R Leg (3): RKNE, RANKLE, RTOE")
    print("  - L Leg (3): LKNE, LANKLE, LTOE")
    
    process_dataset(data_root, output_dir, args.force)
    
    print("\n" + "=" * 70)
    print("s05 Complete!")
    print("=" * 70)
    print("\nNext step:")
    print("  python src/stage1/s12_precompute_training_data.py "
          "--onset_window 0.5 --output_dir MOSA_dataset/training_cache_W05")
    print("  (also re-run with --onset_window 1.0 → _W10 and 2.0 → _B)")


if __name__ == '__main__':
    main()
