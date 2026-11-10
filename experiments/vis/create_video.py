#!/usr/bin/env python3
"""
create_video.py: piano-performance motion video generator.

Visualizes Pianoid motion as video.
- 3D skeleton animation (26 joints - simplified skeleton preserving non-recoverable info)
- GT-only visualization, or GT vs Generated comparison
- Optional audio mux
- Fast rendering via parallel processing

Usage:
    # Default: render the full test split (30 fps)
    python experiments/vis/create_video.py

    # Pick a different split
    python experiments/vis/create_video.py --split train

    # Single-sample
    python experiments/vis/create_video.py --sample_id yp/ba1/yp08/t1 --gt_only
    python experiments/vis/create_video.py --sample_id yp/ba1/yp08/t1 --compare_gt

    # Stage 4 (scripts/train/stage4.sh) output: hierarchical MDM trained from musical features
    python experiments/vis/create_video.py --hierarchical_mdm_dir results/hierarchical_mdm --sample_idx 0 --compare_gt
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import multiprocessing as mp
from functools import partial
import tempfile
import shutil

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.constants import (
    FULL_BODY_JOINTS, NUM_FULL_BODY_JOINTS, FPS,
    FULL_BODY_EDGES, BODY_PARTS,
    PARCO_PARTS, PARCO_PART_NAMES,
)

# Default FPS if not defined
if FPS is None:
    FPS = 120


# ============================================================
# SKELETON CONFIGURATION (26 joints - preserving non-recoverable info)
# 
# ParCo 6-Part structure:
#   1. Backbone: indices 0-7 (RFHD, LFHD, RBHD, LBHD, C7, CLAV, T10, STRN) - Green
#   2. R_Arm: indices 8-11 (RSHO, RELB, RWRIST, RHAND) - Red
#   3. L_Arm: indices 12-15 (LSHO, LELB, LWRIST, LHAND) - Orange
#   4. Root: indices 16-19 (RASI, LASI, RPSI, LPSI) - Purple
#   5. R_Leg: indices 20-22 (RKNE, RANKLE, RTOE) - Blue
#   6. L_Leg: indices 23-25 (LKNE, LANKLE, LTOE) - Teal
# ============================================================

# ParCo 6-Part Colors (primary visualization)
PARCO_PART_COLORS = {
    'Backbone': '#2ecc71',  # Green  - head + spine
    'R_Arm': '#e74c3c',     # Red    - right arm
    'L_Arm': '#e67e22',     # Orange - left arm
    'Root': '#9b59b6',      # Purple - pelvis
    'R_Leg': '#3498db',     # Blue   - right leg
    'L_Leg': '#1abc9c',     # Teal   - left leg
}

# Colors by 3-part training (legacy)
TRAIN_PART_COLORS = {
    'head_torso': '#2ecc71',  # Green - trained together
    'arms': '#e74c3c',        # Red/Orange - trained together
    'lower': '#3498db',       # Blue - trained together
}

# ============================================================
# STAGE-BASED COLORS (Pianoid Hierarchical Generation)
# Color the joints generated at each stage.
# ============================================================
STAGE_COLORS = {
    'stage1_wrist': '#e74c3c',      # Red   - S1 BiLSTM (wrists)
    'stage2_backbone': '#2ecc71',   # Green - S2 MDM backbone (head+torso+shoulders)
    'stage3_elbow': '#f1c40f',      # Gold  - S3 BiLSTM (elbows)
    'stage4_lower': '#3498db',      # Blue  - S4 FiLM (lower body)
}

# Per-stage joint indices (26-joint skeleton)
# S1: MIDI -> wrists (+ hands)
STAGE1_WRIST_INDICES = [10, 11, 14, 15]  # RWRIST, RHAND, LWRIST, LHAND

# S2: wrists -> 10-joint backbone (head+torso+shoulders)
STAGE2_BACKBONE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12]  # head(0-3) + neck/chest(4-7) + shoulders(8, 12)

# S3: wrists+chest+shoulders -> elbows
STAGE3_ELBOW_INDICES = [9, 13]  # RELB, LELB

# S4: spine -> lower body
STAGE4_LOWER_INDICES = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]  # pelvis + legs


def get_joint_color_by_stage(joint_idx):
    """Get color for joint based on generation stage (Pianoid pipeline).

    S1 (BiLSTM wrists):      indices 10, 11, 14, 15 - RED
    S2 (MDM backbone):       indices 0-7, 8, 12     - GREEN
    S3 (BiLSTM elbows):      indices 9, 13          - GOLD
    S4 (FiLM lower body):    indices 16-25          - BLUE
    """
    if joint_idx in STAGE1_WRIST_INDICES:
        return STAGE_COLORS['stage1_wrist']
    elif joint_idx in STAGE2_BACKBONE_INDICES:
        return STAGE_COLORS['stage2_backbone']
    elif joint_idx in STAGE3_ELBOW_INDICES:
        return STAGE_COLORS['stage3_elbow']
    elif joint_idx in STAGE4_LOWER_INDICES:
        return STAGE_COLORS['stage4_lower']
    else:
        return '#95a5a6'  # gray fallback

# Detailed colors for visualization
PART_COLORS = {
    'head': '#27ae60',        # Dark green (4 head markers)
    'neck': '#2ecc71',        # Light green (C7, CLAV)
    'chest': '#1e9b5e',       # Medium green (T10, STRN)
    'right_arm': '#e74c3c',   # Red
    'left_arm': '#e67e22',    # Orange
    'pelvis': '#9b59b6',      # Purple (RASI, LASI, RPSI, LPSI)
    'right_leg': '#3498db',   # Blue
    'left_leg': '#1abc9c',    # Teal
}


def get_joint_color(joint_idx):
    """Get color for joint based on ParCo 6-PART structure (26 joints)
    
    ParCo Part 1 - Backbone (GREEN):
      0-7: head+neck+chest (RFHD, LFHD, RBHD, LBHD, C7, CLAV, T10, STRN)
    
    ParCo Part 2 - R_Arm (RED):
      8-11: right_arm (RSHO, RELB, RWRIST, RHAND)
    
    ParCo Part 3 - L_Arm (ORANGE):
      12-15: left_arm (LSHO, LELB, LWRIST, LHAND)
    
    ParCo Part 4 - Root (PURPLE):
      16-19: pelvis (RASI, LASI, RPSI, LPSI)
    
    ParCo Part 5 - R_Leg (BLUE):
      20-22: right_leg (RKNE, RANKLE, RTOE)
    
    ParCo Part 6 - L_Leg (TEAL):
      23-25: left_leg (LKNE, LANKLE, LTOE)
    """
    # Part 1: Backbone (indices 0-7)
    if joint_idx < 8:
        return PARCO_PART_COLORS['Backbone']  # Green
    
    # Part 2: R_Arm (indices 8-11)
    elif joint_idx < 12:
        return PARCO_PART_COLORS['R_Arm']  # Red
    
    # Part 3: L_Arm (indices 12-15)
    elif joint_idx < 16:
        return PARCO_PART_COLORS['L_Arm']  # Orange
    
    # Part 4: Root (indices 16-19)
    elif joint_idx < 20:
        return PARCO_PART_COLORS['Root']  # Purple
    
    # Part 5: R_Leg (indices 20-22)
    elif joint_idx < 23:
        return PARCO_PART_COLORS['R_Leg']  # Blue
    
    # Part 6: L_Leg (indices 23-25)
    else:
        return PARCO_PART_COLORS['L_Leg']  # Teal


# ============================================================
# FRAME RENDERING FUNCTIONS (for parallel processing)
# ============================================================

def setup_axes(ax, bounds, title=None, view_angle=(20, 60)):
    """Setup 3D axes with proper limits"""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    ax.set_xlabel('X (mm)', fontsize=9)
    ax.set_ylabel('Y (mm)', fontsize=9)
    ax.set_zlabel('Z (mm)', fontsize=9)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    ax.grid(True, alpha=0.3)


def get_edge_color(parent_idx, child_idx):
    """Get edge color based on connected joints.
    
    If both joints belong to same part, use that part's color.
    Otherwise, use the parent's color (hierarchical).
    """
    parent_color = get_joint_color(parent_idx)
    child_color = get_joint_color(child_idx)
    
    # Same part - use that color
    if parent_color == child_color:
        return parent_color
    
    # Cross-part edges: use darker/neutral color for connecting bones
    # Spine/torso connections
    if parent_idx in [4, 5, 6, 7] or child_idx in [4, 5, 6, 7]:
        return '#95a5a6'  # Gray for spine connections
    
    return parent_color


def get_joint_size(joint_idx):
    """Get joint marker size based on body part.
    
    Larger markers for major joints, smaller for extremities.
    """
    # Head markers (small)
    if joint_idx < 4:
        return 40
    # Neck/Chest (medium)
    elif joint_idx < 8:
        return 60
    # Shoulders, elbows (medium-large)
    elif joint_idx in [8, 9, 12, 13]:
        return 70
    # Wrists, hands (small)
    elif joint_idx in [10, 11, 14, 15]:
        return 50
    # Pelvis (large)
    elif joint_idx < 20:
        return 70
    # Knees (medium)
    elif joint_idx in [20, 23]:
        return 65
    # Ankles, toes (small)
    else:
        return 45


def draw_skeleton(ax, frame, alpha=1.0, use_stage_colors=False):
    """Draw skeleton for single frame with improved visualization
    
    Args:
        ax: Matplotlib 3D axis
        frame: (N, 3) joint positions for single frame
        alpha: Transparency
        use_stage_colors: If True, use Pianoid stage-based colors (for Generated motion)
    """
    num_joints = len(frame)
    
    # Select color function based on mode
    color_fn = get_joint_color_by_stage if use_stage_colors else get_joint_color
    
    # Draw edges first (so joints appear on top)
    for parent, child in FULL_BODY_EDGES:
        if parent < num_joints and child < num_joints:
            if use_stage_colors:
                # For stage colors, use child's color (more intuitive for hierarchical generation)
                color = color_fn(child)
            else:
                color = get_edge_color(parent, child)
            
            # Thicker lines for major bones
            if parent < 8 and child < 8:
                # Head/neck/chest - thinner
                linewidth = 1.5
            elif parent >= 16 or child >= 16:
                # Pelvis/legs - medium
                linewidth = 2.0
            else:
                # Arms - medium
                linewidth = 2.0
            
            ax.plot(
                [frame[parent, 0], frame[child, 0]],
                [frame[parent, 1], frame[child, 1]],
                [frame[parent, 2], frame[child, 2]],
                c=color, linewidth=linewidth, alpha=alpha * 0.85
            )
    
    # Draw joints on top
    for i in range(num_joints):
        color = color_fn(i)
        size = get_joint_size(i)
        ax.scatter(
            frame[i, 0], frame[i, 1], frame[i, 2],
            c=color, s=size, alpha=alpha, 
            edgecolors='white', linewidths=0.5,
            zorder=10  # Draw on top
        )


def calculate_bounds(joints, margin_ratio=0.1):
    """Calculate axes bounds from joints data"""
    all_coords = joints.reshape(-1, 3)
    
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
    
    margin = max(x_max - x_min, y_max - y_min, z_max - z_min) * margin_ratio
    
    return (
        x_min - margin, x_max + margin,
        y_min - margin, y_max + margin,
        z_min - margin, z_max + margin
    )


def render_single_frame(args):
    """Render single frame - for parallel processing"""
    frame_idx, frame_data, bounds, output_path, view_angle, title, dpi, figsize = args
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    setup_axes(ax, bounds, title=title, view_angle=view_angle)
    draw_skeleton(ax, frame_data)
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    return frame_idx


def render_comparison_frame(args):
    """Render comparison frame (GT left, Generated right) - for parallel processing"""
    (frame_idx, gt_frame, gen_frame, gt_bounds, gen_bounds, 
     output_path, view_angle, total_frames, dpi, figsize, use_stage_colors) = args
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # GT on LEFT (always use ParCo colors)
    ax1 = fig.add_subplot(121, projection='3d')
    setup_axes(ax1, gt_bounds, title=f'Ground Truth', view_angle=view_angle)
    draw_skeleton(ax1, gt_frame, use_stage_colors=False)
    
    # Generated on RIGHT (use stage colors if enabled)
    ax2 = fig.add_subplot(122, projection='3d')
    gen_title = 'Generated (Stage Colors)' if use_stage_colors else 'Generated'
    setup_axes(ax2, gen_bounds, title=gen_title, view_angle=view_angle)
    draw_skeleton(ax2, gen_frame, use_stage_colors=use_stage_colors)
    
    # Add frame info and legend for stage colors
    if use_stage_colors:
        fig.suptitle(f'Frame {frame_idx} / {total_frames-1}\n'
                     '🔴 Hand (Stage2) | 🟢 Backbone (Stage3-1) | 🟡 Elbow (Stage3-2) | 🔵 Lower (Stage3-3)',
                     fontsize=12, fontweight='bold')
    else:
        fig.suptitle(f'Frame {frame_idx} / {total_frames-1}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    return frame_idx


# ============================================================
# VIDEO CREATOR CLASS
# ============================================================

class PianoMotionVideoCreator:
    """Piano-performance motion video generator (with parallel rendering)."""

    def __init__(
        self,
        video_fps: int = 30,
        render_fps: int = None,
        figsize: tuple = (10, 8),
        dpi: int = 100,
        view_angle: tuple = (20, 60),
        n_workers: int = None,
    ):
        """
        Args:
            video_fps: final video FPS.
            render_fps: FPS at which frames are rendered (defaults to video_fps).
            figsize: figure size.
            dpi: figure DPI.
            view_angle: (elevation, azimuth).
            n_workers: number of parallel workers (defaults to CPU core count).
        """
        self.video_fps = video_fps
        self.render_fps = render_fps or video_fps
        self.figsize = figsize
        self.dpi = dpi
        self.view_angle = view_angle
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        
    def _get_frame_indices(self, total_frames: int, source_fps: int):
        """Get frame indices to render based on render_fps"""
        skip = max(1, source_fps // self.render_fps)
        return list(range(0, total_frames, skip))
    
    def create_video(
        self,
        joints: np.ndarray,
        output_path: Path,
        title: str = "Piano Motion",
        source_fps: int = FPS,
    ):
        """
        Create video from joint positions with parallel rendering
        
        Args:
            joints: (T, 26, 3) joint positions
            output_path: Path to save video (.mp4)
            title: Video title
            source_fps: Source data FPS (default: 120)
        """
        print(f"\nCreating video: {output_path}")
        print(f"  Input: {joints.shape}, source FPS: {source_fps}")
        print(f"  Render FPS: {self.render_fps}, Video FPS: {self.video_fps}")
        print(f"  Workers: {self.n_workers}")
        
        T = len(joints)
        frame_indices = self._get_frame_indices(T, source_fps)
        num_frames = len(frame_indices)
        print(f"  Frames to render: {num_frames}")
        
        # Calculate bounds
        bounds = calculate_bounds(joints)
        
        # Create temp directory for frames
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Prepare render arguments
            render_args = []
            for i, frame_idx in enumerate(frame_indices):
                output_frame_path = temp_dir / f"frame_{i:06d}.png"
                render_args.append((
                    frame_idx,
                    joints[frame_idx],
                    bounds,
                    str(output_frame_path),
                    self.view_angle,
                    f"{title} (Frame {frame_idx}/{T-1})",
                    self.dpi,
                    self.figsize,
                ))
            
            # Parallel rendering
            print(f"  Rendering frames in parallel...")
            with mp.Pool(self.n_workers) as pool:
                list(tqdm(
                    pool.imap(render_single_frame, render_args),
                    total=num_frames,
                    desc="  Rendering",
                ))
            
            # Compile video using ffmpeg
            print(f"  Compiling video...")
            self._compile_video(temp_dir, output_path)
            print(f"  Saved: {output_path}")
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def create_comparison_video(
        self,
        ground_truth: np.ndarray,
        generated: np.ndarray,
        output_path: Path,
        source_fps: int = FPS,
        use_stage_colors: bool = False,
    ):
        """
        Create side-by-side comparison video
        GT on LEFT, Generated on RIGHT
        
        Args:
            ground_truth: (T, 26, 3) GT motion
            generated: (T, 26, 3) generated motion
            output_path: Path to save video
            source_fps: Source data FPS
            use_stage_colors: If True, Generated uses Pianoid stage-based colors
        """
        print(f"\nCreating comparison video: {output_path}")
        print(f"  GT: {ground_truth.shape}, Generated: {generated.shape}")
        print(f"  Render FPS: {self.render_fps}, Video FPS: {self.video_fps}")
        print(f"  Workers: {self.n_workers}")
        print(f"  Stage colors: {use_stage_colors}")
        
        # Align lengths
        T = min(len(ground_truth), len(generated))
        ground_truth = ground_truth[:T]
        generated = generated[:T]
        
        frame_indices = self._get_frame_indices(T, source_fps)
        num_frames = len(frame_indices)
        print(f"  Frames to render: {num_frames}")
        
        # Calculate bounds (use combined for consistent view)
        combined = np.concatenate([ground_truth, generated], axis=0)
        gt_bounds = calculate_bounds(combined)
        gen_bounds = gt_bounds  # Same bounds for fair comparison
        
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Prepare render arguments
            comparison_figsize = (16, 8)  # Wider for side-by-side
            
            render_args = []
            for i, frame_idx in enumerate(frame_indices):
                output_frame_path = temp_dir / f"frame_{i:06d}.png"
                render_args.append((
                    frame_idx,
                    ground_truth[frame_idx],
                    generated[frame_idx],
                    gt_bounds,
                    gen_bounds,
                    str(output_frame_path),
                    self.view_angle,
                    T,
                    self.dpi,
                    comparison_figsize,
                    use_stage_colors,
                ))
            
            # Parallel rendering
            print(f"  Rendering comparison frames in parallel...")
            with mp.Pool(self.n_workers) as pool:
                list(tqdm(
                    pool.imap(render_comparison_frame, render_args),
                    total=num_frames,
                    desc="  Rendering",
                ))
            
            # Compile video
            print(f"  Compiling video...")
            self._compile_video(temp_dir, output_path)
            print(f"  Saved: {output_path}")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _compile_video(self, frame_dir: Path, output_path: Path):
        """Compile frames into video using ffmpeg"""
        import subprocess
        
        frame_pattern = str(frame_dir / "frame_%06d.png")
        
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(self.video_fps),
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'medium',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FFmpeg error: {result.stderr}")
            raise RuntimeError("FFmpeg failed")


def add_audio_to_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_offset: float = 0.0,
    duration: float = None,
):
    """Add audio to video using moviepy"""
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
        
        print(f"\nAdding audio to video...")
        print(f"  Video: {video_path}")
        print(f"  Audio: {audio_path}")
        
        video_clip = VideoFileClip(str(video_path))
        
        if duration is None:
            duration = video_clip.duration
        
        audio_clip = AudioFileClip(str(audio_path))
        if audio_offset > 0:
            audio_clip = audio_clip.subclip(audio_offset, min(audio_offset + duration, audio_clip.duration))
        else:
            audio_clip = audio_clip.subclip(0, min(duration, audio_clip.duration))
        
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None,
        )
        
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
        print(f"  Saved: {output_path}")
        return True
        
    except ImportError:
        print("  Warning: moviepy not installed. Skipping audio.")
        return False
    except Exception as e:
        print(f"  Error adding audio: {e}")
        return False


def load_ground_truth(data_root: Path, sample_id: str) -> np.ndarray:
    """Load ground truth from preprocessed data (26-joint skeleton)"""
    # Try motion_26j.npy first (26-joint skeleton, data is in meters)
    preprocessed_path = data_root / 'preprocessed' / sample_id / 'motion_26j.npy'
    if preprocessed_path.exists():
        print(f"  Loading GT from: {preprocessed_path}")
        data = np.load(preprocessed_path)
        # Convert m to mm for visualization
        return data * 1000
    
    # Try legacy formats
    for legacy_name in ['motion_16j.npy', 'full_body_normalized.npy', 'motion_normalized.npy']:
        legacy_path = data_root / 'preprocessed' / sample_id / legacy_name
        if legacy_path.exists():
            print(f"  Loading from legacy format: {legacy_path}")
            return np.load(legacy_path)
    
    return None


def get_sample_id_from_splits(data_root: Path, sample_idx: int, split: str = 'test') -> str:
    """Get sample_id (path) from splits.json by sample index."""
    splits_path = data_root / 'splits.json'
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path}")
    with open(splits_path) as f:
        splits = json.load(f)
    samples = splits.get('samples', {}).get(split, [])
    if sample_idx < 0 or sample_idx >= len(samples):
        raise IndexError(f"sample_idx {sample_idx} out of range [0, {len(samples)-1}]")
    return samples[sample_idx]['path']


def load_hierarchical_mdm_result(
    hierarchical_mdm_dir: Path,
    sample_idx: int,
) -> np.ndarray:
    """
    Load generated motion from scripts/train/stage4.sh output (results/hierarchical_mdm/sample_XXX.npz).
    Returns (T, 26, 3) in mm for visualization.
    """
    npz_path = hierarchical_mdm_dir / f"sample_{sample_idx:03d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Stage4 output not found: {npz_path}")
    data = np.load(npz_path)
    generated = data['generated']  # (N, T, 26, 3) or (T, 26, 3)
    generated = np.squeeze(generated)
    if generated.ndim != 3:
        raise ValueError(f"Expected (T, 26, 3), got shape {generated.shape}")
    # Stage4 saves in meters (same as gt_joints); convert to mm
    return generated.astype(np.float64) * 1000.0


# ============================================================
# MAIN
# ============================================================

def process_single_sample(args, sample_id: str, output_base: Path, creator) -> dict:
    """Process a single sample for batch mode"""
    try:
        data_root = Path(args.data_root)
        output_dir = output_base / sample_id.replace('/', '_')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load generated motion
        full_body_path = data_root / 'generated' / sample_id / 'generated_motion.npy'
        if not full_body_path.exists():
            # Try legacy name
            full_body_path = data_root / 'generated' / sample_id / 'full_body.npy'
            if not full_body_path.exists():
                return {'status': 'skip', 'sample': sample_id, 'reason': 'no generated data'}
        
        generated = np.load(full_body_path)
        # Convert m to mm for visualization
        generated = generated * 1000
        
        # Load GT for comparison
        ground_truth = load_ground_truth(data_root, sample_id)
        
        if args.max_frames:
            generated = generated[:args.max_frames]
            if ground_truth is not None:
                ground_truth = ground_truth[:args.max_frames]
        
        # Create comparison video
        if ground_truth is not None and args.compare_gt:
            video_path = output_dir / 'comparison.mp4'
            creator.create_comparison_video(
                ground_truth=ground_truth,
                generated=generated,
                output_path=video_path,
            )
        else:
            video_path = output_dir / 'generated.mp4'
            creator.create_video(
                joints=generated,
                output_path=video_path,
                title=f"Generated - {sample_id}",
            )
        
        return {'status': 'success', 'sample': sample_id, 'path': str(video_path)}
    except Exception as e:
        return {'status': 'error', 'sample': sample_id, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Create Piano Motion Video')
    parser.add_argument('--data_root', type=str, default='MOSA_dataset')
    parser.add_argument('--sample_id', type=str, default=None,
                       help='Sample path like yp/ba1/yp08/t1')
    parser.add_argument('--output_dir', type=str, default=None)
    # Stage 4 (scripts/train/stage4.sh) output: hierarchical MDM trained on musical features
    parser.add_argument('--hierarchical_mdm_dir', type=str, default=None,
                       help='Stage4 output dir (e.g. results/hierarchical_mdm); use with --sample_idx')
    parser.add_argument('--sample_idx', type=int, default=None,
                       help='Sample index for Stage4 output (e.g. 0 for sample_000.npz); use with --hierarchical_mdm_dir')
    parser.add_argument('--hierarchical_split', type=str, default='test',
                       help='Split name for mapping sample_idx to sample_id (default: test)')
    
    # Batch mode (default: True when no sample_id provided)
    parser.add_argument('--no_batch', action='store_true',
                       help='Disable batch mode (requires --sample_id)')
    parser.add_argument('--split', type=str, default='test',
                       help='Split to process in batch mode (default: test)')
    parser.add_argument('--batch_workers', type=int, default=None,
                       help='Number of parallel sample workers (default: half of CPU cores)')
    
    # FPS options
    parser.add_argument('--video_fps', type=int, default=30,
                       help='Output video FPS (default: 30)')
    parser.add_argument('--render_fps', type=int, default=30,
                       help='Frames to render per second from source (default: 30)')
    
    # Rendering options
    parser.add_argument('--dpi', type=int, default=100, help='DPI (default: 100)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count - 1)')
    
    # View options
    parser.add_argument('--view_elev', type=float, default=20, help='View elevation angle')
    parser.add_argument('--view_azim', type=float, default=60, help='View azimuth angle')
    
    # Features
    parser.add_argument('--with_audio', action='store_true', help='Add audio to video')
    parser.add_argument('--compare_gt', action='store_true', 
                       help='Create GT vs Generated comparison (GT left, Generated right)')
    parser.add_argument('--gt_only', action='store_true',
                       help='Visualize ground truth only (no generated needed)')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to process (for testing)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Duration in seconds to visualize (e.g., 1.0 for 1 second)')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    use_hierarchical_mdm = (
        args.hierarchical_mdm_dir is not None and args.sample_idx is not None
    )
    
    # Batch mode: default when no sample_id and not hierarchical_mdm single-sample
    run_batch = (
        not args.no_batch
        and args.sample_id is None
        and not use_hierarchical_mdm
    )
    
    if run_batch:
        import json
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # Load samples
        splits_path = data_root / 'splits.json'
        with open(splits_path) as f:
            splits = json.load(f)
        samples = splits['samples'].get(args.split, [])
        
        if not samples:
            print(f"No samples found for split: {args.split}")
            return
        
        # Output directory
        output_base = Path(args.output_dir) if args.output_dir else Path('./results/video')
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Default batch workers: half of CPU cores
        import os
        batch_workers = args.batch_workers if args.batch_workers else max(1, os.cpu_count() // 2)
        
        print("=" * 70)
        print("Batch Video Creation")
        print("=" * 70)
        print(f"Split: {args.split}")
        print(f"Samples: {len(samples)}")
        print(f"Output: {output_base}")
        print(f"Video FPS: {args.video_fps}")
        print(f"Batch workers: {batch_workers}")
        print("=" * 70)
        
        # Create video creator (with single worker for batch - parallelism at sample level)
        creator = PianoMotionVideoCreator(
            video_fps=args.video_fps,
            render_fps=args.render_fps,
            dpi=args.dpi,
            view_angle=(args.view_elev, args.view_azim),
            n_workers=1,  # Single worker per sample
        )
        
        # Process samples in parallel
        results = {'success': 0, 'skip': 0, 'error': 0}
        
        from functools import partial
        process_fn = partial(process_single_sample, args, creator=creator, output_base=output_base)
        
        with ProcessPoolExecutor(max_workers=batch_workers) as executor:
            futures = {executor.submit(process_single_sample, args, s['path'], output_base, creator): s['path'] 
                      for s in samples}
            
            from tqdm import tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc="Creating videos"):
                result = future.result()
                results[result['status']] += 1
                if result['status'] == 'error':
                    print(f"\n  Error: {result['sample']}: {result.get('error', 'Unknown')}")
        
        print("\n" + "=" * 70)
        print("Batch complete!")
        print(f"Success: {results['success']}, Skip: {results['skip']}, Error: {results['error']}")
        print(f"Output: {output_base}")
        print("=" * 70)
        return
    
    # Single sample mode (by sample_id or by Stage4 hierarchical_mdm_dir + sample_idx)
    if use_hierarchical_mdm:
        hierarchical_mdm_dir = Path(args.hierarchical_mdm_dir)
        sample_id = get_sample_id_from_splits(
            data_root, args.sample_idx, args.hierarchical_split
        )
        print("=" * 70)
        print("Stage 4 (Hierarchical MDM) output visualization")
        print("=" * 70)
        print(f"Source: {hierarchical_mdm_dir / f'sample_{args.sample_idx:03d}.npz'}")
        print(f"Sample: {sample_id} (index {args.sample_idx})")
        generated = load_hierarchical_mdm_result(hierarchical_mdm_dir, args.sample_idx)
        print(f"Loaded generated: {generated.shape}")
    else:
        if not args.sample_id:
            print("Error: --sample_id required, or use --hierarchical_mdm_dir and --sample_idx")
            return
        sample_id = args.sample_id
        generated = None  # load below from data_root/generated/...
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / 'outputs' / sample_id.replace('/', '_')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Piano Motion Video Creator (26-joint skeleton)")
    print("=" * 70)
    print(f"Sample: {sample_id}")
    print(f"Output: {output_dir}")
    print(f"Video FPS: {args.video_fps}, Render FPS: {args.render_fps}")
    
    # Calculate max_frames from duration if specified
    if args.duration:
        args.max_frames = int(args.duration * FPS)
        print(f"Duration: {args.duration}s = {args.max_frames} frames (at {FPS} FPS)")
    
    # Create video creator
    creator = PianoMotionVideoCreator(
        video_fps=args.video_fps,
        render_fps=args.render_fps,
        dpi=args.dpi,
        view_angle=(args.view_elev, args.view_azim),
        n_workers=args.workers,
    )
    
    # GT only mode (only when not using hierarchical_mdm generated)
    if args.gt_only and not use_hierarchical_mdm:
        print("\n[1/1] Creating ground truth visualization...")
        
        ground_truth = load_ground_truth(data_root, sample_id)
        
        if ground_truth is None:
            print(f"\nError: Ground truth not found for {sample_id}")
            print("Please run preprocessing first:")
            print(f"  python src/stage1/s05_preprocess_motion.py --force")
            return
        
        print(f"  Loaded GT: {ground_truth.shape} (T, {ground_truth.shape[1]} joints, 3)")
        
        # Limit frames if specified
        if args.max_frames:
            ground_truth = ground_truth[:args.max_frames]
            print(f"  Limited to {args.max_frames} frames ({args.max_frames/FPS:.2f}s)")
        
        video_path = output_dir / 'gt_motion.mp4'
        creator.create_video(
            joints=ground_truth,
            output_path=video_path,
            title=f"GT - {sample_id}",
        )
        
        print("\n" + "=" * 70)
        print("Video creation complete!")
        print(f"Output: {video_path}")
        print("=" * 70)
        return
    
    # Load generated results (skip if already loaded from hierarchical_mdm)
    if not use_hierarchical_mdm:
        generated_dir = data_root / 'generated' / sample_id
        full_body_path = generated_dir / 'generated_motion.npy'
        if not full_body_path.exists():
            full_body_path = generated_dir / 'full_body.npy'
        if not full_body_path.exists():
            print(f"\nError: generated motion not found in {generated_dir}")
            print("Please run inference first:")
            print(f"  python src/stage4/s40_inference.py --sample_id {sample_id}")
            print("\nOr use --hierarchical_mdm_dir and --sample_idx for Stage4 output")
            return
        generated = np.load(full_body_path)
        generated = generated.astype(np.float64) * 1000  # m -> mm
        print(f"\nLoaded generated motion: {generated.shape}")
    
    # Limit frames if specified
    if args.max_frames:
        generated = generated[:args.max_frames]
        print(f"  Limited to {args.max_frames} frames")
    
    # 1. Create comparison video if GT available
    if args.compare_gt:
        print("\n[1/2] Creating GT vs Generated comparison video...")
        print("      (GT: Left, Generated: Right)")
        
        ground_truth = load_ground_truth(data_root, sample_id)
        
        if ground_truth is not None:
            # Limit GT to same length
            if args.max_frames:
                ground_truth = ground_truth[:args.max_frames]
            
            comparison_path = output_dir / 'comparison_gt_vs_generated.mp4'
            creator.create_comparison_video(
                ground_truth=ground_truth,
                generated=generated,
                output_path=comparison_path,
            )
        else:
            print("  Ground truth not found, skipping comparison")
    else:
        print("\n[1/2] Creating generated motion video...")
        video_path = output_dir / 'motion_generated.mp4'
        creator.create_video(
            joints=generated,
            output_path=video_path,
            title=f"Generated - {sample_id}",
        )
    
    # 2. Add audio if requested
    if args.with_audio:
        print("\n[2/2] Adding audio...")
        parts = sample_id.split('/')
        piece, performer, trial = parts[1], parts[2], parts[3]
        audio_path = data_root / sample_id / f"{piece}_{performer}_{trial}_audio.wav"
        
        if audio_path.exists():
            # Determine which video to add audio to
            if args.compare_gt:
                input_video = output_dir / 'comparison_gt_vs_generated.mp4'
                output_video = output_dir / 'comparison_gt_vs_generated_with_audio.mp4'
            else:
                input_video = output_dir / 'motion_generated.mp4'
                output_video = output_dir / 'motion_generated_with_audio.mp4'
            
            add_audio_to_video(
                video_path=input_video,
                audio_path=audio_path,
                output_path=output_video,
            )
        else:
            print(f"  Audio not found: {audio_path}")
    else:
        print("\n[2/2] Skipping audio (use --with_audio to enable)")
    
    print("\n" + "=" * 70)
    print("Video creation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()
