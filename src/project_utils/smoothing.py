"""
Motion Smoothing Utilities

Provides zero-phase, edge-preserving smoothing filters for motion data.
"""

import numpy as np
from typing import Optional


class OneEuroFilter:
    """
    1€ Filter: Adaptive smoothing based on motion speed.
    
    Key properties:
    - Fast motion → less smoothing (preserve responsiveness)
    - Slow motion → more smoothing (reduce jitter)
    - Zero phase when applied bidirectionally (offline)
    
    Reference: 
        Casiez et al. "1€ Filter: A Simple Speed-based Low-pass Filter 
        for Noisy Input in Interactive Systems" CHI 2012
    
    Usage:
        filter = OneEuroFilter(min_cutoff=1.5, beta=0.007)
        smoothed = filter.filter_offline(motion, fps=60)
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.5,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        """
        Args:
            min_cutoff: Minimum cutoff frequency (Hz). 
                        Lower = smoother. Range: 0.5 ~ 3.0
                        - 0.5: Very smooth (may lose detail)
                        - 1.5: Balanced (recommended)
                        - 3.0: Less smoothing
            beta: Speed coefficient. Higher = more responsive to fast motion.
                  Range: 0.001 ~ 0.1
                  - 0.001: Uniform smoothing regardless of speed
                  - 0.007: Balanced (recommended)
                  - 0.05: Very adaptive
            d_cutoff: Derivative (velocity) cutoff frequency (Hz).
                      Usually keep at 1.0.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
    
    def _compute_alpha(self, cutoff: float, fps: float) -> float:
        """Compute smoothing factor alpha from cutoff frequency."""
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / fps
        return 1.0 / (1.0 + tau / te)
    
    def _filter_pass(self, data: np.ndarray, fps: float) -> np.ndarray:
        """
        Single pass of 1€ filter (causal, has phase delay).
        
        Args:
            data: (T,) 1D signal
            fps: Frame rate
        
        Returns:
            Filtered signal (T,)
        """
        T = len(data)
        result = np.zeros(T, dtype=np.float64)
        result[0] = data[0]
        
        dx_prev = 0.0
        x_prev = float(data[0])
        
        alpha_d = self._compute_alpha(self.d_cutoff, fps)
        
        for i in range(1, T):
            x = float(data[i])
            
            # Compute derivative (velocity)
            dx = (x - x_prev) * fps
            
            # Low-pass filter the derivative
            dx_hat = alpha_d * dx + (1 - alpha_d) * dx_prev
            
            # Adaptive cutoff: faster motion → higher cutoff → less smoothing
            speed = abs(dx_hat)
            cutoff = self.min_cutoff + self.beta * speed
            
            # Compute alpha for this frame
            alpha = self._compute_alpha(cutoff, fps)
            
            # Low-pass filter the position
            x_hat = alpha * x + (1 - alpha) * x_prev
            
            result[i] = x_hat
            x_prev = x_hat
            dx_prev = dx_hat
        
        return result
    
    def filter_1d(self, data: np.ndarray, fps: float = 60.0) -> np.ndarray:
        """
        Apply 1€ filter bidirectionally for zero-phase.
        
        Bidirectional filtering:
        1. Forward pass: start → end
        2. Backward pass: end → start (on reversed signal)
        3. Average both results
        
        This eliminates phase delay completely.
        
        Args:
            data: (T,) 1D signal
            fps: Frame rate
        
        Returns:
            Filtered signal (T,), zero phase delay
        """
        # Forward pass
        forward = self._filter_pass(data, fps)
        
        # Backward pass (reverse, filter, reverse back)
        backward = self._filter_pass(data[::-1], fps)[::-1]
        
        # Average for zero-phase
        return (forward + backward) / 2
    
    def filter_offline(
        self,
        data: np.ndarray,
        fps: float = 60.0,
    ) -> np.ndarray:
        """
        Apply 1€ filter to multi-dimensional motion data (offline, zero-phase).
        
        Args:
            data: Motion data with shape:
                  - (T,) for 1D
                  - (T, D) for multi-dimensional
                  - (T, J, 3) for joint positions
            fps: Frame rate
        
        Returns:
            Filtered data with same shape
        """
        original_shape = data.shape
        
        if data.ndim == 1:
            return self.filter_1d(data, fps)
        
        # Flatten to (T, N) where N = product of other dims
        T = data.shape[0]
        data_flat = data.reshape(T, -1)
        N = data_flat.shape[1]
        
        result = np.zeros_like(data_flat)
        
        for i in range(N):
            result[:, i] = self.filter_1d(data_flat[:, i], fps)
        
        return result.reshape(original_shape)


def smooth_motion(
    motion: np.ndarray,
    fps: float = 60.0,
    min_cutoff: float = 1.5,
    beta: float = 0.007,
) -> np.ndarray:
    """
    Convenience function to smooth motion data using 1€ filter.
    
    Args:
        motion: (T, J, 3) or (T, D) motion data
        fps: Frame rate
        min_cutoff: Minimum cutoff frequency (lower = smoother)
        beta: Speed coefficient (higher = more adaptive)
    
    Returns:
        Smoothed motion with same shape
    
    Example:
        >>> motion = np.random.randn(1000, 26, 3)  # 26 joints
        >>> smoothed = smooth_motion(motion, fps=60, min_cutoff=1.5)
    """
    filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    return filter.filter_offline(motion, fps=fps)


def smooth_motion_preserve_constraints(
    motion: np.ndarray,
    fixed_joint_indices: list = None,
    fps: float = 60.0,
    min_cutoff: float = 1.5,
    beta: float = 0.007,
) -> np.ndarray:
    """
    Smooth motion while preserving certain joints (hard constraints).
    
    Args:
        motion: (T, J, 3) motion data
        fixed_joint_indices: List of joint indices to NOT smooth
        fps: Frame rate
        min_cutoff: Minimum cutoff frequency
        beta: Speed coefficient
    
    Returns:
        Smoothed motion with fixed joints unchanged
    """
    if fixed_joint_indices is None:
        fixed_joint_indices = []
    
    result = motion.copy()
    filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    
    for j in range(motion.shape[1]):
        if j in fixed_joint_indices:
            # Keep original
            continue
        
        for d in range(3):
            result[:, j, d] = filter.filter_1d(motion[:, j, d], fps)
    
    return result


# Presets for different use cases
SMOOTHING_PRESETS = {
    # Minimal smoothing - only remove high-frequency jitter
    'minimal': {'min_cutoff': 3.0, 'beta': 0.01},
    
    # Balanced - good for most cases (recommended)
    'balanced': {'min_cutoff': 1.5, 'beta': 0.007},
    
    # Strong - aggressive smoothing for very noisy data
    'strong': {'min_cutoff': 0.8, 'beta': 0.005},
    
    # Adaptive - very responsive to motion speed
    'adaptive': {'min_cutoff': 1.0, 'beta': 0.05},
    
    # Piano - preserve fast intentional motion, filter jitter
    # For wrist/finger joints in piano playing
    # min_cutoff=5Hz: filter 5Hz+ jitter while preserving slower intentional motion
    'piano': {'min_cutoff': 5.0, 'beta': 0.05},
}


def smooth_motion_preset(
    motion: np.ndarray,
    preset: str = 'balanced',
    fps: float = 60.0,
) -> np.ndarray:
    """
    Smooth motion using a preset configuration.
    
    Args:
        motion: (T, J, 3) motion data
        preset: One of 'minimal', 'balanced', 'strong', 'adaptive'
        fps: Frame rate
    
    Returns:
        Smoothed motion
    """
    if preset not in SMOOTHING_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(SMOOTHING_PRESETS.keys())}")
    
    params = SMOOTHING_PRESETS[preset]
    return smooth_motion(motion, fps=fps, **params)


# ============================================================
# BONE LENGTH NORMALIZATION
# ============================================================

# Full body skeleton bones (26 joints)
# Parent → Child ordering for forward kinematics
SKELETON_BONES = [
    # Head connections
    (0, 1),   # RFHD → LFHD
    (2, 3),   # RBHD → LBHD
    (0, 2),   # RFHD → RBHD
    (1, 3),   # LFHD → LBHD
    # Head → Neck
    (2, 4),   # RBHD → C7
    (3, 4),   # LBHD → C7
    # Spine
    (4, 6),   # C7 → T10
    (5, 7),   # CLAV → STRN
    # Shoulders
    (4, 8),   # C7 → RSHO
    (4, 12),  # C7 → LSHO
    (5, 8),   # CLAV → RSHO
    (5, 12),  # CLAV → LSHO
    # Right arm
    (8, 9),   # RSHO → RELB
    (9, 10),  # RELB → RWRIST
    (10, 11), # RWRIST → RHAND
    # Left arm
    (12, 13), # LSHO → LELB
    (13, 14), # LELB → LWRIST
    (14, 15), # LWRIST → LHAND
    # Torso → Pelvis
    (6, 18),  # T10 → RPSI
    (6, 19),  # T10 → LPSI
    (7, 16),  # STRN → RASI
    (7, 17),  # STRN → LASI
    # Pelvis box
    (16, 17), # RASI → LASI
    (18, 19), # RPSI → LPSI
    (16, 18), # RASI → RPSI
    (17, 19), # LASI → LPSI
    # Right leg
    (16, 20), # RASI → RKNE
    (18, 20), # RPSI → RKNE
    (20, 21), # RKNE → RANKLE
    (21, 22), # RANKLE → RTOE
    # Left leg
    (17, 23), # LASI → LKNE
    (19, 23), # LPSI → LKNE
    (23, 24), # LKNE → LANKLE
    (24, 25), # LANKLE → LTOE
]

# Critical bones for normalization (limbs that can stretch badly)
CRITICAL_BONES = [
    # Arms
    (8, 9),   # RSHO → RELB
    (9, 10),  # RELB → RWRIST
    (10, 11), # RWRIST → RHAND
    (12, 13), # LSHO → LELB
    (13, 14), # LELB → LWRIST
    (14, 15), # LWRIST → LHAND
    # Legs
    (16, 20), # RASI → RKNE
    (20, 21), # RKNE → RANKLE
    (21, 22), # RANKLE → RTOE
    (17, 23), # LASI → LKNE
    (23, 24), # LKNE → LANKLE
    (24, 25), # LANKLE → LTOE
]


def compute_bone_lengths(motion: np.ndarray, bones: list = None) -> dict:
    """
    Compute bone lengths from motion data.
    
    Args:
        motion: (T, J, 3) motion data
        bones: List of (parent_idx, child_idx) tuples. 
               If None, uses SKELETON_BONES.
    
    Returns:
        Dict mapping (parent, child) to mean bone length
    """
    if bones is None:
        bones = SKELETON_BONES
    
    bone_lengths = {}
    for parent, child in bones:
        lengths = np.linalg.norm(motion[:, child] - motion[:, parent], axis=-1)
        bone_lengths[(parent, child)] = float(np.median(lengths))  # Use median for robustness
    
    return bone_lengths


def normalize_bone_lengths(
    motion: np.ndarray,
    target_lengths: dict,
    bones: list = None,
    iterations: int = 3,
) -> np.ndarray:
    """
    Normalize bone lengths to target lengths using iterative projection.
    
    This uses a simple iterative approach:
    1. For each bone, move child to correct distance from parent
    2. Repeat for multiple iterations to propagate through kinematic chain
    
    Args:
        motion: (T, J, 3) motion data
        target_lengths: Dict mapping (parent, child) to target bone length
        bones: List of bones to normalize. If None, uses keys from target_lengths.
        iterations: Number of projection iterations
    
    Returns:
        Motion with normalized bone lengths
    """
    result = motion.copy()
    
    if bones is None:
        bones = list(target_lengths.keys())
    
    for _ in range(iterations):
        for parent, child in bones:
            if (parent, child) not in target_lengths:
                continue
            
            target_len = target_lengths[(parent, child)]
            
            # Current bone vector
            bone_vec = result[:, child] - result[:, parent]
            current_len = np.linalg.norm(bone_vec, axis=-1, keepdims=True)
            
            # Avoid division by zero
            current_len = np.maximum(current_len, 1e-8)
            
            # Normalize and scale to target length
            direction = bone_vec / current_len
            result[:, child] = result[:, parent] + direction * target_len
    
    return result


def normalize_bone_lengths_from_reference(
    motion: np.ndarray,
    reference_motion: np.ndarray,
    bones: list = None,
    iterations: int = 3,
) -> np.ndarray:
    """
    Normalize bone lengths to match a reference motion (e.g., GT).
    
    Args:
        motion: (T, J, 3) generated motion
        reference_motion: (T', J, 3) reference motion (e.g., GT)
        bones: List of bones to normalize. If None, uses CRITICAL_BONES.
        iterations: Number of projection iterations
    
    Returns:
        Motion with bone lengths matching reference
    """
    if bones is None:
        bones = CRITICAL_BONES
    
    # Compute target lengths from reference
    target_lengths = compute_bone_lengths(reference_motion, bones)
    
    return normalize_bone_lengths(motion, target_lengths, bones, iterations)


# Default bone lengths for 26-joint skeleton (in meters)
# Computed from MOSA dataset averages
DEFAULT_BONE_LENGTHS = {
    # Arms
    (8, 9): 0.28,    # RSHO → RELB
    (9, 10): 0.25,   # RELB → RWRIST
    (10, 11): 0.08,  # RWRIST → RHAND
    (12, 13): 0.28,  # LSHO → LELB
    (13, 14): 0.25,  # LELB → LWRIST
    (14, 15): 0.08,  # LWRIST → LHAND
    # Legs
    (16, 20): 0.40,  # RASI → RKNE
    (20, 21): 0.38,  # RKNE → RANKLE
    (21, 22): 0.16,  # RANKLE → RTOE
    (17, 23): 0.40,  # LASI → LKNE
    (23, 24): 0.38,  # LKNE → LANKLE
    (24, 25): 0.14,  # LANKLE → LTOE
}


def filter_velocity_spikes(
    motion: np.ndarray,
    max_speed_mm: float = 5.0,
    joint_indices: Optional[list] = None,
    fps: float = 60.0,
) -> np.ndarray:
    """
    Filter velocity spikes by clamping and interpolating.
    
    When a joint moves faster than max_speed_mm per frame, the position
    is adjusted using interpolation from neighboring frames.
    
    Args:
        motion: (T, J, 3) motion array
        max_speed_mm: Maximum allowed speed in mm/frame
        joint_indices: List of joint indices to filter (None = all)
        fps: Frame rate (used for logging)
    
    Returns:
        (T, J, 3) filtered motion
    """
    motion = motion.copy()
    T, J, _ = motion.shape
    max_speed = max_speed_mm / 1000.0  # Convert to meters
    
    if joint_indices is None:
        joint_indices = list(range(J))
    
    total_fixes = 0
    
    for j in joint_indices:
        joint_pos = motion[:, j, :]  # (T, 3)
        
        # Calculate speed
        velocity = np.diff(joint_pos, axis=0)  # (T-1, 3)
        speed = np.linalg.norm(velocity, axis=1)  # (T-1,)
        
        # Find spike frames
        spike_mask = speed > max_speed
        spike_frames = np.where(spike_mask)[0] + 1  # +1 because diff shifts by 1
        
        if len(spike_frames) == 0:
            continue
        
        total_fixes += len(spike_frames)
        
        # Fix each spike by interpolation
        for frame in spike_frames:
            # Find nearest non-spike frames before and after
            before = frame - 1
            while before > 0 and (before in spike_frames):
                before -= 1
            
            after = frame + 1
            while after < T - 1 and (after in spike_frames):
                after += 1
            
            if before < 0:
                before = 0
            if after >= T:
                after = T - 1
            
            # Linear interpolation
            if after > before:
                alpha = (frame - before) / (after - before)
                motion[frame, j, :] = (1 - alpha) * motion[before, j, :] + alpha * motion[after, j, :]
    
    if total_fixes > 0:
        print(f"  Fixed {total_fixes} velocity spikes (>{max_speed_mm}mm/frame)")
    
    return motion
