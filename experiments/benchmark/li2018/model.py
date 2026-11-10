"""
experiments/benchmark/li2018/model.py

Li et al. ISMIR 2018 "Skeleton Plays Piano" adapted for 3D motion generation.

Original: MIDI + Metric → 8 upper body joints (2D)
Ours: MIDI + Metric → 26 joints (3D)

Architecture:
1. PianoRollCNN: [B, 88, 2τ] → [B, piano_feat_dim]
2. MetricCNN: [B, 3, 2τ] → [B, metric_feat_dim]
3. LSTM: Concatenated features → joint coordinates

Reference: https://arxiv.org/abs/1807.11636
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.benchmark.common.base_model import BaseRegressionModel


class PianoRollCNN(nn.Module):
    """
    CNN for extracting features from piano roll matrix.
    
    Original paper:
    - Input: [128, 60] (MIDI pitch × time window at 30fps)
    - Conv(5×5, 10ch) + BN + LeakyReLU + MaxPool(3×2)
    - Conv(5×5, 20ch) + BN + LeakyReLU + MaxPool(2×2)
    - FC(50)
    
    Our adaptation:
    - Input: [88, 2τ] (piano keys × time window at 60fps)
    - Similar structure, adjusted for our input size
    """
    
    def __init__(
        self,
        input_keys: int = 88,
        window_size: int = 120,  # 2τ at 60fps (±1 second)
        output_dim: int = 50,
    ):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(3, 2)),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        
        # Calculate output size after convolutions
        # After conv1: (88/3, 120/2) ≈ (29, 60)
        # After conv2: (29/2, 60/2) ≈ (14, 30)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_keys, window_size)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            flat_size = dummy.view(1, -1).size(1)
        
        self.fc = nn.Linear(flat_size, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 88, window_size] piano roll
        Returns:
            [B, output_dim] feature vector
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # [B, 1, 88, window_size]
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class MetricCNN(nn.Module):
    """
    CNN for extracting features from metric structure matrix.
    
    Original paper:
    - Input: [3, 60] (downbeat/pickup/other × time window)
    - Conv(3×4, 20ch) + BN + LeakyReLU + MaxPool(3×1)
    - FC(10)
    
    Metric encoding (3 channels):
    - Channel 0: Downbeat (beat 0 in measure)
    - Channel 1: Strong beat (beat 2 in 4/4)
    - Channel 2: Other beats
    """
    
    def __init__(
        self,
        window_size: int = 120,
        output_dim: int = 10,
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=(3, 4), padding=(1, 1)),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(1, 3)),  # Pool only along time
        )
        
        # Calculate output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 3, window_size)
            dummy = self.conv(dummy)
            flat_size = dummy.view(1, -1).size(1)
        
        self.fc = nn.Linear(flat_size, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, window_size] metric matrix
        Returns:
            [B, output_dim] feature vector
        """
        x = x.unsqueeze(1)  # [B, 1, 3, window_size]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Li2018Model(BaseRegressionModel):
    """
    Li et al. ISMIR 2018 "Skeleton Plays Piano" adapted for 3D motion.
    
    Architecture:
    1. PianoRollCNN: Extract features from sliding window piano roll
    2. MetricCNN: Extract features from metric structure (optional)
    3. LSTM: Generate skeleton sequence from concatenated features
    
    Original output: 8 joints × 2D = 16 dimensions
    Our output: 26 joints × 3D = 78 dimensions
    """
    
    def __init__(
        self,
        output_dim: int = 78,  # 26 joints × 3
        window_size: int = 120,  # ±1 second at 60fps
        piano_feat_dim: int = 50,
        metric_feat_dim: int = 10,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        use_metric: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.window_size = window_size
        self.use_metric = use_metric
        self.half_window = window_size // 2
        
        # Feature extractors
        self.piano_cnn = PianoRollCNN(
            input_keys=88,
            window_size=window_size,
            output_dim=piano_feat_dim,
        )
        
        if use_metric:
            self.metric_cnn = MetricCNN(
                window_size=window_size,
                output_dim=metric_feat_dim,
            )
            lstm_input_dim = piano_feat_dim + metric_feat_dim
        else:
            self.metric_cnn = None
            lstm_input_dim = piano_feat_dim
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,  # Online generation (unidirectional)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    
    def _extract_window(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Extract a window centered at frame t with padding."""
        B, C, T = x.shape
        
        start = t - self.half_window
        end = t + self.half_window
        
        # Pad if necessary
        if start < 0 or end > T:
            pad_left = max(0, -start)
            pad_right = max(0, end - T)
            
            # Use replicate padding (works for any padding size)
            # For 3D input [B, C, T], pad format is (left, right)
            x_padded = F.pad(x, (pad_left, pad_right), mode='replicate')
            start = start + pad_left
            end = end + pad_left
        else:
            x_padded = x
        
        return x_padded[:, :, start:end]
    
    def forward(
        self,
        midi_features: Dict[str, torch.Tensor],
        metric_matrix: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate motion sequence from MIDI and metric features.
        
        Args:
            midi_features: {
                'piano_roll': [B, T, 88],
                'velocity': [B, T, 88],
                'onset': [B, T, 88],
            }
            metric_matrix: [B, T, 3] - downbeat, strong beat, other
        
        Returns:
            motion: [B, T, 78]
        """
        # Get piano roll (use onset-weighted velocity)
        piano_roll = midi_features['piano_roll']  # [B, T, 88]
        if 'velocity' in midi_features:
            piano_roll = piano_roll * midi_features['velocity']
        
        B, T, _ = piano_roll.shape
        device = piano_roll.device
        
        # Transpose for CNN: [B, T, 88] → [B, 88, T]
        piano_roll = piano_roll.transpose(1, 2)
        
        if self.use_metric and metric_matrix is not None:
            # [B, T, 3] → [B, 3, T]
            metric_matrix = metric_matrix.transpose(1, 2)
        
        # Extract features for each frame
        frame_features = []
        
        for t in range(T):
            # Extract sliding window for piano roll
            piano_window = self._extract_window(piano_roll, t)  # [B, 88, window_size]
            piano_feat = self.piano_cnn(piano_window)  # [B, piano_feat_dim]
            
            if self.use_metric and metric_matrix is not None:
                # Extract sliding window for metric
                metric_window = self._extract_window(metric_matrix, t)  # [B, 3, window_size]
                metric_feat = self.metric_cnn(metric_window)  # [B, metric_feat_dim]
                feat = torch.cat([piano_feat, metric_feat], dim=-1)
            else:
                feat = piano_feat
            
            frame_features.append(feat)
        
        # Stack features: [B, T, feat_dim]
        features = torch.stack(frame_features, dim=1)
        
        # LSTM
        lstm_out, _ = self.lstm(features)  # [B, T, lstm_hidden]
        
        # Output projection
        motion = self.output_proj(lstm_out)  # [B, T, 78]
        
        return motion
    
    def forward_efficient(
        self,
        midi_features: Dict[str, torch.Tensor],
        metric_matrix: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        More efficient forward pass using strided convolution over time.
        Used during inference for long sequences.
        """
        # For now, use the standard forward
        # TODO: Implement efficient batched window extraction
        return self.forward(midi_features, metric_matrix, **kwargs)
    
    def generate(
        self,
        midi_features: Dict[str, torch.Tensor],
        metric_matrix: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Inference (same as forward for regression model)."""
        motion = self.forward(midi_features, metric_matrix, **kwargs)
        # Reshape to [B, T, 26, 3]
        B, T, _ = motion.shape
        return motion.view(B, T, 26, 3)


# ==============================================================================
# Loss Functions
# ==============================================================================

# Limb pairs for skeleton consistency (based on 26-joint structure)
# Format: (joint_i, joint_j) for limb connections
LIMB_PAIRS = [
    # Spine
    (0, 1),   # pelvis - spine1
    (1, 2),   # spine1 - spine2
    (2, 3),   # spine2 - spine3
    (3, 4),   # spine3 - neck
    (4, 5),   # neck - head
    # Left arm
    (3, 6),   # spine3 - left_shoulder
    (6, 7),   # left_shoulder - left_elbow
    (7, 8),   # left_elbow - left_wrist
    # Right arm
    (3, 9),   # spine3 - right_shoulder
    (9, 10),  # right_shoulder - right_elbow
    (10, 11), # right_elbow - right_wrist
    # Left leg
    (0, 18),  # pelvis - left_hip
    (18, 19), # left_hip - left_knee
    (19, 20), # left_knee - left_ankle
    # Right leg
    (0, 22),  # pelvis - right_hip
    (22, 23), # right_hip - right_knee
    (23, 24), # right_knee - right_ankle
    # Hands (if available)
    (8, 12),  # left_wrist - left_index
    (11, 15), # right_wrist - right_index
]


def compute_limb_lengths(joints: torch.Tensor, limb_pairs: list) -> torch.Tensor:
    """
    Compute limb lengths from joint positions.
    
    Args:
        joints: [B, T, 26, 3] or [B, T, 78]
    
    Returns:
        limb_lengths: [B, T, num_limbs]
    """
    if joints.dim() == 3:
        B, T, D = joints.shape
        joints = joints.view(B, T, 26, 3)
    
    limb_lengths = []
    for i, j in limb_pairs:
        if i < joints.shape[2] and j < joints.shape[2]:
            diff = joints[:, :, i] - joints[:, :, j]  # [B, T, 3]
            length = torch.norm(diff, dim=-1)  # [B, T]
            limb_lengths.append(length)
    
    return torch.stack(limb_lengths, dim=-1)  # [B, T, num_limbs]


def compute_li2018_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    stage: str = 'joint',
    limb_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Li2018 two-stage loss function.
    
    Stage 1 (joint): L1 loss on joint positions only
    Stage 2 (limb): L1 loss on joints + limb length consistency
    
    Args:
        pred: [B, T, 78] predicted motion
        gt: [B, T, 78] ground truth motion
        stage: 'joint' or 'limb'
        limb_weight: Weight for limb consistency loss
    
    Returns:
        dict with 'loss', 'joint_loss', 'limb_loss'
    """
    # Joint constraint (L1 loss)
    joint_loss = F.l1_loss(pred, gt)
    
    if stage == 'joint':
        return {
            'loss': joint_loss,
            'joint_loss': joint_loss,
            'limb_loss': torch.tensor(0.0, device=pred.device),
        }
    
    # Limb constraint
    pred_limbs = compute_limb_lengths(pred, LIMB_PAIRS)
    gt_limbs = compute_limb_lengths(gt, LIMB_PAIRS)
    limb_loss = F.l1_loss(pred_limbs, gt_limbs)
    
    total_loss = joint_loss + limb_weight * limb_loss
    
    return {
        'loss': total_loss,
        'joint_loss': joint_loss,
        'limb_loss': limb_loss,
    }


if __name__ == '__main__':
    # Test
    print("=== Li2018 Model Test ===")
    
    model = Li2018Model(use_metric=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    B, T = 2, 60
    midi = {
        'piano_roll': torch.randn(B, T, 88),
        'velocity': torch.rand(B, T, 88),
        'onset': torch.zeros(B, T, 88),
    }
    metric = torch.zeros(B, T, 3)
    # Set some random beats
    metric[:, ::15, 0] = 1  # downbeat every 15 frames
    metric[:, 7::15, 1] = 1  # strong beat
    
    # Forward
    motion = model(midi, metric)
    print(f"Output shape: {motion.shape}")
    
    # Generate
    motion_3d = model.generate(midi, metric)
    print(f"Generate shape: {motion_3d.shape}")
    
    # Loss
    gt = torch.randn(B, T, 78)
    losses_joint = compute_li2018_loss(motion, gt, stage='joint')
    losses_limb = compute_li2018_loss(motion, gt, stage='limb')
    print(f"Joint loss: {losses_joint['loss']:.4f}")
    print(f"Limb loss: {losses_limb['loss']:.4f} (joint: {losses_limb['joint_loss']:.4f}, limb: {losses_limb['limb_loss']:.4f})")
    
    print("\nOK")
