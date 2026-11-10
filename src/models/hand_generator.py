"""
Hand Generator Network

Generates hand positions (wrist + finger) directly from ARIA embedding and MIDI notes.
No rule-based prior or hand assignment heuristics needed.

Input:
- ARIA embedding (B, 512): global music style/context
- Active MIDI keys (B, T, 88): per-frame active keys (multi-hot)

Output:
- Hand positions (B, T, 12): 
    - left_wrist(3) + left_finger(3) + right_wrist(3) + right_finger(3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MidiEncoder(nn.Module):
    """Encode per-frame MIDI active keys"""
    
    def __init__(self, num_keys: int = 88, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_keys, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(self, active_keys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            active_keys: (B, T, 88) - multi-hot active keys per frame
        Returns:
            (B, T, hidden_dim)
        """
        return self.encoder(active_keys)


class AriaEncoder(nn.Module):
    """Encode global ARIA embedding and broadcast to sequence"""
    
    def __init__(self, aria_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(aria_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(self, aria_embed: torch.Tensor, T: int) -> torch.Tensor:
        """
        Args:
            aria_embed: (B, 512) global ARIA embedding
            T: sequence length to broadcast to
        Returns:
            (B, T, hidden_dim)
        """
        feat = self.encoder(aria_embed)  # (B, hidden_dim)
        return feat.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden_dim)


class HandGeneratorNet(nn.Module):
    """
    Hand Generator Network
    
    Generates hand positions (wrist + finger) directly from MIDI and ARIA embedding.
    
    Architecture:
    1. MIDI encoder: active_keys (88) -> hidden
    2. ARIA encoder: global embedding -> hidden (broadcast)
    3. Fusion: concat + project
    4. Temporal Transformer: model temporal dependencies
    5. Output head: -> (left_wrist, left_finger, right_wrist, right_finger)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        aria_dim: int = 512,
        num_keys: int = 88,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # MIDI encoder (per-frame)
        self.midi_encoder = MidiEncoder(num_keys, hidden_dim)
        
        # ARIA encoder (global)
        self.aria_encoder = AriaEncoder(aria_dim, hidden_dim)
        
        # Fusion layer (concat MIDI + ARIA -> hidden)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: -> 12 (left_wrist + left_finger + right_wrist + right_finger)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 12),
        )
        
        # Initialize output to reasonable default
        self._init_output_head()
    
    def _init_output_head(self):
        """Initialize output head for reasonable starting positions"""
        # Last linear layer
        nn.init.zeros_(self.output_head[-1].weight)
        # Bias: approximate hand positions in front of piano (in meters)
        # Finger is slightly forward (+z) and down (-y) from wrist
        with torch.no_grad():
            self.output_head[-1].bias.data = torch.tensor([
                -0.2, 0.9, 0.5,   # left wrist (x, y, z)
                -0.2, 0.85, 0.6,  # left finger (x, y, z)
                 0.2, 0.9, 0.5,   # right wrist (x, y, z)
                 0.2, 0.85, 0.6,  # right finger (x, y, z)
            ])
    
    def forward(
        self,
        active_keys: torch.Tensor,
        aria_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            active_keys: (B, T, 88) - multi-hot active keys per frame
            aria_embed: (B, 512) - ARIA MIDI embedding
            
        Returns:
            hand: (B, T, 12) - [left_wrist, left_finger, right_wrist, right_finger]
        """
        B, T, _ = active_keys.shape
        
        # Encode MIDI (per-frame)
        midi_feat = self.midi_encoder(active_keys)  # (B, T, hidden)
        
        # Encode ARIA (broadcast to all frames)
        aria_feat = self.aria_encoder(aria_embed, T)  # (B, T, hidden)
        
        # Fuse features
        fused = torch.cat([midi_feat, aria_feat], dim=-1)  # (B, T, hidden*2)
        x = self.fusion(fused)  # (B, T, hidden)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Temporal modeling
        x = self.transformer(x)  # (B, T, hidden)
        
        # Output hand positions
        hand = self.output_head(x)  # (B, T, 12)
        
        return hand
    
    def get_left_wrist(self, hand: torch.Tensor) -> torch.Tensor:
        """Extract left wrist from output"""
        return hand[..., 0:3]
    
    def get_left_finger(self, hand: torch.Tensor) -> torch.Tensor:
        """Extract left finger from output"""
        return hand[..., 3:6]
    
    def get_right_wrist(self, hand: torch.Tensor) -> torch.Tensor:
        """Extract right wrist from output"""
        return hand[..., 6:9]
    
    def get_right_finger(self, hand: torch.Tensor) -> torch.Tensor:
        """Extract right finger from output"""
        return hand[..., 9:12]
    
    def get_left_hand(self, hand: torch.Tensor) -> torch.Tensor:
        """Extract left hand (wrist + finger) from output"""
        return hand[..., 0:6]
    
    def get_right_hand(self, hand: torch.Tensor) -> torch.Tensor:
        """Extract right hand (wrist + finger) from output"""
        return hand[..., 6:12]


class HandGeneratorLoss(nn.Module):
    """
    Loss function for hand generator training
    
    Components:
    - Position loss: L1 distance to ground truth
    - Velocity loss: smooth motion
    - Acceleration loss: avoid jerk
    - Bone length loss: maintain wrist-finger distance
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        vel_weight: float = 0.5,
        acc_weight: float = 0.1,
        bone_weight: float = 0.2,
        expected_wrist_finger_dist: float = 0.15,  # ~15cm wrist to fingertip
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.bone_weight = bone_weight
        self.expected_dist = expected_wrist_finger_dist
    
    def forward(
        self,
        pred_hand: torch.Tensor,
        gt_hand: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred_hand: (B, T, 12) - predicted [left_wrist, left_finger, right_wrist, right_finger]
            gt_hand: (B, T, 12) - ground truth
        
        Returns:
            total_loss: scalar
            loss_dict: breakdown of losses
        """
        # Position loss
        pos_loss = F.l1_loss(pred_hand, gt_hand)
        
        # Velocity loss
        pred_vel = pred_hand[:, 1:] - pred_hand[:, :-1]
        gt_vel = gt_hand[:, 1:] - gt_hand[:, :-1]
        vel_loss = F.l1_loss(pred_vel, gt_vel)
        
        # Acceleration loss
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        acc_loss = F.l1_loss(pred_acc, gt_acc)
        
        # Bone length loss (wrist-finger distance should be consistent)
        # Left hand
        left_wrist = pred_hand[..., 0:3]
        left_finger = pred_hand[..., 3:6]
        left_dist = torch.norm(left_finger - left_wrist, dim=-1)
        
        # Right hand
        right_wrist = pred_hand[..., 6:9]
        right_finger = pred_hand[..., 9:12]
        right_dist = torch.norm(right_finger - right_wrist, dim=-1)
        
        # Penalize deviation from expected distance
        bone_loss = (
            F.mse_loss(left_dist, torch.full_like(left_dist, self.expected_dist)) +
            F.mse_loss(right_dist, torch.full_like(right_dist, self.expected_dist))
        ) / 2
        
        # Total
        total_loss = (
            self.pos_weight * pos_loss +
            self.vel_weight * vel_loss +
            self.acc_weight * acc_loss +
            self.bone_weight * bone_loss
        )
        
        loss_dict = {
            'pos': pos_loss.item(),
            'vel': vel_loss.item(),
            'acc': acc_loss.item(),
            'bone': bone_loss.item(),
            'total': total_loss.item(),
        }
        
        return total_loss, loss_dict
