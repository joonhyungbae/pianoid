"""
Stage 1 hand (wrist) generation model.

Paper Stage 1 uses HandLSTMRegression (BiLSTM, h512 l6, v3 loss).
Loss variants v1/v2/v3 are kept for the Table 3(C) ablation row.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def concat_midi_features(midi_features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate MIDI features [B, T, 264] or [B, T, 352] with dur_ioi_ratio."""
    parts = [
        midi_features['piano_roll'],   # [B, T, 88]
        midi_features['velocity'],     # [B, T, 88]
        midi_features['onset'],        # [B, T, 88]
    ]
    if 'dur_ioi_ratio' in midi_features:
        parts.append(midi_features['dur_ioi_ratio'])  # [B, T, 88]
    return torch.cat(parts, dim=-1)


# ============================================================================
# HandLSTMRegression — paper Stage 1 wrist model
# ============================================================================

class HandLSTMRegression(nn.Module):
    """
    BiLSTM-based regression for wrist generation.

    Architecture: MIDI features -> input projection -> BiLSTM -> wrist positions.

    Args:
        hand_dim: Output wrist dimension (6 = lwrist xyz + rwrist xyz)
        midi_dim: Input MIDI dimension (264 = 88*3, or 352 with dur_ioi_ratio)
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Use bidirectional LSTM (default True for paper Stage 1)
    """

    def __init__(
        self,
        hand_dim: int = 6,
        midi_dim: int = 264,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.hand_dim = hand_dim
        self.midi_dim = midi_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.input_proj = nn.Linear(midi_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hand_dim),
        )

        self._init_weights()
        self._init_output_bias()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _init_output_bias(self):
        """Initialize output bias to a reasonable resting wrist position."""
        default_hand = torch.tensor([
            -0.2, 0.85, 0.75,  # left wrist
            0.2, 0.85, 0.75,   # right wrist
        ])
        with torch.no_grad():
            self.output_proj[-1].bias.copy_(default_hand)

    def forward(
        self,
        midi_features: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            midi_features: {piano_roll, velocity, onset[, dur_ioi_ratio]} each [B, T, 88]

        Returns:
            hand: [B, T, hand_dim]
        """
        x = concat_midi_features(midi_features)
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        return self.output_proj(h)


# ============================================================================
# Loss Functions — Table 3(C) ablation v1/v2/v3
# ============================================================================

# Z-axis indices in hand 6-dim: lwrist(0,1,2), rwrist(3,4,5)
HAND_Z_INDICES = (2, 5)


def compute_hand_regression_loss(
    hand_pred: torch.Tensor,
    hand_gt: torch.Tensor,
    onset: Optional[torch.Tensor] = None,
    vel_weight: float = 0.5,
    onset_weight: float = 2.0,
    z_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """v1 loss: MSE position + velocity, with optional onset and Z-axis up-weighting."""
    device = hand_pred.device
    axis_w = torch.ones(1, 1, hand_pred.shape[-1], device=device, dtype=hand_pred.dtype)
    for i in HAND_Z_INDICES:
        axis_w[:, :, i] = z_weight
    sq = (hand_pred - hand_gt) ** 2
    if onset is not None and onset_weight > 1.0:
        onset_mask = onset.sum(dim=-1) > 0
        frame_w = torch.ones_like(onset_mask, dtype=torch.float, device=device)
        frame_w[onset_mask] = onset_weight
        frame_w = frame_w.unsqueeze(-1)
        pos_loss = (frame_w * axis_w * sq).sum() / (frame_w * axis_w).sum().clamp(min=1e-8)
    else:
        pos_loss = (axis_w * sq).mean()

    vel_pred = hand_pred[:, 1:] - hand_pred[:, :-1]
    vel_gt = hand_gt[:, 1:] - hand_gt[:, :-1]
    vel_loss = F.mse_loss(vel_pred, vel_gt)

    loss = pos_loss + vel_weight * vel_loss

    return {
        'loss': loss,
        'pos_loss': pos_loss,
        'vel_loss': vel_loss,
    }


def compute_hand_regression_loss_v2(
    hand_pred: torch.Tensor,
    hand_gt: torch.Tensor,
    onset: Optional[torch.Tensor] = None,
    vel_weight: float = 2.0,
    onset_weight: float = 2.0,
    z_weight: float = 1.0,
    speed_weight: float = 1.0,
    use_l1: bool = True,
) -> Dict[str, torch.Tensor]:
    """v2 loss: L1 position + higher velocity weight + speed-magnitude term."""
    device = hand_pred.device
    dim = hand_pred.shape[-1]

    axis_w = torch.ones(1, 1, dim, device=device, dtype=hand_pred.dtype)
    for i in HAND_Z_INDICES:
        axis_w[:, :, i] = z_weight

    if use_l1:
        diff = (hand_pred - hand_gt).abs()
    else:
        diff = (hand_pred - hand_gt) ** 2

    if onset is not None and onset_weight > 1.0:
        onset_mask = onset.sum(dim=-1) > 0
        frame_w = torch.ones_like(onset_mask, dtype=torch.float, device=device)
        frame_w[onset_mask] = onset_weight
        frame_w = frame_w.unsqueeze(-1)
        pos_loss = (frame_w * axis_w * diff).sum() / (frame_w * axis_w).sum().clamp(min=1e-8)
    else:
        pos_loss = (axis_w * diff).mean()

    vel_pred = hand_pred[:, 1:] - hand_pred[:, :-1]
    vel_gt = hand_gt[:, 1:] - hand_gt[:, :-1]
    vel_loss = F.mse_loss(vel_pred, vel_gt)

    speed_pred = torch.norm(vel_pred, dim=-1)
    speed_gt = torch.norm(vel_gt, dim=-1)
    speed_loss = F.mse_loss(speed_pred, speed_gt)

    loss = pos_loss + vel_weight * vel_loss + speed_weight * speed_loss

    return {
        'loss': loss,
        'pos_loss': pos_loss,
        'vel_loss': vel_loss,
        'speed_loss': speed_loss,
    }


def compute_hand_regression_loss_v3(
    hand_pred: torch.Tensor,
    hand_gt: torch.Tensor,
    onset: Optional[torch.Tensor] = None,
    vel_weight: float = 2.0,
    onset_weight: float = 2.0,
    z_weight: float = 1.0,
    speed_weight: float = 1.0,
    z_range_weight: float = 2.0,
    z_std_weight: float = 2.0,
    use_l1: bool = True,
) -> Dict[str, torch.Tensor]:
    """v3 loss (paper Stage 1): v2 + Z-amplitude matching (range, std) for keystroke preservation."""
    device = hand_pred.device
    dim = hand_pred.shape[-1]

    axis_w = torch.ones(1, 1, dim, device=device, dtype=hand_pred.dtype)
    for i in HAND_Z_INDICES:
        axis_w[:, :, i] = z_weight

    if use_l1:
        diff = (hand_pred - hand_gt).abs()
    else:
        diff = (hand_pred - hand_gt) ** 2

    if onset is not None and onset_weight > 1.0:
        onset_mask = onset.sum(dim=-1) > 0
        frame_w = torch.ones_like(onset_mask, dtype=torch.float, device=device)
        frame_w[onset_mask] = onset_weight
        frame_w = frame_w.unsqueeze(-1)
        pos_loss = (frame_w * axis_w * diff).sum() / (frame_w * axis_w).sum().clamp(min=1e-8)
    else:
        pos_loss = (axis_w * diff).mean()

    vel_pred = hand_pred[:, 1:] - hand_pred[:, :-1]
    vel_gt = hand_gt[:, 1:] - hand_gt[:, :-1]
    vel_loss = F.mse_loss(vel_pred, vel_gt)

    speed_pred = torch.norm(vel_pred, dim=-1)
    speed_gt = torch.norm(vel_gt, dim=-1)
    speed_loss = F.mse_loss(speed_pred, speed_gt)

    z_idx = list(HAND_Z_INDICES)
    z_pred = hand_pred[:, :, z_idx]
    z_gt = hand_gt[:, :, z_idx]

    z_range_pred = z_pred.max(dim=1).values - z_pred.min(dim=1).values
    z_range_gt = z_gt.max(dim=1).values - z_gt.min(dim=1).values
    z_range_loss = (z_range_pred - z_range_gt).abs().mean()

    z_std_pred = z_pred.std(dim=1, unbiased=False)
    z_std_gt = z_gt.std(dim=1, unbiased=False)
    z_std_loss = (z_std_pred - z_std_gt).abs().mean()

    loss = (pos_loss
            + vel_weight * vel_loss
            + speed_weight * speed_loss
            + z_range_weight * z_range_loss
            + z_std_weight * z_std_loss)

    return {
        'loss': loss,
        'pos_loss': pos_loss,
        'vel_loss': vel_loss,
        'speed_loss': speed_loss,
        'z_range_loss': z_range_loss,
        'z_std_loss': z_std_loss,
    }


if __name__ == '__main__':
    B, T = 4, 480
    midi_features = {
        'piano_roll': torch.randn(B, T, 88),
        'velocity': torch.randn(B, T, 88),
        'onset': torch.randn(B, T, 88),
    }
    hand_gt = torch.randn(B, T, 6)

    model = HandLSTMRegression(hidden_dim=256, num_layers=2, bidirectional=True)
    out = model(midi_features)
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    losses = compute_hand_regression_loss_v3(out, hand_gt, midi_features['onset'])
    print(f"v3 loss: {losses['loss'].item():.4f}")
