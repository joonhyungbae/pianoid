"""
Torso Generation Baseline Models (non-diffusion).

All models follow the same interface:
    Input: Hand predictions (12-dim) + optional temporal/global conditioning
    Output: Torso positions (12-dim) = [C7, CLAV, T10, STRN] x 3

Models (selected based on Stage 1 hand architecture search top performers):
1. TorsoTransformerRegression - Stage 1 MPJPE rank 1, Attention paradigm
2. TorsoMambaRegression - Stage 1 EqScore rank 1, SSM paradigm
3. TorsoLSTMRegression - Stage 1 rank 3, Recurrent paradigm
4. TorsoFiLMRegression - Frame-wise MLP baseline (context window)
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    Mamba = None


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ============================================================================
# 1. TorsoTransformerRegression
# ============================================================================

class TorsoTransformerRegression(nn.Module):
    """Deterministic Transformer for torso generation."""

    def __init__(
        self,
        hand_dim: int = 6,
        torso_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 2048,
        use_aria: bool = False,
        aria_dim: int = 512,
        temporal_cond_dim: int = 0,
    ):
        super().__init__()
        self.use_aria = use_aria
        self.temporal_cond_dim = temporal_cond_dim

        input_dim = hand_dim + temporal_cond_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_aria:
            self.aria_film = nn.Linear(aria_dim, hidden_dim * 2)

        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, torso_dim),
        )

        self._init_weights()
        self._init_output_bias()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_output_bias(self):
        default_torso = torch.tensor([
            0.0, 0.55, 1.45,   # C7
            0.0, 0.45, 1.35,   # CLAV
            0.0, 0.40, 1.15,   # T10
            0.0, 0.50, 1.10,   # STRN
        ])
        with torch.no_grad():
            self.output_proj[-1].bias.copy_(default_torso)

    def forward(
        self,
        hand: torch.Tensor,
        aria: Optional[torch.Tensor] = None,
        temporal_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hand: (B, T, 12) hand predictions
            aria: (B, 512) optional global music embedding
            temporal_cond: (B, T, temporal_cond_dim) optional temporal features
        Returns:
            torso: (B, T, 12)
        """
        x = hand
        if self.temporal_cond_dim > 0 and temporal_cond is not None:
            x = torch.cat([x, temporal_cond], dim=-1)

        h = self.input_proj(x)

        if self.use_aria and aria is not None:
            film = self.aria_film(aria)
            scale, shift = film.chunk(2, dim=-1)
            h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        h = self.pos_encoder(h)
        h = self.transformer(h)
        return self.output_proj(h)


# ============================================================================
# 2. TorsoLSTMRegression
# ============================================================================

class TorsoLSTMRegression(nn.Module):
    """Bidirectional LSTM for torso (or backbone) generation."""

    # Default positions (meters) for output bias initialization
    # Torso (4 joints): C7, CLAV, T10, STRN
    DEFAULT_TORSO = [
        0.0, 0.55, 1.45,  0.0, 0.45, 1.35,
        0.0, 0.40, 1.15,  0.0, 0.50, 1.10,
    ]
    # Backbone (10 joints): head(4) + torso(4) + shoulders(2)
    DEFAULT_BACKBONE = [
        0.07, 0.18, 1.75,  -0.07, 0.18, 1.75,    # RFHD, LFHD
        0.07, -0.05, 1.75, -0.07, -0.05, 1.75,   # RBHD, LBHD
        0.0, 0.55, 1.45,   0.0, 0.45, 1.35,      # C7, CLAV
        0.0, 0.40, 1.15,   0.0, 0.50, 1.10,      # T10, STRN
        0.18, 0.40, 1.45,  -0.18, 0.40, 1.45,    # RSHO, LSHO
    ]

    def __init__(
        self,
        hand_dim: int = 6,
        torso_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_aria: bool = False,
        aria_dim: int = 512,
        temporal_cond_dim: int = 0,
    ):
        super().__init__()
        self.use_aria = use_aria
        self.temporal_cond_dim = temporal_cond_dim
        self.torso_dim = torso_dim

        input_dim = hand_dim + temporal_cond_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_aria:
            self.aria_film = nn.Linear(aria_dim, hidden_dim * 2)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, torso_dim),
        )

        self._init_output_bias()

    def _init_output_bias(self):
        if self.torso_dim >= 24:  # backbone (24 or 30)
            defaults = self.DEFAULT_BACKBONE
        else:
            defaults = self.DEFAULT_TORSO
        with torch.no_grad():
            self.output_proj[-1].bias.copy_(torch.tensor(defaults[:self.torso_dim]))

    def forward(
        self,
        hand: Optional[torch.Tensor] = None,
        aria: Optional[torch.Tensor] = None,
        temporal_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = []
        if hand is not None:
            parts.append(hand)
        if self.temporal_cond_dim > 0 and temporal_cond is not None:
            parts.append(temporal_cond)
        x = torch.cat(parts, dim=-1)

        h = self.input_proj(x)

        if self.use_aria and aria is not None:
            film = self.aria_film(aria)
            scale, shift = film.chunk(2, dim=-1)
            h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        h, _ = self.lstm(h)
        return self.output_proj(h)


# ============================================================================
# 3. TorsoMambaRegression
# ============================================================================

class TorsoMambaRegression(nn.Module):
    """
    Mamba (Selective State Space) for torso generation.
    Stage 1 EqScore rank 1. Falls back to unidirectional LSTM if mamba-ssm not installed.
    """

    def __init__(
        self,
        hand_dim: int = 6,
        torso_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        dropout: float = 0.1,
        use_aria: bool = False,
        aria_dim: int = 512,
        temporal_cond_dim: int = 0,
    ):
        super().__init__()
        self.use_aria = use_aria
        self.temporal_cond_dim = temporal_cond_dim
        self._use_mamba = HAS_MAMBA

        if not HAS_MAMBA:
            warnings.warn(
                "mamba-ssm not installed. TorsoMambaRegression uses LSTM fallback.",
                UserWarning,
            )

        input_dim = hand_dim + temporal_cond_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_aria:
            self.aria_film = nn.Linear(aria_dim, hidden_dim * 2)

        if self._use_mamba:
            self.layers = nn.ModuleList([
                Mamba(d_model=hidden_dim, d_state=d_state, d_conv=4,
                      expand=2, layer_idx=i)
                for i in range(num_layers)
            ])
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, torso_dim),
        )

        self._init_output_bias()

    def _init_output_bias(self):
        default_torso = torch.tensor([
            0.0, 0.55, 1.45, 0.0, 0.45, 1.35,
            0.0, 0.40, 1.15, 0.0, 0.50, 1.10,
        ])
        with torch.no_grad():
            self.output_proj[-1].bias.copy_(default_torso)

    def forward(
        self,
        hand: Optional[torch.Tensor] = None,
        aria: Optional[torch.Tensor] = None,
        temporal_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = []
        if hand is not None:
            parts.append(hand)
        if self.temporal_cond_dim > 0 and temporal_cond is not None:
            parts.append(temporal_cond)
        x = torch.cat(parts, dim=-1)

        h = self.input_proj(x)

        if self.use_aria and aria is not None:
            film = self.aria_film(aria)
            scale, shift = film.chunk(2, dim=-1)
            h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        if self._use_mamba:
            for layer in self.layers:
                h = h + layer(h)
            h = self.norm(h)
        else:
            h, _ = self.lstm(h)

        return self.output_proj(h)


# ============================================================================
# 4. TorsoFiLMRegression
# ============================================================================

class TorsoFiLMRegression(nn.Module):
    """
    Frame-wise MLP with sliding window context for torso generation.
    Uses a context window to capture local temporal patterns.
    """

    def __init__(
        self,
        hand_dim: int = 6,
        torso_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        context_frames: int = 5,
        use_aria: bool = False,
        aria_dim: int = 512,
        temporal_cond_dim: int = 0,
    ):
        super().__init__()
        self.use_aria = use_aria
        self.temporal_cond_dim = temporal_cond_dim
        self.context_frames = context_frames

        # Context window: 2*context_frames+1 frames of hand + temporal features
        per_frame_dim = hand_dim + temporal_cond_dim
        input_dim = per_frame_dim * (2 * context_frames + 1)

        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, torso_dim))
        self.net = nn.Sequential(*layers)

        if use_aria:
            # FiLM modulation applied after first layer
            self.aria_film = nn.Linear(aria_dim, hidden_dim * 2)

        self._init_output_bias()

    def _init_output_bias(self):
        default_torso = torch.tensor([
            0.0, 0.55, 1.45, 0.0, 0.45, 1.35,
            0.0, 0.40, 1.15, 0.0, 0.50, 1.10,
        ])
        with torch.no_grad():
            self.net[-1].bias.copy_(default_torso)

    def _build_context(self, x: torch.Tensor) -> torch.Tensor:
        """Build context windows: (B, T, D) -> (B, T, D*(2k+1))"""
        B, T, D = x.shape
        k = self.context_frames
        # Pad with edge values
        padded = torch.nn.functional.pad(x, (0, 0, k, k), mode='replicate')
        # Unfold to get context windows
        chunks = []
        for i in range(2 * k + 1):
            chunks.append(padded[:, i:i+T, :])
        return torch.cat(chunks, dim=-1)  # (B, T, D*(2k+1))

    def forward(
        self,
        hand: Optional[torch.Tensor] = None,
        aria: Optional[torch.Tensor] = None,
        temporal_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = []
        if hand is not None:
            parts.append(hand)
        if self.temporal_cond_dim > 0 and temporal_cond is not None:
            parts.append(temporal_cond)
        x = torch.cat(parts, dim=-1)

        x = self._build_context(x)  # (B, T, input_dim)

        if self.use_aria and aria is not None:
            # Apply FiLM after first layer
            h = self.net[0](x)  # Linear
            h = self.net[1](h)  # GELU
            film = self.aria_film(aria)
            scale, shift = film.chunk(2, dim=-1)
            h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            # Continue through remaining layers
            for layer in self.net[2:]:
                h = layer(h)
            return h
        else:
            return self.net(x)


# ============================================================================
# Loss Functions
# ============================================================================
# Torso layout: (B, T, 12) = 4 joints × 3 axes
#   C7(0,1,2), CLAV(3,4,5), T10(6,7,8), STRN(9,10,11)
# Per-axis indices across all joints:
#   x = [0,3,6,9], y = [1,4,7,10], z = [2,5,8,11]

TORSO_X_INDICES = [0, 3, 6, 9]
TORSO_Y_INDICES = [1, 4, 7, 10]
TORSO_Z_INDICES = [2, 5, 8, 11]


TORSO_BONE_PAIRS = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2), (0, 3)]


def compute_torso_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    vel_w: float = 0.5,
    smooth_w: float = 0.0,
    range_w: float = 0.0,
    std_w: float = 0.0,
    bone_w: float = 0.0,
    spectral_w: float = 0.0,
    speed_w: float = 0.0,
    under_w: float = 0.0,
    jerk_match_w: float = 0.0,
    range_cap_w: float = 0.0,
    use_l1: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Torso/backbone regression loss with expressiveness terms.

    Base:
        pos_loss:  MSE or L1 on positions
        vel_loss:  L1 on velocity vectors (scale-friendly for zscore)

    Expressiveness (controlled by weights, 0 = disabled):
        range_loss:     |pred_range - gt_range| per axis
        std_loss:       |pred_std - gt_std| per axis
        speed_loss:     MSE on speed magnitudes (||vel||)
        under_loss:     asymmetric penalty when pred speed < gt speed
        smooth_loss:    penalty on pred acceleration > GT
        bone_loss:      bone length consistency
        spectral_loss:  low-freq FFT magnitude matching

    Args:
        pred, gt:      (B, T, D) where D=12 (torso) or 24 (backbone)
        vel_w:         velocity vector matching weight
        speed_w:       speed magnitude matching weight
        under_w:       asymmetric under-movement penalty weight
        smooth_w:      smoothness penalty weight
        range_w:       per-axis range matching weight
        std_w:         per-axis std matching weight
        bone_w:        bone length consistency weight
        spectral_w:    spectral (FFT) loss weight
        use_l1:        True for L1, False for MSE on positions
    """
    # Position loss
    if use_l1:
        pos_loss = (pred - gt).abs().mean()
    else:
        pos_loss = F.mse_loss(pred, gt)

    # Velocity loss — L1 instead of MSE for better scale in zscore space
    vel_pred = pred[:, 1:] - pred[:, :-1]
    vel_gt = gt[:, 1:] - gt[:, :-1]
    vel_loss = (vel_pred - vel_gt).abs().mean()

    loss = pos_loss + vel_w * vel_loss

    loss_dict = {
        'pos': pos_loss.item(),
        'vel': vel_loss.item(),
        'total': 0.0,
    }

    EPS = 1e-6

    # Per-dimension speed (axis-independent, not a single norm)
    if speed_w > 0 or under_w > 0:
        dimspeed_pred = vel_pred.abs().mean(dim=1)  # (B, D) mean speed per dim
        dimspeed_gt = vel_gt.abs().mean(dim=1)

    # Speed matching: log-ratio per dimension (scale-invariant)
    if speed_w > 0:
        speed_ratio = torch.log(dimspeed_pred / (dimspeed_gt + EPS) + EPS)
        speed_loss = speed_ratio.pow(2).mean()
        loss = loss + speed_w * speed_loss
        loss_dict['speed'] = speed_loss.item()

    # Asymmetric under-movement: relative penalty per dimension
    if under_w > 0:
        under_ratio = F.relu(1.0 - dimspeed_pred / (dimspeed_gt + EPS))
        under_loss = under_ratio.mean()
        loss = loss + under_w * under_loss
        loss_dict['under'] = under_loss.item()

    # Smoothness: penalize pred being rougher than GT
    if smooth_w > 0:
        acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
        acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
        smooth_loss = F.relu(acc_pred.pow(2).mean() - acc_gt.pow(2).mean())
        loss = loss + smooth_w * smooth_loss
        loss_dict['smooth'] = smooth_loss.item()

    # Per-axis range matching: log-ratio (scale-invariant across axes)
    if range_w > 0:
        r_pred = pred.max(dim=1).values - pred.min(dim=1).values  # (B, D)
        r_gt = gt.max(dim=1).values - gt.min(dim=1).values
        range_ratio = torch.log(r_pred / (r_gt + EPS) + EPS)
        range_loss = range_ratio.pow(2).mean()
        loss = loss + range_w * range_loss
        loss_dict['range'] = range_loss.item()

    # Per-axis std matching: log-ratio (scale-invariant)
    if std_w > 0:
        s_pred = pred.std(dim=1, unbiased=False)  # (B, D)
        s_gt = gt.std(dim=1, unbiased=False)
        std_ratio = torch.log(s_pred / (s_gt + EPS) + EPS)
        std_loss = std_ratio.pow(2).mean()
        loss = loss + std_w * std_loss
        loss_dict['std'] = std_loss.item()

    # Bone length consistency: |pred_bone_len - gt_bone_len|
    if bone_w > 0:
        pred_j = pred.reshape(*pred.shape[:2], 4, 3)
        gt_j = gt.reshape(*gt.shape[:2], 4, 3)
        bone_loss = torch.tensor(0.0, device=pred.device)
        for i, j in TORSO_BONE_PAIRS:
            pd = (pred_j[:, :, i] - pred_j[:, :, j]).norm(dim=-1)
            gd = (gt_j[:, :, i] - gt_j[:, :, j]).norm(dim=-1)
            bone_loss = bone_loss + (pd - gd).abs().mean()
        bone_loss = bone_loss / len(TORSO_BONE_PAIRS)
        loss = loss + bone_w * bone_loss
        loss_dict['bone'] = bone_loss.item()

    # Jerk-matching: log-ratio of 3rd-derivative magnitude (preserves micro-dynamics)
    if jerk_match_w > 0:
        acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
        acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
        jerk_pred = acc_pred[:, 1:] - acc_pred[:, :-1]
        jerk_gt = acc_gt[:, 1:] - acc_gt[:, :-1]
        jm_pred = jerk_pred.abs().mean(dim=1)  # (B, D)
        jm_gt = jerk_gt.abs().mean(dim=1)
        jerk_ratio = torch.log(jm_pred / (jm_gt + EPS) + EPS)
        jerk_match_loss = jerk_ratio.pow(2).mean()
        loss = loss + jerk_match_w * jerk_match_loss
        loss_dict['jerk_match'] = jerk_match_loss.item()

    # Range cap: asymmetric penalty for over-expression (pred range > GT range)
    if range_cap_w > 0:
        rc_pred = pred.max(dim=1).values - pred.min(dim=1).values  # (B, D)
        rc_gt = gt.max(dim=1).values - gt.min(dim=1).values
        over_ratio = F.relu(rc_pred / (rc_gt + EPS) - 1.0)
        range_cap_loss = over_ratio.mean()
        loss = loss + range_cap_w * range_cap_loss
        loss_dict['range_cap'] = range_cap_loss.item()

    # Spectral loss: match low-frequency FFT magnitudes (torso sway = 0~2Hz)
    if spectral_w > 0:
        T = pred.shape[1]
        n_low = max(1, int(2.0 * T / 60.0))
        pred_fft = torch.fft.rfft(pred.float(), dim=1)[:, 1:n_low+1, :]
        gt_fft = torch.fft.rfft(gt.float(), dim=1)[:, 1:n_low+1, :]
        spectral_loss = (pred_fft.abs() - gt_fft.abs()).pow(2).mean()
        loss = loss + spectral_w * spectral_loss
        loss_dict['spectral'] = spectral_loss.item()

    loss_dict['total'] = loss.item()
    return loss, loss_dict
