#!/usr/bin/env python3
"""
HandMambaInputCond: Mamba with 776-dim input [MIDI_264 ; ARIA_512] for Explore 02.

Same backbone as Explore 01 best Mamba (hidden_dim=512, num_layers=4, d_state=16).
Caller masks unused input to zeros:
  midi_only  -> aria_embed = zeros(B, 512)
  aria_only  -> midi_features = zeros
  midi_aria  -> both real
"""
import sys
import warnings
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.hand_baselines import concat_midi_features

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    Mamba = None

MIDI_DIM = 264
ARIA_DIM = 512
INPUT_DIM = MIDI_DIM + ARIA_DIM  # 776


class HandMambaInputCond(nn.Module):
    """Mamba with [midi_264; aria_512] input. Explore 01 best config defaults."""

    def __init__(
        self,
        hand_dim: int = 12,
        hidden_dim: int = 512,
        num_layers: int = 4,
        d_state: int = 16,
        dropout: float = 0.1,
        force_fallback: bool = False,
    ):
        super().__init__()
        self.hand_dim = hand_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._use_mamba = HAS_MAMBA and not force_fallback

        if not HAS_MAMBA:
            warnings.warn(
                "mamba-ssm not installed. HandMambaInputCond uses LSTM fallback. "
                "Install with: pip install mamba-ssm (requires CUDA).",
                UserWarning,
            )
        elif force_fallback:
            warnings.warn(
                "HandMambaInputCond: force_fallback=True. Using LSTM fallback.",
                UserWarning,
            )

        self.input_proj = nn.Linear(INPUT_DIM, hidden_dim)

        if self._use_mamba:
            self.layers = nn.ModuleList([
                Mamba(
                    d_model=hidden_dim, d_state=d_state,
                    d_conv=4, expand=2, layer_idx=i,
                )
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
            nn.Linear(hidden_dim, hand_dim),
        )

        self._init_weights()
        self._init_output_bias()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_output_bias(self):
        default_hand = torch.tensor([
            -0.2, 0.85, 0.75,   # left wrist
            -0.2, 0.80, 0.73,   # left finger
             0.2, 0.85, 0.75,   # right wrist
             0.2, 0.80, 0.73,   # right finger
        ], dtype=torch.float32)
        with torch.no_grad():
            self.output_proj[-1].bias.copy_(default_hand)

    def forward(
        self,
        midi_features: Dict[str, torch.Tensor],
        aria_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            midi_features: {piano_roll, velocity, onset} each [B, T, 88]
            aria_embed: [B, 512] global ARIA embedding (broadcast to all T)
        Returns:
            hand: [B, T, 12]
        """
        midi = concat_midi_features(midi_features)  # [B, T, 264]
        B, T, _ = midi.shape
        aria = aria_embed.unsqueeze(1).expand(B, T, ARIA_DIM)  # [B, T, 512]
        x = torch.cat([midi, aria], dim=-1)  # [B, T, 776]

        h = self.input_proj(x)  # [B, T, hidden]

        if self._use_mamba:
            for layer in self.layers:
                h = h + layer(h)
            h = self.norm(h)
        else:
            h, _ = self.lstm(h)

        return self.output_proj(h)  # [B, T, 12]
