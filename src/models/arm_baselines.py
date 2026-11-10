"""
Arm Generation Regression Models (Stage 3 alternatives to ArmMDM)

All models follow the same interface:
    Input: spatial_cond (wrists + chest, 12-dim) + optional onset_density (1-dim)
    Output: arm positions (12-dim) = [RSHO, RELB, LSHO, LELB]

Models:
1. ArmTransformerRegression
2. ArmBiLSTMRegression
3. ArmMambaRegression
4. ArmFiLMRegression
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ArmTransformerRegression(nn.Module):
    def __init__(
        self,
        spatial_dim: int = 12,
        arm_dim: int = 12,
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

        input_dim = spatial_dim + temporal_cond_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_aria:
            self.aria_film = nn.Linear(aria_dim, hidden_dim * 2)

        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, arm_dim),
        )

    def forward(self, spatial_cond, aria=None, temporal_cond=None):
        parts = [spatial_cond]
        if self.temporal_cond_dim > 0 and temporal_cond is not None:
            parts.append(temporal_cond)
        x = torch.cat(parts, dim=-1)

        h = self.input_proj(x)

        if self.use_aria and aria is not None:
            film = self.aria_film(aria)
            scale, shift = film.chunk(2, dim=-1)
            h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        h = self.pos_encoder(h)
        h = self.transformer(h)
        return self.output_proj(h)


class ArmBiLSTMRegression(nn.Module):
    def __init__(
        self,
        spatial_dim: int = 12,
        arm_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_aria: bool = False,
        aria_dim: int = 512,
        temporal_cond_dim: int = 0,
    ):
        super().__init__()
        self.use_aria = use_aria
        self.temporal_cond_dim = temporal_cond_dim

        input_dim = spatial_dim + temporal_cond_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_aria:
            self.aria_film = nn.Linear(aria_dim, hidden_dim * 2)

        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, arm_dim),
        )

    def forward(self, spatial_cond, aria=None, temporal_cond=None):
        parts = [spatial_cond]
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


class ArmMambaRegression(nn.Module):
    def __init__(
        self,
        spatial_dim: int = 12,
        arm_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 6,
        d_state: int = 16,
        dropout: float = 0.1,
        use_aria: bool = False,
        aria_dim: int = 512,
        temporal_cond_dim: int = 0,
    ):
        super().__init__()
        self.use_aria = use_aria
        self.temporal_cond_dim = temporal_cond_dim

        input_dim = spatial_dim + temporal_cond_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_aria:
            self.aria_film = nn.Linear(aria_dim, hidden_dim * 2)

        try:
            from mamba_ssm import Mamba
            self._use_mamba = True
            self.layers = nn.ModuleList([
                Mamba(d_model=hidden_dim, d_state=d_state, d_conv=4, expand=2)
                for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(hidden_dim)
        except ImportError:
            self._use_mamba = False
            self.lstm = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

        out_dim = hidden_dim if self._use_mamba else hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(out_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, arm_dim),
        )

    def forward(self, spatial_cond, aria=None, temporal_cond=None):
        parts = [spatial_cond]
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


class ArmFiLMRegression(nn.Module):
    def __init__(
        self,
        spatial_dim: int = 12,
        arm_dim: int = 12,
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

        input_dim = spatial_dim + temporal_cond_dim
        context_input_dim = input_dim * (2 * context_frames + 1)

        layers = [nn.Linear(context_input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, arm_dim))
        self.net = nn.Sequential(*layers)

        if use_aria:
            self.aria_film = nn.Linear(aria_dim, hidden_dim * 2)

    def _build_context(self, x):
        B, T, D = x.shape
        k = self.context_frames
        padded = F.pad(x, (0, 0, k, k), mode='replicate')
        chunks = [padded[:, i:i+T, :] for i in range(2 * k + 1)]
        return torch.cat(chunks, dim=-1)

    def forward(self, spatial_cond, aria=None, temporal_cond=None):
        parts = [spatial_cond]
        if self.temporal_cond_dim > 0 and temporal_cond is not None:
            parts.append(temporal_cond)
        x = torch.cat(parts, dim=-1)

        x = self._build_context(x)

        if self.use_aria and aria is not None:
            h = self.net[0](x)
            h = self.net[1](h)
            film = self.aria_film(aria)
            scale, shift = film.chunk(2, dim=-1)
            h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            for layer in self.net[2:]:
                h = layer(h)
            return h
        else:
            return self.net(x)
