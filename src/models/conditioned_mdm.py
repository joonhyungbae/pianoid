"""
Conditioned Motion Diffusion Model - Base Class

Reusable MDM architecture for hierarchical body generation.
Each body part uses the same diffusion framework with different conditioning.

Based on MDM (Motion Diffusion Model, ICLR 2023).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (MDM style)."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """Embed diffusion timesteps."""
    
    def __init__(self, latent_dim: int, sequence_pos_encoder: PositionalEncoding):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class ConditionedMDM(nn.Module):
    """
    Conditioned Motion Diffusion Model.
    
    Generic MDM that can be conditioned on any input.
    Used as base for Hand/Backbone/Lower MDM.
    
    Architecture:
        x_t (noisy) + timestep + condition -> TransformerEncoder -> x_0 (clean)
    """
    
    def __init__(
        self,
        output_dim: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_mask_prob: float = 0.1,
        temporal_cond_dim: int = 0,
        global_cond_dim: int = 0,
        spatial_cond_dim: int = 0,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.cond_mask_prob = cond_mask_prob
        
        self.input_proj = nn.Linear(output_dim, latent_dim)
        self.output_proj = nn.Linear(latent_dim, output_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        
        self.temporal_cond_dim = temporal_cond_dim
        self.global_cond_dim = global_cond_dim
        self.spatial_cond_dim = spatial_cond_dim
        
        if temporal_cond_dim > 0:
            self.temporal_encoder = nn.Sequential(
                nn.Linear(temporal_cond_dim, latent_dim // 2),
                nn.SiLU(),
                nn.Linear(latent_dim // 2, latent_dim),
            )
        
        if global_cond_dim > 0:
            self.global_encoder = nn.Sequential(
                nn.Linear(global_cond_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )
        
        if spatial_cond_dim > 0:
            self.spatial_encoder = nn.Sequential(
                nn.Linear(spatial_cond_dim, latent_dim // 2),
                nn.SiLU(),
                nn.Linear(latent_dim // 2, latent_dim),
            )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def mask_cond(self, cond: torch.Tensor, force_mask: bool = False) -> torch.Tensor:
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            bs = cond.shape[1]
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(1, bs, 1)
            return cond * (1. - mask)
        else:
            return cond
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        temporal_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        spatial_cond: Optional[torch.Tensor] = None,
        force_mask: bool = False,
    ) -> torch.Tensor:
        B, T, D = x.shape
        
        time_emb = self.embed_timestep(timesteps)
        x_emb = self.input_proj(x).permute(1, 0, 2)
        
        if self.temporal_cond_dim > 0 and temporal_cond is not None:
            temp_emb = self.temporal_encoder(temporal_cond).permute(1, 0, 2)
            temp_emb = self.mask_cond(temp_emb, force_mask)
            x_emb = x_emb + temp_emb
        
        if self.spatial_cond_dim > 0 and spatial_cond is not None:
            spat_emb = self.spatial_encoder(spatial_cond).permute(1, 0, 2)
            spat_emb = self.mask_cond(spat_emb, force_mask)
            x_emb = x_emb + spat_emb
        
        cond_tokens = [time_emb]
        
        if self.global_cond_dim > 0 and global_cond is not None:
            glob_emb = self.global_encoder(global_cond).unsqueeze(0)
            glob_emb = self.mask_cond(glob_emb, force_mask)
            cond_tokens.append(glob_emb)
        
        cond_seq = torch.cat(cond_tokens, dim=0)
        n_cond = cond_seq.shape[0]
        xseq = torch.cat([cond_seq, x_emb], dim=0)
        
        xseq = self.sequence_pos_encoder(xseq)
        output = self.transformer(xseq)
        output = output[n_cond:]
        output = self.output_proj(output).permute(1, 0, 2)
        
        return output
