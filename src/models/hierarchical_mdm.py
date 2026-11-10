"""
Hierarchical MDM Models for Piano Body Generation

New data-driven architecture (body_decomposition_analysis.json):

  Stage 1: HandMDM (4j)        MIDI → [LWRIST(14), LHAND(15), RWRIST(10), RHAND(11)]
  Stage 2: TorsoMDM (4j)       hands → [C7(4), CLAV(5), T10(6), STRN(7)]
  Stage 3: ArmMDM (4j)         wrists+chest → [RSHO(8), RELB(9), LSHO(12), LELB(13)]
  HeadRegressor (4j)           C7+CLAV → [RFHD(0), LFHD(1), RBHD(2), LBHD(3)]  (non-diffusion)
  Stage 4: LowerFiLM (10j)     spine → [PELVIS×4(16-19), LEGS×6(20-25)]

Legacy classes (BackboneMDM, ElbowMDM, LowerMDM, MeanPoseLowerMDM) are retained
for backwards compatibility.

Each uses the same ConditionedMDM base with different conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .conditioned_mdm import ConditionedMDM


class HandMDM(ConditionedMDM):
    """
    Hand MDM: MIDI + Velocity + ARIA -> Hand (wrist + finger)
    
    Input conditions:
        - MIDI: (B, T, 88) per-frame active keys (multi-hot)
        - Velocity: (B, T, 88) per-key velocity (0-1 normalized)
        - ARIA: (B, 512) global music embedding
    
    Output: (B, T, 12) = [left_wrist, left_finger, right_wrist, right_finger]
    
    The velocity information helps the model learn:
        - Stronger strikes -> larger downward wrist motion
        - Softer strikes -> smaller, controlled motion
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_mask_prob: float = 0.1,
        use_velocity: bool = True,
    ):
        # MIDI (88) + Velocity (88) = 176 if velocity is used
        temporal_dim = 176 if use_velocity else 88
        
        super().__init__(
            output_dim=12,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_mask_prob=cond_mask_prob,
            temporal_cond_dim=temporal_dim,
            global_cond_dim=512,
            spatial_cond_dim=0,
        )
        self.use_velocity = use_velocity
        self._init_output_bias()
    
    def _init_output_bias(self):
        default_hand = torch.tensor([
            -0.2, 0.85, 0.5,
            -0.2, 0.80, 0.6,
            0.2, 0.85, 0.5,
            0.2, 0.80, 0.6,
        ])
        with torch.no_grad():
            self.output_proj.bias.copy_(default_hand)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        midi: torch.Tensor,
        aria: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        force_mask: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, 12) noisy hand positions
            timesteps: (B,) diffusion timesteps
            midi: (B, T, 88) active keys (multi-hot)
            aria: (B, 512) ARIA embedding
            velocity: (B, T, 88) per-key velocity (optional)
            force_mask: Force mask conditions for CFG
        
        Returns:
            pred: (B, T, 12) predicted clean hand positions
        """
        # Concatenate MIDI and velocity if velocity is provided
        if self.use_velocity and velocity is not None:
            temporal_cond = torch.cat([midi, velocity], dim=-1)  # (B, T, 176)
        else:
            # Pad with zeros if velocity not provided but model expects it
            if self.use_velocity:
                zeros = torch.zeros_like(midi)
                temporal_cond = torch.cat([midi, zeros], dim=-1)
            else:
                temporal_cond = midi
        
        return super().forward(
            x, timesteps,
            temporal_cond=temporal_cond,
            global_cond=aria,
            force_mask=force_mask,
        )


class BackboneMDM(ConditionedMDM):
    """
    Backbone MDM: Hand + MIDI -> Backbone + Shoulder

    Input conditions:
        - Hand: (B, T, 12) wrist + finger positions
        - MIDI: (B, T, 88) piano roll (optional, frame-level)
        - ARIA: (B, 512) global music embedding (optional)
        - pitch_range: (B, T, 1) normalized pitch range (optional)

    Output: (B, T, 30) = 10 joints x 3 coords
        [Head(4), Neck(2), Chest(2), Shoulder(2)]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_mask_prob: float = 0.1,
        use_aria: bool = True,
        use_midi: bool = False,
        midi_dim: int = 88,
        use_pitch_range: bool = False,
    ):
        # temporal_cond_dim: MIDI (if used) + pitch_range (1-dim, if used)
        temporal_dim = 0
        if use_midi:
            temporal_dim += midi_dim
        if use_pitch_range:
            temporal_dim += 1
        super().__init__(
            output_dim=30,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_mask_prob=cond_mask_prob,
            temporal_cond_dim=temporal_dim if temporal_dim > 0 else 0,
            global_cond_dim=512 if use_aria else 0,
            spatial_cond_dim=12,
        )
        self.use_aria = use_aria
        self.use_midi = use_midi
        self.use_pitch_range = use_pitch_range

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        hand: torch.Tensor,
        aria: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        pitch_range: Optional[torch.Tensor] = None,
        force_mask: bool = False,
    ) -> torch.Tensor:
        # Build temporal conditioning by concatenating available features
        temporal_parts = []
        if self.use_midi and midi is not None:
            temporal_parts.append(midi)
        if self.use_pitch_range and pitch_range is not None:
            # pitch_range: (B, T) -> (B, T, 1)
            if pitch_range.dim() == 2:
                pitch_range = pitch_range.unsqueeze(-1)
            temporal_parts.append(pitch_range)
        temporal_cond = torch.cat(temporal_parts, dim=-1) if temporal_parts else None

        return super().forward(
            x, timesteps,
            spatial_cond=hand,
            global_cond=aria if self.use_aria else None,
            temporal_cond=temporal_cond,
            force_mask=force_mask,
        )


class TorsoMDM(ConditionedMDM):
    """
    Stage 2 – Torso MDM: Hand → Neck + Chest

    Data justification (body_decomposition_analysis.json):
      - hand_x → C7_x: r=0.607 (strong)
      - MIDI → C7_x: r=0.265 (weak; hands are necessary intermediate)

    Input conditions:
        - hand: (B, T, 12) wrist + finger positions from Stage 1
        - aria: (B, 512) global music embedding

    Output: (B, T, 12) = 4 joints × 3 coords
        [C7(4), CLAV(5), T10(6), STRN(7)]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_mask_prob: float = 0.1,
        use_aria: bool = True,
    ):
        super().__init__(
            output_dim=12,          # 4 joints × 3
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_mask_prob=cond_mask_prob,
            temporal_cond_dim=0,
            global_cond_dim=512 if use_aria else 0,
            spatial_cond_dim=12,    # hand (12)
        )
        self.use_aria = use_aria
        self._init_output_bias()

    def _init_output_bias(self):
        """Initialize to mean torso position (approximately upright seated)."""
        default_torso = torch.tensor([
            0.0,  0.55, 1.45,   # C7
            0.0,  0.45, 1.35,   # CLAV
            0.0,  0.40, 1.15,   # T10
            0.0,  0.50, 1.10,   # STRN
        ])
        with torch.no_grad():
            self.output_proj.bias.copy_(default_torso)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        hand: torch.Tensor,
        aria: Optional[torch.Tensor] = None,
        force_mask: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, 12) noisy torso positions
            timesteps: (B,) diffusion timesteps
            hand:      (B, T, 12) hand positions from Stage 1
            aria:      (B, 512) ARIA embedding
            force_mask: for CFG
        Returns:
            (B, T, 12) predicted torso [C7, CLAV, T10, STRN]
        """
        return super().forward(
            x, timesteps,
            spatial_cond=hand,
            global_cond=aria if self.use_aria else None,
            temporal_cond=None,
            force_mask=force_mask,
        )


class ArmMDM(ConditionedMDM):
    """
    Stage 3 – Arm MDM: Wrists + Chest → Shoulders + Elbows

    Data justification (body_decomposition_analysis.json):
      - elbow_x vs wrist_x:   r=0.946 (primary dependency)
      - elbow_x vs shoulder_x: r=0.770 (secondary dependency)
      - Elbows MUST come after both Stage 1 (wrists) and Stage 2 (torso).
      - Shoulders generated jointly with elbows to avoid cascade error
        (shoulder in S2 → 77.5mm error propagates to elbow).

    Input conditions:
        - wrists: (B, T, 6)  [RWRIST(10), LWRIST(14)] from Stage 1
        - chest:  (B, T, 6)  [T10(6), STRN(7)] from Stage 2 (anatomical anchor)
        - aria:   (B, 512) global music embedding

    Output: (B, T, 12) = 4 joints × 3 coords
        [RSHO(8), RELB(9), LSHO(12), LELB(13)]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_mask_prob: float = 0.1,
        use_aria: bool = True,
    ):
        super().__init__(
            output_dim=12,          # 4 joints × 3
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_mask_prob=cond_mask_prob,
            temporal_cond_dim=0,
            global_cond_dim=512 if use_aria else 0,
            spatial_cond_dim=12,    # wrists(6) + chest(6)
        )
        self.use_aria = use_aria
        self._init_output_bias()

    def _init_output_bias(self):
        """Initialize to mean arm position (seated pianist)."""
        default_arm = torch.tensor([
            0.20,  0.40, 1.35,   # RSHO
            0.30,  0.55, 1.00,   # RELB
           -0.20,  0.40, 1.35,   # LSHO
           -0.30,  0.55, 1.00,   # LELB
        ])
        with torch.no_grad():
            self.output_proj.bias.copy_(default_arm)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        wrists: torch.Tensor,
        chest: torch.Tensor,
        aria: Optional[torch.Tensor] = None,
        force_mask: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, 12) noisy arm positions
            timesteps: (B,) diffusion timesteps
            wrists:    (B, T, 6)  [RWRIST, LWRIST] from Stage 1
            chest:     (B, T, 6)  [T10, STRN] from Stage 2
            aria:      (B, 512) ARIA embedding
            force_mask: for CFG
        Returns:
            (B, T, 12) predicted arms [RSHO, RELB, LSHO, LELB]
        """
        spatial_cond = torch.cat([wrists, chest], dim=-1)  # (B, T, 12)
        return super().forward(
            x, timesteps,
            spatial_cond=spatial_cond,
            global_cond=aria if self.use_aria else None,
            temporal_cond=None,
            force_mask=force_mask,
        )


class HeadRegressor(nn.Module):
    """
    Head Regressor: C7 + CLAV → Head (non-diffusion, deterministic)

    Data justification (body_decomposition_analysis.json):
      - head_x vs C7_x:   r=0.934 (near-rigid follow)
      - head_x vs MIDI:   r=0.200 (essentially no direct music coupling)

    Head position is almost completely determined by spine (C7/CLAV),
    so diffusion is unnecessary — a temporal MLP suffices.

    Input:  (B, T, 6)  [C7(4), CLAV(5)] positions from Stage 2
    Output: (B, T, 12) [RFHD(0), LFHD(1), RBHD(2), LBHD(3)] positions
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(6, hidden_dim)
        layers = []
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        self.hidden = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, 12)
        self._init_output_bias()

    def _init_output_bias(self):
        """Initialize to mean head position (above neck, slightly forward)."""
        default_head = torch.tensor([
            0.07,  0.18, 1.75,   # RFHD
           -0.07,  0.18, 1.75,   # LFHD
            0.07, -0.05, 1.75,   # RBHD
           -0.07, -0.05, 1.75,   # LBHD
        ])
        with torch.no_grad():
            self.output_proj.bias.copy_(default_head)

    def forward(self, neck: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neck: (B, T, 6) [C7, CLAV] positions from Stage 2 TorsoMDM
        Returns:
            (B, T, 12) head positions [RFHD, LFHD, RBHD, LBHD]
        """
        h = self.input_proj(neck)       # (B, T, hidden)
        h = h + self.hidden(h)          # residual
        return self.output_proj(h)      # (B, T, 12)


class HeadResidualRegressor(nn.Module):
    """
    Head Residual Regressor: torso → head via mean-offset + learned residual.

    head = neck_center + mean_offset + model(torso)

    The mean-offset baseline alone achieves ~41mm MPJPE (GT neck), which is
    better than the original HeadRegressor MLP (70mm). This model learns only
    the residual (head tilt/rotation variations), guaranteeing at least
    mean-offset quality as a floor.

    Input:  (B, T, 12) full torso [C7, CLAV, T10, STRN] — more context than neck-only
    Output: (B, T, 12) head [RFHD, LFHD, RBHD, LBHD]
    """

    # Per-marker mean offset from neck center (computed from training set)
    MEAN_OFFSET = [
        0.046315, 0.180600, 0.165338,   # RFHD
        -0.050339, 0.177253, 0.168896,  # LFHD
        0.055142, 0.011573, 0.175192,   # RBHD
        -0.030769, 0.001929, 0.180065,  # LBHD
    ]

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.register_buffer('mean_offset', torch.tensor(self.MEAN_OFFSET).float())

        # Residual network: torso(12) → residual(12)
        input_dim = 12  # full torso
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, 12))
        self.residual_net = nn.Sequential(*layers)

        # Init residual output to near-zero so initial pred ≈ mean-offset baseline
        with torch.no_grad():
            self.residual_net[-1].weight.mul_(0.01)
            self.residual_net[-1].bias.zero_()

    def forward(self, torso: torch.Tensor) -> torch.Tensor:
        """
        Args:
            torso: (B, T, 12) [C7(0:3), CLAV(3:6), T10(6:9), STRN(9:12)]
        Returns:
            (B, T, 12) head positions [RFHD, LFHD, RBHD, LBHD]
        """
        # Neck center: mean of C7 and CLAV
        c7 = torso[..., 0:3]
        clav = torso[..., 3:6]
        neck_center = (c7 + clav) * 0.5  # (B, T, 3)

        # Mean-offset baseline: expand neck_center to 4 markers
        base = neck_center.unsqueeze(-2).expand(*neck_center.shape[:-1], 4, 3)  # (B, T, 4, 3)
        base = base + self.mean_offset.view(4, 3)  # broadcast add
        base = base.reshape(*torso.shape[:-1], 12)  # (B, T, 12)

        # Learned residual from full torso context
        residual = self.residual_net(torso)  # (B, T, 12)

        return base + residual


class TorsoMDMLoss(nn.Module):
    """Loss for TorsoMDM: position + velocity + bilateral symmetry."""

    SYMMETRIC_PAIRS = [(0, 0)]  # C7/CLAV are midline; no true sym pairs, just smooth

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """pred, gt: (B, T, 12)"""
        loss_pos = F.mse_loss(pred, gt)
        # Velocity smoothness
        vel_pred = pred[:, 1:] - pred[:, :-1]
        vel_gt   = gt[:, 1:]   - gt[:, :-1]
        loss_vel = F.mse_loss(vel_pred, vel_gt)
        # Speed matching
        speed_pred = torch.norm(vel_pred, dim=-1)
        speed_gt   = torch.norm(vel_gt,   dim=-1)
        loss_speed = F.mse_loss(speed_pred, speed_gt)
        return loss_pos + 0.5 * loss_vel + 0.3 * loss_speed


class ArmMDMLoss(nn.Module):
    """Loss for ArmMDM: position + velocity + bilateral symmetry + bone length."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """pred, gt: (B, T, 12) = [RSHO, RELB, LSHO, LELB] each ×3"""
        B, T = pred.shape[:2]
        loss_pos = F.mse_loss(pred, gt)
        vel_pred = pred[:, 1:] - pred[:, :-1]
        vel_gt   = gt[:, 1:]   - gt[:, :-1]
        loss_vel = F.mse_loss(vel_pred, vel_gt)
        # Bilateral symmetry: RSHO↔LSHO, RELB↔LELB (mirror in x)
        pred_r = pred.reshape(B, T, 4, 3)
        gt_r   = gt.reshape(B, T, 4, 3)
        rsho_pred = pred_r[:, :, 0]; lsho_pred = pred_r[:, :, 2]
        relb_pred = pred_r[:, :, 1]; lelb_pred = pred_r[:, :, 3]
        # Mirror x: right.x ≈ -left.x
        sym_loss = F.mse_loss(rsho_pred[..., 0], -lsho_pred[..., 0]) \
                 + F.mse_loss(relb_pred[..., 0], -lelb_pred[..., 0])
        # Upper-arm bone length consistency
        upper_arm_pred = torch.norm(pred_r[:, :, 0] - pred_r[:, :, 1], dim=-1)  # RSHO-RELB
        upper_arm_gt   = torch.norm(gt_r[:, :, 0]   - gt_r[:, :, 1],   dim=-1)
        loss_bone = F.mse_loss(upper_arm_pred, upper_arm_gt)
        return loss_pos + 0.5 * loss_vel + 0.1 * sym_loss + 0.1 * loss_bone


class HeadRegressorLoss(nn.Module):
    """Loss for HeadRegressor: MSE + velocity smoothness."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss_pos = F.mse_loss(pred, gt)
        vel_pred = pred[:, 1:] - pred[:, :-1]
        vel_gt   = gt[:, 1:]   - gt[:, :-1]
        loss_vel = F.mse_loss(vel_pred, vel_gt)
        return loss_pos + 0.3 * loss_vel


class LowerMDM(ConditionedMDM):
    """
    Lower Body MDM: Torso -> Pelvis + Legs
    
    Input conditions:
        - Torso: (B, T, 6) T10 + STRN positions
        - ARIA: (B, 512) global music embedding
    
    Output: (B, T, 30) = 10 joints x 3 coords
        [Pelvis(4), R_Leg(3), L_Leg(3)]
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_mask_prob: float = 0.1,
        use_aria: bool = True,
    ):
        super().__init__(
            output_dim=30,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_mask_prob=cond_mask_prob,
            temporal_cond_dim=0,
            global_cond_dim=512 if use_aria else 0,
            spatial_cond_dim=6,
        )
        self.use_aria = use_aria
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        torso: torch.Tensor,
        aria: Optional[torch.Tensor] = None,
        force_mask: bool = False,
    ) -> torch.Tensor:
        return super().forward(
            x, timesteps,
            spatial_cond=torso,
            global_cond=aria if self.use_aria else None,
            force_mask=force_mask,
        )


class MeanPoseLowerMDM(nn.Module):
    """
    Mean Pose + FiLM Refinement for Lower Body.
    
    Key insight: Piano performers have nearly static lower body.
    - Pelvis: almost fixed (sitting)
    - Left Leg: almost fixed
    - Right Leg: slight movement (pedal)
    
    Architecture:
        1. Start from learned mean pose
        2. FiLM conditioning modulates mean pose based on torso state
        3. Small residual network predicts per-frame adjustments
    
    This is more efficient than learning from scratch when motion is minimal.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = 30  # 10 joints x 3
        self.latent_dim = latent_dim
        
        # Learnable mean pose (10 joints x 3)
        self.mean_pose = nn.Parameter(torch.zeros(30))
        
        # Learnable std for initialization
        self.pose_std = nn.Parameter(torch.ones(30) * 0.01)
        
        # FiLM generator: torso (6) + timestep -> scale (30) + shift (30)
        self.film_generator = nn.Sequential(
            nn.Linear(6 + 128, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 60),  # 30 scale + 30 shift
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        
        # Small residual predictor for per-frame adjustments
        self.residual_proj_in = nn.Linear(30 + 6, latent_dim)  # input + torso
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.residual_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.residual_proj_out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 30),
        )
        
        # Initialize residual output to near-zero
        nn.init.zeros_(self.residual_proj_out[-1].weight)
        nn.init.zeros_(self.residual_proj_out[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,          # (B, T, 30) noisy input
        timesteps: torch.Tensor,  # (B,) diffusion timesteps
        torso: torch.Tensor,      # (B, T, 6) T10 + STRN
        aria: Optional[torch.Tensor] = None,  # unused but kept for API compatibility
        force_mask: bool = False,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device
        
        # Time embedding
        t_emb = self.time_embed(timesteps.float().unsqueeze(-1) / 1000.0)  # (B, 128)
        
        # FiLM: generate per-sample scale and shift from torso mean
        torso_mean = torso.mean(dim=1)  # (B, 6)
        film_input = torch.cat([torso_mean, t_emb], dim=-1)  # (B, 6+128)
        film_params = self.film_generator(film_input)  # (B, 60)
        
        gamma = film_params[:, :30].unsqueeze(1)  # (B, 1, 30)
        beta = film_params[:, 30:].unsqueeze(1)   # (B, 1, 30)
        
        # Modulated mean pose
        mean_pose = self.mean_pose.unsqueeze(0).unsqueeze(0)  # (1, 1, 30)
        modulated_pose = (1 + gamma) * mean_pose + beta  # (B, 1, 30)
        modulated_pose = modulated_pose.expand(B, T, -1)  # (B, T, 30)
        
        # Residual prediction
        residual_input = torch.cat([x, torso], dim=-1)  # (B, T, 36)
        h = self.residual_proj_in(residual_input)  # (B, T, latent_dim)
        h = self.residual_transformer(h)  # (B, T, latent_dim)
        residual = self.residual_proj_out(h)  # (B, T, 30)
        
        # Final output: modulated mean + small residual
        # Scale residual by pose_std to control magnitude
        output = modulated_pose + residual * self.pose_std.abs().unsqueeze(0).unsqueeze(0)
        
        return output
    
    def get_mean_pose(self) -> torch.Tensor:
        """Get current mean pose for visualization."""
        return self.mean_pose.detach()
    
    def set_mean_pose_from_data(self, lower_data: torch.Tensor):
        """
        Initialize mean pose from training data.
        
        Args:
            lower_data: (N, T, 30) lower body positions from dataset
        """
        with torch.no_grad():
            mean = lower_data.reshape(-1, 30).mean(dim=0)
            std = lower_data.reshape(-1, 30).std(dim=0)
            self.mean_pose.copy_(mean)
            self.pose_std.copy_(std.clamp(min=0.001))


class MDMLoss(nn.Module):
    """
    Common loss function for all MDMs with anti-jittering.
    
    Components:
    - Position + Velocity + Acceleration MSE (standard)
    - Jerk penalty: penalizes rapid acceleration changes (reduces jittering)
    - High-frequency penalty: penalizes rapid oscillations
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        vel_weight: float = 0.5,
        acc_weight: float = 0.1,
        jerk_weight: float = 0.05,  # Anti-jittering: penalize jerk (d³x/dt³)
        smooth_weight: float = 0.02,  # Anti-jittering: penalize high-freq energy
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.jerk_weight = jerk_weight
        self.smooth_weight = smooth_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pos_loss = F.mse_loss(pred, target)
        
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(pred_vel, target_vel)
        
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_acc = target_vel[:, 1:] - target_vel[:, :-1]
        acc_loss = F.mse_loss(pred_acc, target_acc)
        
        # Jerk loss: penalize rapid acceleration changes (anti-jittering)
        pred_jerk = pred_acc[:, 1:] - pred_acc[:, :-1]
        target_jerk = target_acc[:, 1:] - target_acc[:, :-1]
        jerk_loss = F.mse_loss(pred_jerk, target_jerk)
        
        # Smoothness loss: penalize prediction being less smooth than GT
        # Compare second derivative magnitude (lower = smoother)
        pred_acc_mag = pred_acc.pow(2).mean()
        target_acc_mag = target_acc.pow(2).mean()
        # Penalize if pred is rougher (higher acc magnitude) than GT
        smooth_loss = F.relu(pred_acc_mag - target_acc_mag)
        
        total = (
            self.pos_weight * pos_loss +
            self.vel_weight * vel_loss +
            self.acc_weight * acc_loss +
            self.jerk_weight * jerk_loss +
            self.smooth_weight * smooth_loss
        )
        
        loss_dict = {
            'pos': pos_loss.item(),
            'vel': vel_loss.item(),
            'acc': acc_loss.item(),
            'jerk': jerk_loss.item(),
            'smooth': smooth_loss.item(),
            'total': total.item(),
        }
        
        return total, loss_dict


class LowerMDMLoss(nn.Module):
    """
    LowerMDM loss with static joint constraint.
    
    Problem: Left leg (LANKLE, LTOE) barely moves in GT but model generates excessive motion.
    Solution: Penalize variance exceeding GT variance for static joints.
    
    Lower body layout (30-dim = 10 joints x 3):
        [0-2]: RASI, [3-5]: LASI, [6-8]: RPSI, [9-11]: LPSI (pelvis)
        [12-14]: RKNE, [15-17]: RANKLE, [18-20]: RTOE (right leg)
        [21-23]: LKNE, [24-26]: LANKLE, [27-29]: LTOE (left leg)
    """
    
    # Static joint indices in 30-dim output (joints that should barely move)
    STATIC_JOINTS = {
        'LANKLE': (24, 27),  # indices 24-26
        'LTOE': (27, 30),    # indices 27-29
    }
    
    # Also constrain pelvis which has minimal movement
    PELVIS_JOINTS = {
        'RASI': (0, 3),
        'LASI': (3, 6),
        'RPSI': (6, 9),
        'LPSI': (9, 12),
    }
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        vel_weight: float = 0.5,
        acc_weight: float = 0.1,
        jerk_weight: float = 0.05,
        smooth_weight: float = 0.02,
        static_weight: float = 10.0,  # Increased: Static joint constraint weight
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.jerk_weight = jerk_weight
        self.smooth_weight = smooth_weight
        self.static_weight = static_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Basic losses
        pos_loss = F.mse_loss(pred, target)
        
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(pred_vel, target_vel)
        
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_acc = target_vel[:, 1:] - target_vel[:, :-1]
        acc_loss = F.mse_loss(pred_acc, target_acc)
        
        # Jerk loss
        pred_jerk = pred_acc[:, 1:] - pred_acc[:, :-1]
        target_jerk = target_acc[:, 1:] - target_acc[:, :-1]
        jerk_loss = F.mse_loss(pred_jerk, target_jerk)
        
        # Smoothness loss
        pred_acc_mag = pred_acc.pow(2).mean()
        target_acc_mag = target_acc.pow(2).mean()
        smooth_loss = F.relu(pred_acc_mag - target_acc_mag)
        
        # Static joint constraint: penalize excessive movement range
        # Use RANGE (max - min) instead of variance for better scaling
        static_loss = torch.tensor(0.0, device=pred.device)
        all_static_joints = {**self.STATIC_JOINTS, **self.PELVIS_JOINTS}
        
        for name, (start, end) in all_static_joints.items():
            pred_joint = pred[:, :, start:end]  # (B, T, 3)
            tgt_joint = target[:, :, start:end]
            
            # Range along time axis (max - min)
            pred_range = pred_joint.max(dim=1)[0] - pred_joint.min(dim=1)[0]  # (B, 3)
            tgt_range = tgt_joint.max(dim=1)[0] - tgt_joint.min(dim=1)[0]
            
            # Penalize if pred range > 2x GT range
            # For nearly static joints (GT range ~0), this effectively penalizes any movement
            min_threshold = 0.01  # 1cm minimum threshold
            tgt_range_safe = torch.clamp(tgt_range, min=min_threshold)
            
            # Ratio-based penalty: penalize if pred_range / tgt_range > 2
            ratio = pred_range / tgt_range_safe
            excess_ratio = F.relu(ratio - 2.0)  # Allow 2x GT range
            static_loss = static_loss + excess_ratio.mean()
        
        static_loss = static_loss / len(all_static_joints)
        
        total = (
            self.pos_weight * pos_loss +
            self.vel_weight * vel_loss +
            self.acc_weight * acc_loss +
            self.jerk_weight * jerk_loss +
            self.smooth_weight * smooth_loss +
            self.static_weight * static_loss
        )
        
        loss_dict = {
            'pos': pos_loss.item(),
            'vel': vel_loss.item(),
            'acc': acc_loss.item(),
            'jerk': jerk_loss.item(),
            'smooth': smooth_loss.item(),
            'static': static_loss.item(),
            'total': total.item(),
        }
        
        return total, loss_dict


class HandMDMLoss(MDMLoss):
    """Hand MDM loss with bone length constraint."""
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        vel_weight: float = 0.5,
        acc_weight: float = 0.1,
        bone_weight: float = 0.2,
        expected_dist: float = 0.15,
    ):
        super().__init__(pos_weight, vel_weight, acc_weight)
        self.bone_weight = bone_weight
        self.expected_dist = expected_dist
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total, loss_dict = super().forward(pred, target)
        
        # Bone length loss (wrist-finger distance)
        left_dist = torch.norm(pred[..., 3:6] - pred[..., 0:3], dim=-1)
        right_dist = torch.norm(pred[..., 9:12] - pred[..., 6:9], dim=-1)
        bone_loss = (
            F.mse_loss(left_dist, torch.full_like(left_dist, self.expected_dist)) +
            F.mse_loss(right_dist, torch.full_like(right_dist, self.expected_dist))
        ) / 2
        
        total = total + self.bone_weight * bone_loss
        loss_dict['bone'] = bone_loss.item()
        loss_dict['total'] = total.item()
        
        return total, loss_dict


class AxisWeightedLoss(nn.Module):
    """
    Position loss with axis-specific weights.
    Z-axis (vertical) can be weighted higher to emphasize up/down motion.
    
    Hand coordinate layout (12-dim):
        [0-2]: left wrist (x, y, z)
        [3-5]: left finger (x, y, z)
        [6-8]: right wrist (x, y, z)
        [9-11]: right finger (x, y, z)
    
    Z indices: 2, 5, 8, 11 (vertical movement)
    """
    
    def __init__(self, z_weight: float = 2.0, xy_weight: float = 1.0):
        super().__init__()
        self.z_weight = z_weight
        self.xy_weight = xy_weight
        
        # Build weight tensor for 12-dim hand
        # [x, y, z, x, y, z, x, y, z, x, y, z]
        weights = []
        for _ in range(4):  # 4 joints
            weights.extend([xy_weight, xy_weight, z_weight])
        self.register_buffer('axis_weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, T, 12) predicted hand positions
            target: (B, T, 12) target hand positions
        
        Returns:
            Axis-weighted MSE loss
        """
        diff_sq = (pred - target) ** 2  # (B, T, 12)
        weighted_diff = diff_sq * self.axis_weights.to(pred.device)
        return weighted_diff.mean()


class KeypressContactLoss(nn.Module):
    """
    Encourage downward wrist motion at keypress (onset) moments.
    
    When a key is pressed, the wrist should move downward (negative Z velocity).
    This loss penalizes upward or static motion at onset frames.
    
    Piano coordinate system:
        Z-axis = vertical (up is positive, down is negative)
    
    Hand layout (12-dim):
        indices 2, 8 = left/right wrist Z coordinates
    """
    
    def __init__(self, weight: float = 0.5, min_downward_vel: float = 0.005):
        super().__init__()
        self.weight = weight
        self.min_downward_vel = min_downward_vel  # Minimum expected downward velocity
    
    def forward(
        self,
        pred: torch.Tensor,
        key_onset: torch.Tensor,
        onset_density_per_key: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, T, 12) predicted hand positions
            key_onset: (B, T, 88) onset mask (1 at onset frame, 0 otherwise)
            onset_density_per_key: (B, T, 88) optional velocity for weighting
        
        Returns:
            Keypress contact loss
        """
        B, T, _ = pred.shape
        
        # Extract wrist Z coordinates
        left_wrist_z = pred[..., 2]   # (B, T)
        right_wrist_z = pred[..., 8]  # (B, T)
        
        # Compute Z velocity (difference between frames)
        left_vel_z = left_wrist_z[:, 1:] - left_wrist_z[:, :-1]   # (B, T-1)
        right_vel_z = right_wrist_z[:, 1:] - right_wrist_z[:, :-1]  # (B, T-1)
        
        # Onset mask per hand (left: keys 0-43, right: keys 44-87)
        # Using MIDI heuristic: left hand = lower keys, right hand = higher keys
        left_onset = key_onset[:, 1:, :44].sum(dim=-1).clamp(0, 1)   # (B, T-1)
        right_onset = key_onset[:, 1:, 44:].sum(dim=-1).clamp(0, 1)  # (B, T-1)
        
        # Velocity weighting (optional)
        if onset_density_per_key is not None:
            left_vel_weight = onset_density_per_key[:, 1:, :44].max(dim=-1)[0]  # (B, T-1)
            right_vel_weight = onset_density_per_key[:, 1:, 44:].max(dim=-1)[0]
        else:
            left_vel_weight = torch.ones_like(left_onset)
            right_vel_weight = torch.ones_like(right_onset)
        
        # At onset, velocity should be negative (downward)
        # Penalize positive (upward) or insufficient downward velocity
        # Loss = ReLU(vel_z + min_downward) at onset frames
        left_loss = F.relu(left_vel_z + self.min_downward_vel) * left_onset * left_vel_weight
        right_loss = F.relu(right_vel_z + self.min_downward_vel) * right_onset * right_vel_weight
        
        # Average over onset frames only
        num_onsets = (left_onset.sum() + right_onset.sum()).clamp(min=1)
        total_loss = (left_loss.sum() + right_loss.sum()) / num_onsets
        
        return self.weight * total_loss


class EnhancedHandMDMLoss(nn.Module):
    """
    Enhanced Hand MDM loss combining:
    1. Base MDM loss (position, velocity, acceleration)
    2. Bone length constraint
    3. Z-axis weighted position loss (vertical emphasis)
    4. Keypress contact loss (downward motion at onset)
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        vel_weight: float = 0.5,
        acc_weight: float = 0.1,
        bone_weight: float = 0.2,
        z_weight: float = 2.0,
        contact_weight: float = 0.3,
        expected_dist: float = 0.15,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.bone_weight = bone_weight
        self.z_weight = z_weight
        self.contact_weight = contact_weight
        self.expected_dist = expected_dist
        
        self.axis_loss = AxisWeightedLoss(z_weight=z_weight)
        self.contact_loss = KeypressContactLoss(weight=contact_weight)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        key_onset: Optional[torch.Tensor] = None,
        onset_density_per_key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: (B, T, 12) predicted hand positions
            target: (B, T, 12) target hand positions
            key_onset: (B, T, 88) onset mask (optional)
            onset_density_per_key: (B, T, 88) velocity per key (optional)
        
        Returns:
            total_loss, loss_dict
        """
        loss_dict = {}
        
        # 1. Axis-weighted position loss (Z emphasized)
        pos_loss = self.axis_loss(pred, target)
        loss_dict['pos'] = pos_loss.item()
        
        # 2. Velocity loss
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(pred_vel, target_vel)
        loss_dict['vel'] = vel_loss.item()
        
        # 3. Acceleration loss
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_acc = target_vel[:, 1:] - target_vel[:, :-1]
        acc_loss = F.mse_loss(pred_acc, target_acc)
        loss_dict['acc'] = acc_loss.item()
        
        # 4. Bone length loss
        left_dist = torch.norm(pred[..., 3:6] - pred[..., 0:3], dim=-1)
        right_dist = torch.norm(pred[..., 9:12] - pred[..., 6:9], dim=-1)
        bone_loss = (
            F.mse_loss(left_dist, torch.full_like(left_dist, self.expected_dist)) +
            F.mse_loss(right_dist, torch.full_like(right_dist, self.expected_dist))
        ) / 2
        loss_dict['bone'] = bone_loss.item()
        
        # 5. Keypress contact loss (if onset info provided)
        contact_loss_val = 0.0
        if key_onset is not None:
            contact_loss = self.contact_loss(pred, key_onset, onset_density_per_key)
            contact_loss_val = contact_loss.item()
            loss_dict['contact'] = contact_loss_val
        else:
            contact_loss = torch.tensor(0.0, device=pred.device)
            loss_dict['contact'] = 0.0
        
        # Total loss
        total = (
            self.pos_weight * pos_loss +
            self.vel_weight * vel_loss +
            self.acc_weight * acc_loss +
            self.bone_weight * bone_loss +
            contact_loss
        )
        
        loss_dict['total'] = total.item()
        
        return total, loss_dict


class BackboneMDMLoss(nn.Module):
    """
    Backbone MDM loss with symmetry, bone length, and stability constraints.
    
    Key features:
    1. Symmetry Loss: Maintain left-right symmetry (prevent twist)
    2. Bone Length Constraint: Keep bone lengths consistent with GT
    3. Spine Stability: Limit T10, CLAV, C7 movement range to GT range
    4. Mean Position Matching: Match average position to GT
    5. Anti-jittering: Jerk and smoothness penalties
    
    Backbone layout (30-dim = 10 joints x 3):
        [0-2]: RFHD, [3-5]: LFHD, [6-8]: RBHD, [9-11]: LBHD
        [12-14]: C7, [15-17]: CLAV, [18-20]: T10, [21-23]: STRN
        [24-26]: RSHO, [27-29]: LSHO
    """
    
    # Symmetric pairs: (left_idx, right_idx) in joint space (not 30-dim)
    SYMMETRIC_PAIRS = [
        (1, 0),  # LFHD - RFHD
        (3, 2),  # LBHD - RBHD
        (9, 8),  # LSHO - RSHO
    ]
    
    # Bone connections: (joint1, joint2) for length constraint
    BONE_PAIRS = [
        (4, 5),  # C7 - CLAV
        (5, 7),  # CLAV - STRN
        (7, 6),  # STRN - T10
        (4, 0),  # C7 - RFHD
        (8, 9),  # RSHO - LSHO (shoulder width)
    ]
    
    # Spine joints that should have limited movement (indices in 10-joint space)
    SPINE_JOINTS = [4, 5, 6, 7]  # C7, CLAV, T10, STRN
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        vel_weight: float = 0.5,
        acc_weight: float = 0.1,
        jerk_weight: float = 0.05,
        smooth_weight: float = 0.02,
        symmetry_weight: float = 1.0,
        bone_weight: float = 0.5,
        spine_weight: float = 1.0,    # NEW: Spine stability
        mean_pos_weight: float = 0.5,  # NEW: Mean position matching
        range_weight: float = 1.0,     # Anti-under-movement: motion range matching
        vel_match_weight: float = 0.5, # Velocity magnitude matching
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.jerk_weight = jerk_weight
        self.smooth_weight = smooth_weight
        self.symmetry_weight = symmetry_weight
        self.bone_weight = bone_weight
        self.spine_weight = spine_weight
        self.mean_pos_weight = mean_pos_weight
        self.range_weight = range_weight
        self.vel_match_weight = vel_match_weight
    
    def _symmetry_loss(self, pred, target):
        """
        Symmetry Loss: Match left-right relative offset to GT.
        
        Problem: Model can flip left/right causing twist.
        Solution: Match (left - right) difference to GT.
        """
        B, T, D = pred.shape
        pred_3d = pred.reshape(B, T, 10, 3)
        tgt_3d = target.reshape(B, T, 10, 3)
        
        sym_loss = torch.tensor(0.0, device=pred.device)
        
        for left_idx, right_idx in self.SYMMETRIC_PAIRS:
            # Y-axis offset (left - right)
            pred_y_diff = pred_3d[:, :, left_idx, 1] - pred_3d[:, :, right_idx, 1]
            tgt_y_diff = tgt_3d[:, :, left_idx, 1] - tgt_3d[:, :, right_idx, 1]
            
            # Match the sign and magnitude
            sym_loss = sym_loss + F.mse_loss(pred_y_diff, tgt_y_diff)
        
        return sym_loss / len(self.SYMMETRIC_PAIRS)
    
    def _bone_length_loss(self, pred, target):
        """
        Bone Length Loss: Keep bone lengths consistent with GT.
        """
        B, T, D = pred.shape
        pred_3d = pred.reshape(B, T, 10, 3)
        tgt_3d = target.reshape(B, T, 10, 3)
        
        bone_loss = torch.tensor(0.0, device=pred.device)
        
        for j1, j2 in self.BONE_PAIRS:
            pred_len = torch.norm(pred_3d[:, :, j1] - pred_3d[:, :, j2], dim=-1)
            tgt_len = torch.norm(tgt_3d[:, :, j1] - tgt_3d[:, :, j2], dim=-1)
            
            bone_loss = bone_loss + F.mse_loss(pred_len, tgt_len)
        
        return bone_loss / len(self.BONE_PAIRS)
    
    def _spine_stability_loss(self, pred, target):
        """
        Spine Stability Loss: Penalize if spine joints move more than GT.
        
        Problem: T10, CLAV, C7 move 5-11x more than GT.
        Solution: Penalize variance exceeding GT variance.
        """
        B, T, D = pred.shape
        pred_3d = pred.reshape(B, T, 10, 3)
        tgt_3d = target.reshape(B, T, 10, 3)
        
        spine_loss = torch.tensor(0.0, device=pred.device)
        
        for joint_idx in self.SPINE_JOINTS:
            pred_joint = pred_3d[:, :, joint_idx]  # (B, T, 3)
            tgt_joint = tgt_3d[:, :, joint_idx]
            
            # Variance along time axis
            pred_var = pred_joint.var(dim=1)  # (B, 3)
            tgt_var = tgt_joint.var(dim=1)
            
            # Penalize if pred variance > 1.5x GT variance
            excess_var = F.relu(pred_var - tgt_var * 1.5)
            spine_loss = spine_loss + excess_var.mean()
        
        return spine_loss / len(self.SPINE_JOINTS)
    
    def _mean_position_loss(self, pred, target):
        """
        Mean Position Loss: Match average position to GT.
        
        Problem: Model generates positions offset from GT by 7-11cm.
        Solution: Match per-joint mean position.
        """
        B, T, D = pred.shape
        pred_3d = pred.reshape(B, T, 10, 3)
        tgt_3d = target.reshape(B, T, 10, 3)
        
        # Mean position per joint per batch
        pred_mean = pred_3d.mean(dim=1)  # (B, 10, 3)
        tgt_mean = tgt_3d.mean(dim=1)
        
        return F.mse_loss(pred_mean, tgt_mean)
    
    def _motion_range_loss(self, pred, target):
        """
        Motion Range Loss: Penalize under-movement.
        
        Problem: Model may generate too little motion (under-movement).
        Solution: Match motion range (max - min) to GT for each joint.
        """
        B, T, D = pred.shape
        pred_3d = pred.reshape(B, T, 10, 3)
        tgt_3d = target.reshape(B, T, 10, 3)
        
        # Motion range per joint: max - min along time
        pred_range = pred_3d.max(dim=1).values - pred_3d.min(dim=1).values  # (B, 10, 3)
        tgt_range = tgt_3d.max(dim=1).values - tgt_3d.min(dim=1).values
        
        # Penalize if pred range < GT range (under-movement)
        under_movement = F.relu(tgt_range - pred_range)
        
        return under_movement.mean()
    
    def _velocity_magnitude_loss(self, pred, target):
        """
        Velocity Magnitude Loss: Match velocity magnitude to GT.
        
        Problem: Model may move too slowly or too fast.
        Solution: Match velocity magnitude distribution to GT.
        """
        # Compute velocities
        pred_vel = pred[:, 1:] - pred[:, :-1]
        tgt_vel = target[:, 1:] - target[:, :-1]
        
        # Velocity magnitude
        pred_vel_mag = pred_vel.pow(2).sum(dim=-1).sqrt()  # (B, T-1)
        tgt_vel_mag = tgt_vel.pow(2).sum(dim=-1).sqrt()
        
        return F.mse_loss(pred_vel_mag, tgt_vel_mag)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        
        # 1. Position loss
        pos_loss = F.mse_loss(pred, target)
        loss_dict['pos'] = pos_loss.item()
        
        # 2. Velocity loss
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(pred_vel, target_vel)
        loss_dict['vel'] = vel_loss.item()
        
        # 3. Acceleration loss
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_acc = target_vel[:, 1:] - target_vel[:, :-1]
        acc_loss = F.mse_loss(pred_acc, target_acc)
        loss_dict['acc'] = acc_loss.item()
        
        # 4. Jerk loss (anti-jittering)
        pred_jerk = pred_acc[:, 1:] - pred_acc[:, :-1]
        target_jerk = target_acc[:, 1:] - target_acc[:, :-1]
        jerk_loss = F.mse_loss(pred_jerk, target_jerk)
        loss_dict['jerk'] = jerk_loss.item()
        
        # 5. Smoothness loss
        pred_acc_mag = pred_acc.pow(2).mean()
        target_acc_mag = target_acc.pow(2).mean()
        smooth_loss = F.relu(pred_acc_mag - target_acc_mag)
        loss_dict['smooth'] = smooth_loss.item()
        
        # 6. Symmetry loss
        sym_loss = self._symmetry_loss(pred, target)
        loss_dict['sym'] = sym_loss.item()
        
        # 7. Bone length loss
        bone_loss = self._bone_length_loss(pred, target)
        loss_dict['bone'] = bone_loss.item()
        
        # 8. Spine stability loss (NEW)
        spine_loss = self._spine_stability_loss(pred, target)
        loss_dict['spine'] = spine_loss.item()
        
        # 9. Mean position loss (NEW)
        mean_pos_loss = self._mean_position_loss(pred, target)
        loss_dict['mean_pos'] = mean_pos_loss.item()
        
        # 10. Motion range loss (anti-under-movement)
        range_loss = self._motion_range_loss(pred, target)
        loss_dict['range'] = range_loss.item()
        
        # 11. Velocity magnitude loss
        vel_mag_loss = self._velocity_magnitude_loss(pred, target)
        loss_dict['vel_mag'] = vel_mag_loss.item()
        
        # Total
        total = (
            self.pos_weight * pos_loss +
            self.vel_weight * vel_loss +
            self.acc_weight * acc_loss +
            self.jerk_weight * jerk_loss +
            self.smooth_weight * smooth_loss +
            self.symmetry_weight * sym_loss +
            self.bone_weight * bone_loss +
            self.spine_weight * spine_loss +
            self.mean_pos_weight * mean_pos_loss +
            self.range_weight * range_loss +
            self.vel_match_weight * vel_mag_loss
        )
        
        loss_dict['total'] = total.item()
        
        return total, loss_dict


# =============================================================================
# NEW: Hierarchical Lower Body Models (Option B)
# =============================================================================

class PelvisMDM(ConditionedMDM):
    """
    Pelvis MDM: Backbone (10 joints) → Pelvis (4 joints)
    
    Input conditions:
        - Backbone: (B, T, 30) full backbone from BackboneMDM
        - ARIA: (B, 512) global music embedding
    
    Output: (B, T, 12) = [RASI, LASI, RPSI, LPSI] (4 joints × 3)
    
    Pelvis is 2.4x more stable than torso, making this a relatively
    easy prediction task. Uses lightweight architecture.
    """
    
    def __init__(
        self,
        latent_dim: int = 128,  # Smaller than other MDMs
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_mask_prob: float = 0.1,
        use_aria: bool = True,
    ):
        super().__init__(
            output_dim=12,  # 4 joints × 3
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            temporal_cond_dim=30,  # Full backbone
            global_cond_dim=512 if use_aria else 0,
            cond_mask_prob=cond_mask_prob,
        )
        self.use_aria = use_aria
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        backbone: torch.Tensor,
        aria: torch.Tensor = None,
    ) -> torch.Tensor:
        global_cond = aria if self.use_aria else None
        return super().forward(
            x, timesteps,
            temporal_cond=backbone,
            global_cond=global_cond,
        )


class RelativeLegMDM(ConditionedMDM):
    """
    Relative Leg MDM: Pelvis → Legs + Feet in RELATIVE coordinates
    
    Input conditions:
        - Pelvis: (B, T, 12) pelvis joints from PelvisMDM
        - ARIA: (B, 512) global music embedding
    
    Output: (B, T, 18) = relative coordinates of 6 joints:
        - RKNE, LKNE relative to Pelvis center (6 dim)
        - RANKLE, LANKLE relative to corresponding Knee (6 dim)
        - RTOE, LTOE relative to corresponding Ankle (6 dim)
    
    Learning relative coordinates dramatically reduces variance:
        - Toe rel to Ankle: 6.4k mm² (vs 76k for Knee rel to Pelvis)
        - Much easier to learn stable, realistic poses
    
    At inference time, convert back to absolute coordinates.
    """
    
    # Joint indices in output (relative coordinates)
    RKNE_REL = slice(0, 3)    # RKNE relative to Pelvis center
    LKNE_REL = slice(3, 6)    # LKNE relative to Pelvis center
    RANKLE_REL = slice(6, 9)  # RANKLE relative to RKNE
    LANKLE_REL = slice(9, 12) # LANKLE relative to LKNE
    RTOE_REL = slice(12, 15)  # RTOE relative to RANKLE
    LTOE_REL = slice(15, 18)  # LTOE relative to LANKLE
    
    def __init__(
        self,
        latent_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_mask_prob: float = 0.1,
        use_aria: bool = True,
    ):
        super().__init__(
            output_dim=18,  # 6 joints × 3 (relative)
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            temporal_cond_dim=12,  # Pelvis (4 joints × 3)
            global_cond_dim=512 if use_aria else 0,
            cond_mask_prob=cond_mask_prob,
        )
        self.use_aria = use_aria
        
        # Learn mean relative poses (initialized from data)
        self.register_buffer('mean_knee_rel', torch.zeros(6))
        self.register_buffer('mean_ankle_rel', torch.zeros(6))
        self.register_buffer('mean_toe_rel', torch.zeros(6))
    
    def set_mean_poses(self, knee_rel: torch.Tensor, ankle_rel: torch.Tensor, toe_rel: torch.Tensor):
        """Set learned mean relative poses from training data."""
        self.mean_knee_rel.copy_(knee_rel)
        self.mean_ankle_rel.copy_(ankle_rel)
        self.mean_toe_rel.copy_(toe_rel)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        pelvis: torch.Tensor,
        aria: torch.Tensor = None,
    ) -> torch.Tensor:
        global_cond = aria if self.use_aria else None
        return super().forward(
            x, timesteps,
            temporal_cond=pelvis,
            global_cond=global_cond,
        )
    
    @staticmethod
    def absolute_to_relative(
        pelvis: torch.Tensor,  # (B, T, 12) - 4 pelvis joints
        rkne: torch.Tensor,    # (B, T, 3)
        lkne: torch.Tensor,    # (B, T, 3)
        rankle: torch.Tensor,  # (B, T, 3)
        lankle: torch.Tensor,  # (B, T, 3)
        rtoe: torch.Tensor,    # (B, T, 3)
        ltoe: torch.Tensor,    # (B, T, 3)
    ) -> torch.Tensor:
        """Convert absolute positions to relative coordinates."""
        # Pelvis center
        pelvis_center = pelvis.view(*pelvis.shape[:-1], 4, 3).mean(dim=-2)  # (B, T, 3)
        
        # Relative positions
        rkne_rel = rkne - pelvis_center
        lkne_rel = lkne - pelvis_center
        rankle_rel = rankle - rkne
        lankle_rel = lankle - lkne
        rtoe_rel = rtoe - rankle
        ltoe_rel = ltoe - lankle
        
        # Concatenate
        return torch.cat([
            rkne_rel, lkne_rel,
            rankle_rel, lankle_rel,
            rtoe_rel, ltoe_rel
        ], dim=-1)  # (B, T, 18)
    
    @staticmethod
    def relative_to_absolute(
        pelvis: torch.Tensor,    # (B, T, 12) - 4 pelvis joints
        rel_coords: torch.Tensor, # (B, T, 18) - relative coordinates
    ) -> Dict[str, torch.Tensor]:
        """Convert relative coordinates back to absolute positions."""
        # Pelvis center
        pelvis_center = pelvis.view(*pelvis.shape[:-1], 4, 3).mean(dim=-2)  # (B, T, 3)
        
        # Extract relative coords
        rkne_rel = rel_coords[..., 0:3]
        lkne_rel = rel_coords[..., 3:6]
        rankle_rel = rel_coords[..., 6:9]
        lankle_rel = rel_coords[..., 9:12]
        rtoe_rel = rel_coords[..., 12:15]
        ltoe_rel = rel_coords[..., 15:18]
        
        # Reconstruct absolute positions (hierarchical)
        rkne = pelvis_center + rkne_rel
        lkne = pelvis_center + lkne_rel
        rankle = rkne + rankle_rel
        lankle = lkne + lankle_rel
        rtoe = rankle + rtoe_rel
        ltoe = lankle + ltoe_rel
        
        return {
            'rkne': rkne,
            'lkne': lkne,
            'rankle': rankle,
            'lankle': lankle,
            'rtoe': rtoe,
            'ltoe': ltoe,
        }


class RelativeLegLoss(nn.Module):
    """
    Loss for RelativeLegMDM with emphasis on hierarchical consistency.
    
    Key features:
    - Position loss on relative coordinates
    - Bone length constraints (knee-ankle, ankle-toe)
    - Temporal smoothness
    - Mean pose regularization
    """
    
    # Expected bone lengths (meters)
    BONE_LENGTHS = {
        'knee_ankle': 0.38,  # Knee to Ankle
        'ankle_toe': 0.15,   # Ankle to Toe
    }
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        vel_weight: float = 0.3,
        bone_weight: float = 1.0,
        smooth_weight: float = 0.1,
        mean_pose_weight: float = 0.05,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.bone_weight = bone_weight
        self.smooth_weight = smooth_weight
        self.mean_pose_weight = mean_pose_weight
    
    def forward(
        self,
        pred: torch.Tensor,   # (B, T, 18) relative coords
        target: torch.Tensor, # (B, T, 18) relative coords
        mean_pose: torch.Tensor = None,  # (18,) mean relative pose
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        
        # 1. Position loss
        pos_loss = F.mse_loss(pred, target)
        loss_dict['pos'] = pos_loss.item()
        
        # 2. Velocity loss
        pred_vel = pred[:, 1:] - pred[:, :-1]
        tgt_vel = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(pred_vel, tgt_vel)
        loss_dict['vel'] = vel_loss.item()
        
        # 3. Bone length loss (on relative coords that represent bones)
        # Ankle relative to Knee = bone vector
        pred_rankle_bone = pred[..., 6:9]   # RANKLE rel to RKNE
        pred_lankle_bone = pred[..., 9:12]  # LANKLE rel to LKNE
        pred_rtoe_bone = pred[..., 12:15]   # RTOE rel to RANKLE
        pred_ltoe_bone = pred[..., 15:18]   # LTOE rel to LANKLE
        
        tgt_rankle_bone = target[..., 6:9]
        tgt_lankle_bone = target[..., 9:12]
        tgt_rtoe_bone = target[..., 12:15]
        tgt_ltoe_bone = target[..., 15:18]
        
        # Bone length matching
        pred_knee_ankle_len = torch.cat([
            pred_rankle_bone.norm(dim=-1, keepdim=True),
            pred_lankle_bone.norm(dim=-1, keepdim=True)
        ], dim=-1)
        tgt_knee_ankle_len = torch.cat([
            tgt_rankle_bone.norm(dim=-1, keepdim=True),
            tgt_lankle_bone.norm(dim=-1, keepdim=True)
        ], dim=-1)
        
        pred_ankle_toe_len = torch.cat([
            pred_rtoe_bone.norm(dim=-1, keepdim=True),
            pred_ltoe_bone.norm(dim=-1, keepdim=True)
        ], dim=-1)
        tgt_ankle_toe_len = torch.cat([
            tgt_rtoe_bone.norm(dim=-1, keepdim=True),
            tgt_ltoe_bone.norm(dim=-1, keepdim=True)
        ], dim=-1)
        
        bone_loss = (
            F.mse_loss(pred_knee_ankle_len, tgt_knee_ankle_len) +
            F.mse_loss(pred_ankle_toe_len, tgt_ankle_toe_len)
        )
        loss_dict['bone'] = bone_loss.item()
        
        # 4. Smoothness loss (jerk)
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        pred_jerk = pred_acc[:, 1:] - pred_acc[:, :-1]
        tgt_acc = tgt_vel[:, 1:] - tgt_vel[:, :-1]
        tgt_jerk = tgt_acc[:, 1:] - tgt_acc[:, :-1]
        smooth_loss = F.mse_loss(pred_jerk, tgt_jerk)
        loss_dict['smooth'] = smooth_loss.item()
        
        # 5. Mean pose regularization (optional)
        mean_pose_loss = torch.tensor(0.0, device=pred.device)
        if mean_pose is not None:
            pred_mean = pred.mean(dim=1)  # (B, 18)
            mean_pose_loss = F.mse_loss(pred_mean, mean_pose.unsqueeze(0).expand(pred.shape[0], -1))
        loss_dict['mean_pose'] = mean_pose_loss.item()
        
        # Total
        total = (
            self.pos_weight * pos_loss +
            self.vel_weight * vel_loss +
            self.bone_weight * bone_loss +
            self.smooth_weight * smooth_loss +
            self.mean_pose_weight * mean_pose_loss
        )
        loss_dict['total'] = total.item()
        
        return total, loss_dict


class PelvisLoss(nn.Module):
    """
    Loss for PelvisMDM - simple and stable.
    
    Pelvis is relatively static (2.4x more stable than torso),
    so we use straightforward position matching with temporal smoothness.
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        vel_weight: float = 0.5,
        smooth_weight: float = 0.2,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.smooth_weight = smooth_weight
    
    def forward(
        self,
        pred: torch.Tensor,   # (B, T, 12)
        target: torch.Tensor, # (B, T, 12)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        
        # Position
        pos_loss = F.mse_loss(pred, target)
        loss_dict['pos'] = pos_loss.item()
        
        # Velocity
        pred_vel = pred[:, 1:] - pred[:, :-1]
        tgt_vel = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(pred_vel, tgt_vel)
        loss_dict['vel'] = vel_loss.item()
        
        # Smoothness (acceleration)
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        tgt_acc = tgt_vel[:, 1:] - tgt_vel[:, :-1]
        smooth_loss = F.mse_loss(pred_acc, tgt_acc)
        loss_dict['smooth'] = smooth_loss.item()
        
        # Total
        total = (
            self.pos_weight * pos_loss +
            self.vel_weight * vel_loss +
            self.smooth_weight * smooth_loss
        )
        loss_dict['total'] = total.item()
        
        return total, loss_dict


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on {device}...")
    
    B, T = 2, 120
    t = torch.randint(0, 1000, (B,)).to(device)
    aria = torch.randn(B, 512).to(device)
    
    # HandMDM
    print("\n1. HandMDM")
    model = HandMDM(latent_dim=256, num_layers=4).to(device)
    x = torch.randn(B, T, 12).to(device)
    midi = torch.rand(B, T, 88).to(device)
    out = model(x, t, midi, aria)
    print(f"   Output: {out.shape}, Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # BackboneMDM
    print("\n2. BackboneMDM")
    model = BackboneMDM(latent_dim=256, num_layers=4).to(device)
    x = torch.randn(B, T, 30).to(device)
    hand = torch.randn(B, T, 12).to(device)
    out = model(x, t, hand, aria)
    print(f"   Output: {out.shape}, Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # BackboneMDMLoss test
    print("\n3. BackboneMDMLoss")
    criterion = BackboneMDMLoss()
    pred = torch.randn(B, T, 30).to(device)
    target = torch.randn(B, T, 30).to(device)
    loss, loss_dict = criterion(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Dict: {loss_dict}")
    
    # LowerMDM
    print("\n4. LowerMDM")
    model = LowerMDM(latent_dim=128, num_layers=3).to(device)
    x = torch.randn(B, T, 30).to(device)
    torso = torch.randn(B, T, 6).to(device)
    out = model(x, t, torso, aria)
    print(f"   Output: {out.shape}, Params: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nAll tests passed!")


# =============================================================================
# FiLM-based Lower Body Generator (Alternative to Diffusion)
# =============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""
    
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) features
            cond: (B, T, C) or (B, C) conditioning
        """
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, x.shape[1], -1)
        
        gamma = self.gamma(cond)  # (B, T, D)
        beta = self.beta(cond)    # (B, T, D)
        
        return gamma * x + beta


class MeanPoseLowerGenerator(nn.Module):
    """
    FiLM-based Lower Body Generator.
    
    Strategy: Start from learned mean pose, refine with FiLM conditioning.
    
    Advantages over diffusion:
    - Stable output (no mode collapse to wrong mean)
    - Fast inference (single forward pass)
    - Anatomically valid by design (mean pose is valid)
    
    Architecture:
        1. Mean pose (learned from data) as base
        2. FiLM layers modulate features based on backbone/aria/midi
        3. Output: mean pose + small offset
    
    Output: (B, T, 30) = 10 lower body joints × 3
        [RASI, LASI, RPSI, LPSI, RKNE, LKNE, RANKLE, LANKLE, RTOE, LTOE]
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        backbone_dim: int = 30,
        aria_dim: int = 512,
        output_dim: int = 30,
        max_offset: float = 0.1,  # Max offset from mean pose (meters)
        use_aria: bool = True,
        use_midi: bool = False,
        midi_dim: int = 88,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_offset = max_offset
        self.use_aria = use_aria
        self.use_midi = use_midi
        
        # Condition projection
        self.backbone_proj = nn.Linear(backbone_dim, hidden_dim)
        if use_aria:
            self.aria_proj = nn.Linear(aria_dim, hidden_dim)
        else:
            self.aria_proj = None
        
        if use_midi:
            self.midi_proj = nn.Linear(midi_dim, hidden_dim)
        else:
            self.midi_proj = None
        
        # Initial feature from mean pose
        self.mean_proj = nn.Linear(output_dim, hidden_dim)
        
        # FiLM layers
        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Condition dimension: backbone + optional aria + optional midi
        cond_dim = hidden_dim  # backbone
        if use_aria:
            cond_dim += hidden_dim
        if use_midi:
            cond_dim += hidden_dim
        
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.film_layers.append(FiLMLayer(hidden_dim, cond_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection (predicts offset from mean)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # Bound output to [-1, 1], then scale by max_offset
        )
        
        # Learnable mean pose (initialized from data)
        self.register_buffer('mean_pose', torch.zeros(output_dim))
        
        # Motion range per dimension (how much each dim actually moves in GT)
        # This bounds the refinement to realistic ranges
        self.register_buffer('motion_range', torch.ones(output_dim) * 0.05)
    
    def set_mean_pose(self, mean: torch.Tensor, motion_range: torch.Tensor = None):
        """Set mean pose and motion range from training data."""
        self.mean_pose.copy_(mean)
        if motion_range is not None:
            # Clamp to reasonable minimum
            self.motion_range.copy_(motion_range.clamp(min=0.005))
    
    def forward(
        self,
        backbone: torch.Tensor,  # (B, T, 30)
        aria: Optional[torch.Tensor] = None,  # (B, 512)
        midi: Optional[torch.Tensor] = None,  # (B, T, 88)
    ) -> torch.Tensor:
        """
        Generate lower body from backbone + optional midi conditioning.
        
        Returns:
            (B, T, 30) lower body positions
        """
        B, T, _ = backbone.shape
        
        # Project conditions
        backbone_feat = self.backbone_proj(backbone)  # (B, T, D)
        cond_list = [backbone_feat]
        
        if self.use_aria and self.aria_proj is not None and aria is not None:
            aria_feat = self.aria_proj(aria).unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
            cond_list.append(aria_feat)
        
        if self.use_midi and self.midi_proj is not None and midi is not None:
            midi_feat = self.midi_proj(midi)  # (B, T, D)
            cond_list.append(midi_feat)
        
        cond = torch.cat(cond_list, dim=-1)  # (B, T, cond_dim)
        
        # Start from mean pose
        mean_expanded = self.mean_pose.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        x = self.mean_proj(mean_expanded)  # (B, T, D)
        
        # FiLM modulated layers
        for layer, film, norm in zip(self.layers, self.film_layers, self.norms):
            residual = x
            x = layer(x)
            x = film(x, cond)
            x = norm(x + residual)
        
        # Output offset (bounded by GT motion range)
        offset = self.out_proj(x)  # (B, T, 30), in [-1, 1]
        offset = offset * self.motion_range  # Scale by actual GT motion range
        
        # Final output = mean + offset
        output = mean_expanded + offset
        
        return output


class MeanPoseLowerLoss(nn.Module):
    """
    Loss for MeanPoseLowerGenerator.
    
    Focus on:
    - Position accuracy
    - Temporal smoothness
    - Bone length preservation
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        vel_weight: float = 0.5,
        smooth_weight: float = 0.2,
        bone_weight: float = 0.5,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.smooth_weight = smooth_weight
        self.bone_weight = bone_weight
    
    def forward(
        self,
        pred: torch.Tensor,  # (B, T, 30)
        target: torch.Tensor,  # (B, T, 30)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Position loss
        pos_loss = F.mse_loss(pred, target)
        
        # Velocity loss
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(pred_vel, target_vel)
        
        # Smoothness (jerk penalty)
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_acc = target_vel[:, 1:] - target_vel[:, :-1]
        smooth_loss = F.mse_loss(pred_acc, target_acc)
        
        # Bone length loss
        # Reshape to (B, T, 10, 3)
        pred_joints = pred.view(*pred.shape[:-1], 10, 3)
        target_joints = target.view(*target.shape[:-1], 10, 3)
        
        # Knee-Ankle bones (indices 4,5 = knees, 6,7 = ankles)
        pred_knee_ankle_r = (pred_joints[:, :, 6] - pred_joints[:, :, 4]).norm(dim=-1)
        pred_knee_ankle_l = (pred_joints[:, :, 7] - pred_joints[:, :, 5]).norm(dim=-1)
        target_knee_ankle_r = (target_joints[:, :, 6] - target_joints[:, :, 4]).norm(dim=-1)
        target_knee_ankle_l = (target_joints[:, :, 7] - target_joints[:, :, 5]).norm(dim=-1)
        
        bone_loss = (
            F.mse_loss(pred_knee_ankle_r, target_knee_ankle_r) +
            F.mse_loss(pred_knee_ankle_l, target_knee_ankle_l)
        )
        
        total = (
            self.pos_weight * pos_loss +
            self.vel_weight * vel_loss +
            self.smooth_weight * smooth_loss +
            self.bone_weight * bone_loss
        )
        
        return total, {
            'pos': pos_loss.item(),
            'vel': vel_loss.item(),
            'smooth': smooth_loss.item(),
            'bone': bone_loss.item(),
            'total': total.item(),
        }


# ============================================================
# ELBOW MDM - Shoulder + Wrist -> Elbow
# ============================================================

class ElbowMDM(ConditionedMDM):
    """
    Elbow MDM: Wrist-relative-to-Shoulder + ARIA -> Elbow-relative-to-Shoulder
    
    Uses FULLY RELATIVE coordinates to be robust to shoulder position errors.
    
    Input conditions:
        - Wrist relative to Shoulder: (B, T, 6) = [(RWRIST-RSHO), (LWRIST-LSHO)] x 3 coords
        - ARIA: (B, 512) global music embedding
    
    Output: (B, T, 6) = [RELB-RSHO, LELB-LSHO] x 3 coords (RELATIVE to shoulder)
    
    NOTE: 
        - Training: Input (wrist - shoulder), Output (elbow - shoulder)
        - Inference: Compute (wrist - shoulder), run model, add shoulder to output
    
    This architecture is INVARIANT to shoulder absolute position.
    
    Args:
        use_relative: If True (default), use shoulder-relative coordinates.
                      If False, use absolute coordinates (for ablation study).
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_mask_prob: float = 0.1,
        use_aria: bool = True,
        use_relative: bool = True,
    ):
        self.use_relative = use_relative
        
        # Spatial condition dimension depends on use_relative
        # Relative: 6 (wrist-shoulder)
        # Absolute: 12 (shoulder + wrist)
        spatial_dim = 6 if use_relative else 12
        
        super().__init__(
            output_dim=6,  # 2 elbows x 3 coords
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_mask_prob=cond_mask_prob,
            temporal_cond_dim=0,
            global_cond_dim=512 if use_aria else 0,
            spatial_cond_dim=spatial_dim,
        )
        self.use_aria = use_aria
        self._init_output_bias()
    
    def _init_output_bias(self):
        """Initialize output bias to reasonable elbow positions.
        
        For relative mode (use_relative=True):
            Values based on MOSA dataset GT statistics (Elbow - Shoulder):
            - RELB-RSHO mean: (~0.01, 0.05, -0.26) meters
            - LELB-LSHO mean: (~-0.05, 0.05, -0.25) meters
        
        For absolute mode (use_relative=False):
            Values based on MOSA dataset GT absolute elbow positions:
            - RELB mean: (~0.15, 0.65, 0.95) meters
            - LELB mean: (~-0.15, 0.65, 0.95) meters
        """
        if self.use_relative:
            # Relative positions (elbow offset from shoulder)
            default_elbow = torch.tensor([
                0.01, 0.05, -0.26,   # RELB relative to RSHO
                -0.05, 0.05, -0.25,  # LELB relative to LSHO
            ])
        else:
            # Absolute positions
            default_elbow = torch.tensor([
                0.15, 0.65, 0.95,   # RELB absolute
                -0.15, 0.65, 0.95,  # LELB absolute
            ])
        with torch.no_grad():
            self.output_proj.bias.copy_(default_elbow)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        shoulder: torch.Tensor,
        wrist: torch.Tensor,
        aria: Optional[torch.Tensor] = None,
        force_mask: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, 6) noisy elbow positions (relative if use_relative, else absolute)
            timesteps: (B,) diffusion timesteps
            shoulder: (B, T, 6) [RSHO, LSHO] positions (absolute)
            wrist: (B, T, 6) [RWRIST, LWRIST] positions (absolute)
            aria: (B, 512) ARIA embedding (optional)
            force_mask: Force mask conditions for CFG
        
        Returns:
            pred: (B, T, 6) predicted clean elbow positions
                  If use_relative=True: returns relative positions (add shoulder to get absolute)
                  If use_relative=False: returns absolute positions
        """
        if self.use_relative:
            # Compute wrist RELATIVE to shoulder
            # This makes the model invariant to shoulder absolute position
            spatial_cond = wrist - shoulder  # (B, T, 6)
        else:
            # Use absolute coordinates (for ablation study)
            # Concatenate shoulder and wrist as conditions
            spatial_cond = torch.cat([shoulder, wrist], dim=-1)  # (B, T, 12)
        
        return super().forward(
            x, timesteps,
            spatial_cond=spatial_cond,
            global_cond=aria if self.use_aria else None,
            force_mask=force_mask,
        )


class ElbowLoss(nn.Module):
    """
    Loss function for ElbowMDM training.
    
    Components:
        1. Position MSE: direct position matching
        2. Velocity MSE: temporal smoothness and motion capture
        3. Bone length: maintain arm segment lengths
        4. Acceleration (smoothness): reduce jitter
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        vel_weight: float = 0.5,
        bone_weight: float = 0.5,
        smooth_weight: float = 0.1,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.bone_weight = bone_weight
        self.smooth_weight = smooth_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        shoulder: torch.Tensor,
        wrist: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: (B, T, 6) predicted elbow [RELB, LELB]
            target: (B, T, 6) ground truth elbow
            shoulder: (B, T, 6) shoulder positions [RSHO, LSHO]
            wrist: (B, T, 6) wrist positions [RWRIST, LWRIST]
        
        Returns:
            total_loss: scalar
            loss_dict: breakdown of losses
        """
        # 1. Position loss
        pos_loss = F.mse_loss(pred, target)
        
        # 2. Velocity loss
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(pred_vel, target_vel)
        
        # 3. Smoothness loss (acceleration)
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_acc = target_vel[:, 1:] - target_vel[:, :-1]
        smooth_loss = F.mse_loss(pred_acc, target_acc)
        
        # 4. Bone length loss (upper arm: shoulder-elbow, lower arm: elbow-wrist)
        # NOTE: pred and target are RELATIVE coordinates (elbow - shoulder)
        # So the relative coord norm IS the upper arm length!
        
        # Extract relative elbow coords
        pred_relb_rel = pred[:, :, :3]  # (B, T, 3) - RELB relative to RSHO
        pred_lelb_rel = pred[:, :, 3:]  # (B, T, 3) - LELB relative to LSHO
        target_relb_rel = target[:, :, :3]
        target_lelb_rel = target[:, :, 3:]
        
        # Shoulder and wrist (absolute coords)
        rsho = shoulder[:, :, :3]
        lsho = shoulder[:, :, 3:]
        rwrist = wrist[:, :, :3]
        lwrist = wrist[:, :, 3:]
        
        # Upper arm lengths = |relative elbow coord| (distance from shoulder to elbow)
        gt_r_upper = target_relb_rel.norm(dim=-1, keepdim=True)
        gt_l_upper = target_lelb_rel.norm(dim=-1, keepdim=True)
        pred_r_upper = pred_relb_rel.norm(dim=-1, keepdim=True)
        pred_l_upper = pred_lelb_rel.norm(dim=-1, keepdim=True)
        
        # Lower arm lengths = |wrist - absolute_elbow| = |wrist - (shoulder + relative_elbow)|
        # absolute_elbow = shoulder + relative_elbow
        pred_relb_abs = rsho + pred_relb_rel
        pred_lelb_abs = lsho + pred_lelb_rel
        target_relb_abs = rsho + target_relb_rel
        target_lelb_abs = lsho + target_lelb_rel
        
        gt_r_lower = (rwrist - target_relb_abs).norm(dim=-1, keepdim=True)
        gt_l_lower = (lwrist - target_lelb_abs).norm(dim=-1, keepdim=True)
        pred_r_lower = (rwrist - pred_relb_abs).norm(dim=-1, keepdim=True)
        pred_l_lower = (lwrist - pred_lelb_abs).norm(dim=-1, keepdim=True)
        
        bone_loss = (
            F.mse_loss(pred_r_upper, gt_r_upper) +
            F.mse_loss(pred_l_upper, gt_l_upper) +
            F.mse_loss(pred_r_lower, gt_r_lower) +
            F.mse_loss(pred_l_lower, gt_l_lower)
        ) / 4
        
        # Total
        total = (
            self.pos_weight * pos_loss +
            self.vel_weight * vel_loss +
            self.smooth_weight * smooth_loss +
            self.bone_weight * bone_loss
        )
        
        return total, {
            'pos': pos_loss.item(),
            'vel': vel_loss.item(),
            'smooth': smooth_loss.item(),
            'bone': bone_loss.item(),
            'total': total.item(),
        }


# ============================================================
# SWIVEL ANGLE PREDICTOR - IK-based Elbow Generation
# ============================================================

def compute_elbow_ik(
    shoulder: torch.Tensor,
    wrist: torch.Tensor,
    upper_len: torch.Tensor,
    lower_len: torch.Tensor,
    swivel_angle: torch.Tensor,
) -> torch.Tensor:
    """
    Compute elbow position using IK with swivel angle.
    
    This guarantees correct bone lengths while allowing the network
    to control only the swivel angle (elbow rotation around shoulder-wrist axis).
    
    Args:
        shoulder: (B, T, 3) shoulder position
        wrist: (B, T, 3) wrist position
        upper_len: (B, 1, 1) or scalar - upper arm length (shoulder to elbow)
        lower_len: (B, 1, 1) or scalar - lower arm length (elbow to wrist)
        swivel_angle: (B, T, 1) swivel angle in radians
    
    Returns:
        elbow: (B, T, 3) elbow position
    """
    # Vector from shoulder to wrist
    sw = wrist - shoulder  # (B, T, 3)
    sw_len = sw.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, T, 1)
    sw_dir = sw / sw_len  # (B, T, 3) - unit vector
    
    # Check if arm can reach (triangle inequality)
    # Clamp sw_len to valid range
    max_reach = upper_len + lower_len
    min_reach = (upper_len - lower_len).abs()
    sw_len_clamped = sw_len.clamp(min=min_reach + 1e-4, max=max_reach - 1e-4)
    
    # Use law of cosines to find elbow position along the arm
    # cos(theta) = (upper^2 + sw^2 - lower^2) / (2 * upper * sw)
    cos_theta = (upper_len**2 + sw_len_clamped**2 - lower_len**2) / (2 * upper_len * sw_len_clamped + 1e-6)
    cos_theta = cos_theta.clamp(-1 + 1e-6, 1 - 1e-6)
    
    # Distance from shoulder to elbow projection on sw axis
    proj_len = upper_len * cos_theta  # (B, T, 1)
    
    # Perpendicular distance from sw axis to elbow
    perp_len = (upper_len**2 - proj_len**2).clamp(min=0).sqrt()  # (B, T, 1)
    
    # Create orthonormal basis for the plane perpendicular to sw
    # Use world up (0, 1, 0) as reference, but handle edge cases
    world_up = torch.zeros_like(sw)
    world_up[..., 1] = 1.0  # Y-up
    
    # First perpendicular vector (remove sw component from up)
    up_proj = (world_up * sw_dir).sum(dim=-1, keepdim=True) * sw_dir
    perp1 = world_up - up_proj
    perp1_len = perp1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    perp1 = perp1 / perp1_len
    
    # Handle case where sw is parallel to up (use X axis instead)
    parallel_mask = perp1_len.squeeze(-1) < 0.1
    if parallel_mask.any():
        world_x = torch.zeros_like(sw)
        world_x[..., 0] = 1.0
        x_proj = (world_x * sw_dir).sum(dim=-1, keepdim=True) * sw_dir
        perp1_alt = world_x - x_proj
        perp1_alt = perp1_alt / perp1_alt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        perp1 = torch.where(parallel_mask.unsqueeze(-1), perp1_alt, perp1)
    
    # Second perpendicular vector (cross product)
    perp2 = torch.cross(sw_dir, perp1, dim=-1)  # (B, T, 3)
    
    # Rotate around sw axis by swivel_angle
    cos_swivel = torch.cos(swivel_angle)  # (B, T, 1)
    sin_swivel = torch.sin(swivel_angle)  # (B, T, 1)
    
    # Elbow offset from shoulder in perpendicular plane
    elbow_offset = perp1 * (perp_len * cos_swivel) + perp2 * (perp_len * sin_swivel)
    
    # Elbow position
    elbow = shoulder + sw_dir * proj_len + elbow_offset
    
    return elbow


def compute_gt_swivel_angle(
    shoulder: torch.Tensor,
    elbow: torch.Tensor,
    wrist: torch.Tensor,
) -> torch.Tensor:
    """
    Compute swivel angle from GT positions (for training target).
    
    Args:
        shoulder: (B, T, 3)
        elbow: (B, T, 3)
        wrist: (B, T, 3)
    
    Returns:
        swivel_angle: (B, T, 1) in radians
    """
    # Vector from shoulder to wrist
    sw = wrist - shoulder
    sw_len = sw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    sw_dir = sw / sw_len
    
    # Vector from shoulder to elbow
    se = elbow - shoulder
    
    # Project elbow onto sw axis
    se_proj_len = (se * sw_dir).sum(dim=-1, keepdim=True)
    se_proj = sw_dir * se_proj_len
    
    # Perpendicular component (elbow offset from sw axis)
    se_perp = se - se_proj  # (B, T, 3)
    se_perp_len = se_perp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    
    # Normalize
    se_perp_dir = se_perp / se_perp_len
    
    # Create reference basis (same as in compute_elbow_ik)
    world_up = torch.zeros_like(sw)
    world_up[..., 1] = 1.0
    
    up_proj = (world_up * sw_dir).sum(dim=-1, keepdim=True) * sw_dir
    perp1 = world_up - up_proj
    perp1_len = perp1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    perp1 = perp1 / perp1_len
    
    # Handle parallel case
    parallel_mask = perp1_len.squeeze(-1) < 0.1
    if parallel_mask.any():
        world_x = torch.zeros_like(sw)
        world_x[..., 0] = 1.0
        x_proj = (world_x * sw_dir).sum(dim=-1, keepdim=True) * sw_dir
        perp1_alt = world_x - x_proj
        perp1_alt = perp1_alt / perp1_alt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        perp1 = torch.where(parallel_mask.unsqueeze(-1), perp1_alt, perp1)
    
    perp2 = torch.cross(sw_dir, perp1, dim=-1)
    
    # Compute swivel angle using atan2
    cos_component = (se_perp_dir * perp1).sum(dim=-1, keepdim=True)
    sin_component = (se_perp_dir * perp2).sum(dim=-1, keepdim=True)
    
    swivel_angle = torch.atan2(sin_component, cos_component)
    
    return swivel_angle


class SwivelAnglePredictor(nn.Module):
    """
    Predicts swivel angle for elbow IK.
    
    Simple deterministic model (no diffusion needed).
    Input: Wrist-relative-to-Shoulder + ARIA
    Output: Swivel angle for left/right elbows
    
    This is combined with analytical IK to get final elbow positions,
    guaranteeing correct bone lengths.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_aria: bool = True,
    ):
        super().__init__()
        
        self.use_aria = use_aria
        self.hidden_dim = hidden_dim
        
        # Input: wrist_rel (6) + optional aria (512)
        input_dim = 6
        if use_aria:
            self.aria_proj = nn.Linear(512, hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output: 2 swivel angles (left, right)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),  # Output in [-1, 1], then scale to [-pi, pi]
        )
        
        # Initialize output bias to typical swivel angles
        # Piano playing typically has elbows slightly outward
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        wrist_rel: torch.Tensor,
        aria: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            wrist_rel: (B, T, 6) wrist relative to shoulder [RWRI-RSHO, LWRI-LSHO]
            aria: (B, 512) ARIA embedding
        
        Returns:
            swivel_angles: (B, T, 2) swivel angles in radians [-pi, pi]
        """
        B, T, _ = wrist_rel.shape
        
        # Project input
        x = self.input_proj(wrist_rel)  # (B, T, hidden)
        
        # Add ARIA conditioning if available
        if self.use_aria and aria is not None:
            aria_feat = self.aria_proj(aria)  # (B, hidden)
            aria_feat = aria_feat.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden)
            x = x + aria_feat
        
        # Transformer
        x = self.transformer(x)  # (B, T, hidden)
        
        # Output
        swivel = self.output_proj(x)  # (B, T, 2) in [-1, 1]
        swivel = swivel * 3.14159  # Scale to [-pi, pi]
        
        return swivel


class SwivelElbowLoss(nn.Module):
    """
    Loss function for SwivelAnglePredictor.
    
    Computes loss on:
    1. Swivel angle directly
    2. Resulting elbow position (after IK)
    3. Temporal smoothness
    """
    
    def __init__(
        self,
        angle_weight: float = 1.0,
        pos_weight: float = 1.0,
        smooth_weight: float = 0.1,
    ):
        super().__init__()
        self.angle_weight = angle_weight
        self.pos_weight = pos_weight
        self.smooth_weight = smooth_weight
    
    def forward(
        self,
        pred_angle: torch.Tensor,
        target_angle: torch.Tensor,
        pred_elbow: torch.Tensor,
        target_elbow: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_angle: (B, T, 2) predicted swivel angles
            target_angle: (B, T, 2) GT swivel angles
            pred_elbow: (B, T, 2, 3) predicted elbow positions (from IK)
            target_elbow: (B, T, 2, 3) GT elbow positions
        """
        # 1. Angle loss (circular - handle wrap-around)
        angle_diff = pred_angle - target_angle
        # Wrap to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angle_loss = (angle_diff ** 2).mean()
        
        # 2. Position loss
        pos_loss = F.mse_loss(pred_elbow, target_elbow)
        
        # 3. Smoothness loss (angle acceleration)
        pred_vel = pred_angle[:, 1:] - pred_angle[:, :-1]
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        smooth_loss = (pred_acc ** 2).mean()
        
        # Total
        total = (
            self.angle_weight * angle_loss +
            self.pos_weight * pos_loss +
            self.smooth_weight * smooth_loss
        )
        
        return total, {
            'angle': angle_loss.item(),
            'pos': pos_loss.item(),
            'smooth': smooth_loss.item(),
            'total': total.item(),
        }


# ============================================================
# NEW PIPELINE: Hand → Lower → Backbone (v2)
# Based on analysis showing Pelvis-Backbone correlation (r=0.82)
# is stronger than Hand-Backbone correlation (r=0.62)
# ============================================================

class HandConditionedLowerGenerator(nn.Module):
    """
    FiLM-based Lower Body Generator conditioned on HAND motion.
    
    NEW PIPELINE ORDER: Hand → Lower → Backbone
    
    Analysis showed:
    - Pelvis-Backbone correlation: r = 0.82 (very strong)
    - Hand-Backbone correlation: r = 0.62 (strong)
    - Hand cannot predict Pelvis well: R² = 0.32
    
    This means Lower Body contains independent information not in Hand,
    which is useful for Backbone generation.
    
    Input:
        - Hand: (B, T, 12) wrist + finger positions
        - ARIA: (B, 512) global music embedding
        - MIDI: (B, T, 88) optional frame-level piano roll
    
    Output: (B, T, 30) lower body positions
        [RASI, LASI, RPSI, LPSI, RKNE, LKNE, RANKLE, LANKLE, RTOE, LTOE]
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        hand_dim: int = 12,
        aria_dim: int = 512,
        output_dim: int = 30,
        use_aria: bool = True,
        use_midi: bool = False,
        midi_dim: int = 88,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_aria = use_aria
        self.use_midi = use_midi
        
        # Hand projection
        self.hand_proj = nn.Linear(hand_dim, hidden_dim)
        
        # ARIA projection
        if use_aria:
            self.aria_proj = nn.Linear(aria_dim, hidden_dim)
        else:
            self.aria_proj = None
        
        # MIDI projection
        if use_midi:
            self.midi_proj = nn.Linear(midi_dim, hidden_dim)
        else:
            self.midi_proj = None
        
        # Mean pose projection
        self.mean_proj = nn.Linear(output_dim, hidden_dim)
        
        # FiLM layers
        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Condition dimension
        cond_dim = hidden_dim  # hand
        if use_aria:
            cond_dim += hidden_dim
        if use_midi:
            cond_dim += hidden_dim
        
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.film_layers.append(FiLMLayer(hidden_dim, cond_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        
        # Learnable mean pose
        self.register_buffer('mean_pose', torch.zeros(output_dim))
        self.register_buffer('motion_range', torch.ones(output_dim) * 0.05)
    
    def set_mean_pose(self, mean: torch.Tensor, motion_range: torch.Tensor = None):
        """Set mean pose and motion range from training data."""
        self.mean_pose.copy_(mean)
        if motion_range is not None:
            self.motion_range.copy_(motion_range.clamp(min=0.005))
    
    def forward(
        self,
        hand: torch.Tensor,  # (B, T, 12)
        aria: Optional[torch.Tensor] = None,  # (B, 512)
        midi: Optional[torch.Tensor] = None,  # (B, T, 88)
    ) -> torch.Tensor:
        """Generate lower body from hand motion."""
        B, T, _ = hand.shape
        
        # Project conditions
        hand_feat = self.hand_proj(hand)  # (B, T, D)
        cond_list = [hand_feat]
        
        if self.use_aria and self.aria_proj is not None and aria is not None:
            aria_feat = self.aria_proj(aria).unsqueeze(1).expand(-1, T, -1)
            cond_list.append(aria_feat)
        
        if self.use_midi and self.midi_proj is not None and midi is not None:
            midi_feat = self.midi_proj(midi)
            cond_list.append(midi_feat)
        
        cond = torch.cat(cond_list, dim=-1)
        
        # Start from mean pose
        mean_expanded = self.mean_pose.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        x = self.mean_proj(mean_expanded)
        
        # FiLM modulated layers
        for layer, film, norm in zip(self.layers, self.film_layers, self.norms):
            residual = x
            x = layer(x)
            x = film(x, cond)
            x = norm(x + residual)
        
        # Output offset
        offset = self.out_proj(x)
        offset = offset * self.motion_range
        
        return mean_expanded + offset


class HandConditionedLowerMLP(nn.Module):
    """
    MLP-based Lower Body Generator conditioned on Hand motion.
    
    Uses frame-wise MLP with sliding window context, same architecture
    as the benchmark MLPRegressor but conditioned on Hand + ARIA + MIDI
    instead of MIDI-only, and outputs only Lower Body (30 dims).
    
    Input:
        - Hand: (B, T, 12) wrist + finger positions
        - ARIA: (B, 512) global music embedding (optional)
        - MIDI: (B, T, 88) frame-level piano roll (optional)
    
    Output: (B, T, 30) lower body positions
    """
    
    def __init__(
        self,
        hand_dim: int = 12,
        aria_dim: int = 512,
        midi_dim: int = 88,
        output_dim: int = 30,
        hidden_dims: list = None,
        use_aria: bool = True,
        use_midi: bool = False,
        use_context: bool = True,
        context_frames: int = 5,
        dropout: float = 0.1,
        # Accept but ignore FiLM-specific args for compatibility
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 256, 256]
        
        self.output_dim = output_dim
        self.use_aria = use_aria
        self.use_midi = use_midi
        self.use_context = use_context
        self.context_frames = context_frames
        
        # Compute per-frame input dim
        frame_dim = hand_dim  # 12
        if use_midi:
            frame_dim += midi_dim  # + 88
        if use_aria:
            frame_dim += aria_dim  # + 512
        
        # With context, multiply by window size
        if use_context:
            input_dim = frame_dim * (2 * context_frames + 1)
        else:
            input_dim = frame_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Mean pose / motion range (for compatibility with FiLM version)
        self.register_buffer('mean_pose', torch.zeros(output_dim))
        self.register_buffer('motion_range', torch.ones(output_dim) * 0.05)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def set_mean_pose(self, mean: torch.Tensor, motion_range: torch.Tensor = None):
        """Set mean pose and motion range from training data."""
        self.mean_pose.copy_(mean)
        if motion_range is not None:
            self.motion_range.copy_(motion_range.clamp(min=0.005))
    
    def _add_context(self, x: torch.Tensor) -> torch.Tensor:
        """Add sliding window context [B, T, D] -> [B, T, D * (2k+1)]"""
        B, T, D = x.shape
        k = self.context_frames
        x_padded = F.pad(x, (0, 0, k, k), mode='replicate')
        windows = []
        for i in range(2 * k + 1):
            windows.append(x_padded[:, i:i+T])
        return torch.cat(windows, dim=-1)
    
    def forward(
        self,
        hand: torch.Tensor,       # (B, T, 12)
        aria: Optional[torch.Tensor] = None,  # (B, 512)
        midi: Optional[torch.Tensor] = None,  # (B, T, 88)
    ) -> torch.Tensor:
        """Generate lower body from hand motion using frame-wise MLP."""
        B, T, _ = hand.shape
        
        # Build per-frame features
        feat_list = [hand]  # (B, T, 12)
        
        if self.use_midi and midi is not None:
            feat_list.append(midi)  # (B, T, 88)
        
        if self.use_aria and aria is not None:
            if aria.ndim == 2:
                aria_expanded = aria.unsqueeze(1).expand(-1, T, -1)  # (B, T, 512)
            else:
                aria_expanded = aria
            feat_list.append(aria_expanded)
        
        x = torch.cat(feat_list, dim=-1)  # (B, T, frame_dim)
        
        if self.use_context:
            x = self._add_context(x)  # (B, T, frame_dim * (2k+1))
        
        # Frame-wise MLP
        h = self.mlp(x)
        output = self.output_layer(h)  # (B, T, 30)
        
        return output


class PelvisConditionedBackboneMDM(ConditionedMDM):
    """
    Backbone MDM conditioned on Hand + Pelvis.
    
    NEW PIPELINE ORDER: Hand → Lower → Backbone
    
    Analysis showed Pelvis-Backbone correlation (r=0.82) is stronger
    than Hand-Backbone correlation (r=0.62), so adding Pelvis as
    condition should improve Backbone generation.
    
    Input conditions:
        - Hand: (B, T, 12) wrist + finger positions
        - Pelvis: (B, T, 12) pelvis positions (RASI, LASI, RPSI, LPSI) - optional
        - ARIA: (B, 512) optional global music embedding
    
    Output: (B, T, 30) = 10 joints x 3 coords
        [Head(4), Neck(2), Chest(2), Shoulder(2)]
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_mask_prob: float = 0.1,
        use_aria: bool = True,
        use_pelvis: bool = True,
    ):
        # Spatial condition: Hand (12) + optionally Pelvis (12)
        spatial_dim = 12 + (12 if use_pelvis else 0)
        super().__init__(
            output_dim=30,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_mask_prob=cond_mask_prob,
            temporal_cond_dim=0,
            global_cond_dim=512 if use_aria else 0,
            spatial_cond_dim=spatial_dim,
        )
        self.use_aria = use_aria
        self.use_pelvis = use_pelvis
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        hand: torch.Tensor,  # (B, T, 12)
        pelvis: Optional[torch.Tensor] = None,  # (B, T, 12)
        aria: Optional[torch.Tensor] = None,
        force_mask: bool = False,
    ) -> torch.Tensor:
        # Build spatial condition
        if self.use_pelvis and pelvis is not None:
            spatial_cond = torch.cat([hand, pelvis], dim=-1)  # (B, T, 24)
        else:
            spatial_cond = hand  # (B, T, 12)
        
        return super().forward(
            x, timesteps,
            spatial_cond=spatial_cond,
            global_cond=aria if self.use_aria else None,
            force_mask=force_mask,
        )
