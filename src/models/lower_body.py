"""
Lower Body Generator (Lightweight)

Generates lower body joint positions for seated piano playing.
Since the pianist is seated, lower body motion is relatively static.

Uses a simple MLP-based approach conditioned on torso position.

Input: Torso position (T10) + Template
Output: Lower body joints (T, 10, 3)
        [RASI, LASI, RPSI, LPSI, RKNE, RANK, RTOE, LKNE, LANK, LTOE]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

SiLU = nn.SiLU


class LowerBodyGenerator(nn.Module):
    """
    Lightweight generator for lower body (seated position).
    
    Since lower body is mostly static during piano playing,
    we use a simple approach:
    1. Learn a template pose relative to T10
    2. Add small adjustments based on torso motion
    """
    
    # Joint ordering (10 joints)
    JOINT_NAMES = [
        'RASI', 'LASI', 'RPSI', 'LPSI',  # Pelvis
        'RKNE', 'RANK', 'RTOE',           # Right leg
        'LKNE', 'LANK', 'LTOE',           # Left leg
    ]
    
    # Default template offsets from T10 (seated position, in meters)
    DEFAULT_TEMPLATE = {
        'RASI':  [0.15, 0.05, -0.10],
        'LASI':  [-0.15, 0.05, -0.10],
        'RPSI':  [0.10, -0.10, -0.10],
        'LPSI':  [-0.10, -0.10, -0.10],
        'RKNE':  [0.18, 0.25, -0.45],
        'RANK':  [0.18, 0.45, -0.85],
        'RTOE':  [0.18, 0.55, -0.85],
        'LKNE':  [-0.18, 0.25, -0.45],
        'LANK':  [-0.18, 0.45, -0.85],
        'LTOE':  [-0.18, 0.55, -0.85],
    }
    
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_joints = 10
        
        # Learnable template offsets (initialized from defaults)
        template = torch.zeros(self.num_joints, 3)
        for i, name in enumerate(self.JOINT_NAMES):
            template[i] = torch.tensor(self.DEFAULT_TEMPLATE[name])
        self.template = nn.Parameter(template)
        
        # Small adjustment network based on torso motion
        self.adjustment_net = nn.Sequential(
            nn.Linear(12, hidden_dim),  # T10 (3) + CLAV (3) + motion delta (6)
            SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, self.num_joints * 3),
        )
        
        # Clamp adjustment magnitude (in meters)
        self.max_adjustment = 0.05  # 50mm = 0.05m max adjustment
    
    def forward(
        self,
        t10: torch.Tensor,
        clav: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate lower body positions.
        
        Args:
            t10: (B, T, 3) - T10 position from torso generator
            clav: (B, T, 3) - CLAV position from torso generator
        
        Returns:
            lower_body: (B, T, 10, 3) - lower body positions
        """
        B, T, _ = t10.shape
        device = t10.device
        
        # Compute motion delta (change from previous frame)
        t10_delta = torch.zeros_like(t10)
        t10_delta[:, 1:] = t10[:, 1:] - t10[:, :-1]
        
        clav_delta = torch.zeros_like(clav)
        clav_delta[:, 1:] = clav[:, 1:] - clav[:, :-1]
        
        # Combine features
        features = torch.cat([t10, clav, t10_delta, clav_delta], dim=-1)  # (B, T, 12)
        
        # Compute adjustments
        adjustments = self.adjustment_net(features)  # (B, T, 30)
        adjustments = adjustments.view(B, T, self.num_joints, 3)
        
        # Clamp adjustments
        adjustments = torch.clamp(adjustments, -self.max_adjustment, self.max_adjustment)
        
        # Apply template + adjustments relative to T10
        # Template is (10, 3), broadcast to (B, T, 10, 3)
        template_expanded = self.template.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        
        # Output = T10 + template + adjustment
        lower_body = t10.unsqueeze(2) + template_expanded + adjustments
        
        return lower_body


class LowerBodyLoss(nn.Module):
    """
    Loss function for lower body generation.
    
    Emphasizes stability (low velocity) since lower body should be relatively static.
    """
    
    def __init__(
        self,
        w_pos: float = 1.0,
        w_vel: float = 1.0,  # Higher weight for stability
        w_static: float = 0.5,  # Encourage small adjustments
    ):
        super().__init__()
        self.w_pos = w_pos
        self.w_vel = w_vel
        self.w_static = w_static
    
    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        adjustments: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: (B, T, 10, 3) predicted lower body
            gt: (B, T, 10, 3) ground truth
            adjustments: (B, T, 10, 3) optional adjustment values
            mask: (B, T) optional valid frame mask
        """
        if mask is None:
            mask = torch.ones(pred.shape[:2], device=pred.device)
        
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        
        # Position loss
        loss_pos = (F.l1_loss(pred, gt, reduction='none') * mask).sum() / mask.sum()
        
        # Velocity loss (encourage smoothness)
        pred_vel = pred[:, 1:] - pred[:, :-1]
        gt_vel = gt[:, 1:] - gt[:, :-1]
        vel_mask = mask[:, 1:] * mask[:, :-1]
        loss_vel = (F.l1_loss(pred_vel, gt_vel, reduction='none') * vel_mask).sum() / vel_mask.sum()
        
        # Static loss (encourage small adjustments from template)
        loss_static = torch.tensor(0.0, device=pred.device)
        if adjustments is not None:
            loss_static = adjustments.abs().mean()
        
        # Total
        loss = self.w_pos * loss_pos + self.w_vel * loss_vel + self.w_static * loss_static
        
        # MPJPE (data is in meters, convert to mm for display)
        mpjpe_m = ((pred - gt).pow(2).sum(dim=-1).sqrt() * mask.squeeze(-1)).sum() / mask.sum()
        mpjpe = mpjpe_m * 1000  # m -> mm
        
        metrics = {
            'loss_pos': loss_pos.item(),
            'loss_vel': loss_vel.item(),
            'loss_static': loss_static.item() if isinstance(loss_static, torch.Tensor) else 0.0,
            'mpjpe_mm': mpjpe.item(),
        }
        
        return loss, metrics


class StaticLowerBody(nn.Module):
    """
    Even simpler: completely static lower body based on mean pose.
    
    Use this if you want zero lower body motion (pure template).
    """
    
    JOINT_NAMES = LowerBodyGenerator.JOINT_NAMES
    DEFAULT_TEMPLATE = LowerBodyGenerator.DEFAULT_TEMPLATE
    
    def __init__(self):
        super().__init__()
        
        self.num_joints = 10
        
        # Fixed template
        template = torch.zeros(self.num_joints, 3)
        for i, name in enumerate(self.JOINT_NAMES):
            template[i] = torch.tensor(self.DEFAULT_TEMPLATE[name])
        self.register_buffer('template', template)
    
    def forward(
        self,
        t10: torch.Tensor,
        clav: torch.Tensor = None,  # unused
    ) -> torch.Tensor:
        """
        Generate static lower body positions.
        
        Args:
            t10: (B, T, 3) - T10 position
        
        Returns:
            lower_body: (B, T, 10, 3)
        """
        B, T, _ = t10.shape
        
        template_expanded = self.template.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        lower_body = t10.unsqueeze(2) + template_expanded
        
        return lower_body
