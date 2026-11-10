"""
experiments/benchmark/common/base_model.py

Abstract base class for every baseline model.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn


class BaseMotionModel(ABC, nn.Module):
    """Abstract base class for every baseline model.

    All baselines inherit from this class so the comparison stays fair
    via a uniform interface.

    Input MIDI features format:
        piano_roll: [B, T, 88] - active keys (binary or continuous)
        velocity: [B, T, 88] - MIDI velocity (0-127 normalized to 0-1)
        onset: [B, T, 88] - onset events (binary)
    
    Output motion format:
        motion: [B, T, 26, 3] - 26 joints × 3 coordinates (meters)
    """
    
    def __init__(self):
        super().__init__()
        self.num_joints = 26
        self.motion_dim = 26 * 3  # 78
        self.midi_dim = 88
    
    @abstractmethod
    def forward(
        self, 
        midi_features: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Training forward pass
        
        Args:
            midi_features: {
                'piano_roll': [B, T, 88],
                'velocity': [B, T, 88],
                'onset': [B, T, 88]
            }
            **kwargs: model-specific extra args (e.g., gt_motion for training)

        Returns:
            output: model-specific output (used for loss computation)
                - Diffusion: predicted noise or x0
                - VAE: (recon, mu, logvar)
                - Regression: predicted motion
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        midi_features: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Inference-only method.

        Unlike training, this performs full generation
        (e.g., diffusion sampling, VAE decoding).

        Args:
            midi_features: {
                'piano_roll': [B, T, 88],
                'velocity': [B, T, 88],
                'onset': [B, T, 88]
            }
            **kwargs: model-specific extra args (e.g., num_steps for diffusion)

        Returns:
            motion: [B, T, 26, 3] - generated motion (meters)
        """
        pass
    
    def get_model_name(self) -> str:
        """Return the model name (used for logging/saving)."""
        return self.__class__.__name__
    
    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path: str, **extra_info):
        """Save a checkpoint.

        Args:
            path: file path to save to
            **extra_info: extra info (epoch, optimizer_state, etc.)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_name': self.get_model_name(),
            'num_params': self.get_num_params(),
            **extra_info
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cuda', **model_kwargs):
        """Load a checkpoint.

        Args:
            path: checkpoint path
            device: device
            **model_kwargs: model-init args

        Returns:
            model: the loaded model
            extra_info: extra info
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        extra_info = {k: v for k, v in checkpoint.items() 
                      if k not in ['model_state_dict']}
        
        return model, extra_info


class BaseDiffusionModel(BaseMotionModel):
    """Base class for diffusion-based models.

    Inherited by diffusion-based models such as MDM and EDGE.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = 'cosine',
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
    
    @abstractmethod
    def denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        midi_features: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Single denoising step
        
        Args:
            x_t: [B, T, 26, 3] - noisy motion
            t: [B] - timestep
            midi_features: conditioning features
        
        Returns:
            noise_pred or x0_pred: [B, T, 26, 3]
        """
        pass


class BaseVAEModel(BaseMotionModel):
    """Base class for VAE-based models."""
    
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
    
    @abstractmethod
    def encode(
        self,
        motion: torch.Tensor,
        midi_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> tuple:
        """Encode motion to latent space
        
        Returns:
            mu, logvar: [B, latent_dim]
        """
        pass
    
    @abstractmethod
    def decode(
        self,
        z: torch.Tensor,
        midi_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Decode latent to motion
        
        Returns:
            motion: [B, T, 26, 3]
        """
        pass
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class BaseRegressionModel(BaseMotionModel):
    """Base class for regression-based models.

    Inherited by direct-mapping (MIDI -> Motion) models.
    """
    
    def generate(
        self,
        midi_features: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """For regression models, forward and generate are identical."""
        return self.forward(midi_features, **kwargs)
