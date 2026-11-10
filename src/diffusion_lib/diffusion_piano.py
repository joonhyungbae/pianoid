"""
Simplified Gaussian Diffusion for Piano Motion Generation.

Based on MDM's gaussian_diffusion.py, simplified for our use case.
Key changes:
- Removed text/action conditioning specific code
- Added piano-specific interface
- Simplified loss computation

Reference:
- MDM: https://github.com/GuyTevet/motion-diffusion-model
- DDPM: https://arxiv.org/abs/2006.11239
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable
from enum import Enum


class ModelMeanType(Enum):
    """What the model predicts."""
    START_X = "start_x"   # Model predicts x_0
    EPSILON = "epsilon"   # Model predicts noise


class BetaSchedule:
    """Beta schedule for diffusion."""
    
    @staticmethod
    def linear(num_timesteps: int, scale: float = 1.0) -> np.ndarray:
        """Linear beta schedule from Ho et al."""
        scale = scale * 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    
    @staticmethod
    def cosine(num_timesteps: int, s: float = 0.008) -> np.ndarray:
        """Cosine beta schedule from Nichol & Dhariwal."""
        steps = num_timesteps + 1
        t = np.linspace(0, num_timesteps, steps) / num_timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)


class GaussianDiffusion:
    """
    Gaussian Diffusion process for training and sampling.
    
    Supports:
    - DDPM training
    - DDPM/DDIM sampling
    - Predicting x_0 or noise
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        model_mean_type: ModelMeanType = ModelMeanType.START_X,
        lambda_vel: float = 0.5,
        lambda_acc: float = 0.1,
    ):
        self.num_timesteps = num_timesteps
        self.model_mean_type = model_mean_type
        self.lambda_vel = lambda_vel
        self.lambda_acc = lambda_acc
        
        # Get beta schedule
        if beta_schedule == "linear":
            betas = BetaSchedule.linear(num_timesteps)
        elif beta_schedule == "cosine":
            betas = BetaSchedule.cosine(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.betas = betas
        
        # Compute diffusion parameters
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _extract(self, arr: np.ndarray, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from array for given timesteps."""
        batch_size = t.shape[0]
        out = torch.from_numpy(arr).to(device=t.device, dtype=torch.float32)
        out = out.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to x_0 to get x_t.
        
        Args:
            x_start: (B, J, 3, T) clean motion
            t: (B,) timesteps
            noise: (B, J, 3, T) optional noise
        
        Returns:
            x_t: (B, J, 3, T) noisy motion
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Dict = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            model: PianoMDM model
            x_start: (B, J, 3, T) clean motion
            t: (B,) timesteps
            model_kwargs: Dict with midi_cond, aria_cond, musical_context
        
        Returns:
            Dict with 'loss', 'mse', 'vel', 'acc'
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        
        # Model forward
        model_output = model(x_t, t, **model_kwargs)
        
        # Determine target
        if self.model_mean_type == ModelMeanType.START_X:
            target = x_start
        else:
            target = noise
        
        # MSE loss
        mse_loss = F.mse_loss(model_output, target, reduction='mean')
        
        # Geometric losses (only for x_0 prediction)
        if self.model_mean_type == ModelMeanType.START_X:
            # Velocity
            pred_vel = model_output[:, :, :, 1:] - model_output[:, :, :, :-1]
            target_vel = target[:, :, :, 1:] - target[:, :, :, :-1]
            vel_loss = F.mse_loss(pred_vel, target_vel)
            
            # Acceleration
            pred_acc = pred_vel[:, :, :, 1:] - pred_vel[:, :, :, :-1]
            target_acc = target_vel[:, :, :, 1:] - target_vel[:, :, :, :-1]
            acc_loss = F.mse_loss(pred_acc, target_acc)
        else:
            vel_loss = torch.tensor(0.0, device=x_start.device)
            acc_loss = torch.tensor(0.0, device=x_start.device)
        
        total_loss = mse_loss + self.lambda_vel * vel_loss + self.lambda_acc * acc_loss
        
        return {
            'loss': total_loss,
            'mse': mse_loss,
            'vel': vel_loss,
            'acc': acc_loss,
        }
    
    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Dict = None,
        clip_denoised: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t).
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        model_output = model(x_t, t, **model_kwargs)
        
        if self.model_mean_type == ModelMeanType.START_X:
            x_recon = model_output
        else:
            # Convert noise prediction to x_0
            x_recon = (
                self._extract(1.0 / self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
                - self._extract(self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod, t, x_t.shape) * model_output
            )
        
        if clip_denoised:
            x_recon = x_recon.clamp(-1, 1)
        
        # Compute posterior
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_recon
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_xstart": x_recon,
        }
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Dict = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Sample x_{t-1} from p(x_{t-1} | x_t)."""
        out = self.p_mean_variance(model, x_t, t, model_kwargs, clip_denoised)
        noise = torch.randn_like(x_t)
        
        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return sample
    
    @torch.no_grad()
    def ddpm_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        model_kwargs: Dict = None,
        clip_denoised: bool = True,
        progress: bool = True,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        DDPM sampling loop.
        
        Args:
            model: PianoMDM model
            shape: (B, J, 3, T) output shape
            model_kwargs: Conditioning
            clip_denoised: Clip predictions
            progress: Show progress
            device: Device
        
        Returns:
            samples: (B, J, 3, T)
        """
        model.eval()
        
        x = torch.randn(*shape, device=device)
        
        timesteps = list(range(self.num_timesteps - 1, -1, -1))
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for i in timesteps:
            t = torch.tensor([i] * shape[0], device=device)
            x = self.p_sample(model, x, t, model_kwargs, clip_denoised)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        model_kwargs: Dict = None,
        clip_denoised: bool = True,
        ddim_steps: int = 50,
        eta: float = 0.0,
        progress: bool = True,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        DDIM sampling for faster generation.
        
        Args:
            model: PianoMDM model
            shape: (B, J, 3, T) output shape
            ddim_steps: Number of DDIM steps (e.g., 50)
            eta: DDIM stochasticity (0 = deterministic)
        
        Returns:
            samples: (B, J, 3, T)
        """
        model.eval()
        
        # Select timestep subset
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]
        
        x = torch.randn(*shape, device=device)
        
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc=f"DDIM Sampling ({ddim_steps} steps)")
        
        for i, t in enumerate(timesteps):
            t_batch = torch.tensor([t] * shape[0], device=device)
            
            # Get prediction
            out = self.p_mean_variance(model, x, t_batch, model_kwargs, clip_denoised)
            pred_x0 = out["pred_xstart"]
            
            if t == 0:
                x = pred_x0
            else:
                # DDIM update
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                
                alpha_t = self.alphas_cumprod[t]
                alpha_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else 1.0
                
                sigma = eta * np.sqrt((1 - alpha_prev) / (1 - alpha_t)) * np.sqrt(1 - alpha_t / alpha_prev)
                
                pred_noise = (x - np.sqrt(alpha_t) * pred_x0) / np.sqrt(1 - alpha_t)
                
                mean_pred = (
                    np.sqrt(alpha_prev) * pred_x0 +
                    np.sqrt(1 - alpha_prev - sigma**2) * pred_noise
                )
                
                if sigma > 0:
                    noise = torch.randn_like(x)
                    x = mean_pred + sigma * noise
                else:
                    x = mean_pred
        
        return x


class DiffusionTrainer:
    """Convenience wrapper for training diffusion models."""
    
    def __init__(
        self,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        device: str = 'cuda',
    ):
        self.model = model
        self.diffusion = diffusion
        self.device = device
    
    def train_step(
        self,
        x_start: torch.Tensor,
        model_kwargs: Dict = None,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            x_start: (B, J, 3, T) clean motion
            model_kwargs: Conditioning
        
        Returns:
            loss_dict: Dict of losses
        """
        self.model.train()
        
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
        
        losses = self.diffusion.training_losses(self.model, x_start, t, model_kwargs)
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        njoints: int,
        nframes: int,
        model_kwargs: Dict = None,
        ddim_steps: int = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples.
        
        Args:
            batch_size: Number of samples
            njoints: Number of joints
            nframes: Sequence length
            model_kwargs: Conditioning
            ddim_steps: If provided, use DDIM; else DDPM
            progress: Show progress bar
        
        Returns:
            samples: (B, J, 3, T)
        """
        shape = (batch_size, njoints, 3, nframes)
        
        if ddim_steps is not None:
            return self.diffusion.ddim_sample(
                self.model, shape, model_kwargs, 
                ddim_steps=ddim_steps, progress=progress, device=self.device
            )
        else:
            return self.diffusion.ddpm_sample(
                self.model, shape, model_kwargs,
                progress=progress, device=self.device
            )


if __name__ == '__main__':
    from piano_mdm import PianoMDM
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing GaussianDiffusion...")
    
    # Create model
    model = PianoMDM(
        njoints=22,
        nfeats=3,
        latent_dim=256,
        num_layers=4,  # Smaller for test
    ).to(device)
    
    # Create diffusion
    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        beta_schedule="cosine",
        model_mean_type=ModelMeanType.START_X,
    )
    
    # Test training loss
    B, J, T = 2, 22, 60
    x_start = torch.randn(B, J, 3, T).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    
    losses = diffusion.training_losses(model, x_start, t)
    print(f"Training loss: {losses['loss'].item():.4f}")
    
    # Test q_sample
    x_t = diffusion.q_sample(x_start, t)
    print(f"Noisy sample shape: {x_t.shape}")
    
    # Test sampling (just 2 steps for speed)
    print("Testing quick sample...")
    quick_diffusion = GaussianDiffusion(num_timesteps=10)  # Very few steps for test
    trainer = DiffusionTrainer(model, quick_diffusion, device)
    
    samples = trainer.sample(1, 22, 60, ddim_steps=5, progress=False)
    print(f"Sample shape: {samples.shape}")
    
    print("✓ All tests passed!")
