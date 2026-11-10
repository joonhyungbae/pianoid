"""
Diffusion Utilities for Hierarchical MDM Training

Wraps MDM's GaussianDiffusion for easy use with our models.
"""

import sys
from pathlib import Path

# Add MDM path BEFORE any other imports
MDM_PATH = str(Path(__file__).resolve().parent.parent.parent / 'vendor' / 'MDM')

# Insert at position 0 and also add utils path
if MDM_PATH not in sys.path:
    sys.path.insert(0, MDM_PATH)

# Now import other modules
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

# Import MDM diffusion
from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType,
)


def create_diffusion(
    num_timesteps: int = 1000,
    noise_schedule: str = "cosine",
) -> GaussianDiffusion:
    """Create GaussianDiffusion with standard settings."""
    betas = get_named_beta_schedule(noise_schedule, num_timesteps)
    
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )


def q_sample(
    diffusion: GaussianDiffusion,
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Add noise to x_0 to get x_t."""
    if noise is None:
        noise = torch.randn_like(x_start)
    return diffusion.q_sample(x_start, t, noise=noise)


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    shape: Tuple[int, ...],
    model_kwargs: Dict,
    ddim_steps: int = 50,
    eta: float = 0.0,
    device: str = 'cuda',
    progress: bool = True,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """DDIM sampling with optional Classifier-Free Guidance (CFG).

    Args:
        guidance_scale: CFG weight. 1.0 = no guidance (normal sampling).
            > 1.0 amplifies conditioning signal, reducing diversity.
            Requires model trained with cond_mask_prob > 0.
    """
    model.eval()
    use_cfg = guidance_scale != 1.0

    step_size = diffusion.num_timesteps // ddim_steps
    timesteps_list = list(range(0, diffusion.num_timesteps, step_size))[::-1]

    x = torch.randn(*shape, device=device)

    iterator = timesteps_list
    if progress:
        from tqdm import tqdm
        iterator = tqdm(timesteps_list, desc=f"DDIM ({ddim_steps} steps)")

    for i, t in enumerate(iterator):
        t_batch = torch.tensor([t] * shape[0], device=device)

        if use_cfg:
            # Conditional prediction
            pred_cond = model(x, t_batch, **model_kwargs)
            # Unconditional prediction (zero out all conditioning)
            pred_uncond = model(x, t_batch, **model_kwargs, force_mask=True)
            # CFG blend
            pred_x0 = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred_x0 = model(x, t_batch, **model_kwargs)

        if t == 0:
            x = pred_x0
        else:
            t_prev = timesteps_list[i + 1] if i + 1 < len(timesteps_list) else 0

            alpha_t = diffusion.alphas_cumprod[t]
            alpha_prev = diffusion.alphas_cumprod[t_prev] if t_prev > 0 else 1.0

            sigma = eta * np.sqrt((1 - alpha_prev) / (1 - alpha_t)) * np.sqrt(1 - alpha_t / alpha_prev)
            pred_noise = (x - np.sqrt(alpha_t) * pred_x0) / np.sqrt(1 - alpha_t)

            mean_pred = np.sqrt(alpha_prev) * pred_x0 + np.sqrt(1 - alpha_prev - sigma**2) * pred_noise

            if sigma > 0:
                x = mean_pred + sigma * torch.randn_like(x)
            else:
                x = mean_pred

    return x
