"""
Diffusion utilities for Piano Motion Generation.
"""

import sys
from pathlib import Path

# Add MDM path FIRST to avoid conflicts with src/utils
MDM_PATH = str(Path(__file__).resolve().parent.parent.parent / 'vendor' / 'MDM')
if MDM_PATH not in sys.path:
    sys.path.insert(0, MDM_PATH)

from .diffusion_utils import (
    create_diffusion,
    q_sample,
    ddim_sample,
)

# Import schedule sampler from MDM
try:
    from diffusion.resample import create_named_schedule_sampler, UniformSampler
except ImportError:
    # Fallback: define simple UniformSampler
    import numpy as np
    import torch as th
    
    class UniformSampler:
        def __init__(self, diffusion):
            self.diffusion = diffusion
            self._weights = np.ones([diffusion.num_timesteps])
        
        def weights(self):
            return self._weights
        
        def sample(self, batch_size, device):
            w = self.weights()
            p = w / np.sum(w)
            indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
            indices = th.from_numpy(indices_np).long().to(device)
            weights_np = 1 / (len(p) * p[indices_np])
            weights = th.from_numpy(weights_np).float().to(device)
            return indices, weights
    
    def create_named_schedule_sampler(name, diffusion):
        if name == "uniform":
            return UniformSampler(diffusion)
        else:
            raise NotImplementedError(f"unknown schedule sampler: {name}")

__all__ = [
    'create_diffusion',
    'q_sample',
    'ddim_sample',
    'create_named_schedule_sampler',
    'UniformSampler',
]
