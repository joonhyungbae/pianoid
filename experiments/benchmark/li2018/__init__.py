"""
Li et al. ISMIR 2018 "Skeleton Plays Piano" baseline.

Adapted from 2D upper body (8 joints) to 3D full body (26 joints).
"""

from .model import Li2018Model, compute_li2018_loss

__all__ = ['Li2018Model', 'compute_li2018_loss']
