#!/usr/bin/env python3
"""
s20_train_torso_sweep.py: Stage 2 architecture + conditioning sweep.

Supports 4 architectures:
  - mdm: TorsoMDM (diffusion-based, current baseline)
  - transformer: TorsoTransformerRegression (deterministic)
  - lstm: TorsoLSTMRegression (bidirectional)
  - film: TorsoFiLMRegression (frame-wise with context window)

Supports conditioning options:
  - Hand predictions (always on, 12-dim spatial)
  - ARIA (--use_aria, 512-dim global FiLM)
  - pitch_range (--use_pitch_range, 1-dim temporal per frame)
  - onset_density (--use_onset_density, 1-dim temporal per frame)

Usage:
    python src/stage2/s20_train_torso_sweep.py --arch mdm
    python src/stage2/s20_train_torso_sweep.py --arch transformer --use_aria
    bash scripts/train/stage2.sh  # full MDM+zscore sweep (Table 3(A))
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'vendor' / 'MDM'))

from data.constants import (
    C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
    RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX,
    RSHO_IDX, LSHO_IDX,
    LWRIST_IDX, LHAND_IDX, RWRIST_IDX, RHAND_IDX,
)
from models.conditioned_mdm import ConditionedMDM
from models.hierarchical_mdm import MDMLoss
from models.torso_baselines import (
    TorsoTransformerRegression,
    TorsoMambaRegression,
    TorsoLSTMRegression,
    TorsoFiLMRegression,
    compute_torso_loss,
)
from diffusion_lib import create_diffusion, q_sample

TORSO_INDICES = [C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX]
HEAD_INDICES = [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX]
SHOULDER_INDICES = [RSHO_IDX, LSHO_IDX]
# Backbone now includes shoulders: 4 head + 4 torso + 2 shoulder = 10 joints, 30-dim
BACKBONE_INDICES = HEAD_INDICES + TORSO_INDICES + SHOULDER_INDICES


# ============================================================
# Dataset
# ============================================================

class TorsoSweepDataset(Dataset):
    """Extended dataset with optional temporal conditioning features."""

    def __init__(
        self,
        cache_dir: str,
        pred_dir: str,
        split: str = 'train',
        window_size: int = 480,
        stride: int = 240,
        use_gt_hand: bool = False,
        load_temporal: bool = False,
        load_pitch_range: bool = False,
        load_onset_density: bool = False,
        load_onset_per_key: bool = False,
        load_piano_roll: bool = False,
        repr_mode: str = 'absolute',
        mirror_aug: bool = False,
        hand_velocity: bool = False,
        norm_stats: dict = None,
        backbone: bool = False,
        require_aria: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.pred_dir = Path(pred_dir)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.use_gt_hand = use_gt_hand
        self.load_temporal = load_temporal
        self.load_pitch_range = load_pitch_range or load_temporal
        self.load_onset_density = load_onset_density or load_temporal
        self.load_onset_per_key = load_onset_per_key
        self.load_piano_roll = load_piano_roll
        self.repr_mode = repr_mode
        self.mirror_aug = mirror_aug and (split == 'train')
        self.hand_velocity = hand_velocity
        self.backbone = backbone
        self.target_indices = BACKBONE_INDICES if backbone else TORSO_INDICES
        # Only filter on ARIA availability when the caller truly needs it
        # (e.g. use_aria=True). Otherwise the dataset is ARIA-optional.
        self.require_aria = require_aria

        self.data = []
        self._load_all_data()

        if norm_stats is not None:
            self.norm_stats = norm_stats
        else:
            self.norm_stats = self._compute_norm_stats()

        self.windows = []
        self._create_windows()
        print(f"[{split}] {len(self.windows)} windows from {len(self.data)} samples "
              f"(repr={repr_mode}, mirror={self.mirror_aug}, hand_vel={hand_velocity})")

    def _load_all_data(self):
        cache_split_dir = self.cache_dir / self.split
        pred_split_dir = self.pred_dir / self.split

        for cache_path in tqdm(sorted(cache_split_dir.glob('*.npz')),
                               desc=f'Loading {self.split}', leave=False):
            try:
                cache_data = np.load(cache_path)
                has_aria = ('aria_embed' in cache_data
                            and not np.all(cache_data['aria_embed'] == 0))
                # ARIA is only a hard requirement when the model actually consumes it
                # (use_aria=True). Otherwise we keep the sample and hand a zero
                # placeholder that downstream code ignores.
                if getattr(self, 'require_aria', False) and not has_aria:
                    continue

                joints = cache_data['joints'].astype(np.float32)
                T = joints.shape[0]

                torso_gt = joints[:, self.target_indices, :].reshape(T, -1)
                gt_hand = np.concatenate([
                    joints[:, LWRIST_IDX],
                    joints[:, RWRIST_IDX],
                ], axis=-1).astype(np.float32)

                if has_aria:
                    aria_embed = cache_data['aria_embed'].astype(np.float32)
                    aria_is_temporal = aria_embed.ndim == 2
                else:
                    aria_embed = np.zeros((1, 512), dtype=np.float32)
                    aria_is_temporal = True

                # Load hand predictions
                hand_pred = None
                if not self.use_gt_hand:
                    pred_path = pred_split_dir / cache_path.name
                    if pred_path.exists():
                        pred_data = np.load(pred_path)
                        hand_pred = pred_data.get('hand_pred', pred_data.get('pred'))
                        if hand_pred is not None:
                            hand_pred = hand_pred.astype(np.float32)
                if hand_pred is None:
                    hand_pred = gt_hand

                sample = {
                    'torso_gt': torso_gt,
                    'hand_pred': hand_pred if not self.use_gt_hand else gt_hand,
                    'aria_embed': aria_embed,
                    'aria_is_temporal': aria_is_temporal,
                    'num_frames': T,
                }

                # Temporal features (load only what's needed)
                if self.load_pitch_range:
                    sample['pitch_range'] = cache_data.get(
                        'pitch_range', np.zeros(T)).astype(np.float32)
                if self.load_onset_density:
                    sample['onset_density'] = cache_data.get(
                        'onset_density', np.zeros(T)).astype(np.float32)
                if self.load_onset_per_key:
                    sample['onset_density_per_key'] = cache_data.get(
                        'onset_density_per_key', np.zeros((T, 88))).astype(np.float32)
                if self.load_piano_roll:
                    ak = cache_data['active_keys']
                    pr = np.zeros((T, 88), dtype=np.float32)
                    for t in range(T):
                        for k in ak[t]:
                            if 0 <= k < 88:
                                pr[t, k] = 1.0
                    sample['piano_roll'] = pr

                self.data.append(sample)
            except Exception as e:
                print(f"Warning: {cache_path.name}: {e}")

    def _compute_norm_stats(self):
        """Compute normalization statistics from loaded data."""
        stats = {'repr_mode': self.repr_mode}
        if self.repr_mode in ('zscore', 'residual', 'delta'):
            all_target = np.concatenate([s['torso_gt'] for s in self.data], axis=0)
            stats['torso_mean'] = all_target.mean(axis=0).astype(np.float32)
            stats['torso_std'] = (all_target.std(axis=0) + 1e-8).astype(np.float32)
        return stats

    def _create_windows(self):
        for si, s in enumerate(self.data):
            for start in range(0, s['num_frames'] - self.window_size + 1, self.stride):
                self.windows.append({
                    'sample_idx': si,
                    'start': start,
                    'end': start + self.window_size,
                    'aria_idx': start // self.stride,
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        s = self.data[w['sample_idx']]
        start, end = w['start'], w['end']

        torso_raw = s['torso_gt'][start:end].copy()  # (T, 12)
        hand_raw = s['hand_pred'][start:end].copy()   # (T, 6)

        # --- Mirror augmentation: flip X axis with 50% probability ---
        if self.mirror_aug and np.random.random() < 0.5:
            # X-axis indices: every 3rd starting from 0
            x_indices = list(range(0, torso_raw.shape[-1], 3))
            torso_raw[:, x_indices] *= -1
            hand_raw[:, [0, 3]] *= -1

        # --- Representation transform on torso target ---
        torso_first = torso_raw[0].copy()  # (12,) for delta reconstruction
        if self.repr_mode == 'zscore':
            mean = self.norm_stats['torso_mean']
            std = self.norm_stats['torso_std']
            torso_target = (torso_raw - mean) / std
        elif self.repr_mode == 'delta':
            delta = np.zeros_like(torso_raw)
            delta[1:] = torso_raw[1:] - torso_raw[:-1]
            torso_target = delta
        elif self.repr_mode == 'residual':
            mean = self.norm_stats['torso_mean']
            torso_target = torso_raw - mean
        else:
            torso_target = torso_raw

        result = {
            'torso_gt': torch.from_numpy(torso_target).float(),
            'torso_gt_abs': torch.from_numpy(torso_raw).float(),
            'torso_first': torch.from_numpy(torso_first).float(),
        }

        # --- Hand input (optionally append hand velocity) ---
        hand_t = torch.from_numpy(hand_raw).float()
        if self.hand_velocity:
            hand_vel = torch.zeros_like(hand_t)
            hand_vel[1:] = hand_t[1:] - hand_t[:-1]
            hand_t = torch.cat([hand_t, hand_vel], dim=-1)  # (T, 12)
        result['hand_pred'] = hand_t

        # ARIA
        aria = s['aria_embed']
        if s['aria_is_temporal'] and aria.ndim == 2:
            aria = aria[min(w['aria_idx'], aria.shape[0] - 1)]
        result['aria'] = torch.from_numpy(aria.copy()).float()

        # Temporal features
        if self.load_pitch_range and 'pitch_range' in s:
            result['pitch_range'] = torch.tensor(
                s['pitch_range'][start:end], dtype=torch.float32).unsqueeze(-1)
        if self.load_onset_density and 'onset_density' in s:
            result['onset_density'] = torch.tensor(
                s['onset_density'][start:end], dtype=torch.float32).unsqueeze(-1)
        if self.load_onset_per_key and 'onset_density_per_key' in s:
            result['onset_density_per_key'] = torch.tensor(
                s['onset_density_per_key'][start:end], dtype=torch.float32)
        if self.load_piano_roll and 'piano_roll' in s:
            result['piano_roll'] = torch.tensor(
                s['piano_roll'][start:end], dtype=torch.float32)

        return result


# ============================================================
# Model factory
# ============================================================

def create_model(args, device):
    """Create model based on architecture choice."""
    base_hand_dim = 6
    output_dim = 30 if getattr(args, 'backbone', False) else 12
    if getattr(args, 'hand_velocity', False):
        base_hand_dim = 12
    hand_dim = 0 if args.no_hand else base_hand_dim
    temporal_cond_dim = 0
    if args.use_pitch_range:
        temporal_cond_dim += 1
    if args.use_onset_density:
        temporal_cond_dim += 1
    if args.use_onset_per_key:
        temporal_cond_dim += 88
    if args.use_piano_roll:
        temporal_cond_dim += 88

    if args.arch == 'mdm':
        model = ConditionedMDM(
            output_dim=output_dim,
            latent_dim=args.latent_dim,
            num_layers=args.num_layers,
            num_heads=max(1, args.latent_dim // 64),
            dropout=0.1,
            cond_mask_prob=0.1,
            spatial_cond_dim=hand_dim,
            global_cond_dim=512 if args.use_aria else 0,
            temporal_cond_dim=temporal_cond_dim,
        )
        # Initialize output bias to default positions
        if output_dim >= 24:  # backbone (24 = head+torso, 30 = +shoulders)
            defaults = TorsoLSTMRegression.DEFAULT_BACKBONE
        else:
            defaults = TorsoLSTMRegression.DEFAULT_TORSO
        default_pos = torch.tensor(defaults[:output_dim])
        with torch.no_grad():
            model.output_proj.bias.copy_(default_pos)
        diffusion = create_diffusion(num_timesteps=args.diffusion_steps)
    elif args.arch == 'transformer':
        model = TorsoTransformerRegression(
            hand_dim=hand_dim,
            hidden_dim=args.latent_dim,
            num_layers=args.num_layers,
            num_heads=max(1, args.latent_dim // 64),
            use_aria=args.use_aria,
            temporal_cond_dim=temporal_cond_dim,
        )
        diffusion = None
    elif args.arch == 'mamba':
        model = TorsoMambaRegression(
            hand_dim=hand_dim,
            hidden_dim=args.latent_dim,
            num_layers=args.num_layers,
            d_state=16,
            use_aria=args.use_aria,
            temporal_cond_dim=temporal_cond_dim,
        )
        diffusion = None
    elif args.arch == 'lstm':
        model = TorsoLSTMRegression(
            hand_dim=hand_dim,
            torso_dim=output_dim,
            hidden_dim=args.latent_dim,
            num_layers=min(args.num_layers, 4),
            use_aria=args.use_aria,
            temporal_cond_dim=temporal_cond_dim,
        )
        diffusion = None
    elif args.arch == 'film':
        model = TorsoFiLMRegression(
            hand_dim=hand_dim,
            hidden_dim=args.latent_dim,
            num_layers=args.num_layers,
            context_frames=5,
            use_aria=args.use_aria,
            temporal_cond_dim=temporal_cond_dim,
        )
        diffusion = None
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    # For non-absolute representations, reset output bias to zero
    repr_mode = getattr(args, 'repr', 'absolute')
    if repr_mode != 'absolute':
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.out_features in (12, 24, 30):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {args.arch}: {n_params:,} params, latent={args.latent_dim}, "
          f"layers={args.num_layers}, aria={args.use_aria}, "
          f"temporal_cond_dim={temporal_cond_dim}, repr={repr_mode}")
    return model, diffusion


def _get_temporal_cond(batch, args, device):
    """Build temporal conditioning tensor from batch."""
    parts = []
    if args.use_pitch_range:
        parts.append(batch['pitch_range'].to(device))
    if args.use_onset_density:
        parts.append(batch['onset_density'].to(device))
    if args.use_onset_per_key:
        parts.append(batch['onset_density_per_key'].to(device))
    if args.use_piano_roll:
        parts.append(batch['piano_roll'].to(device))
    if parts:
        return torch.cat(parts, dim=-1)
    return None


# ============================================================
# Training loops
# ============================================================

def train_epoch_mdm(model, diffusion, loader, optimizer, criterion, device,
                    scaler, args, norm_stats=None):
    model.train()
    total_losses = {}
    mpjpe_sum, mpjpe_count = 0.0, 0
    n = 0
    lkw = _loss_kwargs(args)
    repr_mode = getattr(args, 'repr', 'absolute')

    for batch in tqdm(loader, desc='Train', leave=False):
        torso_gt = batch['torso_gt'].to(device)
        hand = None if args.no_hand else batch['hand_pred'].to(device)
        aria = batch['aria'].to(device) if args.use_aria else None
        temporal_cond = _get_temporal_cond(batch, args, device)

        B = torso_gt.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
        noise = torch.randn_like(torso_gt)
        x_t = q_sample(diffusion, torso_gt, t, noise=noise)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(x_t, t,
                         spatial_cond=hand,
                         global_cond=aria,
                         temporal_cond=temporal_cond)
            loss, loss_dict = compute_torso_loss(pred, torso_gt, **lkw)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v

        with torch.no_grad():
            pred_abs, gt_abs = _denorm_for_mpjpe(pred, batch, repr_mode, norm_stats, device)
            s, c = _compute_mpjpe(pred_abs, gt_abs)
            mpjpe_sum += s
            mpjpe_count += c

        n += 1

    avg = {k: v / n for k, v in total_losses.items()}
    avg['mpjpe_mm'] = (mpjpe_sum / mpjpe_count) * 1000
    return avg


def _compute_mpjpe(pred, gt, n_joints=None):
    """Compute MPJPE for N joints x 3 xyz."""
    B, T, D = pred.shape
    if n_joints is None:
        n_joints = D // 3
    pred_j = pred.float().reshape(B, T, n_joints, 3)
    gt_j = gt.reshape(B, T, n_joints, 3)
    ae = (pred_j - gt_j).norm(dim=-1)  # (B, T, n_joints)
    return ae.sum().item(), ae.numel()


def _loss_kwargs(args):
    """Extract loss weights from args for compute_torso_loss."""
    return dict(
        vel_w=args.vel_w,
        smooth_w=args.smooth_w,
        range_w=args.range_w,
        std_w=args.std_w,
        bone_w=getattr(args, 'bone_w', 0.0),
        spectral_w=getattr(args, 'spectral_w', 0.0),
        speed_w=getattr(args, 'speed_w', 0.0),
        under_w=getattr(args, 'under_w', 0.0),
        jerk_match_w=getattr(args, 'jerk_match_w', 0.0),
        range_cap_w=getattr(args, 'range_cap_w', 0.0),
        use_l1=args.use_l1,
    )


def _denorm_for_mpjpe(pred, batch, repr_mode, norm_stats, device):
    """Convert model output back to absolute positions for MPJPE."""
    if repr_mode == 'absolute':
        return pred, batch['torso_gt'].to(device)
    gt_abs = batch['torso_gt_abs'].to(device)
    if repr_mode == 'zscore':
        mean = torch.from_numpy(norm_stats['torso_mean']).float().to(device)
        std = torch.from_numpy(norm_stats['torso_std']).float().to(device)
        pred_abs = pred * std + mean
    elif repr_mode == 'delta':
        first = batch['torso_first'].to(device).unsqueeze(1)  # (B, 1, 12)
        pred_abs = first + torch.cumsum(pred, dim=1)
    elif repr_mode == 'residual':
        mean = torch.from_numpy(norm_stats['torso_mean']).float().to(device)
        pred_abs = pred + mean
    else:
        pred_abs = pred
    return pred_abs, gt_abs


def train_epoch_regression(model, loader, optimizer, criterion, device,
                           scaler, args, norm_stats=None):
    model.train()
    total_losses = {}
    mpjpe_sum, mpjpe_count = 0.0, 0
    n = 0
    lkw = _loss_kwargs(args)
    repr_mode = getattr(args, 'repr', 'absolute')

    for batch in tqdm(loader, desc='Train', leave=False):
        torso_gt = batch['torso_gt'].to(device)
        hand = None if args.no_hand else batch['hand_pred'].to(device)
        aria = batch['aria'].to(device) if args.use_aria else None
        temporal_cond = _get_temporal_cond(batch, args, device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(hand, aria=aria, temporal_cond=temporal_cond)
            loss, loss_dict = compute_torso_loss(pred, torso_gt, **lkw)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v

        with torch.no_grad():
            pred_abs, gt_abs = _denorm_for_mpjpe(pred, batch, repr_mode, norm_stats, device)
            s, c = _compute_mpjpe(pred_abs, gt_abs)
            mpjpe_sum += s
            mpjpe_count += c

        n += 1

    avg = {k: v / n for k, v in total_losses.items()}
    avg['mpjpe_mm'] = (mpjpe_sum / mpjpe_count) * 1000
    return avg


@torch.no_grad()
def validate_mdm(model, diffusion, loader, criterion, device, args, norm_stats=None):
    model.eval()
    total_losses = {}
    mpjpe_sum, mpjpe_count = 0.0, 0
    n = 0
    lkw = _loss_kwargs(args)
    repr_mode = getattr(args, 'repr', 'absolute')

    for batch in tqdm(loader, desc='Val', leave=False):
        torso_gt = batch['torso_gt'].to(device)
        hand = None if args.no_hand else batch['hand_pred'].to(device)
        aria = batch['aria'].to(device) if args.use_aria else None
        temporal_cond = _get_temporal_cond(batch, args, device)

        B = torso_gt.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
        noise = torch.randn_like(torso_gt)
        x_t = q_sample(diffusion, torso_gt, t, noise=noise)

        pred = model(x_t, t,
                     spatial_cond=hand,
                     global_cond=aria,
                     temporal_cond=temporal_cond)
        loss, loss_dict = compute_torso_loss(pred, torso_gt, **lkw)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v

        pred_abs, gt_abs = _denorm_for_mpjpe(pred, batch, repr_mode, norm_stats, device)
        s, c = _compute_mpjpe(pred_abs, gt_abs)
        mpjpe_sum += s
        mpjpe_count += c

        n += 1

    avg = {k: v / n for k, v in total_losses.items()}
    avg['mpjpe_mm'] = (mpjpe_sum / mpjpe_count) * 1000
    return avg


@torch.no_grad()
def validate_regression(model, loader, criterion, device, args, norm_stats=None):
    model.eval()
    total_losses = {}
    mpjpe_sum, mpjpe_count = 0.0, 0
    n = 0
    lkw = _loss_kwargs(args)
    repr_mode = getattr(args, 'repr', 'absolute')

    for batch in tqdm(loader, desc='Val', leave=False):
        torso_gt = batch['torso_gt'].to(device)
        hand = None if args.no_hand else batch['hand_pred'].to(device)
        aria = batch['aria'].to(device) if args.use_aria else None
        temporal_cond = _get_temporal_cond(batch, args, device)

        pred = model(hand, aria=aria, temporal_cond=temporal_cond)
        loss, loss_dict = compute_torso_loss(pred, torso_gt, **lkw)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v

        pred_abs, gt_abs = _denorm_for_mpjpe(pred, batch, repr_mode, norm_stats, device)
        s, c = _compute_mpjpe(pred_abs, gt_abs)
        mpjpe_sum += s
        mpjpe_count += c

        n += 1

    avg = {k: v / n for k, v in total_losses.items()}
    avg['mpjpe_mm'] = (mpjpe_sum / mpjpe_count) * 1000
    return avg


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 2 Torso Sweep')
    # Architecture
    parser.add_argument('--arch', type=str, default='mdm',
                        choices=['mdm', 'transformer', 'mamba', 'lstm', 'film'])
    # Conditioning
    parser.add_argument('--use_aria', action='store_true')
    parser.add_argument('--use_pitch_range', action='store_true')
    parser.add_argument('--use_onset_density', action='store_true')
    parser.add_argument('--use_onset_per_key', action='store_true',
                        help='onset_density_per_key (88-dim, MIDI B dense feature)')
    parser.add_argument('--no_hand', action='store_true',
                        help='Exclude wrist input (test MIDI-only torso prediction)')
    parser.add_argument('--use_piano_roll', action='store_true',
                        help='Add raw piano roll (88-dim binary) as temporal conditioning')
    # Data
    parser.add_argument('--cache_dir', default='MOSA_dataset/training_cache_W05')
    parser.add_argument('--pred_dir', default='MOSA_dataset/phase0_hand_predictions')
    parser.add_argument('--output_dir', default='checkpoints/stage2_sweep')
    parser.add_argument('--use_gt_hand', action='store_true')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default 42)')
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--window_size', type=int, default=480)
    parser.add_argument('--stride', type=int, default=240)
    parser.add_argument('--patience', type=int, default=40)
    # Loss weights
    parser.add_argument('--vel_w', type=float, default=0.5,
                        help='Velocity matching weight')
    parser.add_argument('--smooth_w', type=float, default=0.0,
                        help='Smoothness penalty (pred acc > gt acc)')
    parser.add_argument('--range_w', type=float, default=0.0,
                        help='Per-axis range matching weight')
    parser.add_argument('--jerk_match_w', type=float, default=0.0,
                        help='Jerk-matching loss weight (log-ratio of 3rd-deriv magnitude)')
    parser.add_argument('--range_cap_w', type=float, default=0.0,
                        help='Range cap loss weight (asymmetric: penalize over-expression only)')
    parser.add_argument('--std_w', type=float, default=0.0,
                        help='Per-axis std matching weight')
    parser.add_argument('--speed_w', type=float, default=0.0,
                        help='Speed magnitude matching weight')
    parser.add_argument('--under_w', type=float, default=0.0,
                        help='Asymmetric under-movement penalty weight')
    parser.add_argument('--use_l1', action='store_true',
                        help='Use L1 instead of MSE for position loss')
    parser.add_argument('--backbone', action='store_true',
                        help='Predict backbone (head+torso, 8 joints 24-dim) instead of torso only (4 joints 12-dim)')
    parser.add_argument('--bone_w', type=float, default=0.0,
                        help='Bone length consistency loss weight')
    parser.add_argument('--spectral_w', type=float, default=0.0,
                        help='Spectral (FFT) loss weight for low-frequency matching')
    # Representation & augmentation
    parser.add_argument('--repr', type=str, default='absolute',
                        choices=['absolute', 'zscore', 'delta', 'residual'],
                        help='Target representation for torso output')
    parser.add_argument('--mirror_aug', action='store_true',
                        help='X-axis mirror augmentation (50%% prob)')
    parser.add_argument('--hand_velocity', action='store_true',
                        help='Append hand velocity (6d) to hand input -> 12d')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--wandb_project', default='pianoid-stage2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from project_utils.seeding import fix_seed, seed_worker, make_generator
    fix_seed(args.seed)
    print(f"Seed: {args.seed} (cudnn deterministic on)")
    is_diffusion = (args.arch == 'mdm')
    load_temporal = args.use_pitch_range or args.use_onset_density or args.use_onset_per_key or args.use_piano_roll

    # Build condition tag for output dir
    cond_tag = args.arch
    if args.no_hand:
        cond_tag += '_nohand'
    if args.use_aria:
        cond_tag += '_aria'
    if args.use_pitch_range:
        cond_tag += '_pr'
    if args.use_onset_density:
        cond_tag += '_od'
    if args.use_onset_per_key:
        cond_tag += '_opk'
    if args.use_piano_roll:
        cond_tag += '_pr88'

    # Add hyperparams + loss tag for grid search runs
    hp_tag = f"h{args.latent_dim}_l{args.num_layers}_lr{args.lr}"
    loss_parts = []
    if args.use_l1:
        loss_parts.append('l1')
    if args.vel_w != 0.5:
        loss_parts.append(f'v{args.vel_w}')
    if args.smooth_w > 0:
        loss_parts.append(f'sm{args.smooth_w}')
    if args.range_w > 0:
        loss_parts.append(f'rng{args.range_w}')
    if args.std_w > 0:
        loss_parts.append(f'std{args.std_w}')
    if args.bone_w > 0:
        loss_parts.append(f'bone{args.bone_w}')
    if args.spectral_w > 0:
        loss_parts.append(f'fft{args.spectral_w}')
    if args.jerk_match_w > 0:
        loss_parts.append(f'jm{args.jerk_match_w}')
    if args.range_cap_w > 0:
        loss_parts.append(f'rc{args.range_cap_w}')
    loss_tag = '_'.join(loss_parts) if loss_parts else 'mse'

    # Representation / augmentation tags
    repr_tag = args.repr if args.repr != 'absolute' else ''
    aug_parts = []
    if args.mirror_aug:
        aug_parts.append('mirr')
    if args.hand_velocity:
        aug_parts.append('hvel')
    aug_tag = '_'.join(aug_parts)

    tag_parts = [cond_tag, hp_tag, loss_tag]
    if repr_tag:
        tag_parts.append(repr_tag)
    if aug_tag:
        tag_parts.append(aug_tag)
    full_tag = '_'.join(tag_parts)
    output_dir = Path(args.output_dir) / full_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"Stage 2 Sweep: {full_tag}")
    print(f"  device={device}  epochs={args.epochs}  patience={args.patience}")
    print(f"{'=' * 60}")

    # Dataset. ARIA is only a hard requirement when the model actually consumes it.
    require_aria = bool(getattr(args, 'use_aria', False))
    print("\nLoading datasets...")
    train_ds = TorsoSweepDataset(
        args.cache_dir, args.pred_dir, 'train',
        args.window_size, args.stride, args.use_gt_hand, load_temporal,
        load_pitch_range=args.use_pitch_range,
        load_onset_density=args.use_onset_density,
        load_onset_per_key=args.use_onset_per_key,
        load_piano_roll=args.use_piano_roll,
        repr_mode=args.repr, mirror_aug=args.mirror_aug,
        hand_velocity=args.hand_velocity, backbone=args.backbone,
        require_aria=require_aria)
    norm_stats = train_ds.norm_stats
    val_ds = TorsoSweepDataset(
        args.cache_dir, args.pred_dir, 'val',
        args.window_size, args.stride, args.use_gt_hand, load_temporal,
        load_pitch_range=args.use_pitch_range,
        load_onset_density=args.use_onset_density,
        load_onset_per_key=args.use_onset_per_key,
        load_piano_roll=args.use_piano_roll,
        repr_mode=args.repr, mirror_aug=False,
        hand_velocity=args.hand_velocity, norm_stats=norm_stats,
        backbone=args.backbone, require_aria=require_aria)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                              worker_init_fn=seed_worker, generator=make_generator(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True,
                            worker_init_fn=seed_worker, generator=make_generator(args.seed))
    print(f"  Train: {len(train_ds)} windows ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_ds)} windows ({len(val_loader)} batches)")

    # Model
    print("\nCreating model...")
    model, diffusion = create_model(args, device)
    criterion = MDMLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=20, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Wandb
    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=full_tag,
            config={
                'arch': args.arch,
                'latent_dim': args.latent_dim,
                'num_layers': args.num_layers,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'use_aria': args.use_aria,
                'use_pitch_range': args.use_pitch_range,
                'use_onset_density': args.use_onset_density,
                'use_onset_per_key': args.use_onset_per_key,
                'use_piano_roll': args.use_piano_roll,
                'no_hand': args.no_hand,
                'diffusion_steps': args.diffusion_steps,
                'window_size': args.window_size,
                'epochs': args.epochs,
                'patience': args.patience,
                'repr': args.repr,
                'mirror_aug': args.mirror_aug,
                'hand_velocity': args.hand_velocity,
                'bone_w': args.bone_w,
                'spectral_w': args.spectral_w,
            },
            tags=['stage2', args.arch, cond_tag],
        )

    # Training — best model selected by val MPJPE
    print(f"\nTraining ({args.epochs} epochs, patience={args.patience})...")
    print(f"  Model selection: val MPJPE (not composite loss)")
    best_val_mpjpe = float('inf')
    patience_counter = 0

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc=full_tag, unit='ep')
    for epoch in epoch_pbar:
        if is_diffusion:
            train_losses = train_epoch_mdm(
                model, diffusion, train_loader, optimizer, criterion,
                device, scaler, args, norm_stats=norm_stats)
            val_losses = validate_mdm(
                model, diffusion, val_loader, criterion, device, args,
                norm_stats=norm_stats)
        else:
            train_losses = train_epoch_regression(
                model, train_loader, optimizer, criterion, device, scaler, args,
                norm_stats=norm_stats)
            val_losses = validate_regression(
                model, val_loader, criterion, device, args,
                norm_stats=norm_stats)

        scheduler.step(val_losses['total'])
        lr = optimizer.param_groups[0]['lr']

        val_mpjpe = val_losses['mpjpe_mm']
        is_best = val_mpjpe < best_val_mpjpe

        # Progress bar update
        epoch_pbar.set_postfix({
            'loss': f"{val_losses['total']:.4f}",
            'mpjpe': f"{val_mpjpe:.1f}",
            'best': f"{min(best_val_mpjpe, val_mpjpe):.1f}",
            'lr': f"{lr:.0e}",
            'pat': f"{patience_counter}/{args.patience}",
        })

        # Wandb logging
        if not args.no_wandb:
            log_dict = {'epoch': epoch, 'lr': lr}
            for prefix, losses in [('train', train_losses), ('val', val_losses)]:
                for k, v in losses.items():
                    log_dict[f'{prefix}/{k}'] = v
            log_dict['best_val_mpjpe'] = min(best_val_mpjpe, val_mpjpe)
            wandb.log(log_dict)

        # Detailed print every 20 epochs or on improvement
        if epoch % 20 == 0 or is_best:
            marker = " *BEST*" if is_best else ""
            tqdm.write(
                f"  Epoch {epoch:>4}/{args.epochs} | "
                f"train: pos={train_losses['pos']:.4f} vel={train_losses.get('vel',0):.4f} MPJPE={train_losses['mpjpe_mm']:.1f}mm | "
                f"val: loss={val_losses['total']:.5f} MPJPE={val_mpjpe:.1f}mm | "
                f"lr={lr:.1e} | pat={patience_counter}/{args.patience}{marker}"
            )

        if is_best:
            best_val_mpjpe = val_mpjpe
            patience_counter = 0
            save_config = {
                'arch': args.arch,
                'latent_dim': args.latent_dim,
                'num_layers': args.num_layers,
                'use_aria': args.use_aria,
                'use_pitch_range': args.use_pitch_range,
                'use_onset_density': args.use_onset_density,
                'use_onset_per_key': args.use_onset_per_key,
                'use_piano_roll': args.use_piano_roll,
                'no_hand': args.no_hand,
                'lr': args.lr,
                'diffusion_steps': args.diffusion_steps,
                'cond_tag': cond_tag,
                'backbone': args.backbone,
                'vel_w': args.vel_w,
                'smooth_w': args.smooth_w,
                'range_w': args.range_w,
                'std_w': args.std_w,
                'jerk_match_w': getattr(args, 'jerk_match_w', 0.0),
                'range_cap_w': getattr(args, 'range_cap_w', 0.0),
                'bone_w': args.bone_w,
                'spectral_w': args.spectral_w,
                'use_l1': args.use_l1,
                'speed_w': getattr(args, 'speed_w', 0.0),
                'under_w': getattr(args, 'under_w', 0.0),
                'repr': args.repr,
                'mirror_aug': args.mirror_aug,
                'hand_velocity': args.hand_velocity,
            }
            # Serialize norm_stats (numpy arrays -> lists for JSON compat)
            save_ns = {}
            for k, v in norm_stats.items():
                save_ns[k] = v.tolist() if hasattr(v, 'tolist') else v
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_losses['total'],
                'val_mpjpe_mm': val_mpjpe,
                'config': save_config,
                'norm_stats': save_ns,
            }, output_dir / 'best.pt')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            tqdm.write(f"  Early stopping at epoch {epoch}")
            break

    epoch_pbar.close()

    if not args.no_wandb:
        wandb.log({'final/best_val_mpjpe': best_val_mpjpe})
        wandb.finish()

    print(f"\n{'=' * 60}")
    print(f"  {full_tag}")
    print(f"  Best val MPJPE: {best_val_mpjpe:.2f}mm")
    print(f"  Checkpoint: {output_dir / 'best.pt'}")
    print(f"{'=' * 60}")
    return best_val_mpjpe


if __name__ == '__main__':
    main()
