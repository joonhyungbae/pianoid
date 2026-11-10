#!/usr/bin/env python3
"""
s30_train_arm.py: Train Stage 3 arm generation models.

Supports both MDM (diffusion) and regression (Transformer/BiLSTM/FiLM).
Supports absolute and relative (shoulder-relative) coordinate systems.

Usage:
    python src/stage3/s30_train_arm.py --arch bilstm --use_onset_density
    python src/stage3/s30_train_arm.py --arch mdm --use_aria
    python src/stage3/s30_train_arm.py --arch bilstm --relative_coords
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'vendor' / 'MDM'))

from data.constants import (
    RSHO_IDX, RELB_IDX, LSHO_IDX, LELB_IDX,
    RWRIST_IDX, LWRIST_IDX, T10_IDX, STRN_IDX,
)
from diffusion_lib import create_diffusion, q_sample
from models.hierarchical_mdm import MDMLoss
from models.torso_baselines import compute_torso_loss  # reusable for any dim

# Stage 3 predicts elbows (shoulders moved to Stage 2 backbone).
# `--functional_grouping` restores the pre-regroup 4-joint setting (Table 5-B Row 1).
ARM_INDICES_ANATOMICAL = [RELB_IDX, LELB_IDX]
ARM_NAMES_ANATOMICAL = ['RELB', 'LELB']
ARM_INDICES_FUNCTIONAL = [RSHO_IDX, RELB_IDX, LSHO_IDX, LELB_IDX]
ARM_NAMES_FUNCTIONAL = ['RSHO', 'RELB', 'LSHO', 'LELB']

ARM_INDICES = ARM_INDICES_ANATOMICAL
ARM_NAMES = ARM_NAMES_ANATOMICAL


class ArmDataset(Dataset):
    """Dataset for arm generation (MDM and regression)."""

    def __init__(self, cache_dir, hand_pred_dir, torso_pred_dir,
                 split='train', window_size=480, stride=240,
                 use_gt_inputs=False, relative_coords=False,
                 load_onset_density=False, load_pitch_range=False,
                 load_piano_roll=False, no_spatial=False,
                 repr_mode='absolute', norm_stats=None,
                 functional_grouping=False,
                 require_aria=False):
        self.cache_dir = Path(cache_dir)
        self.hand_pred_dir = Path(hand_pred_dir)
        self.torso_pred_dir = Path(torso_pred_dir)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.use_gt_inputs = use_gt_inputs
        self.functional_grouping = functional_grouping
        # Shoulders are the target in functional-grouping mode, so shoulder-relative
        # elbow targets (and shoulder inputs) would cause trivial target leakage.
        if functional_grouping and relative_coords:
            raise ValueError("relative_coords is incompatible with functional_grouping")
        self.relative_coords = relative_coords
        self.arm_indices = ARM_INDICES_FUNCTIONAL if functional_grouping else ARM_INDICES_ANATOMICAL
        self.arm_dim = len(self.arm_indices) * 3
        self.load_onset_density = load_onset_density
        self.load_pitch_range = load_pitch_range
        self.load_piano_roll = load_piano_roll
        self.repr_mode = repr_mode
        self.no_spatial = no_spatial
        # ARIA is consumed only when use_aria=True in the training script.
        self.require_aria = require_aria

        self.data = []
        self._load_all_data()

        # Normalization stats for zscore
        if norm_stats is not None:
            self.norm_stats = norm_stats
        elif self.repr_mode == 'zscore':
            all_arm = np.concatenate([s['arm_gt'] for s in self.data], axis=0)
            self.norm_stats = {
                'arm_mean': all_arm.mean(0).astype(np.float32),
                'arm_std': (all_arm.std(0) + 1e-8).astype(np.float32),
            }
        else:
            self.norm_stats = {}

        self.windows = []
        self._create_windows()
        print(f"[{split}] {len(self.windows)} windows from {len(self.data)} samples (repr={repr_mode})")

    def _load_all_data(self):
        cache_split = self.cache_dir / self.split
        hand_split = self.hand_pred_dir / self.split
        torso_split = self.torso_pred_dir / self.split

        for f in tqdm(sorted(cache_split.glob('*.npz')), desc=f'Loading {self.split}', leave=False):
            try:
                data = np.load(f)
                has_aria = ('aria_embed' in data
                            and not np.all(data['aria_embed'] == 0))
                # ARIA is only a hard requirement when the model consumes it.
                if getattr(self, 'require_aria', False) and not has_aria:
                    continue

                joints = data['joints'].astype(np.float32)
                T = joints.shape[0]

                # GT targets: elbows only (6-dim) by default, or shoulders+elbows (12-dim)
                # when functional_grouping is active.
                arm_gt = joints[:, self.arm_indices, :].reshape(T, self.arm_dim)
                # GT shoulders for relative coords
                gt_rsho = joints[:, RSHO_IDX].astype(np.float32)
                gt_lsho = joints[:, LSHO_IDX].astype(np.float32)

                # ARIA (optional — zero placeholder when absent)
                if has_aria:
                    aria = data['aria_embed'].astype(np.float32)
                    aria_is_temporal = aria.ndim == 2
                else:
                    aria = np.zeros((1, 512), dtype=np.float32)
                    aria_is_temporal = True

                # Wrists
                gt_wrists = np.concatenate([
                    joints[:, RWRIST_IDX], joints[:, LWRIST_IDX]
                ], axis=-1).astype(np.float32)
                # Chest
                gt_chest = joints[:, [T10_IDX, STRN_IDX], :].reshape(T, 6).astype(np.float32)

                if self.use_gt_inputs:
                    wrists, chest = gt_wrists, gt_chest
                    rsho_in, lsho_in = gt_rsho, gt_lsho
                else:
                    # Wrists from Stage 1
                    hp = hand_split / f.name
                    if hp.exists():
                        hd = np.load(hp)
                        hp_data = hd.get('hand_pred', hd.get('pred'))
                        if hp_data is not None:
                            hp_data = hp_data.astype(np.float32)
                            wrists = np.concatenate([hp_data[:, 3:6], hp_data[:, 0:3]], axis=-1)
                        else:
                            wrists = gt_wrists
                    else:
                        wrists = gt_wrists

                    # Chest + shoulders from Stage 2 backbone (10 joints, 30-dim)
                    # Backbone layout: [RFHD,LFHD,RBHD,LBHD,C7,CLAV,T10,STRN,RSHO,LSHO]
                    tp = torso_split / f.name
                    if tp.exists():
                        td = np.load(tp)
                        tp_data = td.get('torso_pred')
                        if tp_data is not None:
                            tp_data = tp_data.astype(np.float32)
                            if tp_data.shape[-1] >= 30:
                                # New backbone with shoulders
                                chest = tp_data[:, 18:24]   # T10+STRN
                                rsho_in = tp_data[:, 24:27]  # RSHO
                                lsho_in = tp_data[:, 27:30]  # LSHO
                            elif tp_data.shape[-1] == 24:
                                # Old backbone (no shoulders)
                                chest = tp_data[:, 18:24]
                                rsho_in, lsho_in = gt_rsho, gt_lsho
                            else:
                                # 12-dim torso only
                                chest = tp_data[:, 6:12]
                                rsho_in, lsho_in = gt_rsho, gt_lsho
                        else:
                            chest = gt_chest
                            rsho_in, lsho_in = gt_rsho, gt_lsho
                    else:
                        chest = gt_chest
                        rsho_in, lsho_in = gt_rsho, gt_lsho

                # Relative coordinates: elbow relative to PREDICTED shoulder
                # arm_gt is GT elbow (absolute); convert to relative to GT shoulder for training
                if self.relative_coords:
                    # Use PREDICTED shoulder (rsho_in/lsho_in) to match inference-time distribution.
                    # Previously used gt_rsho/gt_lsho which caused train/test distribution mismatch
                    # and Y-axis over-expression.
                    relb_rel = arm_gt[:, 0:3] - rsho_in
                    lelb_rel = arm_gt[:, 3:6] - lsho_in
                    arm_gt = np.concatenate([relb_rel, lelb_rel], axis=-1)

                sample = {
                    'arm_gt': arm_gt,
                    'wrists': wrists,
                    'chest': chest,
                    'rsho': rsho_in,
                    'lsho': lsho_in,
                    'aria_embed': aria,
                    'aria_is_temporal': aria_is_temporal,
                    'num_frames': T,
                }

                if self.load_onset_density:
                    sample['onset_density'] = data.get(
                        'onset_density', np.zeros(T)).astype(np.float32)
                if self.load_pitch_range:
                    sample['pitch_range'] = data.get(
                        'pitch_range', np.zeros(T)).astype(np.float32)
                if self.load_piano_roll:
                    ak = data['active_keys']
                    pr = np.zeros((T, 88), dtype=np.float32)
                    for t in range(T):
                        for k in ak[t]:
                            if 0 <= k < 88:
                                pr[t, k] = 1.0
                    sample['piano_roll'] = pr

                self.data.append(sample)
            except Exception as e:
                print(f"Warning: {f.name}: {e}")

    def _create_windows(self):
        for si, s in enumerate(self.data):
            for start in range(0, s['num_frames'] - self.window_size + 1, self.stride):
                self.windows.append({
                    'sample_idx': si, 'start': start,
                    'end': start + self.window_size,
                    'aria_idx': start // self.stride,
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        s = self.data[w['sample_idx']]
        start, end = w['start'], w['end']

        if self.no_spatial:
            spatial_cond = np.zeros((end - start, 0), dtype=np.float32)
        elif self.functional_grouping:
            # Shoulders are targets here, so exclude them from the input.
            # spatial: wrists(6) + chest(6) = 12-dim
            spatial_cond = np.concatenate([
                s['wrists'][start:end], s['chest'][start:end],
            ], axis=-1)
        else:
            # spatial: wrists(6) + chest(6) + shoulders(6) = 18-dim
            spatial_cond = np.concatenate([
                s['wrists'][start:end], s['chest'][start:end],
                s['rsho'][start:end], s['lsho'][start:end],
            ], axis=-1)

        aria = s['aria_embed']
        if s['aria_is_temporal'] and aria.ndim == 2:
            aria = aria[min(w['aria_idx'], aria.shape[0] - 1)]

        arm_raw = s['arm_gt'][start:end].copy()
        if self.repr_mode == 'zscore':
            arm_target = (arm_raw - self.norm_stats['arm_mean']) / self.norm_stats['arm_std']
        else:
            arm_target = arm_raw

        result = {
            'arm_gt': torch.from_numpy(arm_target).float(),
            'arm_gt_abs': torch.from_numpy(arm_raw).float(),
            'spatial_cond': torch.from_numpy(spatial_cond.copy()).float(),
            'aria': torch.from_numpy(aria.copy()).float(),
        }

        if self.load_onset_density and 'onset_density' in s:
            result['onset_density'] = torch.tensor(
                s['onset_density'][start:end], dtype=torch.float32).unsqueeze(-1)
        if self.load_pitch_range and 'pitch_range' in s:
            result['pitch_range'] = torch.tensor(
                s['pitch_range'][start:end], dtype=torch.float32).unsqueeze(-1)
        if self.load_piano_roll and 'piano_roll' in s:
            result['piano_roll'] = torch.tensor(
                s['piano_roll'][start:end], dtype=torch.float32)

        return result


def create_model(args, device):
    from models.model_factory import build_stage3_arm_model
    functional = getattr(args, 'functional_grouping', False)
    # Functional grouping: input = wrists(6)+chest(6); target = 4 joints (12-dim).
    # Anatomical (default): input = wrists(6)+chest(6)+shoulders(6); target = elbows (6-dim).
    if args.no_spatial:
        spatial_dim = 0
    else:
        spatial_dim = 12 if functional else 18
    arm_dim = 12 if functional else 6
    config = {
        'arch': args.arch,
        'latent_dim': args.latent_dim,
        'num_layers': args.num_layers,
        'num_heads': max(1, args.latent_dim // 64),
        'use_aria': args.use_aria,
        'use_onset_density': args.use_onset_density,
        'use_pitch_range': args.use_pitch_range,
        'use_piano_roll': args.use_piano_roll,
        'no_spatial': args.no_spatial,
        'spatial_dim': spatial_dim,
        'arm_dim': arm_dim,
    }
    model = build_stage3_arm_model(config, device)
    n = sum(p.numel() for p in model.parameters())
    print(f"  {args.arch}: {n:,} params")
    return model, config


def _get_temporal_cond(batch, args, device):
    parts = []
    if args.use_onset_density and 'onset_density' in batch:
        parts.append(batch['onset_density'].to(device))
    if args.use_pitch_range and 'pitch_range' in batch:
        parts.append(batch['pitch_range'].to(device))
    if args.use_piano_roll and 'piano_roll' in batch:
        parts.append(batch['piano_roll'].to(device))
    return torch.cat(parts, dim=-1) if parts else None


def _loss_kwargs(args):
    return dict(
        vel_w=args.vel_w, range_w=args.range_w, std_w=args.std_w,
        speed_w=args.speed_w, under_w=args.under_w, use_l1=args.use_l1,
    )


def train_regression(model, loader, optimizer, device, criterion, args, scaler=None):
    model.train()
    total_loss = 0.0
    n = 0
    lkw = _loss_kwargs(args)
    for batch in tqdm(loader, desc='Train', leave=False):
        arm_gt = batch['arm_gt'].to(device)
        spatial_cond = batch['spatial_cond'].to(device)
        aria = batch['aria'].to(device) if args.use_aria else None
        temporal_cond = _get_temporal_cond(batch, args, device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(spatial_cond, aria=aria, temporal_cond=temporal_cond)
            loss, _ = compute_torso_loss(pred, arm_gt, **lkw)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        n += 1
    return {'total': total_loss / n}


@torch.no_grad()
def validate_regression(model, loader, device, criterion, args):
    model.eval()
    total_loss = 0.0
    n = 0
    lkw = _loss_kwargs(args)
    for batch in tqdm(loader, desc='Val', leave=False):
        arm_gt = batch['arm_gt'].to(device)
        spatial_cond = batch['spatial_cond'].to(device)
        aria = batch['aria'].to(device) if args.use_aria else None
        temporal_cond = _get_temporal_cond(batch, args, device)
        pred = model(spatial_cond, aria=aria, temporal_cond=temporal_cond)
        loss, _ = compute_torso_loss(pred, arm_gt, **lkw)
        total_loss += loss.item()
        n += 1
    return {'total': total_loss / n}


def train_mdm(model, diffusion, loader, optimizer, device, criterion, args, scaler=None):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc='Train', leave=False):
        arm_gt = batch['arm_gt'].to(device)
        spatial_cond = batch['spatial_cond'].to(device)
        aria = batch['aria'].to(device) if args.use_aria else None

        B = arm_gt.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
        noise = torch.randn_like(arm_gt)
        x_t = q_sample(diffusion, arm_gt, t, noise=noise)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(x_t, t, spatial_cond=spatial_cond, global_cond=aria)
            loss, _ = criterion(pred, arm_gt)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        n += 1
    return {'total': total_loss / n}


@torch.no_grad()
def validate_mdm(model, diffusion, loader, device, criterion, args):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc='Val', leave=False):
        arm_gt = batch['arm_gt'].to(device)
        spatial_cond = batch['spatial_cond'].to(device)
        aria = batch['aria'].to(device) if args.use_aria else None
        B = arm_gt.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
        noise = torch.randn_like(arm_gt)
        x_t = q_sample(diffusion, arm_gt, t, noise=noise)
        pred = model(x_t, t, spatial_cond=spatial_cond, global_cond=aria)
        loss, _ = criterion(pred, arm_gt)
        total_loss += loss.item()
        n += 1
    return {'total': total_loss / n}


def main():
    parser = argparse.ArgumentParser(description='Train Stage 3 Arm Model')
    parser.add_argument('--arch', type=str, default='bilstm',
                        choices=['mdm', 'transformer', 'bilstm', 'mamba', 'film'])
    parser.add_argument('--cache_dir', default='MOSA_dataset/training_cache_W10')
    parser.add_argument('--hand_pred_dir', default='MOSA_dataset/phase0_hand_predictions')
    parser.add_argument('--torso_pred_dir', default='MOSA_dataset/phase1_torso_predictions')
    parser.add_argument('--output_dir', default='checkpoints/stage3_experiments')
    parser.add_argument('--use_aria', action='store_true')
    parser.add_argument('--use_onset_density', action='store_true')
    parser.add_argument('--use_pitch_range', action='store_true')
    parser.add_argument('--use_piano_roll', action='store_true',
                        help='Add raw piano roll (88-dim binary)')
    parser.add_argument('--no_spatial', action='store_true',
                        help='Exclude spatial conditioning (MIDI-only arm prediction)')
    parser.add_argument('--use_gt_inputs', action='store_true')
    parser.add_argument('--relative_coords', action='store_true',
                        help='Predict elbows relative to shoulders')
    parser.add_argument('--functional_grouping', action='store_true',
                        help='Predict 4 arm joints (RSHO+RELB+LSHO+LELB) jointly, '
                             'reproducing the pre-regroup Stage 3 used for Table 5-B Row 1.')
    parser.add_argument('--repr', type=str, default='absolute', choices=['absolute', 'zscore'],
                        help='Target representation')
    parser.add_argument('--vel_w', type=float, default=0.5, help='Velocity loss weight')
    parser.add_argument('--range_w', type=float, default=0.0, help='Log-ratio range matching')
    parser.add_argument('--std_w', type=float, default=0.0, help='Log-ratio std matching')
    parser.add_argument('--speed_w', type=float, default=0.0, help='Per-dim speed matching')
    parser.add_argument('--under_w', type=float, default=0.0, help='Under-movement penalty')
    parser.add_argument('--use_l1', action='store_true', help='L1 position loss')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default 42)')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--window_size', type=int, default=480)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', default='pianoid-stage3')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from project_utils.seeding import fix_seed, seed_worker, make_generator
    fix_seed(args.seed)
    print(f"Seed: {args.seed} (cudnn deterministic on)")

    is_diffusion = (args.arch == 'mdm')

    # Build tag
    tag = args.arch
    if args.no_spatial:
        tag += '_nospatial'
    if args.functional_grouping:
        tag += '_func'
    if args.relative_coords:
        tag += '_rel'
    if args.use_aria:
        tag += '_aria'
    if args.use_onset_density:
        tag += '_od'
    if args.use_pitch_range:
        tag += '_pr'
    if args.use_piano_roll:
        tag += '_pr88'
    if args.use_gt_inputs:
        tag += '_gt'
    hp_tag = f"h{args.latent_dim}_l{args.num_layers}_lr{args.lr}"
    # Loss tag
    loss_parts = []
    if args.use_l1: loss_parts.append('l1')
    if args.vel_w != 0.5: loss_parts.append(f'v{args.vel_w}')
    if args.range_w > 0: loss_parts.append(f'rng{args.range_w}')
    if args.std_w > 0: loss_parts.append(f'std{args.std_w}')
    if args.speed_w > 0: loss_parts.append(f'spd{args.speed_w}')
    if args.under_w > 0: loss_parts.append(f'und{args.under_w}')
    if args.repr != 'absolute': loss_parts.append(args.repr)
    loss_tag = '_'.join(loss_parts) if loss_parts else ''
    full_tag = f"{tag}_{hp_tag}" + (f"_{loss_tag}" if loss_tag else "")

    output_dir = Path(args.output_dir) / full_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Stage 3 Arm: {full_tag}")
    print(f"  device={device}  epochs={args.epochs}  patience={args.patience}")
    print(f"  relative_coords={args.relative_coords}")
    print(f"{'='*60}")

    # Dataset
    ds_kwargs = dict(
        use_gt_inputs=args.use_gt_inputs, relative_coords=args.relative_coords,
        load_onset_density=args.use_onset_density, load_pitch_range=args.use_pitch_range,
        load_piano_roll=args.use_piano_roll, no_spatial=args.no_spatial,
        repr_mode=args.repr,
        functional_grouping=args.functional_grouping,
    )
    train_ds = ArmDataset(
        args.cache_dir, args.hand_pred_dir, args.torso_pred_dir,
        'train', args.window_size, args.window_size // 2, **ds_kwargs)
    norm_stats = train_ds.norm_stats
    val_ds = ArmDataset(
        args.cache_dir, args.hand_pred_dir, args.torso_pred_dir,
        'val', args.window_size, args.window_size // 2, **ds_kwargs,
        norm_stats=norm_stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              worker_init_fn=seed_worker, generator=make_generator(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                            worker_init_fn=seed_worker, generator=make_generator(args.seed))

    # Model
    model, model_config = create_model(args, device)
    diffusion = create_diffusion(num_timesteps=1000) if is_diffusion else None

    criterion = MDMLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Wandb
    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, name=full_tag,
            config={**model_config, 'relative_coords': args.relative_coords,
                    'use_gt_inputs': args.use_gt_inputs, 'lr': args.lr,
                    'batch_size': args.batch_size, 'epochs': args.epochs},
        )

    # Training
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.epochs), desc=f'{full_tag}', unit='ep'):
        if is_diffusion:
            train_losses = train_mdm(model, diffusion, train_loader, optimizer, device, criterion, args, scaler)
            val_losses = validate_mdm(model, diffusion, val_loader, device, criterion, args)
        else:
            train_losses = train_regression(model, train_loader, optimizer, device, criterion, args, scaler)
            val_losses = validate_regression(model, val_loader, device, criterion, args)

        scheduler.step(val_losses['total'])

        if not args.no_wandb:
            import wandb
            wandb.log({'train/total': train_losses['total'], 'val/total': val_losses['total'],
                       'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch + 1})

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            save_config = {**model_config,
                           'relative_coords': args.relative_coords,
                           'use_gt_inputs': args.use_gt_inputs,
                           'no_spatial': args.no_spatial,
                           'repr': args.repr,
                           'vel_w': args.vel_w, 'range_w': args.range_w,
                           'std_w': args.std_w, 'speed_w': args.speed_w,
                           'under_w': args.under_w, 'use_l1': args.use_l1,
                           'abs_elbow_w': getattr(args, 'abs_elbow_w', 0.0)}
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss, 'config': save_config,
                'norm_stats': norm_stats,
            }, output_dir / 'best.pt')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    if not args.no_wandb:
        import wandb
        wandb.finish()

    print(f"\n{'='*60}")
    print(f"  {full_tag}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoint: {output_dir / 'best.pt'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
