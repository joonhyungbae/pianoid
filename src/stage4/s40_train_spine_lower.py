#!/usr/bin/env python3
"""
s40_train_spine_lower.py: Train Spine-Conditioned Lower Body Generator (Stage 4)

Data justification (body_decomposition_analysis.json):
    spine → pelvis r=0.848  >  hands → pelvis r=0.647
    → Spine is a better predictor of pelvis than hands.

Pipeline:
    phase1_torso_predictions: torso_pred (T, 12)
        spine = concat(C7(0:3), T10(6:9)) -> (T, 6)
    -> HandConditionedLowerGenerator(hand_dim=6) -> lower_gt (T, 30)

Output: lower body [RASI, LASI, RPSI, LPSI, RKNE, LKNE, RANKLE, LANKLE, RTOE, LTOE]

Usage:
    # Prerequisite: Stage 2 trained + torso predictions precomputed
    python src/stage2/s21_precompute_torso.py

    # Train Stage 4
    python src/stage4/s40_train_spine_lower.py
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import (
    FPS,
    C7_IDX, T10_IDX,
    LOWER_BODY_INDICES,  # [16..25] = PELVIS(4) + R_LEG(3) + L_LEG(3) = 10 joints
)
from models.hierarchical_mdm import HandConditionedLowerGenerator


# Lower body joint indices — matches inference assembly order (joints 16–25):
#   RASI(16), LASI(17), RPSI(18), LPSI(19),
#   RKNE(20), RANKLE(21), RTOE(22),
#   LKNE(23), LANKLE(24), LTOE(25)
LOWER_INDICES = LOWER_BODY_INDICES  # 10 joints × 3 = 30-dim output

# Indices within the 10-joint lower array for knee→ankle bone loss:
#   idx 4=RKNE, 5=RANKLE   (right knee to right ankle)
#   idx 7=LKNE, 8=LANKLE   (left  knee to left  ankle)
_RKNE_LOCAL, _RANKLE_LOCAL = 4, 5
_LKNE_LOCAL, _LANKLE_LOCAL = 7, 8

# Spine conditioning: C7(0:3) + T10(6:9) from torso_pred -> 6-dim
SPINE_DIM = 6


class SpineLowerDataset(Dataset):
    """Dataset for Stage 4 spine-conditioned lower body training.

    Loads:
        - GT lower body joints from training_cache_B
        - ARIA embeddings from training_cache_B
        - Spine features from phase1_torso_predictions
            spine = concat(torso_pred[0:3], torso_pred[6:9]) = C7 + T10

    Falls back to GT spine if torso predictions unavailable.
    """

    def __init__(
        self,
        cache_dir: str,
        torso_pred_dir: str,
        split: str = 'train',
        window_size: int = 480,
        stride: int = 240,
        use_gt_spine: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.torso_pred_dir = Path(torso_pred_dir)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.use_gt_spine = use_gt_spine

        self.data = []
        self._load_all_data()

        self.windows = []
        self._create_windows()

        print(f"[{split}] {len(self.windows)} windows from {len(self.data)} samples, "
              f"use_gt_spine={use_gt_spine}")

    def _load_all_data(self):
        cache_split_dir = self.cache_dir / self.split
        torso_pred_split_dir = self.torso_pred_dir / self.split

        if not cache_split_dir.exists():
            raise FileNotFoundError(f"Cache split dir not found: {cache_split_dir}")

        cache_files = sorted(cache_split_dir.glob('*.npz'))

        for cache_path in tqdm(cache_files, desc=f'Loading {self.split}', leave=False):
            try:
                cache_data = np.load(cache_path)
                joints = cache_data['joints'].astype(np.float32)  # (T, 26, 3)
                T = joints.shape[0]

                aria_embed = cache_data['aria_embed'].astype(np.float32)

                # GT lower body (T, 30)
                lower_gt = joints[:, LOWER_INDICES, :].reshape(T, 30)

                # GT spine fallback: C7 + T10 -> (T, 6)
                gt_spine = np.concatenate([
                    joints[:, C7_IDX, :],   # (T, 3)
                    joints[:, T10_IDX, :],  # (T, 3)
                ], axis=-1).astype(np.float32)

                if self.use_gt_spine:
                    spine = gt_spine
                else:
                    torso_pred_path = torso_pred_split_dir / cache_path.name
                    spine = None
                    if torso_pred_path.exists():
                        torso_data = np.load(torso_pred_path)
                        if 'torso_pred' in torso_data:
                            torso_pred = torso_data['torso_pred'].astype(np.float32)
                            # torso layout: [C7(0:3), CLAV(3:6), T10(6:9), STRN(9:12)]
                            spine = np.concatenate([
                                torso_pred[:, 0:3],  # C7
                                torso_pred[:, 6:9],  # T10
                            ], axis=-1)  # (T, 6)
                    if spine is None:
                        spine = gt_spine

                self.data.append({
                    'lower_gt': lower_gt,
                    'spine': spine,
                    'aria_embed': aria_embed,
                    'aria_is_temporal': aria_embed.ndim == 2,
                    'num_frames': T,
                })

            except Exception as e:
                print(f"Warning: failed to load {cache_path.name}: {e}")
                continue

    def _create_windows(self):
        for sample_idx, sample in enumerate(self.data):
            num_frames = sample['num_frames']
            for start in range(0, num_frames - self.window_size + 1, self.stride):
                self.windows.append({
                    'sample_idx': sample_idx,
                    'start': start,
                    'end': start + self.window_size,
                    'aria_window_idx': start // self.stride,
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        sample = self.data[window['sample_idx']]
        start, end = window['start'], window['end']

        lower_gt = sample['lower_gt'][start:end]  # (T, 30)
        spine = sample['spine'][start:end]         # (T, 6)

        aria_data = sample['aria_embed']
        if sample['aria_is_temporal'] and aria_data.ndim == 2:
            aria_idx = min(window['aria_window_idx'], aria_data.shape[0] - 1)
            aria = aria_data[aria_idx]
        else:
            aria = aria_data

        return {
            'lower_gt': torch.from_numpy(lower_gt.copy()).float(),
            'spine': torch.from_numpy(spine.copy()).float(),
            'aria': torch.from_numpy(aria.copy()).float(),
        }


class SpineLowerLoss(nn.Module):
    """Position + velocity + bone length loss for lower body."""

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

    def forward(self, pred, target):
        pos_loss = F.mse_loss(pred, target)

        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(pred_vel, target_vel)

        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_acc = target_vel[:, 1:] - target_vel[:, :-1]
        smooth_loss = F.mse_loss(pred_acc, target_acc)

        # Knee–ankle bone length consistency
        pred_j = pred.view(*pred.shape[:-1], 10, 3)
        target_j = target.view(*target.shape[:-1], 10, 3)
        pred_bone_r = (pred_j[:, :, _RANKLE_LOCAL] - pred_j[:, :, _RKNE_LOCAL]).norm(dim=-1)
        pred_bone_l = (pred_j[:, :, _LANKLE_LOCAL] - pred_j[:, :, _LKNE_LOCAL]).norm(dim=-1)
        target_bone_r = (target_j[:, :, _RANKLE_LOCAL] - target_j[:, :, _RKNE_LOCAL]).norm(dim=-1)
        target_bone_l = (target_j[:, :, _LANKLE_LOCAL] - target_j[:, :, _LKNE_LOCAL]).norm(dim=-1)
        bone_loss = F.mse_loss(pred_bone_r, target_bone_r) + F.mse_loss(pred_bone_l, target_bone_l)

        total = (
            self.pos_weight * pos_loss
            + self.vel_weight * vel_loss
            + self.smooth_weight * smooth_loss
            + self.bone_weight * bone_loss
        )
        return total, {
            'pos': pos_loss.item(),
            'vel': vel_loss.item(),
            'smooth': smooth_loss.item(),
            'bone': bone_loss.item(),
            'total': total.item(),
        }


def compute_lower_stats(loader, device):
    """Compute mean lower pose and per-sample motion range for mean-pose initialization."""
    all_lower = []
    all_ranges = []
    for batch in tqdm(loader, desc='Computing lower body stats'):
        lower = batch['lower_gt']  # (B, T, 30)
        all_lower.append(lower)
        for i in range(lower.shape[0]):
            s = lower[i]
            deviation = s - s.mean(dim=0)
            all_ranges.append(deviation.max(dim=0)[0] - deviation.min(dim=0)[0])
    all_lower = torch.cat(all_lower, dim=0)
    mean = all_lower.mean(dim=(0, 1))
    motion_range = torch.stack(all_ranges).mean(dim=0)
    return mean.to(device), motion_range.to(device)


def train_epoch(model, loader, optimizer, device, criterion, scaler=None):
    model.train()
    total_losses = {}
    num_batches = 0

    for batch in tqdm(loader, desc='Training', leave=False):
        lower_gt = batch['lower_gt'].to(device)
        spine = batch['spine'].to(device)
        aria = batch['aria'].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(spine, aria)
            loss, loss_dict = criterion(pred, lower_gt)

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
        num_batches += 1

    return {k: v / num_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    total_losses = {}
    num_batches = 0

    for batch in tqdm(loader, desc='Validating', leave=False):
        lower_gt = batch['lower_gt'].to(device)
        spine = batch['spine'].to(device)
        aria = batch['aria'].to(device)

        with torch.amp.autocast('cuda'):
            pred = model(spine, aria)
            loss, loss_dict = criterion(pred, lower_gt)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        num_batches += 1

    return {k: v / num_batches for k, v in total_losses.items()}


def main():
    parser = argparse.ArgumentParser(description='Train Spine-Conditioned Lower Body Generator (Stage 4)')
    parser.add_argument('--cache_dir', type=str, default='MOSA_dataset/training_cache_B',
                        help='Training cache with GT joints + ARIA')
    parser.add_argument('--torso_pred_dir', type=str, default='MOSA_dataset/phase1_torso_predictions',
                        help='Phase1 torso predictions (for spine C7+T10 input)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage4')
    parser.add_argument('--use_gt_spine', action='store_true',
                        help='Use GT spine instead of torso predictions')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=480)
    parser.add_argument('--stride', type=int, default=240)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--no_wandb', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project='pianoid',
                name=f'spine_lower_{datetime.now():%Y%m%d_%H%M%S}',
                config=vars(args),
            )
        except ImportError:
            use_wandb = False

    # Datasets
    train_dataset = SpineLowerDataset(
        cache_dir=args.cache_dir, torso_pred_dir=args.torso_pred_dir,
        split='train', window_size=args.window_size, stride=args.stride,
        use_gt_spine=args.use_gt_spine,
    )
    val_dataset = SpineLowerDataset(
        cache_dir=args.cache_dir, torso_pred_dir=args.torso_pred_dir,
        split='test', window_size=args.window_size, stride=args.stride,
        use_gt_spine=args.use_gt_spine,
    )

    num_workers = min(8, max(1, len(train_dataset) // args.batch_size))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Compute mean lower pose from training data
    print("\nComputing lower body mean pose and motion range...")
    mean_pose, motion_range = compute_lower_stats(train_loader, device)
    print(f"  mean_pose[:6] = {mean_pose[:6].tolist()}")
    print(f"  motion_range[:6] = {motion_range[:6].tolist()}")

    # Model: HandConditionedLowerGenerator with hand_dim=6 (spine conditioning)
    model = HandConditionedLowerGenerator(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        hand_dim=SPINE_DIM,  # 6-dim spine (C7+T10) as conditioning
        use_aria=True,
        use_midi=False,
    ).to(device)
    model.set_mean_pose(mean_pose, motion_range)
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  SPINE_DIM={SPINE_DIM}, output=30 (10 lower joints × 3)")
    print(f"  Spine: C7(torso[0:3]) + T10(torso[6:9])")

    criterion = SpineLowerLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nTraining Stage 4 SpineLower (patience={args.patience})...")
    print(f"  Spine source: {'GT' if args.use_gt_spine else 'phase1_torso_predictions'}")

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, criterion, scaler)
        val_metrics = validate(model, val_loader, device, criterion)
        scheduler.step(val_metrics['total'])

        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train: {train_metrics['total']:.4f} | "
            f"Val: {val_metrics['total']:.4f} | "
            f"LR: {lr:.2e} | "
            f"Patience: {patience_counter}/{args.patience}"
        )

        if use_wandb:
            log_dict = {'epoch': epoch + 1, 'lr': lr}
            for k, v in train_metrics.items():
                log_dict[f'train/{k}'] = v
            for k, v in val_metrics.items():
                log_dict[f'val/{k}'] = v
            wandb.log(log_dict)

        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'mean_pose': mean_pose.cpu(),
                'motion_range': motion_range.cpu(),
                'epoch': epoch + 1,
                'val_loss': val_metrics['total'],
                'config': {
                    **vars(args),
                    'hand_dim': SPINE_DIM,
                    'use_aria': True,
                    'use_midi': False,
                    'output_dim': 30,
                    'lower_indices': LOWER_INDICES,
                },
            }, output_dir / 'best.pt')
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {output_dir / 'best.pt'}")
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
