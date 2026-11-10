#!/usr/bin/env python3
"""
s31_train_head_regressor.py: Train HeadRegressor (non-diffusion)

HeadRegressor: Neck (C7 + CLAV) -> Head (RFHD, LFHD, RBHD, LBHD)

This is a simple deterministic regression model (no diffusion) since the head
position is nearly fully determined by the neck (C7+CLAV) positions.

Pipeline:
    phase1_torso_predictions: torso_pred[:, 0:6] = [C7(0:3), CLAV(3:6)] -> neck_input (T, 6)
    -> HeadRegressor -> head_gt (T, 12) [RFHD, LFHD, RBHD, LBHD]

Loss:
    MSE position loss + velocity loss

Usage:
    # Prerequisite: trained TorsoMDM + precomputed torso predictions
    python src/stage2/s21_precompute_torso.py

    # Train HeadRegressor
    python src/stage3/s31_train_head_regressor.py
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
from typing import Optional
from tqdm import tqdm

# Add paths
MDM_PATH = str(Path(__file__).parent.parent.parent / 'vendor' / 'MDM')
sys.path.insert(0, MDM_PATH)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import (
    FPS,
    RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX,
    C7_IDX, CLAV_IDX,
)
from models.hierarchical_mdm import HeadRegressor, HeadResidualRegressor


# Head output indices (global joint indices)
HEAD_INDICES = [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX]  # [0, 1, 2, 3]


class HeadDataset(Dataset):
    """Dataset for HeadRegressor training.

    Loads:
        - Ground truth head joints from training cache (joints[:, HEAD_INDICES, :])
        - Torso predictions from phase1_torso_predictions/ (for C7+CLAV neck input)
        - Falls back to GT neck if torso predictions unavailable

    Neck input: torso_pred[:, 0:6] = [C7(0:3), CLAV(3:6)]
    Head GT:    joints[:, HEAD_INDICES, :].reshape(T, 12)
    """

    def __init__(
        self,
        cache_dir: str,
        torso_pred_dir: str,
        split: str = 'train',
        window_size: int = 480,
        stride: int = 240,
        use_gt_neck: bool = False,
        use_full_torso: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.torso_pred_dir = Path(torso_pred_dir)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.use_gt_neck = use_gt_neck
        self.use_full_torso = use_full_torso

        self.data = []
        self._load_all_data()

        self.windows = []
        self._create_windows()

        print(f"[{split}] {len(self.windows)} windows from {len(self.data)} samples, "
              f"use_gt_neck={use_gt_neck}")

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

                # GT Head: [RFHD, LFHD, RBHD, LBHD] -> (T, 12)
                head_gt = joints[:, HEAD_INDICES, :].reshape(T, 12)

                # GT neck (fallback): C7 + CLAV -> (T, 6)
                gt_neck = joints[:, [C7_IDX, CLAV_IDX], :].reshape(T, 6)

                # GT torso (full 12-dim) for residual model
                gt_torso = joints[:, [C7_IDX, CLAV_IDX] + list(range(6, 8)), :].reshape(T, -1)
                # Actually use TORSO_INDICES properly
                from data.constants import TORSO_INDICES
                gt_torso = joints[:, TORSO_INDICES, :].reshape(T, 12)

                if self.use_gt_neck:
                    neck = gt_neck
                    torso_input = gt_torso
                else:
                    # Load torso predictions
                    torso_pred_path = torso_pred_split_dir / cache_path.name
                    if torso_pred_path.exists():
                        torso_pred_data = np.load(torso_pred_path)
                        if 'torso_pred' in torso_pred_data:
                            torso_pred = torso_pred_data['torso_pred'].astype(np.float32)
                            neck = torso_pred[:, 0:6]
                            torso_input = torso_pred  # full 12-dim
                        else:
                            neck = gt_neck
                            torso_input = gt_torso
                    else:
                        neck = gt_neck
                        torso_input = gt_torso

                sample_data = {
                    'head_gt': head_gt,
                    'neck': neck,
                    'torso': torso_input,
                    'num_frames': T,
                }
                self.data.append(sample_data)

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
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        sample = self.data[window['sample_idx']]
        start, end = window['start'], window['end']

        head_gt = sample['head_gt'][start:end]  # (T, 12)
        neck = sample['neck'][start:end]          # (T, 6)
        torso = sample['torso'][start:end]        # (T, 12)

        return {
            'head_gt': torch.from_numpy(head_gt.copy()).float(),
            'neck': torch.from_numpy(neck.copy()).float(),
            'torso': torch.from_numpy(torso.copy()).float(),
        }


class HeadRegressorLoss(nn.Module):
    """MSE position loss + velocity loss for head regressor."""

    def __init__(
        self,
        pos_weight: float = 1.0,
        vel_weight: float = 0.5,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred:   (B, T, 12)
            target: (B, T, 12)

        Returns:
            total: scalar loss
            loss_dict: dict of loss components
        """
        pos_loss = F.mse_loss(pred, target)

        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(pred_vel, target_vel)

        total = self.pos_weight * pos_loss + self.vel_weight * vel_loss

        loss_dict = {
            'pos': pos_loss.item(),
            'vel': vel_loss.item(),
            'total': total.item(),
        }
        return total, loss_dict


def train_epoch(model, loader, optimizer, device, criterion, scaler=None, use_full_torso=False):
    model.train()
    total_losses = {}
    num_batches = 0

    for batch in tqdm(loader, desc='Training', leave=False):
        head_gt = batch['head_gt'].to(device)
        inp = batch['torso'].to(device) if use_full_torso else batch['neck'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(inp)
            loss, loss_dict = criterion(pred, head_gt)

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
def validate(model, loader, device, criterion, use_full_torso=False):
    model.eval()
    total_losses = {}
    num_batches = 0

    for batch in tqdm(loader, desc='Validating', leave=False):
        head_gt = batch['head_gt'].to(device)
        inp = batch['torso'].to(device) if use_full_torso else batch['neck'].to(device)

        with torch.amp.autocast('cuda'):
            pred = model(inp)
            loss, loss_dict = criterion(pred, head_gt)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        num_batches += 1

    return {k: v / num_batches for k, v in total_losses.items()}


def main():
    parser = argparse.ArgumentParser(description='Train HeadRegressor (non-diffusion)')
    parser.add_argument('--cache_dir', type=str, default='MOSA_dataset/training_cache_B',
                        help='Training cache (GT joints)')
    parser.add_argument('--torso_pred_dir', type=str, default='MOSA_dataset/phase1_torso_predictions',
                        help='Phase1 torso predictions (for C7+CLAV neck input)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage3_head')
    parser.add_argument('--use_gt_neck', action='store_true',
                        help='Use GT neck (C7+CLAV) instead of torso predictions')
    parser.add_argument('--model_type', type=str, default='residual',
                        choices=['mlp', 'residual'],
                        help='mlp: original HeadRegressor (neck 6-dim), '
                             'residual: HeadResidualRegressor (torso 12-dim, mean-offset + residual)')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--window_size', type=int, default=480)
    parser.add_argument('--stride', type=int, default=240)
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--no_wandb', action='store_true', default=True)
    parser.add_argument('--compile', action='store_true', help='Use torch.compile() (PyTorch 2.0+)')
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
                name=f'head_regressor_{datetime.now():%Y%m%d_%H%M%S}',
                config=vars(args),
            )
        except ImportError:
            use_wandb = False

    use_full_torso = (args.model_type == 'residual')

    # Datasets
    train_dataset = HeadDataset(
        cache_dir=args.cache_dir,
        torso_pred_dir=args.torso_pred_dir,
        split='train',
        window_size=args.window_size,
        stride=args.stride,
        use_gt_neck=args.use_gt_neck,
        use_full_torso=use_full_torso,
    )
    val_dataset = HeadDataset(
        cache_dir=args.cache_dir,
        torso_pred_dir=args.torso_pred_dir,
        split='test',
        window_size=args.window_size,
        stride=args.stride,
        use_gt_neck=args.use_gt_neck,
        use_full_torso=use_full_torso,
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

    # Model
    if args.model_type == 'residual':
        model = HeadResidualRegressor(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=0.1,
        ).to(device)
        print(f"HeadResidualRegressor (mean-offset + residual)")
        print(f"  input: full torso (T, 12) [C7, CLAV, T10, STRN]")
    else:
        model = HeadRegressor(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=0.1,
        ).to(device)
        print(f"HeadRegressor (original MLP)")
        print(f"  input: neck (T, 6) [C7, CLAV]")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  output: head (T, 12) [RFHD, LFHD, RBHD, LBHD]")

    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')

    criterion = HeadRegressorLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7,
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nTraining HeadRegressor (patience={args.patience})...")
    print(f"  Neck source: {'GT' if args.use_gt_neck else 'phase1_torso_predictions'}")

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, criterion, scaler,
                                     use_full_torso=use_full_torso)
        val_metrics = validate(model, val_loader, device, criterion,
                               use_full_torso=use_full_torso)
        scheduler.step(val_metrics['total'])

        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train: {train_metrics['total']:.4f} "
            f"(pos={train_metrics['pos']:.4f}, vel={train_metrics['vel']:.4f}) | "
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
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total'],
                'config': {
                    **vars(args),
                    'dropout': 0.1,
                    'head_indices': HEAD_INDICES,
                    'output_dim': 12,
                    'input_dim': 12 if use_full_torso else 6,
                },
            }, output_dir / 'best.pt')
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
                break

    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
