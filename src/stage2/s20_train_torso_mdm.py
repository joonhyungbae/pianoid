#!/usr/bin/env python3
"""
s20_train_torso_mdm.py: Train TorsoMDM (Stage 2)

TorsoMDM: Hand predictions + ARIA -> Torso (C7, CLAV, T10, STRN)

Pipeline:
    Phase 0: Hand predictions (from stage1/hand_bilstm) -> stored in phase0_hand_predictions/
    Stage 2: hand_pred (12) + ARIA (512) -> torso_gt (12) [C7, CLAV, T10, STRN]

Usage:
    python src/stage2/s20_train_torso_mdm.py
    python src/stage2/s20_train_torso_mdm.py --use_gt_hand  # Use GT hand instead
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
from typing import Dict, Optional
from tqdm import tqdm

# Add paths - MDM FIRST to avoid utils conflict
MDM_PATH = str(Path(__file__).parent.parent.parent / 'vendor' / 'MDM')
sys.path.insert(0, MDM_PATH)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import (
    FPS, RWRIST_IDX, LWRIST_IDX, RHAND_IDX, LHAND_IDX,
    C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
)
from diffusion_lib import create_diffusion, q_sample, ddim_sample
from models.conditioned_mdm import ConditionedMDM
from models.hierarchical_mdm import MDMLoss


# Torso output indices (global joint indices)
TORSO_INDICES = [C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX]  # [4, 5, 6, 7]


class TorsoMDMWrapper(ConditionedMDM):
    """
    TorsoMDM: Hand + ARIA -> Torso (C7, CLAV, T10, STRN)

    Input conditions:
        - Hand: (B, T, 12) wrist + finger positions (spatial conditioning)
        - ARIA: (B, 512) global music embedding

    Output: (B, T, 12) = 4 joints x 3 coords [C7, CLAV, T10, STRN]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_mask_prob: float = 0.1,
        use_aria: bool = True,
    ):
        super().__init__(
            output_dim=12,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_mask_prob=cond_mask_prob,
            temporal_cond_dim=0,
            global_cond_dim=512 if use_aria else 0,
            spatial_cond_dim=6,
        )
        self.use_aria = use_aria
        self.use_midi = False
        self.use_pitch_range = False

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        hand: torch.Tensor,
        aria: Optional[torch.Tensor] = None,
        force_mask: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, 12) noisy torso positions
            timesteps: (B,) diffusion timesteps
            hand: (B, T, 12) hand positions (spatial conditioning)
            aria: (B, 512) ARIA embedding
            force_mask: Force mask conditions for CFG

        Returns:
            pred: (B, T, 12) predicted clean torso positions
        """
        return super().forward(
            x, timesteps,
            spatial_cond=hand,
            global_cond=aria if self.use_aria else None,
            temporal_cond=None,
            force_mask=force_mask,
        )


class TorsoDataset(Dataset):
    """Dataset for TorsoMDM training.

    Loads:
        - Ground truth torso joints from training cache (joints[:, TORSO_INDICES, :])
        - Hand predictions from phase0_hand_predictions/
        - ARIA embeddings from training cache

    Args:
        cache_dir: Path to MOSA_dataset/training_cache_B (contains GT joints + ARIA)
        pred_dir:  Path to MOSA_dataset/phase0_hand_predictions (contains hand predictions)
        split: 'train', 'val', or 'test'
        window_size: Sliding window size in frames
        stride: Sliding window stride in frames
        use_gt_hand: If True, use GT hand instead of phase0 predictions
    """

    def __init__(
        self,
        cache_dir: str,
        pred_dir: str,
        split: str = 'train',
        window_size: int = 480,
        stride: int = 240,
        use_gt_hand: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.pred_dir = Path(pred_dir)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.use_gt_hand = use_gt_hand

        self.data = []
        self._load_all_data()

        self.windows = []
        self._create_windows()

        print(f"[{split}] {len(self.windows)} windows from {len(self.data)} samples, "
              f"use_gt_hand={use_gt_hand}")

    def _load_all_data(self):
        cache_split_dir = self.cache_dir / self.split
        pred_split_dir = self.pred_dir / self.split

        if not cache_split_dir.exists():
            raise FileNotFoundError(f"Cache split dir not found: {cache_split_dir}")

        cache_files = sorted(cache_split_dir.glob('*.npz'))

        for cache_path in tqdm(cache_files, desc=f'Loading {self.split}', leave=False):
            try:
                cache_data = np.load(cache_path)

                if 'aria_embed' not in cache_data or np.all(cache_data['aria_embed'] == 0):
                    continue

                joints = cache_data['joints'].astype(np.float32)  # (T, 26, 3)
                T = joints.shape[0]

                # GT Torso: [C7, CLAV, T10, STRN] -> (T, 12)
                torso_gt = joints[:, TORSO_INDICES, :].reshape(T, 12)

                # GT Wrist (for use_gt_hand mode or fallback)
                gt_hand = np.concatenate([
                    joints[:, LWRIST_IDX],   # (T, 3)
                    joints[:, RWRIST_IDX],   # (T, 3)
                ], axis=-1).astype(np.float32)  # (T, 6)

                # ARIA embedding
                aria_embed = cache_data['aria_embed'].astype(np.float32)
                aria_is_temporal = aria_embed.ndim == 2

                # Load phase0 hand predictions if available
                hand_pred = None
                if not self.use_gt_hand:
                    pred_path = pred_split_dir / cache_path.name
                    if pred_path.exists():
                        pred_data = np.load(pred_path)
                        if 'hand_pred' in pred_data:
                            hand_pred = pred_data['hand_pred'].astype(np.float32)
                        elif 'pred' in pred_data:
                            hand_pred = pred_data['pred'].astype(np.float32)

                    if hand_pred is None:
                        # Fall back to GT hand if prediction not available
                        hand_pred = gt_hand

                sample_data = {
                    'torso_gt': torso_gt,
                    'gt_hand': gt_hand,
                    'hand_pred': hand_pred if not self.use_gt_hand else gt_hand,
                    'aria_embed': aria_embed,
                    'aria_is_temporal': aria_is_temporal,
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
                    'aria_window_idx': start // self.stride,
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        sample = self.data[window['sample_idx']]
        start, end = window['start'], window['end']

        torso_gt = sample['torso_gt'][start:end]   # (T, 12)
        hand_pred = sample['hand_pred'][start:end]  # (T, 12)

        aria_data = sample['aria_embed']
        if sample['aria_is_temporal'] and aria_data.ndim == 2:
            aria_idx = min(window['aria_window_idx'], aria_data.shape[0] - 1)
            aria = aria_data[aria_idx]
        else:
            aria = aria_data  # (512,)

        return {
            'torso_gt': torch.from_numpy(torso_gt.copy()).float(),
            'hand_pred': torch.from_numpy(hand_pred.copy()).float(),
            'aria': torch.from_numpy(aria.copy()).float(),
        }


def train_epoch(model, diffusion, loader, optimizer, device, criterion, scaler=None):
    model.train()
    total_losses = {}
    num_batches = 0

    for batch in tqdm(loader, desc='Training', leave=False):
        torso_gt = batch['torso_gt'].to(device)
        hand_pred = batch['hand_pred'].to(device)
        aria = batch['aria'].to(device)

        B = torso_gt.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

        noise = torch.randn_like(torso_gt)
        x_t = q_sample(diffusion, torso_gt, t, noise=noise)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(x_t, t, hand_pred, aria)
            loss, loss_dict = criterion(pred, torso_gt)

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
def validate(model, diffusion, loader, device, criterion):
    model.eval()
    total_losses = {}
    num_batches = 0

    for batch in tqdm(loader, desc='Validating', leave=False):
        torso_gt = batch['torso_gt'].to(device)
        hand_pred = batch['hand_pred'].to(device)
        aria = batch['aria'].to(device)

        B = torso_gt.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

        noise = torch.randn_like(torso_gt)
        x_t = q_sample(diffusion, torso_gt, t, noise=noise)

        with torch.amp.autocast('cuda'):
            pred = model(x_t, t, hand_pred, aria)
            loss, loss_dict = criterion(pred, torso_gt)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        num_batches += 1

    return {k: v / num_batches for k, v in total_losses.items()}


def main():
    parser = argparse.ArgumentParser(description='Train TorsoMDM (Stage 2)')
    parser.add_argument('--cache_dir', type=str, default='MOSA_dataset/training_cache_B')
    parser.add_argument('--pred_dir', type=str, default='MOSA_dataset/phase0_hand_predictions')
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage2_torso')
    parser.add_argument('--use_gt_hand', action='store_true', help='Use GT hand instead of phase0 predictions')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--window_size', type=int, default=480)
    parser.add_argument('--stride', type=int, default=240)
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--no_wandb', action='store_true', default=True)
    parser.add_argument('--no_aria', action='store_true', help='Disable ARIA embedding')
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
                name=f'torso_mdm_{datetime.now():%Y%m%d_%H%M%S}',
                config=vars(args),
            )
        except ImportError:
            use_wandb = False

    # Datasets
    train_dataset = TorsoDataset(
        cache_dir=args.cache_dir,
        pred_dir=args.pred_dir,
        split='train',
        window_size=args.window_size,
        stride=args.stride,
        use_gt_hand=args.use_gt_hand,
    )
    val_dataset = TorsoDataset(
        cache_dir=args.cache_dir,
        pred_dir=args.pred_dir,
        split='test',
        window_size=args.window_size,
        stride=args.stride,
        use_gt_hand=args.use_gt_hand,
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
    use_aria = not args.no_aria
    model = TorsoMDMWrapper(
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        use_aria=use_aria,
    ).to(device)
    print(f"TorsoMDM params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  use_aria={use_aria}, output_dim=12 (4 joints x 3)")

    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')

    diffusion = create_diffusion(num_timesteps=args.diffusion_steps)
    criterion = MDMLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nTraining TorsoMDM (patience={args.patience})...")
    print(f"  TORSO_INDICES: {TORSO_INDICES} (C7, CLAV, T10, STRN)")
    print(f"  Hand source: {'GT' if args.use_gt_hand else 'phase0_hand_predictions'}")

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, diffusion, train_loader, optimizer, device, criterion, scaler)
        val_metrics = validate(model, diffusion, val_loader, device, criterion)
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
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total'],
                'config': {
                    **vars(args),
                    'use_aria': use_aria,
                    'torso_indices': TORSO_INDICES,
                    'output_dim': 12,
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
