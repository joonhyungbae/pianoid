#!/usr/bin/env python3
"""
s30_train_arm_mdm.py: Train ArmMDM (Stage 3)

ArmMDM: Wrists + Chest (T10, STRN) + ARIA -> Arms (RSHO, RELB, LSHO, LELB)

Pipeline:
    phase0_hand_predictions: hand_pred (T, 12) -> wrists (T, 6)
    phase1_torso_predictions: torso_pred (T, 12) -> chest (T10+STRN) (T, 6)
    ARIA (512) from cache
    -> ArmMDM -> arms_gt (T, 12) [RSHO, RELB, LSHO, LELB]

Spatial conditioning:
    - wrists: hand_pred[:, 3:6] (RWRIST) + hand_pred[:, 0:3] (LWRIST) -> (T, 6)
    - chest:  torso_pred[:, 6:12] (T10+STRN) -> (T, 6)
    - combined spatial_cond: concat([wrists, chest]) -> (T, 12)

Usage:
    # Step 1: Train TorsoMDM
    python src/stage2/s20_train_torso_mdm.py

    # Step 2: Precompute torso predictions
    python src/stage2/s21_precompute_torso.py

    # Step 3: Train ArmMDM
    python src/stage3/s30_train_arm_mdm.py
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

# Add paths - MDM FIRST to avoid utils conflict
MDM_PATH = str(Path(__file__).parent.parent.parent / 'vendor' / 'MDM')
sys.path.insert(0, MDM_PATH)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import (
    FPS,
    RSHO_IDX, RELB_IDX, LSHO_IDX, LELB_IDX,
    RWRIST_IDX, LWRIST_IDX, T10_IDX, STRN_IDX,
)
from diffusion_lib import create_diffusion, q_sample, ddim_sample
from models.conditioned_mdm import ConditionedMDM
from models.hierarchical_mdm import MDMLoss


# Arm output indices (global joint indices)
ARM_INDICES = [RSHO_IDX, RELB_IDX, LSHO_IDX, LELB_IDX]  # [8, 9, 12, 13]


class ArmMDMWrapper(ConditionedMDM):
    """
    ArmMDM: Wrists + Chest + ARIA -> Arms (RSHO, RELB, LSHO, LELB)

    Input conditions:
        - spatial_cond: (B, T, 12) = wrists(6) + chest_T10_STRN(6)
            wrists: [RWRIST(0:3), LWRIST(3:6)]  (from hand_pred)
            chest:  [T10(0:3), STRN(3:6)]       (from torso_pred[:, 6:12])
        - ARIA: (B, 512) global music embedding

    Output: (B, T, 12) = 4 joints x 3 coords [RSHO, RELB, LSHO, LELB]
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
            spatial_cond_dim=12,  # wrists(6) + chest(6)
        )
        self.use_aria = use_aria

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        spatial_cond: torch.Tensor,
        aria: Optional[torch.Tensor] = None,
        force_mask: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, 12) noisy arm positions
            timesteps: (B,) diffusion timesteps
            spatial_cond: (B, T, 12) wrists(6) + chest(6)
            aria: (B, 512) ARIA embedding
            force_mask: Force mask conditions for CFG

        Returns:
            pred: (B, T, 12) predicted clean arm positions
        """
        return super().forward(
            x, timesteps,
            spatial_cond=spatial_cond,
            global_cond=aria if self.use_aria else None,
            temporal_cond=None,
            force_mask=force_mask,
        )


class ArmDataset(Dataset):
    """Dataset for ArmMDM training.

    Loads:
        - Ground truth arm joints from training cache (joints[:, ARM_INDICES, :])
        - Hand predictions from phase0_hand_predictions/ (for wrists)
        - Torso predictions from phase1_torso_predictions/ (for chest T10+STRN)
        - ARIA embeddings from training cache

    Spatial conditioning:
        - wrists = [RWRIST, LWRIST] from hand_pred (T, 6)
            hand_pred layout: [LWRIST(0:3), RWRIST(3:6)]
            so: right_wrist = hand_pred[:, 3:6], left_wrist = hand_pred[:, 0:3]
            wrists = concat([right_wrist, left_wrist]) -> (T, 6)
        - chest = torso_pred[:, 6:12] = [T10(6:9), STRN(9:12)] -> (T, 6)
        - spatial_cond = concat([wrists, chest]) -> (T, 12)
    """

    def __init__(
        self,
        cache_dir: str,
        hand_pred_dir: str,
        torso_pred_dir: str,
        split: str = 'train',
        window_size: int = 480,
        stride: int = 240,
        use_gt_inputs: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.hand_pred_dir = Path(hand_pred_dir)
        self.torso_pred_dir = Path(torso_pred_dir)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.use_gt_inputs = use_gt_inputs

        self.data = []
        self._load_all_data()

        self.windows = []
        self._create_windows()

        print(f"[{split}] {len(self.windows)} windows from {len(self.data)} samples, "
              f"use_gt_inputs={use_gt_inputs}")

    def _load_all_data(self):
        cache_split_dir = self.cache_dir / self.split
        hand_pred_split_dir = self.hand_pred_dir / self.split
        torso_pred_split_dir = self.torso_pred_dir / self.split

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

                # GT Arms: [RSHO, RELB, LSHO, LELB] -> (T, 12)
                arm_gt = joints[:, ARM_INDICES, :].reshape(T, 12)

                # ARIA embedding
                aria_embed = cache_data['aria_embed'].astype(np.float32)
                aria_is_temporal = aria_embed.ndim == 2

                # GT wrists (fallback)
                gt_right_wrist = joints[:, RWRIST_IDX]  # (T, 3)
                gt_left_wrist = joints[:, LWRIST_IDX]   # (T, 3)
                gt_wrists = np.concatenate([gt_right_wrist, gt_left_wrist], axis=-1)  # (T, 6)

                # GT chest (fallback): T10 + STRN -> (T, 6)
                gt_chest = joints[:, [T10_IDX, STRN_IDX], :].reshape(T, 6)

                if self.use_gt_inputs:
                    wrists = gt_wrists
                    chest = gt_chest
                else:
                    # Load phase0 hand predictions for wrists
                    hand_pred_path = hand_pred_split_dir / cache_path.name
                    if hand_pred_path.exists():
                        hand_pred_data = np.load(hand_pred_path)
                        if 'hand_pred' in hand_pred_data:
                            hand_pred = hand_pred_data['hand_pred'].astype(np.float32)
                        elif 'pred' in hand_pred_data:
                            hand_pred = hand_pred_data['pred'].astype(np.float32)
                        else:
                            hand_pred = None
                    else:
                        hand_pred = None

                    if hand_pred is not None:
                        # hand_pred layout: [LWRIST(0:3), RWRIST(3:6)]
                        right_wrist = hand_pred[:, 3:6]
                        left_wrist = hand_pred[:, 0:3]
                        wrists = np.concatenate([right_wrist, left_wrist], axis=-1)  # (T, 6)
                    else:
                        wrists = gt_wrists

                    # Load phase1 torso predictions for chest
                    torso_pred_path = torso_pred_split_dir / cache_path.name
                    if torso_pred_path.exists():
                        torso_pred_data = np.load(torso_pred_path)
                        if 'torso_pred' in torso_pred_data:
                            torso_pred = torso_pred_data['torso_pred'].astype(np.float32)
                            # torso layout: [C7(0:3), CLAV(3:6), T10(6:9), STRN(9:12)]
                            chest = torso_pred[:, 6:12]  # T10 + STRN -> (T, 6)
                        else:
                            chest = gt_chest
                    else:
                        chest = gt_chest

                sample_data = {
                    'arm_gt': arm_gt,
                    'wrists': wrists,    # (T, 6) [RWRIST, LWRIST]
                    'chest': chest,      # (T, 6) [T10, STRN]
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

        arm_gt = sample['arm_gt'][start:end]   # (T, 12)
        wrists = sample['wrists'][start:end]   # (T, 6)
        chest = sample['chest'][start:end]     # (T, 6)

        # Combined spatial conditioning: wrists(6) + chest(6) = (T, 12)
        spatial_cond = np.concatenate([wrists, chest], axis=-1)

        aria_data = sample['aria_embed']
        if sample['aria_is_temporal'] and aria_data.ndim == 2:
            aria_idx = min(window['aria_window_idx'], aria_data.shape[0] - 1)
            aria = aria_data[aria_idx]
        else:
            aria = aria_data

        return {
            'arm_gt': torch.from_numpy(arm_gt.copy()).float(),
            'spatial_cond': torch.from_numpy(spatial_cond.copy()).float(),
            'aria': torch.from_numpy(aria.copy()).float(),
        }


def train_epoch(model, diffusion, loader, optimizer, device, criterion, scaler=None):
    model.train()
    total_losses = {}
    num_batches = 0

    for batch in tqdm(loader, desc='Training', leave=False):
        arm_gt = batch['arm_gt'].to(device)
        spatial_cond = batch['spatial_cond'].to(device)
        aria = batch['aria'].to(device)

        B = arm_gt.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

        noise = torch.randn_like(arm_gt)
        x_t = q_sample(diffusion, arm_gt, t, noise=noise)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(x_t, t, spatial_cond, aria)
            loss, loss_dict = criterion(pred, arm_gt)

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
        arm_gt = batch['arm_gt'].to(device)
        spatial_cond = batch['spatial_cond'].to(device)
        aria = batch['aria'].to(device)

        B = arm_gt.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

        noise = torch.randn_like(arm_gt)
        x_t = q_sample(diffusion, arm_gt, t, noise=noise)

        with torch.amp.autocast('cuda'):
            pred = model(x_t, t, spatial_cond, aria)
            loss, loss_dict = criterion(pred, arm_gt)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        num_batches += 1

    return {k: v / num_batches for k, v in total_losses.items()}


def main():
    parser = argparse.ArgumentParser(description='Train ArmMDM (Stage 3)')
    parser.add_argument('--cache_dir', type=str, default='MOSA_dataset/training_cache_B',
                        help='Training cache (GT joints + ARIA)')
    parser.add_argument('--hand_pred_dir', type=str, default='MOSA_dataset/phase0_hand_predictions',
                        help='Phase0 hand predictions (for wrists)')
    parser.add_argument('--torso_pred_dir', type=str, default='MOSA_dataset/phase1_torso_predictions',
                        help='Phase1 torso predictions (for chest T10+STRN)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage3_arm')
    parser.add_argument('--use_gt_inputs', action='store_true',
                        help='Use GT wrists and chest instead of predicted values')
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
                name=f'arm_mdm_{datetime.now():%Y%m%d_%H%M%S}',
                config=vars(args),
            )
        except ImportError:
            use_wandb = False

    # Datasets
    train_dataset = ArmDataset(
        cache_dir=args.cache_dir,
        hand_pred_dir=args.hand_pred_dir,
        torso_pred_dir=args.torso_pred_dir,
        split='train',
        window_size=args.window_size,
        stride=args.stride,
        use_gt_inputs=args.use_gt_inputs,
    )
    val_dataset = ArmDataset(
        cache_dir=args.cache_dir,
        hand_pred_dir=args.hand_pred_dir,
        torso_pred_dir=args.torso_pred_dir,
        split='test',
        window_size=args.window_size,
        stride=args.stride,
        use_gt_inputs=args.use_gt_inputs,
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
    model = ArmMDMWrapper(
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        use_aria=use_aria,
    ).to(device)
    print(f"ArmMDM params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  use_aria={use_aria}, output_dim=12 (4 joints x 3)")
    print(f"  ARM_INDICES: {ARM_INDICES} (RSHO, RELB, LSHO, LELB)")
    print(f"  spatial_cond: wrists(6) [RWRIST, LWRIST] + chest(6) [T10, STRN]")

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

    print(f"\nTraining ArmMDM (patience={args.patience})...")
    print(f"  Input source: {'GT' if args.use_gt_inputs else 'phase0/phase1 predictions'}")

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
                    'arm_indices': ARM_INDICES,
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
