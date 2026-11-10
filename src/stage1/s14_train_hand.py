#!/usr/bin/env python3
"""
s14_train_hand.py: Train Stage 1 wrist (hand) regression model.

Paper Stage 1 model: BiLSTM (h512, l6) with v3 loss
(L1 + velocity + speed + Z-range/std matching).

Input features (352-dim with --use_dur_ioi, 264-dim without):
  piano_roll (88), onset_density (88), onset_mask (88), dur_ioi_ratio (88, optional)

Usage:
    bash scripts/train/stage1.sh                 # production Stage 1
    bash scripts/ablation/stage1_v1v2.sh         # Table 3(C) loss ablation (v1/v2/v3)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import FPS, RWRIST_IDX, LWRIST_IDX
from models.hand_baselines import (
    HandLSTMRegression,
    compute_hand_regression_loss,
    compute_hand_regression_loss_v2,
    compute_hand_regression_loss_v3,
)


class HandDataset(Dataset):
    """Window-based dataset for Stage 1 wrist regression training."""
    
    def __init__(self, cache_dir, split='train', window_size=480, stride=240,
                 require_aria=False):
        self.cache_dir = Path(cache_dir)
        self.window_size = window_size
        self.stride = stride
        self.require_aria = require_aria

        with open(self.cache_dir / 'metadata.json') as f:
            self.metadata = json.load(f)

        self.data = []
        self._load_all_data(split)

        self.windows = []
        self._create_windows()
        print(f"[{split}] {len(self.windows)} windows from {len(self.data)} samples"
              f" (require_aria={require_aria})")

    def _load_all_data(self, split):
        split_dir = self.cache_dir / split
        samples = self.metadata['splits'][split]['samples']

        for sample in tqdm(samples, desc=f'Loading {split}', leave=False):
            npz_path = split_dir / f"{sample['idx']:05d}.npz"
            if not npz_path.exists():
                continue

            data = np.load(npz_path)
            has_aria = 'aria_embed' in data and not np.all(data['aria_embed'] == 0)
            # ARIA is only a hard requirement when the model actually consumes it
            # (use_aria / L4 input level / ARIA FiLM). Otherwise skip the filter
            # so the main pipeline can be reproduced without the ARIA extractor.
            if self.require_aria and not has_aria:
                continue

            joints = data['joints'].astype(np.float32)
            hand = np.concatenate([
                joints[:, LWRIST_IDX],
                joints[:, RWRIST_IDX],
            ], axis=-1)

            if has_aria:
                aria_arr = data['aria_embed'].astype(np.float32)
                aria_is_temporal = aria_arr.ndim == 2
            else:
                # Zero-fill so downstream code can always read the key; it is
                # ignored by models built with use_aria=False.
                aria_arr = np.zeros((1, 512), dtype=np.float32)
                aria_is_temporal = True

            sample_data = {
                'hand': hand,
                'active_keys': data['active_keys'],
                'aria_embed': aria_arr,
                'aria_is_temporal': aria_is_temporal,
                'num_frames': sample['num_frames'],
                'onset_density_per_key': data['onset_density_per_key'].astype(np.float32) if 'onset_density_per_key' in data else None,
                'key_onset': data['key_onset'].astype(np.float32) if 'key_onset' in data else None,
                'dur_ioi_ratio': data['dur_ioi_ratio'].astype(np.float32) if 'dur_ioi_ratio' in data else None,
            }
            self.data.append(sample_data)
    
    def _create_windows(self):
        for idx, sample in enumerate(self.data):
            num_frames = sample['num_frames']
            for start in range(0, num_frames - self.window_size + 1, self.stride):
                self.windows.append({
                    'sample_idx': idx,
                    'start': start,
                    'end': start + self.window_size,
                    'aria_idx': start // self.stride,
                })
    
    def _to_multihot(self, indices):
        T = indices.shape[0]
        mh = np.zeros((T, 88), dtype=np.float32)
        for t in range(T):
            for k in indices[t]:
                if 0 <= k < 88:
                    mh[t, k] = 1.0
        return mh
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        w = self.windows[idx]
        sample = self.data[w['sample_idx']]
        s, e = w['start'], w['end']
        
        hand = sample['hand'][s:e]
        midi = self._to_multihot(sample['active_keys'][s:e])
        
        aria = sample['aria_embed']
        if sample['aria_is_temporal'] and aria.ndim == 2:
            aria = aria[min(w['aria_idx'], aria.shape[0] - 1)]
        
        result = {
            'hand': torch.from_numpy(hand.copy()).float(),
            'midi': torch.from_numpy(midi).float(),
            'aria': torch.from_numpy(aria.copy()).float(),
        }
        
        if sample['onset_density_per_key'] is not None:
            result['velocity'] = torch.from_numpy(sample['onset_density_per_key'][s:e]).float()
        else:
            result['velocity'] = torch.zeros_like(result['midi'])
        
        if sample['key_onset'] is not None:
            result['onset'] = torch.from_numpy(sample['key_onset'][s:e]).float()
        else:
            result['onset'] = torch.zeros_like(result['midi'])

        if sample['dur_ioi_ratio'] is not None:
            result['dur_ioi_ratio'] = torch.from_numpy(sample['dur_ioi_ratio'][s:e]).float()
        else:
            result['dur_ioi_ratio'] = torch.zeros_like(result['midi'])

        return result


def _build_midi_dict(batch, device, no_velocity=False, no_onset=False, use_dur_ioi=False):
    """Build MIDI feature dict for regression models."""
    midi = batch['midi'].to(device)       # piano roll (B, T, 88)
    velocity = batch['velocity'].to(device)
    onset = batch['onset'].to(device)
    if no_velocity:
        velocity = torch.zeros_like(velocity)
    if no_onset:
        onset = torch.zeros_like(onset)
    d = {'piano_roll': midi, 'velocity': velocity, 'onset': onset}
    if use_dur_ioi and 'dur_ioi_ratio' in batch:
        d['dur_ioi_ratio'] = batch['dur_ioi_ratio'].to(device)
    return d


def train_epoch_regression(model, loader, optimizer, device, scaler=None,
                           no_velocity=False, no_onset=False,
                           onset_weight=2.0, z_weight=2.0, vel_weight=0.5,
                           use_dur_ioi=False,
                           loss_version='v1',
                           z_range_weight=2.0, z_std_weight=2.0):
    """Train one epoch (BiLSTM regression)."""
    model.train()
    total_losses = {}
    mpjpe_sum = 0.0
    mpjpe_count = 0
    n_batches = 0

    for batch in tqdm(loader, desc='Training', leave=False):
        hand = batch['hand'].to(device)
        onset_raw = batch['onset'].to(device)
        midi_dict = _build_midi_dict(batch, device, no_velocity, no_onset, use_dur_ioi)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(midi_dict)
            if loss_version == 'v3':
                losses = compute_hand_regression_loss_v3(
                    pred, hand, onset=onset_raw,
                    vel_weight=vel_weight,
                    onset_weight=onset_weight,
                    z_weight=z_weight,
                    z_range_weight=z_range_weight,
                    z_std_weight=z_std_weight,
                )
            elif loss_version == 'v2':
                losses = compute_hand_regression_loss_v2(
                    pred, hand, onset=onset_raw,
                    vel_weight=vel_weight,
                    onset_weight=onset_weight,
                    z_weight=z_weight,
                )
            else:
                losses = compute_hand_regression_loss(
                    pred, hand, onset=onset_raw,
                    vel_weight=vel_weight,
                    onset_weight=onset_weight,
                    z_weight=z_weight,
                )
            loss = losses['loss']

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

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()

        with torch.no_grad():
            B, T, _ = pred.shape
            ae = (pred.float().reshape(B, T, 2, 3) - hand.reshape(B, T, 2, 3)).norm(dim=-1)
            mpjpe_sum += ae.sum().item()
            mpjpe_count += ae.numel()

        n_batches += 1

    avg = {k: v / n_batches for k, v in total_losses.items()}
    avg['mpjpe_mm'] = (mpjpe_sum / mpjpe_count) * 1000
    return avg


@torch.no_grad()
def validate_regression(model, loader, device, no_velocity=False, no_onset=False,
                        onset_weight=2.0, z_weight=2.0, vel_weight=0.5,
                        use_dur_ioi=False,
                        loss_version='v1',
                        z_range_weight=2.0, z_std_weight=2.0):
    """Validate (BiLSTM regression). Returns individual losses + MPJPE."""
    model.eval()
    total_losses = {}
    mpjpe_sum = 0.0
    mpjpe_count = 0
    n_batches = 0

    for batch in tqdm(loader, desc='Validating', leave=False):
        hand = batch['hand'].to(device)
        onset_raw = batch['onset'].to(device)
        midi_dict = _build_midi_dict(batch, device, no_velocity, no_onset, use_dur_ioi)

        pred = model(midi_dict)
        if loss_version == 'v3':
            losses = compute_hand_regression_loss_v3(
                pred, hand, onset=onset_raw,
                vel_weight=vel_weight,
                onset_weight=onset_weight,
                z_weight=z_weight,
                z_range_weight=z_range_weight,
                z_std_weight=z_std_weight,
            )
        elif loss_version == 'v2':
            losses = compute_hand_regression_loss_v2(
                pred, hand, onset=onset_raw,
                vel_weight=vel_weight,
                onset_weight=onset_weight,
                z_weight=z_weight,
            )
        else:
            losses = compute_hand_regression_loss(
                pred, hand, onset=onset_raw,
                vel_weight=vel_weight,
                onset_weight=onset_weight,
                z_weight=z_weight,
            )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()

        B, T, _ = pred.shape
        ae = (pred.float().reshape(B, T, 2, 3) - hand.reshape(B, T, 2, 3)).norm(dim=-1)
        mpjpe_sum += ae.sum().item()
        mpjpe_count += ae.numel()

        n_batches += 1

    avg = {k: v / n_batches for k, v in total_losses.items()}
    avg['mpjpe_mm'] = (mpjpe_sum / mpjpe_count) * 1000
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default='./MOSA_dataset/training_cache_W05')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/stage1')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--window_size', type=int, default=480)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--onset_weight', type=float, default=2.0)
    parser.add_argument('--z_weight', type=float, default=2.0,
                        help='Z-axis weight in loss')
    parser.add_argument('--loss_version', type=str, default='v3', choices=['v1', 'v2', 'v3'],
                        help='v1: MSE+vel(0.5), v2: L1+vel(2.0)+speed(1.0), v3: v2 + z-amplitude matching (paper)')
    parser.add_argument('--z_range_weight', type=float, default=2.0,
                        help='[v3] Weight for z-axis range matching loss')
    parser.add_argument('--z_std_weight', type=float, default=2.0,
                        help='[v3] Weight for z-axis std matching loss')
    parser.add_argument('--vel_w', type=float, default=2.0,
                        help='Velocity matching weight')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='pianoid',
                        help='Wandb project name')
    parser.add_argument('--use_dur_ioi', action='store_true',
                        help='Add dur_ioi_ratio as 4th MIDI feature (matches input_level L3)')
    parser.add_argument('--no_velocity', action='store_true',
                        help='Zero out velocity features (for ablation study)')
    parser.add_argument('--no_onset', action='store_true',
                        help='Zero out onset features (for ablation study)')
    parser.add_argument('--input_level', type=str, default=None,
                        choices=['L0', 'L1', 'L2', 'L3'],
                        help='Input feature level: L0=piano_roll, L1=+onset, L2=+onset_density, L3=+dur_ioi (paper)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default 42)')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='hand_bilstm',
                        choices=['hand_bilstm'],
                        help='Stage 1 model (paper: hand_bilstm, BiLSTM h512 l6).')
    args = parser.parse_args()

    # --input_level overrides individual feature flags
    if args.input_level is not None:
        level = args.input_level
        args.no_onset = level == 'L0'
        args.no_velocity = level in ['L0', 'L1']
        args.use_dur_ioi = level == 'L3'
        print(f"[input_level={level}] no_onset={args.no_onset}, no_velocity={args.no_velocity}, "
              f"use_dur_ioi={args.use_dur_ioi}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from project_utils.seeding import fix_seed, seed_worker, make_generator
    fix_seed(args.seed)
    print(f"Seed: {args.seed} (cudnn deterministic on)")
    print(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading datasets...")
    train_dataset = HandDataset(args.cache_dir, 'train', args.window_size, require_aria=False)
    val_dataset = HandDataset(args.cache_dir, 'val', args.window_size, require_aria=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True,
                              worker_init_fn=seed_worker, generator=make_generator(args.seed))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True,
                            worker_init_fn=seed_worker, generator=make_generator(args.seed))

    # Model: paper Stage 1 = BiLSTM h512 l6
    print(f"\nCreating model (type={args.model_type})...")
    midi_dim = 352 if args.use_dur_ioi else 264  # 88*4 or 88*3
    model = HandLSTMRegression(
        hand_dim=6, midi_dim=midi_dim, hidden_dim=args.latent_dim,
        num_layers=args.num_layers, dropout=0.1,
    ).to(device)
    print(f"  {args.model_type}: hidden={args.latent_dim}, layers={args.num_layers}, "
          f"use_dur_ioi={args.use_dur_ioi}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    # Wandb
    if not args.no_wandb:
        import wandb
        input_level = args.input_level or 'custom'
        run_name = f"s1_{args.model_type}_{input_level}_h{args.latent_dim}_l{args.num_layers}_{datetime.now().strftime('%m%d_%H%M')}"
        tags = ['stage1', args.model_type, f'input_{input_level}']
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=tags,
        )
    
    # Training — best model selected by validation MPJPE (not composite loss)
    best_val_mpjpe = float('inf')
    patience_counter = 0
    start_epoch = 0
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_mpjpe = ckpt.get('val_mpjpe_mm', ckpt.get('val_loss', float('inf')))
        print(f"Resumed from epoch {start_epoch}")
    
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        train_losses = train_epoch_regression(
            model, train_loader, optimizer, device, scaler,
            no_velocity=args.no_velocity, no_onset=args.no_onset,
            onset_weight=args.onset_weight, z_weight=args.z_weight, vel_weight=args.vel_w,
            use_dur_ioi=args.use_dur_ioi,
            loss_version=args.loss_version,
            z_range_weight=args.z_range_weight, z_std_weight=args.z_std_weight,
        )
        val_losses = validate_regression(
            model, val_loader, device,
            no_velocity=args.no_velocity, no_onset=args.no_onset,
            onset_weight=args.onset_weight, z_weight=args.z_weight, vel_weight=args.vel_w,
            use_dur_ioi=args.use_dur_ioi,
            loss_version=args.loss_version,
            z_range_weight=args.z_range_weight, z_std_weight=args.z_std_weight,
        )
        scheduler.step(val_losses['loss'])

        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} | lr: {lr:.1e}")
        t, v = train_losses, val_losses
        print(f"  Train: loss={t['loss']:.5f}  pos={t.get('pos_loss', 0):.5f}  "
              f"vel={t.get('vel_loss', 0):.5f}  spd={t.get('speed_loss', 0):.5f}  "
              f"MPJPE={t['mpjpe_mm']:.1f}mm")
        if 'z_range_loss' in t:
            print(f"         z_range={t['z_range_loss']:.5f}  z_std={t['z_std_loss']:.5f}")
        print(f"  Val:   loss={v['loss']:.5f}  pos={v.get('pos_loss', 0):.5f}  "
              f"vel={v.get('vel_loss', 0):.5f}  spd={v.get('speed_loss', 0):.5f}  "
              f"MPJPE={v['mpjpe_mm']:.1f}mm")
        if 'z_range_loss' in v:
            print(f"         z_range={v['z_range_loss']:.5f}  z_std={v['z_std_loss']:.5f}")

        if not args.no_wandb:
            import wandb
            log_dict = {'lr': lr, 'epoch': epoch + 1}
            for prefix, losses in [('train', train_losses), ('val', val_losses)]:
                for k, v in losses.items():
                    log_dict[f'{prefix}/{k}'] = v
            wandb.log(log_dict)

        val_metric = val_losses['mpjpe_mm']
        is_best = val_metric < best_val_mpjpe

        if is_best:
            best_val_mpjpe = val_metric
            patience_counter = 0

            save_config = vars(args).copy()
            save_config['model_type'] = args.model_type
            save_config['use_dur_ioi'] = args.use_dur_ioi
            ckpt_name = f'{args.model_type}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['loss'],
                'val_mpjpe_mm': val_losses['mpjpe_mm'],
                'config': save_config,
            }, output_dir / ckpt_name)
            print(f"  [Best model saved — MPJPE={val_metric:.1f}mm]")
        else:
            patience_counter += 1
        
        # Periodic save
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_losses['loss'],
                'val_mpjpe_mm': val_losses['mpjpe_mm'],
                'config': vars(args),
            }, output_dir / f'{args.model_type}_epoch{epoch+1:04d}.pt')
        
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    if not args.no_wandb:
        import wandb
        wandb.finish()
    
    print("\nTraining complete!")
    print(f"Best val MPJPE: {best_val_mpjpe:.2f}mm")


if __name__ == '__main__':
    main()
