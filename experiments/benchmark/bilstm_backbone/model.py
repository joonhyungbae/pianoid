#!/usr/bin/env python3
"""
BiLSTM-backbone ablation baseline.

Monolithic 26-joint BiLSTM regression with the SAME z-score normalization
and conditioning as Single-Stage MDM+zs, but deterministic (no diffusion).

Purpose: isolate the CONTRIBUTION OF DIFFUSION from the hierarchy claim.
If Pianoid > Single-Stage MDM+zs, it could be hierarchy OR diffusion.
If Single-Stage MDM+zs > BiLSTM-backbone, we've isolated diffusion's role.
If Pianoid > BiLSTM-backbone on ER metrics, the combined effect is clear.

Architecture:
  MIDI (piano_roll 88 + key_onset 88 + onset_density 1) → BiLSTM (h=512, l=6) → 78-dim
  Trained on MOSA with z-score normalized targets, L1 + velocity loss.

Intended to match Single-Stage MDM+zs's training budget as closely as
possible to ensure fair comparison.

Usage:
    python experiments/benchmark/bilstm_backbone/model.py --mode train
    python experiments/benchmark/bilstm_backbone/model.py --mode eval
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from data.constants import NUM_JOINTS

OUTPUT_DIM = NUM_JOINTS * 3  # 26 × 3 = 78
FPS = 60


def compute_norm_stats(cache_dir: Path, split='train'):
    """Compute per-joint-per-axis mean and std (same protocol as Single-Stage MDM+zs)."""
    split_dir = cache_dir / split
    files = sorted(split_dir.glob('*.npz'))
    all_motion = []
    for f in tqdm(files, desc='norm stats'):
        d = np.load(f)
        joints = d['joints'].reshape(-1, OUTPUT_DIM).astype(np.float64)
        all_motion.append(joints)
    all_motion = np.concatenate(all_motion, axis=0)
    mean = all_motion.mean(axis=0).astype(np.float32)
    std = all_motion.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


class BiLSTMBackbone(nn.Module):
    """Monolithic BiLSTM: MIDI features → 78-dim z-score-normalized joint positions."""
    def __init__(self, input_dim=177, hidden_dim=512, num_layers=6):
        super().__init__()
        # 177 = piano_roll 88 + key_onset 88 + onset_density 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=0.1,
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, OUTPUT_DIM),
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        return self.output_proj(h)


class ZscoreDataset(Dataset):
    def __init__(self, cache_dir, split='train', window=480, stride=240, mean=None, std=None):
        self.cache_dir = Path(cache_dir)
        self.window = window
        self.stride = stride
        self.mean = mean
        self.std = std
        split_dir = self.cache_dir / split
        self.files = sorted(split_dir.glob('*.npz'))
        if not self.files:
            raise ValueError(f"No files in {split_dir}")
        self.windows = []
        for fi, f in enumerate(self.files):
            T = np.load(f)['joints'].shape[0]
            for s in range(0, T - window + 1, stride):
                self.windows.append((fi, s))
            if T >= window and (T - window) % stride != 0:
                self.windows.append((fi, T - window))
        print(f"{split}: {len(self.files)} files -> {len(self.windows)} windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        fi, s = self.windows[idx]
        d = np.load(self.files[fi])
        e = s + self.window

        motion = d['joints'][s:e].reshape(self.window, -1).astype(np.float32)
        if self.mean is not None:
            motion = (motion - self.mean) / self.std

        key_onset = d['key_onset'][s:e].astype(np.float32)
        piano_roll = (key_onset > 0).astype(np.float32)
        onset_density = d['onset_density'][s:e].astype(np.float32).reshape(-1, 1)

        midi = np.concatenate([piano_roll, key_onset, onset_density], axis=-1)  # (T, 177)

        return {
            'motion': torch.from_numpy(motion),
            'midi':   torch.from_numpy(midi),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats_path = out / 'norm_stats.npz'
    if stats_path.exists():
        s = np.load(stats_path)
        mean, std = s['mean'], s['std']
    else:
        mean, std = compute_norm_stats(Path(args.cache_dir))
        np.savez(stats_path, mean=mean, std=std)

    train_ds = ZscoreDataset(args.cache_dir, 'train', mean=mean, std=std)
    val_ds = ZscoreDataset(args.cache_dir, 'val', mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate)

    model = BiLSTMBackbone(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)

    best_val = float('inf')
    patience_ctr = 0

    for epoch in range(args.epochs):
        model.train()
        tr_losses = []
        for batch in tqdm(train_loader, desc=f'Ep {epoch+1}'):
            motion = batch['motion'].to(device)
            midi = batch['midi'].to(device)
            pred = model(midi)
            pos_loss = F.l1_loss(pred, motion)
            vel_pred = pred[:, 1:] - pred[:, :-1]
            vel_gt = motion[:, 1:] - motion[:, :-1]
            vel_loss = F.l1_loss(vel_pred, vel_gt)
            loss = pos_loss + 2.0 * vel_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())
        sched.step()
        avg_tr = np.mean(tr_losses)

        model.eval()
        vl_losses = []
        with torch.no_grad():
            for batch in val_loader:
                motion = batch['motion'].to(device)
                midi = batch['midi'].to(device)
                pred = model(midi)
                pos_loss = F.l1_loss(pred, motion)
                vel_pred = pred[:, 1:] - pred[:, :-1]
                vel_gt = motion[:, 1:] - motion[:, :-1]
                vel_loss = F.l1_loss(vel_pred, vel_gt)
                vl_losses.append((pos_loss + 2.0 * vel_loss).item())
        avg_vl = np.mean(vl_losses)

        print(f"Ep {epoch+1}: train={avg_tr:.4f} val={avg_vl:.4f}")
        if avg_vl < best_val:
            best_val = avg_vl
            patience_ctr = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': {'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers},
                'norm_stats': {'mean': mean.tolist(), 'std': std.tolist()},
                'loss': best_val,
            }, out / 'best_model.pt')
            print(f"  saved best {best_val:.4f}")
        else:
            patience_ctr += 1
        if patience_ctr >= args.patience:
            print("Early stop")
            break
    print(f"Done. Best val {best_val:.4f}")


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output_dir)
    ckpt = torch.load(out / 'best_model.pt', map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    ns = ckpt['norm_stats']
    mean = np.array(ns['mean'], dtype=np.float32)
    std = np.array(ns['std'], dtype=np.float32)

    model = BiLSTMBackbone(hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    cache = Path(args.cache_dir)
    test_files = sorted((cache / 'test').glob('*.npz'))
    results = out / 'results'
    results.mkdir(parents=True, exist_ok=True)

    window = 480
    stride = 240
    print(f"Evaluating {len(test_files)} test samples...")
    for idx, tf in enumerate(tqdm(test_files)):
        d = np.load(tf)
        joints_gt = d['joints'].astype(np.float32)
        T = joints_gt.shape[0]
        key_onset = d['key_onset'].astype(np.float32)
        piano_roll = (key_onset > 0).astype(np.float32)
        onset_density = d['onset_density'].astype(np.float32).reshape(-1, 1)
        midi_full = np.concatenate([piano_roll, key_onset, onset_density], axis=-1)

        pred_sum = np.zeros((T, OUTPUT_DIM), dtype=np.float64)
        weight_sum = np.zeros((T, 1), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, max(1, T - window + 1), stride):
                e = min(s + window, T)
                wl = e - s
                if wl < 60:
                    continue
                chunk = midi_full[s:e]
                if wl < window:
                    chunk = np.concatenate([chunk, np.zeros((window - wl, chunk.shape[-1]), np.float32)])
                x = torch.from_numpy(chunk).unsqueeze(0).to(device)
                pred = model(x)[0].cpu().numpy()[:wl]
                w = np.minimum(np.arange(wl) + 1, np.arange(wl, 0, -1)).astype(np.float64).reshape(-1, 1)
                pred_sum[s:e] += pred * w
                weight_sum[s:e] += w
        mask = weight_sum > 0
        gen_zs = np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)
        # Cache joints are already in meters, so no unit conversion needed.
        generated = (gen_zs * std + mean).reshape(-1, NUM_JOINTS, 3)
        gt = joints_gt.reshape(-1, NUM_JOINTS, 3)
        np.savez(results / f'sample_{idx:03d}.npz',
                 generated=generated.astype(np.float32),
                 gt=gt.astype(np.float32))
    print(f"Results saved to {results}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='train', choices=['train', 'eval'])
    p.add_argument('--cache_dir', default='MOSA_dataset/training_cache_B')
    p.add_argument('--output_dir', default='experiments/benchmark/bilstm_backbone/checkpoints')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=50)
    args = p.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
