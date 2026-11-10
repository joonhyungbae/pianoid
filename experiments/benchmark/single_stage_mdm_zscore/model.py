#!/usr/bin/env python3
"""
Single-Stage MDM with Z-score Normalization (Fair Baseline).

Same architecture tricks as Pianoid's Stage 2 (z-score targets, onset_density,
25 DDIM steps) but applied monolithically to all 26 joints, WITHOUT hierarchical
decomposition. This is the critical ablation: if this baseline matches Pianoid's
ER metrics, the hierarchy is unnecessary; if it doesn't, the decomposition matters.

Changes vs. original single_stage_mdm:
    1. Per-joint-per-axis z-score normalization (computed on train set)
    2. onset_density (1-dim temporal conditioning) instead of raw onset mask
    3. 25 DDIM steps (same as Pianoid Stage 2)
    4. L1 + velocity loss (same as pipeline stages)

Usage:
    python experiments/benchmark/single_stage_mdm_zscore/model.py --mode=train
    python experiments/benchmark/single_stage_mdm_zscore/model.py --mode=eval
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'vendor' / 'MDM'))

from data.constants import NUM_JOINTS, FPS
from diffusion_lib.diffusion_piano import GaussianDiffusion


OUTPUT_DIM = NUM_JOINTS * 3  # 26 × 3 = 78


# ============================================================
# Normalization
# ============================================================

def compute_norm_stats(cache_dir: Path, split='train'):
    """Compute per-joint-per-axis mean and std from training data."""
    split_dir = cache_dir / split
    files = sorted(split_dir.glob('*.npz'))
    all_motion = []
    for f in tqdm(files, desc='Computing norm stats'):
        d = np.load(f)
        joints = d['joints'].reshape(-1, OUTPUT_DIM).astype(np.float64)
        all_motion.append(joints)
    all_motion = np.concatenate(all_motion, axis=0)
    mean = all_motion.mean(axis=0).astype(np.float32)
    std = all_motion.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


# ============================================================
# Model (same architecture as original, + onset_density input)
# ============================================================

class SingleStagePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class SingleStageMDMZscore(nn.Module):
    """Single-Stage MDM with onset_density temporal conditioning."""

    def __init__(self, latent_dim=256, num_layers=8, num_heads=4,
                 dropout=0.1, cond_mask_prob=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_mask_prob = cond_mask_prob
        self.output_dim = OUTPUT_DIM

        self.input_proj = nn.Linear(self.output_dim, latent_dim)

        # MIDI: piano_roll (88) + onset_density_per_key (88) + onset_density (1)
        self.midi_proj = nn.Sequential(
            nn.Linear(88, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
        self.od_key_proj = nn.Sequential(
            nn.Linear(88, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
        self.od_proj = nn.Sequential(
            nn.Linear(1, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))

        self.pos_encoder = SingleStagePositionalEncoding(latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4), nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=latent_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim),
            nn.GELU(), nn.Linear(latent_dim, self.output_dim))

    def _timestep_embedding(self, t):
        half = self.latent_dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return self.time_embed(torch.cat([torch.sin(emb), torch.cos(emb)], dim=1))

    def forward(self, x, timesteps, midi, od_key=None, od_scalar=None,
                force_mask=False):
        B, T, _ = x.shape
        device = x.device

        mask_cond = force_mask
        if self.training and not force_mask:
            mask_cond = torch.rand(B, device=device) < self.cond_mask_prob
            mask_cond = mask_cond[:, None, None]

        h = self.input_proj(x)
        midi_emb = self.midi_proj(midi)
        od_key_emb = self.od_key_proj(od_key) if od_key is not None else torch.zeros_like(midi_emb)
        od_emb = self.od_proj(od_scalar) if od_scalar is not None else torch.zeros(B, T, self.latent_dim, device=device)

        if isinstance(mask_cond, torch.Tensor) and mask_cond.any():
            midi_emb = midi_emb * (~mask_cond).float()
            od_key_emb = od_key_emb * (~mask_cond).float()
            od_emb = od_emb * (~mask_cond).float()

        t_emb = self._timestep_embedding(timesteps).unsqueeze(1).expand(-1, T, -1)
        h = h + midi_emb + od_key_emb + od_emb + t_emb
        h = self.pos_encoder(h)
        h = self.transformer(h)
        return self.output_proj(h)


# ============================================================
# Dataset (z-score normalized targets)
# ============================================================

class ZscoreDataset(Dataset):
    def __init__(self, cache_dir, split='train', window_size=480, stride=240,
                 mean=None, std=None):
        self.cache_dir = Path(cache_dir)
        self.window_size = window_size
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
            for s in range(0, T - window_size + 1, stride):
                self.windows.append((fi, s))
            if T >= window_size and (T - window_size) % stride != 0:
                self.windows.append((fi, T - window_size))
        print(f"{split}: {len(self.files)} files -> {len(self.windows)} windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        fi, s = self.windows[idx]
        d = np.load(self.files[fi])
        e = s + self.window_size

        motion = d['joints'][s:e].reshape(self.window_size, -1).astype(np.float32)
        if self.mean is not None:
            motion = (motion - self.mean) / self.std

        od_key = d['onset_density_per_key'][s:e].astype(np.float32)
        key_onset = d['key_onset'][s:e].astype(np.float32)
        midi = (key_onset > 0).astype(np.float32)
        od_scalar = d['onset_density'][s:e].astype(np.float32).reshape(-1, 1)

        return {
            'motion': torch.from_numpy(motion),
            'midi': torch.from_numpy(midi),
            'od_key': torch.from_numpy(od_key),
            'od_scalar': torch.from_numpy(od_scalar),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute or load norm stats
    stats_path = output_dir / 'norm_stats.npz'
    if stats_path.exists():
        s = np.load(stats_path)
        mean, std = s['mean'], s['std']
        print(f"Loaded norm stats from {stats_path}")
    else:
        mean, std = compute_norm_stats(Path(args.cache_dir))
        np.savez(stats_path, mean=mean, std=std)
        print(f"Computed and saved norm stats to {stats_path}")

    train_ds = ZscoreDataset(args.cache_dir, 'train', mean=mean, std=std)
    val_ds = ZscoreDataset(args.cache_dir, 'val', mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project="pianoid-baseline", name="single_stage_mdm_zscore",
                   config=vars(args))
        use_wandb = True
    else:
        use_wandb = False

    model = SingleStageMDMZscore(
        latent_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule="cosine")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    best_val = float('inf')
    patience_ctr = 0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}"):
            motion = batch['motion'].to(device)
            midi = batch['midi'].to(device)
            od_key = batch['od_key'].to(device)
            od_scalar = batch['od_scalar'].to(device)
            B = motion.shape[0]

            t = torch.randint(0, 1000, (B,), device=device)
            noise = torch.randn_like(motion)
            x_noisy = diffusion.q_sample(motion, t, noise)
            x0_pred = model(x_noisy, t, midi, od_key, od_scalar)

            # L1 position + velocity
            pos_loss = F.l1_loss(x0_pred, motion)
            vel_pred = x0_pred[:, 1:] - x0_pred[:, :-1]
            vel_gt = motion[:, 1:] - motion[:, :-1]
            vel_loss = F.l1_loss(vel_pred, vel_gt)
            loss = pos_loss + 2.0 * vel_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        avg_train = np.mean(losses)

        # --- Val ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                motion = batch['motion'].to(device)
                midi = batch['midi'].to(device)
                od_key = batch['od_key'].to(device)
                od_scalar = batch['od_scalar'].to(device)
                B = motion.shape[0]
                t = torch.randint(0, 1000, (B,), device=device)
                noise = torch.randn_like(motion)
                x_noisy = diffusion.q_sample(motion, t, noise)
                x0_pred = model(x_noisy, t, midi, od_key, od_scalar)
                pos_loss = F.l1_loss(x0_pred, motion)
                vel_pred = x0_pred[:, 1:] - x0_pred[:, :-1]
                vel_gt = motion[:, 1:] - motion[:, :-1]
                vel_loss = F.l1_loss(vel_pred, vel_gt)
                val_losses.append((pos_loss + 2.0 * vel_loss).item())
        avg_val = np.mean(val_losses)

        lr = scheduler.get_last_lr()[0]
        print(f"Ep {epoch+1}: train={avg_train:.4f} val={avg_val:.4f} lr={lr:.2e} pat={patience_ctr}/{args.patience}")

        if use_wandb:
            wandb.log({"epoch": epoch+1, "loss/train": avg_train, "loss/val": avg_val, "lr": lr})

        if avg_val < best_val:
            best_val = avg_val
            patience_ctr = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': {'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers},
                'norm_stats': {'mean': mean.tolist(), 'std': std.tolist()},
                'loss': best_val,
            }, output_dir / 'best_model.pt')
            print(f"  Saved best (val={best_val:.4f})")
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            print(f"Early stop at epoch {epoch+1}")
            break

    if use_wandb:
        wandb.finish()
    print(f"Done. Best val loss: {best_val:.4f}")


# ============================================================
# Evaluation (DDIM 25 steps, denormalize, save per-sample)
# ============================================================

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)

    ckpt = torch.load(output_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    ns = ckpt['norm_stats']
    mean = np.array(ns['mean'], dtype=np.float32)
    std = np.array(ns['std'], dtype=np.float32)

    model = SingleStageMDMZscore(
        latent_dim=cfg['hidden_dim'], num_layers=cfg['num_layers']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule="cosine")
    alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)

    cache_dir = Path(args.cache_dir)
    test_files = sorted((cache_dir / 'test').glob('*.npz'))
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    window_size = 480
    stride = 240
    ddim_steps = args.ddim_steps

    print(f"Evaluating {len(test_files)} test sequences (DDIM {ddim_steps} steps)...")

    for idx, tf in enumerate(tqdm(test_files)):
        d = np.load(tf)
        joints_gt = d['joints']  # (T, 26, 3)
        T_full = joints_gt.shape[0]
        motion_gt = joints_gt.reshape(T_full, -1).astype(np.float32)

        od_key_full = d['onset_density_per_key'].astype(np.float32)
        key_onset_full = d['key_onset'].astype(np.float32)
        midi_full = (key_onset_full > 0).astype(np.float32)
        od_scalar_full = d['onset_density'].astype(np.float32).reshape(-1, 1)

        pred_sum = np.zeros((T_full, OUTPUT_DIM), dtype=np.float64)
        weight_sum = np.zeros((T_full, 1), dtype=np.float64)

        with torch.no_grad():
            for s in range(0, max(1, T_full - window_size + 1), stride):
                e = min(s + window_size, T_full)
                wl = e - s
                if wl < 60:
                    continue

                def pad(arr, dim1):
                    if len(arr) < window_size:
                        return np.concatenate([arr, np.zeros((window_size - len(arr), dim1), np.float32)])
                    return arr

                midi_t = torch.from_numpy(pad(midi_full[s:e], 88)).unsqueeze(0).to(device)
                od_key_t = torch.from_numpy(pad(od_key_full[s:e], 88)).unsqueeze(0).to(device)
                od_sc_t = torch.from_numpy(pad(od_scalar_full[s:e], 1)).unsqueeze(0).to(device)

                # DDIM sampling
                x = torch.randn(1, window_size, OUTPUT_DIM, device=device)
                timesteps = torch.linspace(999, 0, ddim_steps, device=device).long()

                for i, t in enumerate(timesteps):
                    t_batch = t.expand(1)
                    pred = model(x, t_batch, midi_t, od_key_t, od_sc_t)
                    if i < len(timesteps) - 1:
                        alpha_t = alphas_cumprod[t]
                        alpha_prev = alphas_cumprod[timesteps[i + 1]]
                        pred_noise = (x - torch.sqrt(alpha_t) * pred) / torch.sqrt(1 - alpha_t)
                        x = torch.sqrt(alpha_prev) * pred + torch.sqrt(1 - alpha_prev) * pred_noise
                    else:
                        x = pred

                gen_zs = x[0].cpu().numpy()[:wl]

                # Triangular blending
                w = np.minimum(np.arange(wl) + 1, np.arange(wl, 0, -1)).astype(np.float64).reshape(-1, 1)
                pred_sum[s:e] += gen_zs * w
                weight_sum[s:e] += w

        mask = weight_sum > 0
        gen_zs = np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)

        # Denormalize
        generated = gen_zs * std + mean
        generated = generated.reshape(-1, NUM_JOINTS, 3)
        gt = joints_gt.reshape(-1, NUM_JOINTS, 3)

        np.savez(results_dir / f"sample_{idx:03d}.npz", generated=generated, gt=gt)

    print(f"Results saved to {results_dir}")
    print("Next: python experiments/benchmark/evaluate_all.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--cache_dir', default='MOSA_dataset/training_cache_B')
    parser.add_argument('--output_dir', default='experiments/benchmark/single_stage_mdm_zscore/checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--ddim_steps', type=int, default=25)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
