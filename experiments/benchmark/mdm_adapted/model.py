#!/usr/bin/env python3
"""
MDM-adapted: a faithful adaptation of Tevet et al. 2023 (ICLR) MDM
to MIDI-conditioned pianist motion generation.

Reference implementation: github.com/GuyTevet/motion-diffusion-model
Paper: "Human Motion Diffusion Model", ICLR 2023.

Architecture preserved from original MDM:
    - arch='trans_enc' (TransformerEncoder denoiser)
    - latent_dim=512, num_heads=4, num_layers=8, ff_size=1024
    - Timestep embedding as first token in sequence
    - Sinusoidal positional encoding
    - x0 prediction loss
    - Classifier-free guidance (cond_mask_prob=0.1)

Key adaptation: the CLIP text encoder is replaced with a frame-wise MIDI
projection (piano_roll 88 + key_onset 88 + onset_density 1 → latent_dim).
Unlike text (which is a sentence-level embedding), MIDI is a time series
aligned to the target motion. We inject MIDI features at EACH frame by
adding them to the input_process embeddings (like AdaIN-style conditioning).

This is the standard "frame-wise conditioning" adaptation pattern used
when porting MDM to audio/music-driven motion tasks.

Output: 26-joint XYZ positions in z-score-normalized space, denormalized
at inference time. Saved as (T, 26, 3) float32 .npz files matching the
other baseline outputs.

Usage:
    python experiments/benchmark/mdm_adapted/model.py --mode=train
    python experiments/benchmark/mdm_adapted/model.py --mode=eval
"""
import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'vendor' / 'MDM'))

from data.constants import NUM_JOINTS
from diffusion_lib.diffusion_piano import GaussianDiffusion

OUTPUT_DIM = NUM_JOINTS * 3  # 78


# ─── MDM model components (ported from vendor/MDM/model/mdm.py) ─────────────

class PositionalEncoding(nn.Module):
    """Identical to Tevet et al. 2023 PositionalEncoding."""
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """Identical to Tevet et al. 2023 TimestepEmbedder."""
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    """Project input motion features into latent_dim."""
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.poseEmbedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x):
        # x: (B, T, input_feats) → (T, B, latent_dim)
        x = self.poseEmbedding(x)
        return x.permute(1, 0, 2)


class OutputProcess(nn.Module):
    def __init__(self, latent_dim, output_feats):
        super().__init__()
        self.poseFinal = nn.Linear(latent_dim, output_feats)

    def forward(self, x):
        # x: (T, B, latent_dim) → (B, T, output_feats)
        x = self.poseFinal(x)
        return x.permute(1, 0, 2)


class MDMAdapted(nn.Module):
    """
    MDM with MIDI frame-wise conditioning (replacing CLIP text conditioning).

    Architecture: same as Tevet et al. 2023 trans_enc, but cond_mode='midi'.
    """
    def __init__(self, latent_dim=512, ff_size=1024, num_layers=8, num_heads=4,
                 dropout=0.1, cond_mask_prob=0.1, midi_dim=177):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_mask_prob = cond_mask_prob
        self.input_feats = OUTPUT_DIM  # 78

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)

        self.input_process = InputProcess(self.input_feats, latent_dim)
        self.output_process = OutputProcess(latent_dim, self.input_feats)

        # MIDI conditioning: frame-wise projection (replaces CLIP text embedding)
        self.embed_midi = nn.Sequential(
            nn.Linear(midi_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Transformer encoder (MDM's trans_enc arch)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation='gelu',
        )
        self.seqTransEncoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[1]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(1, bs, 1)
            return cond * (1.0 - mask)
        return cond

    def forward(self, x, timesteps, midi, force_mask=False):
        """
        x: (B, T, 78) noisy motion in z-score space
        timesteps: (B,) diffusion timesteps
        midi: (B, T, midi_dim) frame-wise MIDI conditioning features
        """
        bs, T, _ = x.shape

        # Timestep embedding: (1, B, D)
        time_emb = self.embed_timestep(timesteps)

        # MIDI embedding: (T, B, D), masked for CFG
        midi_emb = self.embed_midi(midi).permute(1, 0, 2)  # (T, B, D)
        midi_emb = self.mask_cond(midi_emb, force_mask=force_mask)

        # Input process: motion → latent
        x_emb = self.input_process(x)  # (T, B, D)

        # Combine motion + MIDI at each frame (AdaIN-style additive)
        x_emb = x_emb + midi_emb

        # Prepend timestep token
        xseq = torch.cat([time_emb, x_emb], dim=0)  # (T+1, B, D)
        xseq = self.sequence_pos_encoder(xseq)

        # Denoise via transformer encoder
        output = self.seqTransEncoder(xseq)[1:]  # drop timestep token → (T, B, D)

        # Project to output motion
        return self.output_process(output)  # (B, T, 78)


# ─── Dataset, training, evaluation ─────────────────────────────────────────

def compute_norm_stats(cache_dir: Path, split='train'):
    """Per-joint-per-axis z-score stats."""
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


class MDMMidiDataset(Dataset):
    def __init__(self, cache_dir, split='train', window=480, stride=240, mean=None, std=None):
        self.cache_dir = Path(cache_dir)
        self.window = window
        self.stride = stride
        self.mean = mean
        self.std = std
        split_dir = self.cache_dir / split
        self.files = sorted(split_dir.glob('*.npz'))
        self.windows = []
        for fi, f in enumerate(self.files):
            T = np.load(f)['joints'].shape[0]
            for s in range(0, T - window + 1, stride):
                self.windows.append((fi, s))
            if T >= window and (T - window) % stride != 0:
                self.windows.append((fi, T - window))
        print(f"{split}: {len(self.files)} files → {len(self.windows)} windows")

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

    train_ds = MDMMidiDataset(args.cache_dir, 'train', mean=mean, std=std)
    val_ds = MDMMidiDataset(args.cache_dir, 'val', mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate)

    model = MDMAdapted(latent_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MDMAdapted parameters: {n_params:,}")

    diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule="cosine")
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
            B = motion.shape[0]
            t = torch.randint(0, 1000, (B,), device=device)
            noise = torch.randn_like(motion)
            x_noisy = diffusion.q_sample(motion, t, noise)
            x0_pred = model(x_noisy, t, midi)
            pos_loss = F.l1_loss(x0_pred, motion)
            vel_pred = x0_pred[:, 1:] - x0_pred[:, :-1]
            vel_gt = motion[:, 1:] - motion[:, :-1]
            vel_loss = F.l1_loss(vel_pred, vel_gt)
            loss = pos_loss + 2.0 * vel_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())
        sched.step()

        model.eval()
        vl_losses = []
        with torch.no_grad():
            for batch in val_loader:
                motion = batch['motion'].to(device)
                midi = batch['midi'].to(device)
                B = motion.shape[0]
                t = torch.randint(0, 1000, (B,), device=device)
                noise = torch.randn_like(motion)
                x_noisy = diffusion.q_sample(motion, t, noise)
                x0_pred = model(x_noisy, t, midi)
                pos_loss = F.l1_loss(x0_pred, motion)
                vel_pred = x0_pred[:, 1:] - x0_pred[:, :-1]
                vel_gt = motion[:, 1:] - motion[:, :-1]
                vel_loss = F.l1_loss(vel_pred, vel_gt)
                vl_losses.append((pos_loss + 2.0 * vel_loss).item())
        avg_vl = np.mean(vl_losses)
        avg_tr = np.mean(tr_losses)
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
        else:
            patience_ctr += 1
        if patience_ctr >= args.patience:
            print(f"Early stop at {epoch+1}")
            break
    print(f"Done. Best val: {best_val:.4f}")


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output_dir)
    ckpt = torch.load(out / 'best_model.pt', map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    ns = ckpt['norm_stats']
    mean = np.array(ns['mean'], dtype=np.float32)
    std = np.array(ns['std'], dtype=np.float32)

    model = MDMAdapted(latent_dim=cfg['hidden_dim'], num_layers=cfg['num_layers']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule="cosine")
    alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)

    cache = Path(args.cache_dir)
    test_files = sorted((cache / 'test').glob('*.npz'))
    results = out / 'results'
    results.mkdir(parents=True, exist_ok=True)

    window = 480
    stride = 240
    ddim_steps = args.ddim_steps

    print(f"Evaluating {len(test_files)} test sequences (DDIM {ddim_steps} steps)...")
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
                midi_t = torch.from_numpy(chunk).unsqueeze(0).to(device)

                # DDIM sampling
                x = torch.randn(1, window, OUTPUT_DIM, device=device)
                timesteps = torch.linspace(999, 0, ddim_steps, device=device).long()
                for i, t in enumerate(timesteps):
                    t_batch = t.expand(1)
                    pred = model(x, t_batch, midi_t)
                    if i < len(timesteps) - 1:
                        alpha_t = alphas_cumprod[t]
                        alpha_prev = alphas_cumprod[timesteps[i + 1]]
                        pred_noise = (x - torch.sqrt(alpha_t) * pred) / torch.sqrt(1 - alpha_t)
                        x = torch.sqrt(alpha_prev) * pred + torch.sqrt(1 - alpha_prev) * pred_noise
                    else:
                        x = pred
                gen_zs = x[0].cpu().numpy()[:wl]
                w = np.minimum(np.arange(wl) + 1, np.arange(wl, 0, -1)).astype(np.float64).reshape(-1, 1)
                pred_sum[s:e] += gen_zs * w
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
    p.add_argument('--output_dir', default='experiments/benchmark/mdm_adapted/checkpoints')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_layers', type=int, default=8)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--ddim_steps', type=int, default=25)
    args = p.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
