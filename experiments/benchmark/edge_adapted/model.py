#!/usr/bin/env python3
"""
EDGE-adapted: Tseng et al. 2023 (SIGGRAPH) "EDGE: Editable Dance Generation
from Music" adapted to MIDI-conditioned pianist motion.

Reference: github.com/Stanford-TML/EDGE
Paper: "EDGE: Editable Dance Generation From Music", SIGGRAPH 2023.

Architecture preserved from original EDGE:
    - DanceDecoder (FiLM transformer decoder with cross-attention conditioning)
    - Rotary positional embedding
    - Classifier-free guidance via null_cond_embed
    - Imports the actual DanceDecoder class from vendor/EDGE/model/model.py

Key adaptation: EDGE's Jukebox audio features (cond_feature_dim=4800) are
replaced with frame-wise MIDI features (piano_roll 88 + key_onset 88 +
onset_density 1 = 177). The DanceDecoder treats these identically to
Jukebox features — both are (batch, seq_len, feature_dim) conditioning.

This is the minimal change required to apply EDGE to symbolic music input.
The transformer decoder, cross-attention, FiLM conditioning, rotary
embeddings, and all training dynamics are otherwise unchanged.

Piano-specific: EDGE's foot-contact loss is SMPL-specific and disabled
(MOSA uses XYZ positions not SMPL rotations). We use only position L1 +
velocity L1 losses, matching the other baselines.

Output: 26-joint XYZ positions, float32, .npz format.

Usage:
    python experiments/benchmark/edge_adapted/model.py --mode=train
    python experiments/benchmark/edge_adapted/model.py --mode=eval
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
# When run as a script, Python auto-prepends this file's directory to
# sys.path[0]. That directory contains `model.py` (this file!), which would
# shadow `vendor/EDGE/model/model.py`. Remove it before doing anything else.
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
if '' in sys.path:
    sys.path.remove('')

# Order matters: src/ first (for data.constants / diffusion_lib),
# vendor/EDGE second (for model.model.DanceDecoder).
# vendor/EDGE has its own data/ and model/ subdirs that shadow ours if
# inserted first.
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from data.constants import NUM_JOINTS  # noqa: E402
from diffusion_lib.diffusion_piano import GaussianDiffusion  # noqa: E402

# Now add EDGE and import DanceDecoder. This shadows `data` and `model` in
# sys.path but only AFTER we've already imported our own modules.
sys.path.insert(0, str(ROOT / 'vendor' / 'EDGE'))
import importlib
_edge_model_pkg = importlib.import_module('model.model')
DanceDecoder = _edge_model_pkg.DanceDecoder  # noqa: E402

OUTPUT_DIM = NUM_JOINTS * 3  # 78
MIDI_FEATURE_DIM = 177        # 88 + 88 + 1


def compute_norm_stats(cache_dir: Path, split='train'):
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


class EDGEAdaptedDataset(Dataset):
    """Same dataset layout as other baselines — MIDI features + z-score motion."""
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
        midi = np.concatenate([piano_roll, key_onset, onset_density], axis=-1)

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

    train_ds = EDGEAdaptedDataset(args.cache_dir, 'train', window=args.window,
                                  mean=mean, std=std)
    val_ds = EDGEAdaptedDataset(args.cache_dir, 'val', window=args.window,
                                mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate)

    # Instantiate the EDGE DanceDecoder with MIDI conditioning dim
    model = DanceDecoder(
        nfeats=OUTPUT_DIM,
        seq_len=args.window,
        latent_dim=args.hidden_dim,
        ff_size=args.hidden_dim * 4,
        num_layers=args.num_layers,
        num_heads=4,
        dropout=0.1,
        cond_feature_dim=MIDI_FEATURE_DIM,
        use_rotary=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"EDGE DanceDecoder (MIDI-adapted) parameters: {n_params:,}")

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
            t = torch.randint(0, 1000, (B,), device=device).float()
            noise = torch.randn_like(motion)
            x_noisy = diffusion.q_sample(motion, t.long(), noise)
            # EDGE DanceDecoder signature: (x, cond_embed, times, cond_drop_prob)
            x0_pred = model(x_noisy, midi, t, cond_drop_prob=0.1)
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
                t = torch.randint(0, 1000, (B,), device=device).float()
                noise = torch.randn_like(motion)
                x_noisy = diffusion.q_sample(motion, t.long(), noise)
                x0_pred = model(x_noisy, midi, t, cond_drop_prob=0.0)
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
                'config': {'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers,
                           'window': args.window},
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
    window = cfg.get('window', 480)

    model = DanceDecoder(
        nfeats=OUTPUT_DIM, seq_len=window, latent_dim=cfg['hidden_dim'],
        ff_size=cfg['hidden_dim'] * 4, num_layers=cfg['num_layers'],
        num_heads=4, dropout=0.1, cond_feature_dim=MIDI_FEATURE_DIM, use_rotary=True,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule="cosine")
    alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)

    cache = Path(args.cache_dir)
    test_files = sorted((cache / 'test').glob('*.npz'))
    results = out / 'results'
    results.mkdir(parents=True, exist_ok=True)

    stride = window // 2
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

                x = torch.randn(1, window, OUTPUT_DIM, device=device)
                timesteps = torch.linspace(999, 0, ddim_steps, device=device).long()
                for i, t in enumerate(timesteps):
                    t_batch = t.expand(1).float()
                    pred = model(x, midi_t, t_batch, cond_drop_prob=0.0)
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
    p.add_argument('--output_dir', default='experiments/benchmark/edge_adapted/checkpoints')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--window', type=int, default=480)
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
