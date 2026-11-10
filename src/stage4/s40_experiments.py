#!/usr/bin/env python3
"""
Stage 4 Lower Body Experiments.

Experiments:
    1. mean_offset:  Per-joint mean offset from spine centroid (no learning)
    2. film_spine:   FiLM h128 l4, spine only (6-dim)
    3. film_od:      FiLM h128 l4, spine + onset density (7-dim)
    4. film_aria:    FiLM h128 l4, spine + ARIA (6+FiLM)
    5. bilstm_od:    BiLSTM h256 l4, spine + onset density (7-dim)

Each evaluated on both GT and predicted spine inputs.

Usage:
    python src/stage4/s40_experiments.py --experiment all
    python src/stage4/s40_experiments.py --experiment mean_offset
    python src/stage4/s40_experiments.py --experiment film_spine
"""

import argparse
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from data.constants import C7_IDX, T10_IDX, LOWER_BODY_INDICES

LOWER_INDICES = LOWER_BODY_INDICES  # 10 joints
LOWER_NAMES = ['RASI', 'LASI', 'RPSI', 'LPSI', 'RKNE', 'RANKLE', 'RTOE', 'LKNE', 'LANKLE', 'LTOE']
PELVIS_LOCAL = [0, 1, 2, 3]
LEGS_LOCAL = [4, 5, 6, 7, 8, 9]
WINDOW, STRIDE = 480, 240


# ============================================================
# Dataset
# ============================================================

class LowerDataset(Dataset):
    def __init__(self, cache_dir, torso_pred_dir, split='train',
                 window_size=480, stride=240, use_gt=False,
                 load_onset_density=False, load_aria=False):
        self.window_size = window_size
        self.load_od = load_onset_density
        self.load_aria = load_aria
        self.data = []
        self.windows = []

        cache_split = Path(cache_dir) / split
        torso_split = Path(torso_pred_dir) / split

        for f in tqdm(sorted(cache_split.glob('*.npz')), desc=f'Loading {split}', leave=False):
            data = np.load(f)
            joints = data['joints'].astype(np.float32)
            T = len(joints)

            lower_gt = joints[:, LOWER_INDICES, :].reshape(T, 30)
            gt_spine = np.concatenate([joints[:, C7_IDX], joints[:, T10_IDX]], axis=-1).astype(np.float32)

            if use_gt:
                spine = gt_spine
            else:
                tp = torso_split / f.name
                if tp.exists():
                    td = np.load(tp)
                    tp_data = td.get('torso_pred')
                    if tp_data is not None:
                        tp_data = tp_data.astype(np.float32)
                        spine = np.concatenate([tp_data[:, 0:3], tp_data[:, 6:9]], axis=-1)
                    else:
                        spine = gt_spine
                else:
                    spine = gt_spine

            sample = {'lower_gt': lower_gt, 'spine': spine, 'T': T}

            if self.load_od:
                sample['onset_density'] = data.get('onset_density', np.zeros(T)).astype(np.float32)
            if self.load_aria:
                aria = data.get('aria_embed', np.zeros(512)).astype(np.float32)
                if aria.ndim == 2:
                    aria = aria.mean(axis=0)
                sample['aria'] = aria

            self.data.append(sample)

        for si, s in enumerate(self.data):
            for start in range(0, s['T'] - window_size + 1, stride):
                self.windows.append((si, start, start + window_size))

        print(f'[{split}] {len(self.windows)} windows from {len(self.data)} samples')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        si, s, e = self.windows[idx]
        d = self.data[si]
        result = {
            'lower_gt': torch.from_numpy(d['lower_gt'][s:e].copy()).float(),
            'spine': torch.from_numpy(d['spine'][s:e].copy()).float(),
        }
        if self.load_od and 'onset_density' in d:
            result['od'] = torch.tensor(d['onset_density'][s:e], dtype=torch.float32).unsqueeze(-1)
        if self.load_aria and 'aria' in d:
            result['aria'] = torch.from_numpy(d['aria'].copy()).float()
        return result


# ============================================================
# Models
# ============================================================

class LowerFiLM(nn.Module):
    """FiLM-modulated MLP for lower body, with optional ARIA."""
    def __init__(self, input_dim=6, output_dim=30, hidden_dim=128, num_layers=4,
                 context_frames=5, use_aria=False):
        super().__init__()
        self.use_aria = use_aria
        self.context_frames = context_frames
        ctx_dim = input_dim * (2 * context_frames + 1)

        layers = [nn.Linear(ctx_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        if use_aria:
            self.aria_film = nn.Linear(512, hidden_dim * 2)

    def _build_context(self, x):
        B, T, D = x.shape
        k = self.context_frames
        padded = F.pad(x, (0, 0, k, k), mode='replicate')
        chunks = [padded[:, i:i+T, :] for i in range(2 * k + 1)]
        return torch.cat(chunks, dim=-1)

    def forward(self, spine, od=None, aria=None):
        parts = [spine]
        if od is not None:
            parts.append(od)
        x = torch.cat(parts, dim=-1)
        x = self._build_context(x)

        if self.use_aria and aria is not None:
            h = self.net[0](x)
            h = self.net[1](h)
            film = self.aria_film(aria)
            scale, shift = film.chunk(2, dim=-1)
            h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            for layer in self.net[2:]:
                h = layer(h)
            return h
        else:
            return self.net(x)


class LowerBiLSTM(nn.Module):
    def __init__(self, input_dim=6, output_dim=30, hidden_dim=256, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=0.1 if num_layers > 1 else 0)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim, output_dim))

    def forward(self, spine, od=None, aria=None):
        parts = [spine]
        if od is not None:
            parts.append(od)
        x = torch.cat(parts, dim=-1)
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        return self.output_proj(h)


# ============================================================
# Loss
# ============================================================

class LowerLoss(nn.Module):
    def forward(self, pred, target):
        pos_loss = F.mse_loss(pred, target)
        vel_pred = pred[:, 1:] - pred[:, :-1]
        vel_gt = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(vel_pred, vel_gt)
        total = pos_loss + 0.5 * vel_loss
        return total, {'total': total.item(), 'pos': pos_loss.item(), 'vel': vel_loss.item()}


# ============================================================
# Mean Offset Baseline
# ============================================================

def compute_mean_offset(cache_dir, split='train'):
    """Per-joint mean offset from spine centroid."""
    cache_split = Path(cache_dir) / split
    all_offsets = []
    for f in sorted(cache_split.glob('*.npz')):
        data = np.load(f)
        joints = data['joints'].astype(np.float32)
        spine_center = (joints[:, C7_IDX] + joints[:, T10_IDX]) / 2  # (T, 3)
        lower = joints[:, LOWER_INDICES, :]  # (T, 10, 3)
        offset = lower - spine_center[:, np.newaxis, :]  # (T, 10, 3)
        all_offsets.append(offset.reshape(-1, 30))
    return np.concatenate(all_offsets, axis=0).mean(axis=0)


def eval_mean_offset(cache_dir, torso_pred_dir, offset, use_gt=False):
    cache_test = Path(cache_dir) / 'test'
    torso_test = Path(torso_pred_dir) / 'test'
    offset_r = offset.reshape(10, 3)

    all_errors = []
    for f in sorted(cache_test.glob('*.npz')):
        data = np.load(f)
        joints = data['joints'].astype(np.float32)
        T = len(joints)
        lower_gt = joints[:, LOWER_INDICES, :]

        if use_gt:
            spine_center = (joints[:, C7_IDX] + joints[:, T10_IDX]) / 2
        else:
            tp = torso_test / f.name
            if tp.exists():
                td = np.load(tp)
                tp_data = td.get('torso_pred')
                if tp_data is not None:
                    tp_data = tp_data.astype(np.float32)
                    spine_center = (tp_data[:, 0:3] + tp_data[:, 6:9]) / 2
                else:
                    spine_center = (joints[:, C7_IDX] + joints[:, T10_IDX]) / 2
            else:
                spine_center = (joints[:, C7_IDX] + joints[:, T10_IDX]) / 2

        pred = spine_center[:, np.newaxis, :] + offset_r[np.newaxis, :, :]
        err = np.sqrt(((lower_gt - pred) ** 2).sum(axis=-1))
        all_errors.append(err)

    ae = np.concatenate(all_errors, axis=0)
    pelvis = ae[:, PELVIS_LOCAL].mean() * 1000
    legs = ae[:, LEGS_LOCAL].mean() * 1000
    return {'mpjpe': ae.mean() * 1000, 'pelvis': pelvis, 'legs': legs,
            **{LOWER_NAMES[i].lower(): ae[:, i].mean() * 1000 for i in range(10)}}


# ============================================================
# Training + Eval
# ============================================================

def train_model(model, cache_dir, torso_pred_dir, output_dir,
                load_od=False, load_aria=False,
                epochs=500, patience=50, batch_size=16, lr=1e-4,
                device='cuda', wandb_project=None, run_name='lower'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LowerDataset(cache_dir, torso_pred_dir, 'train', WINDOW, STRIDE,
                            use_gt=False, load_onset_density=load_od, load_aria=load_aria)
    val_ds = LowerDataset(cache_dir, torso_pred_dir, 'val', WINDOW, STRIDE,
                          use_gt=False, load_onset_density=load_od, load_aria=load_aria)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = model.to(device)
    print(f'{run_name}: {sum(p.numel() for p in model.parameters()):,} params')

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)
    criterion = LowerLoss()

    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, name=run_name, config={'run_name': run_name, 'lr': lr})

    best_val = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc=run_name, unit='ep'):
        model.train()
        train_loss = 0
        n = 0
        for batch in train_loader:
            gt = batch['lower_gt'].to(device)
            spine = batch['spine'].to(device)
            od = batch['od'].to(device) if load_od and 'od' in batch else None
            aria = batch['aria'].to(device) if load_aria and 'aria' in batch else None

            optimizer.zero_grad()
            pred = model(spine, od=od, aria=aria)
            loss, _ = criterion(pred, gt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n += 1
        train_loss /= n

        model.eval()
        val_loss = 0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                gt = batch['lower_gt'].to(device)
                spine = batch['spine'].to(device)
                od = batch['od'].to(device) if load_od and 'od' in batch else None
                aria = batch['aria'].to(device) if load_aria and 'aria' in batch else None
                pred = model(spine, od=od, aria=aria)
                loss, _ = criterion(pred, gt)
                val_loss += loss.item()
                n += 1
        val_loss /= n

        scheduler.step(val_loss)
        if use_wandb:
            import wandb
            wandb.log({'train': train_loss, 'val': val_loss, 'epoch': epoch + 1})

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': best_val, 'config': {'run_name': run_name}},
                       output_dir / 'best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if use_wandb:
        import wandb
        wandb.finish()
    print(f'{run_name}: best val={best_val:.6f}')
    return output_dir / 'best.pt'


def eval_model(model, cache_dir, torso_pred_dir, device, load_od=False, load_aria=False, use_gt=False):
    cache_test = Path(cache_dir) / 'test'
    torso_test = Path(torso_pred_dir) / 'test'

    all_errors = []
    for f in sorted(cache_test.glob('*.npz')):
        data = np.load(f)
        joints = data['joints'].astype(np.float32)
        T = len(joints)
        lower_gt = joints[:, LOWER_INDICES, :].reshape(T, 30)

        if use_gt:
            spine = np.concatenate([joints[:, C7_IDX], joints[:, T10_IDX]], axis=-1)
        else:
            tp = torso_test / f.name
            if tp.exists():
                td = np.load(tp)
                tp_data = td.get('torso_pred')
                if tp_data is not None:
                    tp_data = tp_data.astype(np.float32)
                    spine = np.concatenate([tp_data[:, 0:3], tp_data[:, 6:9]], axis=-1)
                else:
                    spine = np.concatenate([joints[:, C7_IDX], joints[:, T10_IDX]], axis=-1)
            else:
                spine = np.concatenate([joints[:, C7_IDX], joints[:, T10_IDX]], axis=-1)

        od_np = None
        if load_od:
            od_np = data.get('onset_density', np.zeros(T)).astype(np.float32).reshape(T, 1)

        aria_np = None
        if load_aria:
            aria_np = data.get('aria_embed', np.zeros(512)).astype(np.float32)
            if aria_np.ndim == 2:
                aria_np = aria_np.mean(axis=0)

        # Windowed inference
        input_dim = spine.shape[-1] + (1 if load_od else 0)
        pred_sum = np.zeros((T, 30), dtype=np.float64)
        weight_sum = np.zeros((T, 1), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, max(1, T - WINDOW + 1), STRIDE):
                e = min(s + WINDOW, T)
                al = e - s

                sp = spine[s:e]
                if len(sp) < WINDOW:
                    sp = np.concatenate([sp, np.zeros((WINDOW - len(sp), 6), dtype=np.float32)])
                sp_t = torch.tensor(sp, dtype=torch.float32).unsqueeze(0).to(device)

                od_t = None
                if od_np is not None:
                    od_chunk = od_np[s:e]
                    if len(od_chunk) < WINDOW:
                        od_chunk = np.concatenate([od_chunk, np.zeros((WINDOW - len(od_chunk), 1), dtype=np.float32)])
                    od_t = torch.tensor(od_chunk, dtype=torch.float32).unsqueeze(0).to(device)

                aria_t = None
                if aria_np is not None:
                    aria_t = torch.tensor(aria_np, dtype=torch.float32).unsqueeze(0).to(device)

                pred = model(sp_t, od=od_t, aria=aria_t).cpu().numpy()[0]
                w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
                pred_sum[s:e] += pred[:al] * w
                weight_sum[s:e] += w

        mask = weight_sum > 0
        pred_final = np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)

        gt_r = lower_gt.reshape(T, 10, 3)
        pred_r = pred_final.reshape(T, 10, 3)
        err = np.sqrt(((gt_r - pred_r) ** 2).sum(axis=-1))
        all_errors.append(err)

    ae = np.concatenate(all_errors, axis=0)
    pelvis = ae[:, PELVIS_LOCAL].mean() * 1000
    legs = ae[:, LEGS_LOCAL].mean() * 1000
    return {'mpjpe': ae.mean() * 1000, 'pelvis': pelvis, 'legs': legs,
            **{LOWER_NAMES[i].lower(): ae[:, i].mean() * 1000 for i in range(10)}}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 4 Lower Body Experiments')
    parser.add_argument('--experiment', default='all',
                        choices=['all', 'mean_offset', 'film_spine', 'film_od', 'film_aria', 'bilstm_od',
                                 'scale_h64', 'scale_h256', 'scale_l2', 'scale_l6', 'scaling'])
    parser.add_argument('--cache_dir', default='MOSA_dataset/training_cache_W10')
    parser.add_argument('--torso_pred_dir', default='MOSA_dataset/phase1_torso_predictions')
    parser.add_argument('--output_dir', default='checkpoints/stage4_experiments')
    parser.add_argument('--output', default='results/stage4_eval.json')
    parser.add_argument('--wandb_project', default='pianoid-stage4')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default 42)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    sys.path.insert(0, str(ROOT / 'src'))
    from project_utils.seeding import fix_seed
    fix_seed(args.seed)
    print(f"Seed: {args.seed} (cudnn deterministic on)")
    results = {}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    ALL_EXPERIMENTS = ['mean_offset', 'film_spine', 'film_od', 'film_aria', 'bilstm_od', 'scaling']
    experiments = [args.experiment] if args.experiment != 'all' else ALL_EXPERIMENTS

    for exp in experiments:
        print(f'\n{"="*60}')
        print(f'Experiment: {exp}')
        print(f'{"="*60}')

        if exp == 'mean_offset':
            offset = compute_mean_offset(args.cache_dir, 'train')
            for tag, use_gt in [('predicted', False), ('gt', True)]:
                r = eval_mean_offset(args.cache_dir, args.torso_pred_dir, offset, use_gt)
                results[f'mean_offset_{tag}'] = r
                print(f'  mean_offset_{tag}: MPJPE={r["mpjpe"]:.2f}mm (pelvis={r["pelvis"]:.1f}, legs={r["legs"]:.1f})')

        elif exp == 'film_spine':
            model = LowerFiLM(input_dim=6, hidden_dim=128, num_layers=4)
            ckpt = train_model(model, args.cache_dir, args.torso_pred_dir,
                               f'{args.output_dir}/film_spine', epochs=args.epochs,
                               patience=args.patience, device=device,
                               wandb_project=args.wandb_project, run_name='film_spine')
            model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict'])
            model.to(device).eval()
            for tag, use_gt in [('predicted', False), ('gt', True)]:
                r = eval_model(model, args.cache_dir, args.torso_pred_dir, device, use_gt=use_gt)
                results[f'film_spine_{tag}'] = r
                print(f'  film_spine_{tag}: MPJPE={r["mpjpe"]:.2f}mm (pelvis={r["pelvis"]:.1f}, legs={r["legs"]:.1f})')

        elif exp == 'film_od':
            model = LowerFiLM(input_dim=7, hidden_dim=128, num_layers=4)
            ckpt = train_model(model, args.cache_dir, args.torso_pred_dir,
                               f'{args.output_dir}/film_od', load_od=True,
                               epochs=args.epochs, patience=args.patience, device=device,
                               wandb_project=args.wandb_project, run_name='film_od')
            model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict'])
            model.to(device).eval()
            for tag, use_gt in [('predicted', False), ('gt', True)]:
                r = eval_model(model, args.cache_dir, args.torso_pred_dir, device, load_od=True, use_gt=use_gt)
                results[f'film_od_{tag}'] = r
                print(f'  film_od_{tag}: MPJPE={r["mpjpe"]:.2f}mm (pelvis={r["pelvis"]:.1f}, legs={r["legs"]:.1f})')

        elif exp == 'film_aria':
            model = LowerFiLM(input_dim=6, hidden_dim=128, num_layers=4, use_aria=True)
            ckpt = train_model(model, args.cache_dir, args.torso_pred_dir,
                               f'{args.output_dir}/film_aria', load_aria=True,
                               epochs=args.epochs, patience=args.patience, device=device,
                               wandb_project=args.wandb_project, run_name='film_aria')
            model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict'])
            model.to(device).eval()
            for tag, use_gt in [('predicted', False), ('gt', True)]:
                r = eval_model(model, args.cache_dir, args.torso_pred_dir, device, load_aria=True, use_gt=use_gt)
                results[f'film_aria_{tag}'] = r
                print(f'  film_aria_{tag}: MPJPE={r["mpjpe"]:.2f}mm (pelvis={r["pelvis"]:.1f}, legs={r["legs"]:.1f})')

        elif exp == 'bilstm_od':
            model = LowerBiLSTM(input_dim=7, hidden_dim=256, num_layers=4)
            ckpt = train_model(model, args.cache_dir, args.torso_pred_dir,
                               f'{args.output_dir}/bilstm_od', load_od=True,
                               epochs=args.epochs, patience=args.patience, device=device,
                               wandb_project=args.wandb_project, run_name='bilstm_od')
            model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict'])
            model.to(device).eval()
            for tag, use_gt in [('predicted', False), ('gt', True)]:
                r = eval_model(model, args.cache_dir, args.torso_pred_dir, device, load_od=True, use_gt=use_gt)
                results[f'bilstm_od_{tag}'] = r
                print(f'  bilstm_od_{tag}: MPJPE={r["mpjpe"]:.2f}mm (pelvis={r["pelvis"]:.1f}, legs={r["legs"]:.1f})')

        elif exp in ('scale_h64', 'scale_h256', 'scale_l2', 'scale_l6', 'scaling'):
            # Scaling spot-checks: vary hidden_dim and num_layers around FiLM h128 l4
            if exp == 'scaling':
                scale_configs = [
                    ('h64_l4', 64, 4), ('h128_l4', 128, 4), ('h256_l4', 256, 4),
                    ('h128_l2', 128, 2), ('h128_l6', 128, 6),
                ]
            else:
                scale_map = {
                    'scale_h64': ('h64_l4', 64, 4),
                    'scale_h256': ('h256_l4', 256, 4),
                    'scale_l2': ('h128_l2', 128, 2),
                    'scale_l6': ('h128_l6', 128, 6),
                }
                scale_configs = [scale_map[exp]]

            for name, hdim, nlayers in scale_configs:
                run = f'scale_{name}'
                print(f'\n--- {run} ---')
                model = LowerFiLM(input_dim=6, hidden_dim=hdim, num_layers=nlayers)
                ckpt = train_model(model, args.cache_dir, args.torso_pred_dir,
                                   f'{args.output_dir}/{run}',
                                   epochs=args.epochs, patience=args.patience, device=device,
                                   wandb_project=args.wandb_project, run_name=run)
                model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict'])
                model.to(device).eval()
                for tag, use_gt in [('predicted', False), ('gt', True)]:
                    r = eval_model(model, args.cache_dir, args.torso_pred_dir, device, use_gt=use_gt)
                    results[f'{run}_{tag}'] = r
                    print(f'  {run}_{tag}: MPJPE={r["mpjpe"]:.2f}mm')

    # Summary
    print(f'\n{"="*60}')
    print('Stage 4 Lower Body Summary')
    print(f'{"="*60}')
    print(f'{"Model":<25} {"Input":<8} {"MPJPE":<8} {"Pelvis":<8} {"Legs":<8}')
    print('-' * 57)
    for key, r in sorted(results.items()):
        tag = 'GT' if '_gt' in key else 'pred'
        print(f'{key:<25} {tag:<8} {r["mpjpe"]:<8.1f} {r["pelvis"]:<8.1f} {r["legs"]:<8.1f}')

    # Save JSON
    out_path = Path(args.output)
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.pop('_best', None)
    existing.update({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()})
    pred_results = {k: v for k, v in existing.items() if 'predicted' in k and not k.startswith('_')}
    if pred_results:
        existing['_best'] = min(pred_results, key=lambda k: pred_results[k]['mpjpe'])
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
