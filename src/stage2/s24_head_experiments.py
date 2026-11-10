#!/usr/bin/env python3
"""
Stage 2b Head Experiments: Mean offset baseline + MLP variants.

Experiments:
    1. Mean offset: C7 + per-marker fixed offset (no learning)
    2. MLP 6-dim:   C7 + CLAV → Head (current implementation)
    3. MLP 12-dim:  C7 + CLAV + T10 + STRN → Head (full torso context)

Each evaluated on both GT and predicted torso inputs.

Usage:
    python src/stage2/s24_head_experiments.py --experiment all
    python src/stage2/s24_head_experiments.py --experiment mean_offset
    python src/stage2/s24_head_experiments.py --experiment mlp_6dim
    python src/stage2/s24_head_experiments.py --experiment mlp_12dim
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

from data.constants import (
    RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX,
    C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
)

HEAD_INDICES = [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX]
HEAD_NAMES = ['RFHD', 'LFHD', 'RBHD', 'LBHD']
WINDOW, STRIDE = 480, 240


# ============================================================
# Dataset
# ============================================================

class HeadDataset(Dataset):
    def __init__(self, cache_dir, torso_pred_dir, split='train',
                 window_size=480, stride=240, use_gt=False, input_dim=6):
        self.window_size = window_size
        self.input_dim = input_dim  # 6 (C7+CLAV) or 12 (full torso)
        self.data = []
        self.windows = []

        cache_split = Path(cache_dir) / split
        torso_split = Path(torso_pred_dir) / split

        for f in tqdm(sorted(cache_split.glob('*.npz')), desc=f'Loading {split}', leave=False):
            data = np.load(f)
            joints = data['joints'].astype(np.float32)
            T = len(joints)

            head_gt = joints[:, HEAD_INDICES, :].reshape(T, 12)

            if input_dim == 6:
                gt_input = joints[:, [C7_IDX, CLAV_IDX], :].reshape(T, 6)
            else:
                gt_input = joints[:, [C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX], :].reshape(T, 12)

            if use_gt:
                torso_input = gt_input
            else:
                tp = torso_split / f.name
                if tp.exists():
                    td = np.load(tp)
                    torso_pred = td.get('torso_pred')
                    if torso_pred is not None:
                        torso_pred = torso_pred.astype(np.float32)
                        if input_dim == 6:
                            torso_input = torso_pred[:, 0:6]  # C7+CLAV
                        else:
                            torso_input = torso_pred[:, 0:12]  # full torso
                    else:
                        torso_input = gt_input
                else:
                    torso_input = gt_input

            self.data.append({'head_gt': head_gt, 'input': torso_input, 'T': T})

        for si, s in enumerate(self.data):
            for start in range(0, s['T'] - window_size + 1, stride):
                self.windows.append((si, start, start + window_size))

        print(f'[{split}] {len(self.windows)} windows from {len(self.data)} samples')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        si, s, e = self.windows[idx]
        d = self.data[si]
        return {
            'head_gt': torch.from_numpy(d['head_gt'][s:e].copy()).float(),
            'input': torch.from_numpy(d['input'][s:e].copy()).float(),
        }


# ============================================================
# Models
# ============================================================

class HeadMLP(nn.Module):
    """Temporal MLP for head prediction."""
    def __init__(self, input_dim=6, output_dim=12, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Mean Offset Baseline
# ============================================================

def compute_mean_offset(cache_dir, split='train'):
    """Compute per-marker mean offset from C7."""
    cache_split = Path(cache_dir) / split
    all_offsets = []
    for f in sorted(cache_split.glob('*.npz')):
        data = np.load(f)
        joints = data['joints'].astype(np.float32)
        c7 = joints[:, C7_IDX]  # (T, 3)
        head = joints[:, HEAD_INDICES, :]  # (T, 4, 3)
        # offset: each head marker relative to C7
        offset = head - c7[:, np.newaxis, :]  # (T, 4, 3)
        all_offsets.append(offset.reshape(-1, 12))
    all_offsets = np.concatenate(all_offsets, axis=0)
    return all_offsets.mean(axis=0)  # (12,)


def eval_mean_offset(cache_dir, torso_pred_dir, mean_offset, use_gt=False):
    """Evaluate mean offset baseline on test set."""
    cache_test = Path(cache_dir) / 'test'
    torso_test = Path(torso_pred_dir) / 'test'
    offset = mean_offset.reshape(4, 3)

    all_errors = []
    for f in sorted(cache_test.glob('*.npz')):
        data = np.load(f)
        joints = data['joints'].astype(np.float32)
        T = len(joints)
        head_gt = joints[:, HEAD_INDICES, :]  # (T, 4, 3)

        if use_gt:
            c7 = joints[:, C7_IDX]
        else:
            tp = torso_test / f.name
            if tp.exists():
                td = np.load(tp)
                tp_data = td.get('torso_pred')
                if tp_data is not None:
                    c7 = tp_data.astype(np.float32)[:, 0:3]  # C7
                else:
                    c7 = joints[:, C7_IDX]
            else:
                c7 = joints[:, C7_IDX]

        pred = c7[:, np.newaxis, :] + offset[np.newaxis, :, :]  # (T, 4, 3)
        err = np.sqrt(((head_gt - pred) ** 2).sum(axis=-1))  # (T, 4)
        all_errors.append(err)

    ae = np.concatenate(all_errors, axis=0)
    return {
        'mpjpe': ae.mean() * 1000,
        **{HEAD_NAMES[i].lower(): ae[:, i].mean() * 1000 for i in range(4)},
    }


# ============================================================
# MLP Training + Eval
# ============================================================

def train_mlp(cache_dir, torso_pred_dir, output_dir, input_dim=6,
              hidden_dim=128, num_layers=3, lr=1e-4, epochs=500, patience=50,
              batch_size=32, device='cuda', wandb_project=None):
    """Train Head MLP and return best checkpoint path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = HeadDataset(cache_dir, torso_pred_dir, 'train', WINDOW, STRIDE,
                           use_gt=False, input_dim=input_dim)
    val_ds = HeadDataset(cache_dir, torso_pred_dir, 'val', WINDOW, STRIDE,
                         use_gt=False, input_dim=input_dim)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = HeadMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    print(f'HeadMLP (input={input_dim}): {sum(p.numel() for p in model.parameters()):,} params')

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)

    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, name=f'head_mlp_{input_dim}dim',
                   config={'input_dim': input_dim, 'hidden_dim': hidden_dim,
                           'num_layers': num_layers, 'lr': lr})

    best_val = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc=f'Head MLP {input_dim}d', unit='ep'):
        model.train()
        train_loss = 0
        n = 0
        for batch in train_loader:
            gt = batch['head_gt'].to(device)
            inp = batch['input'].to(device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = F.mse_loss(pred, gt)
            vel_loss = F.mse_loss(pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1])
            total = loss + 0.5 * vel_loss
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += total.item()
            n += 1
        train_loss /= n

        model.eval()
        val_loss = 0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                gt = batch['head_gt'].to(device)
                inp = batch['input'].to(device)
                pred = model(inp)
                loss = F.mse_loss(pred, gt)
                vel_loss = F.mse_loss(pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1])
                val_loss += (loss + 0.5 * vel_loss).item()
                n += 1
        val_loss /= n

        scheduler.step(val_loss)
        if use_wandb:
            import wandb
            wandb.log({'train': train_loss, 'val': val_loss, 'epoch': epoch + 1})

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': best_val,
                'config': {'input_dim': input_dim, 'hidden_dim': hidden_dim,
                           'num_layers': num_layers},
            }, output_dir / 'best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if use_wandb:
        import wandb
        wandb.finish()

    print(f'Best val loss: {best_val:.6f}')
    return output_dir / 'best.pt'


def eval_mlp(ckpt_path, cache_dir, torso_pred_dir, device='cuda', use_gt=False):
    """Evaluate Head MLP checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    input_dim = config['input_dim']

    model = HeadMLP(input_dim=input_dim, hidden_dim=config['hidden_dim'],
                    num_layers=config['num_layers']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    cache_test = Path(cache_dir) / 'test'
    torso_test = Path(torso_pred_dir) / 'test'

    all_errors = []
    for f in sorted(cache_test.glob('*.npz')):
        data = np.load(f)
        joints = data['joints'].astype(np.float32)
        T = len(joints)
        head_gt = joints[:, HEAD_INDICES, :].reshape(T, 12)

        if use_gt:
            if input_dim == 6:
                inp = joints[:, [C7_IDX, CLAV_IDX], :].reshape(T, 6)
            else:
                inp = joints[:, [C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX], :].reshape(T, 12)
        else:
            tp = torso_test / f.name
            if tp.exists():
                td = np.load(tp)
                tp_data = td.get('torso_pred')
                if tp_data is not None:
                    tp_data = tp_data.astype(np.float32)
                    inp = tp_data[:, :input_dim]
                else:
                    if input_dim == 6:
                        inp = joints[:, [C7_IDX, CLAV_IDX], :].reshape(T, 6)
                    else:
                        inp = joints[:, [C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX], :].reshape(T, 12)
            else:
                if input_dim == 6:
                    inp = joints[:, [C7_IDX, CLAV_IDX], :].reshape(T, 6)
                else:
                    inp = joints[:, [C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX], :].reshape(T, 12)

        # Windowed inference
        pred_sum = np.zeros((T, 12), dtype=np.float64)
        weight_sum = np.zeros((T, 1), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, max(1, T - WINDOW + 1), STRIDE):
                e = min(s + WINDOW, T)
                chunk = inp[s:e]
                if len(chunk) < WINDOW:
                    chunk = np.concatenate([chunk, np.zeros((WINDOW - len(chunk), input_dim), dtype=np.float32)])
                ct = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
                pred = model(ct).cpu().numpy()[0]
                al = e - s
                w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
                pred_sum[s:e] += pred[:al] * w
                weight_sum[s:e] += w
        mask = weight_sum > 0
        pred_final = np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)

        gt_r = head_gt.reshape(T, 4, 3)
        pred_r = pred_final.reshape(T, 4, 3)
        err = np.sqrt(((gt_r - pred_r) ** 2).sum(axis=-1))
        all_errors.append(err)

    ae = np.concatenate(all_errors, axis=0)
    del model
    torch.cuda.empty_cache()

    return {
        'mpjpe': ae.mean() * 1000,
        **{HEAD_NAMES[i].lower(): ae[:, i].mean() * 1000 for i in range(4)},
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 2b Head Experiments')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'mean_offset', 'mlp_6dim', 'mlp_12dim'])
    parser.add_argument('--cache_dir', default='MOSA_dataset/training_cache_W10')
    parser.add_argument('--torso_pred_dir', default='MOSA_dataset/phase1_torso_predictions')
    parser.add_argument('--output_dir', default='checkpoints/stage2_head_experiments')
    parser.add_argument('--output', default='results/stage2b_head_eval.json')
    parser.add_argument('--wandb_project', default='pianoid-stage2')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    results = {}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    experiments = [args.experiment] if args.experiment != 'all' else ['mean_offset', 'mlp_6dim', 'mlp_12dim']

    for exp in experiments:
        print(f'\n{"="*60}')
        print(f'Experiment: {exp}')
        print(f'{"="*60}')

        if exp == 'mean_offset':
            print('Computing mean offset from training set...')
            offset = compute_mean_offset(args.cache_dir, 'train')
            print(f'Mean offset: {offset}')

            for input_type, use_gt in [('predicted', False), ('gt', True)]:
                r = eval_mean_offset(args.cache_dir, args.torso_pred_dir, offset, use_gt=use_gt)
                key = f'mean_offset_{input_type}'
                results[key] = r
                print(f'  {key}: MPJPE={r["mpjpe"]:.2f}mm')

        elif exp == 'mlp_6dim':
            ckpt = train_mlp(
                args.cache_dir, args.torso_pred_dir,
                f'{args.output_dir}/mlp_6dim', input_dim=6,
                epochs=args.epochs, patience=args.patience,
                device=device, wandb_project=args.wandb_project)

            for input_type, use_gt in [('predicted', False), ('gt', True)]:
                r = eval_mlp(ckpt, args.cache_dir, args.torso_pred_dir, device, use_gt=use_gt)
                key = f'mlp_6dim_{input_type}'
                results[key] = r
                print(f'  {key}: MPJPE={r["mpjpe"]:.2f}mm')

        elif exp == 'mlp_12dim':
            ckpt = train_mlp(
                args.cache_dir, args.torso_pred_dir,
                f'{args.output_dir}/mlp_12dim', input_dim=12,
                epochs=args.epochs, patience=args.patience,
                device=device, wandb_project=args.wandb_project)

            for input_type, use_gt in [('predicted', False), ('gt', True)]:
                r = eval_mlp(ckpt, args.cache_dir, args.torso_pred_dir, device, use_gt=use_gt)
                key = f'mlp_12dim_{input_type}'
                results[key] = r
                print(f'  {key}: MPJPE={r["mpjpe"]:.2f}mm')

    # Print summary
    print(f'\n{"="*60}')
    print('Stage 2b Head Experiments Summary')
    print(f'{"="*60}')
    print(f'{"Model":<25} {"Input":<12} {"MPJPE":<10} {"RFHD":<8} {"LFHD":<8} {"RBHD":<8} {"LBHD":<8}')
    print('-' * 79)
    for key, r in sorted(results.items()):
        print(f'{key:<25} {"GT" if "gt" in key else "pred":<12} '
              f'{r["mpjpe"]:<10.2f} {r["rfhd"]:<8.2f} {r["lfhd"]:<8.2f} '
              f'{r["rbhd"]:<8.2f} {r["lbhd"]:<8.2f}')

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
