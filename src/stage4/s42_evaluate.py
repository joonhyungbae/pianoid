#!/usr/bin/env python3
"""
Stage 4 Evaluation: Compute test MPJPE for lower body checkpoints.

Usage:
    python src/stage4/s42_evaluate.py --checkpoint checkpoints/stage4/lower_best.pt
    python src/stage4/s42_evaluate.py --pattern "checkpoints/stage4_experiments/phaseC_*.pt"
"""

import argparse
import glob
import json
import torch
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import (
    C7_IDX, T10_IDX,
    RASI_IDX, LASI_IDX, RPSI_IDX, LPSI_IDX,
    RKNE_IDX, RANKLE_IDX, RTOE_IDX,
    LKNE_IDX, LANKLE_IDX, LTOE_IDX,
)

WINDOW, STRIDE = 480, 240
LOWER_INDICES = [RASI_IDX, LASI_IDX, RPSI_IDX, LPSI_IDX,
                 RKNE_IDX, RANKLE_IDX, RTOE_IDX,
                 LKNE_IDX, LANKLE_IDX, LTOE_IDX]
LOWER_NAMES = ['RASI', 'LASI', 'RPSI', 'LPSI', 'RKNE', 'RANKLE', 'RTOE', 'LKNE', 'LANKLE', 'LTOE']


def load_test_data(cache_dir, torso_pred_dir):
    """Load test data: GT lower body + spine conditioning from torso predictions."""
    test_data = []
    cache_test = Path(cache_dir) / 'test'
    torso_test = Path(torso_pred_dir) / 'test'

    for f in sorted(cache_test.glob('*.npz')):
        data = np.load(f)
        joints = data['joints'].astype(np.float32)
        T = len(joints)

        # GT lower body
        lower_gt = joints[:, LOWER_INDICES, :].reshape(T, -1)  # (T, 30)

        # Spine conditioning from torso predictions: C7 + T10
        tp = torso_test / f.name
        if tp.exists():
            td = np.load(tp)
            torso_pred = td.get('torso_pred')
            if torso_pred is not None:
                torso_pred = torso_pred.astype(np.float32)
                spine = np.concatenate([torso_pred[:, 0:3], torso_pred[:, 6:9]], axis=-1)  # C7 + T10
            else:
                spine = np.concatenate([joints[:, C7_IDX], joints[:, T10_IDX]], axis=-1).astype(np.float32)
        else:
            spine = np.concatenate([joints[:, C7_IDX], joints[:, T10_IDX]], axis=-1).astype(np.float32)

        # ARIA
        aria = data.get('aria_embed', np.zeros(512)).astype(np.float32)
        if aria.ndim == 2:
            aria = aria.mean(axis=0)

        test_data.append((lower_gt, spine, T, aria))

    return test_data


def windowed_inference(model, spine_np, T, device, aria=None):
    pred_sum = np.zeros((T, 30), dtype=np.float64)
    weight_sum = np.zeros((T, 1), dtype=np.float64)

    with torch.no_grad():
        for s in range(0, max(1, T - WINDOW + 1), STRIDE):
            e = min(s + WINDOW, T)
            al = e - s

            sp = spine_np[s:e]
            if len(sp) < WINDOW:
                sp = np.concatenate([sp, np.zeros((WINDOW - len(sp), 6), dtype=np.float32)])
            sp_t = torch.tensor(sp, dtype=torch.float32).unsqueeze(0).to(device)

            aria_t = None
            if aria is not None:
                aria_t = torch.tensor(aria, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(sp_t, aria=aria_t).cpu().numpy()[0]

            w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
            pred_sum[s:e] += pred[:al] * w
            weight_sum[s:e] += w

    mask = weight_sum > 0
    return np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)


def evaluate_checkpoint(ckpt_path, test_data, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})

    # Build model (try to auto-detect)
    # Import the lower body model
    try:
        from models.lower_body import HandConditionedLowerGenerator
        model = HandConditionedLowerGenerator(
            hand_dim=config.get('hand_dim', 6),
            hidden_dim=config.get('hidden_dim', config.get('latent_dim', 256)),
            num_layers=config.get('num_layers', 4),
            use_aria=config.get('use_aria', False),
        ).to(device)
    except ImportError:
        print("Warning: Could not import lower body model. Skipping.")
        return None

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_errors = []
    for lower_gt, spine, T, aria in tqdm(test_data, desc='Samples', leave=False):
        pred = windowed_inference(model, spine, T, device,
                                  aria=aria if config.get('use_aria', False) else None)
        gt_r = lower_gt.reshape(T, 10, 3)
        pred_r = pred.reshape(T, 10, 3)
        err = np.sqrt(((gt_r - pred_r) ** 2).sum(axis=-1))  # (T, 10)
        all_errors.append(err)

    ae = np.concatenate(all_errors, axis=0)
    del model
    torch.cuda.empty_cache()

    result = {
        'mpjpe': ae.mean() * 1000,
        'val_loss': ckpt.get('val_loss', float('nan')),
        'epoch': ckpt.get('epoch', -1),
    }
    # Per-joint
    for i, name in enumerate(LOWER_NAMES):
        result[name.lower()] = ae[:, i].mean() * 1000

    return result


def main():
    parser = argparse.ArgumentParser(description='Stage 4 Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pattern', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='MOSA_dataset/training_cache_W05')
    parser.add_argument('--torso_pred_dir', type=str, default='MOSA_dataset/phase1_torso_predictions')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.checkpoint:
        ckpt_paths = [args.checkpoint]
    elif args.pattern:
        ckpt_paths = sorted(glob.glob(args.pattern))
    else:
        parser.error('Provide --checkpoint or --pattern')

    print(f'Evaluating {len(ckpt_paths)} checkpoint(s)\n')
    test_data = load_test_data(args.cache_dir, args.torso_pred_dir)
    print(f'Loaded {len(test_data)} test samples\n')

    results = {}
    for path in tqdm(ckpt_paths, desc='Evaluating'):
        name = Path(path).stem
        r = evaluate_checkpoint(path, test_data, device)
        if r is not None:
            results[name] = r
            tqdm.write(f'  {name}: {r["mpjpe"]:.2f}mm')

    if not results:
        print("No results.")
        return

    pelvis_names = ['rasi', 'lasi', 'rpsi', 'lpsi']
    leg_names = ['rkne', 'rankle', 'rtoe', 'lkne', 'lankle', 'ltoe']

    print(f'\n{"Name":<40} {"MPJPE":<8} {"Pelvis":<8} {"Legs":<8} {"Epoch":<7}')
    print('-' * 71)
    for name, r in sorted(results.items(), key=lambda x: x[1]['mpjpe']):
        pelvis = np.mean([r[k] for k in pelvis_names])
        legs = np.mean([r[k] for k in leg_names])
        print(f'{name:<40} {r["mpjpe"]:<8.2f} {pelvis:<8.2f} {legs:<8.2f} {r["epoch"]+1:<7}')
    print('-' * 71)
    best = min(results, key=lambda k: results[k]['mpjpe'])
    print(f'Best: {best} (MPJPE={results[best]["mpjpe"]:.2f}mm)')

    # Save results to JSON (merge with existing if present)
    out_path = args.output
    if out_path is None:
        Path('results').mkdir(exist_ok=True)
        tag = Path(ckpt_paths[0]).stem.rsplit('_best', 1)[0].split('_')[0]
        out_path = f'results/stage4_eval_{tag}.json'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.pop('_best', None)
    existing.update({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()})
    all_results = {k: v for k, v in existing.items() if not k.startswith('_')}
    all_results['_best'] = min(all_results, key=lambda k: all_results[k]['mpjpe'])
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_path} ({len(all_results)-1} entries)')


if __name__ == '__main__':
    main()
