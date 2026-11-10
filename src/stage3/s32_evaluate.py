#!/usr/bin/env python3
"""
Stage 3 Evaluation: Compute test MPJPE for arm checkpoints.

Supports both MDM (diffusion) and regression models.
Handles absolute and relative coordinate systems.

Usage:
    python src/stage3/s32_evaluate.py --checkpoint checkpoints/stage3_experiments/phase0_absolute_best.pt
    python src/stage3/s32_evaluate.py --pattern "checkpoints/stage3_experiments/phaseC_*.pt"
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'vendor' / 'MDM'))

from models.model_factory import build_stage3_arm_model
from data.constants import (
    RSHO_IDX, RELB_IDX, LSHO_IDX, LELB_IDX,
    RWRIST_IDX, LWRIST_IDX, T10_IDX, STRN_IDX,
)

WINDOW, STRIDE = 480, 240
ARM_INDICES = [RSHO_IDX, RELB_IDX, LSHO_IDX, LELB_IDX]
ARM_NAMES = ['RSHO', 'RELB', 'LSHO', 'LELB']


def load_test_data(cache_dir, pred_dir, torso_pred_dir, config):
    """Load test data."""
    use_od = config.get('use_onset_density', False)
    use_pr = config.get('use_pitch_range', False)
    use_pr88 = config.get('use_piano_roll', False)
    no_spatial = config.get('no_spatial', False)
    test_data = []
    cache_test = Path(cache_dir) / 'test'
    hand_test = Path(pred_dir) / 'test'
    torso_test = Path(torso_pred_dir) / 'test'

    for f in sorted(cache_test.glob('*.npz')):
        data = np.load(f)
        joints = data['joints'].astype(np.float32)
        T = len(joints)

        arm_gt = joints[:, ARM_INDICES, :].reshape(T, -1)  # (T, 12) absolute

        # Wrists
        hp = hand_test / f.name
        if hp.exists():
            hd = np.load(hp)
            hp_data = hd.get('hand_pred', hd.get('pred'))
            if hp_data is not None:
                hp_data = hp_data.astype(np.float32)
                wrists = np.concatenate([hp_data[:, 3:6], hp_data[:, 0:3]], axis=-1)
            else:
                wrists = np.concatenate([joints[:, RWRIST_IDX], joints[:, LWRIST_IDX]], axis=-1).astype(np.float32)
        else:
            wrists = np.concatenate([joints[:, RWRIST_IDX], joints[:, LWRIST_IDX]], axis=-1).astype(np.float32)

        # Chest
        tp = torso_test / f.name
        if tp.exists():
            td = np.load(tp)
            tp_data = td.get('torso_pred')
            if tp_data is not None:
                chest = tp_data.astype(np.float32)[:, 6:12]
            else:
                chest = joints[:, [T10_IDX, STRN_IDX], :].reshape(T, 6).astype(np.float32)
        else:
            chest = joints[:, [T10_IDX, STRN_IDX], :].reshape(T, 6).astype(np.float32)

        if no_spatial:
            spatial_cond = np.zeros((T, 0), dtype=np.float32)
        else:
            spatial_cond = np.concatenate([wrists, chest], axis=-1)  # (T, 12)

        # ARIA
        aria = data.get('aria_embed', np.zeros(512)).astype(np.float32)
        if aria.ndim == 2:
            aria = aria.mean(axis=0)

        # Temporal conditioning
        tc_parts = []
        if use_od:
            tc_parts.append(data.get('onset_density', np.zeros(T)).astype(np.float32).reshape(T, 1))
        if use_pr:
            tc_parts.append(data.get('pitch_range', np.zeros(T)).astype(np.float32).reshape(T, 1))
        if use_pr88:
            ak = data['active_keys']
            pr88 = np.zeros((T, 88), dtype=np.float32)
            for t in range(T):
                for k in ak[t]:
                    if 0 <= k < 88:
                        pr88[t, k] = 1.0
            tc_parts.append(pr88)
        od = np.concatenate(tc_parts, axis=-1) if tc_parts else None

        test_data.append((arm_gt, spatial_cond, T, aria, od))

    return test_data


def windowed_regression(model, spatial_np, T, device, aria=None, od_np=None, use_aria=False):
    """Windowed inference for regression models."""
    pred_sum = np.zeros((T, 12), dtype=np.float64)
    weight_sum = np.zeros((T, 1), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, max(1, T - WINDOW + 1), STRIDE):
            e = min(s + WINDOW, T)
            al = e - s

            sp_dim = spatial_np.shape[-1]
            chunk = spatial_np[s:e]
            if len(chunk) < WINDOW:
                chunk = np.concatenate([chunk, np.zeros((WINDOW - len(chunk), sp_dim), dtype=np.float32)])
            ct = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)

            aria_t = None
            if use_aria and aria is not None:
                aria_t = torch.tensor(aria, dtype=torch.float32).unsqueeze(0).to(device)

            tc_t = None
            if od_np is not None:
                tc_dim = od_np.shape[-1]
                od_chunk = od_np[s:e]
                if len(od_chunk) < WINDOW:
                    od_chunk = np.concatenate([od_chunk, np.zeros((WINDOW - len(od_chunk), tc_dim), dtype=np.float32)])
                tc_t = torch.tensor(od_chunk, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(ct, aria=aria_t, temporal_cond=tc_t).cpu().numpy()[0]
            w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
            pred_sum[s:e] += pred[:al] * w
            weight_sum[s:e] += w
    mask = weight_sum > 0
    return np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)


def windowed_mdm(model, diffusion, spatial_np, T, device, aria=None, use_aria=False):
    """Windowed DDIM sampling for MDM models."""
    from diffusion_lib import ddim_sample
    pred_sum = np.zeros((T, 12), dtype=np.float64)
    weight_sum = np.zeros((T, 1), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, max(1, T - WINDOW + 1), STRIDE):
            e = min(s + WINDOW, T)
            al = e - s

            sp_dim = spatial_np.shape[-1]
            chunk = spatial_np[s:e]
            if len(chunk) < WINDOW:
                chunk = np.concatenate([chunk, np.zeros((WINDOW - len(chunk), sp_dim), dtype=np.float32)])
            ct = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)

            aria_t = None
            if use_aria and aria is not None:
                aria_t = torch.tensor(aria, dtype=torch.float32).unsqueeze(0).to(device)

            shape = (1, WINDOW, 12)
            sample = ddim_sample(
                model, diffusion, shape,
                model_kwargs={'spatial_cond': ct, 'global_cond': aria_t},
                ddim_steps=50, device=str(device), progress=False,
            )
            pred = sample.cpu().numpy()[0]
            w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
            pred_sum[s:e] += pred[:al] * w
            weight_sum[s:e] += w
    mask = weight_sum > 0
    return np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)


def evaluate_checkpoint(ckpt_path, test_data, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    arch = config.get('arch', 'mdm')
    relative = config.get('relative_coords', False)
    use_aria = config.get('use_aria', False)
    use_od = config.get('use_onset_density', False)
    is_diffusion = (arch == 'mdm')

    model = build_stage3_arm_model(config, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    diffusion = None
    if is_diffusion:
        from diffusion_lib import create_diffusion
        diffusion = create_diffusion(num_timesteps=1000)

    all_errors = []
    for arm_gt, spatial_cond, T, aria, od in tqdm(test_data, desc='Samples', leave=False):
        if is_diffusion:
            pred = windowed_mdm(model, diffusion, spatial_cond, T, device,
                                aria=aria, use_aria=use_aria)
        else:
            pred = windowed_regression(model, spatial_cond, T, device,
                                       aria=aria, od_np=od,
                                       use_aria=use_aria)

        # Convert relative back to absolute for evaluation
        if relative:
            rsho = pred[:, 0:3]
            relb_rel = pred[:, 3:6]
            lsho = pred[:, 6:9]
            lelb_rel = pred[:, 9:12]
            pred = np.concatenate([rsho, rsho + relb_rel, lsho, lsho + lelb_rel], axis=-1)

        gt_r = arm_gt.reshape(T, 4, 3)
        pred_r = pred.reshape(T, 4, 3)
        err = np.sqrt(((gt_r - pred_r) ** 2).sum(axis=-1))
        all_errors.append(err)

    ae = np.concatenate(all_errors, axis=0)
    del model
    torch.cuda.empty_cache()

    result = {
        'mpjpe': ae.mean() * 1000,
        'val_loss': ckpt.get('val_loss', float('nan')),
        'epoch': ckpt.get('epoch', -1),
    }
    for i, name in enumerate(ARM_NAMES):
        result[name.lower()] = ae[:, i].mean() * 1000
    return result


def main():
    parser = argparse.ArgumentParser(description='Stage 3 Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pattern', type=str, default=None)
    # Aliases for backward compatibility
    parser.add_argument('--arm_checkpoint', type=str, default=None)
    parser.add_argument('--arm_pattern', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='MOSA_dataset/training_cache_W10')
    parser.add_argument('--pred_dir', type=str, default='MOSA_dataset/phase0_hand_predictions')
    parser.add_argument('--torso_pred_dir', type=str, default='MOSA_dataset/phase1_torso_predictions')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Handle aliases
    ckpt_arg = args.checkpoint or args.arm_checkpoint
    pattern_arg = args.pattern or args.arm_pattern

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if ckpt_arg:
        ckpt_paths = [ckpt_arg]
    elif pattern_arg:
        ckpt_paths = sorted(glob.glob(pattern_arg))
    else:
        parser.error('Provide --checkpoint or --pattern')

    print(f'Evaluating {len(ckpt_paths)} checkpoint(s)\n')

    # Load test data using first checkpoint's config
    first_config = torch.load(ckpt_paths[0], map_location='cpu', weights_only=False).get('config', {})
    test_data = load_test_data(args.cache_dir, args.pred_dir, args.torso_pred_dir, first_config)
    print(f'Loaded {len(test_data)} test samples\n')

    results = {}
    for path in tqdm(ckpt_paths, desc='Evaluating'):
        name = Path(path).stem
        cfg = torch.load(path, map_location='cpu', weights_only=False).get('config', {})
        # Reload test data if config differs
        if any(cfg.get(k) != first_config.get(k) for k in
               ['use_aria', 'use_onset_density', 'use_pitch_range', 'use_piano_roll', 'no_spatial']):
            td = load_test_data(args.cache_dir, args.pred_dir, args.torso_pred_dir, cfg)
        else:
            td = test_data
        r = evaluate_checkpoint(path, td, device)
        results[name] = r
        tqdm.write(f'  {name}: {r["mpjpe"]:.2f}mm (RSHO={r["rsho"]:.1f} RELB={r["relb"]:.1f} LSHO={r["lsho"]:.1f} LELB={r["lelb"]:.1f})')

    # Print table
    print(f'\n{"Name":<40} {"MPJPE":<8} {"RSHO":<8} {"RELB":<8} {"LSHO":<8} {"LELB":<8} {"Epoch":<7}')
    print('-' * 87)
    for name, r in sorted(results.items(), key=lambda x: x[1]['mpjpe']):
        print(f'{name:<40} {r["mpjpe"]:<8.2f} {r["rsho"]:<8.2f} {r["relb"]:<8.2f} '
              f'{r["lsho"]:<8.2f} {r["lelb"]:<8.2f} {r["epoch"]+1:<7}')
    print('-' * 87)
    best = min(results, key=lambda k: results[k]['mpjpe'])
    print(f'Best: {best} (MPJPE={results[best]["mpjpe"]:.2f}mm)')

    # Save JSON (merge with existing)
    out_path = args.output
    if out_path is None:
        Path('results').mkdir(exist_ok=True)
        tag = Path(ckpt_paths[0]).stem.rsplit('_best', 1)[0].split('_')[0]
        out_path = f'results/stage3_eval_{tag}.json'
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
