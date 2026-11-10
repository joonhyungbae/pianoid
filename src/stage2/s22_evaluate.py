#!/usr/bin/env python3
"""
Stage 2 Evaluation: Compute test MPJPE for torso checkpoints.

Usage:
    python src/stage2/s22_evaluate.py --checkpoint checkpoints/stage2_experiments/phaseA_hand_only_best.pt
    python src/stage2/s22_evaluate.py --pattern "checkpoints/stage2_experiments/phaseC_*.pt"
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

from models.model_factory import build_stage2_model
from data.constants import LWRIST_IDX, RWRIST_IDX, TORSO_INDICES, FPS

WINDOW, STRIDE = 480, 240
JOINT_NAMES = ['C7', 'CLAV', 'T10', 'STRN']


def _safe_corr(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return None
    c = np.corrcoef(a, b)[0, 1]
    return None if np.isnan(c) else float(c)


def compute_torso_expressive_metrics(pred, gt, fps=FPS):
    """Per-sample GT-comparison metrics. pred/gt: (T, D) where D=12 or 24."""
    T, D = pred.shape
    n_joints = D // 3
    pred_j = pred.reshape(T, n_joints, 3).astype(np.float64)
    gt_j = gt.reshape(T, n_joints, 3).astype(np.float64)

    # Velocity & speed
    pv = (pred_j[1:] - pred_j[:-1]) * fps
    gv = (gt_j[1:] - gt_j[:-1]) * fps
    ps = np.linalg.norm(pv, axis=-1)   # (T-1, n_joints)
    gs = np.linalg.norm(gv, axis=-1)

    # Speed correlation (per joint, averaged)
    corrs = [c for c in (_safe_corr(ps[:, j], gs[:, j]) for j in range(n_joints)) if c is not None]
    speed_corr = float(np.mean(corrs)) if corrs else 0.0

    # Speed ratio (pred/gt, averaged)
    speed_ratio = float((ps.mean(0) / np.maximum(gs.mean(0), 1e-8)).mean())

    # Jerk ratio
    pa = (pv[1:] - pv[:-1]) * fps
    ga = (gv[1:] - gv[:-1]) * fps
    jerk_pred = float(np.linalg.norm(pa, axis=-1).mean())
    jerk_gt = float(np.linalg.norm(ga, axis=-1).mean())
    jerk_ratio = jerk_pred / jerk_gt if jerk_gt > 0 else float('nan')

    # Per-axis range ratio (all D dims, then average)
    r_pred = pred.max(0) - pred.min(0)  # (D,)
    r_gt = gt.max(0) - gt.min(0)
    range_ratio = float((r_pred / np.maximum(r_gt, 1e-8)).mean())

    # Per-axis range ratio by axis group (x, y, z)
    x_idx = list(range(0, D, 3))
    y_idx = list(range(1, D, 3))
    z_idx = list(range(2, D, 3))
    rr_x = float((r_pred[x_idx] / np.maximum(r_gt[x_idx], 1e-8)).mean())
    rr_y = float((r_pred[y_idx] / np.maximum(r_gt[y_idx], 1e-8)).mean())
    rr_z = float((r_pred[z_idx] / np.maximum(r_gt[z_idx], 1e-8)).mean())

    # Std ratio
    s_pred = pred.std(0)
    s_gt = gt.std(0)
    std_ratio = float((s_pred / np.maximum(s_gt, 1e-8)).mean())

    return {
        'speed_corr': speed_corr,
        'speed_ratio': speed_ratio,
        'jerk_ratio': float(jerk_ratio),
        'range_ratio': range_ratio,
        'range_ratio_x': rr_x,
        'range_ratio_y': rr_y,
        'range_ratio_z': rr_z,
        'std_ratio': std_ratio,
    }


def build_model(config, device):
    """Build torso model from checkpoint config via centralized factory."""
    return build_stage2_model(config, device)


def windowed_inference(model, hand_np, T, device, aria=None, temporal_cond_np=None,
                       no_hand=False, out_dim=12, is_mdm=False, diffusion=None, ddim_steps=50):
    """Run windowed inference with triangular blending."""
    pred_sum = np.zeros((T, out_dim), dtype=np.float64)
    weight_sum = np.zeros((T, 1), dtype=np.float64)

    with torch.no_grad():
        for s in range(0, max(1, T - WINDOW + 1), STRIDE):
            e = min(s + WINDOW, T)
            al = e - s

            hand_chunk = hand_np[s:e]
            if len(hand_chunk) < WINDOW:
                hand_chunk = np.concatenate([hand_chunk,
                    np.zeros((WINDOW - len(hand_chunk), hand_np.shape[-1]), dtype=np.float32)])
            hand_t = torch.tensor(hand_chunk, dtype=torch.float32).unsqueeze(0).to(device)

            aria_t = None
            if aria is not None:
                aria_t = torch.tensor(aria, dtype=torch.float32).unsqueeze(0).to(device)

            tc_t = None
            if temporal_cond_np is not None:
                tc_chunk = temporal_cond_np[s:e]
                if len(tc_chunk) < WINDOW:
                    tc_chunk = np.concatenate([tc_chunk,
                        np.zeros((WINDOW - len(tc_chunk), temporal_cond_np.shape[-1]), dtype=np.float32)])
                tc_t = torch.tensor(tc_chunk, dtype=torch.float32).unsqueeze(0).to(device)

            if is_mdm and diffusion is not None:
                from diffusion_lib.diffusion_utils import ddim_sample
                model_kwargs = {
                    'spatial_cond': None if no_hand else hand_t,
                    'global_cond': aria_t,
                    'temporal_cond': tc_t,
                }
                shape = (1, WINDOW, out_dim)
                pred = ddim_sample(model, diffusion, shape, model_kwargs,
                                   ddim_steps=ddim_steps, device=device,
                                   progress=False).cpu().numpy()[0]
            else:
                pred = model(None if no_hand else hand_t, aria=aria_t,
                             temporal_cond=tc_t).cpu().numpy()[0]

            w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
            pred_sum[s:e] += pred[:al] * w
            weight_sum[s:e] += w

    mask = weight_sum > 0
    return np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)


def load_test_data(cache_dir, pred_dir, config):
    """Load test data with appropriate conditioning."""
    use_pr = config.get('use_pitch_range', False)
    use_od = config.get('use_onset_density', False)
    use_opk = config.get('use_onset_per_key', False)
    use_pr88 = config.get('use_piano_roll', False)
    use_aria = config.get('use_aria', False)
    is_backbone = config.get('backbone', False)

    test_data = []
    cache_test = Path(cache_dir) / 'test'
    pred_test = Path(pred_dir) / 'test'

    for f in sorted(cache_test.glob('*.npz')):
        data = np.load(f)
        joints = data['joints'].astype(np.float32)
        T = len(joints)

        if is_backbone:
            from data.constants import RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX
            HEAD_IDX = [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX]
            torso_gt = joints[:, HEAD_IDX + TORSO_INDICES, :].reshape(T, -1)  # (T, 24)
        else:
            torso_gt = joints[:, TORSO_INDICES, :].reshape(T, -1)  # (T, 12)

        # Hand pred
        pred_path = pred_test / f.name
        if pred_path.exists():
            pd = np.load(pred_path)
            hand_pred = pd.get('hand_pred', pd.get('pred'))
            if hand_pred is not None:
                hand_pred = hand_pred.astype(np.float32)
            else:
                hand_pred = np.concatenate([joints[:, LWRIST_IDX], joints[:, RWRIST_IDX]], axis=-1).astype(np.float32)
        else:
            hand_pred = np.concatenate([joints[:, LWRIST_IDX], joints[:, RWRIST_IDX]], axis=-1).astype(np.float32)

        # ARIA
        aria = None
        if use_aria:
            aria = data.get('aria_embed', np.zeros(512)).astype(np.float32)
            if aria.ndim == 2:
                aria = aria.mean(axis=0)

        # Temporal conditioning
        tc_parts = []
        if use_pr:
            tc_parts.append(data.get('pitch_range', np.zeros(T)).astype(np.float32).reshape(T, 1))
        if use_od:
            tc_parts.append(data.get('onset_density', np.zeros(T)).astype(np.float32).reshape(T, 1))
        if use_opk:
            tc_parts.append(data.get('onset_density_per_key', np.zeros((T, 88))).astype(np.float32))
        if use_pr88:
            ak = data['active_keys']
            pr88 = np.zeros((T, 88), dtype=np.float32)
            for t in range(T):
                for k in ak[t]:
                    if 0 <= k < 88:
                        pr88[t, k] = 1.0
            tc_parts.append(pr88)
        temporal_cond = np.concatenate(tc_parts, axis=-1) if tc_parts else None

        test_data.append((torso_gt, hand_pred, T, aria, temporal_cond))

    return test_data


def _denorm_pred(pred_np, repr_mode, norm_stats, torso_gt):
    """Denormalize model output back to absolute positions."""
    if repr_mode == 'absolute' or repr_mode is None:
        return pred_np
    if repr_mode == 'zscore':
        mean = np.array(norm_stats['torso_mean'], dtype=np.float32)
        std = np.array(norm_stats['torso_std'], dtype=np.float32)
        return pred_np * std + mean
    elif repr_mode == 'delta':
        first_frame = torso_gt[0:1]  # (1, 12) — use GT first frame
        return first_frame + np.cumsum(pred_np, axis=0)
    elif repr_mode == 'residual':
        mean = np.array(norm_stats['torso_mean'], dtype=np.float32)
        return pred_np + mean
    return pred_np


def _prepare_hand_input(hand_pred, hand_velocity):
    """Optionally append hand velocity to hand positions."""
    if not hand_velocity:
        return hand_pred
    vel = np.zeros_like(hand_pred)
    vel[1:] = hand_pred[1:] - hand_pred[:-1]
    return np.concatenate([hand_pred, vel], axis=-1)


def evaluate_checkpoint(ckpt_path, test_data, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    norm_stats = ckpt.get('norm_stats', {})
    repr_mode = config.get('repr', 'absolute')
    hand_velocity = config.get('hand_velocity', False)
    is_mdm = (config.get('arch') == 'mdm')
    is_backbone = config.get('backbone', False)
    out_dim = 24 if is_backbone else 12

    model = build_model(config, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    diffusion = None
    if is_mdm:
        from diffusion_lib import create_diffusion
        diffusion = create_diffusion(num_timesteps=config.get('diffusion_steps', 1000))

    no_hand = config.get('no_hand', False)
    all_errors = []
    per_sample_expr = []
    for torso_gt, hand_pred, T, aria, temporal_cond in test_data:
        hand_input = _prepare_hand_input(hand_pred, hand_velocity)
        pred_raw = windowed_inference(model, hand_input, T, device, aria=aria,
                                      temporal_cond_np=temporal_cond, no_hand=no_hand,
                                      out_dim=out_dim, is_mdm=is_mdm, diffusion=diffusion)
        pred = _denorm_pred(pred_raw, repr_mode, norm_stats, torso_gt)

        n_joints = out_dim // 3
        gt_eval = torso_gt[:, :out_dim] if torso_gt.shape[-1] >= out_dim else torso_gt
        pred_eval = pred[:, :out_dim] if pred.shape[-1] >= out_dim else pred
        gt_r = gt_eval.reshape(T, n_joints, 3)
        pred_r = pred_eval.reshape(T, n_joints, 3)
        err = np.sqrt(((gt_r - pred_r) ** 2).sum(axis=-1))
        all_errors.append(err)
        per_sample_expr.append(compute_torso_expressive_metrics(pred_eval, gt_eval))

    ae = np.concatenate(all_errors, axis=0)
    n_joints = ae.shape[1]
    del model
    torch.cuda.empty_cache()

    expr_keys = per_sample_expr[0].keys()
    expr_agg = {k: float(np.nanmean([s[k] for s in per_sample_expr])) for k in expr_keys}

    result = {
        'mpjpe': ae.mean() * 1000,
        **expr_agg,
        'val_loss': ckpt.get('val_loss', float('nan')),
        'epoch': ckpt.get('epoch', -1),
        'repr': repr_mode,
        'backbone': is_backbone,
    }

    # Per-joint names
    if is_backbone:
        bnames = ['rfhd','lfhd','rbhd','lbhd','c7','clav','t10','strn']
        for i, n in enumerate(bnames):
            result[n] = ae[:, i].mean() * 1000
    else:
        for i, n in enumerate(['c7','clav','t10','strn']):
            result[n] = ae[:, i].mean() * 1000

    return result


def main():
    parser = argparse.ArgumentParser(description='Stage 2 Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pattern', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='MOSA_dataset/training_cache_W05')
    parser.add_argument('--pred_dir', type=str, default='MOSA_dataset/phase0_hand_predictions')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON')
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

    # Load test data using first checkpoint's config
    first_config = torch.load(ckpt_paths[0], map_location='cpu', weights_only=False).get('config', {})
    test_data = load_test_data(args.cache_dir, args.pred_dir, first_config)
    print(f'Loaded {len(test_data)} test samples\n')

    results = {}
    for path in tqdm(ckpt_paths, desc='Evaluating'):
        name = Path(path).parent.name if Path(path).stem == 'best' else Path(path).stem
        cfg = torch.load(path, map_location='cpu', weights_only=False).get('config', {})
        # Reload test data if conditioning flags differ
        if any(cfg.get(k) != first_config.get(k) for k in
               ['use_aria', 'use_pitch_range', 'use_onset_density', 'use_onset_per_key', 'use_piano_roll', 'no_hand']):
            td = load_test_data(args.cache_dir, args.pred_dir, cfg)
        else:
            td = test_data
        r = evaluate_checkpoint(path, td, device)
        results[name] = r
        tqdm.write(f'  {name}: {r["mpjpe"]:.2f}mm')

    # Position accuracy table
    print(f'\n[Position accuracy]')
    print(f'{"Name":<40} {"MPJPE":>7} {"C7":>7} {"CLAV":>7} {"T10":>7} {"STRN":>7} {"Epoch":>6}')
    print('-' * 90)
    for name, r in sorted(results.items(), key=lambda x: x[1]['mpjpe']):
        print(f'{name:<40} {r["mpjpe"]:>7.2f} {r["c7"]:>7.2f} {r["clav"]:>7.2f} '
              f'{r["t10"]:>7.2f} {r["strn"]:>7.2f} {r["epoch"]+1:>6}')
    print('-' * 90)

    # Expressive dynamics table
    print(f'\n[Expressive dynamics vs GT]  (SpdCorr/SpdR: higher/1.0 better; JerkR/RngR/StdR: closer to 1.0)')
    print(f'{"Name":<40} {"SpdCorr":>8} {"SpdR":>6} {"JerkR":>7} {"RngX":>6} {"RngY":>6} {"RngZ":>6} {"StdR":>6}')
    print('-' * 90)
    for name, r in sorted(results.items(), key=lambda x: x[1]['mpjpe']):
        print(f'{name:<40} {r.get("speed_corr",0):>8.3f} {r.get("speed_ratio",0):>6.2f} '
              f'{r.get("jerk_ratio",0):>7.2f} {r.get("range_ratio_x",0):>6.2f} '
              f'{r.get("range_ratio_y",0):>6.2f} {r.get("range_ratio_z",0):>6.2f} '
              f'{r.get("std_ratio",0):>6.2f}')
    print('-' * 90)
    best = min(results, key=lambda k: results[k]['mpjpe'])
    print(f'Best: {best} (MPJPE={results[best]["mpjpe"]:.2f}mm)')

    # Save results to JSON (merge with existing if present)
    out_path = args.output
    if out_path is None:
        Path('results').mkdir(exist_ok=True)
        tag = Path(ckpt_paths[0]).stem.rsplit('_best', 1)[0].split('_')[0]
        out_path = f'results/stage2_eval_{tag}.json'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.pop('_best', None)
    def _to_json_val(v):
        if isinstance(v, str):
            return v
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    existing.update({k: {kk: _to_json_val(vv) for kk, vv in v.items()} for k, v in results.items()})
    all_results = {k: v for k, v in existing.items() if not k.startswith('_')}
    all_results['_best'] = min(all_results, key=lambda k: all_results[k]['mpjpe'])
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_path} ({len(all_results)-1} entries)')


if __name__ == '__main__':
    main()
