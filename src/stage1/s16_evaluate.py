#!/usr/bin/env python3
"""
Stage 1 Evaluation: Compute test MPJPE for one or more checkpoints.

Usage:
    # Single checkpoint
    python src/stage1/s16_evaluate.py --checkpoint checkpoints/stage1_experiments/phaseC_h512_l6_lr1e-4_best.pt

    # All checkpoints matching a pattern (e.g., Phase C)
    python src/stage1/s16_evaluate.py --pattern "checkpoints/stage1_experiments/phaseC_*.pt"

    # Custom cache dir
    python src/stage1/s16_evaluate.py --checkpoint ckpt.pt --cache_dir MOSA_dataset/training_cache_W05
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
from models.model_factory import build_stage1_model
from data.constants import LWRIST_IDX, RWRIST_IDX, FPS

WINDOW, STRIDE = 480, 240


# ============================================================================
# GT-comparison expressive metrics
# ----------------------------------------------------------------------------
# hand layout: (T, 6) = [lwrist_xyz, rwrist_xyz]  →  reshape (T, 2, 3)
# All metrics use GT as reference. Lower is better unless noted.
#
#   mpjpe_mm           : overall Euclidean position error (mm)  [lower better]
#   mpjpe_z_mm         : z-axis (wrist lift) position error (mm) [lower better]
#   vel_err_mm_s       : wrist velocity vector error (mm/s)     [lower better]
#   speed_corr         : Pearson corr of per-frame speed vs GT  [higher better]
#   jerk_pred/gt_m_s2  : acceleration magnitude (smoothness)
#   jerk_ratio         : jerk_pred / jerk_gt (1.0 = matches GT smoothness)
#   onset_mpjpe_mm     : MPJPE restricted to MIDI onset frames  [lower better]
#   z_range_ratio      : wrist z peak-to-peak ratio (pred/gt)   [1.0 = matched]
#   z_std_ratio        : wrist z std ratio (pred/gt)            [1.0 = matched]
#   dyn_corr_pred/gt   : corr(wrist speed, MIDI velocity mass); compare to GT
# ============================================================================

def _safe_corr(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return None
    c = np.corrcoef(a, b)[0, 1]
    return None if np.isnan(c) else float(c)


def compute_expressive_metrics(pred, gt, key_onset, midi_vel, fps=FPS):
    """Per-sample GT-comparison metrics. pred/gt: (T, 6); key_onset/midi_vel: (T, 88)."""
    T = pred.shape[0]
    pred_j = pred.reshape(T, 2, 3).astype(np.float64)
    gt_j = gt.reshape(T, 2, 3).astype(np.float64)

    # Position errors
    pos_err = np.sqrt(((pred_j - gt_j) ** 2).sum(-1))        # (T, 2) meters
    z_err = np.abs(pred_j[:, :, 2] - gt_j[:, :, 2])          # (T, 2) meters

    # Velocity vectors (m/s) and speeds
    pv = (pred_j[1:] - pred_j[:-1]) * fps                     # (T-1, 2, 3)
    gv = (gt_j[1:] - gt_j[:-1]) * fps
    vel_err = np.linalg.norm(pv - gv, axis=-1)                # (T-1, 2)
    ps = np.linalg.norm(pv, axis=-1)                          # (T-1, 2)
    gs = np.linalg.norm(gv, axis=-1)

    # Speed correlation (averaged across joints)
    corrs = [c for c in (_safe_corr(ps[:, j], gs[:, j]) for j in range(2)) if c is not None]
    speed_corr = float(np.mean(corrs)) if corrs else 0.0

    # Jerk (acceleration magnitude, m/s^2)
    pa = (pv[1:] - pv[:-1]) * fps
    ga = (gv[1:] - gv[:-1]) * fps
    jerk_pred = float(np.linalg.norm(pa, axis=-1).mean())
    jerk_gt = float(np.linalg.norm(ga, axis=-1).mean())

    # Onset-aligned MPJPE (use per-frame mean joint error)
    onset_mask = key_onset.sum(axis=-1) > 0                   # (T,)
    if onset_mask.sum() > 0:
        onset_mpjpe = float(pos_err[onset_mask].mean() * 1000)
    else:
        onset_mpjpe = float('nan')

    # Z amplitude ratios (per-joint, averaged)
    pz = pred_j[:, :, 2]; gz = gt_j[:, :, 2]
    p_range = pz.max(0) - pz.min(0); g_range = gz.max(0) - gz.min(0)
    z_range_ratio = float((p_range / np.maximum(g_range, 1e-8)).mean())
    z_std_ratio = float((pz.std(0) / np.maximum(gz.std(0), 1e-8)).mean())

    # MIDI-velocity ↔ wrist-speed dynamics correlation
    midi_amp = midi_vel.sum(-1)[1:]                            # (T-1,)
    pred_mean_speed = ps.mean(-1)                              # (T-1,)
    gt_mean_speed = gs.mean(-1)
    dyn_corr_pred = _safe_corr(pred_mean_speed, midi_amp) or 0.0
    dyn_corr_gt = _safe_corr(gt_mean_speed, midi_amp) or 0.0

    return {
        'mpjpe_z_mm': float(z_err.mean() * 1000),
        'vel_err_mm_s': float(vel_err.mean() * 1000),
        'speed_corr': speed_corr,
        'jerk_pred_m_s2': jerk_pred,
        'jerk_gt_m_s2': jerk_gt,
        'jerk_ratio': float(jerk_pred / jerk_gt) if jerk_gt > 0 else float('nan'),
        'onset_mpjpe_mm': onset_mpjpe,
        'z_range_ratio': z_range_ratio,
        'z_std_ratio': z_std_ratio,
        'dyn_corr_pred': dyn_corr_pred,
        'dyn_corr_gt': dyn_corr_gt,
    }


def build_model(config, midi_dim, device):
    """Build model from checkpoint config via centralized factory."""
    return build_stage1_model(config, device)


def windowed_inference(model, midi_np, T, midi_dim, device):
    pred_sum = np.zeros((T, 6), dtype=np.float64)
    weight_sum = np.zeros((T, 1), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, max(1, T - WINDOW + 1), STRIDE):
            e = min(s + WINDOW, T)
            chunk = midi_np[s:e]
            if len(chunk) < WINDOW:
                chunk = np.concatenate([chunk, np.zeros((WINDOW - len(chunk), midi_dim), dtype=np.float32)])
            ct = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            md = {'piano_roll': ct[:, :, :88], 'velocity': ct[:, :, 88:176],
                  'onset': ct[:, :, 176:264]}
            if midi_dim > 264:
                md['dur_ioi_ratio'] = ct[:, :, 264:352]
            pred = model(md).cpu().numpy()[0]
            al = e - s
            w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
            pred_sum[s:e] += pred[:al] * w
            weight_sum[s:e] += w
    mask = weight_sum > 0
    return np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)


def load_test_data(cache_dir, config):
    """Load test data, respecting input_level flags."""
    no_velocity = config.get('no_velocity', False)
    no_onset = config.get('no_onset', False)
    use_dur_ioi = config.get('use_dur_ioi', False)

    test_data = []
    test_dir = Path(cache_dir) / 'test'
    for f in sorted(test_dir.glob('*.npz')):
        data = np.load(f)
        joints = data['joints'].astype(np.float32)
        T = len(joints)
        gt = np.concatenate([joints[:, LWRIST_IDX], joints[:, RWRIST_IDX]], axis=-1)

        ak = data['active_keys']
        pr = np.zeros((T, 88), dtype=np.float32)
        for t in range(T):
            for k in ak[t]:
                if 0 <= k < 88:
                    pr[t, k] = 1.0

        vel = np.zeros((T, 88), dtype=np.float32) if no_velocity else \
            data.get('onset_density_per_key', np.zeros((T, 88))).astype(np.float32)
        ons = np.zeros((T, 88), dtype=np.float32) if no_onset else \
            data.get('key_onset', np.zeros((T, 88))).astype(np.float32)

        parts = [pr, vel, ons]
        if use_dur_ioi:
            parts.append(data.get('dur_ioi_ratio', np.zeros((T, 88))).astype(np.float32))
        midi_np = np.concatenate(parts, axis=-1)

        # Raw onset / velocity for expressive-metric computation (not zeroed)
        key_onset = data.get('key_onset', np.zeros((T, 88))).astype(np.float32)
        midi_velocity = data.get('onset_density_per_key', np.zeros((T, 88))).astype(np.float32)
        test_data.append((gt, midi_np, T, key_onset, midi_velocity))
    return test_data


def evaluate_checkpoint(ckpt_path, test_data, device):
    """Evaluate a single checkpoint. Returns (mpjpe, lwrist, rwrist, val_loss, epoch)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    use_dur_ioi = config.get('use_dur_ioi', False)
    midi_dim = 352 if use_dur_ioi else 264

    model = build_model(config, midi_dim, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_errors = []
    per_sample_expr = []
    for gt, midi_np, T, key_onset, midi_velocity in test_data:
        pred = windowed_inference(model, midi_np, T, midi_dim, device)
        err = np.sqrt(((gt.reshape(T, 2, 3) - pred.reshape(T, 2, 3)) ** 2).sum(axis=-1))
        all_errors.append(err)
        per_sample_expr.append(
            compute_expressive_metrics(pred, gt, key_onset, midi_velocity, fps=FPS)
        )

    ae = np.concatenate(all_errors, axis=0)
    del model
    torch.cuda.empty_cache()

    # Aggregate expressive metrics across samples (nanmean for onset_mpjpe which may be nan)
    expr_keys = per_sample_expr[0].keys()
    expr_agg = {k: float(np.nanmean([s[k] for s in per_sample_expr])) for k in expr_keys}

    return {
        'mpjpe': ae.mean() * 1000,
        'lwrist': ae[:, 0].mean() * 1000,
        'rwrist': ae[:, 1].mean() * 1000,
        **expr_agg,
        'val_loss': ckpt.get('val_loss', float('nan')),
        'epoch': ckpt.get('epoch', -1),
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 1 Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Single checkpoint path')
    parser.add_argument('--pattern', type=str, default=None, help='Glob pattern for multiple checkpoints')
    parser.add_argument('--cache_dir', type=str, default='MOSA_dataset/training_cache_W05')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON (auto-generated if not specified)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Collect checkpoint paths
    if args.checkpoint:
        ckpt_paths = [args.checkpoint]
    elif args.pattern:
        ckpt_paths = sorted(glob.glob(args.pattern))
    else:
        parser.error('Provide --checkpoint or --pattern')

    print(f'Evaluating {len(ckpt_paths)} checkpoint(s) on {args.cache_dir}/test')

    # Load test data (use first checkpoint's config for input flags)
    first_config = torch.load(ckpt_paths[0], map_location='cpu', weights_only=False)['config']
    test_data = load_test_data(args.cache_dir, first_config)
    print(f'Loaded {len(test_data)} test samples\n')

    # Evaluate
    results = {}
    for path in tqdm(ckpt_paths, desc='Evaluating'):
        name = Path(path).stem
        # Reload test data if config differs (different input_level)
        cfg = torch.load(path, map_location='cpu', weights_only=False)['config']
        if (cfg.get('use_dur_ioi') != first_config.get('use_dur_ioi') or
            cfg.get('no_velocity') != first_config.get('no_velocity') or
            cfg.get('no_onset') != first_config.get('no_onset')):
            td = load_test_data(args.cache_dir, cfg)
        else:
            td = test_data
        r = evaluate_checkpoint(path, td, device)
        results[name] = r
        tqdm.write(f'  {name}: {r["mpjpe"]:.2f}mm')

    # Print table — position accuracy
    print(f'\n[Position accuracy]  (lower is better)')
    print(f'{"Name":<40} {"MPJPE":>8} {"LWrist":>8} {"RWrist":>8} {"Z-err":>8} {"OnsetMP":>8} {"Epoch":>6}')
    print('-' * 96)
    for name, r in sorted(results.items(), key=lambda x: x[1]['mpjpe']):
        print(f'{name:<40} {r["mpjpe"]:>8.2f} {r["lwrist"]:>8.2f} {r["rwrist"]:>8.2f} '
              f'{r["mpjpe_z_mm"]:>8.2f} {r["onset_mpjpe_mm"]:>8.2f} {r["epoch"]+1:>6}')
    print('-' * 96)

    # Print table — expressive dynamics (vs GT)
    print(f'\n[Expressive dynamics vs GT]  (speed/dyn corr: higher better; ratios: closer to 1.0 better)')
    print(f'{"Name":<40} {"VelErr":>8} {"SpdCorr":>8} {"JerkR":>8} {"ZRngR":>8} {"ZStdR":>8} {"DynP/G":>12}')
    print('-' * 96)
    for name, r in sorted(results.items(), key=lambda x: x[1]['mpjpe']):
        print(f'{name:<40} {r["vel_err_mm_s"]:>8.1f} {r["speed_corr"]:>8.3f} '
              f'{r["jerk_ratio"]:>8.2f} {r["z_range_ratio"]:>8.2f} {r["z_std_ratio"]:>8.2f} '
              f'{r["dyn_corr_pred"]:>5.2f}/{r["dyn_corr_gt"]:<5.2f}')
    print('-' * 96)
    best = min(results, key=lambda k: results[k]['mpjpe'])
    print(f'Best: {best} (MPJPE={results[best]["mpjpe"]:.2f}mm)')

    # Save results to JSON (merge with existing if present)
    out_path = args.output
    if out_path is None:
        Path('results').mkdir(exist_ok=True)
        tag = Path(ckpt_paths[0]).stem.rsplit('_best', 1)[0].split('_')[0]
        out_path = f'results/stage1_eval_{tag}.json'
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
