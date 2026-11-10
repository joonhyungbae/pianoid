#!/usr/bin/env python3
"""
Full Pipeline Inference + Evaluation

Runs all stages sequentially on test set, assembles 26-joint skeleton,
and computes MPJPE, per-body-part MPJPE, FID, Jitter, and Velocity Correlation.

Usage:
    python src/pipeline/full_pipeline.py
    python src/pipeline/full_pipeline.py --output results/full_pipeline.json
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'vendor' / 'MDM'))

from data.constants import (
    LWRIST_IDX, RWRIST_IDX,
    C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
    RSHO_IDX, RELB_IDX, LSHO_IDX, LELB_IDX,
    RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX,
    LOWER_BODY_INDICES, NUM_FULL_BODY_JOINTS,
)
from models.model_factory import load_stage1, load_stage2, build_stage3_arm_model

WINDOW = 480
STRIDE = 240  # default; override via --stride
HEAD_INDICES = [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX]
ARM_INDICES = [RSHO_IDX, RELB_IDX, LSHO_IDX, LELB_IDX]
TORSO_INDICES = [C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX]


def windowed_inference(model, input_np, T, device, input_dim, output_dim, **model_kwargs):
    """Generic windowed inference with triangular blending."""
    pred_sum = np.zeros((T, output_dim), dtype=np.float64)
    weight_sum = np.zeros((T, 1), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, max(1, T - WINDOW + 1), STRIDE):
            e = min(s + WINDOW, T)
            al = e - s
            chunk = input_np[s:e]
            if len(chunk) < WINDOW:
                chunk = np.concatenate([chunk, np.zeros((WINDOW - len(chunk), input_dim), dtype=np.float32)])
            ct = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(ct, **model_kwargs).cpu().numpy()[0]
            w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
            pred_sum[s:e] += pred[:al] * w
            weight_sum[s:e] += w
    mask = weight_sum > 0
    return np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)


def compute_mean_head_offset(cache_dir):
    """Compute per-marker mean offset from C7 (Stage 2b)."""
    cache_train = Path(cache_dir) / 'train'
    all_offsets = []
    for f in sorted(cache_train.glob('*.npz')):
        d = np.load(f)
        j = d['joints'].astype(np.float32)
        c7 = j[:, C7_IDX]
        head = j[:, HEAD_INDICES, :]
        offset = head - c7[:, np.newaxis, :]
        all_offsets.append(offset.reshape(-1, 12))
    return np.concatenate(all_offsets, axis=0).mean(axis=0).reshape(4, 3)


def main():
    parser = argparse.ArgumentParser(description='Full Pipeline Inference + Evaluation')
    parser.add_argument('--stage1_ckpt', default='checkpoints/release/stage1_wrist/model.pt')
    parser.add_argument('--stage2_ckpt', default='checkpoints/release/stage2_backbone/model.pt',
                        help='Stage 2 backbone (MDM zscore, 8 joints = head+torso)')
    parser.add_argument('--stage3_ckpt', default='checkpoints/release/stage3_elbow/model.pt')
    parser.add_argument('--stage4_ckpt', default='checkpoints/release/stage4_lower/model.pt')
    parser.add_argument('--ddim_steps', type=int, default=45,
                        help='DDIM sampling steps for Stage 2 MDM (paper final: 45)')
    parser.add_argument('--arm_smooth_sigma', type=float, default=1.5,
                        help='Gaussian smoothing sigma for Stage 3 arm jitter (frames)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Override STRIDE for windowed inference (default: WINDOW//2)')
    parser.add_argument('--cache_dir', default='MOSA_dataset/training_cache_W05')
    parser.add_argument('--cache_dir_w10', default='MOSA_dataset/training_cache_W10')
    parser.add_argument('--output', default='results/full_pipeline.json')
    parser.add_argument('--save_motion', action='store_true', help='Save generated 26-joint motion to disk')
    parser.add_argument('--save_dir', default='MOSA_dataset/full_body_predictions')
    parser.add_argument('--npz_dir', default=None,
                        help='Save per-sample npz (sample_NNN.npz with keys generated/gt) '
                             'for cross-baseline comparison (e.g. experiments/benchmark/pianoid/results)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    global STRIDE
    if args.stride is not None:
        STRIDE = args.stride

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load models ──────────────────────────────────────────
    print('\nLoading models...')

    # Stage 1: MIDI → wrist (uses W05 cache for onset density features)
    s1_model, s1_config = load_stage1(args.stage1_ckpt, device)
    s1_midi_dim = 352 if s1_config.get('use_dur_ioi', False) else 264
    print(f'  Stage 1: {s1_config.get("model_type")} (midi_dim={s1_midi_dim})')

    # Stage 2: wrist → backbone (head 4 + torso 4 + shoulders 2 = 10 joints, 30-dim) via MDM zscore
    s2_ckpt = torch.load(args.stage2_ckpt, map_location='cpu', weights_only=False)
    s2_config = s2_ckpt.get('config', {})
    s2_config['backbone'] = True
    from models.model_factory import build_stage2_model
    s2_model = build_stage2_model(s2_config, device)
    s2_model.load_state_dict(s2_ckpt['model_state_dict']); s2_model.eval()
    s2_norm_stats = s2_ckpt.get('norm_stats', {})
    # Determine backbone output dim (24 legacy or 30 new)
    s2_out_dim = len(s2_norm_stats.get('torso_mean', [])) if s2_norm_stats else 30
    if s2_out_dim not in (24, 30):
        s2_out_dim = 30
    s2_mean = np.array(s2_norm_stats.get('torso_mean', np.zeros(s2_out_dim)), dtype=np.float32)
    s2_std = np.array(s2_norm_stats.get('torso_std', np.ones(s2_out_dim)), dtype=np.float32)
    s2_std[s2_std < 1e-8] = 1.0
    s2_repr = s2_config.get('repr', 'absolute')

    from diffusion_lib import create_diffusion
    from diffusion_lib.diffusion_utils import ddim_sample
    s2_diffusion = create_diffusion(num_timesteps=s2_config.get('diffusion_steps', 1000))
    print(f'  Stage 2: MDM backbone (repr={s2_repr}, out_dim={s2_out_dim}, DDIM {args.ddim_steps} steps)')

    # Stage 3: wrist+chest → arm
    s3_ckpt = torch.load(args.stage3_ckpt, map_location='cpu', weights_only=False)
    s3_config = s3_ckpt.get('config', {})
    s3_model = build_stage3_arm_model(s3_config, device)
    s3_model.load_state_dict(s3_ckpt['model_state_dict'])
    s3_model.eval()
    s3_relative = s3_config.get('relative_coords', False)
    s3_use_od = s3_config.get('use_onset_density', False)
    s3_use_pr = s3_config.get('use_pitch_range', False)
    print(f'  Stage 3: {s3_config.get("arch")} (relative={s3_relative}, od={s3_use_od}, pr={s3_use_pr})')

    # Stage 4: spine → lower body
    from stage4.s40_experiments import LowerFiLM
    s4_ckpt = torch.load(args.stage4_ckpt, map_location='cpu', weights_only=False)
    s4_model = LowerFiLM(input_dim=6, hidden_dim=128, num_layers=4).to(device)
    s4_model.load_state_dict(s4_ckpt['model_state_dict'])
    s4_model.eval()
    print(f'  Stage 4: FiLM spine-only')

    # ── Run pipeline on test set ─────────────────────────────
    print('\nRunning pipeline on test set...')
    cache_test_w05 = Path(args.cache_dir) / 'test'
    cache_test_w10 = Path(args.cache_dir_w10) / 'test'

    all_gt = []
    all_pred = []
    all_errors_per_joint = []

    if args.npz_dir is not None:
        Path(args.npz_dir).mkdir(parents=True, exist_ok=True)

    for sample_idx, f in enumerate(tqdm(sorted(cache_test_w05.glob('*.npz')), desc='Pipeline')):
        data_w05 = np.load(f)
        joints_gt = data_w05['joints'].astype(np.float32)  # (T, 26, 3)
        T = len(joints_gt)

        # Also load W10 data for onset density (Stage 2a/3/4 use W10)
        f_w10 = cache_test_w10 / f.name
        data_w10 = np.load(f_w10) if f_w10.exists() else data_w05

        # ── Stage 1: MIDI → wrist ────────────────────────────
        ak = data_w05['active_keys']
        pr = np.zeros((T, 88), dtype=np.float32)
        for t in range(T):
            for k in ak[t]:
                if 0 <= k < 88:
                    pr[t, k] = 1.0
        vel = data_w05.get('onset_density_per_key', np.zeros((T, 88))).astype(np.float32)
        ons = data_w05.get('key_onset', np.zeros((T, 88))).astype(np.float32)
        dur = data_w05.get('dur_ioi_ratio', np.zeros((T, 88))).astype(np.float32)

        parts = [pr, vel, ons]
        if s1_config.get('use_dur_ioi', False):
            parts.append(dur)
        midi_np = np.concatenate(parts, axis=-1)

        # Stage 1 needs midi_dict, not raw tensor
        pred_sum = np.zeros((T, 6), dtype=np.float64)
        weight_sum = np.zeros((T, 1), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, max(1, T - WINDOW + 1), STRIDE):
                e = min(s + WINDOW, T)
                chunk = midi_np[s:e]
                if len(chunk) < WINDOW:
                    chunk = np.concatenate([chunk, np.zeros((WINDOW - len(chunk), s1_midi_dim), dtype=np.float32)])
                ct = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
                md = {'piano_roll': ct[:, :, :88], 'velocity': ct[:, :, 88:176], 'onset': ct[:, :, 176:264]}
                if s1_config.get('use_dur_ioi', False):
                    md['dur_ioi_ratio'] = ct[:, :, 264:352]
                pred = s1_model(md).cpu().numpy()[0]
                al = e - s
                w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
                pred_sum[s:e] += pred[:al] * w
                weight_sum[s:e] += w
        mask = weight_sum > 0
        wrist_pred = np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)
        # wrist layout: [LWRIST(0:3), RWRIST(3:6)]

        # ── Stage 2: wrist → backbone via MDM zscore ──
        # Layout: head(0:12) + torso(12:24) + shoulders(24:30 if out_dim==30)
        od_w05 = data_w05.get('onset_density', np.zeros(T)).astype(np.float32).reshape(T, 1)
        od_w10 = data_w10.get('onset_density', np.zeros(T)).astype(np.float32).reshape(T, 1)
        pred_sum = np.zeros((T, s2_out_dim), dtype=np.float64)
        weight_sum = np.zeros((T, 1), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, max(1, T - WINDOW + 1), STRIDE):
                e = min(s + WINDOW, T)
                hand_chunk = wrist_pred[s:e]
                od_chunk = od_w05[s:e]
                if len(hand_chunk) < WINDOW:
                    hand_chunk = np.concatenate([hand_chunk, np.zeros((WINDOW - len(hand_chunk), 6), dtype=np.float32)])
                    od_chunk = np.concatenate([od_chunk, np.zeros((WINDOW - len(od_chunk), 1), dtype=np.float32)])
                ht = torch.tensor(hand_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                odt = torch.tensor(od_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                pred = ddim_sample(
                    s2_model, s2_diffusion, (1, WINDOW, s2_out_dim),
                    {'spatial_cond': ht, 'temporal_cond': odt},
                    ddim_steps=args.ddim_steps, device=device, progress=False,
                ).cpu().numpy()[0]
                al = e - s
                w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
                pred_sum[s:e] += pred[:al] * w
                weight_sum[s:e] += w
        mask = weight_sum > 0
        backbone_zscore = np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)
        if s2_repr == 'zscore':
            backbone_pred = backbone_zscore * s2_std + s2_mean
        else:
            backbone_pred = backbone_zscore

        head_pred = backbone_pred[:, 0:12].reshape(T, 4, 3)
        torso_pred = backbone_pred[:, 12:24]  # [C7, CLAV, T10, STRN] x 3
        if s2_out_dim == 30:
            rsho_pred = backbone_pred[:, 24:27]  # from Stage 2 backbone
            lsho_pred = backbone_pred[:, 27:30]
        else:
            rsho_pred = None
            lsho_pred = None

        # ── Stage 3: wrist + chest + shoulders (from S2) → elbows ──
        rwrist = wrist_pred[:, 3:6]
        lwrist = wrist_pred[:, 0:3]
        wrists = np.concatenate([rwrist, lwrist], axis=-1)  # (T, 6)
        chest = torso_pred[:, 6:12]  # T10+STRN

        s3_arm_dim = s3_config.get('arm_dim', 6)
        s3_spatial_dim = s3_config.get('spatial_dim', 18)

        if s3_spatial_dim == 18:
            # New: wrists(6) + chest(6) + shoulders(6) from Stage 2 backbone
            assert rsho_pred is not None, "Stage 2 must produce shoulders (out_dim=30) for new Stage 3"
            spatial_cond = np.concatenate([wrists, chest, rsho_pred, lsho_pred], axis=-1)  # (T, 18)
        else:
            # Legacy: wrists(6) + chest(6) = 12
            spatial_cond = np.concatenate([wrists, chest], axis=-1)  # (T, 12)

        s3_tc_parts = []
        if s3_use_od:
            s3_tc_parts.append(od_w10)
        if s3_use_pr:
            pr_w10 = data_w10.get('pitch_range', np.zeros(T)).astype(np.float32).reshape(T, 1)
            s3_tc_parts.append(pr_w10)
        s3_tc = np.concatenate(s3_tc_parts, axis=-1) if s3_tc_parts else None

        pred_sum = np.zeros((T, s3_arm_dim), dtype=np.float64)
        weight_sum = np.zeros((T, 1), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, max(1, T - WINDOW + 1), STRIDE):
                e = min(s + WINDOW, T)
                sc = spatial_cond[s:e]
                if len(sc) < WINDOW:
                    sc = np.concatenate([sc, np.zeros((WINDOW - len(sc), s3_spatial_dim), dtype=np.float32)])
                sct = torch.tensor(sc, dtype=torch.float32).unsqueeze(0).to(device)
                tct = None
                if s3_tc is not None:
                    tc_chunk = s3_tc[s:e]
                    if len(tc_chunk) < WINDOW:
                        tc_chunk = np.concatenate([tc_chunk, np.zeros((WINDOW - len(tc_chunk), s3_tc.shape[-1]), dtype=np.float32)])
                    tct = torch.tensor(tc_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                pred = s3_model(sct, temporal_cond=tct).cpu().numpy()[0]
                al = e - s
                w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
                pred_sum[s:e] += pred[:al] * w
                weight_sum[s:e] += w
        mask = weight_sum > 0
        s3_pred = np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)

        # Reconstruct arms: [RSHO, RELB, LSHO, LELB] x 3 = (T, 12)
        if s3_arm_dim == 6:
            # New: elbows only; shoulders from Stage 2 backbone
            if s3_relative:
                relb_abs = s3_pred[:, 0:3] + rsho_pred
                lelb_abs = s3_pred[:, 3:6] + lsho_pred
            else:
                relb_abs = s3_pred[:, 0:3]
                lelb_abs = s3_pred[:, 3:6]
            arm_pred = np.concatenate([rsho_pred, relb_abs, lsho_pred, lelb_abs], axis=-1)
        else:
            # Legacy: 4 joints from Stage 3
            if s3_relative:
                rsho = s3_pred[:, 0:3]
                relb_rel = s3_pred[:, 3:6]
                lsho = s3_pred[:, 6:9]
                lelb_rel = s3_pred[:, 9:12]
                arm_pred = np.concatenate([rsho, rsho + relb_rel, lsho, lsho + lelb_rel], axis=-1)
            else:
                arm_pred = s3_pred

        # Optional Gaussian temporal smoothing on ELBOWS ONLY
        # (shoulders come from Stage 2 backbone and must not be re-smoothed)
        if args.arm_smooth_sigma > 0 and s3_arm_dim == 6:
            from scipy.ndimage import gaussian_filter1d
            arm_r = arm_pred.reshape(-1, 4, 3)  # [rsho, relb, lsho, lelb]
            arm_r[:, 1, :] = gaussian_filter1d(arm_r[:, 1, :], sigma=args.arm_smooth_sigma, axis=0)
            arm_r[:, 3, :] = gaussian_filter1d(arm_r[:, 3, :], sigma=args.arm_smooth_sigma, axis=0)
            arm_pred = arm_r.reshape(-1, 12)
        elif args.arm_smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            arm_pred = gaussian_filter1d(arm_pred, sigma=args.arm_smooth_sigma, axis=0)

        # ── Stage 4: spine → lower body ─────────────────────
        spine = np.concatenate([torso_pred[:, 0:3], torso_pred[:, 6:9]], axis=-1)  # C7+T10
        pred_sum = np.zeros((T, 30), dtype=np.float64)
        weight_sum = np.zeros((T, 1), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, max(1, T - WINDOW + 1), STRIDE):
                e = min(s + WINDOW, T)
                sp = spine[s:e]
                if len(sp) < WINDOW:
                    sp = np.concatenate([sp, np.zeros((WINDOW - len(sp), 6), dtype=np.float32)])
                spt = torch.tensor(sp, dtype=torch.float32).unsqueeze(0).to(device)
                pred = s4_model(spt).cpu().numpy()[0]
                al = e - s
                w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
                pred_sum[s:e] += pred[:al] * w
                weight_sum[s:e] += w
        mask = weight_sum > 0
        lower_pred = np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)

        # ── Assemble 26-joint skeleton ───────────────────────
        full_pred = np.zeros((T, 26, 3), dtype=np.float32)

        # Head (4)
        full_pred[:, HEAD_INDICES] = head_pred

        # Torso (4)
        torso_r = torso_pred.reshape(T, 4, 3)
        for i, idx in enumerate(TORSO_INDICES):
            full_pred[:, idx] = torso_r[:, i]

        # Arms (4)
        arm_r = arm_pred.reshape(T, 4, 3)
        for i, idx in enumerate(ARM_INDICES):
            full_pred[:, idx] = arm_r[:, i]

        # Wrists (2)
        full_pred[:, LWRIST_IDX] = wrist_pred[:, 0:3]
        full_pred[:, RWRIST_IDX] = wrist_pred[:, 3:6]

        # Lower body (10)
        lower_r = lower_pred.reshape(T, 10, 3)
        for i, idx in enumerate(LOWER_BODY_INDICES):
            full_pred[:, idx] = lower_r[:, i]

        # ── Save generated motion ────────────────────────────
        if args.save_motion:
            save_dir = Path(args.save_dir) / 'test' / f.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / 'generated_motion.npy', full_pred)
            np.save(save_dir / 'gt_motion.npy', joints_gt)

        # ── Cross-baseline compatible npz dump ───────────────
        if args.npz_dir is not None:
            np.savez(
                Path(args.npz_dir) / f'sample_{sample_idx:03d}.npz',
                generated=full_pred.astype(np.float32),
                gt=joints_gt.astype(np.float32),
            )

        # ── Compute errors ───────────────────────────────────
        per_joint_err = np.sqrt(((joints_gt - full_pred) ** 2).sum(axis=-1))  # (T, 26)
        all_errors_per_joint.append(per_joint_err)
        all_gt.append(joints_gt)
        all_pred.append(full_pred)

    # ── Aggregate metrics ────────────────────────────────────
    from evaluation.metrics import (
        compute_jitter, compute_velocity_correlation,
        compute_fid_from_motions,
    )

    all_errors = np.concatenate(all_errors_per_joint, axis=0)  # (N, 26)
    overall_mpjpe = all_errors.mean() * 1000

    # Per-body-part MPJPE — shoulders come from Stage 2 backbone, elbows from Stage 3
    body_parts = {
        'Wrists (S1)': [LWRIST_IDX, RWRIST_IDX],
        'Head (S2)': HEAD_INDICES,
        'Torso (S2)': TORSO_INDICES,
        'Shoulders (S2)': [RSHO_IDX, LSHO_IDX],
        'Elbows (S3)': [RELB_IDX, LELB_IDX],
        'Pelvis (S4)': LOWER_BODY_INDICES[:4],
        'Legs (S4)': LOWER_BODY_INDICES[4:],
    }

    part_results = {}
    for name, indices in body_parts.items():
        part_results[name] = all_errors[:, indices].mean() * 1000

    # FID
    print('\nComputing FID...')
    fid = compute_fid_from_motions(all_gt, all_pred)

    # Jitter (mean jerk magnitude)
    print('Computing Jitter...')
    gt_jitters = [compute_jitter(m) for m in all_gt]
    pred_jitters = [compute_jitter(m) for m in all_pred]
    gt_jitter_mean = np.mean(gt_jitters)
    pred_jitter_mean = np.mean(pred_jitters)
    jitter_delta = abs(pred_jitter_mean - gt_jitter_mean)

    # Velocity Correlation
    print('Computing Velocity Correlation...')
    vel_corrs = [compute_velocity_correlation(g, p) for g, p in zip(all_gt, all_pred)]
    vel_corr_mean = np.mean(vel_corrs)

    # ── Print results ────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Full Pipeline Results (26 joints, 80 test sequences)')
    print(f'{"="*60}')
    print(f'Overall MPJPE:       {overall_mpjpe:.1f}mm')
    print(f'FID:                 {fid:.4f}')
    print(f'Jitter (GT):         {gt_jitter_mean:.6f}')
    print(f'Jitter (Pred):       {pred_jitter_mean:.6f}')
    print(f'|ΔJitter|:           {jitter_delta:.6f} ({jitter_delta*1e5:.1f} ×10⁻⁵)')
    print(f'Velocity Corr:       {vel_corr_mean:.4f}')
    print(f'\nPer-body-part MPJPE:')
    for name, mpjpe in sorted(part_results.items(), key=lambda x: x[1]):
        print(f'  {name:<20} {mpjpe:.1f}mm')

    # ── Save results ─────────────────────────────────────────
    results = {
        'overall_mpjpe': float(overall_mpjpe),
        'fid': float(fid),
        'jitter_gt': float(gt_jitter_mean),
        'jitter_pred': float(pred_jitter_mean),
        'jitter_delta': float(jitter_delta),
        'jitter_delta_1e5': float(jitter_delta * 1e5),
        'vel_corr': float(vel_corr_mean),
        'per_part': {k: float(v) for k, v in part_results.items()},
        'per_joint': {str(i): float(all_errors[:, i].mean() * 1000) for i in range(26)},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
