#!/usr/bin/env python3
"""
s21_precompute_torso.py: Precompute torso predictions for Stage 3 training.

Loads the trained Stage 2 model (architecture auto-detected from checkpoint config)
and generates torso predictions for all splits. These predictions are used as inputs
for Stage 3 ArmMDM, ensuring distribution matching between training and inference.

Pipeline:
    phase0_hand_predictions + ARIA + pitch_range + onset_density
        -> TorsoTransformer -> phase1_torso_predictions

Output per .npz:
    torso_pred:  (T, 12) generated torso [C7, CLAV, T10, STRN]
    torso_gt:    (T, 12) ground-truth torso
    hand_pred:   (T, 6) wrist predictions used as input
    sample_idx:  int

Usage:
    python src/stage2/s21_precompute_torso.py
    python src/stage2/s21_precompute_torso.py --torso_path checkpoints/stage2_torso/best.pt
    bash scripts/train/stage2.sh  # trains + precomputes
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'vendor' / 'MDM'))

from data.constants import (
    C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
    RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX,
    RSHO_IDX, LSHO_IDX,
    RWRIST_IDX, LWRIST_IDX, RHAND_IDX, LHAND_IDX,
)
from models.model_factory import load_stage2

TORSO_INDICES = [C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX]
HEAD_INDICES = [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX]
SHOULDER_INDICES = [RSHO_IDX, LSHO_IDX]
BACKBONE_INDICES_24 = HEAD_INDICES + TORSO_INDICES               # 8 joints (legacy)
BACKBONE_INDICES_30 = HEAD_INDICES + TORSO_INDICES + SHOULDER_INDICES  # 10 joints
WINDOW_SIZE = 480
STRIDE = 240


def load_model(ckpt_path: str, device):
    """Load torso model from checkpoint via centralized factory."""
    ckpt_raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model, config = load_stage2(ckpt_path, device)
    repr_mode = config.get('repr', 'absolute')
    norm_stats = ckpt_raw.get('norm_stats', None)
    use_hand_vel = config.get('use_hand_vel', config.get('hand_velocity', False))
    is_backbone = config.get('backbone', False)
    is_mdm = (config.get('arch') == 'mdm')

    # Determine output dim
    if is_backbone:
        if norm_stats and 'torso_mean' in norm_stats:
            out_dim = len(norm_stats['torso_mean'])
        else:
            out_dim = 30
    else:
        out_dim = 12

    print(f"Loaded {config.get('arch', '?')} from {ckpt_path}")
    print(f"  config: h={config.get('latent_dim')} l={config.get('num_layers')} "
          f"aria={config.get('use_aria')} od={config.get('use_onset_density')} "
          f"repr={repr_mode} hand_vel={use_hand_vel} backbone={is_backbone} out_dim={out_dim}")
    return model, config, repr_mode, norm_stats, use_hand_vel, is_mdm, out_dim


def generate_torso_for_sample(
    joints_gt: np.ndarray,
    hand_pred: np.ndarray,
    aria_embed: np.ndarray,
    pitch_range: np.ndarray,
    onset_density: np.ndarray,
    onset_density_per_key: np.ndarray,
    model,
    config: dict,
    device,
    repr_mode: str = 'absolute',
    norm_stats: dict = None,
    use_hand_vel: bool = False,
    is_mdm: bool = False,
    diffusion=None,
    ddim_steps: int = 25,
    out_dim: int = 12,
):
    """Generate torso/backbone predictions using windowed inference with triangular blending."""
    T = joints_gt.shape[0]

    # Build GT for the right output dim
    if out_dim == 30:
        torso_gt = joints_gt[:, BACKBONE_INDICES_30, :].reshape(T, 30).astype(np.float32)
    elif out_dim == 24:
        torso_gt = joints_gt[:, BACKBONE_INDICES_24, :].reshape(T, 24).astype(np.float32)
    else:
        torso_gt = joints_gt[:, TORSO_INDICES, :].reshape(T, 12).astype(np.float32)

    use_aria = config.get('use_aria', False)
    use_pitch_range = config.get('use_pitch_range', False)
    use_onset_density = config.get('use_onset_density', False)
    use_onset_per_key = config.get('use_onset_per_key', False)
    aria_is_temporal = aria_embed.ndim == 2

    pred_sum = np.zeros((T, out_dim), dtype=np.float64)
    weight_sum = np.zeros((T, 1), dtype=np.float64)

    # Build hand velocity if needed
    hand_vel = None
    if use_hand_vel:
        hand_vel = np.zeros_like(hand_pred)
        hand_vel[1:] = hand_pred[1:] - hand_pred[:-1]

    with torch.no_grad():
        for s in range(0, max(1, T - WINDOW_SIZE + 1), STRIDE):
            e = min(s + WINDOW_SIZE, T)
            win_len = e - s
            if win_len < 60:
                continue

            if is_mdm:
                # MDM: spatial_cond=hand, temporal_cond=onset_density (separate args)
                hand_chunk = hand_pred[s:e]
                if len(hand_chunk) < WINDOW_SIZE:
                    hand_chunk = np.concatenate([
                        hand_chunk, np.zeros((WINDOW_SIZE - len(hand_chunk), hand_chunk.shape[-1]), np.float32)
                    ])
                ht = torch.from_numpy(hand_chunk).float().unsqueeze(0).to(device)

                tc_parts = []
                if use_onset_density:
                    tc_parts.append(onset_density[s:e].reshape(-1, 1))
                if use_pitch_range:
                    tc_parts.append(pitch_range[s:e].reshape(-1, 1))
                if use_onset_per_key:
                    tc_parts.append(onset_density_per_key[s:e])
                tct = None
                if tc_parts:
                    tc = np.concatenate(tc_parts, axis=-1).astype(np.float32)
                    if len(tc) < WINDOW_SIZE:
                        tc = np.concatenate([tc, np.zeros((WINDOW_SIZE - len(tc), tc.shape[-1]), np.float32)])
                    tct = torch.from_numpy(tc).unsqueeze(0).to(device)

                aria_t = None
                if use_aria:
                    if aria_is_temporal:
                        aria_idx = min(s // STRIDE, aria_embed.shape[0] - 1)
                        aria = aria_embed[aria_idx]
                    else:
                        aria = aria_embed
                    aria_t = torch.from_numpy(aria.copy()).float().unsqueeze(0).to(device)

                from diffusion_lib.diffusion_utils import ddim_sample
                shape = (1, WINDOW_SIZE, out_dim)
                pred = ddim_sample(
                    model, diffusion, shape,
                    {'spatial_cond': ht, 'temporal_cond': tct, 'global_cond': aria_t},
                    ddim_steps=ddim_steps, device=device, progress=False,
                ).cpu().numpy()[0]
            else:
                # Regression: hand + temporal_cond concatenated
                inp_parts = [hand_pred[s:e]]
                if use_hand_vel and hand_vel is not None:
                    inp_parts.append(hand_vel[s:e])
                if use_onset_density:
                    inp_parts.append(onset_density[s:e].reshape(-1, 1))
                if use_pitch_range:
                    inp_parts.append(pitch_range[s:e].reshape(-1, 1))
                if use_onset_per_key:
                    inp_parts.append(onset_density_per_key[s:e])
                inp = np.concatenate(inp_parts, axis=-1).astype(np.float32)
                inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)

                aria_t = None
                if use_aria:
                    if aria_is_temporal:
                        aria_idx = min(s // STRIDE, aria_embed.shape[0] - 1)
                        aria = aria_embed[aria_idx]
                    else:
                        aria = aria_embed
                    aria_t = torch.from_numpy(aria.copy()).float().unsqueeze(0).to(device)

                pred = model(inp_t, aria=aria_t).cpu().numpy()[0]

            # Triangular blending
            w = np.minimum(
                np.arange(win_len) + 1,
                np.arange(win_len, 0, -1),
            ).astype(np.float64).reshape(-1, 1)

            pred_sum[s:e] += pred[:win_len] * w
            weight_sum[s:e] += w

    mask = weight_sum > 0
    torso_pred = np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)

    # Denormalize repr
    if repr_mode == 'zscore' and norm_stats is not None:
        mean = np.array(norm_stats['torso_mean'], dtype=np.float32)
        std = np.array(norm_stats['torso_std'], dtype=np.float32)
        std[std < 1e-8] = 1.0
        torso_pred = torso_pred * std + mean
    elif repr_mode == 'residual' and norm_stats is not None:
        mean = np.array(norm_stats['torso_mean'], dtype=np.float32)
        torso_pred = torso_pred + mean
    elif repr_mode == 'delta':
        abs_pred = np.zeros_like(torso_pred)
        abs_pred[0] = torso_gt[0] + torso_pred[0]
        for t in range(1, len(torso_pred)):
            abs_pred[t] = abs_pred[t-1] + torso_pred[t]
        torso_pred = abs_pred

    return torso_pred, torso_gt


def main():
    parser = argparse.ArgumentParser(description='Precompute torso predictions')
    parser.add_argument('--cache_dir', default='MOSA_dataset/training_cache_B')
    parser.add_argument('--pred_dir', default='MOSA_dataset/phase0_hand_predictions')
    parser.add_argument('--output_dir', default='MOSA_dataset/phase1_torso_predictions')
    parser.add_argument('--torso_path', default='checkpoints/stage2_torso/best.pt')
    parser.add_argument('--ddim_steps', type=int, default=25,
                        help='DDIM sampling steps for MDM models')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)

    model, config, repr_mode, norm_stats, use_hand_vel, is_mdm, out_dim = load_model(args.torso_path, device)
    diffusion = None
    if is_mdm:
        from diffusion_lib import create_diffusion
        diffusion = create_diffusion(num_timesteps=config.get('diffusion_steps', 1000))
        print(f"  Using DDIM with {args.ddim_steps} steps")

    for split in ['train', 'val', 'test']:
        cache_split = cache_dir / split
        pred_split = pred_dir / split
        out_split = output_dir / split

        if not cache_split.exists():
            print(f"Skipping {split}")
            continue

        out_split.mkdir(parents=True, exist_ok=True)
        cache_files = sorted(cache_split.glob('*.npz'))
        print(f"\n{split}: {len(cache_files)} samples -> {out_split}")

        for cache_path in tqdm(cache_files, desc=split):
            out_path = out_split / cache_path.name

            try:
                cache_data = np.load(cache_path)
                joints = cache_data['joints'].astype(np.float32)
                aria_embed = cache_data['aria_embed'].astype(np.float32)
                T = joints.shape[0]

                pitch_range = cache_data.get('pitch_range', np.zeros(T)).astype(np.float32)
                onset_density = cache_data.get('onset_density', np.zeros(T)).astype(np.float32)
                onset_density_per_key = cache_data.get(
                    'onset_density_per_key', np.zeros((T, 88))).astype(np.float32)

                # Load hand predictions
                hand_path = pred_split / cache_path.name
                if hand_path.exists():
                    pd = np.load(hand_path)
                    hand_pred = pd.get('hand_pred', pd.get('pred'))
                    if hand_pred is not None:
                        hand_pred = hand_pred.astype(np.float32)
                    else:
                        hand_pred = None
                else:
                    hand_pred = None

                if hand_pred is None:
                    hand_pred = np.concatenate([
                        joints[:, LWRIST_IDX],
                        joints[:, RWRIST_IDX],
                    ], axis=-1).astype(np.float32)

                torso_pred, torso_gt = generate_torso_for_sample(
                    joints, hand_pred, aria_embed,
                    pitch_range, onset_density, onset_density_per_key,
                    model, config, device,
                    repr_mode=repr_mode,
                    norm_stats=norm_stats,
                    use_hand_vel=use_hand_vel,
                    is_mdm=is_mdm,
                    diffusion=diffusion,
                    ddim_steps=args.ddim_steps,
                    out_dim=out_dim,
                )

                sample_idx = int(cache_path.stem)
                np.savez_compressed(
                    out_path,
                    torso_pred=torso_pred,
                    torso_gt=torso_gt,
                    hand_pred=hand_pred,
                    sample_idx=sample_idx,
                )
            except Exception as e:
                print(f"Error {cache_path.name}: {e}")
                import traceback; traceback.print_exc()

    print(f"\nDone! Predictions saved to {output_dir}")
    print("Next: bash scripts/train/stage3.sh")


if __name__ == '__main__':
    main()
