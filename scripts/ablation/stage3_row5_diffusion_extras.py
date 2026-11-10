#!/usr/bin/env python3
"""Compute Row 5 (Anatomical + Predicted-train + Diffusion) metrics for Table 5(B).

Loads the MDM checkpoint trained at:
    checkpoints/stage3_arm_mdm_anatomical_predtrain/mdm_rel_od_pr_h256_l6_lr0.0001/best.pt

Computes:
  - MPJPE on elbows only (RELB, LELB) in absolute 3D
  - L/R asymmetry = |MPJPE_LELB - MPJPE_RELB|
  - Z-rng (per-axis range ratio: predicted Z range / GT Z range), elbow-only
  - ER-jerk (mean |3rd-deriv| ratio: predicted / GT), elbow-only

These mirror the columns in Table 5(B): MPJPE / Z-rng / L-R asym (mm) / ER-jerk.

Usage:
    python scripts/ablation/stage3_row5_diffusion_extras.py
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
from glob import glob

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'vendor' / 'MDM'))

from data.constants import (
    RSHO_IDX, RELB_IDX, LSHO_IDX, LELB_IDX,
    RWRIST_IDX, LWRIST_IDX, T10_IDX, STRN_IDX,
)
from models.model_factory import build_stage3_arm_model
from diffusion_lib import create_diffusion, ddim_sample

ARM_INDICES = [RSHO_IDX, RELB_IDX, LSHO_IDX, LELB_IDX]


def load_test_data_anatomical(cache_dir, hand_dir, torso_dir, config):
    """Load test data with shoulders included in spatial_cond (18-dim).

    Mirrors the ArmDataset _load_all_data logic for anatomical mode.
    """
    use_od = config.get('use_onset_density', False)
    use_pr = config.get('use_pitch_range', False)
    cache_test = Path(cache_dir) / 'test'
    hand_test = Path(hand_dir) / 'test'
    torso_test = Path(torso_dir) / 'test'

    out = []
    for f in sorted(cache_test.glob('*.npz')):
        data = np.load(f)
        joints = data['joints'].astype(np.float32)
        T = len(joints)
        arm_gt = joints[:, ARM_INDICES, :].reshape(T, 12)

        # Wrists from phase0
        hp = hand_test / f.name
        if hp.exists():
            hd = np.load(hp)
            hp_data = hd.get('hand_pred', hd.get('pred')).astype(np.float32)
            wrists = np.concatenate([hp_data[:, 3:6], hp_data[:, 0:3]], axis=-1)  # [RWRIST, LWRIST]
        else:
            wrists = np.concatenate([joints[:, RWRIST_IDX], joints[:, LWRIST_IDX]], axis=-1).astype(np.float32)

        # Chest + shoulders from phase1 backbone
        tp = torso_test / f.name
        rsho_in, lsho_in = None, None
        if tp.exists():
            td = np.load(tp)
            tp_data = td.get('torso_pred').astype(np.float32)
            if tp_data.shape[-1] >= 30:
                chest = tp_data[:, 18:24]   # T10+STRN
                rsho_in = tp_data[:, 24:27]
                lsho_in = tp_data[:, 27:30]
            else:
                chest = tp_data[:, 6:12] if tp_data.shape[-1] >= 12 else joints[:, [T10_IDX, STRN_IDX], :].reshape(T, 6).astype(np.float32)
                rsho_in = joints[:, RSHO_IDX].astype(np.float32)
                lsho_in = joints[:, LSHO_IDX].astype(np.float32)
        else:
            chest = joints[:, [T10_IDX, STRN_IDX], :].reshape(T, 6).astype(np.float32)
            rsho_in = joints[:, RSHO_IDX].astype(np.float32)
            lsho_in = joints[:, LSHO_IDX].astype(np.float32)

        spatial_cond = np.concatenate([wrists, chest, rsho_in, lsho_in], axis=-1).astype(np.float32)
        assert spatial_cond.shape[-1] == 18

        aria = data.get('aria_embed', np.zeros(512)).astype(np.float32)
        if aria.ndim == 2:
            aria = aria.mean(axis=0)

        tc_parts = []
        if use_od:
            tc_parts.append(data.get('onset_density', np.zeros(T)).astype(np.float32).reshape(T, 1))
        if use_pr:
            tc_parts.append(data.get('pitch_range', np.zeros(T)).astype(np.float32).reshape(T, 1))
        od = np.concatenate(tc_parts, axis=-1) if tc_parts else None

        out.append((arm_gt, spatial_cond, T, aria, od, rsho_in, lsho_in))
    return out

CKPT_PATTERN = str(REPO / 'checkpoints/stage3_arm_mdm_anatomical_predtrain/*/best.pt')
CACHE_DIR = str(REPO / 'MOSA_dataset/training_cache_W10')
HAND_DIR = str(REPO / 'MOSA_dataset/phase0_hand_predictions')
TORSO_DIR = str(REPO / 'MOSA_dataset/phase1_torso_predictions')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WINDOW = 480
STRIDE = 240
DDIM_STEPS = 25  # Best DDIM count for diffusion variant (sweep on val showed flat performance from 25, this is the most charitable to diffusion)


def windowed_mdm_anatomical(model, diffusion, spatial_np, T, device, aria=None,
                            use_aria=False, od_np=None, output_dim=6,
                            spatial_dim=18):
    """DDIM sample for anatomical (6-dim) MDM with optional temporal cond."""
    pred_sum = np.zeros((T, output_dim), dtype=np.float64)
    weight_sum = np.zeros((T, 1), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, max(1, T - WINDOW + 1), STRIDE):
            e = min(s + WINDOW, T)
            al = e - s
            chunk = spatial_np[s:e]
            if len(chunk) < WINDOW:
                chunk = np.concatenate([chunk, np.zeros((WINDOW - len(chunk), spatial_dim), dtype=np.float32)])
            ct = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            aria_t = None
            if use_aria and aria is not None:
                aria_t = torch.tensor(aria, dtype=torch.float32).unsqueeze(0).to(device)
            tc_t = None
            if od_np is not None:
                od_chunk = od_np[s:e]
                tc_dim = od_np.shape[-1]
                if len(od_chunk) < WINDOW:
                    od_chunk = np.concatenate([od_chunk, np.zeros((WINDOW - len(od_chunk), tc_dim), dtype=np.float32)])
                tc_t = torch.tensor(od_chunk, dtype=torch.float32).unsqueeze(0).to(device)
            shape = (1, WINDOW, output_dim)
            kwargs = {'spatial_cond': ct, 'global_cond': aria_t}
            if tc_t is not None:
                kwargs['temporal_cond'] = tc_t
            sample = ddim_sample(
                model, diffusion, shape,
                model_kwargs=kwargs,
                ddim_steps=DDIM_STEPS, device=str(device), progress=False,
            )
            pred = sample.cpu().numpy()[0]
            w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
            pred_sum[s:e] += pred[:al] * w
            weight_sum[s:e] += w
    mask = weight_sum > 0
    return np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)


def main():
    ckpt_files = glob(CKPT_PATTERN)
    assert len(ckpt_files) >= 1, f'No ckpt at {CKPT_PATTERN}'
    ckpt_path = sorted(ckpt_files)[-1]
    print(f'Loading: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    print(f'  arch={config.get("arch")}, relative_coords={config.get("relative_coords")}, '
          f'arm_dim={config.get("arm_dim")}, spatial_dim={config.get("spatial_dim")}')

    model = build_stage3_arm_model(config, DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    diffusion = create_diffusion(num_timesteps=1000)

    test_data = load_test_data_anatomical(CACHE_DIR, HAND_DIR, TORSO_DIR, config)
    print(f'Loaded {len(test_data)} test samples')

    per_seq = []
    for arm_gt, spatial_cond, T, aria, od, rsho, lsho in tqdm(test_data, desc='Eval'):
        assert spatial_cond.shape[-1] == 18, f'Expected 18-dim spatial (anatomical), got {spatial_cond.shape}'

        pred_rel = windowed_mdm_anatomical(
            model, diffusion, spatial_cond, T, DEVICE,
            aria=aria, use_aria=config.get('use_aria', False),
            od_np=od, output_dim=6, spatial_dim=18,
        )
        # pred_rel: (T, 6) [RELB_rel(0:3), LELB_rel(3:6)] — relative to predicted shoulders
        relb_abs = pred_rel[:, 0:3] + rsho  # (T, 3)
        lelb_abs = pred_rel[:, 3:6] + lsho

        # GT arms: arm_gt is (T, 12) [RSHO, RELB, LSHO, LELB] absolute
        gt = arm_gt.reshape(T, 4, 3)
        gt_relb = gt[:, 1, :]
        gt_lelb = gt[:, 3, :]

        # MPJPE per elbow
        mpjpe_relb = float(np.mean(np.linalg.norm(relb_abs - gt_relb, axis=-1)) * 1000)
        mpjpe_lelb = float(np.mean(np.linalg.norm(lelb_abs - gt_lelb, axis=-1)) * 1000)
        mpjpe_elbows = (mpjpe_relb + mpjpe_lelb) / 2

        # Z range ratio (vertical lift)
        def z_rng(arr): return float(arr[:, 2].max() - arr[:, 2].min())
        z_rng_relb = z_rng(relb_abs) / max(z_rng(gt_relb), 1e-6)
        z_rng_lelb = z_rng(lelb_abs) / max(z_rng(gt_lelb), 1e-6)
        z_rng_mean = (z_rng_relb + z_rng_lelb) / 2

        # ER-jerk (3rd deriv ratio per axis, then mean)
        def jerk(arr):
            j = np.diff(arr, n=3, axis=0)
            return np.abs(j).mean(axis=0)  # (3,)
        if T > 4:
            j_relb_p = jerk(relb_abs); j_relb_g = jerk(gt_relb)
            j_lelb_p = jerk(lelb_abs); j_lelb_g = jerk(gt_lelb)
            jerk_ratio = float(np.mean(np.where(j_relb_g > 1e-9, j_relb_p / j_relb_g, np.nan)))
            jerk_ratio_lelb = float(np.mean(np.where(j_lelb_g > 1e-9, j_lelb_p / j_lelb_g, np.nan)))
            jerk_mean = float(np.nanmean([jerk_ratio, jerk_ratio_lelb]))
        else:
            jerk_mean = float('nan')

        per_seq.append({
            'mpjpe_elbows': mpjpe_elbows,
            'mpjpe_relb': mpjpe_relb,
            'mpjpe_lelb': mpjpe_lelb,
            'z_rng_mean': z_rng_mean,
            'jerk_mean': jerk_mean,
        })

    def agg(key):
        vals = [s[key] for s in per_seq if not np.isnan(s[key])]
        return float(np.mean(vals))

    summary = {
        'n_seq': len(per_seq),
        'MPJPE_elbows': agg('mpjpe_elbows'),
        'MPJPE_RELB': agg('mpjpe_relb'),
        'MPJPE_LELB': agg('mpjpe_lelb'),
        'LR_asym_elbows': abs(agg('mpjpe_relb') - agg('mpjpe_lelb')),
        'Z_rng_mean': agg('z_rng_mean'),
        'Jerk_ratio_mean': agg('jerk_mean'),
    }
    print('\n=== Row 5 (Anatomical + Predicted-train + Diffusion) ===')
    for k, v in summary.items():
        print(f'  {k:<20} {v:.3f}')

    out = REPO / 'checkpoints/stage3_arm_mdm_anatomical_predtrain/row5_extras.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
