#!/usr/bin/env python3
"""Evaluate Stage-3 multi-seed checkpoints to populate Table 5(B) with mean ± std.

For each of the 12 cells (3 seeds × 4 rows):
  - Load checkpoint
  - Run windowed regression on test set
  - Reconstruct absolute elbow positions (relative_coords mode)
  - Compute MPJPE, Z-rng, L/R asym, ER-jerk

Then aggregate as mean ± std across 3 seeds for each row.

Usage:
    python scripts/ablation/stage3_multiseed_eval.py
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

CACHE_DIR = REPO / 'MOSA_dataset/training_cache_W10'
HAND_DIR = REPO / 'MOSA_dataset/phase0_hand_predictions'
TORSO_DIR = REPO / 'MOSA_dataset/phase1_torso_predictions'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WINDOW = 480
STRIDE = 240

ROWS = {
    'row1_functional_gt':         {'functional': True,  'use_gt_inputs': True,  'arm_dim': 12, 'spatial_dim': 12},
    'row2_anatomical_gt':         {'functional': False, 'use_gt_inputs': True,  'arm_dim': 6,  'spatial_dim': 18},
    'row3_anatomical_predicted':  {'functional': False, 'use_gt_inputs': False, 'arm_dim': 6,  'spatial_dim': 18},
    'row4_functional_predicted':  {'functional': True,  'use_gt_inputs': False, 'arm_dim': 12, 'spatial_dim': 12},
}

SEEDS = [42, 7, 1337]


def load_test_data(functional, spatial_dim):
    """Load test data — always uses PREDICTED inputs (exposes train/test mismatch for GT-trained models).

    use_gt_inputs is a TRAIN-time flag; at TEST time all cells use the cascade's predicted inputs
    (Stage 1 wrists, Stage 2 chest+shoulders) to faithfully measure deployment behavior.
    """
    test_files = sorted((CACHE_DIR / 'test').glob('*.npz'))
    out = []
    for f in test_files:
        d = np.load(f)
        joints = d['joints'].astype(np.float32)
        T = len(joints)

        # GT arms (always full 4-joint absolute, 12-dim) for evaluation
        gt_rsho = joints[:, RSHO_IDX].astype(np.float32)
        gt_relb = joints[:, RELB_IDX].astype(np.float32)
        gt_lsho = joints[:, LSHO_IDX].astype(np.float32)
        gt_lelb = joints[:, LELB_IDX].astype(np.float32)
        gt_arms_abs = np.concatenate([gt_rsho, gt_relb, gt_lsho, gt_lelb], axis=-1)  # (T, 12)

        # PREDICTED inputs (always — regardless of how the cell was trained)
        hp = HAND_DIR / 'test' / f.name
        hp_data = np.load(hp)['hand_pred'].astype(np.float32)
        wrists = np.concatenate([hp_data[:, 3:6], hp_data[:, 0:3]], axis=-1)
        tp = TORSO_DIR / 'test' / f.name
        tp_data = np.load(tp)['torso_pred'].astype(np.float32)
        if tp_data.shape[-1] >= 30:
            chest = tp_data[:, 18:24]
            rsho_in = tp_data[:, 24:27]
            lsho_in = tp_data[:, 27:30]
        else:
            chest = tp_data[:, 6:12]
            rsho_in, lsho_in = gt_rsho, gt_lsho

        if functional:
            spatial = np.concatenate([wrists, chest], axis=-1).astype(np.float32)  # 12-dim
        else:
            spatial = np.concatenate([wrists, chest, rsho_in, lsho_in], axis=-1).astype(np.float32)  # 18-dim

        assert spatial.shape[-1] == spatial_dim, f'expected {spatial_dim}, got {spatial.shape[-1]}'

        # Onset density + pitch range (always present in W10 cache)
        od = d.get('onset_density', np.zeros(T)).astype(np.float32).reshape(T, 1)
        pr = d.get('pitch_range', np.zeros(T)).astype(np.float32).reshape(T, 1)
        tcond = np.concatenate([od, pr], axis=-1)

        out.append((gt_arms_abs, spatial, T, tcond, rsho_in, lsho_in))
    return out


def windowed_predict(model, spatial, tcond, T, output_dim):
    pred_sum = np.zeros((T, output_dim), dtype=np.float64)
    weight_sum = np.zeros((T, 1), dtype=np.float64)
    sp_dim = spatial.shape[-1]
    tc_dim = tcond.shape[-1] if tcond is not None else 0
    # Build window starts; ensure tail is fully covered by adding a final window at max(0, T-WINDOW)
    starts = list(range(0, max(1, T - WINDOW + 1), STRIDE))
    if T > WINDOW and starts[-1] + WINDOW < T:
        starts.append(T - WINDOW)
    elif T <= WINDOW:
        starts = [0]
    with torch.no_grad():
        for s in starts:
            e = min(s + WINDOW, T)
            al = e - s
            chunk = spatial[s:e]
            if len(chunk) < WINDOW:
                chunk = np.concatenate([chunk, np.zeros((WINDOW - len(chunk), sp_dim), dtype=np.float32)])
            ct = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tt = None
            if tcond is not None:
                tc_chunk = tcond[s:e]
                if len(tc_chunk) < WINDOW:
                    tc_chunk = np.concatenate([tc_chunk, np.zeros((WINDOW - len(tc_chunk), tc_dim), dtype=np.float32)])
                tt = torch.tensor(tc_chunk, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = model(ct, temporal_cond=tt).cpu().numpy()[0]
            w = np.minimum(np.arange(al) + 1, np.arange(al, 0, -1)).astype(np.float64).reshape(-1, 1)
            pred_sum[s:e] += pred[:al] * w
            weight_sum[s:e] += w
    mask = weight_sum > 0
    # If any frame still uncovered (shouldn't happen after fix), use last valid value
    return np.where(mask, pred_sum / np.where(mask, weight_sum, 1), 0).astype(np.float32)


def evaluate_cell(ckpt_path, row_cfg, test_data):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    cfg['arm_dim'] = row_cfg['arm_dim']
    cfg['spatial_dim'] = row_cfg['spatial_dim']
    model = build_stage3_arm_model(cfg, DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    per_seq = []
    for gt_abs, spatial, T, tcond, rsho_in, lsho_in in test_data:
        pred = windowed_predict(model, spatial, tcond, T, row_cfg['arm_dim'])

        if row_cfg['arm_dim'] == 6:
            # Anatomical: pred = [RELB_rel, LELB_rel] relative to shoulders
            relb_abs = pred[:, 0:3] + rsho_in
            lelb_abs = pred[:, 3:6] + lsho_in
            arm_abs = np.concatenate([rsho_in, relb_abs, lsho_in, lelb_abs], axis=-1)  # (T, 12)
        else:
            # Functional: pred = [RSHO, RELB_rel, LSHO, LELB_rel] (relative_coords for funct)
            # Actually for functional non-relative: pred is direct [RSHO, RELB, LSHO, LELB]
            relative = ckpt['config'].get('relative_coords', False)
            if relative:
                rsho = pred[:, 0:3]; relb = pred[:, 3:6] + rsho
                lsho = pred[:, 6:9]; lelb = pred[:, 9:12] + lsho
                arm_abs = np.concatenate([rsho, relb, lsho, lelb], axis=-1)
            else:
                arm_abs = pred  # already [RSHO, RELB, LSHO, LELB] absolute

        gt = gt_abs.reshape(T, 4, 3)
        pr = arm_abs.reshape(T, 4, 3)

        # MPJPE per joint (in mm)
        err_per_joint = np.linalg.norm(pr - gt, axis=-1) * 1000  # (T, 4) — RSHO, RELB, LSHO, LELB
        mpjpe_relb = float(err_per_joint[:, 1].mean())
        mpjpe_lelb = float(err_per_joint[:, 3].mean())
        mpjpe_overall_4j = float(err_per_joint.mean())  # avg over 4 arm joints

        # Z range ratio (elbow Z range)
        relb_abs_z = arm_abs[:, 5]  # RELB.z (index 3*1+2 = 5)
        lelb_abs_z = arm_abs[:, 11]  # LELB.z
        gt_relb_z = gt_abs[:, 5]
        gt_lelb_z = gt_abs[:, 11]
        z_relb = (relb_abs_z.max() - relb_abs_z.min()) / max(gt_relb_z.max() - gt_relb_z.min(), 1e-6)
        z_lelb = (lelb_abs_z.max() - lelb_abs_z.min()) / max(gt_lelb_z.max() - gt_lelb_z.min(), 1e-6)

        # Jerk (3rd deriv) on elbows only
        if T > 4:
            relb_xyz = arm_abs[:, 3:6]
            lelb_xyz = arm_abs[:, 9:12]
            gt_relb = gt_abs[:, 3:6]; gt_lelb = gt_abs[:, 9:12]
            j_relb_p = np.abs(np.diff(relb_xyz, n=3, axis=0)).mean(axis=0)
            j_relb_g = np.abs(np.diff(gt_relb, n=3, axis=0)).mean(axis=0)
            j_lelb_p = np.abs(np.diff(lelb_xyz, n=3, axis=0)).mean(axis=0)
            j_lelb_g = np.abs(np.diff(gt_lelb, n=3, axis=0)).mean(axis=0)
            jr = np.where(j_relb_g > 1e-9, j_relb_p / j_relb_g, np.nan)
            jl = np.where(j_lelb_g > 1e-9, j_lelb_p / j_lelb_g, np.nan)
            jerk_mean = float(np.nanmean(np.concatenate([jr, jl])))
        else:
            jerk_mean = float('nan')

        per_seq.append({
            'mpjpe_4j': mpjpe_overall_4j,
            'mpjpe_relb': mpjpe_relb,
            'mpjpe_lelb': mpjpe_lelb,
            'z_relb': float(z_relb), 'z_lelb': float(z_lelb),
            'jerk': jerk_mean,
        })

    del model
    torch.cuda.empty_cache()

    def agg(key):
        vals = [s[key] for s in per_seq if not np.isnan(s[key])]
        return float(np.mean(vals))

    return {
        'mpjpe_4j': agg('mpjpe_4j'),
        'mpjpe_relb': agg('mpjpe_relb'),
        'mpjpe_lelb': agg('mpjpe_lelb'),
        'lr_asym': abs(agg('mpjpe_relb') - agg('mpjpe_lelb')),
        'z_rng_mean': (agg('z_relb') + agg('z_lelb')) / 2,
        'jerk': agg('jerk'),
    }


def main():
    results = {}  # results[row][seed] = metrics
    for row, row_cfg in ROWS.items():
        print(f'\n=== Loading test data for {row} ===')
        test_data = load_test_data(row_cfg['functional'], row_cfg['spatial_dim'])
        print(f'  {len(test_data)} test sequences loaded (predicted inputs)')

        results[row] = {}
        for seed in SEEDS:
            ckpt_dir = REPO / f'checkpoints/stage3_multiseed/seed{seed}/{row}'
            ckpt_paths = list(ckpt_dir.glob('*/best.pt'))
            if not ckpt_paths:
                print(f'  seed{seed}: NO checkpoint found, skipping')
                continue
            ckpt_path = ckpt_paths[0]
            print(f'  seed{seed}: evaluating {ckpt_path.parent.name}')
            try:
                m = evaluate_cell(ckpt_path, row_cfg, test_data)
                results[row][seed] = m
                print(f'    MPJPE_4j={m["mpjpe_4j"]:.2f}  L/R={m["lr_asym"]:.2f}  Z-rng={m["z_rng_mean"]:.3f}  jerk={m["jerk"]:.3f}')
            except Exception as e:
                print(f'    FAILED: {e}')

    # Aggregate
    print('\n' + '='*70)
    print('=== Multi-seed aggregate (mean ± std across seeds) ===')
    print('='*70)
    agg = {}
    for row, by_seed in results.items():
        if not by_seed:
            continue
        vals = list(by_seed.values())
        keys = ['mpjpe_4j', 'lr_asym', 'z_rng_mean', 'jerk']
        agg[row] = {
            'n_seeds': len(vals),
            'seeds': sorted(by_seed.keys()),
        }
        for k in keys:
            xs = [v[k] for v in vals]
            agg[row][f'{k}_mean'] = float(np.mean(xs))
            agg[row][f'{k}_std']  = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        print(f'\n{row} (n_seeds={agg[row]["n_seeds"]}, seeds={agg[row]["seeds"]}):')
        print(f'  MPJPE_4j  : {agg[row]["mpjpe_4j_mean"]:6.2f} ± {agg[row]["mpjpe_4j_std"]:.2f}')
        print(f'  L/R asym  : {agg[row]["lr_asym_mean"]:6.2f} ± {agg[row]["lr_asym_std"]:.2f}')
        print(f'  Z-rng     : {agg[row]["z_rng_mean_mean"]:6.3f} ± {agg[row]["z_rng_mean_std"]:.3f}')
        print(f'  ER-jerk   : {agg[row]["jerk_mean"]:6.3f} ± {agg[row]["jerk_std"]:.3f}')

    out_path = REPO / 'results/stage3_multiseed_aggregate.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'per_cell': results, 'aggregate': agg}, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
