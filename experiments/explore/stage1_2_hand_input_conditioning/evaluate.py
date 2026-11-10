#!/usr/bin/env python3
"""
Explore 02: Evaluate input conditioning sweep results.
Loads best checkpoints for each condition and computes test metrics.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

_THIS = Path(__file__).resolve().parent
ROOT = _THIS.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(_THIS))

from dataset import get_hand_aria_loader
from model import HandMambaInputCond

CONDITIONS = ["midi_only", "aria_only", "midi_aria"]
MAMBA_CONFIG = {"hidden_dim": 512, "num_layers": 4, "d_state": 16}

# Z-axis (key-press direction) indices: lwrist_z=2, lhand_z=5, rwrist_z=8, rhand_z=11
HAND_Z_INDICES = (2, 5, 8, 11)
M_TO_CM = 100.0


def compute_mpjpe(pred, gt):
    pred = pred.reshape(-1, 4, 3)
    gt = gt.reshape(-1, 4, 3)
    return np.sqrt(((pred - gt) ** 2).sum(axis=-1)).mean() * 1000


def compute_mpjpe_z(pred, gt):
    """Z-axis-only MPJPE (mm). pred, gt: (B, T, 12) or (T, 12)."""
    pred = np.asarray(pred).reshape(-1, 12)
    gt = np.asarray(gt).reshape(-1, 12)
    err = np.abs(pred[:, HAND_Z_INDICES] - gt[:, HAND_Z_INDICES])
    return float(err.mean() * 1000.0)


def wrist_z_range_cm(hand):
    """Wrist Z range (cm). hand: (..., 12)."""
    hand = np.asarray(hand).reshape(-1, 12)
    lwrist_z = hand[:, 2]
    rwrist_z = hand[:, 8]
    z = np.concatenate([lwrist_z, rwrist_z])
    r = (z.max() - z.min()) * M_TO_CM
    return {"min_cm": z.min() * M_TO_CM, "max_cm": z.max() * M_TO_CM, "range_cm": r}


def compute_onset_alignment(pred, gt, onset):
    pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    onset = onset.reshape(-1, onset.shape[-2], onset.shape[-1])
    pred_vel = np.linalg.norm(pred[:, 1:] - pred[:, :-1], axis=-1)
    onset_mask = (onset[:, 1:].sum(axis=-1) > 0).astype(float)
    corrs = []
    for i in range(len(pred_vel)):
        if onset_mask[i].sum() > 0 and pred_vel[i].std() > 0:
            c = np.corrcoef(pred_vel[i], onset_mask[i])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
    return np.mean(corrs) if corrs else 0.0


def compute_jitter(pred):
    pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    vel = pred[:, 1:] - pred[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    return np.linalg.norm(acc, axis=-1).mean() * 1e4


def compute_velocity_correlation(pred, velocity):
    pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    velocity = velocity.reshape(-1, velocity.shape[-2], velocity.shape[-1])
    motion_amp = np.linalg.norm(pred[:, 1:] - pred[:, :-1], axis=-1)
    midi_vel = velocity[:, 1:].mean(axis=-1)
    corrs = []
    for i in range(len(motion_amp)):
        if midi_vel[i].std() > 0 and motion_amp[i].std() > 0:
            c = np.corrcoef(motion_amp[i], midi_vel[i])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
    return np.mean(corrs) if corrs else 0.0


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", MAMBA_CONFIG)
    model = HandMambaInputCond(
        hidden_dim=cfg.get("hidden_dim", 512),
        num_layers=cfg.get("num_layers", 4),
        d_state=cfg.get("d_state", 16),
        force_fallback=(device != "cuda"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("condition", "unknown")


@torch.no_grad()
def evaluate_condition(model, condition, test_loader, device):
    all_pred, all_gt, all_onset, all_vel = [], [], [], []
    for batch in tqdm(test_loader, desc=f"Eval {condition}"):
        hand_gt = batch["hand"].to(device)
        midi_features = {k: v.to(device) for k, v in batch["midi_features"].items()}
        aria_embed = batch["aria_embed"].to(device)
        if condition == "midi_only":
            aria_embed = torch.zeros_like(aria_embed)
        elif condition == "aria_only":
            midi_features = {k: torch.zeros_like(v) for k, v in midi_features.items()}
        pred = model(midi_features, aria_embed)
        all_pred.append(pred.cpu().numpy())
        all_gt.append(hand_gt.cpu().numpy())
        all_onset.append(batch["midi_features"]["onset"].numpy())
        all_vel.append(batch["midi_features"]["velocity"].numpy())
    pred = np.concatenate(all_pred, 0)
    gt = np.concatenate(all_gt, 0)
    onset = np.concatenate(all_onset, 0)
    velocity = np.concatenate(all_vel, 0)
    gt_z = wrist_z_range_cm(gt)
    gen_z = wrist_z_range_cm(pred)
    z_ratio = gen_z["range_cm"] / (gt_z["range_cm"] + 1e-8)
    return {
        "mpjpe_mm": float(compute_mpjpe(pred, gt)),
        "z_mpjpe_mm": float(compute_mpjpe_z(pred, gt)),
        "wrist_z_range_gt_cm": float(gt_z["range_cm"]),
        "wrist_z_range_gen_cm": float(gen_z["range_cm"]),
        "wrist_z_range_ratio": float(z_ratio),
        "onset_alignment": float(compute_onset_alignment(pred, gt, onset)),
        "jitter_1e4": float(compute_jitter(pred)),
        "velocity_correlation": float(compute_velocity_correlation(pred, velocity)),
    }


def main():
    ap = argparse.ArgumentParser(description="Explore 02: Evaluate Input Conditioning")
    ap.add_argument("--checkpoint_dir", default="checkpoints/02_input_conditioning")
    ap.add_argument("--cache_dir", default="MOSA_dataset/training_cache")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb_project", default="hand-input-conditioning")
    args = ap.parse_args()

    ckpt_root = ROOT / args.checkpoint_dir if not Path(args.checkpoint_dir).is_absolute() else Path(args.checkpoint_dir)
    cache_dir = str(ROOT / args.cache_dir) if not Path(args.cache_dir).is_absolute() else args.cache_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    no_wandb = getattr(args, "no_wandb", False)
    use_wandb = (not no_wandb) and HAS_WANDB

    test_split = "test" if (Path(cache_dir) / "test").exists() else "val"
    test_loader = get_hand_aria_loader(
        cache_dir, test_split, batch_size=args.batch_size,
        shuffle=False, num_workers=4, preload=True,
    )

    if use_wandb:
        wandb.init(project=args.wandb_project, name="02_evaluation",
                   config={"checkpoint_dir": str(ckpt_root)})

    print("=" * 70)
    print("Explore 02: Input Conditioning Evaluation")
    print("=" * 70)

    all_metrics = {}
    for cond in CONDITIONS:
        ckpt_path = ckpt_root / cond / "best.pt"
        if not ckpt_path.exists():
            print(f"  [{cond}] checkpoint not found, skipping")
            continue
        model, _ = load_model(ckpt_path, device)
        metrics = evaluate_condition(model, cond, test_loader, device)
        all_metrics[cond] = metrics
        print(f"\n  [{cond}]")
        print(f"    MPJPE:      {metrics['mpjpe_mm']:.2f} mm")
        print(f"    Z MPJPE:    {metrics['z_mpjpe_mm']:.2f} mm  (key-press direction)")
        print(f"    Wrist Z:    GT range {metrics['wrist_z_range_gt_cm']:.1f} cm, Gen {metrics['wrist_z_range_gen_cm']:.1f} cm, ratio {metrics['wrist_z_range_ratio']:.3f}")
        print(f"    Onset Align: {metrics['onset_alignment']:.4f}")
        print(f"    Jitter:      {metrics['jitter_1e4']:.2f} (x1e-4)")
        print(f"    Vel Corr:    {metrics['velocity_correlation']:.4f}")
        if use_wandb:
            for k, v in metrics.items():
                wandb.log({f"test/{cond}/{k}": v})

    print(f"\n{'='*70}")
    print(f"{'Condition':>12s}  {'MPJPE(mm)':>10s}  {'Z_MPJPE':>10s}  {'Z_ratio':>8s}  {'Onset':>8s}  {'Jitter':>8s}  {'VelCorr':>8s}")
    print("-" * 70)
    for cond, m in all_metrics.items():
        print(f"{cond:>12s}  {m['mpjpe_mm']:10.2f}  {m['z_mpjpe_mm']:10.2f}  {m['wrist_z_range_ratio']:8.3f}  {m['onset_alignment']:8.4f}  {m['jitter_1e4']:8.2f}  {m['velocity_correlation']:8.4f}")
    print("=" * 70)

    out_path = ckpt_root / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if use_wandb:
        wandb.summary.update({"all_metrics": all_metrics})
        table = wandb.Table(columns=[
            "condition", "mpjpe_mm", "z_mpjpe_mm", "wrist_z_range_ratio",
            "onset_alignment", "jitter_1e4", "velocity_correlation",
        ])
        for cond, m in all_metrics.items():
            table.add_data(
                cond, m["mpjpe_mm"], m["z_mpjpe_mm"], m["wrist_z_range_ratio"],
                m["onset_alignment"], m["jitter_1e4"], m["velocity_correlation"],
            )
        wandb.log({"evaluation_table": table})
        wandb.finish()


if __name__ == "__main__":
    main()
