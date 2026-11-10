#!/usr/bin/env python3
"""
Explore 02: Input Conditioning Sweep

Trains HandMambaInputCond (best Mamba from Explore 01) with three input conditions:
  midi_only, aria_only, midi_aria.

Usage:
    python experiments/explore/stage1_2_hand_input_conditioning/train_sweep.py --condition all
    python experiments/explore/stage1_2_hand_input_conditioning/train_sweep.py --condition midi_only
    python experiments/explore/stage1_2_hand_input_conditioning/train_sweep.py --epochs 1 --test
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from models.hand_baselines import compute_hand_regression_loss

CONDITIONS = ["midi_only", "aria_only", "midi_aria"]
MAMBA_CONFIG = {"hidden_dim": 512, "num_layers": 4, "d_state": 16}


def prepare_batch(batch, device, condition):
    """Move to device and apply input masking for the given condition."""
    hand_gt = batch["hand"].to(device)
    midi_features = {k: v.to(device) for k, v in batch["midi_features"].items()}
    aria_embed = batch["aria_embed"].to(device)
    if condition == "midi_only":
        aria_embed = torch.zeros_like(aria_embed)
    elif condition == "aria_only":
        midi_features = {k: torch.zeros_like(v) for k, v in midi_features.items()}
    return hand_gt, midi_features, aria_embed


def train_epoch(model, loader, optimizer, device, scaler, condition, use_amp, z_weight=1.0):
    model.train()
    total_loss, total_pos, total_vel = 0.0, 0.0, 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        hand_gt, midi_features, aria_embed = prepare_batch(batch, device, condition)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(midi_features, aria_embed)
            losses = compute_hand_regression_loss(
                pred, hand_gt, midi_features["onset"],
                vel_weight=0.5, onset_weight=2.0,
                z_weight=z_weight,
            )
            loss = losses["loss"]
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_pos += losses["pos_loss"].item()
        total_vel += losses["vel_loss"].item()
    n = len(loader)
    return {"loss": total_loss / n, "pos_loss": total_pos / n, "vel_loss": total_vel / n}


@torch.no_grad()
def validate(model, loader, device, condition, z_weight=1.0):
    model.eval()
    total_loss, total_pos, total_vel = 0.0, 0.0, 0.0
    for batch in loader:
        hand_gt, midi_features, aria_embed = prepare_batch(batch, device, condition)
        pred = model(midi_features, aria_embed)
        losses = compute_hand_regression_loss(pred, hand_gt, z_weight=z_weight)
        total_loss += losses["loss"].item()
        total_pos += losses["pos_loss"].item()
        total_vel += losses["vel_loss"].item()
    n = len(loader)
    return {"loss": total_loss / n, "pos_loss": total_pos / n, "vel_loss": total_vel / n}


def main():
    ap = argparse.ArgumentParser(description="Explore 02: Input Conditioning Sweep")
    ap.add_argument("--cache_dir", default="MOSA_dataset/training_cache")
    ap.add_argument("--output_dir", default="checkpoints/02_input_conditioning")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--condition", choices=CONDITIONS + ["all"], default="all")
    ap.add_argument("--z_weight", type=float, default=2.0,
                    help="Position-loss weight on the Z axis (key-press direction). 1.0 = same as XY; 2-3 recommended.")
    ap.add_argument("--test", action="store_true", help="Quick test (1 epoch, bs=4)")
    ap.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    ap.add_argument("--wandb_project", default="hand-input-conditioning")
    args = ap.parse_args()

    if args.test:
        args.epochs = 1
        args.batch_size = 4

    out_root = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    cache_dir = str(ROOT / args.cache_dir) if not Path(args.cache_dir).is_absolute() else args.cache_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    conditions = CONDITIONS if args.condition == "all" else [args.condition]
    no_wandb = getattr(args, "no_wandb", False)
    use_wandb = (not no_wandb) and HAS_WANDB
    if (not no_wandb) and (not HAS_WANDB):
        print("Warning: wandb not installed. Continuing without wandb.")
    val_split = "val" if (Path(cache_dir) / "val").exists() else "test"

    z_weight = getattr(args, "z_weight", 1.0)
    print("=" * 60)
    print("Explore 02: Input Conditioning Sweep")
    print("=" * 60)
    print(f"Device: {device}  |  Conditions: {conditions}")
    print(f"Mamba config: {MAMBA_CONFIG}")
    print(f"Z-axis weight: {z_weight} (emphasizes the key-press direction)")
    print(f"Epochs: {args.epochs}  Patience: {args.patience}  LR: {args.lr}  BS: {args.batch_size}")
    print(f"wandb: {'ON' if use_wandb else 'OFF'}")
    print("=" * 60)

    summary = {}

    for cond in conditions:
        out_dir = out_root / cond
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\nCondition: {cond}\n{'='*60}")

        train_loader = get_hand_aria_loader(
            cache_dir, "train", batch_size=args.batch_size,
            shuffle=True, num_workers=4, preload=True,
        )
        val_loader = get_hand_aria_loader(
            cache_dir, val_split, batch_size=args.batch_size,
            shuffle=False, num_workers=4, preload=True,
        )

        model = HandMambaInputCond(
            hidden_dim=MAMBA_CONFIG["hidden_dim"],
            num_layers=MAMBA_CONFIG["num_layers"],
            d_state=MAMBA_CONFIG["d_state"],
            force_fallback=(device != "cuda"),
        ).to(device)
        num_params = sum(pp.numel() for pp in model.parameters())
        print(f"Parameters: {num_params:,}")

        # wandb init per condition run
        if use_wandb:
            wandb.init(
                project=args.wandb_project,
                group="input_conditioning_sweep",
                name=f"02_{cond}",
                config={
                    "condition": cond,
                    "z_weight": z_weight,
                    "num_params": num_params,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    **MAMBA_CONFIG,
                },
                reinit=True,
            )

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best_val = float("inf")
        patience_count = 0

        for epoch in range(args.epochs):
            tm = train_epoch(model, train_loader, optimizer, device, scaler, cond, use_amp, z_weight=z_weight)
            vm = validate(model, val_loader, device, cond, z_weight=z_weight)
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"train={tm['loss']:.4f} (pos={tm['pos_loss']:.4f} vel={tm['vel_loss']:.4f})  "
                f"val={vm['loss']:.4f}  lr={lr_now:.2e}"
            )

            # wandb per-step log
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": tm["loss"],
                    "train/pos_loss": tm["pos_loss"],
                    "train/vel_loss": tm["vel_loss"],
                    "val/loss": vm["loss"],
                    "val/pos_loss": vm["pos_loss"],
                    "val/vel_loss": vm["vel_loss"],
                    "lr": lr_now,
                    "best_val_loss": min(best_val, vm["loss"]),
                })

            if vm["loss"] < best_val:
                best_val = vm["loss"]
                patience_count = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val,
                    "condition": cond,
                    "config": MAMBA_CONFIG,
                    "num_params": num_params,
                }, out_dir / "best.pt")
                print(f"  -> best saved (val={best_val:.4f})")
            else:
                patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stop at epoch {epoch+1}")
                break

        result = {
            "condition": cond,
            "best_val_loss": best_val,
            "z_weight": z_weight,
            "num_params": num_params,
            "config": MAMBA_CONFIG,
            "checkpoint": str(out_dir / "best.pt"),
            "timestamp": datetime.now().isoformat(),
        }
        with open(out_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        summary[cond] = result

        # wandb finish per condition
        if use_wandb:
            wandb.summary["best_val_loss"] = best_val
            wandb.summary["num_params"] = num_params
            wandb.finish()

    # Save overall summary
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}\nSweep Complete!\n{'='*60}")
    print(f"Summary: {summary_path}")
    for c, r in summary.items():
        print(f"  {c:12s}  val_loss={r['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
