#!/usr/bin/env python3
"""
Organize trained Pianoid checkpoints into the canonical release layout.

Canonical layout under `checkpoints/release/`:

    stage1_wrist/
        model.pt     # torch dict: model_state_dict, config, val_mpjpe_mm, ...
        README.md    # auto-generated config + metrics summary
    stage2_backbone/
        model.pt     # + norm_stats (z-score mean/std)
        README.md
    stage3_elbow/
        model.pt
        README.md
    stage4_lower/
        model.pt
        README.md

Usage:
    python src/project_utils/release_checkpoints.py                  # dry-run
    python src/project_utils/release_checkpoints.py --mode copy      # copy files
    python src/project_utils/release_checkpoints.py --mode symlink   # symlink
"""
import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

# (canonical_name, source_relative_path, paper_section)
MAPPING = [
    (
        "stage1_wrist",
        "checkpoints/stage1/hand_bilstm_best.pt",
        "§4-S1: MIDI → wrists (BiLSTM h512 l6, v3 loss)",
    ),
    (
        "stage2_backbone",
        "checkpoints/stage2_backbone/mdm_od_h256_l8_lr0.0001_rng0.5_std0.5_zscore/best.pt",
        "§4-S2: wrists → 10-joint backbone (MDM zscore h256 l8, DDIM 45 at inference)",
    ),
    (
        "stage3_elbow",
        "checkpoints/stage3_arm/bilstm_rel_od_pr_h256_l6_lr0.0001/best.pt",
        "§4-S3: wrists+chest+shoulders → elbows (BiLSTM h256 l6, shoulder-relative, predicted-train)",
    ),
    (
        "stage4_lower",
        "checkpoints/stage4_experiments/film_spine/best.pt",
        "§4-S4: spine → lower body (FiLM h128 l4)",
    ),
]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def render_readme(name: str, paper_section: str, ckpt: dict, sha: str, ts: str) -> str:
    cfg = ckpt.get("config", {})
    mpjpe = ckpt.get("val_mpjpe_mm", "n/a")
    epoch = ckpt.get("epoch", "n/a")
    has_norm = "norm_stats" in ckpt
    # Keep config compact — exclude private keys and ckpt-internal paths
    clean = {
        k: v for k, v in cfg.items()
        if not k.startswith("_") and k not in ("cache_dir", "pred_dir", "output_dir", "wandb_project")
    }
    lines = [
        f"# {name}",
        "",
        f"**Paper section**: {paper_section}",
        "",
        f"- Val MPJPE: **{mpjpe}** mm",
        f"- Trained for {epoch} epochs (early-stopped on val MPJPE)",
        f"- Git SHA: `{sha}`",
        f"- Built: {ts}",
        f"- Contains norm_stats: {has_norm}",
        "",
        "## Training config",
        "",
        "```json",
        json.dumps(clean, indent=2, default=str),
        "```",
        "",
        "## Loading",
        "",
        "```python",
        "import torch",
        f"ckpt = torch.load('checkpoints/release/{name}/model.pt',",
        "                  map_location='cpu', weights_only=False)",
        "# Keys: model_state_dict, config, val_mpjpe_mm, epoch"
        + (", norm_stats" if has_norm else ""),
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode", choices=["dryrun", "copy", "symlink"], default="dryrun",
        help="dryrun reports only; copy duplicates the .pt; symlink creates a link",
    )
    p.add_argument("--release-root", default="checkpoints/release")
    args = p.parse_args()

    release_root = ROOT / args.release_root
    release_root.mkdir(parents=True, exist_ok=True)

    sha = git_sha()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for name, src_rel, paper_sec in MAPPING:
        src = ROOT / src_rel
        out_dir = release_root / name

        if not src.exists():
            print(f"[SKIP] {name:18s} (source missing: {src_rel})")
            continue

        out_dir.mkdir(exist_ok=True)
        dst = out_dir / "model.pt"

        if args.mode == "dryrun":
            size_mb = src.stat().st_size / 1e6
            print(f"[DRY ] {name:18s} <- {src_rel} ({size_mb:.1f} MB)")
            continue

        ckpt = torch.load(src, map_location="cpu", weights_only=False)

        if args.mode == "copy":
            # Strip optimizer/scheduler state so release files stay small.
            keep = {k: v for k, v in ckpt.items()
                    if k not in {"optimizer_state_dict", "scheduler_state_dict"}}
            torch.save(keep, dst)
        elif args.mode == "symlink":
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

        readme = render_readme(name, paper_sec, ckpt, sha, ts)
        (out_dir / "README.md").write_text(readme)
        print(f"[{args.mode.upper():4s}] {name:18s} <- {src_rel}")

    print(f"\nRelease root: {release_root.relative_to(ROOT)}")
    if args.mode == "dryrun":
        print("Dry run — rerun with --mode copy or --mode symlink to finalize.")


if __name__ == "__main__":
    main()
