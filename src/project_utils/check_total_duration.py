#!/usr/bin/env python3
"""Compute total dataset duration from MOSA_dataset (training_cache or metadata)."""
import json
import numpy as np
from pathlib import Path

FPS = 60  # from src/data/constants.py

def main():
    data_root = Path("MOSA_dataset")
    cache_dir = data_root / "training_cache"
    metadata_path = data_root / "metadata.json"
    splits_path = data_root / "splits.json"

    # 1) From metadata.json
    with open(metadata_path) as f:
        meta = json.load(f)
    info = meta["dataset_info"]
    total_sec_meta = info["total_duration_sec"]
    total_min_meta = total_sec_meta / 60
    print("=== metadata.json ===")
    print(f"  total_duration_sec: {total_sec_meta}")
    print(f"  total_duration_min: {total_min_meta:.2f} ({int(total_min_meta)} min {int(total_sec_meta % 60)} sec)")
    print(f"  total_duration_hours: {info['total_duration_hours']}")
    print(f"  total_samples: {info['total_samples']}")

    # 2) From training_cache npz (actual frames)
    total_frames = 0
    by_split = {"train": 0, "val": 0, "test": 0}
    for split in ["train", "val", "test"]:
        split_dir = cache_dir / split
        if not split_dir.exists():
            continue
        for npz_path in sorted(split_dir.glob("*.npz")):
            data = np.load(npz_path)
            T = data["joints"].shape[0]
            total_frames += T
            by_split[split] += T
    total_sec_from_frames = total_frames / FPS
    total_min_from_frames = total_sec_from_frames / 60
    print("\n=== training_cache (npz frames) ===")
    print(f"  total_frames: {total_frames}")
    print(f"  FPS: {FPS}")
    print(f"  total_sec: {total_sec_from_frames:.2f}")
    print(f"  total_min: {total_min_from_frames:.2f} ({int(total_min_from_frames)} min {int(total_sec_from_frames % 60)} sec)")
    for k, v in by_split.items():
        print(f"  {k}: {v} frames = {v/FPS:.1f} sec = {v/FPS/60:.2f} min")

    # 3) From pieces (metadata)
    pieces = meta.get("pieces", {})
    sum_piece_sec = sum(p["total_duration_sec"] for p in pieces.values())
    print("\n=== pieces sum (metadata) ===")
    print(f"  sum total_duration_sec: {sum_piece_sec}")
    print(f"  sum total_min: {sum_piece_sec/60:.2f}")

    # Final answer
    print("\n=== Conclusion (total playtime) ===")
    # Prefer metadata as single source of truth (covers full dataset)
    total_sec = total_sec_meta
    total_min = total_sec / 60
    print(f"  Total length: {total_sec:.0f} sec = {total_min:.2f} min = {total_min/60:.2f} h")
    print(f"  Exact: {int(total_min)} min {int(total_sec % 60)} sec")

if __name__ == "__main__":
    main()
