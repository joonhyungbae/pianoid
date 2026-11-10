#!/usr/bin/env python3
"""
Rebuild MOSA_dataset/splits.json to match the paper's 158/34/80 partition.

Paper (§5.1): performer-disjoint except for two hard pieces (be1, tc1).
  - Train (158): yp01-yp05 × all pieces, MINUS yp04/yp05 × {be1, tc1}
  - Val   (34): yp06 × all pieces
  - Test  (80): (yp07 + yp08) × all pieces + yp04/yp05 × {be1, tc1}

Count check:
  - Each performer has 10 pieces × 3 trials + 4 pieces × 1 trial = 34 samples.
  - Train = 34 × 3 (yp01-yp03) + (34 - 6) × 2 (yp04/yp05 without be1/tc1) = 158
  - Val   = 34
  - Test  = 34 + 34 + 6 + 6 = 80

Usage:
    python scripts/preprocess/build_splits.py                     # default paths
    python scripts/preprocess/build_splits.py --mosa_root PATH    # custom MOSA root
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOSA = ROOT / "MOSA_dataset"

HARD_PIECES = {"be1", "tc1"}
TEST_HARD_PERFORMERS = {"yp04", "yp05"}  # their hard pieces get routed to test
TEST_PERFORMERS = {"yp07", "yp08"}        # all their pieces go to test
VAL_PERFORMER = "yp06"
TRAIN_PERFORMERS = {"yp01", "yp02", "yp03", "yp04", "yp05"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mosa_root", type=Path, default=DEFAULT_MOSA,
                   help="MOSA_dataset root containing the original splits.json")
    p.add_argument("--backup_name", default="splits.mosa_default.json",
                   help="File name to back up the current splits.json to")
    p.add_argument("--no_backup", action="store_true",
                   help="Skip backing up the original (already backed up)")
    args = p.parse_args()

    src = args.mosa_root / "splits.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")

    raw = json.load(open(src))

    # Aggregate all samples from whatever the current partition is
    all_samples = []
    for split, items in raw["samples"].items():
        if isinstance(items, list):
            all_samples.extend(items)

    # Deduplicate by (piece, performer, trial)
    seen, unique = set(), []
    for s in all_samples:
        key = (s["piece"], s["performer"], s["trial"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)

    train, val, test = [], [], []
    for s in unique:
        performer, piece = s["performer"], s["piece"]
        if performer in TEST_PERFORMERS:
            test.append(s)
        elif performer in TEST_HARD_PERFORMERS and piece in HARD_PIECES:
            test.append(s)
        elif performer == VAL_PERFORMER:
            val.append(s)
        elif performer in TRAIN_PERFORMERS:
            train.append(s)
        else:
            print(f"WARNING: unassigned sample {s}")

    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}  total: {len(unique)}")
    assert len(train) == 158, f"expected 158 train, got {len(train)}"
    assert len(val) == 34,    f"expected 34 val, got {len(val)}"
    assert len(test) == 80,   f"expected 80 test, got {len(test)}"

    key = lambda s: (s["piece"], s["performer"], s["trial"])
    out = {"samples": {"train": sorted(train, key=key),
                       "val":   sorted(val,   key=key),
                       "test":  sorted(test,  key=key)}}

    if not args.no_backup:
        backup = args.mosa_root / args.backup_name
        if not backup.exists():
            backup.write_text(src.read_text())
            print(f"Backed up original to {backup.name}")

    src.write_text(json.dumps(out, indent=2))
    print(f"Wrote paper-aligned splits to {src}")


if __name__ == "__main__":
    main()
