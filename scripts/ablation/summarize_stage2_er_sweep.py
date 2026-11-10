#!/usr/bin/env python3
"""
Summarize results/er_mdm_zs_*.json files side-by-side against the paper
targets. Marks each cell as "↑", "↓", or "≈" relative to the paper value.

Usage:
    python scripts/ablation/summarize_stage2_er_sweep.py
    python scripts/ablation/summarize_stage2_er_sweep.py --results_dir results
"""
import argparse
import json
from pathlib import Path

PAPER_TARGET = {
    "mpjpe": 93.5,
    "ER_range": 1.22,
    "ER_speed": 0.93,
    "ER_jerk": 0.58,
    "WZR": 0.85,
    "AccR": 0.62,
    "Delta1": 0.27,
}

# For "closer to 1 is better" metrics, compare |x-1| to |paper-1|.
# For MPJPE / Delta1, lower is better.
LOWER_IS_BETTER = {"mpjpe", "Delta1"}


def better_arrow(key, val, paper):
    if key in LOWER_IS_BETTER:
        return "↑" if val < paper else ("≈" if abs(val - paper) < 0.02 else "↓")
    # target = 1
    return "↑" if abs(val - 1) < abs(paper - 1) else ("≈" if abs(val - paper) < 0.01 else "↓")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--glob", default="er_mdm_zs_*.json")
    args = p.parse_args()

    files = sorted(args.results_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matching {args.results_dir}/{args.glob}")

    rows = []
    for f in files:
        d = json.loads(f.read_text())
        rows.append((d.get("run_tag") or f.stem, d["metrics"]))

    keys = ["mpjpe", "ER_range", "ER_speed", "ER_jerk", "WZR", "AccR", "Delta1"]
    col_w = 22
    hdr = f"{'variant':{col_w}s}" + "".join(f"{k:>12s}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    print(f"{'PAPER':{col_w}s}" + "".join(f"{PAPER_TARGET[k]:>12.3f}" for k in keys))
    for tag, m in rows:
        line = f"{tag:{col_w}s}"
        for k in keys:
            v = m[k]["point"]
            arrow = better_arrow(k, v, PAPER_TARGET[k])
            line += f"{v:>10.3f}{arrow:>2s}"
        print(line)

    print()
    print("↑ = better than paper target, ↓ = worse, ≈ = within noise threshold.")
    print("Targets: MPJPE/Δ1 lower is better; ER ratios / WZR / AccR closer to 1 is better.")


if __name__ == "__main__":
    main()
