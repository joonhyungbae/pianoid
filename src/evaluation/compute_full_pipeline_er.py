#!/usr/bin/env python3
"""
Compute expressiveness-ratio metrics (and bootstrap CIs) on a directory of
Pianoid full-pipeline predictions.

Each subdirectory of `--pred_dir` is expected to contain
`generated_motion.npy` and `gt_motion.npy` shaped (T, 26, 3), as written
by `src/pipeline/full_pipeline.py --save_motion`.

Paper-consistent aggregation:
  - Per-axis, per-joint ER ratios computed on the backbone 10 joints.
  - Pianoid's `Δ₁ = mean(|log max(ER, 1e-6)|)` is computed from the bulk
    (sample-mean) ER values, not from per-sample Δ₁ — matches the
    definition in §5.1 of the paper.
  - Bootstrap resamples sample-level metrics and recomputes the bulk mean
    inside each resample, so CIs reflect the same estimator.

Usage:
    python src/evaluation/compute_full_pipeline_er.py \\
        --pred_dir MOSA_dataset/full_body_predictions/test \\
        --output results/full_pipeline_er.json

    # Tag the run (stored as "run_tag" inside the json)
    python ... --pred_dir ... --output ... --run_tag mdm_zs_rng1_std1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.constants import (
    RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX,
    C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
    RSHO_IDX, LSHO_IDX,
    RWRIST_IDX, LWRIST_IDX,
)

BACKBONE = [RFHD_IDX, LFHD_IDX, RBHD_IDX, LBHD_IDX,
            C7_IDX, CLAV_IDX, T10_IDX, STRN_IDX,
            RSHO_IDX, LSHO_IDX]
WRIST = [LWRIST_IDX, RWRIST_IDX]


def per_sample_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Per-sample metrics: bulk-mean ER ratios, WZR, AccR, MPJPE."""
    rngs, spds, jrks = [], [], []
    for j in BACKBONE:
        for a in range(3):
            p, g = pred[:, j, a], gt[:, j, a]
            gr = g.max() - g.min()
            if gr > 1e-6:
                rngs.append((p.max() - p.min()) / gr)
            gv = np.abs(np.diff(g))
            if gv.mean() > 1e-6:
                spds.append(np.abs(np.diff(p)).mean() / gv.mean())
            ga = np.diff(g, n=3)
            if np.abs(ga).mean() > 1e-6:
                jrks.append(np.abs(np.diff(p, n=3)).mean() / np.abs(ga).mean())
    wzr_vals = []
    for j in WRIST:
        gr = gt[:, j, 2].max() - gt[:, j, 2].min()
        if gr > 1e-6:
            wzr_vals.append((pred[:, j, 2].max() - pred[:, j, 2].min()) / gr)
    acc_vals = []
    for j in BACKBONE:
        for a in range(3):
            pa = np.diff(pred[:, j, a], n=2)
            ga = np.diff(gt[:, j, a], n=2)
            if np.abs(ga).mean() > 1e-6:
                acc_vals.append(np.abs(pa).mean() / np.abs(ga).mean())
    return {
        "er_rng_mean": float(np.mean(rngs)),
        "er_spd_mean": float(np.mean(spds)),
        "er_jrk_mean": float(np.mean(jrks)),
        "wzr_mean":    float(np.mean(wzr_vals)) if wzr_vals else float("nan"),
        "accr_mean":   float(np.mean(acc_vals)),
        "mpjpe":       float(np.sqrt(((pred - gt) ** 2).sum(axis=-1)).mean() * 1000),
    }


def bulk_metrics_from_indices(per_sample: dict, idx: np.ndarray) -> dict:
    rng = float(per_sample["er_rng_mean"][idx].mean())
    spd = float(per_sample["er_spd_mean"][idx].mean())
    jrk = float(per_sample["er_jrk_mean"][idx].mean())
    wzr = float(per_sample["wzr_mean"][idx].mean())
    accr = float(per_sample["accr_mean"][idx].mean())
    mpjpe = float(per_sample["mpjpe"][idx].mean())
    d1 = float(np.mean([abs(np.log(max(x, 1e-6))) for x in (rng, spd, jrk)]))
    return {
        "mpjpe": mpjpe, "ER_range": rng, "ER_speed": spd, "ER_jerk": jrk,
        "WZR": wzr, "AccR": accr, "Delta1": d1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", type=Path, required=True,
                   help="Directory with {sample}/(generated_motion.npy, gt_motion.npy)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run_tag", default=None, help="Optional label stored in the json")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    samples = []
    for d in sorted(args.pred_dir.iterdir()):
        if not d.is_dir():
            continue
        gen_path = d / "generated_motion.npy"
        gt_path = d / "gt_motion.npy"
        if not (gen_path.exists() and gt_path.exists()):
            continue
        gen = np.load(gen_path)
        gt = np.load(gt_path)
        samples.append(per_sample_stats(gen, gt))
    if not samples:
        raise RuntimeError(f"No valid samples in {args.pred_dir}")

    per_sample = {k: np.array([s[k] for s in samples]) for k in samples[0]}
    all_idx = np.arange(len(samples))
    point = bulk_metrics_from_indices(per_sample, all_idx)

    rng_gen = np.random.default_rng(args.seed)
    boots = {k: [] for k in point}
    for _ in range(args.n_bootstrap):
        idx = rng_gen.integers(0, len(samples), size=len(samples))
        for k, v in bulk_metrics_from_indices(per_sample, idx).items():
            boots[k].append(v)

    out = {"run_tag": args.run_tag, "n_samples": len(samples),
           "n_bootstrap": args.n_bootstrap, "seed": args.seed, "metrics": {}}
    for k, v in point.items():
        arr = np.array(boots[k])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out["metrics"][k] = {"point": v,
                              "ci_95_low": float(lo),
                              "ci_95_high": float(hi)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    print(f"=== {args.run_tag or args.pred_dir.name} (N={len(samples)}) ===")
    print(f"{'Metric':10s} {'point':>9s}  {'95% CI':>20s}")
    for k, m in out["metrics"].items():
        print(f"{k:10s} {m['point']:>9.4f}  [{m['ci_95_low']:.4f}, {m['ci_95_high']:.4f}]")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
