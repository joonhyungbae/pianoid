#!/bin/bash
# Pianoid: reproduce all paper numbers (no training).
#
# Reproduces:
#   §3 F1–F6 correlations + Table 1 (inter-body-part correlations)
#   §5.2 Table 2 row "Pianoid" (MPJPE 92.1, ER ratios, Δ_1 0.42)
#   §5.2 Table 4 (per-bodypart MPJPE)
#
# Time: ~10–20 min on a single GPU.
# Requires:
#   - bash setup.sh
#   - dataset preprocessed under MOSA_dataset/ (see README "Preprocessing")
#   - Released checkpoints under checkpoints/release/
#       bash scripts/download_checkpoints.sh    # one-shot fetch from OSF
#
# Usage:
#   bash inference.sh                # GPU 1 (default)
#   bash inference.sh 0              # GPU 0

set -e
cd "$(dirname "$0")"

GPU_ID="${1:-1}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "============================================================"
echo "Pianoid: paper reproduction (inference only)"
echo "GPU $GPU_ID"
echo "============================================================"

echo ""
echo ">>> [1/3] §3 Music-body coupling analysis (F1–F6, Table 1)"
python experiments/analysis/music_body_analysis.py
python experiments/analysis/analyze_body_decomposition.py
python experiments/analysis/compute_f3_shoulder_dominance.py
python experiments/analysis/compute_delta1_axis_correlation.py
python experiments/analysis/analyze_expressiveness.py

echo ""
echo ">>> [2/4] §5.2 Pianoid full pipeline (Table 2 row 'Pianoid', MPJPE/FID/Jitter)"
mkdir -p results MOSA_dataset/full_body_predictions
python src/pipeline/full_pipeline.py \
    --ddim_steps 45 \
    --save_motion \
    --save_dir MOSA_dataset/full_body_predictions \
    --npz_dir results/hierarchical_mdm \
    --output results/full_pipeline.json

echo ""
echo ">>> [3/4] §5.2 Pianoid expressiveness ratios (Table 2: ER-rng/spd/jrk, WZR, AccR, Δ₁)"
python src/evaluation/compute_full_pipeline_er.py \
    --pred_dir MOSA_dataset/full_body_predictions/test \
    --output results/full_pipeline_er.json \
    --run_tag pianoid_release

echo ""
echo ">>> [4/4] §5.2 Table 4: per-bodypart MPJPE"
python experiments/benchmark/compute_bodypart_metrics.py

echo ""
echo "============================================================"
echo "Done!"
echo ""
echo "Outputs:"
echo "  results/full_pipeline.json                   (Table 2: MPJPE / FID / Jitter / VelCorr)"
echo "  results/full_pipeline_er.json                (Table 2: ER ratios / WZR / AccR / Δ₁)"
echo "  experiments/benchmark/bodypart_metrics.json  (Table 4)"
echo "  experiments/analysis/*.json                  (§3 F1–F6, Table 1)"
echo ""
echo "Expected paper numbers (Table 2 row 'Pianoid'):"
echo "  MPJPE 92.1 mm | ER-rng 1.12 | ER-spd 0.92 | ER-jrk 0.34"
echo "  WZR 0.76 | AccR 0.44 | Δ_1 0.42"
echo ""
echo "Optional ablations (require their own training; see README):"
echo "  scripts/ablation/stage{1,2,3}_*.sh"
echo "============================================================"
