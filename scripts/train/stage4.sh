#!/bin/bash
# Experiment 3: Stage 4 Lower Body Ablation (Supplementary)
#
# 5 experiments: mean_offset, film_spine, film_od, film_aria, bilstm_od
# Each evaluated on GT and predicted spine inputs.
#
# ~1-2h total on single GPU
#
# Usage:
#   bash scripts/train/stage4.sh [gpu_id]

set -e
cd "$(dirname "$0")/../.."

GPU_ID="${1:-1}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "============================================================"
echo "Exp 3: Stage 4 Lower Body Ablation (GPU $GPU_ID)"
echo "============================================================"

python src/stage4/s40_experiments.py \
    --experiment all \
    --cache_dir MOSA_dataset/training_cache_W10 \
    --torso_pred_dir MOSA_dataset/phase1_torso_predictions \
    --output results/stage4_ablation.json \
    --wandb_project pianoid-stage4-ablation \
    --epochs 500 \
    --patience 50

echo ""
echo "============================================================"
echo "Exp 3 complete. Results: results/stage4_ablation.json"
echo "============================================================"
