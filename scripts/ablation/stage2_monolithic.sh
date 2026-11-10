#!/bin/bash
# Experiment 1: Single-Stage MDM + zscore (Fair Baseline)
#
# Same tricks as Pianoid Stage 2 (zscore norm, onset_density, DDIM 25)
# but monolithic 26-joint output. Proves hierarchical decomposition matters.
#
# ~4-6h training + ~1h eval on single GPU
#
# Usage:
#   bash scripts/ablation/stage2_monolithic.sh [gpu_id]

set -e
cd "$(dirname "$0")/../.."

GPU_ID="${1:-1}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

CACHE_DIR="MOSA_dataset/training_cache_B"
OUT_DIR="experiments/benchmark/single_stage_mdm_zscore/checkpoints"

echo "============================================================"
echo "Exp 1: Single-Stage MDM + zscore (GPU $GPU_ID)"
echo "============================================================"

# Train
python experiments/benchmark/single_stage_mdm_zscore/model.py \
    --mode train \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUT_DIR" \
    --batch_size 16 \
    --hidden_dim 256 \
    --num_layers 8 \
    --epochs 500 \
    --lr 1e-4 \
    --patience 50 \
    --use_wandb

echo ""
echo "Training complete. Running evaluation..."

# Eval (DDIM 25 steps)
python experiments/benchmark/single_stage_mdm_zscore/model.py \
    --mode eval \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUT_DIR" \
    --ddim_steps 25

echo ""
echo "============================================================"
echo "Exp 1 complete. Results: $OUT_DIR/results/"
echo "============================================================"
echo "Next: python experiments/benchmark/evaluate_all.py"
