#!/bin/bash
# Experiment 2: Stage 1 Loss Ablation (Supplementary)
#
# Compares v1 (L1 only) vs v2 (L1+vel+speed) vs v3 (full, production)
# to demonstrate the value of z-range/std matching losses.
#
# v3 is already trained (production model). Only v1 and v2 need training.
# ~2h each on single GPU
#
# Usage:
#   bash scripts/ablation/stage1_v1v2.sh [gpu_id]

set -e
cd "$(dirname "$0")/../.."

GPU_ID="${1:-1}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

CACHE_DIR="MOSA_dataset/training_cache_W05"
MODEL_TYPE="hand_bilstm"
HIDDEN=512
LAYERS=6
LR=1e-4
BATCH=16
EPOCHS=500
PATIENCE=50

echo "============================================================"
echo "Exp 2: Stage 1 Loss Ablation (GPU $GPU_ID)"
echo "============================================================"

# --- v1: L1 position + velocity only ---
echo ""
echo ">>> Training v1 (L1 + velocity, no z-range/std)..."
mkdir -p checkpoints/stage1_ablation_v1

python src/stage1/s14_train_hand.py \
    --cache_dir "$CACHE_DIR" \
    --output_dir checkpoints/stage1_ablation_v1 \
    --model_type "$MODEL_TYPE" \
    --input_level L3 \
    --epochs $EPOCHS --batch_size $BATCH --lr $LR \
    --latent_dim $HIDDEN --num_layers $LAYERS \
    --window_size 480 --patience $PATIENCE \
    --loss_version v1 \
    --onset_weight 2.0 --z_weight 1.0 --vel_w 0.5 \
    --wandb_project pianoid-stage1-ablation

# --- v2: L1 + velocity + speed matching ---
echo ""
echo ">>> Training v2 (L1 + velocity + speed, no z-range/std)..."
mkdir -p checkpoints/stage1_ablation_v2

python src/stage1/s14_train_hand.py \
    --cache_dir "$CACHE_DIR" \
    --output_dir checkpoints/stage1_ablation_v2 \
    --model_type "$MODEL_TYPE" \
    --input_level L3 \
    --epochs $EPOCHS --batch_size $BATCH --lr $LR \
    --latent_dim $HIDDEN --num_layers $LAYERS \
    --window_size 480 --patience $PATIENCE \
    --loss_version v2 \
    --onset_weight 2.0 --z_weight 2.0 --vel_w 2.0 --speed_w 1.0 \
    --wandb_project pianoid-stage1-ablation

echo ""
echo "============================================================"
echo "Exp 2 complete."
echo "  v1: checkpoints/stage1_ablation_v1/"
echo "  v2: checkpoints/stage1_ablation_v2/"
echo "  v3: checkpoints/stage1/ (production, already trained)"
echo "============================================================"
echo "Next: evaluate all three with src/stage1/s16_evaluate.py + ER metrics"
