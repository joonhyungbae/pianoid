#!/bin/bash
# Stage 1: Wrist Generation (Final)
#
# Architecture: BiLSTM (Phase B winner)
# Preprocessing: W05 (0.5s onset density window, Phase 0 winner)
# Input: L3 = piano_roll + onset_mask + onset_density + dur_ioi_ratio (352-dim)
# Config: hidden=512, layers=6, lr=1e-4 (Phase C winner)
# Loss: v3 (L1 + velocity + speed + z-range/std matching)
# Model selection: val MPJPE (not composite loss)
# Output: Wrist motion (2 joints x 3 = 6-dim)
#
# Usage:
#   bash scripts/train/stage1.sh
#
# For the Stage 1 loss ablation (Table 3(C)): bash scripts/ablation/stage1_v1v2.sh

set -e
cd "$(dirname "$0")/../.."

GPU_ID=1
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Best config from 4-phase search
MODEL_TYPE="hand_bilstm"
CACHE_DIR="./MOSA_dataset/training_cache_W05"
HIDDEN_DIM=512
NUM_LAYERS=6
LR=1e-4
BATCH_SIZE=16
EPOCHS=500
PATIENCE=50
INPUT_LEVEL="L3"

echo "============================================================"
echo "Stage 1: Wrist Generation (BiLSTM, v3 loss)"
echo "============================================================"
echo "GPU: $GPU_ID"
echo "Model: $MODEL_TYPE"
echo "Cache: $CACHE_DIR"
echo "Input: $INPUT_LEVEL (352-dim)"
echo "Config: hidden=$HIDDEN_DIM, layers=$NUM_LAYERS, lr=$LR"
echo "Loss: v3 (L1 + vel + speed + z_range + z_std)"
echo "Selection: val MPJPE"
echo ""

mkdir -p ./checkpoints/stage1

python src/stage1/s14_train_hand.py \
    --cache_dir "$CACHE_DIR" \
    --output_dir ./checkpoints/stage1 \
    --model_type "$MODEL_TYPE" \
    --input_level "$INPUT_LEVEL" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --latent_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --window_size 480 \
    --patience $PATIENCE \
    --loss_version v3 \
    --onset_weight 2.0 \
    --z_weight 2.0 \
    --vel_w 2.0 \
    --z_range_weight 2.0 \
    --z_std_weight 2.0 \
    --wandb_project pianoid-stage1

echo ""
echo "============================================================"
echo "Stage 1 Complete!"
echo "============================================================"
echo "Checkpoint: ./checkpoints/stage1/${MODEL_TYPE}_best.pt"

# Generate wrist predictions for all splits
echo ""
echo "Generating wrist predictions for train/val/test..."
python src/pipeline/phase0_precompute.py \
    --skip_train \
    --checkpoint "./checkpoints/stage1/${MODEL_TYPE}_best.pt" \
    --cache_dir "$CACHE_DIR"
echo "Predictions: ./MOSA_dataset/phase0_hand_predictions/"

echo ""
echo "Next: bash scripts/train/stage2.sh"
