#!/bin/bash
# Stage 3: Elbow Generation (BiLSTM, shoulder-relative, predicted-shoulder inputs)
#
# Architecture: BiLSTM h=256, 6 layers
# Target:       Elbows only (6-dim), predicted relative to the Stage 2 shoulder
# Input:        wrists (6) + chest (6) + shoulders (6) = 18-dim spatial,
#               plus onset density + pitch range (2-dim temporal)
# Cascade:      Upstream Stage 1/2 predictions are used at train time (not GT),
#               fixing the exposure-bias L/R asymmetry (paper §4, Table 5-B Row 3)
#
# Usage:
#   bash scripts/train/stage3.sh              # GPU 1 (default)
#   bash scripts/train/stage3.sh 0            # GPU 0
#
# For the full Table 5-B ablation block: bash scripts/ablation/stage3_rows.sh all

set -e
cd "$(dirname "$0")/../.."

GPU_ID="${1:-1}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

ARCH="bilstm"
CACHE_DIR="./MOSA_dataset/training_cache_W10"
HAND_DIR="./MOSA_dataset/phase0_hand_predictions"
TORSO_DIR="./MOSA_dataset/phase1_torso_predictions"
HIDDEN=256
LAYERS=6
LR=1e-4

echo "============================================================"
echo "Stage 3: Elbow Generation ($ARCH, shoulder-relative, predicted-train)"
echo "============================================================"
python src/stage3/s30_train_arm.py \
    --arch $ARCH \
    --cache_dir $CACHE_DIR \
    --hand_pred_dir $HAND_DIR \
    --torso_pred_dir $TORSO_DIR \
    --output_dir ./checkpoints/stage3_arm \
    --relative_coords \
    --use_onset_density --use_pitch_range \
    --latent_dim $HIDDEN --num_layers $LAYERS --lr $LR \
    --batch_size 16 --epochs 500 --patience 50 \
    --wandb_project pianoid-stage3

echo ""
echo "Stage 3 Complete!"
echo "Checkpoint: ./checkpoints/stage3_arm/bilstm_rel_od_pr_h${HIDDEN}_l${LAYERS}_lr${LR}/best.pt"
echo "Next: bash scripts/train/stage4.sh"
