#!/bin/bash
# Pianoid: full training cascade from scratch.
#
# Trains all four stages and the published baselines.
# After this completes, run `bash inference.sh` to reproduce paper numbers.
#
# Time: ~3–5 days on a single GPU (Stage 2 + baselines dominate).
# Requires:
#   - bash setup.sh
#   - dataset preprocessed under MOSA_dataset/ (see README "Preprocessing")
#
# Usage:
#   bash train.sh                    # GPU 1 (default), full cascade
#   bash train.sh 0                  # GPU 0
#   bash train.sh 1 --skip_baselines # skip Table 2 baseline training

set -e
cd "$(dirname "$0")"

GPU_ID="${1:-1}"
SKIP_BASELINES=false
for arg in "$@"; do
    if [ "$arg" = "--skip_baselines" ]; then SKIP_BASELINES=true; fi
done
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "============================================================"
echo "Pianoid: full training cascade"
echo "GPU $GPU_ID | skip_baselines=$SKIP_BASELINES"
echo "============================================================"

echo ""
echo ">>> [Stage 1] BiLSTM wrist (paper Stage 1)"
bash scripts/train/stage1.sh

echo ""
echo ">>> [Stage 2] MDM+zscore backbone sweep (paper Stage 2)"
bash scripts/train/stage2.sh sweep

echo ""
echo ">>> [Stage 2 → Stage 3] Precompute backbone predictions (DDIM 45)"
python src/stage2/s21_precompute_torso.py \
    --torso_path checkpoints/stage2_backbone/mdm_od_h256_l8_lr0.0001_rng0.5_std0.5_zscore/best.pt \
    --ddim_steps 45 \
    --output_dir MOSA_dataset/phase1_torso_predictions

echo ""
echo ">>> [Stage 3] BiLSTM elbow on predicted shoulders (paper Stage 3)"
bash scripts/train/stage3.sh

echo ""
echo ">>> [Stage 4] FiLM lower body sweep (paper Stage 4)"
bash scripts/train/stage4.sh

if [ "$SKIP_BASELINES" = false ]; then
    echo ""
    echo ">>> [Baselines] Table 2: Li2018, MDM-adapted, EDGE-adapted"
    bash scripts/baselines.sh all
fi

echo ""
echo "============================================================"
echo "Training cascade complete."
echo "Run \`bash inference.sh\` to reproduce paper numbers."
echo "============================================================"
