#!/usr/bin/env bash
# Option C baseline training + inference orchestration.
#
# This script documents (and can run) all baseline training + inference
# for the Pianoid perceptual study's hierarchical-framing design.
#
# Baselines (paper Table 2):
#   1. BiLSTM-backbone — ~2-4h on single GPU. Quickest training baseline.
#   2. Li2018          — ~3-6h on single GPU. Li et al. ISMIR 2018 CNN-LSTM (published baseline).
#   3. MDM-adapted     — ~1-2 days on single GPU. Full Tevet 2023 MDM faithful adaptation.
#   4. EDGE-adapted    — ~1-2 days on single GPU. Full Tseng 2023 EDGE adaptation.
#
# Usage:
#   bash scripts/baselines.sh bilstm_backbone         # train BiLSTM-backbone
#   bash scripts/baselines.sh li2018                  # train Li2018
#   bash scripts/baselines.sh mdm_adapted             # train MDM-adapted
#   bash scripts/baselines.sh edge_adapted            # train EDGE-adapted
#   bash scripts/baselines.sh status                  # show current baseline status

set -euo pipefail
cd "$(dirname "$0")/.."

PIANOID_PY=/root/miniconda3/envs/pianoid/bin/python
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

case "${1:-status}" in

  status)
    echo "============================================================"
    echo " Baseline status"
    echo "============================================================"
    for BL in bilstm_backbone li2018 mdm_adapted edge_adapted; do
      case "$BL" in
        bilstm_backbone) DIR="experiments/benchmark/bilstm_backbone/checkpoints/results" ;;
        li2018)          DIR="experiments/benchmark/li2018/results" ;;
        mdm_adapted)     DIR="experiments/benchmark/mdm_adapted/checkpoints/results" ;;
        edge_adapted)    DIR="experiments/benchmark/edge_adapted/checkpoints/results" ;;
      esac
      if [ -d "$DIR" ]; then
        N=$(ls "$DIR" 2>/dev/null | grep -c '\.npz' || true)
        echo "  $BL: $N npz files at $DIR"
      else
        echo "  $BL: (not trained yet)"
      fi
    done
    ;;

  bilstm|bilstm_backbone)
    echo ">>> Training BiLSTM-backbone on GPU $GPU_ID"
    echo "    Expected time: ~2-4 hours"
    $PIANOID_PY experiments/benchmark/bilstm_backbone/model.py --mode train \
        --cache_dir MOSA_dataset/training_cache_B \
        --output_dir experiments/benchmark/bilstm_backbone/checkpoints \
        --batch_size 16 --hidden_dim 512 --num_layers 6 \
        --epochs 500 --lr 1e-4 --patience 50
    echo ">>> Running eval (DDIM not needed, deterministic)"
    $PIANOID_PY experiments/benchmark/bilstm_backbone/model.py --mode eval \
        --cache_dir MOSA_dataset/training_cache_B \
        --output_dir experiments/benchmark/bilstm_backbone/checkpoints
    ;;

  li2018)
    echo ">>> Training Li2018 (Li et al. ISMIR 2018, CNN-LSTM) on GPU $GPU_ID"
    echo "    Expected time: ~10-15 hours (500 epochs unified budget, 2-stage schedule)"
    $PIANOID_PY experiments/benchmark/li2018/train.py \
        --cache_dir MOSA_dataset/training_cache_B \
        --checkpoint_dir experiments/benchmark/li2018/checkpoints \
        --stage1_epochs 30 --stage2_epochs 470 \
        --lstm_hidden 256 --lstm_layers 2 \
        --batch_size 16 --lr 1e-4 --weight_decay 1e-8 \
        --patience 50
    echo ">>> Running Li2018 inference (deterministic)"
    $PIANOID_PY experiments/benchmark/li2018/inference.py \
        --cache_dir MOSA_dataset/training_cache_B \
        --checkpoint experiments/benchmark/li2018/checkpoints/best_model.pt \
        --output_dir experiments/benchmark/li2018/results
    ;;

  mdm|mdm_adapted)
    echo ">>> Training MDM-adapted (Tevet 2023) on GPU $GPU_ID"
    echo "    Expected time: ~1-2 days"
    $PIANOID_PY experiments/benchmark/mdm_adapted/model.py --mode train \
        --cache_dir MOSA_dataset/training_cache_B \
        --output_dir experiments/benchmark/mdm_adapted/checkpoints \
        --batch_size 16 --hidden_dim 512 --num_layers 8 \
        --epochs 500 --lr 1e-4 --patience 50
    echo ">>> Running eval (DDIM 25 steps)"
    $PIANOID_PY experiments/benchmark/mdm_adapted/model.py --mode eval \
        --cache_dir MOSA_dataset/training_cache_B \
        --output_dir experiments/benchmark/mdm_adapted/checkpoints \
        --ddim_steps 25
    ;;

  edge|edge_adapted)
    echo ">>> Training EDGE-adapted (Tseng 2023) on GPU $GPU_ID"
    echo "    Expected time: ~1-2 days"
    $PIANOID_PY experiments/benchmark/edge_adapted/model.py --mode train \
        --cache_dir MOSA_dataset/training_cache_B \
        --output_dir experiments/benchmark/edge_adapted/checkpoints \
        --batch_size 16 --hidden_dim 256 --num_layers 4 --window 480 \
        --epochs 500 --lr 1e-4 --patience 50
    echo ">>> Running eval (DDIM 25 steps)"
    $PIANOID_PY experiments/benchmark/edge_adapted/model.py --mode eval \
        --cache_dir MOSA_dataset/training_cache_B \
        --output_dir experiments/benchmark/edge_adapted/checkpoints \
        --ddim_steps 25
    ;;

  all|full)
    echo ">>> Running all baselines"
    bash "$0" bilstm_backbone
    bash "$0" li2018
    bash "$0" mdm_adapted
    bash "$0" edge_adapted
    ;;

  *)
    echo "Usage: bash scripts/baselines.sh {status|bilstm|li2018|mdm|edge|all}"
    echo ""
    echo "Suggested order (with --gpu_id):"
    echo "  1. GPU_ID=1 bash scripts/baselines.sh bilstm            (~2-4h)"
    echo "  2. GPU_ID=1 bash scripts/baselines.sh li2018            (~3-6h)"
    echo "  3. GPU_ID=1 bash scripts/baselines.sh mdm_adapted       (~1-2d)"
    echo "  4. GPU_ID=1 bash scripts/baselines.sh edge_adapted      (~1-2d)"
    ;;

esac
