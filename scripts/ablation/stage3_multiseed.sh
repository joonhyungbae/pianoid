#!/bin/bash
# Stage 3 Anatomical Regrouping Ablation — MULTI-SEED version (Table 5-B v2)
#
# Runs the 2x2 design (grouping × training-input-source) across SEEDS={42,7,1337}
# to isolate the F3 anatomical-regrouping effect from single-seed noise, per
# ISMIR reviewer request (interaction test on grouping × predicted-train).
#
# Total: 4 cells × 3 seeds = 12 runs.
#
# Usage:
#   bash scripts/ablation/stage3_multiseed.sh seed42
#   bash scripts/ablation/stage3_multiseed.sh seed7
#   bash scripts/ablation/stage3_multiseed.sh seed1337
#   bash scripts/ablation/stage3_multiseed.sh all        # sequential all 12 runs
#   GPU_ID=1 bash scripts/ablation/stage3_multiseed.sh seed42  # pick a specific GPU
#
# Results populate ./checkpoints/stage3_multiseed/seed{42,7,1337}/row{1..4}_*/.
# Downstream analysis: scripts/ablation/stage3_multiseed_summarize.py (TBD).

set -e
cd "$(dirname "$0")/../.."

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

ARCH="bilstm"
CACHE_DIR="./MOSA_dataset/training_cache_W10"
HAND_DIR="./MOSA_dataset/phase0_hand_predictions"
TORSO_DIR="./MOSA_dataset/phase1_torso_predictions"
OUT_ROOT="./checkpoints/stage3_multiseed"
HIDDEN=256
LAYERS=6
LR=1e-4
BATCH=16
EPOCHS=500
PATIENCE=50

SEEDS_DEFAULT=(42 7 1337)

run_one() {
    local seed=$1
    local row=$2
    local extra_flags=$3
    local out="$OUT_ROOT/seed${seed}/${row}"
    mkdir -p "$(dirname "$out")"
    if [[ -f "$out/best.pt" ]]; then
        echo "[skip] $out/best.pt exists"
        return
    fi
    echo ""
    echo "=========================================================="
    echo "seed=$seed  row=$row  flags=$extra_flags"
    echo "=========================================================="
    python src/stage3/s30_train_arm.py \
        --arch "$ARCH" \
        --cache_dir "$CACHE_DIR" \
        --hand_pred_dir "$HAND_DIR" \
        --torso_pred_dir "$TORSO_DIR" \
        --latent_dim "$HIDDEN" --num_layers "$LAYERS" --lr "$LR" \
        --batch_size "$BATCH" --epochs "$EPOCHS" --patience "$PATIENCE" \
        --use_onset_density --use_pitch_range \
        --seed "$seed" \
        $extra_flags \
        --output_dir "$out" \
        --no_wandb
}

run_seed() {
    local seed=$1
    # 2x2 cells: (grouping ∈ {functional, anatomical}) × (train ∈ {gt, predicted})
    run_one "$seed" row1_functional_gt        "--functional_grouping --use_gt_inputs"
    run_one "$seed" row2_anatomical_gt        "--relative_coords --use_gt_inputs"
    run_one "$seed" row3_anatomical_predicted "--relative_coords"
    run_one "$seed" row4_functional_predicted "--functional_grouping"
}

CMD="${1:-help}"
case "$CMD" in
    seed42)   run_seed 42   ;;
    seed7)    run_seed 7    ;;
    seed1337) run_seed 1337 ;;
    all)
        for s in "${SEEDS_DEFAULT[@]}"; do run_seed "$s"; done
        ;;
    help|*)
        echo "Usage: [GPU_ID=N] bash scripts/ablation/stage3_multiseed.sh {seed42|seed7|seed1337|all}"
        echo ""
        echo "Total: 12 runs (4 cells × 3 seeds). Each Stage-3 BiLSTM run ≈ 500 epochs"
        echo "(or earlier with patience=50) × batch 16 × window 10s."
        ;;
esac
