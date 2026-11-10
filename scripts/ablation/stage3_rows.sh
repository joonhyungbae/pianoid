#!/bin/bash
# Stage 3 Anatomical Regrouping Ablation (Table 5-B)
#
# Reproduces the four rows of Table 5-B (2x2 design: grouping × training-input-source):
#
#   Row 1  Functional grouping, GT-train              MPJPE 82.5  Z-rng 0.44  L/R 27 mm  JerkR 1.14
#   Row 2  Anatomical regroup, GT-train               MPJPE 99.5  Z-rng 0.67  L/R 27 mm  JerkR 1.14
#   Row 3  Anatomical + Predicted-train (ours)        MPJPE 82.6  Z-rng 0.72  L/R  0.1   JerkR 1.02
#   Row 4  Functional + Predicted-train               MPJPE ??    Z-rng ??    L/R ??     JerkR ??
#          (2x2 completion — isolates the effect of anatomical regrouping from predicted-train)
#
# Row 1 predicts 4 joints (shoulders+elbows, 12-dim). Requires `--functional_grouping`
# support added to src/stage3/s30_train_arm.py. Input conditioning excludes shoulders
# (wrists+chest, 12-dim) to avoid target leakage.
#
# Row 2 predicts elbows only (shoulder-relative) but trained on GT shoulders,
# surfacing the exposure-bias L/R asymmetry.
#
# Row 3 is the production configuration (same as scripts/train/stage3.sh) — included
# so that `all` trains the full Table 5-B block in one run.
#
# Usage:
#   bash scripts/ablation/stage3_rows.sh row1
#   bash scripts/ablation/stage3_rows.sh row2
#   bash scripts/ablation/stage3_rows.sh row3
#   bash scripts/ablation/stage3_rows.sh all
#   bash scripts/ablation/stage3_rows.sh list
#
# GPU selection: `GPU_ID=0 bash scripts/ablation/stage3_rows.sh row1`

set -e
cd "$(dirname "$0")/../.."

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

ARCH="bilstm"
CACHE_DIR="./MOSA_dataset/training_cache_W10"
HAND_DIR="./MOSA_dataset/phase0_hand_predictions"
TORSO_DIR="./MOSA_dataset/phase1_torso_predictions"
OUT_BASE="./checkpoints/stage3_ablation"
HIDDEN=256
LAYERS=6
LR=1e-4
BATCH=16
EPOCHS=500
PATIENCE=50

COMMON=(
    --arch "$ARCH"
    --cache_dir "$CACHE_DIR"
    --hand_pred_dir "$HAND_DIR"
    --torso_pred_dir "$TORSO_DIR"
    --latent_dim "$HIDDEN" --num_layers "$LAYERS" --lr "$LR"
    --batch_size "$BATCH" --epochs "$EPOCHS" --patience "$PATIENCE"
    --use_onset_density --use_pitch_range
    --wandb_project pianoid-stage3-ablation
)

banner() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

row1() {
    banner "Row 1: Functional grouping, GT-train (shoulders+elbows, 12-dim)"
    python src/stage3/s30_train_arm.py "${COMMON[@]}" \
        --functional_grouping \
        --use_gt_inputs \
        --output_dir "$OUT_BASE/row1_functional_gt_train"
}

row2() {
    banner "Row 2: + Anatomical regroup (elbows only, shoulder-relative, GT-train)"
    python src/stage3/s30_train_arm.py "${COMMON[@]}" \
        --relative_coords \
        --use_gt_inputs \
        --output_dir "$OUT_BASE/row2_anatomical_gt_train"
}

row3() {
    banner "Row 3: + Predicted-train (ours) — matches scripts/train/stage3.sh"
    python src/stage3/s30_train_arm.py "${COMMON[@]}" \
        --relative_coords \
        --output_dir "$OUT_BASE/row3_anatomical_predicted_train"
}

row4() {
    banner "Row 4: Functional grouping, Predicted-train (2x2 completion)"
    python src/stage3/s30_train_arm.py "${COMMON[@]}" \
        --functional_grouping \
        --output_dir "$OUT_BASE/row4_functional_predicted_train"
}

list_rows() {
    printf "  %-6s  %s\n" "ROW"  "CONFIG"
    printf "  %-6s  %s\n" "row1" "--functional_grouping --use_gt_inputs   (predicts 4 joints, GT wrists/chest)"
    printf "  %-6s  %s\n" "row2" "--relative_coords --use_gt_inputs       (elbows only, GT shldr ref)"
    printf "  %-6s  %s\n" "row3" "--relative_coords                        (predicted shldr, production)"
    printf "  %-6s  %s\n" "row4" "--functional_grouping                    (predicts 4 joints, predicted wrists/chest)"
}

CMD="${1:-help}"
case "$CMD" in
    row1) row1 ;;
    row2) row2 ;;
    row3) row3 ;;
    row4) row4 ;;
    all)  row1; row2; row3; row4 ;;
    list) list_rows ;;
    help|*)
        echo "Usage: bash scripts/ablation/stage3_rows.sh {row1|row2|row3|all|list}"
        echo ""
        echo "Reproduces Table 5-B rows."
        list_rows
        ;;
esac
