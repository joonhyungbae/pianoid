#!/bin/bash
# Stage 2 — Distal head error reduction sweep
#
# Paper reports Stage 2 head MPJPE 114.3 mm (worst joint group). This script
# sweeps model-capacity variants aimed at reducing that head error while
# keeping the core recipe (MDM + zscore + rng0.5/std0.5 + onset density).
#
# Variants:
#   v_base   — production h256 l8 (reference, same as scripts/train/stage2.sh main)
#   v_h512   — wider (h512 l8): more representation capacity per frame
#   v_l12    — deeper (h256 l12): more reasoning depth
#   v_ep1k   — epochs 1000 (double the production 500; lets earlyStop act later)
#
# Usage:
#   bash scripts/ablation/stage2_distal_head.sh sweep          # all variants
#   bash scripts/ablation/stage2_distal_head.sh v_h512         # single variant
#   bash scripts/ablation/stage2_distal_head.sh list
#
# Prereq: Stage 1 trained, phase0_hand_predictions/ available.

set -e
cd "$(dirname "$0")/../.."

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

ARCH="mdm"
REPR="zscore"
CACHE_DIR="./MOSA_dataset/training_cache_W05"
PRED_DIR="./MOSA_dataset/phase0_hand_predictions"
OUTDIR="checkpoints/stage2_distal_head"
LR=1e-4
BATCH_SIZE=16
PATIENCE=50
WANDB_PROJECT="pianoid-stage2-distalhead"

# NAME  HIDDEN  LAYERS  EPOCHS  RANGE_W  STD_W  EXTRA
VARIANTS=(
    "v_base    256   8   500   0.5  0.5  --use_onset_density"
    "v_h512    512   8   500   0.5  0.5  --use_onset_density"
    "v_l12     256  12   500   0.5  0.5  --use_onset_density"
    "v_ep1k    256   8  1000   0.5  0.5  --use_onset_density"
)

run_one() {
    local NAME=$1 HIDDEN=$2 LAYERS=$3 EPOCHS=$4 RANGE_W=$5 STD_W=$6
    shift 6
    local EXTRA="$@"
    echo ""
    echo "============================================================"
    echo "Distal-head S2: $NAME (h=$HIDDEN l=$LAYERS ep=$EPOCHS rng=$RANGE_W std=$STD_W)"
    echo "  extra: $EXTRA"
    echo "============================================================"
    python src/stage2/s20_train_torso_sweep.py \
        --arch $ARCH --backbone --repr $REPR \
        --latent_dim $HIDDEN --num_layers $LAYERS --lr $LR \
        --range_w "$RANGE_W" --std_w "$STD_W" \
        $EXTRA \
        --output_dir "$OUTDIR/$NAME" \
        --cache_dir "$CACHE_DIR" --pred_dir "$PRED_DIR" \
        --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE \
        --wandb_project "$WANDB_PROJECT"
}

list_variants() {
    printf "  %-10s %s\n" "NAME" "CONFIG"
    for V in "${VARIANTS[@]}"; do
        read -r NAME H L E R S REST <<< "$V"
        printf "  %-10s h=%s l=%s ep=%s rng=%s std=%s %s\n" "$NAME" "$H" "$L" "$E" "$R" "$S" "$REST"
    done
}

CMD="${1:-help}"
case "$CMD" in
    sweep)
        for V in "${VARIANTS[@]}"; do
            read -r NAME H L E R S REST <<< "$V"
            run_one "$NAME" "$H" "$L" "$E" "$R" "$S" $REST
        done
        ;;
    list) list_variants ;;
    v_base|v_h512|v_l12|v_ep1k)
        for V in "${VARIANTS[@]}"; do
            read -r NAME H L E R S REST <<< "$V"
            if [ "$NAME" = "$CMD" ]; then
                run_one "$NAME" "$H" "$L" "$E" "$R" "$S" $REST
                exit 0
            fi
        done
        echo "Variant $CMD not found"; exit 1 ;;
    help|*)
        echo "Usage: bash scripts/ablation/stage2_distal_head.sh {sweep|list|v_base|v_h512|v_l12|v_ep1k}"
        list_variants ;;
esac
