#!/bin/bash
# Stage 2 MDM ZScore Backbone Sweep
#
# MDM + zscore repr showed excellent expressiveness (range ratio 0.81-1.32).
# This sweep tunes range_w/std_w to reduce Z-axis over-expression (1.67→1.0)
# while preserving the dynamic motion.
#
# Usage:
#   bash scripts/train/stage2.sh sweep     # run all variants
#   bash scripts/train/stage2.sh eval      # evaluate all
#   bash scripts/train/stage2.sh list      # show grid

set -e
cd "$(dirname "$0")/../.."

GPU_ID=1
export CUDA_VISIBLE_DEVICES=$GPU_ID

# ── Fixed: MDM zscore backbone ──────────────────────────────────────
ARCH="mdm"
CACHE_DIR="./MOSA_dataset/training_cache_W05"
PRED_DIR="./MOSA_dataset/phase0_hand_predictions"
HIDDEN_DIM=256
NUM_LAYERS=8
LR=1e-4
BATCH_SIZE=16
EPOCHS=500
PATIENCE=50
REPR="zscore"
WANDB_PROJECT="pianoid-stage2"
OUTDIR="checkpoints/stage2_backbone"

# Each row: "NAME  RANGE_W  STD_W  SPEED_W  UNDER_W  COND_FLAGS"
VARIANTS=(
    # Baseline: no expressiveness loss
    "mdm_zs_base            0.0  0.0  0.0  0.0  --use_onset_density"
    # Range/std only (different weights)
    "mdm_zs_rng05_std05     0.5  0.5  0.0  0.0  --use_onset_density"
    "mdm_zs_rng1_std1       1.0  1.0  0.0  0.0  --use_onset_density"
    "mdm_zs_rng2_std2       2.0  2.0  0.0  0.0  --use_onset_density"
    # With speed/under
    "mdm_zs_rng1_spd1       1.0  1.0  1.0  0.0  --use_onset_density"
    "mdm_zs_rng1_spd1_und1  1.0  1.0  1.0  1.0  --use_onset_density"
    # Onset per key (88-dim, richer music info)
    "mdm_zs_opk_rng1        1.0  1.0  0.0  0.0  --use_onset_per_key"
)

run_one() {
    local NAME=$1 RANGE_W=$2 STD_W=$3 SPEED_W=$4 UNDER_W=$5
    shift 5
    local COND_FLAGS="$@"

    echo ""
    echo "============================================================"
    echo "MDM Sweep: ${NAME}"
    echo "  repr=zscore range_w=${RANGE_W} std_w=${STD_W} speed_w=${SPEED_W} under_w=${UNDER_W}"
    echo "  cond: ${COND_FLAGS}"
    echo "============================================================"

    python src/stage2/s20_train_torso_sweep.py \
        --arch $ARCH --backbone \
        --latent_dim $HIDDEN_DIM --num_layers $NUM_LAYERS --lr $LR \
        --repr $REPR \
        --range_w "$RANGE_W" --std_w "$STD_W" \
        --speed_w "$SPEED_W" --under_w "$UNDER_W" \
        $COND_FLAGS \
        --output_dir "$OUTDIR" \
        --cache_dir "$CACHE_DIR" --pred_dir "$PRED_DIR" \
        --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE \
        --wandb_project "$WANDB_PROJECT"
}

eval_all() {
    echo ""
    echo "============================================================"
    echo "Evaluating all MDM zscore backbone checkpoints"
    echo "============================================================"
    mkdir -p results
    for CKPT in ${OUTDIR}/mdm_*zscore*/best.pt; do
        if [ -f "$CKPT" ]; then
            echo "  $CKPT"
            python src/stage2/s22_evaluate.py \
                --checkpoint "$CKPT" \
                --cache_dir "$CACHE_DIR" --pred_dir "$PRED_DIR" \
                --output results/stage2_mdm_sweep.json --gpu 0
        fi
    done
    echo "Results: results/stage2_mdm_sweep.json"
}

list_variants() {
    printf "  %-28s %-8s %-8s %-8s %-8s %s\n" "NAME" "RNG_W" "STD_W" "SPD_W" "UND_W" "COND"
    echo "  -------------------------------------------------------------------"
    for V in "${VARIANTS[@]}"; do
        read -r NAME RNG STD SPD UND COND <<< "$V"
        printf "  %-28s %-8s %-8s %-8s %-8s %s\n" "$NAME" "$RNG" "$STD" "$SPD" "$UND" "$COND"
    done
    echo "  ${#VARIANTS[@]} variants"
}

CMD="${1:-help}"
shift 2>/dev/null || true

case "$CMD" in
    main)
        # Production variant only — the one cited in the paper checkpoints
        # (mdm_od_h256_l8_lr0.0001_rng0.5_std0.5_zscore).
        run_one "mdm_zs_rng05_std05" 0.5 0.5 0.0 0.0 --use_onset_density
        ;;
    variant)
        # Train a single named variant. Usage: bash stage2.sh variant mdm_zs_rng1_std1
        WANT="${1:-}"
        if [ -z "$WANT" ]; then
            echo "Error: bash stage2.sh variant <name>"
            echo "Available names:"
            list_variants
            exit 1
        fi
        for V in "${VARIANTS[@]}"; do
            read -r NAME RNG STD SPD UND COND <<< "$V"
            if [ "$NAME" = "$WANT" ]; then
                run_one "$NAME" "$RNG" "$STD" "$SPD" "$UND" $COND
                exit 0
            fi
        done
        echo "Error: variant '$WANT' not found. Run 'bash stage2.sh list' for options."
        exit 1
        ;;
    sweep)
        for V in "${VARIANTS[@]}"; do
            read -r NAME RNG STD SPD UND COND <<< "$V"
            run_one "$NAME" "$RNG" "$STD" "$SPD" "$UND" $COND
        done
        eval_all
        ;;
    eval)  eval_all ;;
    list)  list_variants ;;
    help|*)
        echo "Usage: bash scripts/train/stage2.sh {main|variant <name>|sweep|eval|list}"
        echo "  main:             train only the production variant (rng0.5_std0.5, paper final)"
        echo "  variant <name>:   train a single named variant (e.g. mdm_zs_rng1_std1)"
        echo "  sweep:            run all 7 MDM zscore backbone variants (Table 3A)"
        echo "  eval:             evaluate all checkpoints"
        echo "  list:             show variant grid"
        ;;
esac
