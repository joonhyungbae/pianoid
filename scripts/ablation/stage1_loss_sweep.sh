#!/bin/bash
# Stage 1 Loss Weight Sweep
#
# Sweeps loss weight combinations for v3 loss (z_range, z_std amplitude matching).
# All runs share the same architecture (BiLSTM, L3, h=512, layers=6).
#
# Usage:
#   # Grid sweep over z_range_weight × z_std_weight
#   bash scripts/ablation/stage1_loss_sweep.sh sweep
#
#   # Custom single run
#   bash scripts/ablation/stage1_loss_sweep.sh run --vel_w 2.0 --z_range_w 0.5 --z_std_w 0.5
#
#   # Run a named variant (v1, v2, v3, v3a, v3b)
#   bash scripts/ablation/stage1_loss_sweep.sh v3a
#
#   # Evaluate all sweep checkpoints
#   bash scripts/ablation/stage1_loss_sweep.sh eval
#
#   # List available named variants
#   bash scripts/ablation/stage1_loss_sweep.sh list

set -e
cd "$(dirname "$0")/../.."

GPU_ID=1
export CUDA_VISIBLE_DEVICES=$GPU_ID

# ── Shared architecture (Phase C winner) ────────────────────────────
MODEL_TYPE="hand_bilstm"
CACHE_DIR="./MOSA_dataset/training_cache_W05"
HIDDEN_DIM=512
NUM_LAYERS=6
LR=1e-4
BATCH_SIZE=16
EPOCHS=500
PATIENCE=50
INPUT_LEVEL="L3"
WANDB_PROJECT="pianoid-loss"
EVAL_OUTPUT="results/loss_experiment_eval.json"

# ── Sweep grid (edit these to change search space) ──────────────────
#
# Each weight list is space-separated. The sweep runs all combinations.
# To sweep a single axis, set the other to one value.
#
SWEEP_LOSS_VERSION="v3"
SWEEP_VEL_W="2.0"
SWEEP_ONSET_W="2.0"
SWEEP_Z_W="2.0"
SWEEP_Z_RANGE_W="0.25 0.5 1.0 2.0"
SWEEP_Z_STD_W="0.25 0.5 1.0 2.0"

# ── Core functions ──────────────────────────────────────────────────

run_one() {
    local NAME=$1
    local LOSS_VER=$2
    local VEL_W=$3
    local ONSET_W=$4
    local Z_W=$5
    local Z_RANGE_W=$6
    local Z_STD_W=$7
    local OUTDIR="checkpoints/loss_sweep_${NAME}"

    # Skip if best checkpoint already exists
    if [ -f "${OUTDIR}/${MODEL_TYPE}_best.pt" ]; then
        echo "[SKIP] ${NAME} — checkpoint already exists at ${OUTDIR}/${MODEL_TYPE}_best.pt"
        return
    fi

    echo ""
    echo "============================================================"
    echo "Loss Sweep: ${NAME}"
    echo "  loss=${LOSS_VER} vel_w=${VEL_W} onset_w=${ONSET_W}"
    echo "  z_w=${Z_W} z_range_w=${Z_RANGE_W} z_std_w=${Z_STD_W}"
    echo "============================================================"

    mkdir -p "$OUTDIR"

    python src/stage1/s14_train_hand.py \
        --cache_dir "$CACHE_DIR" \
        --output_dir "$OUTDIR" \
        --model_type "$MODEL_TYPE" \
        --input_level "$INPUT_LEVEL" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --latent_dim $HIDDEN_DIM \
        --num_layers $NUM_LAYERS \
        --window_size 480 \
        --patience $PATIENCE \
        --loss_version "$LOSS_VER" \
        --onset_weight "$ONSET_W" \
        --z_weight "$Z_W" \
        --vel_w "$VEL_W" \
        --z_range_weight "$Z_RANGE_W" \
        --z_std_weight "$Z_STD_W" \
        --wandb_project "$WANDB_PROJECT"

    echo "  → Checkpoint: ${OUTDIR}/${MODEL_TYPE}_best.pt"
}

eval_all() {
    echo ""
    echo "============================================================"
    echo "Evaluating all loss sweep checkpoints"
    echo "============================================================"

    mkdir -p results
    local COUNT=0
    for CKPT in checkpoints/loss_sweep_*/hand_bilstm_best.pt; do
        if [ -f "$CKPT" ]; then
            echo "  Evaluating: $CKPT"
            python src/stage1/s16_evaluate.py \
                --checkpoint "$CKPT" \
                --cache_dir "$CACHE_DIR" \
                --output "$EVAL_OUTPUT" \
                --gpu 0
            COUNT=$((COUNT + 1))
        fi
    done

    if [ "$COUNT" -eq 0 ]; then
        echo "  No checkpoints found. Run sweep first."
        exit 1
    fi

    echo ""
    echo "Evaluated ${COUNT} checkpoints → ${EVAL_OUTPUT}"
}

# ── Named variants (for quick single runs) ─────────────────────────
#               NAME   LOSS  VEL_W  ONSET_W  Z_W  Z_RNG_W  Z_STD_W
VARIANTS=(
    "v1         v1    0.5    2.0      2.0  0.0      0.0"
    "v2         v2    2.0    2.0      2.0  0.0      0.0"
    "v3         v3    2.0    2.0      2.0  2.0      2.0"
    "v3a        v3    2.0    2.0      2.0  0.5      0.5"
    "v3b        v3    2.0    2.0      2.0  1.0      1.0"
)

run_named() {
    local TARGET=$1
    for V in "${VARIANTS[@]}"; do
        read -r NAME LOSS VEL_W ONSET_W Z_W Z_RNG_W Z_STD_W <<< "$V"
        if [ "$NAME" = "$TARGET" ]; then
            run_one "$NAME" "$LOSS" "$VEL_W" "$ONSET_W" "$Z_W" "$Z_RNG_W" "$Z_STD_W"
            return 0
        fi
    done
    return 1
}

list_variants() {
    echo "Named variants:"
    printf "  %-8s %-6s %-6s %-8s %-6s %-9s %-9s\n" \
        "NAME" "LOSS" "VEL_W" "ONSET_W" "Z_W" "Z_RNG_W" "Z_STD_W"
    echo "  ---------------------------------------------------------------"
    for V in "${VARIANTS[@]}"; do
        read -r NAME LOSS VEL_W ONSET_W Z_W Z_RNG_W Z_STD_W <<< "$V"
        printf "  %-8s %-6s %-6s %-8s %-6s %-9s %-9s\n" \
            "$NAME" "$LOSS" "$VEL_W" "$ONSET_W" "$Z_W" "$Z_RNG_W" "$Z_STD_W"
    done
    echo ""
    echo "Sweep grid (z_range_w × z_std_w):"
    echo "  z_range_w: ${SWEEP_Z_RANGE_W}"
    echo "  z_std_w:   ${SWEEP_Z_STD_W}"
    echo "  → $(($(echo $SWEEP_Z_RANGE_W | wc -w) * $(echo $SWEEP_Z_STD_W | wc -w))) combinations"
}

# ── Main ────────────────────────────────────────────────────────────

CMD="${1:-help}"
shift 2>/dev/null || true

case "$CMD" in
    sweep)
        echo "Starting grid sweep: z_range_w=[${SWEEP_Z_RANGE_W}] × z_std_w=[${SWEEP_Z_STD_W}]"
        for ZR in $SWEEP_Z_RANGE_W; do
            for ZS in $SWEEP_Z_STD_W; do
                NAME="v3_zr${ZR}_zs${ZS}"
                run_one "$NAME" "$SWEEP_LOSS_VERSION" "$SWEEP_VEL_W" \
                        "$SWEEP_ONSET_W" "$SWEEP_Z_W" "$ZR" "$ZS"
            done
        done
        eval_all
        ;;

    run)
        # Parse custom args: --vel_w 2.0 --z_range_w 0.5 --z_std_w 0.5 ...
        VEL_W=2.0; ONSET_W=2.0; Z_W=2.0; Z_RANGE_W=0.5; Z_STD_W=0.5
        LOSS_VER=v3; RUN_NAME=""
        while [ $# -gt 0 ]; do
            case "$1" in
                --loss_version) LOSS_VER=$2; shift 2 ;;
                --vel_w)        VEL_W=$2;    shift 2 ;;
                --onset_w)      ONSET_W=$2;  shift 2 ;;
                --z_w)          Z_W=$2;      shift 2 ;;
                --z_range_w)    Z_RANGE_W=$2; shift 2 ;;
                --z_std_w)      Z_STD_W=$2;  shift 2 ;;
                --name)         RUN_NAME=$2; shift 2 ;;
                *) echo "Unknown arg: $1"; exit 1 ;;
            esac
        done
        [ -z "$RUN_NAME" ] && RUN_NAME="${LOSS_VER}_zr${Z_RANGE_W}_zs${Z_STD_W}"
        run_one "$RUN_NAME" "$LOSS_VER" "$VEL_W" "$ONSET_W" "$Z_W" "$Z_RANGE_W" "$Z_STD_W"
        ;;

    eval)
        eval_all
        ;;

    list)
        list_variants
        ;;

    v1|v2|v3|v3a|v3b)
        run_named "$CMD" || { echo "Unknown variant: $CMD"; exit 1; }
        ;;

    help|*)
        echo "Usage: bash scripts/ablation/stage1_loss_sweep.sh <command>"
        echo ""
        echo "Commands:"
        echo "  sweep                               Grid sweep (z_range_w × z_std_w)"
        echo "  run --z_range_w 0.5 --z_std_w 0.5   Custom single run"
        echo "  v1 | v2 | v3 | v3a | v3b            Named variant"
        echo "  eval                                 Evaluate all checkpoints"
        echo "  list                                 Show variants & sweep grid"
        echo ""
        echo "Examples:"
        echo "  bash scripts/ablation/stage1_loss_sweep.sh sweep"
        echo "  bash scripts/ablation/stage1_loss_sweep.sh run --z_range_w 0.3 --z_std_w 0.7 --name custom1"
        echo "  bash scripts/ablation/stage1_loss_sweep.sh v3a"
        echo "  bash scripts/ablation/stage1_loss_sweep.sh eval"
        ;;
esac
