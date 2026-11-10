#!/bin/bash
# Stage-2 loss-weight sweep to hunt for the original paper's ER regime.
#
# Release checkpoint (rng0.5_std0.5) reproduces MPJPE and per-body-part
# numbers closely but converges to an under-expressive ER regime
# (Δ1≈0.42 vs paper 0.27, ER-jerk 0.34 vs 0.58). The paper's Table 3A
# row for the final model is one cell of a broader loss-weight sweep we
# now revisit. For each variant this script:
#
#   1. Trains Stage 2 (scripts/train/stage2.sh variant <name>)
#   2. Runs s21_precompute_torso with DDIM 45 → phase1_torso_predictions_<tag>/
#   3. Runs full_pipeline with the new Stage-2 checkpoint → full_body_predictions_<tag>/
#   4. Computes ER + bootstrap CI → results/er_<tag>.json
#
# After all variants finish, `results/er_<tag>.json` can be compared to the
# paper targets with `scripts/ablation/summarize_stage2_er_sweep.py`
# (auto-created below).
#
# Usage:
#   bash scripts/ablation/stage2_loss_weight_sweep.sh                   # all 3
#   bash scripts/ablation/stage2_loss_weight_sweep.sh mdm_zs_rng1_std1  # single

set -e
cd "$(dirname "$0")/../.."

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export WANDB_MODE="${WANDB_MODE:-offline}"

PY=/root/miniconda3/envs/pianoid/bin/python

# Variants to sweep (must match names in scripts/train/stage2.sh VARIANTS)
VARIANTS=(
    "mdm_zs_rng1_std1"
    "mdm_zs_rng2_std2"
    "mdm_zs_rng1_spd1"
)

# Allow single-variant invocation
if [ -n "${1:-}" ]; then
    VARIANTS=("$1")
fi

# Fixed paths that full_pipeline.py expects
STAGE3_CKPT="checkpoints/stage3_arm/bilstm_rel_od_pr_h256_l6_lr0.0001/best.pt"
STAGE4_CKPT="checkpoints/stage4_experiments/film_spine/best.pt"

for VARIANT in "${VARIANTS[@]}"; do
    TAG="$VARIANT"

    echo ""
    echo "============================================================"
    echo " [$TAG] step 1/4 — training Stage 2"
    echo "============================================================"
    SWEEP_START=$(date +%s)
    bash scripts/train/stage2.sh variant "$VARIANT"

    # Training writes to a name that includes the full hp string
    # (e.g. mdm_od_h256_l8_lr0.0001_rng1.0_std1.0_zscore). Since the
    # sweep runs sequentially we pick the freshest best.pt newer than
    # the start of this step.
    CKPT=$(ls -t checkpoints/stage2_backbone/mdm_od_h256_l8_lr0.0001_*_zscore/best.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "ERROR: no Stage-2 checkpoint produced for $VARIANT"
        exit 1
    fi
    # Sanity: ckpt must have been written after this step started
    MTIME=$(stat -c %Y "$CKPT")
    if [ "$MTIME" -lt "$SWEEP_START" ]; then
        echo "ERROR: newest ckpt $CKPT is older than sweep step start"
        exit 1
    fi
    echo "  Found Stage-2 ckpt: $CKPT"

    PHASE1_OUT="MOSA_dataset/phase1_torso_predictions_${TAG}"
    PIPE_OUT="MOSA_dataset/full_body_predictions_${TAG}"
    ER_OUT="results/er_${TAG}.json"
    PIPE_JSON="results/full_pipeline_${TAG}.json"

    echo ""
    echo "============================================================"
    echo " [$TAG] step 2/4 — s21 precompute torso (DDIM 45)"
    echo "============================================================"
    $PY src/stage2/s21_precompute_torso.py \
        --torso_path "$CKPT" \
        --ddim_steps 45 \
        --output_dir "$PHASE1_OUT"

    echo ""
    echo "============================================================"
    echo " [$TAG] step 3/4 — full pipeline (DDIM 45)"
    echo "============================================================"
    $PY src/pipeline/full_pipeline.py \
        --ddim_steps 45 \
        --stage2_ckpt "$CKPT" \
        --stage3_ckpt "$STAGE3_CKPT" \
        --stage4_ckpt "$STAGE4_CKPT" \
        --save_motion --save_dir "$PIPE_OUT" \
        --output "$PIPE_JSON"

    echo ""
    echo "============================================================"
    echo " [$TAG] step 4/4 — ER + bootstrap CI"
    echo "============================================================"
    $PY src/evaluation/compute_full_pipeline_er.py \
        --pred_dir "$PIPE_OUT/test" \
        --output "$ER_OUT" \
        --run_tag "$TAG"

    echo ""
    echo ">>> [$TAG] complete. Metrics in $ER_OUT"
done

echo ""
echo "============================================================"
echo " Stage-2 loss-weight sweep DONE"
echo "============================================================"
ls -la results/er_mdm_zs_*.json 2>/dev/null || true
