#!/bin/bash
# Stage 2 Torso Mega-Sweep: 3-Round Staged Experiment
#
# Round 1: Representation (zscore/delta/residual/augmentation)
# Round 2: Architecture (BiLSTM/Mamba/Transformer with best repr)
# Round 3: Loss Refinement (bone/spectral/range/std with best repr+arch)
#
# Usage:
#   bash scripts/ablation/stage2_mega_sweep.sh all        # full 3-round sweep
#   bash scripts/ablation/stage2_mega_sweep.sh round1     # only Round 1
#   bash scripts/ablation/stage2_mega_sweep.sh round2     # only Round 2 (needs R1 results)
#   bash scripts/ablation/stage2_mega_sweep.sh round3     # only Round 3 (needs R1+R2 results)
#   bash scripts/ablation/stage2_mega_sweep.sh eval1      # evaluate Round 1 only
#   bash scripts/ablation/stage2_mega_sweep.sh eval2      # evaluate Round 2 only
#   bash scripts/ablation/stage2_mega_sweep.sh eval3      # evaluate Round 3 only

set -e
cd "$(dirname "$0")/../.."

GPU_ID=1
export CUDA_VISIBLE_DEVICES=$GPU_ID

CACHE_DIR="./MOSA_dataset/training_cache_W10"
PRED_DIR="./MOSA_dataset/phase0_hand_predictions"
OUTBASE="checkpoints/stage2_mega_sweep"
WANDB_PROJECT="pianoid-stage2-mega"
EVAL_DIR="results"

EPOCHS=500
PATIENCE=50
BATCH_SIZE=16

mkdir -p "$OUTBASE" "$EVAL_DIR"

# ── Helper: find best from eval JSON → checkpoint name ──
best_from_json() {
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
data.pop('_best', None)
best = min(data, key=lambda k: data[k]['mpjpe'])
print(best)
" "$1"
}

# ── Helper: extract repr/aug flags from best R1 checkpoint config ──
extract_r1_flags() {
    python3 -c "
import sys, torch
from pathlib import Path
best_name = sys.argv[1]
ckpt = torch.load(str(Path(sys.argv[2]) / best_name / 'best.pt'), map_location='cpu', weights_only=False)
cfg = ckpt.get('config', {})
flags = []
r = cfg.get('repr', 'absolute')
if r != 'absolute':
    flags.append(f'--repr {r}')
if cfg.get('mirror_aug', False):
    flags.append('--mirror_aug')
if cfg.get('hand_velocity', False):
    flags.append('--hand_velocity')
print(' '.join(flags))
" "$1" "$2"
}

# ── Helper: extract arch flags from best R2 checkpoint config ──
extract_r2_flags() {
    python3 -c "
import sys, torch
from pathlib import Path
best_name = sys.argv[1]
ckpt = torch.load(str(Path(sys.argv[2]) / best_name / 'best.pt'), map_location='cpu', weights_only=False)
cfg = ckpt.get('config', {})
print(f\"--arch {cfg.get('arch','lstm')} --latent_dim {cfg.get('latent_dim',512)} --num_layers {cfg.get('num_layers',6)} --lr {cfg.get('lr',1e-4)}\")
" "$1" "$2"
}

# ── Helper: evaluate all checkpoints in a directory ──
eval_round() {
    local ROUND_DIR="$1"
    local OUTPUT_JSON="$2"
    echo ""
    echo "Evaluating checkpoints in $ROUND_DIR ..."
    for CKPT in "$ROUND_DIR"/*/best.pt; do
        if [ -f "$CKPT" ]; then
            echo "  → $CKPT"
            python src/stage2/s22_evaluate.py \
                --checkpoint "$CKPT" \
                --cache_dir "$CACHE_DIR" --pred_dir "$PRED_DIR" \
                --output "$OUTPUT_JSON" --gpu 0
        fi
    done
    echo ""
    echo "Results saved to $OUTPUT_JSON"
}

# ── Helper: print evaluation summary ──
print_summary() {
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
data.pop('_best', None)
print(f'{\"Name\":<60} {\"MPJPE\":>7} {\"SpdCorr\":>8} {\"SpdR\":>6} {\"JerkR\":>7} {\"RngX\":>6} {\"RngY\":>6} {\"RngZ\":>6}')
print('-' * 110)
for k in sorted(data, key=lambda x: data[x]['mpjpe']):
    r = data[k]
    print(f'{k:<60} {r[\"mpjpe\"]:>7.2f} {r.get(\"speed_corr\",0):>8.3f} {r.get(\"speed_ratio\",0):>6.2f} {r.get(\"jerk_ratio\",0):>7.2f} {r.get(\"range_ratio_x\",0):>6.2f} {r.get(\"range_ratio_y\",0):>6.2f} {r.get(\"range_ratio_z\",0):>6.2f}')
best = min(data, key=lambda k: data[k]['mpjpe'])
print('-' * 110)
print(f'Best: {best} (MPJPE={data[best][\"mpjpe\"]:.2f}mm)')
" "$1"
}


# ============================================================
# ROUND 1: Representation (6 variants)
# Fixed: BiLSTM h512 l6, L1 + vel 2.0, onset_density
# ============================================================
round1() {
    echo ""
    echo "============================================================"
    echo "  ROUND 1: Representation Sweep (6 variants)"
    echo "  Fixed: BiLSTM h512 l6, L1 + vel 2.0"
    echo "============================================================"

    local COMMON="--arch lstm --latent_dim 512 --num_layers 6 --lr 1e-4 \
        --use_onset_density --use_l1 --vel_w 2.0 \
        --cache_dir $CACHE_DIR --pred_dir $PRED_DIR \
        --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE \
        --wandb_project $WANDB_PROJECT --seed 42"

    echo "--- R1-1: absolute (baseline) ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --repr absolute --output_dir "$OUTBASE/R1"

    echo "--- R1-2: zscore ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --repr zscore --output_dir "$OUTBASE/R1"

    echo "--- R1-3: delta ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --repr delta --output_dir "$OUTBASE/R1"

    echo "--- R1-4: residual ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --repr residual --output_dir "$OUTBASE/R1"

    echo "--- R1-5: zscore + mirror ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --repr zscore --mirror_aug --output_dir "$OUTBASE/R1"

    echo "--- R1-6: zscore + mirror + hand_vel ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --repr zscore --mirror_aug --hand_velocity --output_dir "$OUTBASE/R1"

    eval_round "$OUTBASE/R1" "$EVAL_DIR/stage2_mega_R1.json"
    echo ""
    echo "=== ROUND 1 RESULTS ==="
    print_summary "$EVAL_DIR/stage2_mega_R1.json"

    BEST_R1=$(best_from_json "$EVAL_DIR/stage2_mega_R1.json")
    echo ""
    echo "Round 1 winner: $BEST_R1"
    echo "$BEST_R1" > "$OUTBASE/R1_best.txt"
}

eval1() {
    eval_round "$OUTBASE/R1" "$EVAL_DIR/stage2_mega_R1.json"
    echo "=== ROUND 1 RESULTS ==="
    print_summary "$EVAL_DIR/stage2_mega_R1.json"
    BEST_R1=$(best_from_json "$EVAL_DIR/stage2_mega_R1.json")
    echo "Round 1 winner: $BEST_R1"
    echo "$BEST_R1" > "$OUTBASE/R1_best.txt"
}


# ============================================================
# ROUND 2: Architecture (6 variants)
# Uses best repr from Round 1, fixed loss: L1 + vel 2.0
# ============================================================
round2() {
    if [ ! -f "$OUTBASE/R1_best.txt" ]; then
        echo "ERROR: Round 1 results not found. Run round1 first."
        exit 1
    fi
    BEST_R1=$(cat "$OUTBASE/R1_best.txt")
    R1_FLAGS=$(extract_r1_flags "$BEST_R1" "$OUTBASE/R1")

    echo ""
    echo "============================================================"
    echo "  ROUND 2: Architecture Sweep (6 variants)"
    echo "  Using R1 best repr: $BEST_R1"
    echo "  Repr flags: $R1_FLAGS"
    echo "============================================================"

    local COMMON="--use_onset_density --use_l1 --vel_w 2.0 \
        --cache_dir $CACHE_DIR --pred_dir $PRED_DIR \
        --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE \
        --wandb_project $WANDB_PROJECT --seed 42 $R1_FLAGS"

    echo "--- R2-1: BiLSTM h512 l6 ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --arch lstm --latent_dim 512 --num_layers 6 --lr 1e-4 \
        --output_dir "$OUTBASE/R2"

    echo "--- R2-2: Mamba h512 l8 ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --arch mamba --latent_dim 512 --num_layers 8 --lr 5e-4 \
        --output_dir "$OUTBASE/R2"

    echo "--- R2-3: Transformer h512 l8 ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --arch transformer --latent_dim 512 --num_layers 8 --lr 5e-4 \
        --output_dir "$OUTBASE/R2"

    echo "--- R2-4: Mamba h512 l4 (light) ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --arch mamba --latent_dim 512 --num_layers 4 --lr 5e-4 \
        --output_dir "$OUTBASE/R2"

    echo "--- R2-5: BiLSTM h768 l4 (wide) ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --arch lstm --latent_dim 768 --num_layers 4 --lr 1e-4 \
        --output_dir "$OUTBASE/R2"

    echo "--- R2-6: Mamba h512 l8 + mirror + hand_vel ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --arch mamba --latent_dim 512 --num_layers 8 --lr 5e-4 \
        --mirror_aug --hand_velocity \
        --output_dir "$OUTBASE/R2"

    eval_round "$OUTBASE/R2" "$EVAL_DIR/stage2_mega_R2.json"
    echo ""
    echo "=== ROUND 2 RESULTS ==="
    print_summary "$EVAL_DIR/stage2_mega_R2.json"

    BEST_R2=$(best_from_json "$EVAL_DIR/stage2_mega_R2.json")
    echo ""
    echo "Round 2 winner: $BEST_R2"
    echo "$BEST_R2" > "$OUTBASE/R2_best.txt"
}

eval2() {
    eval_round "$OUTBASE/R2" "$EVAL_DIR/stage2_mega_R2.json"
    echo "=== ROUND 2 RESULTS ==="
    print_summary "$EVAL_DIR/stage2_mega_R2.json"
    BEST_R2=$(best_from_json "$EVAL_DIR/stage2_mega_R2.json")
    echo "Round 2 winner: $BEST_R2"
    echo "$BEST_R2" > "$OUTBASE/R2_best.txt"
}


# ============================================================
# ROUND 3: Loss Refinement (6 variants)
# Uses best repr + arch from Rounds 1-2
# ============================================================
round3() {
    if [ ! -f "$OUTBASE/R1_best.txt" ] || [ ! -f "$OUTBASE/R2_best.txt" ]; then
        echo "ERROR: Need Round 1+2 results. Run round1 and round2 first."
        exit 1
    fi
    BEST_R1=$(cat "$OUTBASE/R1_best.txt")
    BEST_R2=$(cat "$OUTBASE/R2_best.txt")
    R1_FLAGS=$(extract_r1_flags "$BEST_R1" "$OUTBASE/R1")
    R2_FLAGS=$(extract_r2_flags "$BEST_R2" "$OUTBASE/R2")

    echo ""
    echo "============================================================"
    echo "  ROUND 3: Loss Refinement (6 variants)"
    echo "  Using R1 repr: $BEST_R1 → $R1_FLAGS"
    echo "  Using R2 arch: $BEST_R2 → $R2_FLAGS"
    echo "============================================================"

    local COMMON="--use_onset_density --use_l1 \
        --cache_dir $CACHE_DIR --pred_dir $PRED_DIR \
        --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE \
        --wandb_project $WANDB_PROJECT --seed 42 $R1_FLAGS $R2_FLAGS"

    echo "--- R3-1: L1 + vel 2.0 (baseline) ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --vel_w 2.0 --output_dir "$OUTBASE/R3"

    echo "--- R3-2: + bone 1.0 ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --vel_w 2.0 --bone_w 1.0 --output_dir "$OUTBASE/R3"

    echo "--- R3-3: + spectral 0.5 ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --vel_w 2.0 --spectral_w 0.5 --output_dir "$OUTBASE/R3"

    echo "--- R3-4: + bone + spectral ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --vel_w 2.0 --bone_w 1.0 --spectral_w 0.5 --output_dir "$OUTBASE/R3"

    echo "--- R3-5: + range 1.0 + std 1.0 ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --vel_w 2.0 --range_w 1.0 --std_w 1.0 --output_dir "$OUTBASE/R3"

    echo "--- R3-6: bone + spectral + range + std (all) ---"
    python src/stage2/s20_train_torso_sweep.py $COMMON \
        --vel_w 2.0 --bone_w 1.0 --spectral_w 0.5 --range_w 1.0 --std_w 1.0 \
        --output_dir "$OUTBASE/R3"

    eval_round "$OUTBASE/R3" "$EVAL_DIR/stage2_mega_R3.json"
    echo ""
    echo "=== ROUND 3 RESULTS ==="
    print_summary "$EVAL_DIR/stage2_mega_R3.json"

    BEST_R3=$(best_from_json "$EVAL_DIR/stage2_mega_R3.json")
    echo ""
    echo "Round 3 winner: $BEST_R3"
    echo "$BEST_R3" > "$OUTBASE/R3_best.txt"
}

eval3() {
    eval_round "$OUTBASE/R3" "$EVAL_DIR/stage2_mega_R3.json"
    echo "=== ROUND 3 RESULTS ==="
    print_summary "$EVAL_DIR/stage2_mega_R3.json"
    BEST_R3=$(best_from_json "$EVAL_DIR/stage2_mega_R3.json")
    echo "Round 3 winner: $BEST_R3"
    echo "$BEST_R3" > "$OUTBASE/R3_best.txt"
}


# ============================================================
# Final summary
# ============================================================
final_summary() {
    echo ""
    echo "============================================================"
    echo "  MEGA-SWEEP FINAL SUMMARY"
    echo "============================================================"
    [ -f "$OUTBASE/R1_best.txt" ] && echo "  Round 1 (Repr):  $(cat $OUTBASE/R1_best.txt)"
    [ -f "$OUTBASE/R2_best.txt" ] && echo "  Round 2 (Arch):  $(cat $OUTBASE/R2_best.txt)"
    [ -f "$OUTBASE/R3_best.txt" ] && echo "  Round 3 (Loss):  $(cat $OUTBASE/R3_best.txt)"
    echo ""
    for R in R1 R2 R3; do
        JSON="$EVAL_DIR/stage2_mega_${R}.json"
        if [ -f "$JSON" ]; then
            echo "--- Round ${R} ---"
            print_summary "$JSON"
            echo ""
        fi
    done
    echo "============================================================"
    echo "  Done! Best checkpoint per round saved in $OUTBASE/R*_best.txt"
    echo "============================================================"
}


# ── Main ────────────────────────────────────────────────────
CMD="${1:-help}"

case "$CMD" in
    all)
        round1
        round2
        round3
        final_summary
        ;;
    round1) round1 ;;
    round2) round2 ;;
    round3) round3 ;;
    eval1)  eval1 ;;
    eval2)  eval2 ;;
    eval3)  eval3 ;;
    summary) final_summary ;;
    help|*)
        echo "Usage: bash scripts/ablation/stage2_mega_sweep.sh <command>"
        echo ""
        echo "Commands:"
        echo "  all       Full 3-round sweep (Round 1 → 2 → 3)"
        echo "  round1    Round 1 only: Representation sweep (6 variants)"
        echo "  round2    Round 2 only: Architecture sweep (needs R1 results)"
        echo "  round3    Round 3 only: Loss refinement (needs R1+R2 results)"
        echo "  eval1     Evaluate Round 1 checkpoints"
        echo "  eval2     Evaluate Round 2 checkpoints"
        echo "  eval3     Evaluate Round 3 checkpoints"
        echo "  summary   Print final summary of all rounds"
        echo ""
        echo "Examples:"
        echo "  bash scripts/ablation/stage2_mega_sweep.sh all      # full sweep"
        echo "  bash scripts/ablation/stage2_mega_sweep.sh round1   # representation only"
        echo "  bash scripts/ablation/stage2_mega_sweep.sh eval1    # re-eval Round 1"
        ;;
esac
