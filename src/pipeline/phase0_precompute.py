#!/usr/bin/env python3
"""
Stage 1 prediction generation.

Loads a trained Stage 1 wrist checkpoint (paper: BiLSTM h512 l6) and writes
overlap-add wrist predictions for all train/val/test samples. Stage 2/3
training scripts read these predictions from MOSA_dataset/phase0_hand_predictions/.

Usage (called from scripts/train/stage1.sh after Stage 1 training):
    python src/pipeline/phase0_precompute.py \\
        --skip_train \\
        --checkpoint ./checkpoints/stage1/hand_bilstm_best.pt \\
        --cache_dir ./MOSA_dataset/training_cache_W05
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.constants import LWRIST_IDX, RWRIST_IDX
from src.models.model_factory import load_stage1


CACHE_DIR = ROOT / "MOSA_dataset" / "training_cache_W05"
PRED_DIR = ROOT / "MOSA_dataset" / "phase0_hand_predictions"

WINDOW_SIZE = 480
STRIDE = 240

# Must match s14_train_hand.py ordering: [LWRIST, RWRIST]
HAND_INDICES = [LWRIST_IDX, RWRIST_IDX]


def generate_predictions(checkpoint_path: Path):
    """Generate wrist predictions for all splits using the trained Stage 1 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_stage1(checkpoint_path, device)
    use_dur_ioi = config.get('use_dur_ioi', False)
    midi_dim = 352 if use_dur_ioi else 264

    print(f"Loaded model from {checkpoint_path} "
          f"(model_type={config.get('model_type')}, use_dur_ioi={use_dur_ioi})")

    with open(CACHE_DIR / 'metadata.json') as f:
        metadata = json.load(f)

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        split_info = metadata['splits'][split]
        split_dir = CACHE_DIR / split
        pred_split_dir = PRED_DIR / split
        pred_split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {split} predictions ({split_info['num_samples']} samples)...")

        for sample in tqdm(split_info['samples'], desc=split):
            idx = sample['idx']
            path = split_dir / f"{idx:05d}.npz"
            if not path.exists():
                continue

            data = np.load(path)
            joints = data['joints'].astype(np.float32)
            T = len(joints)

            ak = data['active_keys']
            pr = np.zeros((T, 88), dtype=np.float32)
            for t in range(T):
                for k in ak[t]:
                    if 0 <= k < 88:
                        pr[t, k] = 1.0
            vel = data.get('onset_density_per_key', np.zeros((T, 88))).astype(np.float32)
            ons = data.get('key_onset', np.zeros((T, 88))).astype(np.float32)
            parts = [pr, vel, ons]
            if use_dur_ioi:
                parts.append(data.get('dur_ioi_ratio', np.zeros((T, 88))).astype(np.float32))
            midi_np = np.concatenate(parts, axis=-1)

            pred_sum = np.zeros((T, 6), dtype=np.float64)
            weight_sum = np.zeros((T, 1), dtype=np.float64)

            with torch.no_grad():
                for s in range(0, max(1, T - WINDOW_SIZE + 1), STRIDE):
                    e = min(s + WINDOW_SIZE, T)
                    chunk = midi_np[s:e]
                    actual_len = e - s
                    if actual_len < WINDOW_SIZE:
                        chunk = np.concatenate(
                            [chunk, np.zeros((WINDOW_SIZE - actual_len, midi_dim), dtype=np.float32)]
                        )
                    ct = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
                    md = {
                        'piano_roll': ct[:, :, :88],
                        'velocity':   ct[:, :, 88:176],
                        'onset':      ct[:, :, 176:264],
                    }
                    if midi_dim > 264:
                        md['dur_ioi_ratio'] = ct[:, :, 264:352]

                    pred = model(md).cpu().numpy()[0]
                    w = np.minimum(np.arange(actual_len) + 1,
                                   np.arange(actual_len, 0, -1)).astype(np.float64).reshape(-1, 1)
                    pred_sum[s:e] += pred[:actual_len] * w
                    weight_sum[s:e] += w

            mask = weight_sum > 0
            hand_pred = np.where(mask, pred_sum / weight_sum, 0).astype(np.float32)

            np.savez_compressed(
                pred_split_dir / f"{idx:05d}.npz",
                hand_pred=hand_pred,
                hand_gt=joints[:, HAND_INDICES].reshape(T, -1),
                sample_info=json.dumps(sample),
            )

        print(f"  Saved to {pred_split_dir}")

    print(f"\nAll predictions saved to {PRED_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 → wrist predictions for all splits")
    parser.add_argument('--skip_train', action='store_true',
                        help='Compatibility flag (training is no longer in this script; always required).')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained Stage 1 checkpoint (e.g. checkpoints/stage1/hand_bilstm_best.pt).')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Override training cache directory (default: MOSA_dataset/training_cache_W05).')
    args = parser.parse_args()

    global CACHE_DIR
    if args.cache_dir:
        CACHE_DIR = Path(args.cache_dir)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print("=" * 70)
    print("Stage 1 prediction generation")
    print("=" * 70)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Cache:      {CACHE_DIR}")
    print(f"Output:     {PRED_DIR}")
    print()

    generate_predictions(ckpt_path)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"Predictions: {PRED_DIR}/{{train,val,test}}/")
    print("\nNext: bash scripts/train/stage2.sh")


if __name__ == "__main__":
    main()
