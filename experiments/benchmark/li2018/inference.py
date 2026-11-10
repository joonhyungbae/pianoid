#!/usr/bin/env python3
"""
experiments/benchmark/li2018/inference.py

Run inference and compute metrics for Li2018 baseline.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.benchmark.li2018.model import Li2018Model
from experiments.benchmark.li2018.dataset import get_li2018_test_loader
from src.evaluation import (
    compute_mpjpe,
    compute_mpjpe_hands,
    compute_jitter,
    compute_velocity_correlation,
    compute_fid_from_motions,
)


def load_model(checkpoint_path: str, device: str = 'cuda') -> Li2018Model:
    """Load trained Li2018 model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    
    model = Li2018Model(
        window_size=config.get('window_size', 120),
        piano_feat_dim=config.get('piano_feat_dim', 50),
        metric_feat_dim=config.get('metric_feat_dim', 10),
        lstm_hidden=config.get('lstm_hidden', 256),
        lstm_layers=config.get('lstm_layers', 2),
        use_metric=config.get('use_metric', True),
        dropout=config.get('dropout', 0.1),
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}, Val Loss: {ckpt.get('val_loss', 'N/A'):.4f}")
    
    return model


@torch.no_grad()
def run_inference(
    model: Li2018Model,
    test_loader,
    device: str,
    output_dir: Path,
) -> dict:
    """Run inference on test set and compute metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_gt = []
    all_pred = []
    all_metrics = []
    
    for batch in tqdm(test_loader, desc="Generating"):
        joints_gt = batch['joints'].to(device)  # [1, T, 78]
        midi = {k: v.to(device) for k, v in batch['midi_features'].items()}
        metric = batch['metric_matrix'].to(device)
        sample_idx = batch['sample_idx'][0].item()
        
        # Generate
        motion_pred = model.generate(midi, metric)  # [1, T, 26, 3]
        
        # To numpy
        gt = batch['joints_3d'][0].numpy()  # [T, 26, 3]
        pred = motion_pred[0].cpu().numpy()  # [T, 26, 3]
        onset = midi['onset'][0].cpu().numpy()  # [T, 88]
        
        # Save
        np.savez(
            output_dir / f'sample_{sample_idx:03d}.npz',
            gt_joints=gt,
            generated=pred,
        )
        
        # Compute metrics
        metrics = {
            'sample_idx': sample_idx,
            'mpjpe': compute_mpjpe(gt, pred),
            'mpjpe_hands': compute_mpjpe_hands(gt, pred),
            'jitter': compute_jitter(pred),
            'vel_corr': compute_velocity_correlation(gt, pred),
        }
        all_metrics.append(metrics)
        all_gt.append(gt)
        all_pred.append(pred)
        
        print(f"  Sample {sample_idx}: MPJPE={metrics['mpjpe']:.1f}mm, Jitter={metrics['jitter']:.2f}")
    
    # Aggregate metrics
    agg_metrics = {
        'mpjpe': np.mean([m['mpjpe'] for m in all_metrics]),
        'mpjpe_std': np.std([m['mpjpe'] for m in all_metrics]),
        'mpjpe_hands': np.mean([m['mpjpe_hands'] for m in all_metrics]),
        'jitter': np.mean([m['jitter'] for m in all_metrics]),
        'jitter_std': np.std([m['jitter'] for m in all_metrics]),
        'vel_corr': np.mean([m['vel_corr'] for m in all_metrics]),
        'num_samples': len(all_metrics),
    }
    
    # FID
    try:
        agg_metrics['fid'] = compute_fid_from_motions(all_gt, all_pred)
    except Exception as e:
        print(f"FID computation failed: {e}")
        agg_metrics['fid'] = -1
    
    return agg_metrics


def main():
    parser = argparse.ArgumentParser(description="Li2018 inference")
    parser.add_argument('--checkpoint', default='experiments/benchmark/li2018/checkpoints/best_model.pt')
    parser.add_argument('--cache_dir', default='MOSA_dataset/training_cache')
    parser.add_argument('--output_dir', default='experiments/benchmark/li2018/results')
    args = parser.parse_args()
    
    # Check checkpoint
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run train.py first.")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load test data
    test_loader = get_li2018_test_loader(args.cache_dir)
    
    # Run inference
    output_dir = Path(args.output_dir)
    metrics = run_inference(model, test_loader, device, output_dir)
    
    # Print results
    print("\n" + "=" * 50)
    print("Li2018 Results")
    print("=" * 50)
    print(f"MPJPE:      {metrics['mpjpe']:.1f} ± {metrics['mpjpe_std']:.1f} mm")
    print(f"MPJPE Hand: {metrics['mpjpe_hands']:.1f} mm")
    print(f"FID:        {metrics['fid']:.4f}")
    print(f"Jitter:     {metrics['jitter']:.2f} ± {metrics['jitter_std']:.2f}")
    print(f"Vel Corr:   {metrics['vel_corr']:.4f}")
    print(f"Samples:    {metrics['num_samples']}")
    print("=" * 50)
    
    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")


if __name__ == '__main__':
    main()
