#!/usr/bin/env python3
"""
experiments/benchmark/evaluate_all.py

Evaluate every baseline model and compare the results.

Usage:
    python experiments/benchmark/evaluate_all.py
    python experiments/benchmark/evaluate_all.py --output results/benchmark.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.benchmark.common import (
    get_test_loader,
    compute_all_metrics,
    compute_fid_from_motions,
    format_metrics,
)


# Paper Table 2 baselines + ablation variants (precomputed .npz outputs).
BASELINES = {
    'proposed': {
        'name': 'Proposed (Pianoid)',
        'results_dir': 'results/hierarchical_mdm',
        'precomputed': True,
    },
    'single_stage_mdm_zscore': {
        'name': 'Single-Stage MDM+zs',
        'results_dir': 'experiments/benchmark/single_stage_mdm_zscore/checkpoints/results',
        'precomputed': True,
    },
    'bilstm_backbone': {
        'name': 'BiLSTM-backbone',
        'results_dir': 'experiments/benchmark/bilstm_backbone/checkpoints/results',
        'precomputed': True,
    },
    'mdm_adapted': {
        'name': 'MDM-adapted',
        'results_dir': 'experiments/benchmark/mdm_adapted/checkpoints/results',
        'precomputed': True,
    },
    'edge_adapted': {
        'name': 'EDGE-adapted',
        'results_dir': 'experiments/benchmark/edge_adapted/checkpoints/results',
        'precomputed': True,
    },
    'li2018': {
        'name': 'Li2018',
        'results_dir': 'experiments/benchmark/li2018/results',
        'precomputed': True,
    },
}


def load_precomputed_results(results_dir: Path, test_loader) -> Dict[str, List[np.ndarray]]:
    """Load pre-computed results."""
    gt_motions = []
    pred_motions = []
    onsets = []
    
    for batch in test_loader:
        idx = batch['sample_idx'][0].item()
        
        # Load precomputed result
        result_path = results_dir / f'sample_{idx:03d}.npz'
        if not result_path.exists():
            print(f"Warning: {result_path} not found")
            continue
        
        data = np.load(result_path)
        gt = data['gt_joints']
        pred = data['generated']
        
        # Handle batch dimension
        if pred.ndim == 4 and pred.shape[0] == 1:
            pred = pred[0]
        
        gt_motions.append(gt)
        pred_motions.append(pred)
        onsets.append(batch['midi_features']['onset'][0].numpy())
    
    return {
        'gt': gt_motions,
        'pred': pred_motions,
        'onset': onsets,
    }


def evaluate_baseline(
    baseline_name: str,
    config: Dict,
    test_loader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Evaluate a single baseline."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {config['name']}")
    print('='*60)
    
    if config.get('precomputed', False):
        # Load precomputed results
        results_dir = Path(config['results_dir'])
        data = load_precomputed_results(results_dir, test_loader)
    else:
        # TODO: Load model and generate
        raise NotImplementedError(f"Model inference not implemented for {baseline_name}")
    
    # Compute metrics
    all_metrics = []
    
    for gt, pred, onset in tqdm(
        zip(data['gt'], data['pred'], data['onset']),
        total=len(data['gt']),
        desc="Computing metrics"
    ):
        metrics = compute_all_metrics(gt, pred, onset)
        all_metrics.append(metrics)
    
    # Aggregate
    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        aggregated[key] = float(np.mean(values))
    
    # Compute FID
    aggregated['fid'] = compute_fid_from_motions(data['gt'], data['pred'])
    
    print(f"\nResults for {config['name']}:")
    print(format_metrics(aggregated))
    
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baselines', type=str, default='all',
                        help='Comma-separated baseline names, or "all"')
    parser.add_argument('--cache_dir', type=str, default='MOSA_dataset/training_cache')
    parser.add_argument('--output', type=str, default='experiments/benchmark/results.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Select baselines
    if args.baselines == 'all':
        baseline_names = list(BASELINES.keys())
    else:
        baseline_names = args.baselines.split(',')
    
    # Load test data
    test_loader = get_test_loader(args.cache_dir)
    
    # Evaluate each baseline
    results = {}
    
    for name in baseline_names:
        if name not in BASELINES:
            print(f"Warning: Unknown baseline '{name}', skipping")
            continue
        
        try:
            config = BASELINES[name]
            metrics = evaluate_baseline(name, config, test_loader, args.device)
            results[name] = {
                'name': config['name'],
                'metrics': metrics,
            }
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print('='*60)
    
    # Print comparison table
    header = f"{'Model':<25} {'MPJPE':>8} {'FID':>8} {'Jitter':>8} {'VelC':>8}"
    print(header)
    print('-' * len(header))
    
    for name, data in results.items():
        m = data['metrics']
        row = f"{data['name']:<25} {m.get('mpjpe', 0):>8.1f} {m.get('fid', 0):>8.2f} "
        row += f"{m.get('jitter_pred', 0)*1e4:>8.1f} {m.get('vel_corr', 0):>8.2f}"
        print(row)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
