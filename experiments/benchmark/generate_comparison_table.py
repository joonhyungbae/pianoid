#!/usr/bin/env python3
"""
Generate comparison table from all benchmark results.

Usage:
    python experiments/benchmark/generate_comparison_table.py
"""

import json
from pathlib import Path
from typing import Dict

# Paper Table 2 benchmark baselines (and ablation variants).
MODELS = {
    'Li2018': 'experiments/benchmark/li2018/results/metrics.json',
    'MDM-adapted': 'experiments/benchmark/mdm_adapted/checkpoints/results/metrics.json',
    'EDGE-adapted': 'experiments/benchmark/edge_adapted/checkpoints/results/metrics.json',
    'Single-Stage MDM+zs': 'experiments/benchmark/single_stage_mdm_zscore/checkpoints/results/metrics.json',
    'BiLSTM-backbone': 'experiments/benchmark/bilstm_backbone/checkpoints/results/metrics.json',
    'Pianoid': 'experiments/benchmark/pianoid/results_final/metrics.json',
}


def load_metrics(path: str) -> Dict:
    """Load metrics from JSON file."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 80)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    
    # Load all metrics
    results = {}
    for name, path in MODELS.items():
        metrics = load_metrics(path)
        if metrics:
            results[name] = metrics
            print(f"✓ Loaded {name}")
        else:
            print(f"✗ Missing {name} ({path})")
    
    if not results:
        print("\nNo results found. Run evaluation first:")
        print("  bash experiments/benchmark/run_evaluation.sh")
        return
    
    print("\n")
    
    # Print table header
    header = f"{'Model':<20} {'MPJPE↓':>10} {'Hand↓':>10} {'FID↓':>10} {'Jitter↓':>12} {'OnsetSync↑':>12} {'VelCorr↑':>10}"
    print(header)
    print("-" * len(header))
    
    # Print each model's results
    for name, m in results.items():
        mpjpe = m.get('mpjpe', 0)
        mpjpe_hands = m.get('mpjpe_hands', 0)
        fid = m.get('fid', 0)
        jitter = m.get('jitter_pred', m.get('jitter', 0))
        vel = m.get('vel_corr', 0)
        
        # Format jitter (×10⁻⁴)
        jitter_fmt = f"{jitter*1e4:.1f}×10⁻⁴"
        
        row = f"{name:<20} {mpjpe:>10.1f} {mpjpe_hands:>10.1f} {fid:>10.2f} {jitter_fmt:>12} {vel:>10.2f}"
        print(row)
    
    print("-" * len(header))
    
    # Find best values
    best = {}
    for metric in ['mpjpe', 'fid', 'vel_corr']:
        values = [(name, m.get(metric, float('inf') if metric in ['mpjpe', 'fid'] else 0)) 
                  for name, m in results.items()]
        if metric in ['mpjpe', 'fid']:  # Lower is better
            best[metric] = min(values, key=lambda x: x[1])
        else:  # Higher is better
            best[metric] = max(values, key=lambda x: x[1])
    
    print("\nBest performers:")
    print(f"  MPJPE: {best['mpjpe'][0]} ({best['mpjpe'][1]:.1f} mm)")
    print(f"  FID: {best['fid'][0]} ({best['fid'][1]:.2f})")
    print(f"  Vel Corr: {best['vel_corr'][0]} ({best['vel_corr'][1]:.2f})")
    
    # Save comparison JSON
    output = {
        'models': results,
        'best': {k: {'model': v[0], 'value': v[1]} for k, v in best.items()}
    }
    
    output_path = Path('experiments/benchmark/comparison_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Generate LaTeX table
    print("\n\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    print(r"""
\begin{table}[t]
\centering
\caption{Comparison with baseline methods on piano motion generation.}
\label{tab:benchmark}
\begin{tabular}{lcccc}
\toprule
Method & MPJPE $\downarrow$ & FID $\downarrow$ & Jitter $\downarrow$ & Vel. Corr. $\uparrow$ \\
\midrule""")
    
    for name, m in results.items():
        mpjpe = m.get('mpjpe', 0)
        fid = m.get('fid', 0)
        jitter = m.get('jitter_pred', m.get('jitter', 0)) * 1e4
        vel = m.get('vel_corr', 0)
        
        print(f"{name} & {mpjpe:.1f} & {fid:.2f} & {jitter:.1f} & {vel:.2f} \\\\")
    
    print(r"""\midrule
\textbf{Proposed} & \textbf{XX.X} & \textbf{X.XX} & \textbf{X.X} & \textbf{X.XX} \\
\bottomrule
\end{tabular}
\end{table}""")


if __name__ == '__main__':
    main()
