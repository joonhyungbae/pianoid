#!/usr/bin/env python3
"""
per_sequence_analysis.py: Per-sequence metrics and distribution analysis

Computes metrics for each test sequence individually and generates:
1. Per-sequence metric distributions (histograms, box plots)
2. Comparison between model variants
3. Statistical summaries (mean ± std)
4. Difficulty-based analysis (if available)

Usage:
    python evaluation/statistical/per_sequence_analysis.py
    python evaluation/statistical/per_sequence_analysis.py --variants=full_model,single_stage
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import NUM_JOINTS
from src.evaluation import (
    compute_mpjpe, compute_jitter,
    compute_velocity_correlation, compute_motion_features, compute_fid,
    load_npz_result as _load_npz_result,
)

# Matplotlib settings for paper-quality figures
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150


def compute_per_sequence_metrics(
    results_dir: Path,
    cache_dir: Path,
) -> Dict[str, List[float]]:
    """Compute metrics for each test sequence.
    
    Args:
        results_dir: Path to inference results (*.npz files)
        cache_dir: Path to training cache (for MIDI data)
    
    Returns:
        Dict mapping metric name to list of per-sequence values
    """
    npz_files = sorted(results_dir.glob('*.npz'))
    if not npz_files:
        print(f"No result files found in {results_dir}")
        return {}
    
    metrics = {
        'mpjpe': [],
        'mpjpe_hands': [],
        'mpjpe_body': [],
        'jitter': [],
        'jitter_ratio': [],  # pred_jitter / gt_jitter
        'vel_corr': [],
        'sequence_id': [],
    }
    
    cache_path = Path(cache_dir)
    
    for npz_file in tqdm(npz_files, desc="Computing per-sequence metrics"):
        data = np.load(npz_file)
        gt, pred, idx = _load_npz_result(data, npz_file)
        
        # Ensure correct shape
        if gt.ndim == 2:
            gt = gt.reshape(-1, NUM_JOINTS, 3)
        if pred.ndim == 2:
            pred = pred.reshape(-1, NUM_JOINTS, 3)
        
        # MPJPE (overall)
        mpjpe = compute_mpjpe(gt, pred)
        metrics['mpjpe'].append(mpjpe)
        
        # MPJPE (hands only: indices 10, 11, 14, 15)
        hand_indices = [10, 11, 14, 15]
        mpjpe_hands = compute_mpjpe(gt[:, hand_indices], pred[:, hand_indices])
        metrics['mpjpe_hands'].append(mpjpe_hands)
        
        # MPJPE (body only: excluding hands)
        body_indices = [i for i in range(NUM_JOINTS) if i not in hand_indices]
        mpjpe_body = compute_mpjpe(gt[:, body_indices], pred[:, body_indices])
        metrics['mpjpe_body'].append(mpjpe_body)
        
        # Jitter
        jitter_pred = compute_jitter(pred)
        jitter_gt = compute_jitter(gt)
        metrics['jitter'].append(jitter_pred)
        metrics['jitter_ratio'].append(jitter_pred / (jitter_gt + 1e-8))
        
        # Velocity correlation
        vel_corr = compute_velocity_correlation(gt, pred)
        metrics['vel_corr'].append(vel_corr)
        
        metrics['sequence_id'].append(idx)
    
    return metrics


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistical summary."""
    values = np.array(values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
    }


def significance_test(values1: List[float], values2: List[float]) -> Dict[str, float]:
    """Perform significance tests between two groups.
    
    Returns:
        Dict with t-test p-value, Wilcoxon p-value, and effect size (Cohen's d)
    """
    values1 = np.array(values1)
    values2 = np.array(values2)
    
    # Check normality
    _, p_norm1 = stats.shapiro(values1[:50] if len(values1) > 50 else values1)
    _, p_norm2 = stats.shapiro(values2[:50] if len(values2) > 50 else values2)
    
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(values1, values2)
    
    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pvalue = stats.wilcoxon(values1, values2)
    except ValueError:
        w_pvalue = 1.0
    
    # Effect size (Cohen's d)
    diff = values1 - values2
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-8)
    
    return {
        't_test_pvalue': float(t_pvalue),
        'wilcoxon_pvalue': float(w_pvalue),
        'cohens_d': float(cohens_d),
        'is_normal_1': p_norm1 > 0.05,
        'is_normal_2': p_norm2 > 0.05,
        'mean_diff': float(np.mean(diff)),
        'std_diff': float(np.std(diff)),
    }


def plot_distributions(
    metrics_dict: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
):
    """Generate distribution plots comparing variants.
    
    Args:
        metrics_dict: Dict mapping variant_name -> metrics
        output_dir: Path to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    variants = list(metrics_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    
    # 1. MPJPE Box Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    mpjpe_data = [metrics_dict[v]['mpjpe'] for v in variants]
    bp = ax.boxplot(mpjpe_data, labels=variants, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('MPJPE (mm)')
    ax.set_title('Per-Sequence MPJPE Distribution')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mpjpe_boxplot.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'mpjpe_boxplot.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. MPJPE Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, v in enumerate(variants):
        ax.hist(metrics_dict[v]['mpjpe'], bins=20, alpha=0.5, label=v, color=colors[i])
    ax.set_xlabel('MPJPE (mm)')
    ax.set_ylabel('Count')
    ax.set_title('MPJPE Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mpjpe_histogram.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'mpjpe_histogram.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Jitter Ratio Box Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    jitter_data = [metrics_dict[v].get('jitter_ratio', [1.0] * len(metrics_dict[v]['mpjpe'])) for v in variants]
    bp = ax.boxplot(jitter_data, labels=variants, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='GT level')
    ax.set_ylabel('Jitter Ratio (pred/GT)')
    ax.set_title('Motion Smoothness (1.0 = same as GT)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'jitter_ratio_boxplot.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'jitter_ratio_boxplot.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 4. Multi-metric comparison
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    metric_configs = [
        ('mpjpe', 'MPJPE (mm)', axes[0, 0]),
        ('mpjpe_hands', 'Hand MPJPE (mm)', axes[0, 1]),
        ('vel_corr', 'Velocity Correlation', axes[1, 0]),
        ('jitter', 'Jitter', axes[1, 1]),
    ]
    
    for metric_key, metric_label, ax in metric_configs:
        data = [metrics_dict[v].get(metric_key, []) for v in variants]
        data = [d for d in data if d]  # Filter empty
        if data:
            bp = ax.boxplot(data, labels=variants[:len(data)], patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
            ax.set_ylabel(metric_label)
            ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Per-Sequence Metric Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_metric_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'multi_metric_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 5. Scatter plot: Hand MPJPE vs Body MPJPE
    if len(variants) >= 1:
        fig, ax = plt.subplots(figsize=(7, 7))
        for i, v in enumerate(variants):
            if 'mpjpe_hands' in metrics_dict[v] and 'mpjpe_body' in metrics_dict[v]:
                ax.scatter(
                    metrics_dict[v]['mpjpe_hands'],
                    metrics_dict[v]['mpjpe_body'],
                    alpha=0.6,
                    label=v,
                    color=colors[i],
                )
        ax.set_xlabel('Hand MPJPE (mm)')
        ax.set_ylabel('Body MPJPE (mm)')
        ax.set_title('Hand vs Body Error')
        ax.plot([0, 200], [0, 200], 'k--', alpha=0.3)  # Diagonal
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'hand_vs_body_scatter.pdf', bbox_inches='tight')
        plt.savefig(output_dir / 'hand_vs_body_scatter.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"Plots saved to: {output_dir}")


def generate_latex_table_with_std(
    metrics_dict: Dict[str, Dict[str, List[float]]],
    output_path: Path,
):
    """Generate LaTeX table with mean ± std."""
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Per-sequence evaluation metrics (mean $\pm$ std).}',
        r'\label{tab:per_sequence}',
        r'\small',
        r'\begin{tabular}{lccccc}',
        r'\toprule',
        r'Variant & MPJPE$\downarrow$ & FID$\downarrow$ & Jitter$\downarrow$ & Vel$\uparrow$ \\',
        r'\midrule',
    ]
    
    for variant, metrics in metrics_dict.items():
        # Compute statistics
        mpjpe_stats = compute_statistics(metrics['mpjpe'])
        jitter_stats = compute_statistics(metrics['jitter'])
        vel_stats = compute_statistics(metrics['vel_corr'])
        
        # Format row
        row = f"{variant} & "
        row += f"${mpjpe_stats['mean']:.1f} \\pm {mpjpe_stats['std']:.1f}$ & "
        row += f"— & "  # FID is computed globally, not per-sequence
        row += f"${jitter_stats['mean']:.2f} \\pm {jitter_stats['std']:.2f}$ & "
        row += f"${vel_stats['mean']:.2f} \\pm {vel_stats['std']:.2f}$ \\\\"
        
        lines.append(row)
    
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    latex = '\n'.join(lines)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"LaTeX table saved to: {output_path}")
    return latex


def main():
    parser = argparse.ArgumentParser(description='Per-sequence analysis')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Base directory for inference results')
    parser.add_argument('--cache_dir', type=str, default='MOSA_dataset/training_cache')
    parser.add_argument('--output_dir', type=str, default='evaluation/statistical/outputs')
    parser.add_argument('--variants', type=str, default='full_model',
                        help='Comma-separated list of variants to analyze')
    args = parser.parse_args()
    
    results_base = Path(args.results_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    
    variants = [v.strip() for v in args.variants.split(',')]
    
    print("="*60)
    print("Per-Sequence Analysis")
    print("="*60)
    
    # Compute metrics for each variant
    all_metrics = {}
    
    for variant in variants:
        variant_dir = results_base / variant
        if not variant_dir.exists():
            print(f"Warning: Variant directory not found: {variant_dir}")
            continue
        
        print(f"\nProcessing: {variant}")
        metrics = compute_per_sequence_metrics(variant_dir, cache_dir)
        
        if metrics:
            all_metrics[variant] = metrics
            
            # Print summary
            stats = compute_statistics(metrics['mpjpe'])
            print(f"  MPJPE: {stats['mean']:.1f} ± {stats['std']:.1f} mm")
            print(f"         [min: {stats['min']:.1f}, max: {stats['max']:.1f}]")
    
    if not all_metrics:
        print("No metrics computed. Check your results directories.")
        return
    
    # Significance tests (if multiple variants)
    if len(all_metrics) >= 2:
        print("\n" + "="*60)
        print("Statistical Significance Tests")
        print("="*60)
        
        variant_names = list(all_metrics.keys())
        for i in range(len(variant_names)):
            for j in range(i + 1, len(variant_names)):
                v1, v2 = variant_names[i], variant_names[j]
                print(f"\n{v1} vs {v2}:")
                
                sig = significance_test(all_metrics[v1]['mpjpe'], all_metrics[v2]['mpjpe'])
                print(f"  MPJPE: diff={sig['mean_diff']:.1f}±{sig['std_diff']:.1f} mm, "
                      f"p={sig['t_test_pvalue']:.4f} (t-test), "
                      f"Cohen's d={sig['cohens_d']:.2f}")
                
                if sig['t_test_pvalue'] < 0.05:
                    print(f"  → Statistically significant (p < 0.05)")
                else:
                    print(f"  → NOT statistically significant")
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)
    plot_distributions(all_metrics, output_dir)
    
    # Generate LaTeX table
    latex_path = output_dir / 'per_sequence_table.tex'
    generate_latex_table_with_std(all_metrics, latex_path)
    
    # Save raw metrics
    metrics_path = output_dir / 'per_sequence_metrics.json'
    # Convert numpy to python types for JSON serialization
    json_metrics = {}
    for variant, metrics in all_metrics.items():
        json_metrics[variant] = {k: [float(v) for v in vals] for k, vals in metrics.items()}
    
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    print(f"\nRaw metrics saved to: {metrics_path}")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
