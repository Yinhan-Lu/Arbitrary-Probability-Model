#!/usr/bin/env python3
"""
Cross-experiment comparison visualization tool.

Generates comparison plots with multiple experiments' curves overlaid on the same figure.
Outputs to comparison_between_experiments/YYYYMMDD_HHMMSS/ with a comparison_info.json.

Usage:
    python utils/plot_comparison_metrics.py experiments/exp1 experiments/exp2
    python utils/plot_comparison_metrics.py experiments/exp1 experiments/exp2 experiments/exp3
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color palette for multiple experiments (colorblind-friendly)
COMPARISON_COLORS = [
    '#E63946',  # Red
    '#2E86AB',  # Blue
    '#06D6A0',  # Teal
    '#F77F00',  # Orange
    '#7B2CBF',  # Purple
    '#2D6A4F',  # Forest Green
    '#E9C46A',  # Gold
    '#264653',  # Dark Teal
    '#D62828',  # Crimson
    '#023E8A',  # Navy
]

LINE_STYLES = ['-', '--', '-.', ':']


def extract_label(exp_name):
    """Remove timestamp suffix from experiment name.

    Example: 'conditional_moderate_cond_20251104_180059' -> 'conditional_moderate_cond'
    """
    parts = exp_name.split('_')
    # Remove last 2 parts (date YYYYMMDD and time HHMMSS)
    if len(parts) >= 3 and len(parts[-1]) == 6 and len(parts[-2]) == 8:
        # Verify they look like date/time
        try:
            int(parts[-1])  # HHMMSS
            int(parts[-2])  # YYYYMMDD
            return '_'.join(parts[:-2])
        except ValueError:
            pass
    return exp_name  # Return original if pattern doesn't match


def load_experiment_data(exp_dir):
    """Load metrics.csv and return DataFrame with experiment info."""
    exp_path = Path(exp_dir)
    metrics_csv = exp_path / 'logs' / 'metrics.csv'

    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv not found at {metrics_csv}")

    df = pd.read_csv(metrics_csv)
    df = df.replace('', np.nan)
    df = df.replace('inf', np.inf)

    exp_name = exp_path.name
    label = extract_label(exp_name)

    return df, exp_name, label


MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']


def get_style(index):
    """Get color, line style, and marker for experiment index."""
    color = COMPARISON_COLORS[index % len(COMPARISON_COLORS)]
    linestyle = LINE_STYLES[index // len(COMPARISON_COLORS) % len(LINE_STYLES)]
    marker = MARKERS[index % len(MARKERS)]
    return color, linestyle, marker


def compare_train_loss(experiments_data, output_dir):
    """Compare training loss across experiments."""
    has_data = False
    plt.figure(figsize=(12, 7))

    for i, (df, exp_name, label) in enumerate(experiments_data):
        if 'train_loss' not in df.columns or df['train_loss'].isna().all():
            continue
        data = df[['step', 'train_loss']].dropna()
        if len(data) == 0:
            continue

        has_data = True
        color, linestyle, marker = get_style(i)
        plt.plot(data['step'], data['train_loss'],
                linewidth=2, color=color, linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, label=label)

    if not has_data:
        plt.close()
        print("  - Skipping compare_train_loss (no data)")
        return None

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'compare_train_loss.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved: {output_path.name}")
    return output_path.name


def compare_train_perplexity(experiments_data, output_dir):
    """Compare training perplexity across experiments."""
    has_data = False
    plt.figure(figsize=(12, 7))

    for i, (df, exp_name, label) in enumerate(experiments_data):
        if 'train_perplexity' not in df.columns or df['train_perplexity'].isna().all():
            continue
        data = df[['step', 'train_perplexity']].dropna()
        if len(data) == 0:
            continue

        has_data = True
        color, linestyle, marker = get_style(i)
        plt.plot(data['step'], data['train_perplexity'],
                linewidth=2, color=color, linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, label=label)

    if not has_data:
        plt.close()
        print("  - Skipping compare_train_perplexity (no data)")
        return None

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Training Perplexity', fontsize=12)
    plt.title('Training Perplexity Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'compare_train_perplexity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved: {output_path.name}")
    return output_path.name


def compare_learning_rate(experiments_data, output_dir):
    """Compare learning rate schedules across experiments."""
    has_data = False
    plt.figure(figsize=(12, 7))

    for i, (df, exp_name, label) in enumerate(experiments_data):
        if 'learning_rate' not in df.columns or df['learning_rate'].isna().all():
            continue
        data = df[['step', 'learning_rate']].dropna()
        if len(data) == 0:
            continue

        has_data = True
        color, linestyle, marker = get_style(i)
        plt.plot(data['step'], data['learning_rate'],
                linewidth=2, color=color, linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, label=label)

    if not has_data:
        plt.close()
        print("  - Skipping compare_learning_rate (no data)")
        return None

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    plt.tight_layout()

    output_path = output_dir / 'compare_learning_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved: {output_path.name}")
    return output_path.name


def compare_mode_loss(experiments_data, mode_num, output_dir):
    """Compare mode N loss across experiments."""
    loss_col = f'mode{mode_num}_loss'
    has_data = False
    plt.figure(figsize=(12, 7))

    for i, (df, exp_name, label) in enumerate(experiments_data):
        if loss_col not in df.columns or df[loss_col].isna().all():
            continue
        data = df[['step', loss_col]].dropna()
        if len(data) == 0:
            continue

        has_data = True
        color, linestyle, marker = get_style(i)
        plt.plot(data['step'], data[loss_col],
                linewidth=2, color=color, linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, label=label)

    if not has_data:
        plt.close()
        print(f"  - Skipping compare_mode{mode_num}_loss (no data)")
        return None

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Mode {mode_num} Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f'compare_mode{mode_num}_loss.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved: {output_path.name}")
    return output_path.name


def compare_mode_perplexity(experiments_data, mode_num, output_dir):
    """Compare mode N perplexity across experiments."""
    ppl_col = f'mode{mode_num}_ppl'
    has_data = False
    plt.figure(figsize=(12, 7))

    for i, (df, exp_name, label) in enumerate(experiments_data):
        if ppl_col not in df.columns or df[ppl_col].isna().all():
            continue
        data = df[['step', ppl_col]].dropna()
        if len(data) == 0:
            continue

        has_data = True
        color, linestyle, marker = get_style(i)
        plt.plot(data['step'], data[ppl_col],
                linewidth=2, color=color, linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, label=label)

    if not has_data:
        plt.close()
        print(f"  - Skipping compare_mode{mode_num}_perplexity (no data)")
        return None

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title(f'Mode {mode_num} Perplexity Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f'compare_mode{mode_num}_perplexity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved: {output_path.name}")
    return output_path.name


def compare_eval_loss(experiments_data, output_dir):
    """Compare evaluation loss (sigmagpt) across experiments."""
    has_data = False
    plt.figure(figsize=(12, 7))

    for i, (df, exp_name, label) in enumerate(experiments_data):
        if 'eval_loss' not in df.columns or df['eval_loss'].isna().all():
            continue
        data = df[['step', 'eval_loss']].dropna()
        if len(data) == 0:
            continue

        has_data = True
        color, linestyle, marker = get_style(i)
        plt.plot(data['step'], data['eval_loss'],
                linewidth=2, color=color, linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, label=label)

    if not has_data:
        plt.close()
        print("  - Skipping compare_eval_loss (no data)")
        return None

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Evaluation Loss', fontsize=12)
    plt.title('Evaluation Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'compare_eval_loss.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved: {output_path.name}")
    return output_path.name


def compare_eval_perplexity(experiments_data, output_dir):
    """Compare evaluation perplexity (sigmagpt) across experiments."""
    has_data = False
    plt.figure(figsize=(12, 7))

    for i, (df, exp_name, label) in enumerate(experiments_data):
        if 'eval_perplexity' not in df.columns or df['eval_perplexity'].isna().all():
            continue
        data = df[['step', 'eval_perplexity']].dropna()
        if len(data) == 0:
            continue

        has_data = True
        color, linestyle, marker = get_style(i)
        plt.plot(data['step'], data['eval_perplexity'],
                linewidth=2, color=color, linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, label=label)

    if not has_data:
        plt.close()
        print("  - Skipping compare_eval_perplexity (no data)")
        return None

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Evaluation Perplexity', fontsize=12)
    plt.title('Evaluation Perplexity Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'compare_eval_perplexity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved: {output_path.name}")
    return output_path.name


def get_experiment_summary(df, exp_path, label):
    """Get summary statistics for an experiment."""
    summary = {
        'path': str(exp_path),
        'label': label,
    }

    # Total steps
    if 'step' in df.columns:
        step_max = df['step'].max()
        if pd.notna(step_max):
            summary['total_steps'] = int(step_max)

    # Training loss stats
    if 'train_loss' in df.columns:
        train_loss = df['train_loss'].dropna()
        if len(train_loss) > 0:
            summary['final_train_loss'] = float(train_loss.iloc[-1])
            summary['min_train_loss'] = float(train_loss.min())
            summary['initial_train_loss'] = float(train_loss.iloc[0])

    # Mode losses (final values)
    for mode_num in range(1, 6):
        loss_col = f'mode{mode_num}_loss'
        if loss_col in df.columns:
            mode_loss = df[loss_col].dropna()
            if len(mode_loss) > 0:
                summary[f'final_mode{mode_num}_loss'] = float(mode_loss.iloc[-1])
                summary[f'min_mode{mode_num}_loss'] = float(mode_loss.min())

    return summary


def plot_all_comparisons(exp_dirs, output_base_dir=None):
    """Generate all comparison plots for given experiments."""
    # Determine output directory
    if output_base_dir is None:
        project_root = Path(__file__).parent.parent
        output_base_dir = project_root / 'comparison_between_experiments'

    output_base_dir = Path(output_base_dir)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Cross-Experiment Comparison")
    print("=" * 80)
    print(f"\nLoading {len(exp_dirs)} experiments...")

    # Load all experiments
    experiments_data = []
    for exp_dir in exp_dirs:
        try:
            df, exp_name, label = load_experiment_data(exp_dir)
            experiments_data.append((df, exp_name, label))
            print(f"  + Loaded: {label} ({len(df)} rows)")
        except FileNotFoundError as e:
            print(f"  ! Error loading {exp_dir}: {e}")

    if len(experiments_data) < 2:
        print("\nError: Need at least 2 experiments to compare")
        return

    print(f"\nOutput directory: {output_dir}")
    print("\n" + "=" * 80)
    print("Generating comparison plots...")
    print("=" * 80)

    # Generate all comparison plots
    plots_generated = []

    # Basic training metrics
    result = compare_train_loss(experiments_data, output_dir)
    if result:
        plots_generated.append(result)

    result = compare_train_perplexity(experiments_data, output_dir)
    if result:
        plots_generated.append(result)

    result = compare_learning_rate(experiments_data, output_dir)
    if result:
        plots_generated.append(result)

    # Mode-specific plots (1-5)
    for mode_num in range(1, 6):
        result = compare_mode_loss(experiments_data, mode_num, output_dir)
        if result:
            plots_generated.append(result)

        result = compare_mode_perplexity(experiments_data, mode_num, output_dir)
        if result:
            plots_generated.append(result)

    # Eval metrics (sigmagpt)
    result = compare_eval_loss(experiments_data, output_dir)
    if result:
        plots_generated.append(result)

    result = compare_eval_perplexity(experiments_data, output_dir)
    if result:
        plots_generated.append(result)

    # Generate comparison_info.json
    print("\n" + "=" * 80)
    print("Generating comparison_info.json...")
    print("=" * 80)

    comparison_info = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': [],
        'plots_generated': plots_generated,
    }

    for df, exp_name, label in experiments_data:
        exp_path = None
        for exp_dir in exp_dirs:
            if Path(exp_dir).name == exp_name:
                exp_path = exp_dir
                break
        summary = get_experiment_summary(df, exp_path, label)
        comparison_info['experiments'].append(summary)

    info_path = output_dir / 'comparison_info.json'
    with open(info_path, 'w') as f:
        json.dump(comparison_info, f, indent=2)
    print(f"  + Saved: {info_path.name}")

    # Print summary
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)

    print(f"\nExperiments compared: {len(experiments_data)}")
    for i, (df, exp_name, label) in enumerate(experiments_data):
        color, _, marker = get_style(i)
        print(f"  [{i+1}] {label} (color: {color}, marker: {marker})")

    print(f"\nPlots generated: {len(plots_generated)}")
    for plot_name in plots_generated:
        print(f"  - {plot_name}")

    print(f"\nAll outputs saved to: {output_dir}")
    print("=" * 80)

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Compare metrics across multiple experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare 2 experiments
  python utils/plot_comparison_metrics.py experiments/exp1 experiments/exp2

  # Compare 3 experiments
  python utils/plot_comparison_metrics.py experiments/exp1 experiments/exp2 experiments/exp3

Output:
  comparison_between_experiments/YYYYMMDD_HHMMSS/
    - compare_train_loss.png
    - compare_train_perplexity.png
    - compare_learning_rate.png
    - compare_mode{1-5}_loss.png
    - compare_mode{1-5}_perplexity.png
    - compare_eval_loss.png
    - compare_eval_perplexity.png
    - comparison_info.json
        """
    )

    parser.add_argument('experiments', nargs='+', type=str,
                       help='Paths to experiment directories (at least 2)')

    args = parser.parse_args()

    if len(args.experiments) < 2:
        print("Error: Please provide at least 2 experiment directories to compare")
        sys.exit(1)

    plot_all_comparisons(args.experiments)


if __name__ == '__main__':
    main()
