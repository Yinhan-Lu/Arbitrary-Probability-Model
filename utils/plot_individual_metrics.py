#!/usr/bin/env python3
"""
Plot individual metrics from experiment metrics.csv as separate files.

Usage:
    python utils/plot_individual_metrics.py experiments/exp_name
    python utils/plot_individual_metrics.py experiments/exp_name --output-dir custom_plots
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_train_loss(df, output_dir):
    """Plot training loss vs step."""
    if 'train_loss' not in df.columns or df['train_loss'].isna().all():
        print("  ⚠ Skipping train_loss plot (no data)")
        return

    data = df[['step', 'train_loss']].dropna()

    plt.figure(figsize=(10, 6))
    plt.plot(data['step'], data['train_loss'], linewidth=2, color='#2E86AB', marker='o', markersize=3, alpha=0.7)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss vs Step', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'train_loss_vs_step.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_train_perplexity(df, output_dir):
    """Plot training perplexity vs step."""
    if 'train_perplexity' not in df.columns or df['train_perplexity'].isna().all():
        print("  ⚠ Skipping train_perplexity plot (no data)")
        return

    data = df[['step', 'train_perplexity']].dropna()

    plt.figure(figsize=(10, 6))
    plt.plot(data['step'], data['train_perplexity'], linewidth=2, color='#A23B72', marker='o', markersize=3, alpha=0.7)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Training Perplexity', fontsize=12)
    plt.title('Training Perplexity vs Step', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'train_perplexity_vs_step.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_learning_rate(df, output_dir):
    """Plot learning rate vs step."""
    if 'learning_rate' not in df.columns or df['learning_rate'].isna().all():
        print("  ⚠ Skipping learning_rate plot (no data)")
        return

    data = df[['step', 'learning_rate']].dropna()

    plt.figure(figsize=(10, 6))
    plt.plot(data['step'], data['learning_rate'], linewidth=2, color='#F18F01', marker='o', markersize=3, alpha=0.7)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    plt.tight_layout()

    output_path = output_dir / 'learning_rate_vs_step.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_all_modes_loss(df, output_dir):
    """Plot all 5 evaluation modes loss comparison."""
    mode_cols = [f'mode{i}_loss' for i in range(1, 6)]
    available_modes = [col for col in mode_cols if col in df.columns and not df[col].isna().all()]

    if not available_modes:
        print("  ⚠ Skipping all_modes_loss plot (no data)")
        return

    plt.figure(figsize=(12, 6))
    colors = ['#E63946', '#F77F00', '#06D6A0', '#118AB2', '#073B4C']

    for i, mode_col in enumerate(available_modes):
        data = df[['step', mode_col]].dropna()
        if len(data) > 0:
            mode_num = mode_col.replace('mode', '').replace('_loss', '')
            plt.plot(data['step'], data[mode_col], linewidth=2,
                    color=colors[i % len(colors)],
                    marker='o', markersize=4, alpha=0.7,
                    label=f'Mode {mode_num}')

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('All Evaluation Modes - Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'all_modes_loss_vs_step.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_all_modes_perplexity(df, output_dir):
    """Plot all 5 evaluation modes perplexity comparison."""
    mode_cols = [f'mode{i}_ppl' for i in range(1, 6)]
    available_modes = [col for col in mode_cols if col in df.columns and not df[col].isna().all()]

    if not available_modes:
        print("  ⚠ Skipping all_modes_perplexity plot (no data)")
        return

    plt.figure(figsize=(12, 6))
    colors = ['#E63946', '#F77F00', '#06D6A0', '#118AB2', '#073B4C']

    for i, mode_col in enumerate(available_modes):
        data = df[['step', mode_col]].dropna()
        if len(data) > 0:
            mode_num = mode_col.replace('mode', '').replace('_ppl', '')
            plt.plot(data['step'], data[mode_col], linewidth=2,
                    color=colors[i % len(colors)],
                    marker='o', markersize=4, alpha=0.7,
                    label=f'Mode {mode_num}')

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('All Evaluation Modes - Perplexity Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'all_modes_perplexity_vs_step.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_train_vs_eval(df, output_dir):
    """Plot training loss vs evaluation loss (using mode3 as eval)."""
    if 'train_loss' not in df.columns or df['train_loss'].isna().all():
        print("  ⚠ Skipping train_vs_eval plot (no train_loss data)")
        return
    if 'mode3_loss' not in df.columns or df['mode3_loss'].isna().all():
        print("  ⚠ Skipping train_vs_eval plot (no mode3_loss data)")
        return

    train_data = df[['step', 'train_loss']].dropna()
    eval_data = df[['step', 'mode3_loss']].dropna()

    plt.figure(figsize=(12, 6))

    if len(train_data) > 0:
        plt.plot(train_data['step'], train_data['train_loss'],
                linewidth=2, color='#2E86AB', marker='o', markersize=3,
                alpha=0.7, label='Train Loss')

    if len(eval_data) > 0:
        plt.plot(eval_data['step'], eval_data['mode3_loss'],
                linewidth=2, color='#E63946', marker='s', markersize=4,
                alpha=0.7, label='Eval Loss (Mode 3)')

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss vs Evaluation Loss', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'train_vs_eval_loss.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_train_vs_all_eval_losses(df, output_dir):
    """Plot training loss vs all evaluation modes losses together."""
    has_train = 'train_loss' in df.columns and not df['train_loss'].isna().all()

    mode_cols = [f'mode{i}_loss' for i in range(1, 6)]
    available_modes = [col for col in mode_cols if col in df.columns and not df[col].isna().all()]

    if not has_train and not available_modes:
        print("  ⚠ Skipping train_vs_all_eval_losses plot (no data)")
        return

    plt.figure(figsize=(14, 7))
    colors = ['#2E86AB', '#E63946', '#F77F00', '#06D6A0', '#118AB2', '#073B4C']

    # Plot training loss
    if has_train:
        train_data = df[['step', 'train_loss']].dropna()
        if len(train_data) > 0:
            plt.plot(train_data['step'], train_data['train_loss'],
                    linewidth=2.5, color=colors[0], marker='o', markersize=3,
                    alpha=0.8, label='Train Loss', zorder=10)

    # Plot all eval mode losses
    for i, mode_col in enumerate(available_modes):
        mode_data = df[['step', mode_col]].dropna()
        if len(mode_data) > 0:
            mode_num = mode_col.replace('mode', '').replace('_loss', '')
            plt.plot(mode_data['step'], mode_data[mode_col],
                    linewidth=2, color=colors[i+1], marker='s', markersize=4,
                    alpha=0.7, label=f'Mode {mode_num} (Eval)', zorder=5)

    plt.xlabel('Step', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.title('Training Loss vs All Evaluation Modes', fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'train_vs_all_eval_losses.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_single_mode(df, mode_num, output_dir):
    """Plot loss and perplexity for a single evaluation mode."""
    loss_col = f'mode{mode_num}_loss'
    ppl_col = f'mode{mode_num}_ppl'

    has_loss = loss_col in df.columns and not df[loss_col].isna().all()
    has_ppl = ppl_col in df.columns and not df[ppl_col].isna().all()

    if not has_loss and not has_ppl:
        print(f"  ⚠ Skipping mode{mode_num} plot (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss plot
    if has_loss:
        loss_data = df[['step', loss_col]].dropna()
        axes[0].plot(loss_data['step'], loss_data[loss_col],
                    linewidth=2, color='#2E86AB', marker='o', markersize=3, alpha=0.7)
        axes[0].set_xlabel('Step', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'Mode {mode_num} - Loss', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
        axes[0].set_title(f'Mode {mode_num} - Loss', fontsize=13, fontweight='bold')

    # Perplexity plot
    if has_ppl:
        ppl_data = df[['step', ppl_col]].dropna()
        axes[1].plot(ppl_data['step'], ppl_data[ppl_col],
                    linewidth=2, color='#A23B72', marker='o', markersize=3, alpha=0.7)
        axes[1].set_xlabel('Step', fontsize=12)
        axes[1].set_ylabel('Perplexity', fontsize=12)
        axes[1].set_title(f'Mode {mode_num} - Perplexity', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
        axes[1].set_title(f'Mode {mode_num} - Perplexity', fontsize=13, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / f'mode{mode_num}_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_epoch_comparison(df, output_dir):
    """Plot loss distribution by epoch using boxplots."""
    if 'train_loss' not in df.columns or df['train_loss'].isna().all():
        print("  ⚠ Skipping epoch_comparison plot (no train_loss data)")
        return
    if 'epoch' not in df.columns:
        print("  ⚠ Skipping epoch_comparison plot (no epoch data)")
        return

    data = df[['epoch', 'train_loss']].dropna()
    if len(data) == 0:
        print("  ⚠ Skipping epoch_comparison plot (no valid data)")
        return

    epochs = sorted(data['epoch'].unique())
    epoch_data = [data[data['epoch'] == epoch]['train_loss'].values for epoch in epochs]

    # Filter out empty arrays
    valid_epochs = []
    valid_data = []
    for epoch, loss_data in zip(epochs, epoch_data):
        if len(loss_data) > 0:
            valid_epochs.append(epoch)
            valid_data.append(loss_data)

    if not valid_data:
        print("  ⚠ Skipping epoch_comparison plot (no valid epoch data)")
        return

    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(valid_data, tick_labels=valid_epochs, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Distribution by Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_path = output_dir / 'train_loss_by_epoch.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_all_metrics(exp_dir, output_dir=None):
    """Generate all individual plots for an experiment."""
    exp_path = Path(exp_dir)

    # Find metrics.csv
    metrics_csv = exp_path / 'logs' / 'metrics.csv'
    if not metrics_csv.exists():
        print(f"❌ Error: metrics.csv not found at {metrics_csv}")
        return

    # Set output directory
    if output_dir is None:
        output_dir = exp_path / 'plots_individual'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n{'=' * 80}")
    print(f"Loading metrics from: {metrics_csv}")
    print('=' * 80)

    df = pd.read_csv(metrics_csv)
    print(f"✓ Loaded {len(df)} rows")
    print(f"\nAvailable columns: {', '.join(df.columns)}")

    # Generate all plots
    print(f"\n{'=' * 80}")
    print(f"Generating individual plots...")
    print('=' * 80)

    plot_train_loss(df, output_dir)
    plot_train_perplexity(df, output_dir)
    plot_learning_rate(df, output_dir)
    plot_all_modes_loss(df, output_dir)
    plot_all_modes_perplexity(df, output_dir)
    plot_train_vs_eval(df, output_dir)
    plot_train_vs_all_eval_losses(df, output_dir)
    plot_epoch_comparison(df, output_dir)

    # Individual mode plots
    for mode_num in range(1, 6):
        plot_single_mode(df, mode_num, output_dir)

    print(f"\n{'=' * 80}")
    print(f"✓ All plots saved to: {output_dir}")
    print('=' * 80)

    # Print summary statistics
    print(f"\n{'=' * 80}")
    print("Experiment Summary")
    print('=' * 80)

    if 'train_loss' in df.columns:
        train_loss_data = df['train_loss'].dropna()
        if len(train_loss_data) > 0:
            print(f"\nTraining Loss:")
            print(f"  Initial: {train_loss_data.iloc[0]:.4f}")
            print(f"  Final: {train_loss_data.iloc[-1]:.4f}")
            print(f"  Min: {train_loss_data.min():.4f} (step {df.loc[train_loss_data.idxmin(), 'step']:.0f})")
            print(f"  Max: {train_loss_data.max():.4f}")

    for mode_num in range(1, 6):
        loss_col = f'mode{mode_num}_loss'
        if loss_col in df.columns:
            mode_loss_data = df[loss_col].dropna()
            if len(mode_loss_data) > 0:
                print(f"\nMode {mode_num} Loss:")
                print(f"  First: {mode_loss_data.iloc[0]:.4f}")
                print(f"  Last: {mode_loss_data.iloc[-1]:.4f}")
                print(f"  Min: {mode_loss_data.min():.4f} (step {df.loc[mode_loss_data.idxmin(), 'step']:.0f})")

    if 'step' in df.columns:
        total_steps = df['step'].max()
        print(f"\nTotal Steps: {total_steps:.0f}")

    if 'epoch' in df.columns:
        total_epochs = df['epoch'].max()
        print(f"Total Epochs: {total_epochs:.0f}")

    print(f"\n{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate individual metric plots from experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots for a single experiment
  python utils/plot_individual_metrics.py experiments/conditional_moderate_cond_20251104_180059

  # Specify custom output directory
  python utils/plot_individual_metrics.py experiments/exp_name --output-dir custom_plots
        """
    )

    parser.add_argument('exp_dir', type=str,
                       help='Path to experiment directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory for plots (default: exp_dir/plots_individual)')

    args = parser.parse_args()

    plot_all_metrics(args.exp_dir, args.output_dir)


if __name__ == '__main__':
    main()
