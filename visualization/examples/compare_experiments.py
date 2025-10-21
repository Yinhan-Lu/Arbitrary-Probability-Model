#!/usr/bin/env python3
"""
Example script for comparing multiple experiments.

Usage:
    python compare_experiments.py <exp_dir1> <exp_dir2> [exp_dir3 ...]
    python compare_experiments.py experiments/exp1 experiments/exp2 experiments/exp3
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization import MultiExperimentLoader, ExperimentPlotter


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_experiments.py <exp_dir1> <exp_dir2> [exp_dir3 ...]")
        print("\nExample:")
        print("  python compare_experiments.py experiments/exp1 experiments/exp2 experiments/exp3")
        sys.exit(1)

    experiment_dirs = sys.argv[1:]

    print("=" * 80)
    print("Multi-Experiment Comparison")
    print("=" * 80)

    # Load all experiments
    print(f"\nLoading {len(experiment_dirs)} experiments...")
    loader = MultiExperimentLoader(experiment_dirs)

    print(f"Successfully loaded {len(loader.loaders)} experiments:")
    for name in loader.loaders.keys():
        print(f"  - {name}")

    # Get all summaries
    print("\n" + "=" * 80)
    print("Experiment Summaries")
    print("=" * 80)

    all_summaries = loader.get_all_summaries()
    for exp_name, summary in all_summaries.items():
        print(f"\n{exp_name}:")
        print("-" * 40)
        print(f"  Model: {summary['model_type']}")
        print(f"  Dataset: {summary['dataset']}")
        print(f"  Total Steps: {summary['total_steps']:,}")
        print(f"  Total Epochs: {summary['total_epochs']}")

        if 'training_duration_hours' in summary:
            print(f"  Duration: {summary['training_duration_hours']:.2f} hours")

        if 'train_loss' in summary.get('metrics', {}):
            loss_stats = summary['metrics']['train_loss']
            print(f"  Initial Loss: {loss_stats['initial']:.4f}")
            print(f"  Final Loss: {loss_stats['final']:.4f}")
            print(f"  Min Loss: {loss_stats['min']:.4f}")
            improvement = (loss_stats['initial'] - loss_stats['final']) / loss_stats['initial'] * 100
            print(f"  Improvement: {improvement:.2f}%")

    # Create comparison plots
    print("\n" + "=" * 80)
    print("Generating Comparison Plots")
    print("=" * 80)

    plotter = ExperimentPlotter(loader, output_dir="plots/comparison")

    # Plot 1: Training curves comparison
    print("\n1. Creating training curves comparison...")
    plotter.plot_training_curves(save=True, show=False)

    # Plot 2: Comparison dashboard
    print("2. Creating comparison dashboard...")
    plotter.plot_summary_dashboard(save=True, show=False)

    print("\n" + "=" * 80)
    print(f"All comparison plots saved to: {plotter.output_dir}")
    print("=" * 80)

    # Compare specific metric
    print("\n" + "=" * 80)
    print("Metric Comparison: Training Loss")
    print("=" * 80)

    loss_comparison = loader.compare_metrics('train_loss')
    print("\nFinal training loss for each experiment:")
    for exp_name in loss_comparison.columns:
        final_loss = loss_comparison[exp_name].dropna().iloc[-1]
        print(f"  {exp_name}: {final_loss:.4f}")

    # Find best experiment
    best_exp = loss_comparison.iloc[-1].idxmin()
    best_loss = loss_comparison.iloc[-1].min()
    print(f"\nBest experiment: {best_exp} (final loss: {best_loss:.4f})")

    # List generated files
    print("\nGenerated files:")
    for plot_file in sorted(plotter.output_dir.glob("*.png")):
        size_mb = plot_file.stat().st_size / (1024 * 1024)
        print(f"  - {plot_file.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
