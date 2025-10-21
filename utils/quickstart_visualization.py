#!/usr/bin/env python3
"""
Quick-start script for visualizing experiments.

This is a simple entry point for the visualization pipeline.

Run from project root: python utils/quickstart_visualization.py

Usage:
    # Visualize a single experiment
    python utils/quickstart_visualization.py experiments/distilgpt2_wikipedia_full_20251020_033212

    # Compare multiple experiments
    python utils/quickstart_visualization.py experiments/exp1 experiments/exp2 --compare

    # Analyze all experiments
    python utils/quickstart_visualization.py --all
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization import (
    quick_plot,
    discover_experiments,
    MultiExperimentLoader,
    ExperimentPlotter,
    ExperimentLoader
)


def visualize_single(experiment_dir):
    """Visualize a single experiment."""
    print(f"\n{'=' * 80}")
    print(f"Visualizing: {experiment_dir}")
    print('=' * 80)

    quick_plot(experiment_dir)

    # Print summary
    loader = ExperimentLoader(experiment_dir)
    summary = loader.get_summary()

    print(f"\nExperiment Summary:")
    print(f"  Model: {summary['model_type']}")
    print(f"  Dataset: {summary['dataset']}")
    print(f"  Total Steps: {summary['total_steps']:,}")
    print(f"  Total Epochs: {summary['total_epochs']}")

    if 'training_duration_hours' in summary:
        print(f"  Duration: {summary['training_duration_hours']:.2f} hours")

    if 'train_loss' in summary.get('metrics', {}):
        loss = summary['metrics']['train_loss']
        print(f"\n  Training Loss:")
        print(f"    Initial: {loss['initial']:.4f}")
        print(f"    Final: {loss['final']:.4f}")
        print(f"    Min: {loss['min']:.4f}")
        improvement = (loss['initial'] - loss['final']) / loss['initial'] * 100
        print(f"    Improvement: {improvement:.2f}%")

    print(f"\nPlots saved to: {loader.exp_dir / 'plots'}")


def compare_experiments(experiment_dirs):
    """Compare multiple experiments."""
    print(f"\n{'=' * 80}")
    print(f"Comparing {len(experiment_dirs)} experiments")
    print('=' * 80)

    loader = MultiExperimentLoader(experiment_dirs)

    print(f"\nLoaded experiments:")
    for name in loader.loaders.keys():
        print(f"  - {name}")

    # Create comparison plots
    plotter = ExperimentPlotter(loader, output_dir="plots/comparison")
    print(f"\nGenerating comparison plots...")
    plotter.plot_summary_dashboard(save=True)
    plotter.plot_training_curves(save=True)

    # Print comparison
    all_summaries = loader.get_all_summaries()
    print(f"\n{'Experiment':<50} {'Final Loss':<15} {'Duration (h)':<15}")
    print('-' * 80)

    for exp_name, summary in all_summaries.items():
        final_loss = "N/A"
        if 'train_loss' in summary.get('metrics', {}):
            final_loss = f"{summary['metrics']['train_loss']['final']:.4f}"

        duration = "N/A"
        if 'training_duration_hours' in summary:
            duration = f"{summary['training_duration_hours']:.2f}"

        exp_name_short = exp_name[:48] + ".." if len(exp_name) > 50 else exp_name
        print(f"{exp_name_short:<50} {final_loss:<15} {duration:<15}")

    print(f"\nComparison plots saved to: {plotter.output_dir}")


def analyze_all(base_dir='experiments', pattern='*'):
    """Analyze all experiments in the base directory."""
    print(f"\n{'=' * 80}")
    print(f"Analyzing all experiments in: {base_dir}")
    print('=' * 80)

    experiment_dirs = discover_experiments(base_dir, pattern)

    if len(experiment_dirs) == 0:
        print(f"\nNo experiments found in {base_dir} with pattern '{pattern}'")
        return

    print(f"\nFound {len(experiment_dirs)} experiments:")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir.name}")

    # Compare all experiments
    compare_experiments(experiment_dirs)

    # Generate individual plots
    print(f"\n{'=' * 80}")
    print("Generating individual experiment plots...")
    print('=' * 80)

    for i, exp_dir in enumerate(experiment_dirs, 1):
        print(f"\n{i}/{len(experiment_dirs)}: {exp_dir.name}")
        try:
            quick_plot(exp_dir)
        except Exception as e:
            print(f"  Warning: Failed to generate plots - {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick-start visualization for ML experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a single experiment
  python quickstart_visualization.py experiments/distilgpt2_wikipedia_full_20251020_033212

  # Compare two experiments
  python quickstart_visualization.py experiments/exp1 experiments/exp2 --compare

  # Analyze all experiments
  python quickstart_visualization.py --all

  # Analyze experiments matching a pattern
  python quickstart_visualization.py --all --pattern "distilgpt2_*"
        """
    )

    parser.add_argument('experiments', nargs='*',
                       help='Experiment directories to visualize')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple experiments')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all experiments in the base directory')
    parser.add_argument('--base-dir', type=str, default='experiments',
                       help='Base directory for --all mode (default: experiments)')
    parser.add_argument('--pattern', type=str, default='*',
                       help='Pattern for discovering experiments (default: *)')

    args = parser.parse_args()

    # Validate arguments
    if args.all:
        analyze_all(args.base_dir, args.pattern)
    elif args.compare:
        if len(args.experiments) < 2:
            print("Error: --compare requires at least 2 experiment directories")
            return
        compare_experiments(args.experiments)
    elif len(args.experiments) == 1:
        visualize_single(args.experiments[0])
    elif len(args.experiments) > 1:
        # Multiple experiments without --compare flag
        print(f"Visualizing {len(args.experiments)} experiments individually...")
        for exp_dir in args.experiments:
            visualize_single(exp_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
