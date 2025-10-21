#!/usr/bin/env python3
"""
Example script for analyzing all experiments in the experiments directory.

Usage:
    python analyze_all_experiments.py
    python analyze_all_experiments.py --base-dir experiments
    python analyze_all_experiments.py --pattern "distilgpt2_*"
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization import discover_experiments, MultiExperimentLoader, ExperimentPlotter


def main():
    parser = argparse.ArgumentParser(description="Analyze all experiments in a directory")
    parser.add_argument('--base-dir', type=str, default='experiments',
                       help='Base directory containing experiments (default: experiments)')
    parser.add_argument('--pattern', type=str, default='*',
                       help='Glob pattern for experiment directories (default: *)')
    parser.add_argument('--output-dir', type=str, default='plots/all_experiments',
                       help='Output directory for plots (default: plots/all_experiments)')

    args = parser.parse_args()

    print("=" * 80)
    print("Analyzing All Experiments")
    print("=" * 80)

    # Discover experiments
    print(f"\nSearching for experiments in: {args.base_dir}")
    print(f"Pattern: {args.pattern}")

    experiment_dirs = discover_experiments(args.base_dir, args.pattern)

    if len(experiment_dirs) == 0:
        print("\nNo experiments found!")
        sys.exit(1)

    print(f"\nFound {len(experiment_dirs)} experiments:")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir.name}")

    # Load all experiments
    print("\n" + "=" * 80)
    print("Loading Experiments")
    print("=" * 80)

    loader = MultiExperimentLoader(experiment_dirs)

    print(f"\nSuccessfully loaded {len(loader.loaders)} experiments")

    # Get all summaries
    print("\n" + "=" * 80)
    print("Experiment Overview")
    print("=" * 80)

    all_summaries = loader.get_all_summaries()

    # Create summary table
    print("\n{:<40} {:<15} {:<12} {:<10} {:<12} {:<12}".format(
        "Experiment", "Model", "Dataset", "Steps", "Final Loss", "Duration (h)"
    ))
    print("-" * 110)

    for exp_name, summary in all_summaries.items():
        model = summary['model_type']
        dataset = summary['dataset']
        steps = summary['total_steps']

        final_loss = "N/A"
        if 'train_loss' in summary.get('metrics', {}):
            final_loss = f"{summary['metrics']['train_loss']['final']:.4f}"

        duration = "N/A"
        if 'training_duration_hours' in summary:
            duration = f"{summary['training_duration_hours']:.2f}"

        # Truncate long names
        exp_name_short = exp_name[:38] + ".." if len(exp_name) > 40 else exp_name

        print("{:<40} {:<15} {:<12} {:<10} {:<12} {:<12}".format(
            exp_name_short, model[:15], dataset[:12], f"{steps:,}", final_loss, duration
        ))

    # Group experiments by model type
    print("\n" + "=" * 80)
    print("Experiments by Model Type")
    print("=" * 80)

    model_groups = {}
    for exp_name, summary in all_summaries.items():
        model = summary['model_type']
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append(exp_name)

    for model, exps in model_groups.items():
        print(f"\n{model} ({len(exps)} experiments):")
        for exp in exps:
            print(f"  - {exp}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    plotter = ExperimentPlotter(loader, output_dir=args.output_dir)

    print("\n1. Creating comparison dashboard...")
    plotter.plot_summary_dashboard(save=True, show=False)

    print("2. Creating training curves comparison...")
    plotter.plot_training_curves(save=True, show=False)

    # Generate individual plots for each experiment
    print("\n" + "=" * 80)
    print("Generating Individual Experiment Plots")
    print("=" * 80)

    from visualization import ExperimentLoader, quick_plot

    for i, (exp_name, exp_loader) in enumerate(loader.loaders.items(), 1):
        print(f"\n{i}. {exp_name}")
        try:
            quick_plot(exp_loader.exp_dir)
        except Exception as e:
            print(f"   Warning: Failed to generate plots - {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)

    print(f"\nComparison plots saved to: {plotter.output_dir}")
    print(f"Individual plots saved to: <experiment_dir>/plots/")

    print("\nSummary Statistics:")
    print("-" * 40)
    print(f"Total experiments analyzed: {len(all_summaries)}")
    print(f"Model types: {', '.join(model_groups.keys())}")

    # Find best experiment by final loss
    best_exp = None
    best_loss = float('inf')
    for exp_name, summary in all_summaries.items():
        if 'train_loss' in summary.get('metrics', {}):
            final_loss = summary['metrics']['train_loss']['final']
            if final_loss < best_loss:
                best_loss = final_loss
                best_exp = exp_name

    if best_exp:
        print(f"\nBest experiment (by final loss): {best_exp}")
        print(f"Final loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
