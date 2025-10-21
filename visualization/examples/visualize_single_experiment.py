#!/usr/bin/env python3
"""
Example script for visualizing a single experiment.

Usage:
    python visualize_single_experiment.py <experiment_dir>
    python visualize_single_experiment.py experiments/distilgpt2_wikipedia_full_20251020_033212
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization import ExperimentLoader, ExperimentPlotter, quick_plot


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_single_experiment.py <experiment_dir>")
        print("\nExample:")
        print("  python visualize_single_experiment.py experiments/distilgpt2_wikipedia_full_20251020_033212")
        sys.exit(1)

    experiment_dir = sys.argv[1]

    # Option 1: Quick plot (generates all plots automatically)
    print("=" * 80)
    print("OPTION 1: Quick Plot (All visualizations)")
    print("=" * 80)
    quick_plot(experiment_dir)

    # Option 2: Custom plotting with more control
    print("\n" + "=" * 80)
    print("OPTION 2: Custom Plotting")
    print("=" * 80)

    # Load experiment
    loader = ExperimentLoader(experiment_dir)

    # Print summary
    print("\nExperiment Summary:")
    print("-" * 40)
    summary = loader.get_summary()
    for key, value in summary.items():
        if key != 'metrics':
            print(f"{key}: {value}")

    print("\nMetric Statistics:")
    print("-" * 40)
    for metric, stats in summary.get('metrics', {}).items():
        print(f"\n{metric}:")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {stat_value:.6f}")

    # Create plotter
    plotter = ExperimentPlotter(loader)

    # Generate specific plots
    print("\n" + "=" * 80)
    print("Generating Custom Plots...")
    print("=" * 80)

    # Plot 1: Comprehensive dashboard
    print("\n1. Creating summary dashboard...")
    plotter.plot_summary_dashboard(save=True, show=False)

    # Plot 2: Detailed training curves
    print("2. Creating training curves...")
    plotter.plot_training_curves(save=True, show=False)

    # Plot 3: Loss landscape
    print("3. Creating loss landscape...")
    plotter.plot_loss_landscape(metric='train_loss', save=True, show=False)

    # Plot 4: Optimization dynamics
    print("4. Creating optimization dynamics...")
    plotter.plot_optimization_dynamics(save=True, show=False)

    print("\n" + "=" * 80)
    print(f"All plots saved to: {plotter.output_dir}")
    print("=" * 80)

    # List generated files
    print("\nGenerated files:")
    for plot_file in sorted(plotter.output_dir.glob("*.png")):
        size_mb = plot_file.stat().st_size / (1024 * 1024)
        print(f"  - {plot_file.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
