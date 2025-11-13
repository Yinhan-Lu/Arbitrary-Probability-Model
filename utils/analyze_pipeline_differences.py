"""
Analyze Performance Differences Between Legacy and New Pipelines

This script performs comprehensive analysis of training curves and metrics
from two different training pipelines to understand why Mode 2 (boundary filling)
shows dramatically different performance (16.8x difference).

Usage:
    python utils/analyze_pipeline_differences.py \
        result_of_two_pipeline/legacy_pipeline \
        result_of_two_pipeline/new_pipeline \
        --output results_analysis

    Or with custom experiment directories:
    python utils/analyze_pipeline_differences.py \
        experiments/exp1 \
        experiments/exp2 \
        --output comparison_output

Run from project root.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Dict, List, Tuple
import json

print("=" * 100)
print("PIPELINE DIFFERENCES ANALYSIS")
print("=" * 100)


def list_experiments(base_dir: str = "experiments") -> List[Path]:
    """
    List all experiment directories in base_dir

    Args:
        base_dir: Base directory containing experiments

    Returns:
        List of experiment directory paths
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        return []

    # Find directories that contain logs/metrics.csv
    experiments = []
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if this looks like an experiment (has logs/metrics.csv)
            has_metrics = (item / "logs" / "metrics.csv").exists() or \
                         (item / "logs.pdf" / "metrics.csv").exists() or \
                         (item / "metrics.csv").exists()
            if has_metrics:
                experiments.append(item)

    return sorted(experiments)


def resolve_experiment_path(name_or_path: str, base_dir: str = "experiments") -> Path:
    """
    Resolve experiment name to full path

    Supports:
    - Full paths (returned as-is)
    - Experiment names (prefixed with base_dir/)
    - Partial matches (glob)

    Args:
        name_or_path: Experiment name or path
        base_dir: Base directory for experiments

    Returns:
        Resolved path

    Raises:
        FileNotFoundError: If experiment not found or ambiguous
    """
    path = Path(name_or_path)

    # If it's an absolute path or starts with base_dir, use as-is
    if path.is_absolute() or str(path).startswith(f"{base_dir}/"):
        if path.exists():
            return path
        else:
            raise FileNotFoundError(f"Experiment not found: {path}")

    # Try prefixing with base_dir
    base_path = Path(base_dir)
    candidate = base_path / name_or_path

    if candidate.exists():
        return candidate

    # Try glob matching
    matches = list(base_path.glob(f"*{name_or_path}*"))
    matches = [m for m in matches if m.is_dir()]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No experiment matching '{name_or_path}' found in {base_dir}/\n"
            f"Available experiments: {[e.name for e in list_experiments(base_dir)]}"
        )
    elif len(matches) == 1:
        return matches[0]
    else:
        raise ValueError(
            f"Ambiguous experiment name '{name_or_path}'. Multiple matches:\n" +
            "\n".join([f"  - {m.name}" for m in matches]) +
            "\nPlease be more specific."
        )


def select_experiments_interactive(base_dir: str = "experiments") -> Tuple[Path, Path]:
    """
    Interactive selection of two experiments

    Args:
        base_dir: Base directory containing experiments

    Returns:
        Tuple of (exp1_path, exp2_path)
    """
    experiments = list_experiments(base_dir)

    if len(experiments) < 2:
        raise ValueError(f"Need at least 2 experiments in {base_dir}/, found {len(experiments)}")

    print("\nAvailable experiments:")
    print("=" * 80)
    for i, exp in enumerate(experiments, 1):
        print(f"  [{i:2d}] {exp.name}")
    print("=" * 80)
    print("")

    # Get first selection
    while True:
        try:
            choice1 = int(input("Select first experiment (number): "))
            if 1 <= choice1 <= len(experiments):
                exp1 = experiments[choice1 - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(experiments)}")
        except (ValueError, KeyboardInterrupt):
            print("\nSelection cancelled")
            sys.exit(0)

    # Get second selection
    while True:
        try:
            choice2 = int(input("Select second experiment (number): "))
            if 1 <= choice2 <= len(experiments):
                if choice2 != choice1:
                    exp2 = experiments[choice2 - 1]
                    break
                else:
                    print("Please select a different experiment")
            else:
                print(f"Please enter a number between 1 and {len(experiments)}")
        except (ValueError, KeyboardInterrupt):
            print("\nSelection cancelled")
            sys.exit(0)

    print("")
    print("Selected experiments:")
    print(f"  Exp 1: {exp1}")
    print(f"  Exp 2: {exp2}")
    print("")

    # Confirm
    confirm = input("Proceed with analysis? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Analysis cancelled")
        sys.exit(0)

    return exp1, exp2


def load_metrics(exp_dir: Path) -> pd.DataFrame:
    """
    Load metrics.csv from experiment directory

    Args:
        exp_dir: Path to experiment directory

    Returns:
        DataFrame with all metrics
    """
    # Try different possible locations
    possible_paths = [
        exp_dir / "logs" / "metrics.csv",
        exp_dir / "logs.pdf" / "metrics.csv",  # Legacy location
        exp_dir / "metrics.csv",
    ]

    for path in possible_paths:
        if path.exists():
            print(f"✓ Found metrics at: {path}")
            df = pd.read_csv(path)
            return df

    raise FileNotFoundError(f"Could not find metrics.csv in {exp_dir}")


def identify_divergence_point(legacy_values: np.ndarray,
                              new_values: np.ndarray,
                              steps: np.ndarray,
                              threshold: float = 2.0) -> int:
    """
    Identify the step where two curves start to diverge significantly

    Args:
        legacy_values: Values from legacy pipeline
        new_values: Values from new pipeline
        steps: Training steps
        threshold: Ratio threshold to consider significant divergence

    Returns:
        Step index where divergence starts
    """
    ratios = legacy_values / (new_values + 1e-8)

    # Find first point where ratio exceeds threshold
    diverged = np.where(ratios > threshold)[0]

    if len(diverged) > 0:
        return int(steps[diverged[0]])
    else:
        return -1  # No significant divergence


def plot_mode_comparison(legacy_df: pd.DataFrame,
                         new_df: pd.DataFrame,
                         mode_name: str,
                         output_dir: Path):
    """
    Create side-by-side comparison plots for a specific mode

    Args:
        legacy_df: Metrics from legacy pipeline
        new_df: Metrics from new pipeline
        mode_name: Name of the mode (e.g., "mode1", "mode2", etc.)
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{mode_name.upper()} Performance Comparison', fontsize=16, fontweight='bold')

    ppl_col = f"{mode_name}_ppl"
    loss_col = f"{mode_name}_loss"

    # Get steps that exist in both datasets
    legacy_steps = legacy_df['step'].values
    new_steps = new_df['step'].values

    legacy_ppl = legacy_df[ppl_col].values
    new_ppl = new_df[ppl_col].values
    legacy_loss = legacy_df[loss_col].values
    new_loss = new_df[loss_col].values

    # Plot 1: Perplexity over steps
    ax1 = axes[0, 0]
    ax1.plot(legacy_steps, legacy_ppl, 'o-', label='Legacy Pipeline', linewidth=2, markersize=6)
    ax1.plot(new_steps, new_ppl, 's-', label='New Pipeline', linewidth=2, markersize=6)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Perplexity', fontsize=12)
    ax1.set_title('Perplexity over Training', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Loss over steps
    ax2 = axes[0, 1]
    ax2.plot(legacy_steps, legacy_loss, 'o-', label='Legacy Pipeline', linewidth=2, markersize=6)
    ax2.plot(new_steps, new_loss, 's-', label='New Pipeline', linewidth=2, markersize=6)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss over Training', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Ratio Legacy/New Perplexity
    ax3 = axes[1, 0]
    ratios = legacy_ppl / (new_ppl + 1e-8)
    ax3.plot(legacy_steps, ratios, 'o-', color='purple', linewidth=2, markersize=6)
    ax3.axhline(y=1.0, color='black', linestyle='--', label='No difference', linewidth=1.5)
    ax3.axhline(y=2.0, color='red', linestyle=':', label='2x threshold', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Training Step', fontsize=12)
    ax3.set_ylabel('Ratio (Legacy / New)', fontsize=12)
    ax3.set_title('Performance Ratio Over Training', fontsize=13)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Identify divergence point
    div_step = identify_divergence_point(legacy_ppl, new_ppl, legacy_steps, threshold=2.0)
    if div_step > 0:
        ax3.axvline(x=div_step, color='orange', linestyle='--',
                   label=f'Divergence at step {div_step}', linewidth=2)
        ax3.legend(fontsize=11)

    # Plot 4: Absolute difference in perplexity
    ax4 = axes[1, 1]
    abs_diff = legacy_ppl - new_ppl
    ax4.plot(legacy_steps, abs_diff, 'o-', color='red', linewidth=2, markersize=6)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Training Step', fontsize=12)
    ax4.set_ylabel('Perplexity Difference (Legacy - New)', fontsize=12)
    ax4.set_title('Absolute Performance Difference', fontsize=13)
    ax4.grid(True, alpha=0.3)

    # Add final values as text
    final_legacy = legacy_ppl[-1]
    final_new = new_ppl[-1]
    final_ratio = final_legacy / final_new

    fig.text(0.5, 0.02,
             f'Final Values: Legacy={final_legacy:.2f}, New={final_new:.2f}, '
             f'Ratio={final_ratio:.2f}x {"(New Better)" if final_ratio > 1 else "(Legacy Better)"}',
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    output_file = output_dir / f'{mode_name}_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_all_modes_overview(legacy_df: pd.DataFrame,
                           new_df: pd.DataFrame,
                           output_dir: Path):
    """
    Create overview plot showing all modes together

    Args:
        legacy_df: Metrics from legacy pipeline
        new_df: Metrics from new pipeline
        output_dir: Directory to save plots
    """
    modes = ['mode1', 'mode2', 'mode3', 'mode4', 'mode5']
    colors_legacy = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
    colors_new = plt.cm.Reds(np.linspace(0.4, 0.9, 5))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('All Modes Performance Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Legacy pipeline all modes
    ax1 = axes[0]
    for i, mode in enumerate(modes):
        ppl_col = f"{mode}_ppl"
        steps = legacy_df['step'].values
        ppl = legacy_df[ppl_col].values
        ax1.plot(steps, ppl, 'o-', label=mode.upper(), color=colors_legacy[i],
                linewidth=2, markersize=5)

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Perplexity', fontsize=12)
    ax1.set_title('Legacy Pipeline - All Modes', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: New pipeline all modes
    ax2 = axes[1]
    for i, mode in enumerate(modes):
        ppl_col = f"{mode}_ppl"
        steps = new_df['step'].values
        ppl = new_df[ppl_col].values
        ax2.plot(steps, ppl, 's-', label=mode.upper(), color=colors_new[i],
                linewidth=2, markersize=5)

    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('New Pipeline - All Modes', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    output_file = output_dir / 'all_modes_overview.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_training_loss_comparison(legacy_df: pd.DataFrame,
                                  new_df: pd.DataFrame,
                                  output_dir: Path):
    """
    Compare training loss between pipelines

    Args:
        legacy_df: Metrics from legacy pipeline
        new_df: Metrics from new pipeline
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Loss Comparison', fontsize=16, fontweight='bold')

    legacy_steps = legacy_df['step'].values
    new_steps = new_df['step'].values
    legacy_train_loss = legacy_df['train_loss'].values
    new_train_loss = new_df['train_loss'].values

    # Plot 1: Training loss over steps
    ax1 = axes[0]
    ax1.plot(legacy_steps, legacy_train_loss, 'o-', label='Legacy Pipeline',
            linewidth=2, markersize=6, color='blue')
    ax1.plot(new_steps, new_train_loss, 's-', label='New Pipeline',
            linewidth=2, markersize=6, color='red')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Progression', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss difference
    ax2 = axes[1]
    loss_diff = legacy_train_loss - new_train_loss
    ax2.plot(legacy_steps, loss_diff, 'o-', color='purple', linewidth=2, markersize=6)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Loss Difference (Legacy - New)', fontsize=12)
    ax2.set_title('Training Loss Difference Over Time', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Add final values
    final_legacy = legacy_train_loss[-1]
    final_new = new_train_loss[-1]

    fig.text(0.5, 0.02,
             f'Final Training Loss: Legacy={final_legacy:.4f}, New={final_new:.4f}, '
             f'Difference={final_legacy-final_new:.4f}',
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    output_file = output_dir / 'training_loss_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def generate_summary_report(legacy_df: pd.DataFrame,
                            new_df: pd.DataFrame,
                            output_dir: Path):
    """
    Generate comprehensive text summary report

    Args:
        legacy_df: Metrics from legacy pipeline
        new_df: Metrics from new pipeline
        output_dir: Directory to save report
    """
    modes = ['mode1', 'mode2', 'mode3', 'mode4', 'mode5']

    report = []
    report.append("=" * 100)
    report.append("PIPELINE DIFFERENCES ANALYSIS REPORT")
    report.append("=" * 100)
    report.append("")

    # Final performance comparison
    report.append("## FINAL PERFORMANCE COMPARISON (Step 2500)")
    report.append("")
    report.append("| Mode | Legacy PPL | New PPL | Ratio (Legacy/New) | Change |")
    report.append("|------|-----------|---------|-------------------|--------|")

    for mode in modes:
        ppl_col = f"{mode}_ppl"
        legacy_final = legacy_df[ppl_col].iloc[-1]
        new_final = new_df[ppl_col].iloc[-1]
        ratio = legacy_final / new_final

        if ratio > 1.5:
            change = f"🟢 New {ratio:.2f}x better"
        elif ratio < 0.67:
            change = f"🔴 Legacy {1/ratio:.2f}x better"
        else:
            change = "🟡 Similar"

        report.append(f"| {mode.upper()} | {legacy_final:.2f} | {new_final:.2f} | {ratio:.2f}x | {change} |")

    report.append("")

    # Training loss comparison
    report.append("## TRAINING LOSS")
    report.append("")
    legacy_train_loss = legacy_df['train_loss'].iloc[-1]
    new_train_loss = new_df['train_loss'].iloc[-1]
    report.append(f"- Legacy final training loss: {legacy_train_loss:.4f}")
    report.append(f"- New final training loss: {new_train_loss:.4f}")
    report.append(f"- Difference: {legacy_train_loss - new_train_loss:.4f}")

    if new_train_loss < legacy_train_loss:
        report.append("- ⚠️  New pipeline has lower training loss (potential overfitting?)")

    report.append("")

    # Divergence analysis for Mode 2
    report.append("## MODE 2 DIVERGENCE ANALYSIS")
    report.append("")

    mode2_ppl_legacy = legacy_df['mode2_ppl'].values
    mode2_ppl_new = new_df['mode2_ppl'].values
    steps = legacy_df['step'].values

    div_step = identify_divergence_point(mode2_ppl_legacy, mode2_ppl_new, steps, threshold=2.0)

    if div_step > 0:
        report.append(f"- **Divergence Point**: Step {div_step}")
        report.append(f"- Mode 2 perplexities started diverging significantly (>2x ratio) at step {div_step}")
    else:
        report.append("- No clear divergence point found (curves never reached 2x ratio)")

    report.append("")

    # Step-by-step progression
    report.append("## MODE 2 PROGRESSION")
    report.append("")
    report.append("| Step | Legacy PPL | New PPL | Ratio | Status |")
    report.append("|------|-----------|---------|-------|--------|")

    for i, step in enumerate(steps):
        legacy_ppl = mode2_ppl_legacy[i]
        new_ppl = mode2_ppl_new[i]
        ratio = legacy_ppl / new_ppl

        if ratio > 2.0:
            status = "🔴 Large divergence"
        elif ratio > 1.2:
            status = "🟡 Moderate divergence"
        elif ratio < 0.8:
            status = "🟢 New winning"
        else:
            status = "⚪ Similar"

        report.append(f"| {step} | {legacy_ppl:.2f} | {new_ppl:.2f} | {ratio:.2f}x | {status} |")

    report.append("")
    report.append("=" * 100)

    # Save report
    report_text = "\n".join(report)
    output_file = output_dir / 'analysis_report.txt'
    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"  Saved: {output_file}")

    # Also print to console
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze pipeline performance differences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python utils/analyze_pipeline_differences.py --list

  # Interactive mode
  python utils/analyze_pipeline_differences.py --interactive

  # Use experiment names (auto-detected in experiments/)
  python utils/analyze_pipeline_differences.py exp1_name exp2_name

  # Use full paths
  python utils/analyze_pipeline_differences.py experiments/exp1 experiments/exp2
        """
    )
    parser.add_argument('exp1', type=str, nargs='?', default=None,
                       help='First experiment (name or path)')
    parser.add_argument('exp2', type=str, nargs='?', default=None,
                       help='Second experiment (name or path)')
    parser.add_argument('--output', type=str, default='pipeline_analysis',
                       help='Output directory for results (default: pipeline_analysis)')
    parser.add_argument('--base-dir', type=str, default='experiments',
                       help='Base directory for experiments (default: experiments)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available experiments and exit')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode: select experiments from list')

    args = parser.parse_args()

    # Handle --list mode
    if args.list:
        experiments = list_experiments(args.base_dir)
        if not experiments:
            print(f"No experiments found in {args.base_dir}/")
            sys.exit(1)

        print(f"\nAvailable experiments in {args.base_dir}/:")
        print("=" * 80)
        for i, exp in enumerate(experiments, 1):
            print(f"  [{i:2d}] {exp.name}")
        print("=" * 80)
        print(f"\nTotal: {len(experiments)} experiments")
        sys.exit(0)

    # Handle --interactive mode
    if args.interactive:
        exp1_path, exp2_path = select_experiments_interactive(args.base_dir)
    else:
        # Normal mode: require both arguments
        if not args.exp1 or not args.exp2:
            parser.error("Two experiments required (or use --interactive mode)")

        # Resolve paths
        try:
            exp1_path = resolve_experiment_path(args.exp1, args.base_dir)
            exp2_path = resolve_experiment_path(args.exp2, args.base_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n❌ Error: {e}\n")
            sys.exit(1)

    # Use resolved paths
    legacy_dir = exp1_path
    new_dir = exp2_path
    output_dir = Path(args.output)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir.absolute()}\n")

    # Load metrics
    print("📊 Loading metrics...")
    legacy_df = load_metrics(legacy_dir)
    new_df = load_metrics(new_dir)
    print(f"  Legacy: {len(legacy_df)} evaluation checkpoints")
    print(f"  New: {len(new_df)} evaluation checkpoints")
    print("")

    # Generate plots
    print("📈 Generating comparison plots...")

    modes = ['mode1', 'mode2', 'mode3', 'mode4', 'mode5']
    for mode in modes:
        print(f"  Analyzing {mode.upper()}...")
        plot_mode_comparison(legacy_df, new_df, mode, output_dir)

    print(f"  Generating overview plot...")
    plot_all_modes_overview(legacy_df, new_df, output_dir)

    print(f"  Generating training loss comparison...")
    plot_training_loss_comparison(legacy_df, new_df, output_dir)

    print("")

    # Generate report
    print("📝 Generating summary report...")
    generate_summary_report(legacy_df, new_df, output_dir)

    print("")
    print("=" * 100)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\n📂 All results saved to: {output_dir.absolute()}")
    print(f"   - Individual mode plots: {output_dir}/*_comparison.png")
    print(f"   - Overview plot: {output_dir}/all_modes_overview.png")
    print(f"   - Training loss: {output_dir}/training_loss_comparison.png")
    print(f"   - Text report: {output_dir}/analysis_report.txt")
    print("")


if __name__ == "__main__":
    main()
