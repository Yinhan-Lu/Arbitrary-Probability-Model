"""
Detailed Statistical Comparison of Two Experiments

This script provides comprehensive statistical analysis of performance differences
between two training runs, focusing on:
- Mean, variance, and trend analysis
- Statistical significance tests
- Anomaly detection
- Convergence analysis

Usage:
    python utils/compare_experiments.py \
        result_of_two_pipeline/legacy_pipeline \
        result_of_two_pipeline/new_pipeline

Run from project root.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
from typing import Dict, Tuple
from scipy import stats

print("=" * 100)
print("DETAILED STATISTICAL COMPARISON")
print("=" * 100)


def list_experiments(base_dir: str = "experiments"):
    """List all experiment directories in base_dir"""
    from pathlib import Path
    base_path = Path(base_dir)

    if not base_path.exists():
        return []

    experiments = []
    for item in base_path.iterdir():
        if item.is_dir():
            has_metrics = (item / "logs" / "metrics.csv").exists() or \
                         (item / "logs.pdf" / "metrics.csv").exists() or \
                         (item / "metrics.csv").exists()
            if has_metrics:
                experiments.append(item)

    return sorted(experiments)


def resolve_experiment_path(name_or_path: str, base_dir: str = "experiments"):
    """Resolve experiment name to full path"""
    from pathlib import Path
    path = Path(name_or_path)

    if path.is_absolute() or str(path).startswith(f"{base_dir}/"):
        if path.exists():
            return path
        else:
            raise FileNotFoundError(f"Experiment not found: {path}")

    base_path = Path(base_dir)
    candidate = base_path / name_or_path

    if candidate.exists():
        return candidate

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


def select_experiments_interactive(base_dir: str = "experiments"):
    """Interactive selection of two experiments"""
    experiments = list_experiments(base_dir)

    if len(experiments) < 2:
        raise ValueError(f"Need at least 2 experiments in {base_dir}/, found {len(experiments)}")

    print("\nAvailable experiments:")
    print("=" * 80)
    for i, exp in enumerate(experiments, 1):
        print(f"  [{i:2d}] {exp.name}")
    print("=" * 80)
    print("")

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

    confirm = input("Proceed with comparison? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Comparison cancelled")
        sys.exit(0)

    return exp1, exp2


def load_metrics(exp_dir: Path) -> pd.DataFrame:
    """Load metrics.csv from experiment directory"""
    possible_paths = [
        exp_dir / "logs" / "metrics.csv",
        exp_dir / "logs.pdf" / "metrics.csv",
        exp_dir / "metrics.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    raise FileNotFoundError(f"Could not find metrics.csv in {exp_dir}")


def compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a metric

    Args:
        values: Array of metric values over time

    Returns:
        Dictionary of statistics
    """
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'final': values[-1],
        'initial': values[0],
        'improvement': (values[0] - values[-1]) / values[0] * 100,  # % improvement
        'variance': np.var(values),
        'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,  # Coefficient of variation
    }


def detect_anomalies(values: np.ndarray, threshold: float = 2.0) -> Tuple[int, List[int]]:
    """
    Detect anomalous spikes or drops in training curves

    Args:
        values: Array of metric values
        threshold: Z-score threshold for anomaly detection

    Returns:
        Number of anomalies and list of indices where they occur
    """
    # Compute differences between consecutive points
    diffs = np.diff(values)

    # Compute z-scores of differences
    z_scores = np.abs(stats.zscore(diffs))

    # Find anomalies
    anomalies = np.where(z_scores > threshold)[0]

    return len(anomalies), anomalies.tolist()


def analyze_convergence(values: np.ndarray, window: int = 3) -> Dict[str, float]:
    """
    Analyze convergence behavior

    Args:
        values: Array of metric values
        window: Window size for moving average

    Returns:
        Dictionary of convergence metrics
    """
    # Compute moving average
    ma = np.convolve(values, np.ones(window) / window, mode='valid')

    # Compute slope of last few points (is it still improving?)
    last_points = values[-window:]
    slope, _ = np.polyfit(np.arange(len(last_points)), last_points, 1)

    # Compute stability (variance of last few points)
    stability = np.std(last_points)

    return {
        'final_slope': slope,
        'final_stability': stability,
        'is_converged': abs(slope) < 0.1 and stability < 1.0,
    }


def compare_modes(legacy_df: pd.DataFrame,
                 new_df: pd.DataFrame,
                 modes: List[str]) -> pd.DataFrame:
    """
    Create comprehensive comparison table for all modes

    Args:
        legacy_df: Metrics from legacy pipeline
        new_df: Metrics from new pipeline
        modes: List of mode names

    Returns:
        DataFrame with comparison statistics
    """
    results = []

    for mode in modes:
        ppl_col = f"{mode}_ppl"
        loss_col = f"{mode}_loss"

        legacy_ppl = legacy_df[ppl_col].values
        new_ppl = new_df[ppl_col].values

        # Compute statistics
        legacy_stats = compute_statistics(legacy_ppl)
        new_stats = compute_statistics(new_ppl)

        # Convergence analysis
        legacy_conv = analyze_convergence(legacy_ppl)
        new_conv = analyze_convergence(new_ppl)

        # Anomaly detection
        legacy_anomalies, _ = detect_anomalies(legacy_ppl)
        new_anomalies, _ = detect_anomalies(new_ppl)

        # Performance ratio
        final_ratio = legacy_stats['final'] / new_stats['final']

        results.append({
            'Mode': mode.upper(),
            'Legacy Final': f"{legacy_stats['final']:.2f}",
            'New Final': f"{new_stats['final']:.2f}",
            'Ratio': f"{final_ratio:.2f}x",
            'Legacy Improvement': f"{legacy_stats['improvement']:.1f}%",
            'New Improvement': f"{new_stats['improvement']:.1f}%",
            'Legacy Stability': f"{legacy_stats['cv']:.3f}",
            'New Stability': f"{new_stats['cv']:.3f}",
            'Legacy Converged': "✓" if legacy_conv['is_converged'] else "✗",
            'New Converged': "✓" if new_conv['is_converged'] else "✗",
            'Legacy Anomalies': legacy_anomalies,
            'New Anomalies': new_anomalies,
        })

    return pd.DataFrame(results)


def analyze_training_dynamics(legacy_df: pd.DataFrame,
                              new_df: pd.DataFrame) -> Dict:
    """
    Analyze training dynamics (learning curves, loss progression)

    Args:
        legacy_df: Metrics from legacy pipeline
        new_df: Metrics from new pipeline

    Returns:
        Dictionary of analysis results
    """
    legacy_train_loss = legacy_df['train_loss'].values
    new_train_loss = new_df['train_loss'].values

    legacy_stats = compute_statistics(legacy_train_loss)
    new_stats = compute_statistics(new_train_loss)

    # Check for overfitting indicators
    # Lower training loss but not proportionally better eval loss = overfitting
    legacy_mode2_loss = legacy_df['mode2_loss'].values[-1]
    new_mode2_loss = new_df['mode2_loss'].values[-1]

    train_gap_legacy = legacy_mode2_loss - legacy_stats['final']
    train_gap_new = new_mode2_loss - new_stats['final']

    return {
        'legacy': legacy_stats,
        'new': new_stats,
        'legacy_train_eval_gap': train_gap_legacy,
        'new_train_eval_gap': train_gap_new,
        'potential_overfitting_new': train_gap_new > train_gap_legacy * 1.5,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Detailed statistical comparison of experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python utils/compare_experiments.py --list

  # Interactive mode
  python utils/compare_experiments.py --interactive

  # Use experiment names (auto-detected in experiments/)
  python utils/compare_experiments.py exp1_name exp2_name

  # Use full paths
  python utils/compare_experiments.py experiments/exp1 experiments/exp2
        """
    )
    parser.add_argument('exp1', type=str, nargs='?', default=None,
                       help='First experiment (name or path)')
    parser.add_argument('exp2', type=str, nargs='?', default=None,
                       help='Second experiment (name or path)')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional output file for results (default: print to console)')
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

    print(f"\n📊 Loading experiments...")
    print(f"  Legacy: {legacy_dir}")
    print(f"  New: {new_dir}")
    print("")

    # Load metrics
    legacy_df = load_metrics(legacy_dir)
    new_df = load_metrics(new_dir)

    print(f"✓ Loaded {len(legacy_df)} checkpoints from each experiment\n")

    # Compare all modes
    print("=" * 100)
    print("MODE-BY-MODE COMPARISON")
    print("=" * 100)
    print("")

    modes = ['mode1', 'mode2', 'mode3', 'mode4', 'mode5']
    comparison_df = compare_modes(legacy_df, new_df, modes)

    print(comparison_df.to_string(index=False))
    print("")

    # Training dynamics analysis
    print("=" * 100)
    print("TRAINING DYNAMICS ANALYSIS")
    print("=" * 100)
    print("")

    dynamics = analyze_training_dynamics(legacy_df, new_df)

    print(f"Legacy Training Loss:")
    print(f"  Initial: {dynamics['legacy']['initial']:.4f}")
    print(f"  Final: {dynamics['legacy']['final']:.4f}")
    print(f"  Improvement: {dynamics['legacy']['improvement']:.1f}%")
    print(f"  Stability (CV): {dynamics['legacy']['cv']:.4f}")
    print("")

    print(f"New Training Loss:")
    print(f"  Initial: {dynamics['new']['initial']:.4f}")
    print(f"  Final: {dynamics['new']['final']:.4f}")
    print(f"  Improvement: {dynamics['new']['improvement']:.1f}%")
    print(f"  Stability (CV): {dynamics['new']['cv']:.4f}")
    print("")

    print(f"Train-Eval Gap Analysis (Mode 2):")
    print(f"  Legacy gap: {dynamics['legacy_train_eval_gap']:.4f}")
    print(f"  New gap: {dynamics['new_train_eval_gap']:.4f}")

    if dynamics['potential_overfitting_new']:
        print(f"  ⚠️  WARNING: New pipeline shows larger train-eval gap (potential overfitting)")
    else:
        print(f"  ✓ No significant overfitting detected")

    print("")

    # Detailed Mode 2 analysis
    print("=" * 100)
    print("DETAILED MODE 2 ANALYSIS")
    print("=" * 100)
    print("")

    mode2_legacy = legacy_df['mode2_ppl'].values
    mode2_new = new_df['mode2_ppl'].values
    steps = legacy_df['step'].values

    print("Step-by-Step Progression:")
    print("")
    print(f"{'Step':<10} {'Legacy PPL':<15} {'New PPL':<15} {'Ratio':<10} {'Abs Diff':<15}")
    print("-" * 70)

    for i, step in enumerate(steps):
        ratio = mode2_legacy[i] / mode2_new[i]
        diff = mode2_legacy[i] - mode2_new[i]
        print(f"{step:<10} {mode2_legacy[i]:<15.2f} {mode2_new[i]:<15.2f} "
              f"{ratio:<10.2f} {diff:<15.2f}")

    print("")

    # Statistical significance test
    print("=" * 100)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("=" * 100)
    print("")

    # T-test for Mode 2
    t_stat, p_value = stats.ttest_ind(mode2_legacy, mode2_new)
    print(f"Mode 2 Independent T-Test:")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")

    if p_value < 0.001:
        print(f"  ✓ Highly significant difference (p < 0.001)")
    elif p_value < 0.05:
        print(f"  ✓ Significant difference (p < 0.05)")
    else:
        print(f"  ✗ Not statistically significant (p >= 0.05)")

    print("")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(mode2_legacy)**2 + np.std(mode2_new)**2) / 2)
    cohens_d = (np.mean(mode2_legacy) - np.mean(mode2_new)) / pooled_std
    print(f"Effect Size (Cohen's d): {cohens_d:.4f}")

    if abs(cohens_d) > 0.8:
        print(f"  → Large effect size")
    elif abs(cohens_d) > 0.5:
        print(f"  → Medium effect size")
    else:
        print(f"  → Small effect size")

    print("")

    # Save results if output specified
    if args.output:
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            f.write(comparison_df.to_string(index=False))
        print(f"✓ Results saved to: {output_file}")
        print("")

    print("=" * 100)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 100)
    print("")


if __name__ == "__main__":
    main()
