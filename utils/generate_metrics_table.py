#!/usr/bin/env python3
"""
Generate metrics table from comparison results.

Reads final_metrics.json from a comparison folder and outputs a formatted table
where rows are models and columns are metrics.

Usage:
    # Text formats
    python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456
    python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456 --format markdown
    python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456 --format latex

    # Figure formats (for slides)
    python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456 --format figure
    python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456 --format heatmap
    python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456 --format bar
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Metric display names for better readability
METRIC_DISPLAY_NAMES = {
    'train_loss': 'Train Loss',
    'train_perplexity': 'Train PPL',
    'eval_loss': 'Eval Loss',
    'eval_perplexity': 'Eval PPL',
    'mode1_loss': 'M1 Loss',
    'mode1_ppl': 'M1 PPL',
    'mode2_loss': 'M2 Loss',
    'mode2_ppl': 'M2 PPL',
    'mode3_loss': 'M3 Loss',
    'mode3_ppl': 'M3 PPL',
    'mode4_loss': 'M4 Loss',
    'mode4_ppl': 'M4 PPL',
    'mode5_loss': 'M5 Loss',
    'mode5_ppl': 'M5 PPL',
}

# Default metrics to show (in order)
DEFAULT_METRICS = [
    'mode1_ppl',
    'mode2_ppl',
    'mode3_ppl',
    'mode4_ppl',
    'mode5_ppl',
]

# Metric presets for convenience
METRIC_PRESETS = {
    'ppl': ['mode1_ppl', 'mode2_ppl', 'mode3_ppl', 'mode4_ppl', 'mode5_ppl'],
    'loss': ['mode1_loss', 'mode2_loss', 'mode3_loss', 'mode4_loss', 'mode5_loss'],
    'all': [
        'mode1_loss', 'mode1_ppl',
        'mode2_loss', 'mode2_ppl',
        'mode3_loss', 'mode3_ppl',
        'mode4_loss', 'mode4_ppl',
        'mode5_loss', 'mode5_ppl',
    ],
}


def load_final_metrics(comparison_dir):
    """Load final_metrics.json from comparison directory.

    Args:
        comparison_dir: Path to comparison results folder

    Returns:
        Dictionary with metrics data
    """
    comparison_path = Path(comparison_dir)
    metrics_file = comparison_path / 'final_metrics.json'

    if not metrics_file.exists():
        raise FileNotFoundError(f"final_metrics.json not found at {metrics_file}")

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    return data


def build_table_data(metrics_data, selected_metrics=None):
    """Build table data from metrics dictionary.

    Args:
        metrics_data: Dictionary from final_metrics.json
        selected_metrics: List of metric names to include (None = use defaults)

    Returns:
        Tuple of (headers, rows) where:
        - headers: List of column names ['Model', 'M1 PPL', 'M2 PPL', ...]
        - rows: List of [model_name, value1, value2, ...] lists
    """
    metrics = metrics_data.get('metrics', {})

    if not metrics:
        raise ValueError("No metrics found in final_metrics.json")

    # Determine which metrics to use
    if selected_metrics is None:
        # Use default metrics that exist in the data
        metric_names = [m for m in DEFAULT_METRICS if m in metrics]
    else:
        # Use specified metrics that exist in the data
        metric_names = [m for m in selected_metrics if m in metrics]

    if not metric_names:
        # Fall back to all available metrics
        metric_names = list(metrics.keys())

    # Get all models across all metrics
    all_models = set()
    for metric_name in metric_names:
        all_models.update(metrics.get(metric_name, {}).keys())

    # Sort models for consistent ordering
    models = sorted(all_models)

    # Build headers
    headers = ['Model']
    for metric_name in metric_names:
        display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        headers.append(display_name)

    # Build rows
    rows = []
    for model in models:
        row = [model]
        for metric_name in metric_names:
            value = metrics.get(metric_name, {}).get(model)
            if value is not None:
                # Format number with appropriate precision
                if 'ppl' in metric_name.lower() or 'perplexity' in metric_name.lower():
                    row.append(f"{value:.2f}")
                else:
                    row.append(f"{value:.4f}")
            else:
                row.append('-')
        rows.append(row)

    return headers, rows, metric_names


def format_console_table(headers, rows):
    """Format table for console output.

    Args:
        headers: List of column headers
        rows: List of row data

    Returns:
        Formatted string
    """
    # Calculate column widths
    col_widths = []
    for i in range(len(headers)):
        width = len(headers[i])
        for row in rows:
            if i < len(row):
                width = max(width, len(str(row[i])))
        col_widths.append(width + 2)  # Add padding

    # Build table
    lines = []

    # Header separator
    sep = '+' + '+'.join('-' * w for w in col_widths) + '+'
    lines.append(sep)

    # Headers
    header_line = '|' + '|'.join(
        str(headers[i]).center(col_widths[i])
        for i in range(len(headers))
    ) + '|'
    lines.append(header_line)
    lines.append(sep)

    # Data rows
    for row in rows:
        row_line = '|' + '|'.join(
            str(row[i] if i < len(row) else '').center(col_widths[i])
            for i in range(len(headers))
        ) + '|'
        lines.append(row_line)

    lines.append(sep)

    return '\n'.join(lines)


def format_markdown_table(headers, rows):
    """Format table as Markdown.

    Args:
        headers: List of column headers
        rows: List of row data

    Returns:
        Markdown table string
    """
    lines = []

    # Headers
    header_line = '| ' + ' | '.join(headers) + ' |'
    lines.append(header_line)

    # Separator (right-align numeric columns)
    sep_parts = []
    for i, h in enumerate(headers):
        if i == 0:
            sep_parts.append(':---')  # Left align model name
        else:
            sep_parts.append('---:')  # Right align numbers
    sep_line = '| ' + ' | '.join(sep_parts) + ' |'
    lines.append(sep_line)

    # Data rows
    for row in rows:
        row_line = '| ' + ' | '.join(str(v) for v in row) + ' |'
        lines.append(row_line)

    return '\n'.join(lines)


def format_latex_table(headers, rows, metric_names):
    """Format table as LaTeX.

    Args:
        headers: List of column headers
        rows: List of row data
        metric_names: List of metric names (for caption)

    Returns:
        LaTeX table string
    """
    lines = []

    # Begin table
    num_cols = len(headers)
    col_spec = 'l' + 'r' * (num_cols - 1)  # Left for model, right for numbers

    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')

    # Headers
    header_line = ' & '.join(f'\\textbf{{{h}}}' for h in headers) + ' \\\\'
    lines.append(header_line)
    lines.append('\\midrule')

    # Data rows
    for row in rows:
        row_line = ' & '.join(str(v) for v in row) + ' \\\\'
        lines.append(row_line)

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\caption{Final metrics comparison across models.}')
    lines.append('\\label{tab:metrics_comparison}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


def generate_table_figure(headers, rows, output_path, title=None, highlight_best=True):
    """Generate a styled table as PNG image for slides.

    Args:
        headers: List of column headers
        rows: List of row data (first column is model name, rest are values)
        output_path: Path to save the PNG file
        title: Optional title for the table
        highlight_best: Whether to highlight best values per column (green)

    Returns:
        Path to saved file
    """
    # Parse numeric values for highlighting
    num_cols = len(headers)
    num_rows = len(rows)

    # Convert string values to floats where possible
    numeric_data = []
    for row in rows:
        row_values = []
        for i, val in enumerate(row[1:], 1):  # Skip model name
            try:
                row_values.append(float(val))
            except (ValueError, TypeError):
                row_values.append(None)
        numeric_data.append(row_values)

    # Find best (minimum) value per column for highlighting
    best_indices = []
    if highlight_best and numeric_data:
        for col_idx in range(num_cols - 1):
            col_values = [row[col_idx] for row in numeric_data if row[col_idx] is not None]
            if col_values:
                min_val = min(col_values)
                best_idx = None
                for row_idx, row in enumerate(numeric_data):
                    if row[col_idx] == min_val:
                        best_idx = row_idx
                        break
                best_indices.append(best_idx)
            else:
                best_indices.append(None)

    # Calculate max model name length for dynamic sizing
    max_model_len = max(len(str(row[0])) for row in rows) if rows else 10
    model_col_width = max(3, max_model_len * 0.12)  # Scale factor for width

    # Create figure with dynamic width based on content
    fig_width = max(12, model_col_width + 1.2 * (num_cols - 1))
    fig_height = max(3, 0.6 * (num_rows + 1))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * num_cols,  # Header background (blue)
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)

    # Auto-fit column widths based on content
    table.auto_set_column_width(col=list(range(num_cols)))

    # Style header row
    for j in range(num_cols):
        cell = table[(0, j)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4472C4')
        cell.set_height(0.15)

    # Style data rows
    for i in range(num_rows):
        for j in range(num_cols):
            cell = table[(i + 1, j)]

            # Alternate row colors
            if i % 2 == 0:
                cell.set_facecolor('#D6DCE5')
            else:
                cell.set_facecolor('#FFFFFF')

            # Model name column (first column) - left align, bold
            if j == 0:
                cell.set_text_props(weight='bold', ha='left')
                cell.get_text().set_position((0.05, 0.5))
            else:
                # Highlight best values (green background, bold)
                if highlight_best and j - 1 < len(best_indices):
                    if best_indices[j - 1] == i:
                        cell.set_facecolor('#C6EFCE')  # Light green
                        cell.set_text_props(weight='bold', color='#006100')

    # Add title if provided
    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate metrics table from comparison results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (console table with default metrics)
  python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456

  # Markdown format
  python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456 --format markdown

  # LaTeX format
  python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456 --format latex

  # PNG figure for slides (default: generates BOTH ppl and loss figures)
  python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456 --format figure
  # Output: metrics_table_ppl.png, metrics_table_loss.png

  # Generate single figure with specific preset
  python utils/generate_metrics_table.py ... --format figure -o my_table.png
  python utils/generate_metrics_table.py ... --format figure --metrics mode1_ppl mode3_ppl -o custom.png

  # Select specific metrics
  python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456 --metrics mode1_ppl mode3_ppl

  # Show all available metrics
  python utils/generate_metrics_table.py comparison_between_experiments/20241224_123456 --all-metrics

Presets: ppl (default), loss, all

Available metrics:
  - train_loss, train_perplexity
  - mode1_loss, mode1_ppl (Autoregressive)
  - mode2_loss, mode2_ppl (Filling In)
  - mode3_loss, mode3_ppl (Training Distribution)
  - mode4_loss, mode4_ppl (Mode 2 Eval + Mode 1 Logits)
  - mode5_loss, mode5_ppl (Mode 3 Eval + Mode 1 Logits)
        """
    )

    parser.add_argument('comparison_dir', type=str,
                       help='Path to comparison results folder')
    parser.add_argument('--format', '-f', type=str,
                       choices=['console', 'markdown', 'latex', 'figure'],
                       default='console',
                       help='Output format (default: console, use "figure" for PNG)')
    parser.add_argument('--preset', '-p', type=str,
                       choices=['ppl', 'loss', 'all'],
                       default='ppl',
                       help='Metric preset: ppl (default), loss, or all')
    parser.add_argument('--metrics', '-m', type=str, nargs='+',
                       help='Specific metrics to include (overrides preset)')
    parser.add_argument('--all-metrics', '-a', action='store_true',
                       help='Include all available metrics (overrides preset)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (default: print to stdout, or metrics_table.png for figure)')
    parser.add_argument('--title', '-t', type=str,
                       help='Title for figure format')
    parser.add_argument('--no-highlight', action='store_true',
                       help='Disable highlighting of best values in figure format')

    args = parser.parse_args()

    try:
        # Load data
        data = load_final_metrics(args.comparison_dir)

        # Determine metrics to show (priority: --all-metrics > --metrics > --preset)
        if args.all_metrics:
            selected_metrics = None  # Will use all available
        elif args.metrics:
            selected_metrics = args.metrics
        else:
            selected_metrics = METRIC_PRESETS.get(args.preset, DEFAULT_METRICS)

        # Build table
        headers, rows, metric_names = build_table_data(data, selected_metrics)

        if not rows:
            print("No data found in final_metrics.json")
            return

        # Handle figure format separately
        if args.format == 'figure':
            comparison_path = Path(args.comparison_dir)

            # If user specified --output or --metrics, generate single figure
            if args.output or args.metrics or args.all_metrics:
                if args.output:
                    output_path = Path(args.output)
                else:
                    output_path = comparison_path / 'metrics_table.png'

                generate_table_figure(
                    headers, rows, output_path,
                    title=args.title,
                    highlight_best=not args.no_highlight
                )
                print(f"Table figure saved to: {output_path}")
                print(f"  - Models: {len(rows)}")
                print(f"  - Metrics: {len(headers) - 1}")
                print(f"  - Best values highlighted: {not args.no_highlight}")
            else:
                # Default: generate both PPL and Loss figures
                for preset_name in ['ppl', 'loss']:
                    preset_metrics = METRIC_PRESETS[preset_name]
                    h, r, _ = build_table_data(data, preset_metrics)

                    output_path = comparison_path / f'metrics_table_{preset_name}.png'
                    title = f"{'Perplexity' if preset_name == 'ppl' else 'Loss'} Comparison"

                    generate_table_figure(
                        h, r, output_path,
                        title=title,
                        highlight_best=not args.no_highlight
                    )
                    print(f"Table figure saved to: {output_path}")

                print(f"\nGenerated 2 figures:")
                print(f"  - metrics_table_ppl.png (Perplexity)")
                print(f"  - metrics_table_loss.png (Loss)")
                print(f"  - Models: {len(rows)}")
                print(f"  - Best values highlighted: {not args.no_highlight}")
            return

        # Format text output
        if args.format == 'console':
            output = format_console_table(headers, rows)
        elif args.format == 'markdown':
            output = format_markdown_table(headers, rows)
        elif args.format == 'latex':
            output = format_latex_table(headers, rows, metric_names)
        else:
            output = format_console_table(headers, rows)

        # Output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Table saved to: {args.output}")
        else:
            print(output)

        # Print metadata
        if args.format == 'console':
            print(f"\nSource: {args.comparison_dir}/final_metrics.json")
            print(f"Created: {data.get('created_at', 'unknown')}")
            print(f"Models: {len(rows)}, Metrics: {len(headers) - 1}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
