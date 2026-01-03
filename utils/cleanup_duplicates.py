#!/usr/bin/env python3
"""
Duplicate Experiment Cleanup Script

For experiment folders with the same prefix (before timestamp), keeps only one:
1. If any folder has both plots/ and plots_individual/ → keep that one
2. Otherwise → keep the one with most lines in logs/metrics.csv

Usage:
    # Preview what will be deleted (dry run - default)
    python utils/cleanup_duplicates.py experiments/

    # Actually delete
    python utils/cleanup_duplicates.py experiments/ --execute

Run from project root: python utils/cleanup_duplicates.py
"""

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def get_folder_size(folder: Path) -> int:
    """Get total size of a folder in bytes."""
    total = 0
    for item in folder.rglob('*'):
        if item.is_file():
            total += item.stat().st_size
    return total


def extract_prefix(folder_name: str) -> str | None:
    """
    Extract prefix from folder name (everything before the timestamp).

    Examples:
        cond0-20_max_block_rope_distilgpt2_sigmagpt_random_scramble_20260101_224905
        → cond0-20_max_block_rope_distilgpt2_sigmagpt_random_scramble

        cond0-20_max_block_rope_distilgpt2_sigmagpt_random_scramble_think_expectation_20260101_225403
        → cond0-20_max_block_rope_distilgpt2_sigmagpt_random_scramble_think_expectation
    """
    # Match timestamp pattern: _YYYYMMDD_HHMMSS at the end
    match = re.match(r'^(.+)_(\d{8}_\d{6})$', folder_name)
    if match:
        return match.group(1)
    return None


def has_plot_folders(folder: Path) -> bool:
    """Check if folder has both plots/ and plots_individual/ directories."""
    plots = folder / "plots"
    plots_individual = folder / "plots_individual"
    return plots.is_dir() and plots_individual.is_dir()


def get_metrics_lines(folder: Path) -> int:
    """Get number of lines in logs/metrics.csv (0 if not found)."""
    metrics_file = folder / "logs" / "metrics.csv"
    if not metrics_file.exists():
        return 0
    try:
        with open(metrics_file, 'r') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def select_best_folder(folders: list[Path]) -> tuple[Path, str]:
    """
    Select the best folder to keep from a list of duplicates.

    Returns:
        (best_folder, reason)
    """
    # First priority: folder with both plots/ and plots_individual/
    for folder in folders:
        if has_plot_folders(folder):
            return folder, "has plots/ and plots_individual/"

    # Second priority: most lines in metrics.csv
    best_folder = None
    best_lines = -1
    for folder in folders:
        lines = get_metrics_lines(folder)
        if lines > best_lines:
            best_lines = lines
            best_folder = folder

    if best_folder:
        return best_folder, f"most metrics lines ({best_lines})"

    # Fallback: just keep the first one
    return folders[0], "fallback (first folder)"


def main():
    parser = argparse.ArgumentParser(
        description="Clean up duplicate experiment folders, keeping the best one per config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/cleanup_duplicates.py experiments/          # Dry run (preview)
  python utils/cleanup_duplicates.py experiments/ --execute  # Actually delete
        """
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="experiments",
        help="Directory containing experiment folders (default: experiments/)"
    )
    parser.add_argument(
        "-e", "--execute",
        action="store_true",
        help="Actually delete folders (default: dry run)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed information about each folder"
    )

    args = parser.parse_args()

    # Resolve directory
    base_dir = Path(args.directory)
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    if not base_dir.is_dir():
        print(f"Error: Not a directory: {base_dir}")
        sys.exit(1)

    # Header
    print("=" * 70)
    print("Duplicate Experiment Cleanup Script")
    print("=" * 70)
    print(f"Directory: {base_dir.resolve()}")
    print(f"Mode: {'EXECUTE (will delete folders)' if args.execute else 'DRY RUN (preview only)'}")
    print("-" * 70)

    # Group folders by prefix
    prefix_groups: dict[str, list[Path]] = defaultdict(list)

    for item in sorted(base_dir.iterdir()):
        if not item.is_dir():
            continue
        prefix = extract_prefix(item.name)
        if prefix:
            prefix_groups[prefix].append(item)

    # Process groups with duplicates
    total_deleted = 0
    total_bytes = 0
    groups_with_duplicates = 0

    for prefix, folders in sorted(prefix_groups.items()):
        if len(folders) <= 1:
            continue  # No duplicates

        groups_with_duplicates += 1
        best_folder, reason = select_best_folder(folders)

        print(f"\n{prefix}")
        print(f"  Found {len(folders)} folders, keeping: {best_folder.name}")
        print(f"  Reason: {reason}")

        if args.verbose:
            for folder in folders:
                has_plots = "✓" if has_plot_folders(folder) else "✗"
                metrics_lines = get_metrics_lines(folder)
                keep_marker = " [KEEP]" if folder == best_folder else ""
                print(f"    - {folder.name}: plots={has_plots}, metrics={metrics_lines} lines{keep_marker}")

        # Delete non-best folders
        for folder in folders:
            if folder == best_folder:
                continue

            folder_size = get_folder_size(folder)
            total_bytes += folder_size
            total_deleted += 1

            print(f"  {'Deleting' if args.execute else 'Would delete'}: {folder.name} ({format_size(folder_size)})")

            if args.execute:
                shutil.rmtree(folder)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total experiment folders scanned: {sum(len(f) for f in prefix_groups.values())}")
    print(f"Groups with duplicates: {groups_with_duplicates}")
    print(f"Folders {'deleted' if args.execute else 'to delete'}: {total_deleted}")
    print(f"Space {'freed' if args.execute else 'to free'}: {format_size(total_bytes)}")

    # Reminder for dry run
    if not args.execute and total_deleted > 0:
        print("\n" + "-" * 70)
        print("This was a DRY RUN. To actually delete folders, run with --execute:")
        print(f"  python utils/cleanup_duplicates.py {args.directory} --execute")


if __name__ == "__main__":
    main()
