#!/usr/bin/env python3
"""
Duplicate Experiment Cleanup Script

For experiment folders with the same prefix (before timestamp), keeps only one.

Two modes available:
  --mode best (default):
    1. If any folder has both plots/ and plots_individual/ → keep that one
    2. Otherwise → keep the one with most lines in logs/metrics.csv

  --mode earliest:
    1. Keep only the folder with earliest timestamp, delete all others
    2. If the earliest folder has no checkpoints → delete it too

Usage:
    # Preview what will be deleted (dry run - default)
    python utils/cleanup_duplicates.py experiments/

    # Use earliest mode
    python utils/cleanup_duplicates.py experiments/ --mode earliest

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


def extract_timestamp(folder_name: str) -> str | None:
    """Extract timestamp from folder name (YYYYMMDD_HHMMSS)."""
    match = re.match(r'^(.+)_(\d{8}_\d{6})$', folder_name)
    if match:
        return match.group(2)
    return None


def has_checkpoints(folder: Path) -> bool:
    """Check if folder has any checkpoint files."""
    checkpoints_dir = folder / "checkpoints"
    if not checkpoints_dir.is_dir():
        return False
    # Check for any .pt files
    pt_files = list(checkpoints_dir.glob("*.pt"))
    return len(pt_files) > 0


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


def select_earliest_folder(folders: list[Path]) -> tuple[Path, str]:
    """
    Select the folder with earliest timestamp.

    Returns:
        (earliest_folder, reason)
    """
    earliest_folder = None
    earliest_timestamp = None

    for folder in folders:
        timestamp = extract_timestamp(folder.name)
        if timestamp:
            if earliest_timestamp is None or timestamp < earliest_timestamp:
                earliest_timestamp = timestamp
                earliest_folder = folder

    if earliest_folder:
        return earliest_folder, f"earliest timestamp ({earliest_timestamp})"

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
    parser.add_argument(
        "-m", "--mode",
        choices=["best", "earliest"],
        default="best",
        help="Selection mode: 'best' (plots/metrics), 'earliest' (timestamp)"
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
    print(f"Selection mode: {args.mode}")
    print(f"Execute: {'YES (will delete folders)' if args.execute else 'NO (dry run)'}")
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
    empty_earliest_deleted = 0

    for prefix, folders in sorted(prefix_groups.items()):
        if len(folders) <= 1:
            # For earliest mode, still check if single folder has no checkpoints
            if args.mode == "earliest" and len(folders) == 1:
                folder = folders[0]
                if not has_checkpoints(folder):
                    folder_size = get_folder_size(folder)
                    total_bytes += folder_size
                    total_deleted += 1
                    empty_earliest_deleted += 1

                    print(f"\n{prefix}")
                    print(f"  Single folder with NO checkpoints")
                    print(f"  {'Deleting' if args.execute else 'Would delete'}: {folder.name} ({format_size(folder_size)})")

                    if args.execute:
                        shutil.rmtree(folder)
            continue

        groups_with_duplicates += 1

        # Select folder based on mode
        if args.mode == "earliest":
            keep_folder, reason = select_earliest_folder(folders)
        else:
            keep_folder, reason = select_best_folder(folders)

        print(f"\n{prefix}")
        print(f"  Found {len(folders)} folders, keeping: {keep_folder.name}")
        print(f"  Reason: {reason}")

        if args.verbose:
            for folder in folders:
                has_plots = "✓" if has_plot_folders(folder) else "✗"
                has_ckpt = "✓" if has_checkpoints(folder) else "✗"
                metrics_lines = get_metrics_lines(folder)
                timestamp = extract_timestamp(folder.name) or "?"
                keep_marker = " [KEEP]" if folder == keep_folder else ""
                print(f"    - {folder.name}: ts={timestamp}, ckpt={has_ckpt}, plots={has_plots}, metrics={metrics_lines}{keep_marker}")

        # Delete non-kept folders
        for folder in folders:
            if folder == keep_folder:
                continue

            folder_size = get_folder_size(folder)
            total_bytes += folder_size
            total_deleted += 1

            print(f"  {'Deleting' if args.execute else 'Would delete'}: {folder.name} ({format_size(folder_size)})")

            if args.execute:
                shutil.rmtree(folder)

        # For earliest mode: also delete the kept folder if it has no checkpoints
        if args.mode == "earliest" and not has_checkpoints(keep_folder):
            folder_size = get_folder_size(keep_folder)
            total_bytes += folder_size
            total_deleted += 1
            empty_earliest_deleted += 1

            print(f"  Earliest folder has NO checkpoints!")
            print(f"  {'Deleting' if args.execute else 'Would delete'}: {keep_folder.name} ({format_size(folder_size)})")

            if args.execute:
                shutil.rmtree(keep_folder)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Selection mode: {args.mode}")
    print(f"Total experiment folders scanned: {sum(len(f) for f in prefix_groups.values())}")
    print(f"Groups with duplicates: {groups_with_duplicates}")
    print(f"Folders {'deleted' if args.execute else 'to delete'}: {total_deleted}")
    if args.mode == "earliest" and empty_earliest_deleted > 0:
        print(f"  (includes {empty_earliest_deleted} earliest folders with no checkpoints)")
    print(f"Space {'freed' if args.execute else 'to free'}: {format_size(total_bytes)}")

    # Reminder for dry run
    if not args.execute and total_deleted > 0:
        print("\n" + "-" * 70)
        print("This was a DRY RUN. To actually delete folders, run with --execute:")
        print(f"  python utils/cleanup_duplicates.py {args.directory} --mode {args.mode} --execute")


if __name__ == "__main__":
    main()
