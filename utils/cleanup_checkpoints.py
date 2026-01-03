#!/usr/bin/env python3
"""
Checkpoint Cleanup Script

Removes intermediate checkpoints from experiment folders while keeping
best_model.pt and final_model.pt.

Usage:
    # Preview what will be deleted (dry run - default)
    python utils/cleanup_checkpoints.py experiments/

    # Actually delete files
    python utils/cleanup_checkpoints.py experiments/ --execute

    # Custom folder
    python utils/cleanup_checkpoints.py /path/to/experiments_archive --execute

Run from project root: python utils/cleanup_checkpoints.py
"""

import argparse
import os
import sys
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


def get_intermediate_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """Find all intermediate checkpoint files to delete."""
    patterns = [
        "checkpoint_step_*.pt",
        "checkpoint_epoch*.pt",
    ]

    files_to_delete = []
    for pattern in patterns:
        files_to_delete.extend(checkpoints_dir.glob(pattern))

    return sorted(files_to_delete)


def process_experiment(exp_dir: Path, execute: bool, verbose: bool) -> tuple[int, int, list[str]]:
    """
    Process a single experiment directory.

    Returns:
        (files_count, bytes_freed, warnings)
    """
    checkpoints_dir = exp_dir / "checkpoints"
    warnings = []

    if not checkpoints_dir.exists():
        return 0, 0, []

    # Check if best_model.pt exists
    best_model = checkpoints_dir / "best_model.pt"
    final_model = checkpoints_dir / "final_model.pt"

    if not best_model.exists() and not final_model.exists():
        warnings.append(f"WARNING: {exp_dir.name} has no best_model.pt or final_model.pt!")
    elif not best_model.exists():
        warnings.append(f"WARNING: {exp_dir.name} missing best_model.pt (has final_model.pt)")

    # Find intermediate checkpoints
    files_to_delete = get_intermediate_checkpoints(checkpoints_dir)

    if not files_to_delete:
        return 0, 0, warnings

    # Calculate size
    total_size = sum(f.stat().st_size for f in files_to_delete)

    # Delete or report
    for f in files_to_delete:
        if verbose:
            print(f"  {'Deleting' if execute else 'Would delete'}: {f.name} ({format_size(f.stat().st_size)})")
        if execute:
            f.unlink()

    return len(files_to_delete), total_size, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Clean up intermediate checkpoints from experiment folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/cleanup_checkpoints.py experiments/          # Dry run (preview)
  python utils/cleanup_checkpoints.py experiments/ --execute  # Actually delete
  python utils/cleanup_checkpoints.py /path/to/exps -e -v   # Delete with verbose
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
        help="Actually delete files (default: dry run)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show each file being deleted"
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
    print("=" * 60)
    print("Checkpoint Cleanup Script")
    print("=" * 60)
    print(f"Directory: {base_dir.resolve()}")
    print(f"Mode: {'EXECUTE (will delete files)' if args.execute else 'DRY RUN (preview only)'}")
    print("-" * 60)

    # Process all experiment directories
    total_files = 0
    total_bytes = 0
    all_warnings = []
    experiments_processed = 0

    # Get all subdirectories (experiment folders)
    exp_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    for exp_dir in exp_dirs:
        files, bytes_freed, warnings = process_experiment(exp_dir, args.execute, args.verbose)

        if files > 0 or warnings:
            experiments_processed += 1
            if files > 0:
                print(f"{exp_dir.name}: {files} files, {format_size(bytes_freed)}")
            total_files += files
            total_bytes += bytes_freed
            all_warnings.extend(warnings)

    # Summary
    print("-" * 60)
    print(f"Experiments scanned: {len(exp_dirs)}")
    print(f"Experiments with intermediates: {experiments_processed}")
    print(f"Total files {'deleted' if args.execute else 'to delete'}: {total_files}")
    print(f"Total space {'freed' if args.execute else 'to free'}: {format_size(total_bytes)}")

    # Warnings
    if all_warnings:
        print("\n" + "=" * 60)
        print("WARNINGS")
        print("=" * 60)
        for w in all_warnings:
            print(w)

    # Reminder for dry run
    if not args.execute and total_files > 0:
        print("\n" + "-" * 60)
        print("This was a DRY RUN. To actually delete files, run with --execute:")
        print(f"  python utils/cleanup_checkpoints.py {args.directory} --execute")


if __name__ == "__main__":
    main()
