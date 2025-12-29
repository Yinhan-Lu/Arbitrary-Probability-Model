#!/usr/bin/env python3
"""
Consolidate experiment folders that were split due to checkpoint resume bug.

The bug caused resumed training to create new folders with appended timestamps:
- Original: cond0-60_..._20251227_070135/
- After resume: cond0-60_..._20251227_070135_20251227_204239/

This script merges them back into a single folder.

Usage:
    python scripts/consolidate_experiments.py --dry-run      # Preview changes
    python scripts/consolidate_experiments.py                # Run merge
    python scripts/consolidate_experiments.py --delete-source  # Merge and delete source
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


# Timestamp pattern: YYYYMMDD_HHMMSS
TIMESTAMP_PATTERN = re.compile(r'_(\d{8}_\d{6})')


def count_timestamps(folder_name: str) -> int:
    """Count number of timestamps in folder name."""
    return len(TIMESTAMP_PATTERN.findall(folder_name))


def find_experiment_groups(experiments_dir: Path) -> dict[str, list[Path]]:
    """
    Group experiment folders by their resume chain.

    Key insight: If folder B's name starts with folder A's name + "_",
    then B is a resume of A (due to the bug that appended timestamps).

    Example chain:
        cond0-60_..._20251227_070135                                    (original)
        cond0-60_..._20251227_070135_20251227_204239                    (resume 1)
        cond0-60_..._20251227_070135_20251227_204239_20251228_115938    (resume 2)

    NOT a chain (different experiments):
        cond0-60_..._20251227_070135
        cond0-60_..._20251228_100000  <- Different first timestamp = different experiment

    Returns:
        Dict mapping original_folder_name -> list of folders in chain (sorted by name length)
    """
    # Get all folders with at least one timestamp
    folders_with_timestamps = []
    for folder in experiments_dir.iterdir():
        if folder.is_dir() and count_timestamps(folder.name) >= 1:
            folders_with_timestamps.append(folder)

    # Sort by name length (shorter = more likely to be original)
    folders_with_timestamps.sort(key=lambda f: len(f.name))

    # Build chains: find folders that are prefixes of other folders
    chains = {}  # original_name -> [original, resume1, resume2, ...]
    assigned = set()  # folders already assigned to a chain

    for folder in folders_with_timestamps:
        if folder.name in assigned:
            continue

        # Start a new chain with this folder as the original
        chain = [folder]
        assigned.add(folder.name)

        # Find all folders that start with this folder's name + "_" + timestamp
        prefix = folder.name + "_"
        for other in folders_with_timestamps:
            if other.name in assigned:
                continue
            if other.name.startswith(prefix):
                # Verify the suffix is timestamp(s)
                suffix = other.name[len(folder.name):]
                if TIMESTAMP_PATTERN.match(suffix):
                    chain.append(other)
                    assigned.add(other.name)

        # Sort chain by name length (original first, then resumes in order)
        chain.sort(key=lambda f: len(f.name))

        # Only keep chains with more than one folder (needs merging)
        if len(chain) > 1:
            chains[folder.name] = chain

    return chains


def merge_metrics_csv(target_csv: Path, source_csvs: list[Path], dry_run: bool = False) -> int:
    """
    Merge multiple metrics.csv files, removing duplicates by step.

    Returns:
        Number of rows in merged CSV
    """
    all_dfs = []

    # Read target CSV first (has priority for duplicates)
    if target_csv.exists():
        try:
            df = pd.read_csv(target_csv)
            all_dfs.append(df)
            print(f"    - Target CSV: {len(df)} rows")
        except Exception as e:
            print(f"    - Warning: Could not read {target_csv}: {e}")

    # Read source CSVs
    for csv_path in source_csvs:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
                print(f"    - Source CSV: {csv_path.parent.parent.name} ({len(df)} rows)")
            except Exception as e:
                print(f"    - Warning: Could not read {csv_path}: {e}")

    if not all_dfs:
        print("    - No CSV files found")
        return 0

    # Concatenate and remove duplicates
    merged = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates by step
    # Keep 'last' = prefer resume folder data (cleaner, from checkpoint)
    # over original folder data (may have dirty data from interruption)
    if 'step' in merged.columns:
        merged = merged.drop_duplicates(subset=['step'], keep='last')
        merged = merged.sort_values('step').reset_index(drop=True)

    print(f"    - Merged: {len(merged)} rows (after deduplication)")

    if not dry_run:
        # Backup original
        if target_csv.exists():
            backup_path = target_csv.with_suffix('.csv.bak')
            shutil.copy2(target_csv, backup_path)

        # Write merged CSV
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(target_csv, index=False)
        print(f"    - Saved to: {target_csv}")

    return len(merged)


def merge_checkpoints(target_dir: Path, source_dirs: list[Path], dry_run: bool = False) -> int:
    """
    Copy checkpoints from source directories to target directory.

    Returns:
        Number of checkpoints copied
    """
    copied = 0

    for source_dir in source_dirs:
        if not source_dir.exists():
            continue

        for ckpt in source_dir.glob('*.pt'):
            target_ckpt = target_dir / ckpt.name

            if target_ckpt.exists():
                print(f"    - Skip (exists): {ckpt.name}")
                continue

            print(f"    - Copy: {ckpt.name} from {source_dir.parent.name}")

            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ckpt, target_ckpt)

            copied += 1

    return copied


def consolidate_group(original_name: str, folders: list[Path], dry_run: bool = False, delete_source: bool = False):
    """
    Consolidate a chain of experiment folders into one.

    Args:
        original_name: Name of the original experiment folder
        folders: List of folders in the chain [original, resume1, resume2, ...]
    """
    if len(folders) <= 1:
        return  # Nothing to merge

    target_folder = folders[0]  # Original folder (shortest name)
    source_folders = folders[1:]  # Resume folders (with appended timestamps)

    print(f"\n{'='*60}")
    print(f"RESUME CHAIN DETECTED ({len(folders)} folders)")
    print(f"{'='*60}")
    print(f"  [ORIGINAL] {target_folder.name}")
    for i, sf in enumerate(source_folders, 1):
        print(f"  [RESUME {i}] {sf.name}")

    # Merge metrics.csv
    print("\n  [Merging metrics.csv]")
    target_csv = target_folder / 'logs' / 'metrics.csv'
    source_csvs = [sf / 'logs' / 'metrics.csv' for sf in source_folders]
    merge_metrics_csv(target_csv, source_csvs, dry_run)

    # Merge checkpoints
    print("\n  [Merging checkpoints]")
    target_ckpt_dir = target_folder / 'checkpoints'
    source_ckpt_dirs = [sf / 'checkpoints' for sf in source_folders]
    copied = merge_checkpoints(target_ckpt_dir, source_ckpt_dirs, dry_run)
    print(f"    - Total copied: {copied} checkpoints")

    # Copy config.json if missing in target
    print("\n  [Checking config.json]")
    target_config = target_folder / 'config.json'
    if not target_config.exists():
        for sf in source_folders:
            source_config = sf / 'config.json'
            if source_config.exists():
                print(f"    - Copy config.json from {sf.name}")
                if not dry_run:
                    shutil.copy2(source_config, target_config)
                break
    else:
        print(f"    - config.json exists in target")

    # Delete source folders if requested
    if delete_source and not dry_run:
        print("\n  [Deleting source folders]")
        for sf in source_folders:
            print(f"    - Deleting: {sf.name}")
            shutil.rmtree(sf)
    elif delete_source and dry_run:
        print("\n  [Would delete source folders]")
        for sf in source_folders:
            print(f"    - Would delete: {sf.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate experiment folders split by checkpoint resume bug'
    )
    parser.add_argument(
        '--experiments-dir',
        type=Path,
        default=Path('./experiments'),
        help='Path to experiments directory (default: ./experiments)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--delete-source',
        action='store_true',
        help='Delete source folders after merging'
    )
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Only process experiments matching this pattern (e.g., "cond0-60")'
    )

    args = parser.parse_args()

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("=" * 60)

    experiments_dir = args.experiments_dir.resolve()
    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found: {experiments_dir}")
        return 1

    print(f"Scanning: {experiments_dir}")

    # Find experiment groups
    groups = find_experiment_groups(experiments_dir)

    # Filter if specified
    if args.filter:
        groups = {k: v for k, v in groups.items() if args.filter in k}

    # Find groups that need merging (more than one folder)
    merge_needed = {k: v for k, v in groups.items() if len(v) > 1}

    print(f"\nFound {len(groups)} experiment groups")
    print(f"Groups needing merge: {len(merge_needed)}")

    if not merge_needed:
        print("\nNo experiments need consolidation.")
        return 0

    # Show summary
    print("\n" + "=" * 60)
    print("RESUME CHAINS TO CONSOLIDATE:")
    print("=" * 60)
    for original_name, folders in merge_needed.items():
        print(f"\nChain ({len(folders)} folders):")
        print(f"  [ORIGINAL] {folders[0].name}")
        for i, f in enumerate(folders[1:], 1):
            print(f"  [RESUME {i}] {f.name}")

    # Consolidate each group
    for base_name, folders in merge_needed.items():
        consolidate_group(base_name, folders, args.dry_run, args.delete_source)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN COMPLETE - No changes were made")
    else:
        print("CONSOLIDATION COMPLETE")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
