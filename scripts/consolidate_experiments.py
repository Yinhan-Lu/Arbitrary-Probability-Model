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


def extract_base_name_and_timestamps(folder_name: str) -> tuple[str, list[str]]:
    """
    Extract base experiment name and list of timestamps from folder name.

    Example:
        'cond0-60_max_block_rope_gpt2_conditional_20251227_070135_20251227_204239'
        -> ('cond0-60_max_block_rope_gpt2_conditional', ['20251227_070135', '20251227_204239'])
    """
    timestamps = TIMESTAMP_PATTERN.findall(folder_name)

    if not timestamps:
        return folder_name, []

    # Remove all timestamps to get base name
    base_name = folder_name
    for ts in timestamps:
        base_name = base_name.replace(f'_{ts}', '')

    return base_name, timestamps


def find_experiment_groups(experiments_dir: Path) -> dict[str, list[Path]]:
    """
    Group experiment folders by their base name.

    Returns:
        Dict mapping base_name -> list of folders (sorted by timestamp count)
    """
    groups = defaultdict(list)

    for folder in experiments_dir.iterdir():
        if not folder.is_dir():
            continue

        base_name, timestamps = extract_base_name_and_timestamps(folder.name)

        # Only group folders that have timestamps (ignore non-timestamped folders)
        if timestamps:
            groups[base_name].append((folder, len(timestamps), timestamps[0] if timestamps else ''))

    # Sort each group by: 1) number of timestamps (fewer = earlier), 2) first timestamp
    result = {}
    for base_name, folders in groups.items():
        sorted_folders = sorted(folders, key=lambda x: (x[1], x[2]))
        result[base_name] = [f[0] for f in sorted_folders]

    return result


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

    # Remove duplicates by step (keep first occurrence = earlier data)
    if 'step' in merged.columns:
        merged = merged.drop_duplicates(subset=['step'], keep='first')
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


def consolidate_group(base_name: str, folders: list[Path], dry_run: bool = False, delete_source: bool = False):
    """
    Consolidate a group of experiment folders into one.
    """
    if len(folders) <= 1:
        return  # Nothing to merge

    target_folder = folders[0]  # Folder with fewest timestamps
    source_folders = folders[1:]

    print(f"\n{'='*60}")
    print(f"Consolidating: {base_name}")
    print(f"  Target: {target_folder.name}")
    for sf in source_folders:
        print(f"  Source: {sf.name}")
    print(f"{'='*60}")

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
    print("EXPERIMENTS TO CONSOLIDATE:")
    print("=" * 60)
    for base_name, folders in merge_needed.items():
        print(f"\n{base_name}:")
        for i, f in enumerate(folders):
            marker = "[TARGET]" if i == 0 else "[SOURCE]"
            print(f"  {marker} {f.name}")

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
