#!/bin/bash
# Cleanup incomplete experiment folders
#
# This script removes incomplete experiment runs when a completed version exists.
# A completed experiment has plots/ OR plots_individual/ subdirectory.
#
# Usage:
#   ./cleanup_incomplete_experiments.sh [OPTIONS] [EXPERIMENTS_DIR]
#
# Options:
#   --execute    Actually delete folders (default is dry-run)
#   --help       Show this help message
#
# Examples:
#   ./cleanup_incomplete_experiments.sh                    # Dry-run on ./experiments/
#   ./cleanup_incomplete_experiments.sh --execute          # Execute on ./experiments/
#   ./cleanup_incomplete_experiments.sh /path/to/experiments --execute

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DRY_RUN=true
EXPERIMENTS_DIR="./experiments"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --execute)
            DRY_RUN=false
            shift
            ;;
        --help|-h)
            head -20 "$0" | tail -18
            exit 0
            ;;
        *)
            if [[ -d "$1" ]]; then
                EXPERIMENTS_DIR="$1"
            else
                echo "Error: Directory '$1' does not exist"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate experiments directory
if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
    echo "Error: Experiments directory '$EXPERIMENTS_DIR' does not exist"
    exit 1
fi

echo "=== Experiment Cleanup Script ==="
if $DRY_RUN; then
    echo -e "Mode: ${YELLOW}DRY-RUN${NC} (use --execute to delete)"
else
    echo -e "Mode: ${RED}EXECUTE${NC} (will delete folders)"
fi
echo "Directory: $EXPERIMENTS_DIR"
echo ""

# Create temporary files for tracking
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Regex pattern to match timestamp suffix: _YYYYMMDD_HHMMSS
TIMESTAMP_PATTERN='_[0-9]{8}_[0-9]{6}$'

# Counters
total_prefixes=0
total_to_delete=0
total_to_keep=0
total_skipped=0

# Collect all experiment folders and extract prefixes
for folder in "$EXPERIMENTS_DIR"/*/; do
    folder_name=$(basename "$folder")

    # Skip if folder doesn't match timestamp pattern
    if ! echo "$folder_name" | grep -qE "$TIMESTAMP_PATTERN"; then
        continue
    fi

    # Extract prefix by removing timestamp suffix
    prefix=$(echo "$folder_name" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')

    # Check if this folder is completed (has plots/ OR plots_individual/)
    if [[ -d "${folder}plots" || -d "${folder}plots_individual" ]]; then
        completed="true"
    else
        completed="false"
    fi

    # Store folder info: prefix|folder_name|completed
    echo "${prefix}|${folder_name}|${completed}" >> "$TMP_DIR/folders.txt"
done

# Check if we found any folders
if [[ ! -f "$TMP_DIR/folders.txt" ]]; then
    echo "No experiment folders found with timestamp pattern."
    exit 0
fi

# Get unique prefixes
cut -d'|' -f1 "$TMP_DIR/folders.txt" | sort -u > "$TMP_DIR/prefixes.txt"

# Track folders to delete
> "$TMP_DIR/to_delete.txt"

# Process each prefix
while read -r prefix; do
    # Get all folders for this prefix
    grep "^${prefix}|" "$TMP_DIR/folders.txt" | sort -t'|' -k2 > "$TMP_DIR/current_prefix.txt"

    folder_count=$(wc -l < "$TMP_DIR/current_prefix.txt" | tr -d ' ')

    # Skip if only one folder
    if [[ $folder_count -eq 1 ]]; then
        continue
    fi

    ((total_prefixes++)) || true

    # Count completed folders
    completed_count=$(grep "|true$" "$TMP_DIR/current_prefix.txt" | wc -l | tr -d ' ')

    # If no completed folders, skip this prefix
    if [[ $completed_count -eq 0 ]]; then
        echo -e "[PREFIX: ${BLUE}$prefix${NC}]"
        echo -e "  ${YELLOW}SKIP${NC}: No completed experiment (0/${folder_count} have plots or plots_individual)"
        echo ""
        ((total_skipped++)) || true
        continue
    fi

    # Print prefix header
    echo -e "[PREFIX: ${BLUE}$prefix${NC}]"

    # Process each folder
    while IFS='|' read -r p folder_name completed; do
        if [[ "$completed" == "true" ]]; then
            echo -e "  ${GREEN}KEEP${NC}:   $folder_name (completed)"
            ((total_to_keep++)) || true
        else
            echo -e "  ${RED}DELETE${NC}: $folder_name (incomplete)"
            echo "$EXPERIMENTS_DIR/$folder_name" >> "$TMP_DIR/to_delete.txt"
            ((total_to_delete++)) || true
        fi
    done < "$TMP_DIR/current_prefix.txt"
    echo ""
done < "$TMP_DIR/prefixes.txt"

# Print summary
echo "=========================================="
echo "Summary:"
echo "  - Prefixes with duplicates: $total_prefixes"
echo "  - Folders to keep: $total_to_keep"
echo -e "  - Folders to delete: ${RED}$total_to_delete${NC}"
echo "  - Prefixes skipped (no completed): $total_skipped"
echo ""

# Execute deletion if not dry-run
if [[ $total_to_delete -eq 0 ]]; then
    echo "No folders to delete."
elif $DRY_RUN; then
    echo -e "${YELLOW}To execute deletion, run:${NC}"
    echo "  $0 --execute $EXPERIMENTS_DIR"
else
    echo -e "${RED}Deleting ${total_to_delete} folders...${NC}"
    while read -r folder; do
        echo "  Deleting: $folder"
        rm -rf "$folder"
    done < "$TMP_DIR/to_delete.txt"
    echo -e "${GREEN}Done!${NC}"
fi
