#!/bin/bash
# ==========================================================================
# EXPERIMENT FOLDER CLEANER
# ==========================================================================
# Deletes experiment folders containing specified keywords.
#
# Usage:
#   ./scripts/clean_experiments.sh "keyword1" "keyword2" ...
#   ./scripts/clean_experiments.sh --dry-run "keyword1" "keyword2" ...
#
# Examples:
#   # Delete all folders containing "think" in the name
#   ./scripts/clean_experiments.sh think
#
#   # Delete folders containing "think" OR "expectation"
#   ./scripts/clean_experiments.sh think expectation
#
#   # Dry run (show what would be deleted without actually deleting)
#   ./scripts/clean_experiments.sh --dry-run think
#
#   # Delete folders matching multiple patterns
#   ./scripts/clean_experiments.sh think_expectation upper_bound
# ==========================================================================

set -e

# Default experiment directory
EXPERIMENTS_DIR="./experiments"

# Parse arguments
DRY_RUN=false
KEYWORDS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --dir)
            EXPERIMENTS_DIR="$2"
            shift 2
            ;;
        --dir=*)
            EXPERIMENTS_DIR="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] KEYWORD1 [KEYWORD2 ...]"
            echo ""
            echo "Delete experiment folders containing specified keywords."
            echo ""
            echo "Options:"
            echo "  --dry-run, -n    Show what would be deleted without actually deleting"
            echo "  --dir DIR        Specify experiments directory (default: ./experiments)"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 think                      # Delete folders containing 'think'"
            echo "  $0 --dry-run think upper      # Dry run for 'think' or 'upper'"
            echo "  $0 think_expectation          # Delete folders with 'think_expectation'"
            exit 0
            ;;
        *)
            KEYWORDS+=("$1")
            shift
            ;;
    esac
done

# Check if keywords were provided
if [ ${#KEYWORDS[@]} -eq 0 ]; then
    echo "Error: No keywords specified."
    echo "Usage: $0 [--dry-run] KEYWORD1 [KEYWORD2 ...]"
    echo "Use --help for more information."
    exit 1
fi

# Check if experiments directory exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Error: Experiments directory not found: $EXPERIMENTS_DIR"
    exit 1
fi

echo "========================================="
echo "EXPERIMENT FOLDER CLEANER"
echo "========================================="
echo "Directory: $EXPERIMENTS_DIR"
echo "Keywords: ${KEYWORDS[*]}"
if [ "$DRY_RUN" == "true" ]; then
    echo "Mode: DRY RUN (no files will be deleted)"
else
    echo "Mode: DELETE"
fi
echo "========================================="
echo ""

# Find matching folders
TOTAL_SIZE=0
MATCHED_FOLDERS=()

for dir in "$EXPERIMENTS_DIR"/*/; do
    if [ ! -d "$dir" ]; then
        continue
    fi

    dirname=$(basename "$dir")

    # Check if any keyword matches
    for keyword in "${KEYWORDS[@]}"; do
        if [[ "$dirname" == *"$keyword"* ]]; then
            MATCHED_FOLDERS+=("$dir")
            # Calculate size
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "  [MATCH] $dirname ($size)"
            break
        fi
    done
done

echo ""
echo "========================================="
echo "Found ${#MATCHED_FOLDERS[@]} matching folder(s)"
echo "========================================="

if [ ${#MATCHED_FOLDERS[@]} -eq 0 ]; then
    echo "No folders match the specified keywords."
    exit 0
fi

# Confirm deletion (unless dry run)
if [ "$DRY_RUN" == "true" ]; then
    echo ""
    echo "[DRY RUN] Would delete ${#MATCHED_FOLDERS[@]} folder(s)"
    echo "Run without --dry-run to actually delete."
else
    echo ""
    read -p "Are you sure you want to delete ${#MATCHED_FOLDERS[@]} folder(s)? [y/N] " confirm

    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Cancelled."
        exit 0
    fi

    echo ""
    echo "Deleting..."

    DELETED=0
    for dir in "${MATCHED_FOLDERS[@]}"; do
        dirname=$(basename "$dir")
        echo "  Deleting: $dirname"
        rm -rf "$dir"
        DELETED=$((DELETED + 1))
    done

    echo ""
    echo "========================================="
    echo "Deleted $DELETED folder(s)"
    echo "========================================="
fi
