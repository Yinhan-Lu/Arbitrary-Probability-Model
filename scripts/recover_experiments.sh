#!/bin/bash
# ==========================================================================
# RECOVER EXPERIMENTS FROM LOCAL BACKUP
# ==========================================================================
# This script helps recover experiment data after the resume bug corrupted
# metrics.csv files on the cluster.
#
# What it does:
# 1. Lists all LOCAL experiment folders (these are the "valid" ones)
# 2. For each local folder with metrics.csv, syncs it to the cluster
# 3. Optionally deletes folders on cluster that don't exist locally
#
# Usage:
#   ./scripts/recover_experiments.sh --list              # List local experiments
#   ./scripts/recover_experiments.sh --sync-metrics      # Sync metrics.csv to cluster
#   ./scripts/recover_experiments.sh --delete-invalid    # Delete invalid folders on cluster
#   ./scripts/recover_experiments.sh --dry-run --sync-metrics  # Preview sync
#   ./scripts/recover_experiments.sh --dry-run --delete-invalid  # Preview delete
#
# IMPORTANT: Configure CLUSTER_HOST and CLUSTER_PATH before running!
# ==========================================================================

set -e

# ========== CONFIGURATION ==========
# EDIT THESE to match your cluster setup!
CLUSTER_HOST="mila"  # SSH alias or hostname (e.g., "user@cluster.example.com")
CLUSTER_PATH="/home/mila/l/luyinhan/scratch/Arbitrary Probability Model"
# ===================================

DRY_RUN=false
ACTION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --list)
            ACTION="list"
            shift
            ;;
        --sync-metrics)
            ACTION="sync"
            shift
            ;;
        --delete-invalid)
            ACTION="delete"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--list|--sync-metrics|--delete-invalid]"
            exit 1
            ;;
    esac
done

if [ -z "$ACTION" ]; then
    echo "Please specify an action: --list, --sync-metrics, or --delete-invalid"
    exit 1
fi

# Get list of local experiment folders (rope experiments only)
LOCAL_EXPERIMENTS=$(ls -d experiments/cond0-*rope* 2>/dev/null || true)

echo "========================================="
echo "EXPERIMENT RECOVERY TOOL"
echo "========================================="
echo "Cluster: $CLUSTER_HOST:$CLUSTER_PATH"
echo "Dry run: $DRY_RUN"
echo ""

# ========== LIST ACTION ==========
if [ "$ACTION" == "list" ]; then
    echo "--- Local Experiment Folders ---"
    echo "$LOCAL_EXPERIMENTS" | wc -l
    echo ""

    echo "--- Folders with metrics.csv ---"
    count=0
    for exp in $LOCAL_EXPERIMENTS; do
        if [ -f "$exp/logs/metrics.csv" ]; then
            size=$(wc -l < "$exp/logs/metrics.csv")
            echo "  $exp (${size} lines)"
            count=$((count + 1))
        fi
    done
    echo ""
    echo "Total: $count folders with metrics.csv"
    exit 0
fi

# ========== SYNC ACTION ==========
if [ "$ACTION" == "sync" ]; then
    echo "--- Syncing metrics.csv to cluster ---"

    for exp in $LOCAL_EXPERIMENTS; do
        if [ -f "$exp/logs/metrics.csv" ]; then
            local_file="$exp/logs/metrics.csv"
            remote_dir="$CLUSTER_PATH/$exp/logs/"

            if [ "$DRY_RUN" == "true" ]; then
                echo "[DRY-RUN] rsync $local_file -> $CLUSTER_HOST:$remote_dir"
            else
                echo "Syncing: $exp"
                # Create remote directory if it doesn't exist
                ssh "$CLUSTER_HOST" "mkdir -p '$remote_dir'" 2>/dev/null || true
                rsync -av "$local_file" "$CLUSTER_HOST:$remote_dir"
            fi
        fi
    done

    echo ""
    echo "Done!"
    exit 0
fi

# ========== DELETE ACTION ==========
if [ "$ACTION" == "delete" ]; then
    echo "--- Finding invalid folders on cluster ---"

    # Get list of local folder NAMES (basenames only)
    LOCAL_NAMES=$(for exp in $LOCAL_EXPERIMENTS; do basename "$exp"; done | sort)

    # Save to temp file
    echo "$LOCAL_NAMES" > /tmp/valid_experiments.txt

    # Get list of cluster folders
    echo "Fetching cluster experiment list..."
    CLUSTER_FOLDERS=$(ssh "$CLUSTER_HOST" "ls -d '$CLUSTER_PATH/experiments/cond0-'*rope* 2>/dev/null | xargs -n1 basename" || true)

    if [ -z "$CLUSTER_FOLDERS" ]; then
        echo "No experiments found on cluster or connection failed."
        exit 1
    fi

    echo ""
    echo "--- Invalid folders (exist on cluster but not locally) ---"

    to_delete=""
    for folder in $CLUSTER_FOLDERS; do
        if ! echo "$LOCAL_NAMES" | grep -q "^${folder}$"; then
            echo "  [INVALID] $folder"
            to_delete="$to_delete $folder"
        fi
    done

    if [ -z "$to_delete" ]; then
        echo "No invalid folders found."
        exit 0
    fi

    echo ""
    count=$(echo "$to_delete" | wc -w)
    echo "Total invalid folders: $count"

    if [ "$DRY_RUN" == "true" ]; then
        echo ""
        echo "[DRY-RUN] Would delete these folders on cluster."
        echo "Run without --dry-run to actually delete."
    else
        echo ""
        read -p "Are you sure you want to delete these $count folders? (yes/no): " confirm
        if [ "$confirm" == "yes" ]; then
            for folder in $to_delete; do
                echo "Deleting: $folder"
                ssh "$CLUSTER_HOST" "rm -rf '$CLUSTER_PATH/experiments/$folder'"
            done
            echo "Done!"
        else
            echo "Aborted."
        fi
    fi

    exit 0
fi
