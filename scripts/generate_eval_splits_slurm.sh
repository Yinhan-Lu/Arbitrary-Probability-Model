#!/bin/bash
#SBATCH --job-name=gen_splits
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1

# ==========================================================================
# Generate Deterministic Evaluation Splits
# ==========================================================================
# This script generates fixed evaluation splits for fair model comparison.
# Run this ONCE before submitting comparison experiments.
# CPU-only task, no GPU needed.

echo "========================================="
echo "Generating Deterministic Evaluation Splits"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
mkdir -p utils/evaluation_splits

# Distribution parameters - MUST match training config (submit_conditional_moderate_cond.sh)
# Conditioning: 0-40% of sequence
# Evaluation: 100% of non-conditioning tokens (no unseen set)
python3 scripts/generate_eval_splits.py \
    --dataset wikitext \
    --dataset_config wikitext-103-raw-v1 \
    --split validation \
    --output utils/evaluation_splits/wikitext103_valid_seed42.pt \
    --seed 42 \
    --cond_pct_min 0.0 \
    --cond_pct_max 0.4 \
    --eval_pct_min 1.0 \
    --eval_pct_max 1.0 \
    --max_cond_blocks 3 \

EXIT_CODE=$?

echo "========================================="
echo "Generation Completed"
echo "Exit code: $EXIT_CODE"
echo "Duration: $SECONDS seconds"
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Splits file created: utils/evaluation_splits/wikitext103_valid_seed42.pt"
    ls -lh utils/evaluation_splits/
fi

exit $EXIT_CODE
