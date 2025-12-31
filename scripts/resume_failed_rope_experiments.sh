#!/bin/bash
# ==========================================================================
# RESUME FAILED ROPE EXPERIMENTS
# ==========================================================================
# This script checks all conditional and sigmagpt rope experiments (excluding
# thinking token experiments) and re-submits those that don't have
# plots_individual output (i.e., didn't complete successfully).
#
# Currently running experiments are skipped:
#   - c100_gpt2 (conditional)
#   - c80_gpt2_m (conditional)
#   - c60_gpt2_m (conditional)
#   - c100_gpt2_m (conditional)
#
# Usage:
#   ./scripts/resume_failed_rope_experiments.sh            # Submit all failed
#   ./scripts/resume_failed_rope_experiments.sh --dry-run  # Preview only
# ==========================================================================

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No jobs will be submitted ==="
    echo ""
fi

# Currently running experiments (skip these)
RUNNING_JOBS=(
    "cond0-100_max_block_rope_gpt2_conditional"
    "cond0-80_max_block_rope_gpt2_medium_conditional"
    "cond0-60_max_block_rope_gpt2_medium_conditional"
    "cond0-100_max_block_rope_gpt2_medium_conditional"
)

# Create logs directory
mkdir -p logs

echo "========================================="
echo "RESUME FAILED ROPE EXPERIMENTS"
echo "========================================="
echo ""

# Arrays to collect experiments to resume
declare -a CONDITIONAL_TO_RESUME=()
declare -a SIGMAGPT_TO_RESUME=()

# ==========================================================================
# Check conditional rope experiments
# ==========================================================================
echo "--- Checking Conditional RoPE Experiments ---"

for MODEL in distilgpt2 gpt2 gpt2_medium; do
    for COND in 0.2 0.4 0.6 0.8 1.0; do
        COND_PCT=$(python3 -c "print(int($COND * 100))")

        # Pattern to match this experiment config
        PATTERN="cond0-${COND_PCT}_max_block_rope_${MODEL}_conditional"

        # Check if this is a running job
        IS_RUNNING=false
        for RUNNING in "${RUNNING_JOBS[@]}"; do
            if [[ "$PATTERN" == "$RUNNING"* ]]; then
                IS_RUNNING=true
                break
            fi
        done

        if [ "$IS_RUNNING" == "true" ]; then
            echo "  [RUNNING] $PATTERN - skipping"
            continue
        fi

        # Find matching experiment folders
        MATCHING_DIRS=$(ls -d ./experiments/${PATTERN}_* 2>/dev/null | tail -1 || true)

        if [ -z "$MATCHING_DIRS" ]; then
            echo "  [NO FOLDER] $PATTERN - will submit new"
            CONDITIONAL_TO_RESUME+=("${MODEL}:${COND}")
        elif [ -d "$MATCHING_DIRS/plots_individual" ]; then
            echo "  [COMPLETE] $PATTERN - has plots_individual"
        else
            echo "  [INCOMPLETE] $PATTERN - no plots_individual, will resume"
            CONDITIONAL_TO_RESUME+=("${MODEL}:${COND}")
        fi
    done
done

echo ""

# ==========================================================================
# Check sigmagpt rope experiments (without thinking tokens)
# ==========================================================================
echo "--- Checking SigmaGPT RoPE Experiments (no thinking) ---"

for MODEL in distilgpt2 gpt2 gpt2_medium; do
    for ORDERING in temporal random_scramble; do
        for COND in 0.2 0.4 0.6 0.8 1.0; do
            COND_PCT=$(python3 -c "print(int($COND * 100))")

            # Pattern for sigmagpt WITHOUT thinking tokens
            PATTERN="cond0-${COND_PCT}_max_block_rope_${MODEL}_sigmagpt_${ORDERING}"

            # Find matching folders that DON'T have "think" in the name
            MATCHING_DIRS=$(ls -d ./experiments/${PATTERN}_* 2>/dev/null | grep -v "think" | tail -1 || true)

            if [ -z "$MATCHING_DIRS" ]; then
                echo "  [NO FOLDER] $PATTERN - will submit new"
                SIGMAGPT_TO_RESUME+=("${MODEL}:${ORDERING}:${COND}")
            elif [ -d "$MATCHING_DIRS/plots_individual" ]; then
                echo "  [COMPLETE] $PATTERN - has plots_individual"
            else
                echo "  [INCOMPLETE] $PATTERN - no plots_individual, will resume"
                SIGMAGPT_TO_RESUME+=("${MODEL}:${ORDERING}:${COND}")
            fi
        done
    done
done

echo ""
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo "Conditional experiments to resume: ${#CONDITIONAL_TO_RESUME[@]}"
echo "SigmaGPT experiments to resume: ${#SIGMAGPT_TO_RESUME[@]}"
echo ""

# ==========================================================================
# Submit failed experiments
# ==========================================================================
SUBMITTED=0

if [ ${#CONDITIONAL_TO_RESUME[@]} -gt 0 ]; then
    echo "--- Submitting Conditional Experiments ---"
    for item in "${CONDITIONAL_TO_RESUME[@]}"; do
        IFS=':' read -r MODEL COND <<< "$item"
        echo "  Submitting: conditional $MODEL cond=$COND"
        if [ "$DRY_RUN" == "false" ]; then
            ./scripts/conditional_rope_control_scale_maxtoken.sh --model "$MODEL" --cond "$COND"
            SUBMITTED=$((SUBMITTED + 1))
            sleep 1
        fi
    done
    echo ""
fi

if [ ${#SIGMAGPT_TO_RESUME[@]} -gt 0 ]; then
    echo "--- Submitting SigmaGPT Experiments ---"
    for item in "${SIGMAGPT_TO_RESUME[@]}"; do
        IFS=':' read -r MODEL ORDERING COND <<< "$item"
        echo "  Submitting: sigmagpt $MODEL $ORDERING cond=$COND"
        if [ "$DRY_RUN" == "false" ]; then
            ./scripts/submit_sigmagpt_rope.sh --model "$MODEL" --ordering "$ORDERING" --cond "$COND"
            SUBMITTED=$((SUBMITTED + 1))
            sleep 1
        fi
    done
    echo ""
fi

echo "========================================="
if [ "$DRY_RUN" == "true" ]; then
    echo "DRY RUN COMPLETE"
    echo "Would submit: $((${#CONDITIONAL_TO_RESUME[@]} + ${#SIGMAGPT_TO_RESUME[@]})) experiments"
else
    echo "SUBMISSION COMPLETE"
    echo "Submitted: $SUBMITTED experiments"
fi
echo "========================================="
echo ""
echo "Monitor jobs: squeue -u \$USER"
echo "Cancel all:   scancel -u \$USER"
