#!/bin/bash
# ==========================================================================
# CONDITIONING PERCENTAGE SWEEP: SIGMAGPT TEMPORAL
# ==========================================================================
# Systematic study of conditioning percentage effects on SigmaGPT with
# temporal ordering (Eric's Method 1 - maintains left-to-right order).
#
# Configurations:
#   Main experiments (performance curve):
#     - 0-20%, 0-40%, 0-60%, 0-80%
#   Ablation experiments (narrow vs wide):
#     - 40-60%, 60-80%
#
# Total: 6 experiments
#
# Usage:
#   ./scripts/submit_sweep_sigmagpt_temporal.sh           # Submit all experiments
#   ./scripts/submit_sweep_sigmagpt_temporal.sh --dry-run # Show what would be submitted
#   ./scripts/submit_sweep_sigmagpt_temporal.sh --skip-baseline # Skip 0-40% (already done)
# ==========================================================================

set -e

# Parse arguments
DRY_RUN=false
SKIP_BASELINE=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            ;;
    esac
done

if [ "$DRY_RUN" == "true" ]; then
    echo "=== DRY RUN MODE - No jobs will be submitted ==="
    echo ""
fi

# Create logs directory
mkdir -p logs

# Define conditioning ranges and corresponding Mode 2 settings
# Format: "cond_min cond_max mode2_min mode2_max label"
declare -a CONFIGS=(
    "0.0 0.2 0.05 0.15 cond0_20"
    "0.0 0.4 0.10 0.30 cond0_40"
    "0.0 0.6 0.15 0.45 cond0_60"
    "0.0 0.8 0.20 0.60 cond0_80"
    "0.4 0.6 0.40 0.50 cond40_60"
    "0.6 0.8 0.60 0.70 cond60_80"
)

# Common training parameters (matching converge experiments)
MODEL_CONFIG="distilgpt2"
BATCH_SIZE=8
GRAD_ACCUM=16
NUM_SAMPLES=1000000
EVAL_SAMPLES=10000
LEARNING_RATE="5e-4"
NUM_EPOCHS=20
WEIGHT_DECAY=0.1

echo "========================================="
echo "CONDITIONING PERCENTAGE SWEEP: SIGMAGPT TEMPORAL"
echo "========================================="
echo "Configurations: ${#CONFIGS[@]} conditioning ranges"
echo "Model: SigmaGPT (temporal ordering - Eric's Method 1)"
if [ "$SKIP_BASELINE" == "true" ]; then
    echo "Skipping: 0-40% baseline (already done)"
fi
echo ""

# Counter for submitted jobs
SUBMITTED=0
SKIPPED=0

# Generate and submit jobs
for config in "${CONFIGS[@]}"; do
    read -r COND_MIN COND_MAX MODE2_MIN MODE2_MAX LABEL <<< "$config"

    # Skip baseline if requested
    if [ "$SKIP_BASELINE" == "true" ] && [ "$LABEL" == "cond0_40" ]; then
        echo "----------------------------------------"
        echo "SKIPPING: $LABEL (baseline already done)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Construct experiment name
    EXP_NAME="sweep_${LABEL}_sgpt_temp"

    # Construct job name (max 15 chars for SLURM)
    JOB_NAME="${LABEL:0:10}_temp"

    echo "----------------------------------------"
    echo "Experiment: $EXP_NAME"
    echo "  Model: sigmagpt"
    echo "  Ordering: temporal"
    echo "  Cond: ${COND_MIN}-${COND_MAX}"
    echo "  Mode2: ${MODE2_MIN}-${MODE2_MAX}"

    # Create temporary job script
    SCRIPT_FILE="/tmp/submit_${EXP_NAME}.sh"

    # Write SLURM header and training command
    cat > "$SCRIPT_FILE" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1

# ==========================================================================
# Experiment: ${EXP_NAME}
# Model: sigmagpt, Ordering: temporal (Eric's Method 1)
# Conditioning: ${COND_MIN}-${COND_MAX}, Mode2: ${MODE2_MIN}-${MODE2_MAX}
# ==========================================================================

echo "========================================="
echo "EXPERIMENT: ${EXP_NAME}"
echo "  Model: sigmagpt"
echo "  Ordering: temporal"
echo "  Cond %: ${COND_MIN}-${COND_MAX}"
echo "  Mode2 %: ${MODE2_MIN}-${MODE2_MAX}"
echo "========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start Time: \$(date)"
echo "========================================="

nvidia-smi
echo "========================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="\${SLURM_SUBMIT_DIR}:\${PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=1

cd "\$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load modules (Mila cluster)
module load cuda/12.1.1/cudnn/8.9
source /cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/etc/profile.d/conda.sh
conda activate arbprob

echo "Python: \$(which python3)"
echo "PyTorch: \$(python3 -c 'import torch; print(torch.__version__)')"
echo "========================================="

python3 ./train.py \\
    --model_type sigmagpt \\
    --model_config ${MODEL_CONFIG} \\
    --dataset_name wikitext \\
    --dataset_config wikitext-103-raw-v1 \\
    --num_train_samples ${NUM_SAMPLES} \\
    --num_eval_samples ${EVAL_SAMPLES} \\
    --num_epochs ${NUM_EPOCHS} \\
    --batch_size ${BATCH_SIZE} \\
    --eval_batch_size 16 \\
    --gradient_accumulation_steps ${GRAD_ACCUM} \\
    --learning_rate ${LEARNING_RATE} \\
    --weight_decay ${WEIGHT_DECAY} \\
    --warmup_steps 2000 \\
    --max_grad_norm 1.0 \\
    --adam_beta1 0.9 \\
    --adam_beta2 0.999 \\
    --sigmagpt_mode fair \\
    --ordering_mode temporal \\
    --cond_pct_min ${COND_MIN} \\
    --cond_pct_max ${COND_MAX} \\
    --eval_pct_min 1.0 \\
    --eval_pct_max 1.0 \\
    --conditioning_sampling blockwise \\
    --evaluation_sampling blockwise \\
    --max_cond_blocks 3 \\
    --mode2_boundary_cond_pct_min ${MODE2_MIN} \\
    --mode2_boundary_cond_pct_max ${MODE2_MAX} \\
    --logging_steps 10 \\
    --eval_steps 100 \\
    --save_steps 1000 \\
    --early_stopping_patience 0 \\
    --do_eval \\
    --max_eval_batches 10 \\
    --output_dir ./experiments \\
    --exp_name ${EXP_NAME} \\
    --device cuda \\
    --num_workers 4

EXIT_CODE=\$?

echo "========================================="
echo "Training Completed"
echo "Exit code: \$EXIT_CODE"
echo "Duration: \$SECONDS seconds (~\$((SECONDS / 60)) minutes)"
echo "========================================="

# Auto-generate visualization plots
if [ \$EXIT_CODE -eq 0 ]; then
    LATEST_EXP=\$(ls -dt ./experiments/${EXP_NAME}_* 2>/dev/null | head -1)
    if [ -n "\$LATEST_EXP" ] && [ -d "\$LATEST_EXP" ]; then
        echo "========================================="
        echo "AUTO-GENERATING VISUALIZATION PLOTS"
        echo "========================================="

        python3 utils/plot_individual_metrics.py "\$LATEST_EXP" 2>/dev/null || true
        python3 utils/quickstart_visualization.py "\$LATEST_EXP" 2>/dev/null || true

        echo "Results: \$LATEST_EXP"
    fi
fi

exit \$EXIT_CODE
EOF

    # Submit or show
    if [ "$DRY_RUN" == "true" ]; then
        echo "  [DRY-RUN] Would submit: $SCRIPT_FILE"
    else
        JOB_ID=$(sbatch "$SCRIPT_FILE" | awk '{print $4}')
        echo "  Submitted: Job ID $JOB_ID"
        SUBMITTED=$((SUBMITTED + 1))
    fi

    # Small delay to avoid overwhelming scheduler
    sleep 0.5
done

echo ""
echo "========================================="
if [ "$DRY_RUN" == "true" ]; then
    echo "DRY RUN COMPLETE"
    echo "Would submit: $((${#CONFIGS[@]} - SKIPPED)) jobs"
else
    echo "SUBMISSION COMPLETE"
    echo "Submitted: $SUBMITTED jobs"
    if [ "$SKIPPED" -gt 0 ]; then
        echo "Skipped: $SKIPPED jobs (baseline)"
    fi
fi
echo "========================================="
echo ""
echo "Monitor jobs:    squeue -u \$USER"
echo "Cancel all:      scancel -u \$USER"
echo "Job details:     scontrol show job <JOB_ID>"
echo ""
echo "After completion, compare results with:"
echo "  python utils/plot_comparison_metrics.py experiments/sweep_*_sgpt_temp_*"
