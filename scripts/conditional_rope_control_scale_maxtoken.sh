#!/bin/bash
# ==========================================================================
# ROPE ABLATION STUDY: Model Size × Conditioning Percentage
# ==========================================================================
# Two-dimensional ablation study for conditional model with RoPE:
#   1. Model size: distilgpt2 (82M), gpt2 (117M), gpt2_medium (180M)
#   2. Conditioning %: 0-20%, 0-40%, 0-60%, 0-80%, 0-100%
#
# Key settings:
#   - position_encoding_type = "rope"
#   - min_conditioning = 0 (always)
#   - max_cond_blocks = max_n_cond (blocks = cond tokens)
#   - eval_pct = 1.0 (all non-cond = eval)
#
# Total: 3 sizes × 5 cond_pct = 15 experiments
#
# Usage:
#   ./scripts/submit_rope_ablation.sh              # Submit all experiments
#   ./scripts/submit_rope_ablation.sh --dry-run    # Show what would be submitted
#   ./scripts/submit_rope_ablation.sh --model gpt2 # Submit only gpt2 model
#   ./scripts/submit_rope_ablation.sh --cond 0.4   # Submit only 0-40% cond
# ==========================================================================

set -e

# Parse arguments
DRY_RUN=false
FILTER_MODEL=""
FILTER_COND=""

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --model)
            shift
            FILTER_MODEL="$1"
            ;;
        --model=*)
            FILTER_MODEL="${arg#*=}"
            ;;
        --cond)
            shift
            FILTER_COND="$1"
            ;;
        --cond=*)
            FILTER_COND="${arg#*=}"
            ;;
    esac
    shift 2>/dev/null || true
done

if [ "$DRY_RUN" == "true" ]; then
    echo "=== DRY RUN MODE - No jobs will be submitted ==="
    echo ""
fi

# Create logs directory
mkdir -p logs

# ==========================================================================
# EXPERIMENT CONFIGURATION
# ==========================================================================

# Model sizes: distilgpt2 (82M), gpt2 (117M), gpt2_medium (180M)
declare -a MODEL_CONFIGS=("distilgpt2" "gpt2" "gpt2_medium")

# Conditioning percentages: 0-20%, 0-40%, 0-60%, 0-80%, 0-100%
# Format: "cond_max mode2_min mode2_max label"
# mode2 boundary is set to ~75% of cond_max for reasonable evaluation
declare -a COND_CONFIGS=(
    "0.2 0.05 0.15 cond0_20"
    "0.4 0.10 0.30 cond0_40"
    "0.6 0.15 0.45 cond0_60"
    "0.8 0.20 0.60 cond0_80"
    "1.0 0.25 0.75 cond0_100"
)

# Common training parameters
BATCH_SIZE=8
GRAD_ACCUM=16
NUM_SAMPLES=-1          # Use full dataset (WikiText-103: ~100K sequences)
EVAL_SAMPLES=-1         # Use full eval set
LEARNING_RATE="5e-4"
NUM_EPOCHS=20
WEIGHT_DECAY=0.1
WARMUP_STEPS=2000

echo "========================================="
echo "ROPE ABLATION STUDY"
echo "========================================="
echo "Model sizes: ${MODEL_CONFIGS[*]}"
echo "Conditioning configs: ${#COND_CONFIGS[@]}"
echo "Total experiments: $((${#MODEL_CONFIGS[@]} * ${#COND_CONFIGS[@]}))"
if [ -n "$FILTER_MODEL" ]; then
    echo "Filter: model=$FILTER_MODEL"
fi
if [ -n "$FILTER_COND" ]; then
    echo "Filter: cond=$FILTER_COND"
fi
echo ""

# Counter for submitted jobs
SUBMITTED=0
SKIPPED=0

# ==========================================================================
# GENERATE AND SUBMIT JOBS
# ==========================================================================

for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
    # Apply model filter if specified
    if [ -n "$FILTER_MODEL" ] && [ "$MODEL_CONFIG" != "$FILTER_MODEL" ]; then
        continue
    fi

    for config in "${COND_CONFIGS[@]}"; do
        read -r COND_MAX MODE2_MIN MODE2_MAX LABEL <<< "$config"

        # Apply cond filter if specified
        if [ -n "$FILTER_COND" ] && [ "$COND_MAX" != "$FILTER_COND" ]; then
            continue
        fi

        # Calculate max_cond_blocks = ceil(1023 * cond_max)
        # 1023 = body tokens (1024 positions - 1 BOS)
        MAX_COND_BLOCKS=$(python3 -c "import math; print(math.ceil(1023 * $COND_MAX))")

        # Generate timestamp
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)

        # Convert COND_MAX to percentage (0.2 -> 20, 1.0 -> 100)
        COND_PCT=$(python3 -c "print(int($COND_MAX * 100))")

        # Construct experiment name: cond0-{pct}_max_block_rope_{scale}_conditional_{timestamp}
        EXP_NAME="cond0-${COND_PCT}_max_block_rope_${MODEL_CONFIG}_conditional_${TIMESTAMP}"

        # Construct job name (max 15 chars for SLURM)
        JOB_NAME="c${COND_PCT}_${MODEL_CONFIG:0:6}"

        echo "----------------------------------------"
        echo "Experiment: $EXP_NAME"
        echo "  Model: $MODEL_CONFIG"
        echo "  Cond: 0.0-${COND_MAX}"
        echo "  Max blocks: $MAX_COND_BLOCKS"
        echo "  Mode2: ${MODE2_MIN}-${MODE2_MAX}"

        # Create temporary job script
        SCRIPT_FILE="/tmp/submit_${EXP_NAME}.sh"

        # Determine GPU and memory based on model size
        if [ "$MODEL_CONFIG" == "gpt2_medium" ]; then
            GPU_TYPE="a100l:1"
            MEMORY="48G"
        else
            GPU_TYPE="a100l:1"
            MEMORY="32G"
        fi

        # Write SLURM script
        cat > "$SCRIPT_FILE" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:${GPU_TYPE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${MEMORY}
#SBATCH --ntasks=1

# ==========================================================================
# ROPE ABLATION EXPERIMENT
# Model: ${MODEL_CONFIG}
# Conditioning: 0.0-${COND_MAX} (max_blocks=${MAX_COND_BLOCKS})
# Position Encoding: RoPE
# ==========================================================================

echo "========================================="
echo "EXPERIMENT: ${EXP_NAME}"
echo "  Model: ${MODEL_CONFIG}"
echo "  Position Encoding: RoPE"
echo "  Cond %: 0.0-${COND_MAX}"
echo "  Max Cond Blocks: ${MAX_COND_BLOCKS}"
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

# Run training
python3 ./train.py \\
    --model_type conditional \\
    --model_config ${MODEL_CONFIG} \\
    --position_encoding_type rope \\
    --num_epochs ${NUM_EPOCHS} \\
    --batch_size ${BATCH_SIZE} \\
    --eval_batch_size 16 \\
    --gradient_accumulation_steps ${GRAD_ACCUM} \\
    --num_train_samples ${NUM_SAMPLES} \\
    --num_eval_samples ${EVAL_SAMPLES} \\
    --learning_rate ${LEARNING_RATE} \\
    --warmup_steps ${WARMUP_STEPS} \\
    --max_grad_norm 1.0 \\
    --weight_decay ${WEIGHT_DECAY} \\
    --adam_beta1 0.9 \\
    --adam_beta2 0.999 \\
    --adam_epsilon 1e-8 \\
    --cond_pct_min 0.0 \\
    --cond_pct_max ${COND_MAX} \\
    --eval_pct_min 1.0 \\
    --eval_pct_max 1.0 \\
    --conditioning_sampling blockwise \\
    --evaluation_sampling blockwise \\
    --min_conditioning 0 \\
    --min_evaluation 1 \\
    --max_cond_blocks ${MAX_COND_BLOCKS} \\
    --mode2_boundary_cond_pct_min ${MODE2_MIN} \\
    --mode2_boundary_cond_pct_max ${MODE2_MAX} \\
    --use_attention_mask_for_valid \\
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
done

echo ""
echo "========================================="
if [ "$DRY_RUN" == "true" ]; then
    echo "DRY RUN COMPLETE"
    echo "Would submit: experiments"
else
    echo "SUBMISSION COMPLETE"
    echo "Submitted: $SUBMITTED jobs"
fi
echo "========================================="
echo ""
echo "Monitor jobs:    squeue -u \$USER"
echo "Cancel all:      scancel -u \$USER"
echo "Job details:     scontrol show job <JOB_ID>"
echo ""
echo "After completion, compare results with:"
echo "  python utils/plot_comparison_metrics.py experiments/rope_ablation_*"
echo ""
