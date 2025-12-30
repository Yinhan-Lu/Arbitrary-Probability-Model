#!/bin/bash
# ==========================================================================
# SIGMAGPT ROPE + THINKING TOKENS ABLATION STUDY
# ==========================================================================
# Four-dimensional ablation study for SigmaGPT with RoPE and thinking tokens:
#   1. Model size: distilgpt2 (82M), gpt2 (117M), gpt2_medium (180M)
#   2. Conditioning %: 0-20%, 0-40%, 0-60%, 0-80%, 0-100%
#   3. Ordering mode: temporal, random_scramble
#   4. Thinking token mode: expectation, upper_bound
#
# Thinking tokens provide "extra computation" for fair comparison with
# conditioning tokens in the conditional model:
#   - expectation: n = 0.5 * cond_pct_max * 1024 tokens
#   - upper_bound: n = cond_pct_max * 1024 tokens
#
# Total: 3 sizes x 5 cond_pct x 2 orderings x 2 thinking = 60 experiments
#
# Usage:
#   ./scripts/submit_sigmagpt_rope_thinking.sh                       # Submit all
#   ./scripts/submit_sigmagpt_rope_thinking.sh --dry-run             # Show what would be submitted
#   ./scripts/submit_sigmagpt_rope_thinking.sh --model gpt2          # Filter by model
#   ./scripts/submit_sigmagpt_rope_thinking.sh --cond 0.4            # Filter by cond
#   ./scripts/submit_sigmagpt_rope_thinking.sh --thinking expectation # Filter by thinking mode
# ==========================================================================

set -e

# Parse arguments
DRY_RUN=false
FILTER_MODEL=""
FILTER_COND=""
FILTER_ORDERING=""
FILTER_THINKING=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --model=*)
            FILTER_MODEL="${1#*=}"
            shift
            ;;
        --cond)
            FILTER_COND="$2"
            shift 2
            ;;
        --cond=*)
            FILTER_COND="${1#*=}"
            shift
            ;;
        --ordering)
            FILTER_ORDERING="$2"
            shift 2
            ;;
        --ordering=*)
            FILTER_ORDERING="${1#*=}"
            shift
            ;;
        --thinking)
            FILTER_THINKING="$2"
            shift 2
            ;;
        --thinking=*)
            FILTER_THINKING="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
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

# Ordering modes: temporal (left-to-right), random_scramble (random permutation)
declare -a ORDERING_MODES=("temporal" "random_scramble")

# Thinking token modes: expectation (0.5*x), upper_bound (x)
declare -a THINKING_MODES=("expectation" "upper_bound")

# Conditioning percentages: 0-20%, 0-40%, 0-60%, 0-80%, 0-100%
# Format: "cond_max mode2_min mode2_max label"
declare -a COND_CONFIGS=(
    "0.2 0.00 0.20 cond0_20"
    "0.4 0.00 0.40 cond0_40"
    "0.6 0.00 0.60 cond0_60"
    "0.8 0.00 0.80 cond0_80"
    "1.0 0.00 1.00 cond0_100"
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
echo "SIGMAGPT ROPE + THINKING TOKENS ABLATION"
echo "========================================="
echo "Model sizes: ${MODEL_CONFIGS[*]}"
echo "Ordering modes: ${ORDERING_MODES[*]}"
echo "Thinking modes: ${THINKING_MODES[*]}"
echo "Conditioning configs: ${#COND_CONFIGS[@]}"
echo "Total experiments: $((${#MODEL_CONFIGS[@]} * ${#ORDERING_MODES[@]} * ${#THINKING_MODES[@]} * ${#COND_CONFIGS[@]}))"
if [ -n "$FILTER_MODEL" ]; then
    echo "Filter: model=$FILTER_MODEL"
fi
if [ -n "$FILTER_COND" ]; then
    echo "Filter: cond=$FILTER_COND"
fi
if [ -n "$FILTER_ORDERING" ]; then
    echo "Filter: ordering=$FILTER_ORDERING"
fi
if [ -n "$FILTER_THINKING" ]; then
    echo "Filter: thinking=$FILTER_THINKING"
fi
echo ""

# Counter for submitted jobs
SUBMITTED=0

# ==========================================================================
# GENERATE AND SUBMIT JOBS
# ==========================================================================

for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
    # Apply model filter if specified
    if [ -n "$FILTER_MODEL" ] && [ "$MODEL_CONFIG" != "$FILTER_MODEL" ]; then
        continue
    fi

    for ORDERING in "${ORDERING_MODES[@]}"; do
        # Apply ordering filter if specified
        if [ -n "$FILTER_ORDERING" ] && [ "$ORDERING" != "$FILTER_ORDERING" ]; then
            continue
        fi

        for THINKING in "${THINKING_MODES[@]}"; do
            # Apply thinking filter if specified
            if [ -n "$FILTER_THINKING" ] && [ "$THINKING" != "$FILTER_THINKING" ]; then
                continue
            fi

            for config in "${COND_CONFIGS[@]}"; do
                read -r COND_MAX MODE2_MIN MODE2_MAX LABEL <<< "$config"

                # Apply cond filter if specified
                if [ -n "$FILTER_COND" ] && [ "$COND_MAX" != "$FILTER_COND" ]; then
                    continue
                fi

                # Generate timestamp
                TIMESTAMP=$(date +%Y%m%d_%H%M%S)

                # Convert COND_MAX to percentage (0.2 -> 20, 1.0 -> 100)
                COND_PCT=$(python3 -c "print(int($COND_MAX * 100))")

                # Thinking mode short (exp/ub)
                THINK_SHORT="${THINKING:0:3}"

                # Construct experiment name:
                # cond0-{pct}_max_block_rope_{model}_sigmagpt_{ordering}_think_{mode}_{timestamp}
                EXP_NAME="cond0-${COND_PCT}_max_block_rope_${MODEL_CONFIG}_sigmagpt_${ORDERING}_think_${THINKING}_${TIMESTAMP}"

                # Construct job name (max 15 chars for SLURM)
                # sr=sigmagpt_rope, t/s=temporal/scramble, e/u=expectation/upper_bound
                ORDER_SHORT="${ORDERING:0:1}"
                JOB_NAME="sr${ORDER_SHORT}${THINK_SHORT:0:1}_c${COND_PCT}"

                echo "----------------------------------------"
                echo "Experiment: $EXP_NAME"
                echo "  Model: $MODEL_CONFIG"
                echo "  Ordering: $ORDERING"
                echo "  Thinking: $THINKING"
                echo "  Cond: 0.0-${COND_MAX}"

                # Create temporary job script
                SCRIPT_FILE="/tmp/submit_${EXP_NAME}.sh"

                # Determine GPU and memory based on model size and thinking tokens
                # Thinking tokens add to sequence length, may need more memory
                if [ "$MODEL_CONFIG" == "gpt2_medium" ]; then
                    GPU_TYPE="a100l:1"
                    MEMORY="64G"  # More memory for thinking tokens
                else
                    GPU_TYPE="a100l:1"
                    MEMORY="48G"
                fi

                # Write SLURM script
                cat > "$SCRIPT_FILE" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/cond0-${COND_PCT}_sigmagpt_${ORDERING}_think_${THINKING}_%j.out
#SBATCH --error=logs/cond0-${COND_PCT}_sigmagpt_${ORDERING}_think_${THINKING}_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:${GPU_TYPE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${MEMORY}
#SBATCH --ntasks=1

# ==========================================================================
# SIGMAGPT ROPE + THINKING TOKENS EXPERIMENT
# Model: ${MODEL_CONFIG}
# Ordering: ${ORDERING}
# Thinking: ${THINKING}
# Conditioning: 0.0-${COND_MAX}
# ==========================================================================

echo "========================================="
echo "EXPERIMENT: ${EXP_NAME}"
echo "  Model: ${MODEL_CONFIG}"
echo "  Position Encoding: RoPE (dual-axis)"
echo "  Ordering: ${ORDERING}"
echo "  Thinking Mode: ${THINKING}"
echo "  Cond %: 0.0-${COND_MAX}"
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

# =========================================================================
# AUTO-RESUME LOGIC
# =========================================================================
EXP_NAME="${EXP_NAME}"

# Look for existing experiment folder matching this config (ignoring timestamp)
EXP_PATTERN="cond0-${COND_PCT}_max_block_rope_${MODEL_CONFIG}_sigmagpt_${ORDERING}_think_${THINKING}_*"
EXISTING_EXP=\$(ls -dt ./experiments/\${EXP_PATTERN} 2>/dev/null | head -1)
RESUME_ARG=""

if [ -n "\$EXISTING_EXP" ] && [ -d "\$EXISTING_EXP/checkpoints" ]; then
    LATEST_CKPT=\$(ls -v "\$EXISTING_EXP/checkpoints/checkpoint_step_"*.pt 2>/dev/null | tail -1)

    if [ -n "\$LATEST_CKPT" ]; then
        echo "========================================="
        echo "RESUMING FROM CHECKPOINT"
        echo "  Experiment: \$EXISTING_EXP"
        echo "  Checkpoint: \$LATEST_CKPT"
        echo "========================================="
        RESUME_ARG="--resume_from \$LATEST_CKPT"
    fi
fi

# Run training with thinking tokens
python3 ./train.py \\
    --model_type sigmagpt \\
    --model_config ${MODEL_CONFIG} \\
    --position_encoding_type rope \\
    --sigmagpt_mode fair \\
    --ordering_mode ${ORDERING} \\
    --use_thinking_tokens \\
    --thinking_token_mode ${THINKING} \\
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
    --mode2_boundary_cond_pct_min ${MODE2_MIN} \\
    --mode2_boundary_cond_pct_max ${MODE2_MAX} \\
    --logging_steps 10 \\
    --eval_steps 100 \\
    --save_steps 1000 \\
    --early_stopping_patience 0 \\
    --do_eval \\
    --max_eval_batches 10 \\
    --output_dir ./experiments \\
    --exp_name \${EXP_NAME} \\
    --device cuda \\
    --num_workers 4 \\
    \$RESUME_ARG

EXIT_CODE=\$?

echo "========================================="
echo "Training Completed"
echo "Exit code: \$EXIT_CODE"
echo "Duration: \$SECONDS seconds (~\$((SECONDS / 60)) minutes)"
echo "========================================="

# Auto-generate visualization plots
if [ \$EXIT_CODE -eq 0 ]; then
    LATEST_EXP=\$(ls -dt ./experiments/\${EXP_PATTERN} 2>/dev/null | head -1)
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
    done
done

echo ""
echo "========================================="
if [ "$DRY_RUN" == "true" ]; then
    echo "DRY RUN COMPLETE"
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
echo "Thinking token count per config:"
echo "  0-20% (exp): 102 tokens,  0-20% (ub): 205 tokens"
echo "  0-40% (exp): 205 tokens,  0-40% (ub): 410 tokens"
echo "  0-60% (exp): 307 tokens,  0-60% (ub): 614 tokens"
echo "  0-80% (exp): 410 tokens,  0-80% (ub): 819 tokens"
echo "  0-100% (exp): 512 tokens, 0-100% (ub): 1024 tokens"
echo ""
