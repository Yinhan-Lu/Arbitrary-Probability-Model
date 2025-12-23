#!/bin/bash
# ==========================================================================
# TABLE 3: REAL UNSEEN RATIO ABLATION
# ==========================================================================
# Research Question: How does the ratio of Real Unseen tokens ([M] without
# loss) affect learning?
#
# Fixed Parameters:
#   - cond_pct: 30% (fixed)
#   - max_cond_blocks: 3
#
# Sweep Parameters:
#   - eval_pct: 100%, 75%, 50%, 25% (controls real unseen ratio)
#   - models: conditional, sigmagpt_temporal, sigmagpt_scramble
#   - seeds: 42
#
# Naming Format: {model}_c{cond}_e{eval}_b{blocks}_s{seed}
# Example: cond_c30_e75_b3_s42
#
# Total: 3 models × 4 eval_pct × 1 seed = 12 experiments
#
# Usage:
#   ./scripts/submit_ablation_unseen_ratio.sh           # Submit all
#   ./scripts/submit_ablation_unseen_ratio.sh --dry-run # Show what would run
# ==========================================================================

set -e

# Parse arguments
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
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
# CONFIGURATION
# ==========================================================================

# Fixed parameters
COND_PCT=0.3           # 30% conditioning
MAX_COND_BLOCKS=3      # 3 blocks
MODEL_CONFIG="distilgpt2"
BATCH_SIZE=8
GRAD_ACCUM=16
NUM_SAMPLES=1000000
EVAL_SAMPLES=10000
LEARNING_RATE="5e-4"
NUM_EPOCHS=20
WEIGHT_DECAY=0.1

# Sweep parameters
declare -a MODELS=("conditional" "sigmagpt" "sigmagpt")
declare -a MODEL_LABELS=("cond" "sgpt_t" "sgpt_s")
declare -a SIGMAGPT_MODES=("" "temporal" "scramble")  # "" for conditional

declare -a EVAL_PCTS=("1.0" "0.75" "0.50" "0.25")
declare -a EVAL_LABELS=("e100" "e75" "e50" "e25")

declare -a SEEDS=("42")

# Mode 2 boundary (fixed, based on 30% conditioning)
MODE2_MIN=0.10
MODE2_MAX=0.20

# ==========================================================================
# SUMMARY
# ==========================================================================

echo "========================================="
echo "TABLE 3: REAL UNSEEN RATIO ABLATION"
echo "========================================="
echo "Fixed: cond=${COND_PCT}, max_blocks=${MAX_COND_BLOCKS}"
echo "Sweep: eval_pct=${EVAL_PCTS[*]}"
echo "Models: ${#MODELS[@]} (conditional, sigmagpt_temporal, sigmagpt_scramble)"
echo "Seeds: ${SEEDS[*]}"
echo "Total experiments: $((${#MODELS[@]} * ${#EVAL_PCTS[@]} * ${#SEEDS[@]}))"
echo ""
echo "Naming format: {model}_c{cond}_e{eval}_b{blocks}_s{seed}"
echo "========================================="
echo ""

# Counter
SUBMITTED=0

# ==========================================================================
# GENERATE AND SUBMIT JOBS
# ==========================================================================

for m_idx in "${!MODELS[@]}"; do
    MODEL_TYPE="${MODELS[$m_idx]}"
    MODEL_LABEL="${MODEL_LABELS[$m_idx]}"
    SIGMAGPT_MODE="${SIGMAGPT_MODES[$m_idx]}"

    for e_idx in "${!EVAL_PCTS[@]}"; do
        EVAL_PCT="${EVAL_PCTS[$e_idx]}"
        EVAL_LABEL="${EVAL_LABELS[$e_idx]}"

        for SEED in "${SEEDS[@]}"; do
            # Construct experiment name with rich information
            # Format: {model}_c{cond}_e{eval}_b{blocks}_s{seed}
            COND_LABEL="c30"
            BLOCK_LABEL="b${MAX_COND_BLOCKS}"
            SEED_LABEL="s${SEED}"

            EXP_NAME="${MODEL_LABEL}_${COND_LABEL}_${EVAL_LABEL}_${BLOCK_LABEL}_${SEED_LABEL}"

            # SLURM job name (max 15 chars)
            JOB_NAME="${MODEL_LABEL}_${EVAL_LABEL}_${SEED_LABEL}"

            echo "----------------------------------------"
            echo "Experiment: $EXP_NAME"
            if [ -n "$SIGMAGPT_MODE" ]; then
                echo "  Model: $MODEL_TYPE ($SIGMAGPT_MODE)"
            else
                echo "  Model: $MODEL_TYPE"
            fi
            echo "  Cond: ${COND_PCT} (30%)"
            echo "  Eval: ${EVAL_PCT} ($(echo "scale=0; (1-${EVAL_PCT})*100" | bc)% real unseen)"
            echo "  Blocks: ${MAX_COND_BLOCKS}"
            echo "  Seed: ${SEED}"

            # Create temporary job script
            SCRIPT_FILE="/tmp/submit_${EXP_NAME}.sh"

            # Build model-specific arguments
            if [ "$MODEL_TYPE" == "conditional" ]; then
                MODEL_ARGS="--model_type conditional"
            else
                MODEL_ARGS="--model_type sigmagpt --sigmagpt_order_type ${SIGMAGPT_MODE}"
            fi

            # Write SLURM script
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
# Model: ${MODEL_TYPE} ${SIGMAGPT_MODE}
# Cond: ${COND_PCT}, Eval: ${EVAL_PCT}, Blocks: ${MAX_COND_BLOCKS}, Seed: ${SEED}
# ==========================================================================

echo "========================================="
echo "EXPERIMENT: ${EXP_NAME}"
echo "  Model: ${MODEL_TYPE} ${SIGMAGPT_MODE}"
echo "  Cond %: ${COND_PCT}"
echo "  Eval %: ${EVAL_PCT}"
echo "  Real Unseen %: \$(echo \"scale=2; (1-${EVAL_PCT})*100\" | bc)%"
echo "  Max Cond Blocks: ${MAX_COND_BLOCKS}"
echo "  Seed: ${SEED}"
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
    ${MODEL_ARGS} \\
    --model_config ${MODEL_CONFIG} \\
    --seed ${SEED} \\
    --num_epochs ${NUM_EPOCHS} \\
    --batch_size ${BATCH_SIZE} \\
    --eval_batch_size 16 \\
    --gradient_accumulation_steps ${GRAD_ACCUM} \\
    --num_train_samples ${NUM_SAMPLES} \\
    --num_eval_samples ${EVAL_SAMPLES} \\
    --learning_rate ${LEARNING_RATE} \\
    --warmup_steps 2000 \\
    --max_grad_norm 1.0 \\
    --weight_decay ${WEIGHT_DECAY} \\
    --adam_beta1 0.9 \\
    --adam_beta2 0.999 \\
    --adam_epsilon 1e-8 \\
    --cond_pct_min ${COND_PCT} \\
    --cond_pct_max ${COND_PCT} \\
    --eval_pct_min ${EVAL_PCT} \\
    --eval_pct_max ${EVAL_PCT} \\
    --conditioning_sampling blockwise \\
    --evaluation_sampling blockwise \\
    --min_conditioning 1 \\
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

            # Small delay
            sleep 0.3
        done
    done
done

# ==========================================================================
# SUMMARY
# ==========================================================================

echo ""
echo "========================================="
if [ "$DRY_RUN" == "true" ]; then
    echo "DRY RUN COMPLETE"
    echo "Would submit: $((${#MODELS[@]} * ${#EVAL_PCTS[@]} * ${#SEEDS[@]})) jobs"
else
    echo "SUBMISSION COMPLETE"
    echo "Submitted: $SUBMITTED jobs"
fi
echo "========================================="
echo ""
echo "Experiment naming: {model}_c{cond}_e{eval}_b{blocks}_s{seed}"
echo "Examples:"
echo "  cond_c30_e100_b3_s42   = conditional, 30% cond, 100% eval, 3 blocks, seed 42"
echo "  sgpt_t_c30_e50_b3_s123 = sigmagpt temporal, 30% cond, 50% eval, 3 blocks, seed 123"
echo ""
echo "Monitor jobs:    squeue -u \$USER"
echo "Cancel all:      scancel -u \$USER"
echo "Job details:     scontrol show job <JOB_ID>"
echo ""
echo "After completion, compare results with:"
echo "  python utils/plot_comparison_metrics.py experiments/*_c30_*"
