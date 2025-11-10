#!/bin/bash
#SBATCH --job-name=cond_min_quick_cpu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --partition=cpu

# CPU debug version for minimal conditioning training
# NO unseen set: evaluation = all non-conditioning tokens
# Model: distilgpt2 (81.9M params)

echo "========================================="
echo "CPU DEBUG - MINIMAL CONDITIONING"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Quick test parameters for CPU debugging
EXP_NAME="conditional_minimal_cond_quick_cpu"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilgpt2"
BATCH_SIZE=2              # Smaller batch for CPU
GRAD_ACCUM=4              # Reduced gradient accumulation
NUM_SAMPLES=1000          # Very small for quick debugging
EVAL_SAMPLES=200          # Small eval set
LEARNING_RATE=5e-4
NUM_EPOCHS=1

echo "CPU Debug Configuration:"
echo "  Model: $MODEL_CONFIG"
echo "  Epochs: $NUM_EPOCHS"
echo "  Training Samples: $NUM_SAMPLES (small for debugging)"
echo "  Eval Samples: $EVAL_SAMPLES"
echo "  Conditioning: 0-10% (minimal)"
echo "  Evaluation: 100% of non-conditioning (no unseen)"
echo "  Device: CPU"
echo "  Expected Time: ~20-30 minutes (CPU is slower)"
echo "========================================="

# Run quick test on CPU
python3 ./train.py \
    --model_type conditional \
    --model_config $MODEL_CONFIG \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_samples $NUM_SAMPLES \
    --num_eval_samples $EVAL_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --warmup_steps 50 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --cond_pct_min 0.0 \
    --cond_pct_max 0.1 \
    --eval_pct_min 1.0 \
    --eval_pct_max 1.0 \
    --conditioning_sampling blockwise \
    --evaluation_sampling blockwise \
    --min_conditioning 0 \
    --min_evaluation 1 \
    --mode2_boundary_cond_pct_min 0.1 \
    --mode2_boundary_cond_pct_max 0.3 \
    --logging_steps 5 \
    --eval_steps 50 \
    --save_steps 200 \
    --do_eval \
    --max_eval_batches 10 \
    --output_dir $OUTPUT_DIR \
    --exp_name $EXP_NAME \
    --device cpu \
    --num_workers 2

EXIT_CODE=$?

echo "========================================="
echo "CPU Debug Completed"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "Duration: $((SECONDS / 60)) minutes"
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ CPU debug test passed!"
    echo ""
    echo "Configuration verified:"
    echo "  ✓ Minimal conditioning (0-10%) works"
    echo "  ✓ No unseen set (eval=100%) works"
    echo "  ✓ 5-mode evaluation runs successfully"
    echo "  ✓ CPU mode works correctly"
else
    echo "✗ CPU debug test failed!"
    echo "  Check logs: logs/slurm_${SLURM_JOB_ID}.out/err"
fi

exit $EXIT_CODE

