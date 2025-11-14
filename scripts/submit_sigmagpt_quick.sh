#!/bin/bash
#SBATCH --job-name=sigmagpt_quick
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1

# Sigma GPT Quick Test - Fair Mode
#
# Quick training run to verify:
# - Training pipeline works correctly
# - No crashes or errors
# - GPU memory is sufficient
# - Data loading works
#
# This should complete in ~30-60 minutes

echo "========================================="
echo "SIGMA GPT QUICK TEST"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo "========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Print environment info
echo "Environment Information:"
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo "========================================="

# Quick test configuration
EXP_NAME="sigmagpt_quick_test"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilgpt2"
MODE="fair"

# Small batch configuration for quick testing
BATCH_SIZE=4
GRAD_ACCUM=2
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))

# Small dataset for quick testing
NUM_SAMPLES=1000              # Only 1K samples
EVAL_SAMPLES=500
NUM_EPOCHS=1                  # Just 1 epoch

# Standard hyperparameters
LEARNING_RATE=2.5e-4
WARMUP_STEPS=100              # Short warmup
WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95
MAX_GRAD_NORM=1.0

# Conditioning configuration
MAX_COND_BLOCKS=2
MAX_EVAL_BLOCKS=1

# Frequent logging for debugging
LOGGING_STEPS=10
EVAL_STEPS=50
SAVE_STEPS=100
MAX_EVAL_BATCHES=10

echo "========================================="
echo "Quick Test Configuration:"
echo "========================================="
echo "Model: $MODEL_CONFIG"
echo "Mode: $MODE"
echo ""
echo "Quick Test Settings:"
echo "  Training Samples: $NUM_SAMPLES (1K)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $EFFECTIVE_BATCH_SIZE"
echo ""
echo "Expected completion time: ~30-60 minutes"
echo "========================================="

# Run quick test
python3 ./train_sigmagpt.py \
    --model_config $MODEL_CONFIG \
    --mode $MODE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 8 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_samples $NUM_SAMPLES \
    --num_eval_samples $EVAL_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --adam_beta1 $ADAM_BETA1 \
    --adam_beta2 $ADAM_BETA2 \
    --max_grad_norm $MAX_GRAD_NORM \
    --warmup_steps $WARMUP_STEPS \
    --conditioning_sampling blockwise \
    --evaluation_sampling blockwise \
    --max_cond_blocks $MAX_COND_BLOCKS \
    --max_eval_blocks $MAX_EVAL_BLOCKS \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --max_eval_batches $MAX_EVAL_BATCHES \
    --output_dir $OUTPUT_DIR \
    --exp_name $EXP_NAME \
    --device cuda \
    --num_workers 2 \
    --fp16

EXIT_CODE=$?

echo "========================================="
echo "Quick Test Completed"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "Duration: $((SECONDS / 60)) minutes"
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Quick test passed!"
    echo ""
    echo "The training pipeline is working correctly."
    echo "You can now run full-scale training with:"
    echo "  sbatch scripts/submit_sigmagpt_fair.sh"
    echo "  sbatch scripts/submit_sigmagpt_full.sh"
    echo ""
    echo "Results: $OUTPUT_DIR/$EXP_NAME*"
else
    echo "✗ Quick test failed"
    echo "Check logs: logs/slurm_${SLURM_JOB_ID}.err"
    echo ""
    echo "Common issues:"
    echo "  1. GPU memory insufficient - reduce batch_size"
    echo "  2. Dataset loading issues - check internet connection"
    echo "  3. Missing dependencies - check requirements"
fi
echo "========================================="

exit $EXIT_CODE
