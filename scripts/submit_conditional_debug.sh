#!/bin/bash
#SBATCH --job-name=cond_debug
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --ntasks=1

# Quick 5-minute debugging job for prefix conditioning
# Model: tiny (6.9M params)
# Training time: ~2-3 minutes

echo "========================================="
echo "DEBUG CONDITIONAL TRAINING"
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

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Create logs directory
mkdir -p logs

# Print environment info
echo "Environment Information:"
echo "Python version:"
python3 --version
echo "PyTorch version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo "========================================="

# Training parameters - minimal for quick testing
EXP_NAME="conditional_debug"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="tiny"
BATCH_SIZE=4
GRAD_ACCUM=1
NUM_SAMPLES=500
LEARNING_RATE=5e-4

echo "Debug Configuration:"
echo "  Model: $MODEL_CONFIG (6.9M params)"
echo "  Batch Size: $BATCH_SIZE"
echo "  Training Samples: $NUM_SAMPLES (minimal)"
echo "  Purpose: Quick verification"
echo "  Expected Time: ~2-3 minutes"
echo "========================================="

# Run minimal training for debugging
python3 train_conditional.py \
    --model_config $MODEL_CONFIG \
    --num_epochs 1 \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_samples $NUM_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --warmup_steps 100 \
    --conditioning_sampling blockwise \
    --evaluation_sampling random \
    --max_cond_blocks 2 \
    --max_eval_blocks 2 \
    --cond_pct_min 0.2 \
    --cond_pct_max 0.4 \
    --eval_pct_min 0.2 \
    --eval_pct_max 0.4 \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir $OUTPUT_DIR \
    --exp_name $EXP_NAME \
    --device cuda \
    --num_workers 2

EXIT_CODE=$?

echo "========================================="
echo "Debug job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Duration: $SECONDS seconds"
echo "========================================="

# Print GPU memory usage
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used --format=csv
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Debug run successful!"
    echo "  Your code is working correctly."
    echo "  Ready for larger experiments."
else
    echo "✗ Debug run failed with exit code $EXIT_CODE"
    echo "  Check logs for errors:"
    echo "  - STDOUT: logs/slurm_${SLURM_JOB_ID}.out"
    echo "  - STDERR: logs/slurm_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
