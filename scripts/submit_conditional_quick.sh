#!/bin/bash
#SBATCH --job-name=cond_quick
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1

# Quick 10-minute GPU test for prefix conditioning
# Model: small (38.6M params)
# Training time: ~5-8 minutes

echo "========================================="
echo "QUICK CONDITIONAL TRAINING TEST"
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
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "========================================="

# Training parameters
EXP_NAME="conditional_quick_test"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="small"
BATCH_SIZE=32
GRAD_ACCUM=2
NUM_SAMPLES=20000
LEARNING_RATE=5e-4

echo "Training Configuration:"
echo "  Model: $MODEL_CONFIG (38.6M params)"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Training Samples: $NUM_SAMPLES"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Expected Time: ~5-8 minutes"
echo "========================================="

# Run training
python3 train.py \
    --model_type conditional \
    --model_config $MODEL_CONFIG \
    --num_epochs 1 \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_samples $NUM_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --conditioning_sampling blockwise \
    --evaluation_sampling random \
    --cond_pct_min 0.2 \
    --cond_pct_max 0.4 \
    --eval_pct_min 0.2 \
    --eval_pct_max 0.4 \
    --logging_steps 50 \
    --save_steps 1000 \
    --output_dir $OUTPUT_DIR \
    --exp_name $EXP_NAME \
    --device cuda \
    --num_workers 4

EXIT_CODE=$?

echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Duration: $SECONDS seconds"
echo "========================================="

# Print GPU memory usage at end
echo "Final GPU Memory Usage:"
nvidia-smi
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results: $OUTPUT_DIR/$EXP_NAME*"
else
    echo "Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
