#!/bin/bash
#SBATCH --job-name=distilgpt2_train
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1

# DistilGPT-2 Training Job on Wikipedia Dataset
# This script submits a training job to SLURM with optimal resource allocation

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

# Load modules if needed
# module load python/3.9 cuda/11.8

# Activate virtual environment if using one
# source ~/venv/bin/activate

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# Set Python path
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Change to submission directory
cd "$SLURM_SUBMIT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print Python and CUDA versions
echo "Python version:"
python3 --version
echo "PyTorch version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo "========================================="

# Install missing dependencies if needed
echo "Checking and installing dependencies..."
python3 -m pip install --user --quiet filelock transformers datasets matplotlib tensorboard || echo "Warning: Some dependencies may not be installed"
echo "Dependencies check complete"
echo "========================================="

# Training parameters
EXP_NAME="distilgpt2_wikipedia_full"
OUTPUT_DIR="./experiments"
BATCH_SIZE=8
GRAD_ACCUM_STEPS=16  # Effective batch size = 8 * 16 = 128
LEARNING_RATE=5e-4
NUM_EPOCHS=3
WARMUP_STEPS=2000
LOGGING_STEPS=10
EVAL_STEPS=500
SAVE_STEPS=1000

echo "Training Configuration:"
echo "  Experiment Name: $EXP_NAME"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM_STEPS"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Number of Epochs: $NUM_EPOCHS"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "========================================="

# Run training
python3 train.py \
    --model_type baseline \
    --model_config distilgpt2 \
    --dataset_name wikipedia \
    --dataset_config "20220301.en" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --warmup_steps $WARMUP_STEPS \
    --warmup_start_factor 0.1 \
    --min_lr_ratio 0.1 \
    --fp16 \
    --do_eval \
    --num_eval_samples 10000 \
    --max_eval_batches 100 \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --output_dir $OUTPUT_DIR \
    --exp_name $EXP_NAME \
    --device cuda \
    --num_workers 4

EXIT_CODE=$?

echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

# Print GPU memory usage at end
echo "Final GPU Memory Usage:"
nvidia-smi
echo "========================================="

exit $EXIT_CODE
