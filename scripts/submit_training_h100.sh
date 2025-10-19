#!/bin/bash
#SBATCH --job-name=distilgpt2_h100
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1

# DistilGPT-2 Training on H100 GPU (Best Available)
# Optimized for maximum performance with H100's capabilities

echo "========================================="
echo "DISTILGPT-2 TRAINING ON H100 GPU"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "========================================="

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo "========================================="

# Set environment variables for H100 optimization
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# Enable TF32 for H100 (faster than FP32, almost as accurate)
export NVIDIA_TF32_OVERRIDE=1

# Set Python path
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Create logs directory
mkdir -p logs

# Print environment info
echo "Environment Information:"
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "========================================="

# Training parameters - optimized for H100
EXP_NAME="distilgpt2_wikipedia_h100"
OUTPUT_DIR="./experiments"
BATCH_SIZE=16  # Larger batch size for H100's 80GB memory
GRAD_ACCUM_STEPS=8  # Effective batch size = 16 * 8 = 128
LEARNING_RATE=5e-4
NUM_EPOCHS=3
WARMUP_STEPS=2000
LOGGING_STEPS=10
EVAL_STEPS=500
SAVE_STEPS=1000

echo "Training Configuration (H100 Optimized):"
echo "  Experiment Name: $EXP_NAME"
echo "  Batch Size per Device: $BATCH_SIZE"
echo "  Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Number of Epochs: $NUM_EPOCHS"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "  Mixed Precision: FP16 Enabled"
echo "  Data Loading Workers: 8"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Run training with H100-optimized settings
python train_distilgpt2.py \
    --model_config distilgpt2 \
    --dataset_name wikipedia \
    --dataset_config "20220301.en" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 32 \
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
echo "Training Completed"
echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Total duration: $SECONDS seconds"
echo "========================================="

# Print final GPU stats
echo "Final GPU Memory Usage:"
nvidia-smi
echo "========================================="

# Print experiment directory
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Experiment directory: $OUTPUT_DIR/$EXP_NAME*"
    echo "Checkpoints: $OUTPUT_DIR/$EXP_NAME*/checkpoints/"
    echo "Logs: $OUTPUT_DIR/$EXP_NAME*/logs/"
    echo "TensorBoard: tensorboard --logdir=$OUTPUT_DIR/$EXP_NAME*/logs/tensorboard"
else
    echo "Training failed with exit code $EXIT_CODE"
    echo "Check error log: logs/slurm_${SLURM_JOB_ID}.err"
fi
echo "========================================="

exit $EXIT_CODE
