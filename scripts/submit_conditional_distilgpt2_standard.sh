#!/bin/bash
#SBATCH --job-name=distilgpt2_std
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1

# Standard DistilGPT-2 training with standard hyperparameters
# Model: distilgpt2 (81.9M params)
# Hyperparameters match standard GPT-2/DistilGPT-2 training

echo "========================================="
echo "STANDARD DISTILGPT2 TRAINING"
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

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Enable TF32 for faster training on A100
export NVIDIA_TF32_OVERRIDE=1

# Change to submission directory
cd "$SLURM_SUBMIT_DIR"

# Create logs directory
mkdir -p logs

# Print environment info
echo "Environment Information:"
echo "Python version:"
python3 --version
echo "PyTorch version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "========================================="

# Standard DistilGPT-2 hyperparameters
EXP_NAME="distilgpt2_standard_hyperparams"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilgpt2"

# Standard GPT-2 training hyperparameters
BATCH_SIZE=8                    # Per-device batch size
GRAD_ACCUM=64                   # Gradient accumulation steps
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))  # = 512 (standard)

NUM_SAMPLES=500000              # Training samples
EVAL_SAMPLES=10000              # Evaluation samples
LEARNING_RATE=2.5e-4           # Standard GPT-2 learning rate
NUM_EPOCHS=3                    # Number of epochs

# Standard conditioning range (20-40%)
COND_PCT_MIN=0.2
COND_PCT_MAX=0.4
EVAL_PCT_MIN=0.2
EVAL_PCT_MAX=0.4

echo "========================================="
echo "Training Configuration:"
echo "========================================="
echo "Model: $MODEL_CONFIG (81.9M parameters)"
echo ""
echo "Standard DistilGPT-2 Hyperparameters:"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size per Device: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $EFFECTIVE_BATCH_SIZE (standard: 512)"
echo "  Warmup Steps: 2000"
echo "  Weight Decay: 0.01"
echo "  Adam Beta1: 0.9"
echo "  Adam Beta2: 0.999"
echo "  Adam Epsilon: 1e-8"
echo "  Max Gradient Norm: 1.0"
echo ""
echo "Dataset:"
echo "  Training Samples: $NUM_SAMPLES"
echo "  Eval Samples: $EVAL_SAMPLES"
echo "  Epochs: $NUM_EPOCHS"
echo ""
echo "Conditioning Strategy:"
echo "  Conditioning: ${COND_PCT_MIN}-${COND_PCT_MAX} (20-40%, standard)"
echo "  Evaluation: ${EVAL_PCT_MIN}-${EVAL_PCT_MAX} (20-40%, standard)"
echo "  Sampling: Blockwise"
echo ""
echo "Evaluation:"
echo "  5-Mode Evaluation: Enabled"
echo "  Max Eval Batches: 20 (320 samples)"
echo "========================================="

# Run training with standard hyperparameters
python3 ./train.py \
    --model_type conditional \
    --model_config $MODEL_CONFIG \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_samples $NUM_SAMPLES \
    --num_eval_samples $EVAL_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --warmup_steps 2000 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --cond_pct_min $COND_PCT_MIN \
    --cond_pct_max $COND_PCT_MAX \
    --eval_pct_min $EVAL_PCT_MIN \
    --eval_pct_max $EVAL_PCT_MAX \
    --conditioning_sampling blockwise \
    --evaluation_sampling blockwise \
    --min_conditioning 1 \
    --min_evaluation 1 \
    --mode2_boundary_cond_pct_min 0.1 \
    --mode2_boundary_cond_pct_max 0.3 \
    --logging_steps 100 \
    --eval_steps 500 \
    --save_steps 1000 \
    --do_eval \
    --max_eval_batches 20 \
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
echo "Total duration: $SECONDS seconds (~$((SECONDS / 60)) minutes)"
echo "========================================="

# Print GPU memory usage at end
echo "Final GPU Memory Usage:"
nvidia-smi
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "Results:"
    echo "  Experiment directory: $OUTPUT_DIR/$EXP_NAME*"
    echo "  Checkpoints: $OUTPUT_DIR/$EXP_NAME*/checkpoints/"
    echo "  Logs: $OUTPUT_DIR/$EXP_NAME*/logs/"
    echo "  CSV Metrics: $OUTPUT_DIR/$EXP_NAME*/logs/metrics.csv"
    echo ""
    echo "Model characteristics:"
    echo "  - 81.9M parameters (same as DistilGPT-2)"
    echo "  - Standard GPT-2 hyperparameters"
    echo "  - Effective batch size: 512"
    echo "  - Conditioning: 20-40% (standard range)"
    echo ""
    echo "To view training curves:"
    echo "  python3 utils/quickstart_visualization.py $OUTPUT_DIR/$EXP_NAME*"
    echo ""
    echo "To use the trained model:"
    echo "  python3 train.py \
    --model_type conditional \\"
    echo "    --pretrained_model_path $OUTPUT_DIR/$EXP_NAME*/checkpoints/best_model.pt \\"
    echo "    --resume_training"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check error log: logs/slurm_${SLURM_JOB_ID}.err"
    echo "  2. Check output log: logs/slurm_${SLURM_JOB_ID}.out"
    echo "  3. Check GPU memory: nvidia-smi"
    echo "  4. Verify data loading: ls experiments/"
fi
echo "========================================="

exit $EXIT_CODE
