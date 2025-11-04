#!/bin/bash
#SBATCH --job-name=cond_minimal
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1

# Conditional training with minimal conditioning (0-10%)
# NO unseen set: evaluation = all non-conditioning tokens
# Model: distilgpt2 (81.9M params)

echo "========================================="
echo "CONDITIONAL TRAINING - MINIMAL CONDITIONING"
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

# Change to submission directory (project root)
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

# Training parameters
EXP_NAME="conditional_minimal_cond"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilgpt2"
BATCH_SIZE=8
GRAD_ACCUM=16
NUM_SAMPLES=200000
EVAL_SAMPLES=10000
LEARNING_RATE=5e-4
NUM_EPOCHS=5

echo "Training Configuration:"
echo "  Model: $MODEL_CONFIG (81.9M params)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size per Device: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Training Samples: $NUM_SAMPLES"
echo "  Eval Samples: $EVAL_SAMPLES"
echo "  Learning Rate: $LEARNING_RATE"
echo "========================================="
echo "Conditioning Strategy:"
echo "  Conditioning: 0-10% of sequence (minimal)"
echo "  Evaluation: 100% of non-conditioning (no unseen set)"
echo "  Max Cond Blocks: 3"
echo "  Max Eval Blocks: 2"
echo "========================================="

# Run training with distribution-based sampling
python3 ./train_conditional.py \
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
    --logging_steps 100 \
    --eval_steps 500 \
    --save_steps 1000 \
    --do_eval \
    --max_eval_batches 100 \
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
    echo "Key characteristics of this training:"
    echo "  - Minimal conditioning (0-10% of sequence)"
    echo "  - No unseen tokens (evaluation = all non-conditioning)"
    echo "  - Tests model's ability with very limited context"
    echo ""
    echo "To view 5-mode evaluation results:"
    echo "  cat $OUTPUT_DIR/$EXP_NAME*/logs/metrics.csv | column -t -s,"
    echo ""
    echo "To visualize training curves:"
    echo "  python3 utils/quickstart_visualization.py $OUTPUT_DIR/$EXP_NAME*"
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
