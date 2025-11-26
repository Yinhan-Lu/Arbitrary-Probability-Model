#!/bin/bash
#SBATCH --job-name=distilbert_std
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1

# Large-scale DistilBERT training (MLM)
# Model: DistilBertForMaskedLM (GPT-2 tokenizer, seq_len=1024)
# Uses your DistilBertTrainer with 3 evaluation modes

echo "========================================="
echo "STANDARD DISTILBERT TRAINING"
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

# Standard DistilBert hyperparameters
EXP_NAME="distilbert_standard_scale"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilbert-base-uncased"

# Standard DistilBert training hyperparameters
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

echo "Training Configuration:"
echo "  Model:            $MODEL_CONFIG (DistilBERT ~80M params)"
echo "  Epochs:           $NUM_EPOCHS"
echo "  Train samples:    $NUM_SAMPLES"
echo "  Eval samples:     $EVAL_SAMPLES"
echo "  Batch size:       $BATCH_SIZE"
echo "  Grad accumulation $GRAD_ACCUM"
echo "  Effective batch:  $EFFECTIVE_BATCH_SIZE"
echo "  Learning rate:    $LEARNING_RATE"
echo "  cond_pct:         ${COND_PCT_MIN}-${COND_PCT_MAX} (BERT Mode 3)"
echo "  Sampling:         pure MLM + BERT eval modes"
echo "========================================="

# Run training with standard hyperparameters
python -u train.py \
  --model_type distilbert \
  --model_config distilbert-base-uncased \
  --dataset_name wikitext \
  --dataset_config wikitext-103-v1 \
  --num_epochs 1 \
  --num_train_samples 1500 \
  --num_eval_samples 300 \
  --batch_size 1 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --warmup_steps 50 \
  --logging_steps 20 \
  --eval_steps 60 \
  --max_eval_batches 3 \
  --save_steps 999999 \
  --do_eval \
  --device cpu \
  --output_dir ./experiments \
  --exp_name distilbert_10min_debug


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
    echo "✓ DistilBERT training completed successfully!"
    echo ""
    echo "Results:"
    echo "  Experiment dir:  $OUTPUT_DIR/$EXP_NAME*/"
    echo "  Checkpoints:     $OUTPUT_DIR/$EXP_NAME*/checkpoints/"
    echo "  Logs:            $OUTPUT_DIR/$EXP_NAME*/logs/"
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
