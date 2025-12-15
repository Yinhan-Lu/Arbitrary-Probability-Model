#!/bin/bash
#SBATCH --job-name=distilbert_large
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1


# Large-scale DistilBERT training (MLM)
# Model: DistilBertForMaskedLM (GPT-2 tokenizer, seq_len=1024)
# Uses your DistilBertTrainer with 3 evaluation modes

echo "========================================="
echo "LARGE-SCALE DISTILBERT TRAINING"
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

# Activate conda environment
echo "Activating conda environment 'arbprob'..."
source ~/miniconda3/etc/profile.d/conda.sh || {
    echo "ERROR: Failed to source conda.sh"
    exit 1
}

conda activate arbprob || {
    echo "ERROR: Failed to activate conda environment 'arbprob'"
    conda env list
    exit 1
}
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python3)"
echo "========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Print environment info
echo "Environment Information:"
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo "========================================="

# Large-scale training configuration
EXP_NAME="distilbert_large_scale"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilbert-base-uncased"

# Batch configuration (effective batch size = 512)
BATCH_SIZE=8
GRAD_ACCUM=64
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))

# Large-scale dataset
# Use full WikiText-103 train set multiple times to simulate larger dataset
NUM_SAMPLES=2000000             # 2M samples (use full dataset ~20 times)
EVAL_SAMPLES=10000
NUM_EPOCHS=50                   # Many passes over the data

# This gives us approximately:
# Steps per epoch: 2M / 512 = 3,906 steps
# Total steps: 3,906 × 50 = 195,312 steps (~195K steps)
# Total tokens seen: 512 × 195K × avg_seq_len ≈ 100M × avg_seq_len

# Standard hyperparameters
LEARNING_RATE=2.5e-4
WARMUP_STEPS=10000              # Longer warmup for large-scale training

# Conditioning configuration
COND_PCT_MIN=0.2
COND_PCT_MAX=0.4

echo "========================================="
echo "Large-Scale DistilBERT Configuration:"
echo "========================================="
echo "Model config: $MODEL_CONFIG"
echo "Training samples: $NUM_SAMPLES"
echo "Eval samples:     $EVAL_SAMPLES"
echo "Epochs:           $NUM_EPOCHS"
echo "Batch size:       $BATCH_SIZE"
echo "Grad accum:       $GRAD_ACCUM"
echo "Effective batch:  $EFFECTIVE_BATCH_SIZE"
echo "Learning rate:    $LEARNING_RATE"
echo "Warmup steps:     $WARMUP_STEPS"
echo "MLM eval cond%:   ${COND_PCT_MIN}-${COND_PCT_MAX}"
echo "========================================="

# Run large-scale training
python3 ./train.py \
  --model_type distilbert \
  --model_config $MODEL_CONFIG \
  --dataset_name wikitext \
  --dataset_config wikitext-103-v1 \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --eval_batch_size 32 \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --num_train_samples $NUM_SAMPLES \
  --num_eval_samples $EVAL_SAMPLES \
  --learning_rate $LEARNING_RATE \
  --logging_steps 200 \
  --eval_steps 5000 \
  --save_steps 10000 \
  --do_eval \
  --output_dir $OUTPUT_DIR \
  --exp_name $EXP_NAME \
  --device cuda \
  --num_workers 4

EXIT_CODE=$?

echo "========================================="
echo "Training Completed"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "Duration: $((SECONDS / 3600)) hours"
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ DistilBert Large-scale training completed!"
    echo ""
    echo "Training Statistics:"
    echo "  Expected ~195K steps completed"
    echo "  Model: DistilBert"
    echo ""
    echo "Results: $OUTPUT_DIR/$EXP_NAME*"
else
    echo "✗ Training failed"
    echo "Check logs: logs/slurm_${SLURM_JOB_ID}.err"
fi
echo "========================================="

exit $EXIT_CODE
