#!/bin/bash
#SBATCH --job-name=distilgpt2_large
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1

# Large-scale DistilGPT-2 training (closer to standard scale)
# Model: distilgpt2 (81.9M params)
# Target: ~100K training steps

echo "========================================="
echo "LARGE-SCALE DISTILGPT2 TRAINING"
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
export NVIDIA_TF32_OVERRIDE=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Print environment info
echo "Environment Information:"
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo "========================================="

# Large-scale training configuration
EXP_NAME="distilgpt2_large_scale"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilgpt2"

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
EVAL_PCT_MIN=0.2
EVAL_PCT_MAX=0.4

echo "========================================="
echo "Large-Scale Training Configuration:"
echo "========================================="
echo "Model: $MODEL_CONFIG (81.9M parameters)"
echo ""
echo "Scale:"
echo "  Training Samples: $NUM_SAMPLES (2M samples)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Expected Total Steps: ~195,000"
echo "  (Compared to standard DistilGPT-2: ~300K steps)"
echo ""
echo "Hyperparameters:"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Effective Batch Size: $EFFECTIVE_BATCH_SIZE"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "  Weight Decay: 0.01"
echo ""
echo "Conditioning:"
echo "  Conditioning: ${COND_PCT_MIN}-${COND_PCT_MAX} (20-40%)"
echo "  Evaluation: ${EVAL_PCT_MIN}-${EVAL_PCT_MAX} (20-40%)"
echo ""
echo "Estimated Training Time:"
echo "  ~5-7 days on A100"
echo "========================================="

# Run large-scale training
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
    --warmup_steps $WARMUP_STEPS \
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
    --eval_steps 1000 \
    --save_steps 5000 \
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
echo "Exit code: $EXIT_CODE"
echo "Duration: $((SECONDS / 3600)) hours"
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Large-scale training completed!"
    echo ""
    echo "Training Statistics:"
    echo "  Expected ~195K steps completed"
    echo "  Model: DistilGPT-2 (81.9M params)"
    echo "  Effective batch size: 512"
    echo ""
    echo "Results: $OUTPUT_DIR/$EXP_NAME*"
else
    echo "✗ Training failed"
    echo "Check logs: logs/slurm_${SLURM_JOB_ID}.err"
fi
echo "========================================="

exit $EXIT_CODE
