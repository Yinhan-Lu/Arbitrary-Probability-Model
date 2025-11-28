#!/bin/bash
#SBATCH --job-name=sigmagpt_old_paper
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/%x_%j/logs/slurm-%j.out
#SBATCH --error=experiments/%x_%j/logs/slurm-%j.out

# ==============================================================================
# Sigma GPT OLD Architecture (Double Position Encoding) with Paper's Evaluation
# ==============================================================================
# This script trains the ORIGINAL Sigma GPT model from the paper:
# - Double position encoding (n_embd // 2 per position)
# - Encodes both current position and next position
# - Uses paper's autoregressive evaluation (left-to-right, Mode 1)
#
# Reference: https://arxiv.org/abs/2404.09562
# ==============================================================================

set -e

# Configuration
MODEL_CONFIG=${MODEL_CONFIG:-"small"}
SIGMAGPT_MODE=${SIGMAGPT_MODE:-"fair"}
NUM_EPOCHS=${NUM_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-16}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
EVAL_STEPS=${EVAL_STEPS:-500}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOGGING_STEPS=${LOGGING_STEPS:-100}
NUM_TRAIN_SAMPLES=${NUM_TRAIN_SAMPLES:-100000}
NUM_EVAL_SAMPLES=${NUM_EVAL_SAMPLES:-5000}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-50}

# Generate experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="sigmagpt_old_${SIGMAGPT_MODE}_${MODEL_CONFIG}_${TIMESTAMP}"

echo "=============================================="
echo "Sigma GPT OLD (Paper's Double Position Encoding)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Experiment: $EXP_NAME"
echo "Model config: $MODEL_CONFIG"
echo "Architecture: OLD (double position encoding)"
echo "Training mode: $SIGMAGPT_MODE"
echo "Evaluation: autoregressive (paper's method)"
echo "=============================================="

# Create experiment directory
EXP_DIR="experiments/${EXP_NAME}"
mkdir -p "${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}/checkpoints"

# Activate environment
source ~/.bashrc
conda activate apm || source ~/venv/bin/activate || true

# Change to project directory
cd /home/mila/l/luy/Arbitrary-Probability-Model

# Print Python and PyTorch info
echo ""
echo "Environment:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Run training with OLD architecture and autoregressive evaluation
echo "Starting training..."
python train.py \
    --model_type sigmagpt \
    --model_config ${MODEL_CONFIG} \
    --sigmagpt_mode ${SIGMAGPT_MODE} \
    --sigmagpt_arch old \
    --sigmagpt_eval_mode autoregressive \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE} \
    --do_eval \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --num_train_samples ${NUM_TRAIN_SAMPLES} \
    --num_eval_samples ${NUM_EVAL_SAMPLES} \
    --max_eval_batches ${MAX_EVAL_BATCHES} \
    --exp_name ${EXP_NAME} \
    --ordering_mode temporal \
    --conditioning_sampling blockwise \
    --evaluation_sampling blockwise

echo ""
echo "=============================================="
echo "Training completed!"
echo "Results saved to: ${EXP_DIR}"
echo "=============================================="
echo ""
echo "Key differences from NEW architecture:"
echo "  - Position embedding: n_embd // 2 (not n_embd)"
echo "  - Encodes: current position + next position"
echo "  - Evaluation: autoregressive (left-to-right)"
echo "=============================================="
