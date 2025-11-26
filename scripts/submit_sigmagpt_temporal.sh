#!/bin/bash
#SBATCH --job-name=sigmagpt_temporal
#SBATCH --output=logs/sigmagpt_temporal_%j.out
#SBATCH --error=logs/sigmagpt_temporal_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=long

# Eric's Method 1: Temporal Ordering
# Maintains left-to-right order within conditioning and evaluation sets
# Enables fair comparison with existing conditional model

echo "=========================================="
echo "Sigma GPT Training - Temporal Ordering"
echo "Eric's Method 1"
echo "=========================================="

# Activate environment
source ~/.bashrc
conda activate arbitrary_prob

# Run training
python train_sigmagpt.py \
    --model_config distilgpt2 \
    --dataset_name wikitext \
    --dataset_config wikitext-103-raw-v1 \
    --mode fair \
    --ordering_mode temporal \
    --eval_splits_file utils/evaluation_splits/wikitext103_valid_seed42.pt \
    --exp_name sigmagpt_temporal_fair_distilgpt2 \
    --num_epochs 100 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_cond_blocks 2 \
    --max_eval_blocks 1 \
    --seed 42 \
    --num_workers 4 \
    --logging_steps 100 \
    --eval_steps 1000 \
    --save_steps 5000

echo "=========================================="
echo "Training Complete"
echo "=========================================="
