#!/bin/bash
# 10-minute GPU training script for prefix conditioning
# Model: small (~38.6M params)
# Expected time: ~10 minutes on GPU

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arbprob

python3 train_conditional.py \
    --model_config small \
    --num_epochs 3 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --num_train_samples 50000 \
    --num_eval_samples 5000 \
    --num_workers 4 \
    --learning_rate 5e-4 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --conditioning_sampling blockwise \
    --evaluation_sampling blockwise \
    --max_cond_blocks 2 \
    --max_eval_blocks 2 \
    --cond_pct_min 0.2 \
    --cond_pct_max 0.4 \
    --eval_pct_min 0.2 \
    --eval_pct_max 0.4 \
    --logging_steps 50 \
    --save_steps 500 \
    --output_dir ./experiments \
    --exp_name small_prefix_10min \
    --device cuda

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
