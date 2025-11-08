#!/bin/bash
# M-chip (MPS) GPU Quick Test
# Testing new concatenate + chunk data pipeline on MacBook M-chip
# Expected time: ~10-15 minutes

echo "=========================================="
echo "M-chip (MPS) Training Quick Test"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: small (38.6M params)"
echo "  Training samples: 10,000"
echo "  Evaluation samples: 1,000"
echo "  Batch size: 8"
echo "  Device: MPS (M-chip GPU)"
echo "  Expected time: ~10-15 minutes"
echo ""
echo "This test validates:"
echo "  1. MPS device support works correctly"
echo "  2. New concatenate + chunk data pipeline"
echo "  3. No placeholder text contamination"
echo ""
echo "=========================================="
echo ""

python3 train_conditional.py \
    --model_config small \
    --num_epochs 1 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_train_samples 10000 \
    --num_eval_samples 1000 \
    --num_workers 2 \
    --learning_rate 5e-4 \
    --warmup_steps 200 \
    --cond_pct_min 0.2 \
    --cond_pct_max 0.4 \
    --eval_pct_min 0.2 \
    --eval_pct_max 0.4 \
    --max_cond_blocks 3 \
    --max_eval_blocks 2 \
    --conditioning_sampling blockwise \
    --evaluation_sampling blockwise \
    --logging_steps 50 \
    --save_steps 500 \
    --do_eval \
    --eval_steps 500 \
    --max_eval_batches 50 \
    --output_dir ./experiments \
    --exp_name mps_pipeline_test \
    --device mps

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Results saved to: experiments/mps_pipeline_test_*"
echo ""
echo "To visualize results:"
echo "  python utils/quickstart_visualization.py experiments/mps_pipeline_test_*"
echo ""
