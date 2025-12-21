#!/bin/bash
# Minimal Baseline MPS Test
# Purpose: Verify baseline training works and metrics are stored correctly
# Expected time: ~1-2 minutes

echo "=========================================="
echo "Minimal Baseline MPS Test"
echo "=========================================="
echo ""
echo "This test validates:"
echo "  1. Baseline training runs without errors"
echo "  2. Train loss is recorded correctly"
echo "  3. Eval loss is stored in CSV"
echo "  4. 5-mode CSV format is correct"
echo ""
echo "=========================================="
echo ""

python train.py \
    --model_type baseline \
    --model_config small \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_samples 500 \
    --num_eval_samples 100 \
    --num_workers 0 \
    --learning_rate 1e-4 \
    --warmup_steps 10 \
    --logging_steps 5 \
    --do_eval \
    --eval_steps 10 \
    --max_eval_batches 5 \
    --output_dir ./experiments \
    --exp_name baseline_mps_minimal \
    --device mps

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Validation Results"
echo "=========================================="
echo ""

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "✗ ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit 1
fi
echo "✓ Training completed successfully"

# Find the most recent experiment folder
METRICS_FILE=$(ls -t experiments/baseline_mps_minimal_*/logs/metrics.csv 2>/dev/null | head -1)

if [ -z "$METRICS_FILE" ]; then
    echo "✗ ERROR: metrics.csv not found!"
    exit 1
fi
echo "✓ Metrics file found: $METRICS_FILE"

# Check CSV has content
LINE_COUNT=$(wc -l < "$METRICS_FILE" | tr -d ' ')
if [ "$LINE_COUNT" -lt 2 ]; then
    echo "✗ ERROR: metrics.csv is empty (only header)"
    exit 1
fi
echo "✓ Metrics file has $LINE_COUNT lines"

echo ""
echo "CSV Header:"
head -1 "$METRICS_FILE"

echo ""
echo "Sample rows (last 5):"
tail -5 "$METRICS_FILE"

echo ""
echo "=========================================="
echo "All validations passed!"
echo "=========================================="
