#!/bin/bash

# ==========================================================================
# QUICK CONVERGENCE TEST: BERT (DistilBERT MLM)
# ==========================================================================
# Small-scale test to verify BERT training/evaluation pipeline
# Uses same hyperparameters as conditional/sigmagpt but tiny dataset
# Should complete in ~5-10 minutes on CPU

echo "========================================="
echo "QUICK CONVERGENCE TEST: BERT"
echo "  Dataset: tiny (1000 train, 200 eval)"
echo "  Epochs: 3"
echo "  Early stopping: disabled (patience=0)"
echo "  Device: auto (will use MPS/CUDA if available)"
echo "========================================="

# Activate virtual environment
if [ -d "myenv" ]; then
    echo "Activating myenv..."
    source myenv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating venv..."
    source venv/bin/activate
fi

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Training parameters (matching converge scripts but tiny)
EXP_NAME="bert_converge_test_quick"
MODEL_CONFIG="distilbert-base-uncased"

# Tiny dataset for quick testing
BATCH_SIZE=4
GRAD_ACCUM=2
NUM_SAMPLES=1000           # Very small for quick test
EVAL_SAMPLES=200
NUM_EPOCHS=3               # Just a few epochs
LEARNING_RATE=5e-4
WEIGHT_DECAY=0.01

# Conditioning configuration (matching conditional/sigmagpt)
COND_PCT_MIN=0.2
COND_PCT_MAX=0.4

echo "Configuration:"
echo "  Model: $MODEL_CONFIG"
echo "  Train Samples: $NUM_SAMPLES"
echo "  Eval Samples: $EVAL_SAMPLES"
echo "  Epochs: $NUM_EPOCHS (quick test)"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Conditioning: ${COND_PCT_MIN}-${COND_PCT_MAX}"
echo ""
echo "  ** Eval Steps: 20 (frequent for testing) **"
echo "  ** Early Stopping: DISABLED (patience=0) **"
echo "  ** Max Eval Batches: 3 (quick eval) **"
echo "========================================="

python ./train.py \
    --model_type distilbert \
    --model_config $MODEL_CONFIG \
    --dataset_name wikitext \
    --dataset_config wikitext-103-v1 \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 8 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_samples $NUM_SAMPLES \
    --num_eval_samples $EVAL_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --warmup_steps 50 \
    --max_grad_norm 1.0 \
    --weight_decay $WEIGHT_DECAY \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --cond_pct_min $COND_PCT_MIN \
    --cond_pct_max $COND_PCT_MAX \
    --logging_steps 10 \
    --eval_steps 20 \
    --save_steps 999999 \
    --early_stopping_patience 0 \
    --do_eval \
    --max_eval_batches 3 \
    --output_dir ./experiments \
    --exp_name $EXP_NAME \
    --device auto \
    --num_workers 2

EXIT_CODE=$?

echo "========================================="
echo "Test Completed"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ BERT quick convergence test completed!"
    echo ""
    echo "Results: ./experiments/$EXP_NAME*"
    echo ""
    echo "Check metrics:"
    echo "  cat ./experiments/${EXP_NAME}*/logs/metrics.csv"
    echo ""
    echo "View all 5 modes in CSV (train_loss, mode1-5_loss, mode1-5_ppl)"
else
    echo "✗ Test failed"
fi
echo "========================================="

exit $EXIT_CODE
