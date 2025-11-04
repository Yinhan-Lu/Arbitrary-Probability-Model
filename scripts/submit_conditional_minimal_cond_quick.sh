#!/bin/bash
#SBATCH --job-name=cond_quick
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1

# Quick test for minimal conditioning training (15 minutes)
# NO unseen set: evaluation = all non-conditioning tokens
# Model: distilgpt2 (81.9M params)

echo "========================================="
echo "QUICK TEST - MINIMAL CONDITIONING"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Quick test parameters
EXP_NAME="conditional_minimal_cond_quick"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilgpt2"
BATCH_SIZE=4
GRAD_ACCUM=8
NUM_SAMPLES=5000      # Small for quick test
EVAL_SAMPLES=500
LEARNING_RATE=5e-4
NUM_EPOCHS=1

echo "Quick Test Configuration:"
echo "  Model: $MODEL_CONFIG"
echo "  Epochs: $NUM_EPOCHS"
echo "  Training Samples: $NUM_SAMPLES (small for testing)"
echo "  Eval Samples: $EVAL_SAMPLES"
echo "  Conditioning: 0-10% (minimal)"
echo "  Evaluation: 100% of non-conditioning (no unseen)"
echo "  Expected Time: ~10 minutes"
echo "========================================="

# Run quick test
python3 ./train_conditional.py \
    --model_config $MODEL_CONFIG \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 8 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_samples $NUM_SAMPLES \
    --num_eval_samples $EVAL_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --warmup_steps 100 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
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
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --do_eval \
    --max_eval_batches 5 \
    --output_dir $OUTPUT_DIR \
    --exp_name $EXP_NAME \
    --device cuda \
    --num_workers 2

EXIT_CODE=$?

echo "========================================="
echo "Quick Test Completed"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "Duration: $((SECONDS / 60)) minutes"
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Quick test passed!"
    echo ""
    echo "Configuration verified:"
    echo "  ✓ Minimal conditioning (0-10%) works"
    echo "  ✓ No unseen set (eval=100%) works"
    echo "  ✓ 5-mode evaluation runs successfully"
    echo ""
    echo "Ready for full training!"
    echo "  Run: sbatch scripts/submit_conditional_minimal_cond.sh"
else
    echo "✗ Quick test failed!"
    echo "  Check logs: logs/slurm_${SLURM_JOB_ID}.out/err"
fi

exit $EXIT_CODE
