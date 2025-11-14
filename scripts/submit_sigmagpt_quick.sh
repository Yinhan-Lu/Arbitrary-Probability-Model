#!/bin/bash
#SBATCH --job-name=sigmagpt_quick
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1

# Sigma GPT Quick Test - Fair Mode
#
# Quick training run to verify:
# - Training pipeline works correctly
# - No crashes or errors
# - GPU memory is sufficient
# - Data loading works
#
# This should complete in ~30-60 minutes

echo "========================================="
echo "SIGMA GPT QUICK TEST"
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

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Print environment info
echo "Environment Information:"
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo "========================================="

# Quick test configuration
EXP_NAME="sigmagpt_quick_test"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilgpt2"
MODE="fair"

# Small batch configuration for quick testing
BATCH_SIZE=4
GRAD_ACCUM=2
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))

# Small dataset for quick testing
NUM_SAMPLES=1000              # Only 1K samples
EVAL_SAMPLES=500
NUM_EPOCHS=1                  # Just 1 epoch

# Standard hyperparameters
LEARNING_RATE=2.5e-4
WARMUP_STEPS=100              # Short warmup
WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95
MAX_GRAD_NORM=1.0

# Conditioning configuration
MAX_COND_BLOCKS=2
MAX_EVAL_BLOCKS=1

# Frequent logging for debugging
# Note: Quick test only has ~7 steps, so use very small logging intervals
LOGGING_STEPS=1       # Log every step (only 7 steps total)
EVAL_STEPS=3          # Evaluate every 3 steps
SAVE_STEPS=5          # Save every 5 steps
MAX_EVAL_BATCHES=10

echo "========================================="
echo "Quick Test Configuration:"
echo "========================================="
echo "Model: $MODEL_CONFIG"
echo "Mode: $MODE"
echo ""
echo "Quick Test Settings:"
echo "  Training Samples: $NUM_SAMPLES (1K)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $EFFECTIVE_BATCH_SIZE"
echo ""
echo "Expected completion time: ~30-60 minutes"
echo "========================================="

# Run quick test using unified train.py
python3 ./train.py \
    --model_type sigmagpt \
    --model_config $MODEL_CONFIG \
    --sigmagpt_mode $MODE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 8 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_samples $NUM_SAMPLES \
    --num_eval_samples $EVAL_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --adam_beta1 $ADAM_BETA1 \
    --adam_beta2 $ADAM_BETA2 \
    --max_grad_norm $MAX_GRAD_NORM \
    --warmup_steps $WARMUP_STEPS \
    --conditioning_sampling blockwise \
    --evaluation_sampling blockwise \
    --max_cond_blocks $MAX_COND_BLOCKS \
    --max_eval_blocks $MAX_EVAL_BLOCKS \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --max_eval_batches $MAX_EVAL_BATCHES \
    --output_dir $OUTPUT_DIR \
    --exp_name $EXP_NAME \
    --device cuda \
    --num_workers 2 \
    --primary_dataset_only \
    --fp16

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

    # Find the most recent experiment directory
    LATEST_EXP=$(ls -dt $OUTPUT_DIR/${EXP_NAME}_* 2>/dev/null | head -1)

    echo "The training pipeline is working correctly."
    echo ""
    echo "Results:"
    echo "  Experiment directory: $LATEST_EXP"
    echo "  Checkpoints: $LATEST_EXP/checkpoints/"
    echo "  Logs: $LATEST_EXP/logs/"
    echo "  CSV Metrics: $LATEST_EXP/logs/metrics.csv"
    echo ""

    # Auto-generate visualization plots
    if [ -n "$LATEST_EXP" ] && [ -d "$LATEST_EXP" ]; then
        echo "========================================="
        echo "AUTO-GENERATING VISUALIZATION PLOTS"
        echo "========================================="

        # Generate detailed individual plots (PRIORITY)
        echo "Generating individual metric plots..."
        python3 utils/plot_individual_metrics.py "$LATEST_EXP"
        PLOT_EXIT_CODE=$?

        if [ $PLOT_EXIT_CODE -eq 0 ]; then
            echo "✓ Individual plots generated successfully!"
            echo "  Location: $LATEST_EXP/plots_individual/"
            echo "  Files: train_loss.png, train_perplexity.png, learning_rate.png, etc."
        else
            echo "⚠ Warning: Failed to generate individual plots (exit code: $PLOT_EXIT_CODE)"
        fi

        # Generate comprehensive visualization dashboard
        echo ""
        echo "Generating comprehensive dashboard..."
        python3 utils/quickstart_visualization.py "$LATEST_EXP"
        DASH_EXIT_CODE=$?

        if [ $DASH_EXIT_CODE -eq 0 ]; then
            echo "✓ Dashboard generated successfully!"
            echo "  Location: $LATEST_EXP/plots/"
        else
            echo "⚠ Warning: Failed to generate dashboard (exit code: $DASH_EXIT_CODE)"
        fi

        echo "========================================="
        echo ""
    fi

    echo "Visualization results:"
    echo "  📊 Individual plots: $LATEST_EXP/plots_individual/"
    echo "  📈 Dashboard: $LATEST_EXP/plots/"
    echo ""
    echo "You can now run full-scale training with:"
    echo "  sbatch scripts/submit_sigmagpt_fair.sh"
    echo "  sbatch scripts/submit_sigmagpt_full.sh"
else
    echo "✗ Quick test failed"
    echo "Check logs: logs/slurm_${SLURM_JOB_ID}.err"
    echo ""
    echo "Common issues:"
    echo "  1. GPU memory insufficient - reduce batch_size"
    echo "  2. Dataset loading issues - check internet connection"
    echo "  3. Missing dependencies - check requirements"
fi
echo "========================================="

exit $EXIT_CODE
