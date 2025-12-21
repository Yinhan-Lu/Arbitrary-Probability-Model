#!/bin/bash
#SBATCH --job-name=sigmagpt_full
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1

# Sigma GPT Training - Full Mode (100% learning efficiency)
#
# Full mode trains on all tokens in the sequence, providing 100% learning
# efficiency (same as standard autoregressive training). This is useful for:
# - Maximum learning efficiency
# - Best model performance
# - Comparison with fair mode to understand the impact of learning efficiency
#
# Based on Sigma GPT paper (ArXiv 2404.09562)

echo "========================================="
echo "SIGMA GPT TRAINING - FULL MODE"
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

# Training configuration
EXP_NAME="sigmagpt_full_distilgpt2"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilgpt2"
MODE="full"

# Batch configuration
# Effective batch size = 512 (Sigma GPT paper recommendation)
BATCH_SIZE=8
GRAD_ACCUM=64
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))

# Dataset configuration
# WikiText-103 full training
NUM_SAMPLES=2000000           # 2M samples (multiple passes over WikiText-103)
EVAL_SAMPLES=10000
NUM_EPOCHS=50

# Hyperparameters (from Sigma GPT paper and GPT-2)
LEARNING_RATE=2.5e-4          # Sigma GPT recommendation
WARMUP_STEPS=10000            # Longer warmup for large-scale training
WEIGHT_DECAY=0.1              # Sigma GPT uses 0.1 (GPT-2 standard)
ADAM_BETA1=0.9
ADAM_BETA2=0.95               # Sigma GPT uses 0.95
MAX_GRAD_NORM=1.0

# Conditioning configuration
# Use blockwise sampling (default in ConditionalAugmenter)
MAX_COND_BLOCKS=2

# Logging and checkpointing
LOGGING_STEPS=100
EVAL_STEPS=1000
SAVE_STEPS=5000
MAX_EVAL_BATCHES=50

echo "========================================="
echo "Sigma GPT Full Mode Training Configuration:"
echo "========================================="
echo "Model: $MODEL_CONFIG (DistilGPT-2, ~82M parameters)"
echo ""
echo "Training Mode: FULL (100% learning efficiency)"
echo "  - All tokens contribute to loss"
echo "  - Maximum learning efficiency"
echo "  - Equivalent to standard autoregressive training"
echo ""
echo "Scale:"
echo "  Training Samples: $NUM_SAMPLES (2M samples)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Expected Total Steps: ~195,000"
echo ""
echo "Hyperparameters (Sigma GPT recommendations):"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Effective Batch Size: $EFFECTIVE_BATCH_SIZE"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Adam Beta: ($ADAM_BETA1, $ADAM_BETA2)"
echo "  Gradient Clip: $MAX_GRAD_NORM"
echo ""
echo "Conditioning:"
echo "  Sampling: Blockwise"
echo "  Max Cond Blocks: $MAX_COND_BLOCKS"
echo ""
echo "Estimated Training Time:"
echo "  ~5-7 days on A100"
echo "========================================="

# Run training using unified train.py
python3 ./train.py \
    --model_type sigmagpt \
    --model_config $MODEL_CONFIG \
    --sigmagpt_mode $MODE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 16 \
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
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --max_eval_batches $MAX_EVAL_BATCHES \
    --output_dir $OUTPUT_DIR \
    --exp_name $EXP_NAME \
    --device cuda \
    --num_workers 4 \
    --primary_dataset_only \
    --fp16

EXIT_CODE=$?

echo "========================================="
echo "Training Completed"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "Duration: $((SECONDS / 3600)) hours"
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Sigma GPT Full mode training completed!"
    echo ""

    # Find the most recent experiment directory
    LATEST_EXP=$(ls -dt $OUTPUT_DIR/${EXP_NAME}_* 2>/dev/null | head -1)

    echo "Training Statistics:"
    echo "  Mode: Full (100% learning efficiency)"
    echo "  Expected ~195K steps completed"
    echo "  Model: DistilGPT-2 (82M params)"
    echo "  Effective batch size: 512"
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
    echo "Next Steps:"
    echo "  1. Review individual metric plots (especially train/eval loss curves)"
    echo "  2. Evaluate model on test set"
    echo "  3. Compare with fair mode: python3 utils/quickstart_visualization.py $LATEST_EXP {fair_mode_exp} --compare"
else
    echo "✗ Training failed"
    echo "Check logs: logs/slurm_${SLURM_JOB_ID}.err"
fi
echo "========================================="

exit $EXIT_CODE
