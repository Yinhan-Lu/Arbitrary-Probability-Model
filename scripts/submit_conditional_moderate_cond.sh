#!/bin/bash
#SBATCH --job-name=cond_moderate
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1

# Conditional training with moderate conditioning (20-40%) - Plan A
# Improved regularization to prevent overfitting
# NO unseen set: evaluation = all non-conditioning tokens
# Model: distilgpt2 (81.9M params)

echo "========================================="
echo "CONDITIONAL TRAINING - MODERATE CONDITIONING (PLAN A)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
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

# Enable TF32 for faster training on A100
export NVIDIA_TF32_OVERRIDE=1

# Change to submission directory (project root)
cd "$SLURM_SUBMIT_DIR"

# Create logs directory
mkdir -p logs

# Print environment info
echo "Environment Information:"
echo "Python version:"
python3 --version
echo "PyTorch version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "========================================="

# Training parameters
EXP_NAME="conditional_moderate_cond"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilgpt2"
BATCH_SIZE=8
GRAD_ACCUM=16
NUM_SAMPLES=1000000
EVAL_SAMPLES=10000
LEARNING_RATE=5e-4
NUM_EPOCHS=5

echo "Training Configuration (PLAN A - Conservative Anti-Overfitting):"
echo "  Model: $MODEL_CONFIG (81.9M params)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size per Device: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Training Samples: $NUM_SAMPLES"
echo "  Eval Samples: $EVAL_SAMPLES"
echo "  Learning Rate: $LEARNING_RATE"
echo "========================================="
echo "Conditioning Strategy (MODIFIED):"
echo "  Conditioning: 0-40% of sequence (moderate, up from 0-10%)"
echo "  Evaluation: 100% of non-conditioning (no unseen set)"
echo "  Max Cond Blocks: 3"
echo "  Max Eval Blocks: 2"
echo "========================================="
echo "Regularization Changes:"
echo "  Weight Decay: 0.1 (increased from 0.01)"
echo "  More meaningful conditioning task (harder to memorize)"
echo "========================================="

# Run training with distribution-based sampling
python3 ./train_conditional.py \
    --model_config $MODEL_CONFIG \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_samples $NUM_SAMPLES \
    --num_eval_samples $EVAL_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --warmup_steps 2000 \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --cond_pct_min 0.0 \
    --cond_pct_max 0.4 \
    --eval_pct_min 1.0 \
    --eval_pct_max 1.0 \
    --conditioning_sampling blockwise \
    --evaluation_sampling blockwise \
    --min_conditioning 0 \
    --min_evaluation 1 \
    --mode2_boundary_cond_pct_min 0.1 \
    --mode2_boundary_cond_pct_max 0.3 \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 1000 \
    --do_eval \
    --max_eval_batches 10 \
    --output_dir $OUTPUT_DIR \
    --exp_name $EXP_NAME \
    --device cuda \
    --num_workers 4

EXIT_CODE=$?

echo "========================================="
echo "Training Completed"
echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Total duration: $SECONDS seconds (~$((SECONDS / 60)) minutes)"
echo "========================================="

# Print GPU memory usage at end
echo "Final GPU Memory Usage:"
nvidia-smi
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""

    # Find the most recent experiment directory
    LATEST_EXP=$(ls -dt $OUTPUT_DIR/${EXP_NAME}_* 2>/dev/null | head -1)

    echo "Results:"
    echo "  Experiment directory: $LATEST_EXP"
    echo "  Checkpoints: $LATEST_EXP/checkpoints/"
    echo "  Logs: $LATEST_EXP/logs/"
    echo "  CSV Metrics: $LATEST_EXP/logs/metrics.csv"
    echo ""
    echo "Key improvements in this training (Plan A):"
    echo "  ✓ Moderate conditioning (20-40%, up from 0-10%)"
    echo "  ✓ Stronger weight decay (0.1, up from 0.01)"
    echo "  ✓ More meaningful task (harder to memorize)"
    echo ""
    echo "Expected improvements:"
    echo "  - Better generalization (lower eval loss)"
    echo "  - Less overfitting (train/eval gap reduced)"
    echo "  - Train loss should stay closer to eval loss"
    echo ""

    # Auto-generate visualization plots
    if [ -n "$LATEST_EXP" ] && [ -d "$LATEST_EXP" ]; then
        echo "========================================="
        echo "AUTO-GENERATING VISUALIZATION PLOTS"
        echo "========================================="

        # Generate detailed individual plots
        echo "Generating individual metric plots..."
        python3 utils/plot_individual_metrics.py "$LATEST_EXP"
        PLOT_EXIT_CODE=$?

        if [ $PLOT_EXIT_CODE -eq 0 ]; then
            echo "✓ Individual plots generated successfully!"
            echo "  Location: $LATEST_EXP/plots_individual/"
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

    echo "To view 5-mode evaluation results:"
    echo "  cat $LATEST_EXP/logs/metrics.csv | column -t -s,"
    echo ""
    echo "Plots generated:"
    echo "  Individual plots: $LATEST_EXP/plots_individual/"
    echo "  Dashboard: $LATEST_EXP/plots/"
    echo ""
    echo "To compare with previous training:"
    echo "  python3 utils/quickstart_visualization.py experiments/conditional_minimal_cond_* experiments/conditional_moderate_cond_* --compare"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check error log: logs/slurm_${SLURM_JOB_ID}.err"
    echo "  2. Check output log: logs/slurm_${SLURM_JOB_ID}.out"
    echo "  3. Check GPU memory: nvidia-smi"
    echo "  4. Verify data loading: ls experiments/"
fi
echo "========================================="

exit $EXIT_CODE
