#!/bin/bash
#SBATCH --job-name=sigmagpt_old
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1

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

echo "========================================="
echo "SIGMA GPT OLD - PAPER'S ARCHITECTURE"
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

# Load CUDA and conda environment (Mila cluster)
module load cuda/12.1.1/cudnn/8.9
source /cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/etc/profile.d/conda.sh
conda activate arbprob

# Print environment info
echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "========================================="

# Training parameters (aligned with Sigma GPT paper: arXiv 2404.09562)
MODEL_CONFIG=${MODEL_CONFIG:-"distilgpt2"}
SIGMAGPT_MODE=${SIGMAGPT_MODE:-"fair"}
EXP_NAME="sigmagpt_old_${SIGMAGPT_MODE}_${MODEL_CONFIG}"
OUTPUT_DIR="./experiments"
BATCH_SIZE=8
GRAD_ACCUM=64                # Paper uses effective batch size 512 (8 * 64 = 512)
NUM_SAMPLES=2000000          # Same as fair/full scripts
EVAL_SAMPLES=10000
LEARNING_RATE=2.5e-4         # Paper's learning rate
NUM_EPOCHS=50                # Same as fair/full scripts
WARMUP_STEPS=10000           # Paper's warmup steps

echo "Training Configuration:"
echo "  Model: $MODEL_CONFIG"
echo "  Architecture: OLD (double position encoding)"
echo "  Position embedding: n_embd // 2 per position"
echo "  Sigmagpt Mode: $SIGMAGPT_MODE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size per Device: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Training Samples: $NUM_SAMPLES"
echo "  Eval Samples: $EVAL_SAMPLES"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "========================================="
echo "Paper Hyperparameters (arXiv 2404.09562):"
echo "  - LR=2.5e-4, BS=512, Warmup=10K, WD=0.1"
echo "========================================="
echo "Conditioning Strategy:"
echo "  Conditioning: 0-40% of sequence"
echo "  Evaluation: 100% of non-conditioning"
echo "  Max Cond Blocks: 3"
echo "  Max Eval Blocks: 2"
echo "========================================="
echo "Evaluation Mode:"
echo "  Mode: autoregressive (paper's left-to-right)"
echo "  This evaluates standard next-token prediction"
echo "========================================="

# Run training with OLD architecture and autoregressive evaluation
python3 ./train.py \
    --model_type sigmagpt \
    --model_config $MODEL_CONFIG \
    --sigmagpt_mode $SIGMAGPT_MODE \
    --sigmagpt_arch old \
    --sigmagpt_eval_mode autoregressive \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_samples $NUM_SAMPLES \
    --num_eval_samples $EVAL_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --cond_pct_min 0.0 \
    --cond_pct_max 0.4 \
    --eval_pct_min 1.0 \
    --eval_pct_max 1.0 \
    --conditioning_sampling blockwise \
    --evaluation_sampling blockwise \
    --max_cond_blocks 3 \
    --max_eval_blocks 2 \
    --ordering_mode temporal \
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
    echo "Key differences from NEW architecture:"
    echo "  - Position embedding: n_embd // 2 (not n_embd)"
    echo "  - Encodes: current position + next position"
    echo "  - Evaluation: autoregressive (left-to-right)"
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

    echo "To view metrics:"
    echo "  cat $LATEST_EXP/logs/metrics.csv | column -t -s,"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check error log: logs/slurm_${SLURM_JOB_ID}.err"
    echo "  2. Check output log: logs/slurm_${SLURM_JOB_ID}.out"
    echo "  3. Check GPU memory: nvidia-smi"
fi
echo "========================================="

exit $EXIT_CODE
