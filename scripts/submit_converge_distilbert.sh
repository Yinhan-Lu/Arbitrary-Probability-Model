#!/bin/bash
#SBATCH --job-name=conv_dbert
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1

# ==========================================================================
# CONVERGENCE EXPERIMENT: DistilBERT (with Early Stopping)
# ==========================================================================
# DistilBERT with MLM training and 5 evaluation modes
# Stops training when mode3_loss converges (no improvement for 5 evals)

echo "========================================="
echo "CONVERGENCE EXPERIMENT: DistilBERT"
echo "  Early stopping: patience=5 evals"
echo "  Eval frequency: every 100 steps"
echo "  MLM Training with 5 evaluation modes"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

# Print GPU information
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

# Load CUDA and conda environment (Mila cluster)
module load cuda/12.1.1/cudnn/8.9
source /cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/etc/profile.d/conda.sh
conda activate arbprob

echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "========================================="

# Training parameters (matching other convergence experiments)
EXP_NAME="converge_distilbert"
MODEL_CONFIG="distilbert-base-uncased"
BATCH_SIZE=8
GRAD_ACCUM=16
NUM_SAMPLES=1000000
EVAL_SAMPLES=10000
LEARNING_RATE=5e-4
NUM_EPOCHS=20
WEIGHT_DECAY=0.1

echo "Configuration:"
echo "  Model: $MODEL_CONFIG (66M params)"
echo "  Train Samples: $NUM_SAMPLES"
echo "  Eval Samples: $EVAL_SAMPLES"
echo "  Epochs: $NUM_EPOCHS (with early stopping)"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  MLM Masking: 15%"
echo "  Conditioning: 0-40%"
echo "  Max Cond Blocks: 3"
echo "  Max Eval Blocks: 2"
echo ""
echo "  ** Eval Steps: 100 (5x more frequent) **"
echo "  ** Early Stopping: ENABLED (patience=5) **"
echo "========================================="

# Using unified train.py pipeline
python3 ./train.py \
    --model_type distilbert \
    --model_config $MODEL_CONFIG \
    --dataset_name wikitext \
    --dataset_config wikitext-103-v1 \
    --num_train_samples $NUM_SAMPLES \
    --num_eval_samples $EVAL_SAMPLES \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 1000 \
    --early_stopping_patience 5 \
    --do_eval \
    --max_eval_batches 10 \
    --output_dir ./experiments \
    --exp_name $EXP_NAME \
    --device cuda \
    --num_workers 4
    # --weight_decay $WEIGHT_DECAY \ 
    # --warmup_steps 2000 \
    # --max_grad_norm 1.0 \
    # --adam_beta1 0.9 \
    # --adam_beta2 0.999 \
    # --cond_pct_min 0.0 \
    # --cond_pct_max 0.4 \
    # --eval_pct_min 1.0 \
    # --eval_pct_max 1.0 \
    # --conditioning_sampling blockwise \
    # --evaluation_sampling blockwise \
    # --max_cond_blocks 3 \
    # --max_eval_blocks 2 \
    # --mode2_boundary_cond_pct_min 0.1 \
    # --mode2_boundary_cond_pct_max 0.3 \

EXIT_CODE=$?

echo "========================================="
echo "Training Completed"
echo "Exit code: $EXIT_CODE"
echo "Duration: $SECONDS seconds (~$((SECONDS / 60)) minutes)"
echo "========================================="

# Auto-generate visualization plots
if [ $EXIT_CODE -eq 0 ]; then
    LATEST_EXP=$(ls -dt ./experiments/${EXP_NAME}_* 2>/dev/null | head -1)
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
        echo "Results:"
        echo "  Experiment directory: $LATEST_EXP"
        echo "  Individual plots: $LATEST_EXP/plots_individual/"
        echo "  Dashboard: $LATEST_EXP/plots/"
        echo "  CSV Metrics: $LATEST_EXP/logs/metrics.csv"
    fi
fi

exit $EXIT_CODE
