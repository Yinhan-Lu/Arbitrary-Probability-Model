#!/bin/bash
#SBATCH --job-name=diffusion_baseline
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1

# Diffusion Model Baseline Training
# Model: MDLM-style discrete diffusion with GPT-2 backbone
# Training time: ~2-3 hours for full training

echo "========================================="
echo "DIFFUSION MODEL BASELINE TRAINING"
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

# Change to submission directory
cd $SLURM_SUBMIT_DIR

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
EXP_NAME="diffusion_distilgpt2"
OUTPUT_DIR="./experiments"
MODEL_CONFIG="distilgpt2"
BATCH_SIZE=8
GRAD_ACCUM=16
NUM_SAMPLES=200000
EVAL_SAMPLES=10000
LEARNING_RATE=5e-4
NUM_EPOCHS=5

# Diffusion-specific parameters
NUM_DIFFUSION_STEPS=1000
NOISE_SCHEDULE="cosine"
TIME_EMB_TYPE="sinusoidal"
NUM_NLL_SAMPLES=10

echo "Training Configuration:"
echo "  Model Type: diffusion"
echo "  Model: $MODEL_CONFIG"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size per Device: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Training Samples: $NUM_SAMPLES"
echo "  Eval Samples: $EVAL_SAMPLES"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Diffusion Timesteps: $NUM_DIFFUSION_STEPS"
echo "  Noise Schedule: $NOISE_SCHEDULE"
echo "  Time Embedding: $TIME_EMB_TYPE"
echo "  NLL Samples (eval): $NUM_NLL_SAMPLES"
echo "  Mixed Precision: Enabled"
echo "  Expected Time: ~2-3 hours"
echo "========================================="

# Run training with evaluation
python3 train.py \
    --model_type diffusion \
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
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --num_diffusion_steps $NUM_DIFFUSION_STEPS \
    --noise_schedule $NOISE_SCHEDULE \
    --time_emb_type $TIME_EMB_TYPE \
    --num_nll_samples $NUM_NLL_SAMPLES \
    --conditioning_sampling blockwise \
    --evaluation_sampling blockwise \
    --cond_pct_min 0.0 \
    --cond_pct_max 0.4 \
    --eval_pct_min 0.2 \
    --eval_pct_max 0.4 \
    --mode2_boundary_cond_pct_min 0.1 \
    --mode2_boundary_cond_pct_max 0.3 \
    --logging_steps 100 \
    --save_steps 1000 \
    --do_eval \
    --fp16 \
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
    echo "Training completed successfully!"
    echo ""
    echo "Results:"
    echo "  Experiment directory: $OUTPUT_DIR/$EXP_NAME*"
    echo "  Checkpoints: $OUTPUT_DIR/$EXP_NAME*/checkpoints/"
    echo "  Logs: $OUTPUT_DIR/$EXP_NAME*/logs/"
    echo ""
    echo "To view training progress:"
    echo "  tail -f $OUTPUT_DIR/$EXP_NAME*/logs/training.log"
    echo ""
    echo "Model Info:"
    echo "  - Uses same GPT-2 backbone as conditional model"
    echo "  - Bidirectional attention (not causal)"
    echo "  - Timestep conditioning via sinusoidal embedding"
    echo "  - MDLM-style absorbing state diffusion"
else
    echo "Training failed with exit code $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check error log: logs/slurm_${SLURM_JOB_ID}.err"
    echo "  2. Check output log: logs/slurm_${SLURM_JOB_ID}.out"
    echo "  3. Check GPU memory: nvidia-smi"
    echo "  4. Verify data loading: ls experiments/"
fi
echo "========================================="

exit $EXIT_CODE
