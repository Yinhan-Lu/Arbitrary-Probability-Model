#!/bin/bash
#SBATCH --job-name=debug_mode2
#SBATCH --output=logs/debug_mode2_%j.out
#SBATCH --error=logs/debug_mode2_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Debug script to compare Legacy vs New Pipeline Mode 2 evaluation

echo "========================================"
echo "Debug Mode 2 Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Load modules (adjust based on your cluster setup)
module load python/3.9
module load cuda/11.8

# Activate conda environment if needed
# source ~/miniconda3/bin/activate your_env_name

# Run the debug script
python debug_mode2_comparison.py

echo "========================================"
echo "Debug complete! Check the output above."
echo "========================================"
