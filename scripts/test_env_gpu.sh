#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --output=logs/test_env_%j.out
#SBATCH --error=logs/test_env_%j.err
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=main

echo "=========================================="
echo "Testing arbprob Environment on GPU Node"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Load conda
module load miniconda/3

# Activate environment
conda activate arbprob

echo "Python version: $(python --version)"
echo ""

# Test imports
echo "Testing package imports..."
python -c "
import torch
import transformers
import datasets
import matplotlib
import numpy
import pandas

print('✓ All packages imported successfully')
print('')
print('=' * 50)
print('Environment Information')
print('=' * 50)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'Current GPU: {torch.cuda.current_device()}')
else:
    print('WARNING: CUDA not available')
print(f'Transformers version: {transformers.__version__}')
print(f'Datasets version: {datasets.__version__}')
print('=' * 50)
"

echo ""
echo "=========================================="
echo "Test completed at: $(date)"
echo "=========================================="
