#!/bin/bash

# Incremental Package Installation Script for arbprob Environment
# This script installs packages step-by-step to avoid memory issues

set -e  # Exit on error

echo "=========================================="
echo "Incremental Environment Setup for arbprob"
echo "=========================================="
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "arbprob" ]]; then
    echo "ERROR: Please activate the arbprob environment first:"
    echo "  conda activate arbprob"
    exit 1
fi

echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo ""

# Step 1: Install PyTorch
echo "=========================================="
echo "Step 1/5: Installing PyTorch (CUDA 12.1)"
echo "=========================================="
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "✓ PyTorch installed successfully"
echo ""

# Step 2: Install HuggingFace packages
echo "=========================================="
echo "Step 2/5: Installing HuggingFace packages"
echo "=========================================="
pip install transformers==4.46.3 datasets==3.1.0 tokenizers huggingface-hub
echo "✓ HuggingFace packages installed successfully"
echo ""

# Step 3: Install visualization packages
echo "=========================================="
echo "Step 3/5: Installing visualization packages"
echo "=========================================="
pip install matplotlib seaborn pillow
echo "✓ Visualization packages installed successfully"
echo ""

# Step 4: Install data processing packages
echo "=========================================="
echo "Step 4/5: Installing data processing packages"
echo "=========================================="
pip install numpy pandas pyyaml
echo "✓ Data processing packages installed successfully"
echo ""

# Step 5: Install utility packages
echo "=========================================="
echo "Step 5/5: Installing utility packages"
echo "=========================================="
pip install tqdm tensorboard ipython jupyter
echo "✓ Utility packages installed successfully"
echo ""

# Verification
echo "=========================================="
echo "Verification: Testing imports"
echo "=========================================="
python -c "
import sys

# Test non-PyTorch packages first
try:
    import transformers
    import datasets
    import matplotlib
    import numpy
    import pandas
    print('✓ Core packages imported successfully')
    print(f'Transformers version: {transformers.__version__}')
    print(f'Datasets version: {datasets.__version__}')
except Exception as e:
    print(f'✗ Error importing packages: {e}')
    sys.exit(1)

# Test PyTorch (may fail on login node without CUDA)
try:
    import torch
    print(f'✓ PyTorch version: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    if 'libnvJitLink' in str(e) or 'CUDA' in str(e):
        print('⚠ PyTorch installed but CUDA not available (normal on login node)')
        print('  Run on GPU compute node to test CUDA: sbatch scripts/test_env_gpu.sh')
    else:
        print(f'✗ Error importing PyTorch: {e}')
        sys.exit(1)
"

echo ""
echo "=========================================="
echo "Environment setup completed successfully!"
echo "=========================================="
echo ""
echo "To use this environment:"
echo "  conda activate arbprob"
echo ""
echo "To verify CUDA on GPU compute node:"
echo "  sbatch scripts/test_env_gpu.sh"
echo ""
