#!/bin/bash

# One-click sanity check script for GPT-2 implementation
# This script runs all sanity checks to verify the implementation is correct

set -e  # Exit on error

echo "======================================================================"
echo "GPT-2 Implementation Sanity Check"
echo "======================================================================"
echo ""

# Check if we're in the right directory
if [ ! -d "model" ] || [ ! -d "train" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs/sanity
mkdir -p checkpoints/sanity

# Check Python and required packages
echo ""
echo "Checking Python environment..."
python3 --version

echo ""
echo "Checking required packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "Error: PyTorch not installed. Please install with: pip install torch"
    exit 1
}

python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || {
    echo "Error: Transformers not installed. Please install with: pip install transformers"
    exit 1
}

python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')" || {
    echo "Error: Datasets not installed. Please install with: pip install datasets"
    exit 1
}

python3 -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')" || {
    echo "Warning: Matplotlib not installed. Plotting may not work. Install with: pip install matplotlib"
}

# Run sanity checks
echo ""
echo "======================================================================"
echo "Running Sanity Checks"
echo "======================================================================"
echo ""

# M0: Model instantiation
echo "----------------------------------------------------------------------"
echo "M0: Testing model instantiation and forward pass..."
echo "----------------------------------------------------------------------"
cd "$(dirname "$0")/.." || exit 1

python3 tests/sanity.py --test m0 --config tiny || {
    echo "✗ M0 failed"
    exit 1
}

echo ""
echo "----------------------------------------------------------------------"
echo "M1: Testing data loading and tokenization..."
echo "----------------------------------------------------------------------"
python3 tests/sanity.py --test m1 --config tiny || {
    echo "✗ M1 failed"
    exit 1
}

echo ""
echo "----------------------------------------------------------------------"
echo "M2: Testing training loop and loss convergence..."
echo "----------------------------------------------------------------------"
python3 tests/sanity.py --test m2 --config tiny || {
    echo "✗ M2 failed"
    exit 1
}

echo ""
echo "======================================================================"
echo "✓ ALL SANITY CHECKS PASSED!"
echo "======================================================================"
echo ""
echo "Your GPT-2 implementation is ready for training."
echo ""
echo "Next steps:"
echo "  1. For CPU testing with tiny model:"
echo "     python scripts/train.py --config tiny --max_epochs 1 --num_train_samples 100"
echo ""
echo "  2. For GPU training with DistilGPT-2:"
echo "     python scripts/train.py --config distilgpt2 --max_epochs 3 --batch_size 16"
echo ""
echo "  3. View training logs:"
echo "     cat logs/training_log.csv"
echo "     open logs/training_curves.png"
echo ""
