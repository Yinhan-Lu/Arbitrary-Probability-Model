#!/bin/bash
# Quick GPU access script
# Usage:
#   ./quick_gpu.sh              # 1 hour, 16GB (default)
#   ./quick_gpu.sh 00:30:00     # 30 minutes, 16GB
#   ./quick_gpu.sh 02:00:00 32G # 2 hours, 32GB

TIME=${1:-01:00:00}  # Default 1 hour
MEM=${2:-16G}        # Default 16GB
CPUS=${3:-4}         # Default 4 CPUs

echo "========================================"
echo "Quick GPU Access"
echo "========================================"
echo "Time requested: $TIME"
echo "Memory: $MEM"
echo "CPUs: $CPUS"
echo "========================================"
echo "Requesting GPU..."
echo ""

srun --gres=gpu:1 --cpus-per-task=$CPUS --mem=$MEM --time=$TIME --pty bash
