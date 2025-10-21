# Utils Cheatsheet

Quick reference for common utility commands. Run all commands from **project root**.

---

## 🚀 Quick Start

### Before Training
```bash
# Quick model test (~10s)
python utils/quick_test.py

# Full sanity check (~3min)
python train/sanity.py --test all
```

### After Training
```bash
# Visualize single experiment
python utils/quickstart_visualization.py experiments/distilgpt2_wikipedia_full_20251020_033212

# Compare experiments
python utils/quickstart_visualization.py experiments/exp1 experiments/exp2 --compare

# Visualize all experiments
python utils/quickstart_visualization.py --all
```

---

## 📋 Testing Commands

### Quick Test (Fast)
```bash
# Basic test - all configs
python utils/quick_test.py
```

### Sanity Check (Comprehensive)
```bash
# All tests
python train/sanity.py --test all

# Individual tests
python train/sanity.py --test m0          # Model only
python train/sanity.py --test m1          # Data only
python train/sanity.py --test m2          # Training only

# Different config
python train/sanity.py --test all --config tiny
python train/sanity.py --test all --config nano
```

---

## 📊 Visualization Commands

### Single Experiment
```bash
# Basic visualization
python utils/quickstart_visualization.py experiments/your_experiment_name

# Specific experiment
python utils/quickstart_visualization.py experiments/distilgpt2_wikipedia_full_20251020_033212
```

### Multiple Experiments
```bash
# Visualize each separately
python utils/quickstart_visualization.py exp1 exp2 exp3

# Compare side-by-side
python utils/quickstart_visualization.py exp1 exp2 exp3 --compare
```

### All Experiments
```bash
# All experiments in default dir
python utils/quickstart_visualization.py --all

# Pattern matching
python utils/quickstart_visualization.py --all --pattern "distilgpt2_*"
python utils/quickstart_visualization.py --all --pattern "*_20251020_*"

# Custom base directory
python utils/quickstart_visualization.py --all --base-dir custom_experiments
```

---

## 🔧 Development Workflow

### Pre-Commit Checks
```bash
# Quick validation before commit
python utils/quick_test.py && git add . && git commit -m "your message"

# Full validation (slower)
python train/sanity.py --test all && git add . && git commit -m "your message"
```

### Training Pipeline
```bash
# Test → Train → Visualize
python utils/quick_test.py && \
sbatch scripts/submit_training.sh && \
echo "Training submitted!"

# After training completes
python utils/quickstart_visualization.py experiments/latest_experiment
```

### Debugging
```bash
# Quick model check
python utils/quick_test.py

# Check data loading
python train/sanity.py --test m1

# Mini training run (local)
python train_distilgpt2.py \
    --model_config tiny \
    --num_epochs 1 \
    --max_steps 10 \
    --batch_size 2 \
    --logging_steps 1 \
    --output_dir experiments \
    --exp_name debug_run \
    --device cuda
```

---

## 🎯 Common Tasks

### 1. Validate Changes
```bash
python utils/quick_test.py
```

### 2. Check Training Progress
```bash
# View latest experiment
ls -lt experiments/ | head -5

# Visualize most recent
EXP=$(ls -t experiments/ | head -1)
python utils/quickstart_visualization.py "experiments/$EXP"
```

### 3. Compare Hyperparameters
```bash
# Compare two runs
python utils/quickstart_visualization.py \
    experiments/run_lr_1e-4 \
    experiments/run_lr_5e-4 \
    --compare
```

### 4. Monitor SLURM Job
```bash
# Submit and get job ID
sbatch scripts/submit_training.sh
# Output: Submitted batch job 1234567

# Monitor
squeue -u $USER
tail -f logs/slurm_1234567.out

# After job completes
python utils/quickstart_visualization.py experiments/distilgpt2_*
```

---

## 📁 File Locations

| Script | Location | Purpose |
|--------|----------|---------|
| `quick_test.py` | `utils/` | Fast model validation |
| `quickstart_visualization.py` | `utils/` | Experiment visualization |
| `sanity.py` | `train/` | Comprehensive testing |
| `sanity_run.sh` | `scripts/` | SLURM sanity check |
| Training scripts | `scripts/` | SLURM job submission |

---

## ⚠️ Troubleshooting

### Import Errors
Always run from project root:
```bash
# ✅ Correct
cd /path/to/Arbitrary\ Probability\ Model
python utils/quick_test.py

# ❌ Wrong
cd utils
python quick_test.py
```

### GPU Not Found
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Force CPU for testing
python utils/quick_test.py  # Uses CPU automatically
python train/sanity.py --test all  # Detects device automatically
```

### Dataset Loading Fails
```bash
# Check internet connection
ping huggingface.co

# Try with smaller dataset
python train/sanity.py --test m1  # Uses tiny sample

# Check cache
ls ~/.cache/huggingface/datasets/
```

### Visualization No Data
```bash
# Check experiment has logs
ls experiments/your_exp_name/logs/

# Check metrics.csv exists
cat experiments/your_exp_name/logs/metrics.csv | head
```

---

## 🔗 Quick Links

- [Full Utils Documentation](README.md)
- [Sanity Check Details](../train/sanity.py)
- [SLURM Scripts](../scripts/)
- [Visualization Library](../visualization/)

---

**Tip:** Bookmark this file for quick reference!

**Last updated:** October 21, 2025
