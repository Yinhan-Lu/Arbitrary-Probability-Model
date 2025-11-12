# Utilities (utils/)

Quick-access utility scripts for testing, visualization, and development workflows.

---

## 📋 Overview

The `utils/` folder contains standalone scripts for common development tasks:
- **Visualization:** Plotting training curves and experiment comparisons
- **Debugging:** Fast iteration scripts for development
- **Analysis:** Experiment summaries and comparisons

These are **convenience wrappers** - they don't contain core logic (which lives in `model/`, `train/`, etc.), but provide easy entry points for common tasks.

**Note:** Test scripts have been moved to `tests/` folder. See `tests/README.md` for testing utilities.

---

## 📂 Directory Structure

```
utils/
├── README.md                           # This file
├── CHEATSHEET.md                       # Quick command reference
├── quickstart_visualization.py         # Experiment visualization
├── analyze_pipeline_differences.py     # Deep analysis of pipeline differences
├── compare_experiments.py              # Statistical comparison of experiments
└── (future utilities...)
```

---

## 🚀 Available Utilities

### 1. `quickstart_visualization.py` - Experiment Visualization

**Purpose:** Visualize and compare training experiments

**Features:**
- 📊 Plot training curves (loss, perplexity, LR, gradient norms)
- 📈 Compare multiple experiments side-by-side
- 🔍 Auto-discover experiments in directory
- 📋 Generate experiment summaries

**Usage:**

```bash
# Visualize a single experiment
python utils/quickstart_visualization.py experiments/distilgpt2_wikipedia_full_20251020_033212

# Compare multiple experiments
python utils/quickstart_visualization.py experiments/exp1 experiments/exp2 --compare

# Analyze all experiments
python utils/quickstart_visualization.py --all

# Analyze experiments matching pattern
python utils/quickstart_visualization.py --all --pattern "distilgpt2_*"
```

**Outputs:**
- Loss curves (train/val)
- Perplexity over time
- Learning rate schedule
- Gradient norm tracking
- Summary statistics

**Saved to:**
- Single experiment: `experiments/*/plots/`
- Comparisons: `plots/comparison/`

**When to use:**
- After training completes
- Debugging training instabilities
- Comparing hyperparameter configurations
- Preparing figures for reports/papers

**Dependencies:**
- matplotlib
- pandas (optional, for better data handling)

---

### 2. `analyze_pipeline_differences.py` - Pipeline Comparison Analysis

**Purpose:** Deep analysis of performance differences between two training pipelines

**Features:**
- 📊 Side-by-side training curve comparisons for all modes
- 🔍 Divergence point identification (when curves start separating)
- 📈 Training loss progression analysis
- 📋 Comprehensive text summary reports
- 🎯 Automatic anomaly detection

**Usage:**

```bash
# Analyze legacy vs new pipeline
python utils/analyze_pipeline_differences.py \
    result_of_two_pipeline/legacy_pipeline \
    result_of_two_pipeline/new_pipeline \
    --output pipeline_analysis

# Compare any two experiments
python utils/analyze_pipeline_differences.py \
    experiments/exp1 \
    experiments/exp2 \
    --output comparison_results
```

**Outputs:**
- Individual mode comparison plots (mode1-5)
- All modes overview plot
- Training loss comparison
- Detailed text report with:
  - Final performance table
  - Divergence point analysis
  - Step-by-step progression
  - Statistical summaries

**Saved to:** `<output_dir>/` containing:
- `mode*_comparison.png` - Per-mode plots
- `all_modes_overview.png` - Overview visualization
- `training_loss_comparison.png` - Loss curves
- `analysis_report.txt` - Comprehensive text report

**When to use:**
- Investigating why different pipelines produce different results
- Understanding when performance divergence occurs during training
- Comparing refactored vs original implementations
- Debugging unexpected evaluation results

**Dependencies:**
- matplotlib
- pandas
- numpy
- scipy

---

### 3. `compare_experiments.py` - Statistical Comparison

**Purpose:** Detailed statistical analysis of experiment differences

**Features:**
- 📊 Comprehensive statistics (mean, std, variance, CV)
- 🔬 Statistical significance testing (t-test, effect size)
- 📈 Convergence analysis
- 🔍 Anomaly detection in training curves
- 📋 Overfitting indicators

**Usage:**

```bash
# Compare two experiments
python utils/compare_experiments.py \
    result_of_two_pipeline/legacy_pipeline \
    result_of_two_pipeline/new_pipeline

# Save results to file
python utils/compare_experiments.py \
    experiments/exp1 \
    experiments/exp2 \
    --output comparison_stats.txt
```

**Outputs (printed to console):**
- Mode-by-mode comparison table
- Training dynamics analysis
- Detailed Mode 2 progression
- Statistical significance tests
- Effect size calculations
- Overfitting warnings

**When to use:**
- Verifying statistical significance of performance differences
- Analyzing training stability
- Detecting overfitting patterns
- Checking convergence behavior
- Quantifying effect sizes

**Dependencies:**
- pandas
- numpy
- scipy (for statistical tests)

---

## 🔧 Related Scripts (Outside utils/)

While `utils/` contains convenience scripts, core testing is elsewhere:

### `train/sanity.py` - Comprehensive Sanity Check

**More thorough than `quick_test.py`:**
- M0: Model instantiation & forward pass
- M1: Data loading & tokenization
- M2: Training loop & loss convergence

**Usage:**
```bash
# Full sanity check
python train/sanity.py --test all

# Individual tests
python train/sanity.py --test m0  # Model only
python train/sanity.py --test m1  # Data only
python train/sanity.py --test m2  # Training only

# Different config
python train/sanity.py --test all --config tiny
```

**When to use:**
- Before starting a long training run
- After major architectural changes
- Debugging data pipeline issues
- ~2-3 minutes runtime

### `scripts/sanity_run.sh` - SLURM Sanity Check

Runs `train/sanity.py` on the cluster with proper resource allocation.

```bash
sbatch scripts/sanity_run.sh
```

---

## 🎯 Quick Reference

| Task | Script | Runtime |
|------|--------|---------|
| Quick model check | `python utils/quick_test.py` | ~10s |
| Full sanity check | `python train/sanity.py` | ~2-3min |
| Visualize experiment | `python utils/quickstart_visualization.py experiments/exp_name` | ~5s |
| Compare experiments (visual) | `python utils/quickstart_visualization.py exp1 exp2 --compare` | ~10s |
| Analyze pipeline differences | `python utils/analyze_pipeline_differences.py exp1 exp2` | ~15s |
| Statistical comparison | `python utils/compare_experiments.py exp1 exp2` | ~5s |
| SLURM sanity test | `sbatch scripts/sanity_run.sh` | ~5min |

---

## 📝 Adding New Utilities

When adding new utility scripts to this folder:

### 1. Naming Convention
- Use lowercase with underscores: `my_utility.py`
- Be descriptive: `debug_attention_weights.py` not `debug.py`
- Prefix by category if needed: `test_*`, `viz_*`, `bench_*`

### 2. Structure Template

```python
#!/usr/bin/env python3
"""
Brief description of what this utility does.

Usage:
    python utils/my_utility.py [args]

Examples:
    python utils/my_utility.py --option value
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="...")
    # Add arguments
    args = parser.parse_args()

    # Your utility logic here
    pass

if __name__ == "__main__":
    main()
```

### 3. Documentation Requirements
- Add docstring at top with usage examples
- Add entry to this README under "Available Utilities"
- Include in Quick Reference table
- Document dependencies if any

### 4. Best Practices
- Keep utilities **standalone** - minimal dependencies
- Use relative imports from project root
- Add `--help` argument descriptions
- Print clear success/failure messages
- Exit with proper codes (0 = success, 1 = failure)

---

## 🔍 Design Philosophy

**Utils vs Core Code:**

| `utils/` | `model/`, `train/`, `scripts/` |
|----------|--------------------------------|
| Convenience wrappers | Core implementation |
| Quick iteration | Production code |
| Interactive usage | Long-running jobs |
| Development/debugging | Training/inference |
| Fast (~seconds) | Slower (~hours) |

**Key principle:** Utilities should be **composable** - you can mix and match them with core scripts as needed.

---

## 🚫 What NOT to Put Here

- Core model implementations → `model/`
- Training logic → `train/`
- Dataset loaders → `train/dataset.py`
- SLURM job scripts → `scripts/`
- Production inference → Create separate `inference/` folder

---

## 💡 Tips

1. **Always run from project root:**
   ```bash
   # Good
   python utils/quick_test.py

   # Bad (may break imports)
   cd utils && python quick_test.py
   ```

2. **Check exit codes in scripts:**
   ```bash
   python utils/quick_test.py && echo "Ready to train!" || echo "Fix errors first"
   ```

3. **Chain utilities:**
   ```bash
   # Test, train, then visualize
   python utils/quick_test.py && \
   python train_distilgpt2.py --max_steps 100 && \
   python utils/quickstart_visualization.py experiments/latest
   ```

4. **Use in pre-commit hooks:**
   Add `python utils/quick_test.py` to `.git/hooks/pre-commit` for automatic validation

---

## 🔗 See Also

- [train/sanity.py](../train/sanity.py) - Comprehensive testing
- [scripts/](../scripts/) - SLURM job submission scripts
- [visualization/](../visualization/) - Full visualization library
- [implementation_record/](../implementation_record/) - Implementation documentation

---

**Last updated:** October 21, 2025
