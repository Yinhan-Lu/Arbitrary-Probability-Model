# Visualization Pipeline - Quick Start

## 30-Second Quick Start

```python
from visualization import quick_plot
quick_plot('experiments/distilgpt2_wikipedia_full_20251020_033212')
```

That's it! All plots will be saved to `experiments/<name>/plots/`

## 2-Minute Quick Start

```bash
# From the project root
python3 quickstart_visualization.py experiments/distilgpt2_wikipedia_full_20251020_033212
```

Output:
```
Generating plots for distilgpt2_wikipedia_full_20251020_033212...
Saved plot to experiments/.../plots/dashboard_....png
Saved plot to experiments/.../plots/training_curves_....png
Saved plot to experiments/.../plots/loss_landscape_....png
Saved plot to experiments/.../plots/optimization_dynamics_....png

Experiment Summary:
  Model: distilgpt2
  Dataset: wikipedia
  Total Steps: 42,210
  Total Epochs: 3
  Duration: 22.23 hours

  Training Loss:
    Initial: 33.0616
    Final: 3.2925
    Min: 2.8603
    Improvement: 90.04%

Plots saved to: experiments/distilgpt2_wikipedia_full_20251020_033212/plots
```

## What Gets Generated

**4 comprehensive visualizations:**

1. **Dashboard** (`dashboard_<name>.png`)
   - Training/validation loss curves
   - Perplexity evolution
   - Learning rate schedule
   - Gradient norms
   - Training throughput
   - Loss distribution by epoch
   - Configuration summary

2. **Training Curves** (`training_curves_<name>.png`)
   - All metrics plotted individually
   - Auto-smoothing for noisy metrics
   - Customizable metric selection

3. **Loss Landscape** (`loss_landscape_<name>.png`)
   - Loss evolution with epoch boundaries
   - Loss distribution per epoch (box plots)

4. **Optimization Dynamics** (`optimization_dynamics_<name>.png`)
   - Learning rate schedule (log scale)
   - Gradient norm evolution and distribution
   - Training throughput over time
   - Time efficiency metrics

## Common Use Cases

### 1. Quick Analysis After Training

```bash
python3 quickstart_visualization.py experiments/my_latest_experiment
```

### 2. Compare Two Models

```bash
python3 quickstart_visualization.py \
    experiments/distilgpt2_run1 \
    experiments/distilgpt2_run2 \
    --compare
```

### 3. Analyze All Experiments

```bash
python3 quickstart_visualization.py --all
```

### 4. Compare Specific Model Type

```bash
python3 quickstart_visualization.py --all --pattern "distilgpt2_*"
```

## Requirements

**Already installed:**
- matplotlib (✓ 3.10.7)
- pandas (✓ 2.3.3)
- numpy (✓ 1.26.4)

**Optional (for better styling):**
- seaborn

Install optional packages:
```bash
pip install seaborn
```

## Expected Experiment Structure

The pipeline works with experiments that have this structure:

```
experiments/
└── <experiment_name>/
    ├── config.json          # Required: experiment configuration
    └── logs/
        └── metrics.csv      # Required: training metrics
```

**metrics.csv format:**
```csv
step,epoch,train_loss,val_loss,learning_rate,grad_norm,tokens_per_second,...
10,1,33.06,,,5.22e-05,0.72,65670.94
20,1,10.25,,,5.24e-05,0.65,66234.12
...
```

## Next Steps

- **Full documentation**: See [README.md](README.md)
- **Usage guide**: See [../VISUALIZATION_GUIDE.md](../VISUALIZATION_GUIDE.md)
- **Examples**: Check [examples/](examples/) directory

## Model Compatibility

✅ Works with **any model type**:
- Autoregressive (GPT-2, GPT-3, DistilGPT2, etc.)
- Masked Language Models (BERT, RoBERTa, etc.)
- Your custom architectures
- Any model that logs to CSV

**No modifications needed for new models!**
