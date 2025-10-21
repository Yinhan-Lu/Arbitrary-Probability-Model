# Experiment Visualization Pipeline

A robust, reusable visualization toolkit for analyzing machine learning experiments across different model architectures (GPT, BERT, autoregressive models, custom models, etc.).

## Features

- **Model-Agnostic**: Works with any model type (DistilGPT2, BERT, custom architectures)
- **Flexible Data Loading**: Automatically detects and loads metrics from CSV and TensorBoard
- **Comprehensive Visualizations**: Training curves, loss landscapes, optimization dynamics, summary dashboards
- **Multi-Experiment Comparison**: Compare multiple experiments side-by-side
- **Easy to Use**: Simple API with both quick-plot and advanced customization options
- **Extensible**: Easy to add new metrics and plot types

## Directory Structure

```
visualization/
├── __init__.py                  # Package initialization
├── experiment_loader.py         # Data loading utilities
├── plotting.py                  # Plotting functions
├── README.md                    # This file
└── examples/                    # Example usage scripts
    ├── visualize_single_experiment.py
    ├── compare_experiments.py
    └── analyze_all_experiments.py
```

## Installation

No additional installation required! The visualization pipeline uses standard Python libraries:
- `matplotlib`
- `seaborn`
- `pandas`
- `numpy`

Install if needed:
```bash
pip install matplotlib seaborn pandas numpy
```

## Quick Start

### 1. Visualize a Single Experiment

**Quickest way:**
```python
from visualization import quick_plot

quick_plot('experiments/distilgpt2_wikipedia_full_20251020_033212')
```

**With more control:**
```python
from visualization import ExperimentLoader, ExperimentPlotter

# Load experiment
loader = ExperimentLoader('experiments/distilgpt2_wikipedia_full_20251020_033212')

# Create plotter
plotter = ExperimentPlotter(loader)

# Generate specific plots
plotter.plot_summary_dashboard(save=True)
plotter.plot_training_curves(save=True)
plotter.plot_loss_landscape(save=True)
plotter.plot_optimization_dynamics(save=True)
```

**Using example script:**
```bash
cd visualization/examples
python visualize_single_experiment.py ../../experiments/distilgpt2_wikipedia_full_20251020_033212
```

### 2. Compare Multiple Experiments

```python
from visualization import MultiExperimentLoader, ExperimentPlotter

# Load multiple experiments
experiment_dirs = [
    'experiments/distilgpt2_wikipedia_full_20251020_033212',
    'experiments/bert_wikipedia_20251021_120000',
    'experiments/custom_model_20251022_140000'
]

loader = MultiExperimentLoader(experiment_dirs)

# Create comparison plots
plotter = ExperimentPlotter(loader, output_dir='plots/comparison')
plotter.plot_summary_dashboard(save=True)
plotter.plot_training_curves(save=True)

# Compare specific metric
loss_comparison = loader.compare_metrics('train_loss')
print(loss_comparison.tail())
```

**Using example script:**
```bash
cd visualization/examples
python compare_experiments.py ../../experiments/exp1 ../../experiments/exp2
```

### 3. Analyze All Experiments

```python
from visualization import discover_experiments, MultiExperimentLoader

# Auto-discover all experiments
experiment_dirs = discover_experiments('experiments', pattern='distilgpt2_*')

# Load and analyze
loader = MultiExperimentLoader(experiment_dirs)
summaries = loader.get_all_summaries()

for exp_name, summary in summaries.items():
    print(f"{exp_name}: final loss = {summary['metrics']['train_loss']['final']:.4f}")
```

**Using example script:**
```bash
cd visualization/examples
python analyze_all_experiments.py --base-dir ../../experiments --pattern "distilgpt2_*"
```

## API Reference

### ExperimentLoader

Load and parse data from a single experiment.

```python
from visualization import ExperimentLoader

loader = ExperimentLoader('experiments/my_experiment')

# Load metrics
metrics = loader.load_metrics()  # Returns pandas DataFrame

# Get summary statistics
summary = loader.get_summary()

# Get configuration
config_value = loader.get_config_value('learning_rate')

# List checkpoints
checkpoints = loader.list_checkpoints()
```

### MultiExperimentLoader

Load and compare multiple experiments.

```python
from visualization import MultiExperimentLoader

loader = MultiExperimentLoader([
    'experiments/exp1',
    'experiments/exp2',
    'experiments/exp3'
])

# Load all metrics
all_metrics = loader.load_all_metrics()  # Dict of DataFrames

# Get all summaries
all_summaries = loader.get_all_summaries()

# Compare specific metric
comparison = loader.compare_metrics('train_loss')

# Filter experiments by config
filtered = loader.filter_experiments(model_config='distilgpt2')
```

### ExperimentPlotter

Generate visualizations.

```python
from visualization import ExperimentLoader, ExperimentPlotter

loader = ExperimentLoader('experiments/my_experiment')
plotter = ExperimentPlotter(loader, output_dir='custom_plots')

# Generate plots
plotter.plot_summary_dashboard(save=True, show=False)
plotter.plot_training_curves(metrics=['train_loss', 'learning_rate'], save=True)
plotter.plot_loss_landscape(metric='train_loss', save=True)
plotter.plot_optimization_dynamics(save=True)
```

### Available Plots

1. **Summary Dashboard**: Comprehensive overview with all key metrics
   - Training/validation loss
   - Perplexity evolution
   - Learning rate schedule
   - Gradient norm
   - Training throughput
   - Loss distribution by epoch
   - Configuration summary

2. **Training Curves**: Detailed metric evolution over training
   - Auto-detects available metrics
   - Smoothing for noisy metrics
   - Customizable metric selection

3. **Loss Landscape**: Loss analysis with epoch boundaries
   - Full loss curve with smoothing
   - Loss distribution per epoch (box plot)

4. **Optimization Dynamics**: Training optimization analysis
   - Learning rate schedule
   - Gradient norm evolution and distribution
   - Training throughput
   - Time efficiency

## Expected Experiment Structure

The pipeline expects experiments to follow this structure:

```
experiments/
└── experiment_name/
    ├── config.json                  # Experiment configuration
    ├── logs/
    │   ├── metrics.csv             # Training metrics (required)
    │   └── tensorboard/            # TensorBoard events (optional)
    └── checkpoints/                # Model checkpoints (optional)
        ├── checkpoint_step_*.pt
        ├── best_model.pt
        └── final_model.pt
```

### Required Files

- `config.json`: Experiment configuration
- `logs/metrics.csv`: Training metrics with columns:
  - `step`, `epoch`, `train_loss`, `val_loss`, `learning_rate`, `grad_norm`, etc.

### Metrics CSV Format

```csv
step,epoch,train_loss,train_perplexity,val_loss,val_perplexity,learning_rate,grad_norm,time_elapsed_seconds,tokens_per_second
10,1,33.06,228287393562624.0,,,5.22e-05,0.72,175.27,65670.94
20,1,10.25,28193.53,,,5.24e-05,0.65,350.54,66234.12
...
```

## Extending the Pipeline

### Adding Custom Metrics

1. Update your training script to log new metrics to `metrics.csv`
2. The pipeline will automatically detect and plot them

### Adding Custom Plots

```python
from visualization import ExperimentPlotter

class CustomPlotter(ExperimentPlotter):
    def plot_custom_metric(self, save=True, show=False):
        df = self.loader.load_metrics()

        # Your custom plotting code
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(df['step'], df['custom_metric'])
        ax.set_title('Custom Metric')

        if save:
            fig.savefig(self.output_dir / 'custom_metric.png', dpi=300)

        return fig

# Use it
loader = ExperimentLoader('experiments/my_experiment')
plotter = CustomPlotter(loader)
plotter.plot_custom_metric()
```

### Supporting Different Model Types

The pipeline is already model-agnostic! It works with:
- **Autoregressive models**: GPT-2, GPT-3, DistilGPT2, your custom AR models
- **Masked language models**: BERT, RoBERTa, your custom MLM models
- **Custom architectures**: Any model that logs metrics to CSV

Just ensure your training script saves metrics in the expected format.

## Examples Output

All plots are saved as high-resolution PNG files (300 DPI) in the `plots/` directory:

- `dashboard_<experiment_name>.png`: Comprehensive dashboard
- `training_curves_<experiment_name>.png`: All training curves
- `loss_landscape_<experiment_name>.png`: Loss analysis
- `optimization_dynamics_<experiment_name>.png`: Optimization details

For multi-experiment comparison:
- `comparison_dashboard.png`: Side-by-side comparison
- `training_curves_comparison.png`: Overlaid training curves

## Tips and Best Practices

1. **Use quick_plot for rapid analysis**: Great for initial exploration
2. **Use MultiExperimentLoader for comparisons**: Compare hyperparameters, architectures, etc.
3. **Filter experiments by config**: Use `filter_experiments()` to compare specific subsets
4. **Check summaries first**: Use `get_summary()` before plotting to verify data
5. **Customize output directory**: Organize plots by analysis type
6. **Save plots programmatically**: Set `save=True, show=False` for batch processing

## Troubleshooting

### "No config.json found"
- Make sure your experiment directory contains `config.json`
- The loader will still work but won't have configuration metadata

### "Metrics file not found"
- Ensure `logs/metrics.csv` exists in the experiment directory
- Check that the file path is correct

### "No data for metric"
- The metric column might be empty (all NaN values)
- Check that your training script is logging this metric

### Memory issues with many experiments
- Use `discover_experiments()` with patterns to filter
- Analyze experiments in batches
- Use `MultiExperimentLoader.filter_experiments()` to reduce the dataset

## Future Enhancements

Potential additions:
- [ ] Interactive plots with Plotly
- [ ] Real-time monitoring during training
- [ ] Automatic report generation (HTML/PDF)
- [ ] Statistical significance testing for comparisons
- [ ] Hyperparameter correlation analysis
- [ ] Checkpoint analysis (model weight statistics)
- [ ] TensorBoard integration for scalars
- [ ] W&B integration

## Contributing

To add new features:
1. Add new plotting methods to `ExperimentPlotter` class
2. Update this README with usage examples
3. Add example scripts to `examples/` directory

## License

Part of the Arbitrary Probability Model project.
