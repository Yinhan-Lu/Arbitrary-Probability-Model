"""
Experiment data loader for flexible experiment analysis.

Supports loading metrics, configurations, and checkpoints from various experiment types.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings


class ExperimentLoader:
    """
    Load and parse experiment data from standardized experiment directories.

    Supports:
    - Multiple experiment formats (autoregressive, BERT, custom models)
    - Flexible metric loading from CSV and TensorBoard
    - Configuration parsing
    - Checkpoint metadata extraction
    """

    def __init__(self, experiment_dir: Union[str, Path]):
        """
        Initialize loader with experiment directory.

        Args:
            experiment_dir: Path to experiment directory
        """
        self.exp_dir = Path(experiment_dir)
        if not self.exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        self.config = None
        self.metrics = None
        self._load_config()

    def _load_config(self) -> None:
        """Load experiment configuration from config.json."""
        config_path = self.exp_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            warnings.warn(f"No config.json found in {self.exp_dir}")
            self.config = {}

    def load_metrics(self, metrics_file: str = "logs/metrics.csv") -> pd.DataFrame:
        """
        Load training metrics from CSV file.

        Args:
            metrics_file: Relative path to metrics CSV file

        Returns:
            DataFrame with training metrics
        """
        metrics_path = self.exp_dir / metrics_file

        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

        self.metrics = pd.read_csv(metrics_path)

        # Add derived columns if they don't exist
        self._add_derived_metrics()

        return self.metrics

    def _add_derived_metrics(self) -> None:
        """Add derived metrics for better analysis."""
        if self.metrics is None:
            return

        # Convert time to hours and minutes
        if 'time_elapsed_seconds' in self.metrics.columns:
            self.metrics['time_elapsed_hours'] = self.metrics['time_elapsed_seconds'] / 3600
            self.metrics['time_elapsed_minutes'] = self.metrics['time_elapsed_seconds'] / 60

        # Add loss improvement metrics
        if 'train_loss' in self.metrics.columns:
            self.metrics['train_loss_improvement'] = (
                self.metrics['train_loss'].iloc[0] - self.metrics['train_loss']
            )
            self.metrics['train_loss_pct_improvement'] = (
                (self.metrics['train_loss'].iloc[0] - self.metrics['train_loss']) /
                self.metrics['train_loss'].iloc[0] * 100
            )

    def get_metric_columns(self) -> List[str]:
        """Get list of available metric columns."""
        if self.metrics is None:
            self.load_metrics()
        return list(self.metrics.columns)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        if self.metrics is None:
            self.load_metrics()

        summary = {
            'experiment_name': self.exp_dir.name,
            'model_type': self.config.get('model_config', 'unknown'),
            'dataset': self.config.get('dataset_name', 'unknown'),
            'total_steps': int(self.metrics['step'].max()) if 'step' in self.metrics.columns else 0,
            'total_epochs': int(self.metrics['epoch'].max()) if 'epoch' in self.metrics.columns else 0,
        }

        # Add metric statistics
        metric_stats = {}
        for col in ['train_loss', 'val_loss', 'train_perplexity', 'val_perplexity',
                    'learning_rate', 'grad_norm', 'tokens_per_second']:
            if col in self.metrics.columns:
                data = self.metrics[col].dropna()
                if len(data) > 0:
                    metric_stats[col] = {
                        'initial': float(data.iloc[0]),
                        'final': float(data.iloc[-1]),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                    }

        summary['metrics'] = metric_stats

        # Training duration
        if 'time_elapsed_seconds' in self.metrics.columns:
            total_time = self.metrics['time_elapsed_seconds'].max()
            summary['training_duration_seconds'] = float(total_time)
            summary['training_duration_hours'] = float(total_time / 3600)

        return summary

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self.config.get(key, default)

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoint files in the experiment."""
        checkpoint_dir = self.exp_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return []
        return sorted(checkpoint_dir.glob("*.pt"))

    def get_experiment_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata including config and directory info."""
        return {
            'experiment_name': self.exp_dir.name,
            'experiment_path': str(self.exp_dir),
            'config': self.config,
            'num_checkpoints': len(self.list_checkpoints()),
            'has_metrics': (self.exp_dir / "logs/metrics.csv").exists(),
            'has_tensorboard': (self.exp_dir / "logs/tensorboard").exists(),
        }


class MultiExperimentLoader:
    """
    Load and compare multiple experiments.

    Useful for comparing different model architectures, hyperparameters, or training runs.
    """

    def __init__(self, experiment_dirs: List[Union[str, Path]]):
        """
        Initialize with multiple experiment directories.

        Args:
            experiment_dirs: List of paths to experiment directories
        """
        self.loaders = {}
        for exp_dir in experiment_dirs:
            exp_path = Path(exp_dir)
            try:
                loader = ExperimentLoader(exp_path)
                self.loaders[exp_path.name] = loader
            except Exception as e:
                warnings.warn(f"Failed to load experiment {exp_dir}: {e}")

    def load_all_metrics(self) -> Dict[str, pd.DataFrame]:
        """Load metrics from all experiments."""
        return {name: loader.load_metrics() for name, loader in self.loaders.items()}

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all experiments."""
        return {name: loader.get_summary() for name, loader in self.loaders.items()}

    def compare_metrics(self, metric_name: str) -> pd.DataFrame:
        """
        Compare a specific metric across all experiments.

        Args:
            metric_name: Name of metric to compare

        Returns:
            DataFrame with metric values aligned by step
        """
        comparison_data = {}
        for name, loader in self.loaders.items():
            metrics = loader.load_metrics()
            if metric_name in metrics.columns:
                comparison_data[name] = metrics.set_index('step')[metric_name]

        return pd.DataFrame(comparison_data)

    def filter_experiments(self, **config_filters) -> 'MultiExperimentLoader':
        """
        Filter experiments by configuration values.

        Args:
            **config_filters: Key-value pairs to filter by

        Returns:
            New MultiExperimentLoader with filtered experiments
        """
        filtered_dirs = []
        for name, loader in self.loaders.items():
            match = True
            for key, value in config_filters.items():
                if loader.get_config_value(key) != value:
                    match = False
                    break
            if match:
                filtered_dirs.append(loader.exp_dir)

        return MultiExperimentLoader(filtered_dirs)


def discover_experiments(base_dir: Union[str, Path],
                         pattern: str = "*") -> List[Path]:
    """
    Discover all experiment directories in a base directory.

    Args:
        base_dir: Base directory to search
        pattern: Glob pattern for experiment directory names

    Returns:
        List of experiment directory paths
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    experiments = []
    for exp_dir in base_path.glob(pattern):
        if exp_dir.is_dir() and (exp_dir / "config.json").exists():
            experiments.append(exp_dir)

    return sorted(experiments)
