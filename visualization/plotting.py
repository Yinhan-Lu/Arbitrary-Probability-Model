"""
Flexible plotting utilities for experiment visualization.

Provides high-level plotting functions that work with various experiment types.
"""

# Set matplotlib backend for non-GUI environments (cluster/server)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .experiment_loader import ExperimentLoader, MultiExperimentLoader

# Set default style
if HAS_SEABORN:
    sns.set_style("whitegrid")
else:
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class ExperimentPlotter:
    """
    High-level plotting interface for experiment visualization.

    Supports:
    - Single experiment analysis
    - Multi-experiment comparison
    - Automatic metric detection and plotting
    - Customizable plot styles
    """

    def __init__(self, loader: Union[ExperimentLoader, MultiExperimentLoader],
                 output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize plotter with experiment loader.

        Args:
            loader: ExperimentLoader or MultiExperimentLoader instance
            output_dir: Directory to save plots (default: experiment_dir/plots)
        """
        self.loader = loader
        self.is_multi = isinstance(loader, MultiExperimentLoader)

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        elif not self.is_multi:
            self.output_dir = loader.exp_dir / "plots"
        else:
            self.output_dir = Path("plots")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(self, metrics: Optional[List[str]] = None,
                            save: bool = True,
                            show: bool = False,
                            figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Plot comprehensive training curves.

        Args:
            metrics: List of metrics to plot (auto-detect if None)
            save: Whether to save the plot
            show: Whether to display the plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if self.is_multi:
            return self._plot_training_curves_multi(metrics, save, show, figsize)
        else:
            return self._plot_training_curves_single(metrics, save, show, figsize)

    def _plot_training_curves_single(self, metrics: Optional[List[str]],
                                     save: bool, show: bool,
                                     figsize: Tuple[int, int]) -> plt.Figure:
        """Plot training curves for single experiment."""
        df = self.loader.load_metrics()

        # Auto-detect metrics if not provided
        if metrics is None:
            available_metrics = []
            priority_metrics = ['train_loss', 'val_loss', 'train_perplexity',
                              'val_perplexity', 'learning_rate', 'grad_norm',
                              'tokens_per_second']
            for m in priority_metrics:
                if m in df.columns and df[m].notna().any():
                    available_metrics.append(m)
            metrics = available_metrics

        # Create subplots
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]

        exp_name = self.loader.exp_dir.name

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = df[['step', metric]].dropna()

            if len(data) == 0:
                ax.text(0.5, 0.5, f'No data for {metric}',
                       ha='center', va='center', transform=ax.transAxes)
                continue

            ax.plot(data['step'], data[metric], linewidth=2, alpha=0.8)
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs Training Step')
            ax.grid(True, alpha=0.3)

            # Add smoothed line for noisy metrics
            if metric in ['train_loss', 'grad_norm', 'tokens_per_second']:
                window = min(50, len(data) // 10)
                if window > 1:
                    smoothed = data[metric].rolling(window=window, center=True).mean()
                    ax.plot(data['step'], smoothed, 'r--', linewidth=2,
                           alpha=0.6, label=f'Smoothed (window={window})')
                    ax.legend()

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        fig.suptitle(f'Training Curves: {exp_name}', fontsize=16, y=0.995)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f'training_curves_{exp_name}.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def _plot_training_curves_multi(self, metrics: Optional[List[str]],
                                    save: bool, show: bool,
                                    figsize: Tuple[int, int]) -> plt.Figure:
        """Plot training curves comparing multiple experiments."""
        all_metrics = self.loader.load_all_metrics()

        # Auto-detect common metrics
        if metrics is None:
            metric_sets = [set(df.columns) for df in all_metrics.values()]
            common_metrics = set.intersection(*metric_sets) if metric_sets else set()
            metrics = [m for m in ['train_loss', 'val_loss', 'learning_rate',
                                  'grad_norm', 'tokens_per_second']
                      if m in common_metrics]

        # Create subplots
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            for (exp_name, df), color in zip(all_metrics.items(), colors):
                if metric in df.columns:
                    data = df[['step', metric]].dropna()
                    if len(data) > 0:
                        ax.plot(data['step'], data[metric],
                               label=exp_name, linewidth=2, alpha=0.7, color=color)

            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Multi-Experiment Training Curves Comparison', fontsize=16, y=0.995)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'training_curves_comparison.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_loss_landscape(self, metric: str = 'train_loss',
                           save: bool = True, show: bool = False,
                           figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Plot loss landscape with epoch boundaries and trends.

        Args:
            metric: Metric to plot (default: 'train_loss')
            save: Whether to save the plot
            show: Whether to display the plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if self.is_multi:
            raise NotImplementedError("Loss landscape not supported for multi-experiment comparison")

        df = self.loader.load_metrics()

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in experiment data")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Full loss curve with epoch boundaries
        data = df[['step', 'epoch', metric]].dropna()
        ax1.plot(data['step'], data[metric], linewidth=1.5, alpha=0.7, label='Raw')

        # Add smoothed line
        window = min(100, len(data) // 20)
        if window > 1:
            smoothed = data[metric].rolling(window=window, center=True).mean()
            ax1.plot(data['step'], smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')

        # Add epoch boundaries
        epoch_changes = data[data['epoch'] != data['epoch'].shift()]['step']
        for step in epoch_changes[1:]:
            ax1.axvline(x=step, color='gray', linestyle='--', alpha=0.5)

        ax1.set_xlabel('Training Step')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title(f'{metric.replace("_", " ").title()} Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Loss per epoch (box plot)
        epoch_data = [group[metric].values for _, group in data.groupby('epoch')]
        epochs = sorted(data['epoch'].unique())

        bp = ax2.boxplot(epoch_data, labels=epochs, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric.replace('_', ' ').title())
        ax2.set_title(f'{metric.replace("_", " ").title()} Distribution per Epoch')
        ax2.grid(True, alpha=0.3, axis='y')

        exp_name = self.loader.exp_dir.name
        fig.suptitle(f'Loss Landscape: {exp_name}', fontsize=16)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f'loss_landscape_{exp_name}.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_optimization_dynamics(self, save: bool = True, show: bool = False,
                                   figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """
        Plot optimization dynamics (learning rate, gradient norm, throughput).

        Args:
            save: Whether to save the plot
            show: Whether to display the plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if self.is_multi:
            raise NotImplementedError("Optimization dynamics not supported for multi-experiment")

        df = self.loader.load_metrics()

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 2, figure=fig)

        exp_name = self.loader.exp_dir.name

        # Plot 1: Learning rate schedule
        if 'learning_rate' in df.columns:
            ax1 = fig.add_subplot(gs[0, :])
            data = df[['step', 'learning_rate']].dropna()
            ax1.plot(data['step'], data['learning_rate'], linewidth=2)
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Learning Rate')
            ax1.set_title('Learning Rate Schedule')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')

        # Plot 2: Gradient norm
        if 'grad_norm' in df.columns:
            ax2 = fig.add_subplot(gs[1, 0])
            data = df[['step', 'grad_norm']].dropna()
            ax2.plot(data['step'], data['grad_norm'], linewidth=1, alpha=0.6)

            # Add smoothed line
            window = min(50, len(data) // 10)
            if window > 1:
                smoothed = data['grad_norm'].rolling(window=window, center=True).mean()
                ax2.plot(data['step'], smoothed, 'r-', linewidth=2, label='Smoothed')

            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Gradient Norm')
            ax2.set_title('Gradient Norm Evolution')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        # Plot 3: Gradient norm histogram
        if 'grad_norm' in df.columns:
            ax3 = fig.add_subplot(gs[1, 1])
            data = df['grad_norm'].dropna()
            # Filter out inf and very large values
            data_clean = data[np.isfinite(data)]
            if len(data_clean) > 0:
                ax3.hist(data_clean, bins=50, alpha=0.7, edgecolor='black')
                ax3.axvline(x=data_clean.median(), color='r', linestyle='--',
                           linewidth=2, label=f'Median: {data_clean.median():.3f}')
                ax3.set_xlabel('Gradient Norm')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Gradient Norm Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3, axis='y')
            else:
                ax3.text(0.5, 0.5, 'No finite gradient norm values',
                        ha='center', va='center', transform=ax3.transAxes)

        # Plot 4: Throughput
        if 'tokens_per_second' in df.columns:
            ax4 = fig.add_subplot(gs[2, 0])
            data = df[['step', 'tokens_per_second']].dropna()
            ax4.plot(data['step'], data['tokens_per_second'], linewidth=1, alpha=0.6)

            # Add smoothed line
            window = min(50, len(data) // 10)
            if window > 1:
                smoothed = data['tokens_per_second'].rolling(window=window, center=True).mean()
                ax4.plot(data['step'], smoothed, 'r-', linewidth=2, label='Smoothed')

            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Tokens/Second')
            ax4.set_title('Training Throughput')
            ax4.grid(True, alpha=0.3)
            ax4.legend()

        # Plot 5: Time per step
        if 'time_elapsed_seconds' in df.columns:
            ax5 = fig.add_subplot(gs[2, 1])
            df_temp = df[['step', 'time_elapsed_seconds']].dropna()
            time_per_step = df_temp['time_elapsed_seconds'].diff() / df_temp['step'].diff()
            steps = df_temp['step'].iloc[1:]

            ax5.plot(steps, time_per_step.iloc[1:], linewidth=1, alpha=0.6)

            # Add smoothed line
            window = min(50, len(time_per_step) // 10)
            if window > 1:
                smoothed = time_per_step.rolling(window=window, center=True).mean()
                ax5.plot(steps, smoothed.iloc[1:], 'r-', linewidth=2, label='Smoothed')

            ax5.set_xlabel('Training Step')
            ax5.set_ylabel('Seconds per Step')
            ax5.set_title('Time Efficiency')
            ax5.grid(True, alpha=0.3)
            ax5.legend()

        fig.suptitle(f'Optimization Dynamics: {exp_name}', fontsize=16, y=0.995)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f'optimization_dynamics_{exp_name}.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_summary_dashboard(self, save: bool = True, show: bool = False,
                              figsize: Tuple[int, int] = (18, 12)) -> plt.Figure:
        """
        Create comprehensive summary dashboard with all key metrics.

        Args:
            save: Whether to save the plot
            show: Whether to display the plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if self.is_multi:
            return self._plot_comparison_dashboard(save, show, figsize)

        df = self.loader.load_metrics()
        config = self.loader.config
        summary = self.loader.get_summary()

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        exp_name = self.loader.exp_dir.name

        # Plot 1: Training & Validation Loss
        ax1 = fig.add_subplot(gs[0, :2])
        if 'train_loss' in df.columns:
            data = df[['step', 'train_loss']].dropna()
            ax1.plot(data['step'], data['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in df.columns:
            data = df[['step', 'val_loss']].dropna()
            if len(data) > 0:
                ax1.plot(data['step'], data['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Summary Statistics (text)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        summary_text = f"""Experiment Summary

Model: {config.get('model_config', 'N/A')}
Dataset: {config.get('dataset_name', 'N/A')}

Total Steps: {summary.get('total_steps', 0):,}
Total Epochs: {summary.get('total_epochs', 0)}
Duration: {summary.get('training_duration_hours', 0):.2f}h

Initial Loss: {summary['metrics'].get('train_loss', {}).get('initial', 0):.4f}
Final Loss: {summary['metrics'].get('train_loss', {}).get('final', 0):.4f}
Min Loss: {summary['metrics'].get('train_loss', {}).get('min', 0):.4f}

Batch Size: {config.get('batch_size', 'N/A')}
Learning Rate: {config.get('learning_rate', 'N/A')}
Grad Accum: {config.get('gradient_accumulation_steps', 'N/A')}
"""
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Plot 3: Perplexity
        ax3 = fig.add_subplot(gs[1, :2])
        if 'train_perplexity' in df.columns:
            data = df[['step', 'train_perplexity']].dropna()
            # Clip extreme values for better visualization
            data_clipped = data[data['train_perplexity'] < data['train_perplexity'].quantile(0.99)]
            ax3.plot(data_clipped['step'], data_clipped['train_perplexity'],
                    label='Train Perplexity', linewidth=2)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Perplexity')
        ax3.set_title('Perplexity Evolution')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Learning Rate
        ax4 = fig.add_subplot(gs[1, 2])
        if 'learning_rate' in df.columns:
            data = df[['step', 'learning_rate']].dropna()
            ax4.plot(data['step'], data['learning_rate'], linewidth=2, color='orange')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('LR Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        # Plot 5: Gradient Norm
        ax5 = fig.add_subplot(gs[2, 0])
        if 'grad_norm' in df.columns:
            data = df[['step', 'grad_norm']].dropna()
            window = min(50, len(data) // 10)
            smoothed = data['grad_norm'].rolling(window=window, center=True).mean()
            ax5.plot(data['step'], smoothed, linewidth=2, color='green')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Gradient Norm')
        ax5.set_title('Gradient Norm (Smoothed)')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Throughput
        ax6 = fig.add_subplot(gs[2, 1])
        if 'tokens_per_second' in df.columns:
            data = df[['step', 'tokens_per_second']].dropna()
            window = min(50, len(data) // 10)
            smoothed = data['tokens_per_second'].rolling(window=window, center=True).mean()
            ax6.plot(data['step'], smoothed, linewidth=2, color='purple')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Tokens/Second')
        ax6.set_title('Training Throughput')
        ax6.grid(True, alpha=0.3)

        # Plot 7: Loss Distribution by Epoch
        ax7 = fig.add_subplot(gs[2, 2])
        if 'train_loss' in df.columns and 'epoch' in df.columns:
            data = df[['epoch', 'train_loss']].dropna()
            epochs = sorted(data['epoch'].unique())
            epoch_data = [group['train_loss'].values for _, group in data.groupby('epoch')]
            ax7.boxplot(epoch_data, labels=epochs, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral', alpha=0.7))
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Loss')
        ax7.set_title('Loss per Epoch')
        ax7.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'Experiment Dashboard: {exp_name}', fontsize=18, y=0.995)

        if save:
            save_path = self.output_dir / f'dashboard_{exp_name}.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def _plot_comparison_dashboard(self, save: bool, show: bool,
                                   figsize: Tuple[int, int]) -> plt.Figure:
        """Create comparison dashboard for multiple experiments."""
        all_metrics = self.loader.load_all_metrics()
        all_summaries = self.loader.get_all_summaries()

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))

        # Plot 1: Loss comparison
        ax1 = fig.add_subplot(gs[0, 0])
        for (exp_name, df), color in zip(all_metrics.items(), colors):
            if 'train_loss' in df.columns:
                data = df[['step', 'train_loss']].dropna()
                ax1.plot(data['step'], data['train_loss'],
                        label=exp_name, linewidth=2, alpha=0.7, color=color)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Final loss comparison (bar chart)
        ax2 = fig.add_subplot(gs[0, 1])
        exp_names = []
        final_losses = []
        for exp_name, summary in all_summaries.items():
            if 'train_loss' in summary.get('metrics', {}):
                exp_names.append(exp_name[:20])  # Truncate long names
                final_losses.append(summary['metrics']['train_loss']['final'])
        ax2.barh(exp_names, final_losses, color=colors[:len(exp_names)])
        ax2.set_xlabel('Final Training Loss')
        ax2.set_title('Final Loss Comparison')
        ax2.grid(True, alpha=0.3, axis='x')

        # Plot 3: Learning rate comparison
        ax3 = fig.add_subplot(gs[1, 0])
        for (exp_name, df), color in zip(all_metrics.items(), colors):
            if 'learning_rate' in df.columns:
                data = df[['step', 'learning_rate']].dropna()
                ax3.plot(data['step'], data['learning_rate'],
                        label=exp_name, linewidth=2, alpha=0.7, color=color)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule Comparison')
        ax3.set_yscale('log')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Training efficiency
        ax4 = fig.add_subplot(gs[1, 1])
        exp_names = []
        throughputs = []
        for exp_name, summary in all_summaries.items():
            if 'tokens_per_second' in summary.get('metrics', {}):
                exp_names.append(exp_name[:20])
                throughputs.append(summary['metrics']['tokens_per_second']['mean'])
        ax4.barh(exp_names, throughputs, color=colors[:len(exp_names)])
        ax4.set_xlabel('Avg Tokens/Second')
        ax4.set_title('Training Efficiency Comparison')
        ax4.grid(True, alpha=0.3, axis='x')

        fig.suptitle('Multi-Experiment Comparison Dashboard', fontsize=18, y=0.995)

        if save:
            save_path = self.output_dir / 'comparison_dashboard.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


def quick_plot(experiment_dir: Union[str, Path],
              output_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Quick plot function for rapid visualization.

    Args:
        experiment_dir: Path to experiment directory
        output_dir: Directory to save plots (default: experiment_dir/plots)
    """
    loader = ExperimentLoader(experiment_dir)
    plotter = ExperimentPlotter(loader, output_dir)

    print(f"\nGenerating plots for {loader.exp_dir.name}...")

    # Generate all plots
    plotter.plot_summary_dashboard()
    plotter.plot_training_curves()
    plotter.plot_loss_landscape()
    plotter.plot_optimization_dynamics()

    print(f"\nAll plots saved to {plotter.output_dir}")
