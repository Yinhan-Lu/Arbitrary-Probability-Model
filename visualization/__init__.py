"""
Visualization Pipeline for Experiment Analysis

A robust, reusable visualization toolkit for analyzing machine learning experiments
across different model architectures (GPT, BERT, custom models, etc.).

Quick start:
    from visualization import quick_plot
    quick_plot('experiments/my_experiment_dir')

Advanced usage:
    from visualization import ExperimentLoader, ExperimentPlotter
    loader = ExperimentLoader('experiments/my_experiment_dir')
    plotter = ExperimentPlotter(loader)
    plotter.plot_summary_dashboard()
"""

from .experiment_loader import (
    ExperimentLoader,
    MultiExperimentLoader,
    discover_experiments
)

from .plotting import (
    ExperimentPlotter,
    quick_plot
)

__all__ = [
    'ExperimentLoader',
    'MultiExperimentLoader',
    'discover_experiments',
    'ExperimentPlotter',
    'quick_plot',
]

__version__ = '1.0.0'
