from visualization import ExperimentLoader, ExperimentPlotter

if __name__ == "__main__":
    exp_dir = "experiments/distilbert_debug_256_v2_20251114_132749"  
    loader = ExperimentLoader(exp_dir)
    plotter = ExperimentPlotter(loader)
    plotter.plot_training_curves(metrics=["loss"])
