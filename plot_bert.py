import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# CHANGE THIS to your experiment folder
exp_dir = "experiments/distilbert_debug_run_20251125_160014"
csv_path = f"{exp_dir}/logs/metrics.csv"

df = pd.read_csv(csv_path)

# Separate train/eval rows
train = df[df["split"] == "train"]
val = df[df["split"] == "val"]

plt.figure(figsize=(14, 8))

# === TRAIN LOSS ===
plt.plot(train["step"], train["loss"], "-", label="Train Loss", alpha=0.7)

# === EVAL MODES ===
def plot_eval_mode(column, label, marker):
    if column in val.columns:
        plt.plot(val["step"], val[column], marker, label=label)

plot_eval_mode("mode1_loss", "Mode 1 (Eval)", "o-")
plot_eval_mode("mode2_loss", "Mode 2 (Eval)", "s-")
plot_eval_mode("mode3_loss", "Mode 3 (Eval)", "^-")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss + DistilBERT Evaluation Modes")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save plot
plots_dir = Path(exp_dir) / "plots"
plots_dir.mkdir(exist_ok=True)

out_path = plots_dir / "train_vs_modes.png"
plt.savefig(out_path, dpi=300)

print("Saved plot:", out_path)


