import pandas as pd
import matplotlib.pyplot as plt

exp_dir = "experiments/distilbert_medium_run_20251114_135003"
csv_path = f"{exp_dir}/logs/metrics.csv"

df = pd.read_csv(csv_path)

train = df[df["split"] == "train"]
val = df[df["split"] == "val"]

plt.figure(figsize=(10, 6))

# (Optional) smooth train loss with rolling mean if you have many steps
# train_loss_smooth = train["loss"].rolling(window=5, min_periods=1).mean()
# plt.plot(train["step"], train_loss_smooth, label="Train Loss (smoothed)")

# For now just plot raw train loss
plt.plot(train["step"], train["loss"], "-o", label="Train Loss")

# Plot eval loss as red squares
if len(val) > 0:
    plt.plot(val["step"], val["loss"], "s-", label="Eval Loss")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss vs Evaluation Loss (DistilBERT debug)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(f"{exp_dir}/plots/train_vs_val_loss_simple.png", dpi=300)
print("Saved to", f"{exp_dir}/plots/train_vs_val_loss_simple.png")
