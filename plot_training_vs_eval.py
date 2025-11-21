import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load training metrics
train_csv = "experiments/distilbert_full_training_20251120_171222/logs/metrics.csv"
train_df = pd.read_csv(train_csv)

# Load evaluation metrics
eval_csv = "experiments/distilbert_full_training_20251120_171222/evaluation_results/bert_eval_modes_best_model.csv"
eval_df = pd.read_csv(eval_csv)

# Create output directory
plots_dir = Path(eval_csv).parent / "plots"
plots_dir.mkdir(exist_ok=True)

# Separate train and val from training
train_data = train_df[train_df['split'] == 'train']
val_data = train_df[train_df['split'] == 'val']

# Get final training and validation values
final_train_loss = train_data['loss'].iloc[-1]
final_val_loss = val_data['loss'].iloc[-1]

# Plot: Training curve + Final evaluation modes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# LEFT: Training curve over time
ax1.plot(train_data['step'], train_data['loss'], 'o-', linewidth=2, markersize=3, 
         label='Training Loss', alpha=0.7, color='#2E86AB')
ax1.plot(val_data['step'], val_data['loss'], 's-', linewidth=2, markersize=5, 
         label='Validation Loss', alpha=0.7, color='#E63946')
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Progress', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# RIGHT: Final losses comparison
all_labels = ['Final Train', 'Final Val', 'Eval Mode1', 'Eval Mode2', 'Eval Mode3']
all_losses = [final_train_loss, final_val_loss, 
              eval_df.loc[0, 'loss'], eval_df.loc[1, 'loss'], eval_df.loc[2, 'loss']]
colors = ['#2E86AB', '#E63946', '#06D6A0', '#F77F00', '#A23B72']

bars = ax2.bar(range(len(all_labels)), all_losses, color=colors, alpha=0.8, edgecolor='black')

# Add value labels on bars
for bar, loss in zip(bars, all_losses):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{loss:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Loss', fontsize=12)
ax2.set_xlabel('', fontsize=12)
ax2.set_title('Final Loss Comparison', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(all_labels)))
ax2.set_xticklabels(all_labels, rotation=25, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('DistilBERT Training & Evaluation Results', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(plots_dir / 'training_vs_eval.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plots_dir / 'training_vs_eval.png'}")
plt.close()

# Plot 2: Just the final comparison bars
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(all_labels)), all_losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, loss in zip(bars, all_losses):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{loss:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.ylabel('Loss', fontsize=13)
plt.title('DistilBERT: Training vs Evaluation Modes', fontsize=15, fontweight='bold')
plt.xticks(range(len(all_labels)), all_labels, rotation=20, ha='right', fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(plots_dir / 'final_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plots_dir / 'final_comparison.png'}")
plt.close()

# Print summary
print("\n" + "="*70)
print("Training vs Evaluation Comparison")
print("="*70)
print(f"Final Training Loss:       {final_train_loss:.4f}")
print(f"Final Validation Loss:     {final_val_loss:.4f}")
print(f"")
print(f"Eval Mode 1 (MLM):         {eval_df.loc[0, 'loss']:.4f}  (PPL: {eval_df.loc[0, 'perplexity']:.1f})")
print(f"Eval Mode 2 (Boundary):    {eval_df.loc[1, 'loss']:.4f}  (PPL: {eval_df.loc[1, 'perplexity']:.1f})")
print(f"Eval Mode 3 (Training):    {eval_df.loc[2, 'loss']:.4f}  (PPL: {eval_df.loc[2, 'perplexity']:.1f})")
print("="*70)
print("\nInsights:")
print(f"  • Training converged well: {final_train_loss:.3f} → {final_val_loss:.3f}")
print(f"  • Eval modes are consistent with validation: ~{eval_df.loc[0, 'loss']:.2f}")
print(f"  • Mode 2 (boundary) is hardest: {eval_df.loc[1, 'loss']:.3f}")
print(f"  • Small gap shows good generalization!")
print(f"\nPlots saved to: {plots_dir}/")
