# Evaluation Functions Added to train_conditional.py

This document summarizes all the evaluation functionality added to `train_conditional.py` to match the capabilities of `train_distilgpt2.py`.

## Summary of Changes

All changes follow the same evaluation pattern as `train_distilgpt2.py`, adapted for conditional probability training with custom attention masks.

---

## 1. New Method: `evaluate()`

**Location:** After `train_step()` method

**Purpose:** Run evaluation on validation set with conditional augmentation

**Key Features:**
- Decorated with `@torch.no_grad()` for efficiency
- Applies same conditional augmentation as training
- Computes average loss and perplexity over validation batches
- Respects `max_eval_batches` limit
- Returns detailed evaluation metrics

**Code:**
```python
@torch.no_grad()
def evaluate(self):
    """Run evaluation on validation set with conditional augmentation"""
    self.model.eval()
    total_loss = 0
    total_tokens = 0
    num_batches = 0

    logger.info("Running evaluation...")

    for batch in self.val_loader:
        # Get original input
        input_ids = batch["input_ids"].to(self.device)

        # Apply conditional augmentation (same as training)
        aug_batch = self.augmenter.augment_batch(input_ids, device=self.device)

        # Forward pass
        logits, loss = self.model(
            input_ids=aug_batch["input_ids"],
            attention_mask=aug_batch["attention_mask"],
            labels=aug_batch["labels"],
            position_ids=aug_batch["position_ids"]
        )

        total_loss += loss.item()
        total_tokens += (aug_batch["labels"] != -100).sum().item()
        num_batches += 1

        # Limit evaluation batches if specified
        if self.args.max_eval_batches > 0 and num_batches >= self.args.max_eval_batches:
            break

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    self.model.train()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_batches": num_batches,
        "total_tokens": total_tokens
    }
```

---

## 2. Updated Training Loop: Evaluation Logic

**Location:** Inside `train()` method, after logging section

**Purpose:** Periodically evaluate during training and save best model

**Key Features:**
- Evaluates every `eval_steps` steps
- Logs validation loss and perplexity
- Tracks best validation loss
- Automatically saves best model checkpoint
- Logs evaluation metrics to CSV

**Code Added:**
```python
# Evaluation
if self.args.do_eval and self.global_step % self.args.eval_steps == 0:
    eval_results = self.evaluate()

    logger.info(
        f"Evaluation at step {self.global_step} | "
        f"Val Loss: {eval_results['loss']:.4f} | "
        f"Val PPL: {eval_results['perplexity']:.2f}"
    )

    # Create evaluation metrics entry
    eval_metrics = {
        'step': self.global_step,
        'epoch': epoch + 1,
        'val_loss': eval_results["loss"],
        'val_perplexity': eval_results["perplexity"],
        'learning_rate': self.optimizer.param_groups[0]["lr"]
    }
    self._log_to_csv(eval_metrics)

    # Save best model
    if eval_results["loss"] < self.best_val_loss:
        self.best_val_loss = eval_results["loss"]
        self._save_checkpoint("best_model")
        logger.info(f"New best model saved! Val Loss: {self.best_val_loss:.4f}")
```

---

## 3. Updated CSV Logging

**Location:** `_init_csv_logger()` and `_log_to_csv()` methods

**Purpose:** Support both training and validation metrics in CSV logs

**Changes:**

### CSV Header (in `_init_csv_logger()`):
```python
writer.writerow([
    "step",
    "epoch",
    "train_loss",         # Changed from "loss"
    "train_perplexity",   # Changed from "perplexity"
    "val_loss",           # NEW
    "val_perplexity",     # NEW
    "learning_rate"
])
```

### CSV Writing (in `_log_to_csv()`):
```python
writer.writerow([
    metrics.get('step', ''),
    metrics.get('epoch', ''),
    metrics.get('train_loss', ''),      # Changed
    metrics.get('train_perplexity', ''), # Changed
    metrics.get('val_loss', ''),        # NEW
    metrics.get('val_perplexity', ''),  # NEW
    metrics.get('learning_rate', '')
])
```

**Note:** Uses `.get()` to allow partial logging (training-only or evaluation-only rows)

---

## 4. Updated Training Metrics Logging

**Location:** Inside `train()` method, logging section

**Purpose:** Match new CSV column names

**Changes:**
```python
metrics = {
    'step': self.global_step,
    'epoch': epoch + 1,
    'train_loss': avg_loss,         # Changed from 'loss'
    'train_perplexity': perplexity, # Changed from 'perplexity'
    'learning_rate': lr
}
```

---

## 5. New Command-Line Arguments

**Location:** `parse_args()` function

**Purpose:** Control evaluation behavior

**New Arguments:**
```python
parser.add_argument("--eval_steps", type=int, default=500,
                    help="Evaluate every N steps")
parser.add_argument("--max_eval_batches", type=int, default=100,
                    help="Maximum evaluation batches")
```

**Updated Arguments:**
```python
parser.add_argument("--logging_steps", type=int, default=10,
                    help="Log every N steps")  # Added help text
parser.add_argument("--save_steps", type=int, default=1000,
                    help="Save checkpoint every N steps")  # Added help text
```

---

## 6. Updated Checkpoint Saving

**Location:** `_save_checkpoint()` method

**Purpose:** Save best validation loss in checkpoints

**Change:**
```python
checkpoint = {
    "global_step": self.global_step,
    "epoch": self.epoch,
    "model_state_dict": self.model.state_dict(),
    "optimizer_state_dict": self.optimizer.state_dict(),
    "scheduler_state_dict": self.scheduler.state_dict(),
    "best_val_loss": self.best_val_loss,  # NEW
    "config": self.config.__dict__,
    "args": vars(self.args)
}
```

---

## 7. Updated Training Completion Logging

**Location:** End of `train()` method

**Purpose:** Report best validation loss at end of training

**Change:**
```python
logger.info("=" * 80)
logger.info("Training completed!")
if self.args.do_eval:  # NEW conditional
    logger.info(f"Best validation loss: {self.best_val_loss:.4f}")  # NEW
logger.info("=" * 80)
```

---

## Usage Examples

### 1. Training WITHOUT evaluation (as before):
```bash
python train_conditional.py \
    --model_config distilgpt2 \
    --num_epochs 3 \
    --batch_size 8
```

### 2. Training WITH evaluation:
```bash
python train_conditional.py \
    --model_config distilgpt2 \
    --num_epochs 3 \
    --batch_size 8 \
    --do_eval \
    --eval_steps 500 \
    --max_eval_batches 100
```

### 3. With custom evaluation frequency:
```bash
python train_conditional.py \
    --model_config distilgpt2 \
    --num_epochs 3 \
    --batch_size 8 \
    --do_eval \
    --eval_steps 250 \        # Evaluate more frequently
    --max_eval_batches 50     # Faster evaluation
```

---

## Key Differences from train_distilgpt2.py

While the evaluation structure is nearly identical, there are important differences:

1. **Conditional Augmentation**: `train_conditional.py` applies `ConditionalAugmenter` to both training and validation data, while `train_distilgpt2.py` uses standard sequences.

2. **Custom Attention Masks**: Evaluation uses the same prefix-based conditional probability setup with custom attention masks and position IDs.

3. **Loss Computation**: Only evaluation tokens contribute to loss (labels = -100 for conditioning and unknown tokens), whereas `train_distilgpt2.py` computes loss over all non-padding tokens.

4. **No TensorBoard/WandB**: `train_conditional.py` only has CSV logging, while `train_distilgpt2.py` includes TensorBoard and WandB integration.

---

## CSV Output Format

The metrics CSV now has the following columns:

| step | epoch | train_loss | train_perplexity | val_loss | val_perplexity | learning_rate |
|------|-------|------------|------------------|----------|----------------|---------------|
| 10   | 1     | 3.456      | 31.68            |          |                | 0.0005        |
| 20   | 1     | 3.234      | 25.37            |          |                | 0.0005        |
| 500  | 1     |            |                  | 2.987    | 19.81          | 0.0005        |
| 510  | 1     | 3.012      | 20.33            |          |                | 0.0005        |

**Note:** Training and evaluation metrics are logged in separate rows since they occur at different steps.

---

## Testing the Changes

To verify the evaluation functionality:

1. **Quick test with small dataset:**
```bash
python train_conditional.py \
    --model_config tiny \
    --num_train_samples 1000 \
    --num_eval_samples 100 \
    --num_epochs 1 \
    --do_eval \
    --eval_steps 50 \
    --max_eval_batches 10
```

2. **Check outputs:**
   - Look for "Running evaluation..." log messages
   - Verify validation loss and perplexity are logged
   - Check that `best_model.pt` is saved when validation improves
   - Inspect CSV file for both training and validation metrics

3. **Verify best model saving:**
```bash
# After training, check checkpoint directory
ls experiments/conditional_gpt2_*/checkpoints/
# Should see: best_model.pt, checkpoint_step_*.pt, final_model.pt
```

---

## Summary

All evaluation functionality from `train_distilgpt2.py` has been successfully adapted to `train_conditional.py`:

✅ `evaluate()` method with conditional augmentation  
✅ Periodic evaluation during training  
✅ Best model tracking and saving  
✅ Validation metrics in CSV logs  
✅ Command-line arguments for evaluation control  
✅ Checkpoint includes best_val_loss  
✅ Training completion reports best validation loss  

The implementation maintains the conditional probability training approach while adding comprehensive evaluation capabilities.
