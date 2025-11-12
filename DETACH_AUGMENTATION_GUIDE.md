# Detach Augmentation Feature Guide

## Overview

The `--detach_augmentation` flag is a training-time option that controls whether gradients flow through the internal augmentation operations during training.

### Purpose

This feature was added to investigate performance differences between:
- **Legacy pipeline**: External augmentation (CPU, in DataLoader)
- **New pipeline**: Internal augmentation (GPU, in model forward)

## Background

### The Problem

During refactoring from external to internal augmentation, Mode 2 (boundary filling) evaluation showed dramatically different performance:
- **Legacy pipeline**: Mode 2 ppl = ~120 (worst performer)
- **New pipeline**: Mode 2 ppl = ~7 (best performer)

This 17x performance flip suggested a fundamental difference in how the models were trained.

### The Hypothesis

**Internal augmentation uses differentiable operations** (like `torch.where()` and tensor indexing) that are part of the computation graph. This means:

1. Gradients can flow through augmentation operations during backpropagation
2. The model might learn to exploit patterns in the augmentation process
3. This creates a different training signal compared to external augmentation

**External augmentation creates static tensors** before the forward pass, so gradients stop at the augmented tensors and don't flow through the augmentation logic.

## How It Works

### When `--detach_augmentation` is **False** (default)

```python
# Internal augmentation
aug_input_ids, aug_position_ids, aug_attention_mask, aug_labels = self._augment_batch(...)

# Use augmented tensors directly (gradients can flow)
input_ids = aug_input_ids
position_ids = aug_position_ids
attention_mask = aug_attention_mask
labels = aug_labels
```

- Augmentation operations are part of the computation graph
- Gradients flow through tensor operations like `torch.where()`, indexing, etc.
- **This is the current behavior** that may cause Mode 2 artifacts

### When `--detach_augmentation` is **True**

```python
# Internal augmentation
aug_input_ids, aug_position_ids, aug_attention_mask, aug_labels = self._augment_batch(...)

# Detach before using (block gradient flow)
input_ids = aug_input_ids.detach()
position_ids = aug_position_ids.detach()
attention_mask = aug_attention_mask.detach()
labels = aug_labels.detach()
```

- Augmented tensors are detached from the computation graph
- Gradients **cannot** flow through augmentation operations
- **Mimics legacy external augmentation behavior**

## Usage

### Basic Training Command

**Without detach (default behavior)**:
```bash
python train.py \
    --model_type conditional \
    --model_config distilgpt2 \
    --num_epochs 3 \
    ... (other arguments)
```

**With detach (legacy-like behavior)**:
```bash
python train.py \
    --model_type conditional \
    --model_config distilgpt2 \
    --detach_augmentation \
    --num_epochs 3 \
    ... (other arguments)
```

### SLURM Script Example

Modify your SLURM submission script:

```bash
#!/bin/bash
#SBATCH --job-name=train_detached
...

python train.py \
    --model_type conditional \
    --model_config distilgpt2 \
    --detach_augmentation \  # Add this line
    --cond_pct_min 0.0 \
    --cond_pct_max 0.4 \
    --eval_pct_min 1.0 \
    --eval_pct_max 1.0 \
    ...
```

## Comparison Experiment Design

To test the hypothesis, run two experiments in parallel:

### Experiment A: With Gradient Flow (Default)
```bash
sbatch scripts/submit_experiment_a.sh
# Uses: no --detach_augmentation flag
```

### Experiment B: Without Gradient Flow (Detached)
```bash
sbatch scripts/submit_experiment_b.sh
# Uses: --detach_augmentation flag
```

### What to Compare

After training both models, evaluate them and compare:

1. **Mode 2 Performance**:
   - Experiment A (no detach): Mode 2 ppl = ?
   - Experiment B (detached): Mode 2 ppl = ?
   - **Hypothesis**: Experiment B should be closer to legacy (~120)

2. **Other Modes**:
   - Mode 0, 1, 3, 4, 5 performance
   - Check if detaching affects all modes or just Mode 2

3. **Training Dynamics**:
   - Training loss curves
   - Convergence speed
   - Final training perplexity

## Expected Results

### If the hypothesis is correct:

| Experiment | detach_augmentation | Mode 2 PPL | Explanation |
|------------|---------------------|------------|-------------|
| Legacy     | N/A (external aug)  | ~120       | Baseline (correct) |
| New - A    | False (default)     | ~7         | Gradient flow artifact |
| New - B    | True (detached)     | ~120       | Fixed, matches legacy |

### If the hypothesis is incorrect:

- Experiment B's Mode 2 will still show ppl ~7
- This would indicate the issue is elsewhere (e.g., different random seed handling, padding logic, etc.)

## Implementation Details

### Files Modified

1. **`model/arbitrary_prob_gpt2.py`**:
   - Added `detach_augmentation` parameter to `GPT2Config.__init__()`
   - Added conditional detach logic in `GPT2Model.forward()` (lines 585-595)

2. **`train.py`**:
   - Added `--detach_augmentation` command-line argument (line 349-355)

3. **`train/conditional_trainer.py`**:
   - Pass `detach_augmentation` from args to config (lines 68-71)

### Performance Impact

- **Memory**: No significant difference (detach is a lightweight operation)
- **Speed**: No measurable difference in training time
- **Accuracy**: This is what we're testing!

## Testing

A test script is provided to verify the implementation:

```bash
python test_detach_augmentation.py
```

This tests:
- Config parameter can be set correctly
- Model initialization works with both values
- Forward and backward passes work correctly

## Troubleshooting

### Issue: --detach_augmentation flag not recognized

**Solution**: Make sure you're using the updated code:
```bash
git pull origin main
git log --oneline -1  # Should show commit with detach_augmentation
```

### Issue: No performance difference between experiments

**Possible causes**:
1. The hypothesis is wrong (gradient flow is not the issue)
2. Both experiments are using the same config (check SLURM output)
3. Random seed differences are dominating the effect

**Next steps**: Check other potential causes (padding, random seeds, etc.)

### Issue: Training fails with detach_augmentation=True

This shouldn't happen, but if it does:
1. Check the test script passes: `python test_detach_augmentation.py`
2. Check the error message for clues
3. Report the issue with full error traceback

## Additional Notes

### Backward Compatibility

- Default value is `False` (current behavior)
- Existing training scripts work without modification
- Only affects training if explicitly enabled

### When to Use

Use `--detach_augmentation` when:
- ✓ Investigating Mode 2 performance artifacts
- ✓ Trying to reproduce legacy pipeline results
- ✓ Debugging training dynamics
- ✓ Comparing augmentation strategies

Don't use if:
- ✗ You want the current default behavior
- ✗ You're not investigating the Mode 2 issue
- ✗ You're running baseline (non-conditional) models

## References

- Issue investigation: Mode 2 performance flip between pipelines
- Related commits: [list relevant commits]
- Debug script: `debug_mode2_comparison.py`
- Test script: `test_detach_augmentation.py`

---

**Last Updated**: 2025-01-12
**Status**: Experimental feature for debugging
**Contact**: Check project documentation for questions
