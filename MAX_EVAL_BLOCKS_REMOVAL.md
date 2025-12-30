# max_eval_blocks Parameter Removal Summary

## Date
December 21, 2025

## Rationale

The `max_eval_blocks` parameter was removed because it was fundamentally problematic:

1. **Insufficient Token Selection**: When `eval_pct` is high (e.g., 99%), limiting blocks to 2-3 prevents selecting enough evaluation tokens
2. **Natural Segmentation**: Evaluation blocks should be naturally determined by how conditioning blocks split the sequence
3. **No Input Impact**: Unlike conditioning blocks (which affect model input complexity), evaluation blocks don't impact memory or computation
4. **Better Gradient Signal**: More evaluation blocks = better gradient signal from diverse positions

## What Was Kept

- **`max_cond_blocks`**: Still exists and is meaningful
  - Controls input complexity to the model
  - Affects memory usage and training stability
  - Default: 5 blocks

## What Was Removed

- **`--max_eval_blocks`** command-line argument from `train.py`
- All references to `max_eval_blocks` in trainers:
  - ConditionalTrainer
  - SigmaGPTTrainer
  - DistilBertTrainer
- All references in scripts:
  - generate_eval_splits.py
  - train_sigmagpt_mps_test.py
  - All sweep scripts (submit_sweep_*.sh, submit_cond_pct_sweep.sh)
  - All converge scripts
  - All comparison scripts
  - All sigmagpt scripts
- All references in tests:
  - test_sigmagpt_old.py
  - test_sigmagpt_5mode_eval.py
  - test_prefix_conditioning.py

## Changes Made

### 1. Core Training Files
- `train.py`: Removed `--max_eval_blocks` argument
- `train/conditional_trainer.py`: Use `uniform_num_blocks_distribution` directly
- `train/sigmagpt_trainer.py`: Use `uniform_num_blocks_distribution` directly
- `train/distilbert_trainer.py`: Simplified SimpleAugmenter to not use max_eval_blocks

### 2. Scripts (13 files updated)
- `scripts/generate_eval_splits.py`
- `scripts/train_sigmagpt_mps_test.py`
- `scripts/submit_sweep_sigmagpt_scramble.sh`
- `scripts/submit_sweep_sigmagpt_temporal.sh`
- `scripts/submit_sweep_conditional.sh`
- `scripts/submit_cond_pct_sweep.sh` (2 locations)
- `scripts/submit_sigmagpt_temporal.sh`
- `scripts/submit_sigmagpt_scramble.sh`
- `scripts/submit_sigmagpt_quick.sh`
- `scripts/submit_sigmagpt_old_paper.sh`
- `scripts/submit_sigmagpt_full.sh`
- `scripts/submit_sigmagpt_fair.sh`
- `scripts/submit_converge_sigmagpt_temporal.sh`
- `scripts/submit_converge_sigmagpt_scramble.sh`
- `scripts/submit_comparison_sigmagpt_temporal.sh`
- `scripts/submit_comparison_sigmagpt_scramble.sh`
- `scripts/generate_eval_splits_slurm.sh`

### 3. Tests (3 files updated)
- `tests/test_sigmagpt_old.py`
- `tests/test_sigmagpt_5mode_eval.py`
- `tests/test_prefix_conditioning.py`

### 4. Test Suite
- `train/blockwise_sampling.py`: Updated Test 5 to only test conditioning block limits

## Behavior Changes

### Before
```python
# Conditioning: limited to max_cond_blocks (e.g., 5)
# Evaluation: limited to max_eval_blocks (e.g., 3) ← PROBLEMATIC
```

**Problem Example**:
- Sequence: 100 tokens
- Conditioning creates 4 unknown segments
- eval_pct = 99% → need 99 tokens
- max_eval_blocks = 2 → can only select ~40 tokens
- **Result**: Only 40% evaluated instead of 99%!

### After
```python
# Conditioning: limited to max_cond_blocks (e.g., 5)
# Evaluation: unlimited, naturally follows unknown segments
```

**Fixed Behavior**:
- Sequence: 100 tokens
- Conditioning creates 4 unknown segments
- eval_pct = 99% → need 99 tokens
- No block limit → selects all 99 tokens across 4 natural segments
- **Result**: 99% evaluated as intended ✓

## Verification

All tests pass:
```bash
python train/blockwise_sampling.py
# ✓ All blockwise sampling tests passed!
```

Test 5 now verifies:
- Conditioning blocks respect max_cond_blocks limit
- Evaluation blocks are unlimited

## Remaining References

Only in commented-out or archived code:
- `scripts/test_eval_bug.sh` (commented line)
- `scripts/submit_converge_distilbert.sh` (commented line)
- `model/archived/train_sigmagpt_deprecated.py` (archived)

These are intentionally left as they don't affect active code.

## Usage Going Forward

### Training with Block Limits
```bash
# Only specify max_cond_blocks
python train.py \
    --max_cond_blocks 5 \
    ...other args...
```

### Sweep Scripts
All sweep scripts now use:
```bash
--max_cond_blocks 3 \
# No max_eval_blocks!
```

## Impact

- ✅ Evaluation now works correctly with high eval_pct values
- ✅ Natural block segmentation from conditioning
- ✅ Better gradient signal from more evaluation positions
- ✅ Simpler API (one less parameter to tune)
- ✅ More intuitive behavior

## Conclusion

Removing `max_eval_blocks` fixes a fundamental design flaw where high evaluation percentages couldn't be achieved due to artificial block limits. The evaluation set now naturally follows the segmentation created by conditioning blocks, which is the correct behavior.

