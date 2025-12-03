# Sigma GPT Training Guide

This guide explains how to train Sigma GPT models using the SLURM scripts.

---

## Quick Start

### 1. Quick Test (Recommended First Step)

Run a quick test to verify everything works:

```bash
sbatch scripts/submit_sigmagpt_quick.sh
```

**Duration**: ~30-60 minutes
**Purpose**: Verify training pipeline, GPU memory, data loading
**Output**: `experiments/sigmagpt_quick_test_*/`

### 2. Full Training

Once the quick test passes, run full-scale training:

**Fair Mode** (~40% learning efficiency):
```bash
sbatch scripts/submit_sigmagpt_fair.sh
```

**Full Mode** (100% learning efficiency):
```bash
sbatch scripts/submit_sigmagpt_full.sh
```

**Duration**: ~5-7 days on A100
**Output**: `experiments/sigmagpt_{fair|full}_distilgpt2_*/`

---

## Training Scripts

### `submit_sigmagpt_quick.sh`
- **Purpose**: Quick test (~1 hour)
- **Samples**: 1,000 training samples
- **Epochs**: 1
- **Batch Size**: 4 × 2 = 8 effective
- **Use Case**: Verify pipeline before full training

### `submit_sigmagpt_fair.sh`
- **Purpose**: Fair mode training (~40% learning efficiency)
- **Samples**: 2,000,000 training samples
- **Epochs**: 50
- **Batch Size**: 8 × 64 = 512 effective
- **Use Case**: Baseline comparison with existing conditional model

### `submit_sigmagpt_full.sh`
- **Purpose**: Full mode training (100% learning efficiency)
- **Samples**: 2,000,000 training samples
- **Epochs**: 50
- **Batch Size**: 8 × 64 = 512 effective
- **Use Case**: Maximum performance, comparison with fair mode

---

## Training Configuration

All scripts use hyperparameters from the Sigma GPT paper (ArXiv 2404.09562):

| Parameter | Value | Source |
|-----------|-------|--------|
| Learning Rate | 2.5e-4 | Sigma GPT paper |
| Batch Size | 512 (effective) | Sigma GPT paper |
| Weight Decay | 0.1 | Sigma GPT / GPT-2 |
| Adam Beta | (0.9, 0.95) | Sigma GPT paper |
| Warmup Steps | 10,000 | Large-scale training |
| Gradient Clip | 1.0 | Standard |

### Model Configuration

- **Default**: `distilgpt2` (~82M parameters)
- **Architecture**: Sigma GPT with double position encoding
- **Dataset**: WikiText-103
- **Max Sequence Length**: 1024 tokens

### Fair vs Full Mode

**Fair Mode**:
- Only evaluation tokens contribute to loss
- ~40% learning efficiency compared to standard autoregressive
- Useful for fair comparison with existing conditional models
- Faster training (less gradient computation)

**Full Mode**:
- All tokens contribute to loss
- 100% learning efficiency (same as standard autoregressive)
- Maximum model performance
- Slightly slower training (more gradient computation)

---

## Monitoring Training

### Check Job Status

```bash
# List your jobs
squeue -u $USER

# Check specific job
squeue -j <job_id>
```

### View Logs

```bash
# Training output
tail -f logs/slurm_<job_id>.out

# Error messages
tail -f logs/slurm_<job_id>.err

# Training metrics
cat experiments/sigmagpt_*/logs/training_log.csv
```

### Experiment Directory Structure

```
experiments/sigmagpt_fair_distilgpt2_<timestamp>/
├── config.json                    # Training configuration
├── logs/
│   └── training_log.csv          # Step-by-step metrics
├── checkpoint-5000/
│   └── model.pt                  # Checkpoint at step 5000
├── checkpoint-10000/
│   └── model.pt                  # Checkpoint at step 10000
└── ...
```

---

## Customization

### Modify Hyperparameters

Edit the SLURM script before submission:

```bash
# Example: Change batch size
BATCH_SIZE=16           # Increase per-GPU batch size
GRAD_ACCUM=32          # Reduce gradient accumulation

# Example: Change learning rate
LEARNING_RATE=3e-4     # Increase learning rate

# Example: Train for fewer epochs
NUM_EPOCHS=20          # Reduce from 50 to 20
```

### Use Different Model Size

Edit `MODEL_CONFIG` in the script:

```bash
MODEL_CONFIG="gpt2"              # GPT-2 small (124M params)
MODEL_CONFIG="gpt2-medium"       # GPT-2 medium (355M params)
MODEL_CONFIG="gpt2-large"        # GPT-2 large (774M params)
```

**Note**: Larger models require more GPU memory. Adjust `BATCH_SIZE` accordingly.

### Use Different Dataset

Edit dataset parameters in the script:

```bash
# Example: Use different WikiText version
DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"  # Smaller dataset

# Then modify the training script call:
python3 ./train_sigmagpt.py \
    --dataset_name $DATASET_NAME \
    --dataset_config $DATASET_CONFIG \
    ...
```

---

## Resource Requirements

### GPU Memory

| Batch Size | Gradient Accum | Effective BS | GPU Memory | Recommended GPU |
|------------|----------------|--------------|------------|-----------------|
| 4 | 2 | 8 | ~8GB | RTX 3090, V100 |
| 8 | 16 | 128 | ~12GB | RTX 3090, V100 |
| 8 | 64 | 512 | ~12GB | A100, V100 |
| 16 | 32 | 512 | ~20GB | A100 |

**Tips**:
- Use `--fp16` for mixed precision (reduces memory by ~40%)
- Reduce `batch_size` if OOM (Out of Memory)
- Increase `gradient_accumulation_steps` to maintain effective batch size

### Storage

- Model checkpoints: ~300MB each
- Logs: ~10MB per experiment
- Total per experiment: ~5GB (with checkpoints every 5K steps)

### Time Estimates

| Configuration | GPU | Duration |
|---------------|-----|----------|
| Quick test | Any GPU | ~30-60 min |
| Fair mode (50 epochs) | A100 | ~5-7 days |
| Full mode (50 epochs) | A100 | ~5-7 days |

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: `CUDA out of memory` error

**Solutions**:
1. Reduce `BATCH_SIZE` in the script
2. Increase `GRAD_ACCUM` to maintain effective batch size
3. Reduce `max_seq_len` (if using custom config)
4. Ensure `--fp16` is enabled

Example:
```bash
BATCH_SIZE=4      # Reduce from 8
GRAD_ACCUM=128    # Increase from 64
# Effective batch size remains 512
```

### Dataset Loading Hangs

**Symptoms**: Training hangs at "Loading datasets..."

**Solutions**:
1. Check internet connection (WikiText downloads from HuggingFace)
2. Set `num_workers=0` in script
3. Pre-download dataset:
   ```bash
   python3 -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-raw-v1')"
   ```

### Training Diverges (Loss = NaN)

**Symptoms**: Loss becomes NaN after some steps

**Solutions**:
1. Reduce learning rate (e.g., from 2.5e-4 to 1e-4)
2. Increase warmup steps
3. Check gradient clipping is enabled (`max_grad_norm=1.0`)
4. Try `--fp16` off if using mixed precision

### Checkpoints Not Saving

**Symptoms**: No checkpoint directories created

**Solutions**:
1. Check disk space
2. Verify `output_dir` has write permissions
3. Check `save_steps` is set correctly
4. Look for error messages in `logs/slurm_*.err`

---

## Best Practices

### 1. Always Run Quick Test First

Before full training, run the quick test to catch issues early:
```bash
sbatch scripts/submit_sigmagpt_quick.sh
```

### 2. Monitor Training Metrics

Check training progress regularly:
```bash
# View recent logs
tail -f logs/slurm_<job_id>.out

# Plot training curves
python3 -c "
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('experiments/sigmagpt_*/logs/training_log.csv')
df['loss'].plot()
plt.savefig('loss_curve.png')
"
```

### 3. Save Checkpoints Frequently

The default `save_steps=5000` means checkpoints every ~5K steps. For long training:
- Keep at least 2-3 recent checkpoints
- Delete old checkpoints to save disk space
- Back up best checkpoint based on eval loss

### 4. Compare Fair vs Full Mode

To understand the impact of learning efficiency:
1. Train both fair and full mode
2. Evaluate on same test set
3. Compare perplexity and generation quality

---

## Citation

If you use Sigma GPT in your research, please cite:

```bibtex
@article{sigmagpt2024,
  title={Sigma GPT: Arbitrary Order Autoregressive Language Modeling with Double Position Encoding},
  journal={arXiv preprint arXiv:2404.09562},
  year={2024}
}
```

---

## Support

For issues or questions:
1. Check this guide first
2. Look at experiment logs: `logs/slurm_*.err`
3. Review training metrics: `experiments/*/logs/training_log.csv`
4. Check GPU memory: `nvidia-smi` in the SLURM output

---

**Last Updated**: 2025-11-14
**Version**: Implementation 07 (Sigma GPT Integration)
