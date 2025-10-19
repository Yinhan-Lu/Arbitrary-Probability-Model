# GPT-2 PyTorch Implementation from Scratch

A complete implementation of GPT-2 (decoder-only Transformer) in PyTorch with support for custom attention masks, designed for research on different attention mechanisms.

## Features

- **Pure PyTorch implementation** of GPT-2 architecture
- **Flexible attention mask support** for experimenting with different attention patterns
- **DistilGPT-2 configuration** using official Hugging Face parameters
- **Lightweight configurations** (tiny, nano) for CPU testing
- **Wikipedia dataset** integration for language model pre-training
- **Complete training pipeline** with logging, checkpointing, and visualization

## Architecture

The model implements the standard GPT-2 stack:
```
Input Embeddings (Token + Position)
    ↓
N × Transformer Blocks
    ├── Layer Norm
    ├── Multi-Head Self-Attention (with configurable masks)
    ├── Residual Connection
    ├── Layer Norm
    ├── Feed-Forward Network (MLP)
    └── Residual Connection
    ↓
Final Layer Norm
    ↓
Language Model Head
```

## Project Structure

```
.
├── model/
│   ├── arbitrary_prob_gpt2.py  # GPT-2 model implementation
│   └── config.py                # Model configurations (DistilGPT-2, Tiny, Nano)
├── train/
│   ├── dataset.py               # Wikipedia dataset loader
│   ├── loop.py                  # Training loop with logging
│   └── sanity.py                # Sanity check tests
├── scripts/
│   ├── train.py                 # Main training script
│   └── sanity_run.sh            # One-click sanity check
├── logs/                        # Training logs and plots (created during training)
├── checkpoints/                 # Model checkpoints (created during training)
└── README.md                    # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (for tokenizer)
- Datasets (for Wikipedia data)
- Matplotlib (for plotting)

### Install Dependencies

```bash
pip install torch transformers datasets matplotlib
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Sanity Check (Recommended First Step)

Run all tests to verify the implementation:

```bash
./scripts/sanity_run.sh
```

This will test:
- **M0**: Model instantiation and forward pass
- **M1**: Data loading and tokenization
- **M2**: Training loop and loss convergence

### 2. CPU Testing (Tiny Configuration)

Quick training on CPU with minimal resources:

```bash
python scripts/train.py \
    --config tiny \
    --max_epochs 1 \
    --num_train_samples 100 \
    --num_val_samples 20 \
    --batch_size 4 \
    --device cpu
```

### 3. GPU Training (DistilGPT-2 Configuration)

Full training on GPU with DistilGPT-2 parameters:

```bash
python scripts/train.py \
    --config distilgpt2 \
    --max_epochs 3 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --device cuda
```

## Model Configurations

### DistilGPT-2 (Official Hugging Face Parameters)

```python
vocab_size: 50257
n_layer: 6
n_head: 12
n_embd: 768
max_seq_len: 1024
dropout: 0.1
Parameters: ~82M
```

Source: [Hugging Face DistilGPT-2](https://huggingface.co/distilbert/distilgpt2)

### Tiny (CPU Testing)

```python
vocab_size: 50257
n_layer: 2
n_head: 4
n_embd: 128
max_seq_len: 256
dropout: 0.1
Parameters: ~7M
```

### Nano (Quick Debugging)

```python
vocab_size: 50257
n_layer: 1
n_head: 2
n_embd: 64
max_seq_len: 128
dropout: 0.0
Parameters: ~3M
```

## Custom Attention Masks

The model supports custom attention masks for research on different attention mechanisms:

```python
from model.arbitrary_prob_gpt2 import GPT2Model, create_causal_mask
from model.config import get_config

config = get_config("tiny")
model = GPT2Model(config)

# Default: Causal (left-to-right) mask
logits, loss = model(input_ids)

# Custom mask (e.g., bidirectional, sliding window, etc.)
custom_mask = create_your_custom_mask(seq_len)  # Shape: (batch, seq_len, seq_len)
logits, loss = model(input_ids, attention_mask=custom_mask)
```

## Training

### Basic Training

```bash
python scripts/train.py --config tiny --max_epochs 3
```

### Advanced Options

```bash
python scripts/train.py \
    --config distilgpt2 \
    --max_epochs 10 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --num_train_samples 100000 \
    --num_val_samples 5000 \
    --log_interval 100 \
    --eval_interval 1000 \
    --save_interval 5000 \
    --device cuda
```

### Resume Training

```bash
python scripts/train.py \
    --config distilgpt2 \
    --resume checkpoints/checkpoint_epoch3_step15000.pt
```

## Monitoring Training

### View Logs

```bash
# CSV log with step, epoch, loss, lr, etc.
cat logs/training_log.csv

# Training curves (loss and learning rate)
open logs/training_curves.png
```

### TensorBoard (Optional)

You can extend the trainer to add TensorBoard logging.

## Development Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M0 | Model skeleton and attention mask module | ✓ Complete |
| M1 | Tokenizer + Wikipedia data pipeline | ✓ Complete |
| M2 | Lightweight validation (tiny config) | ✓ Complete |
| M3 | Full training (DistilGPT-2 config) | Ready |
| M4 | Special attention mechanism extensions | Ready |

## Key Design Decisions

### 1. Attention Mask Interface

The model's forward pass accepts an optional `attention_mask` parameter:
- If `None` (default): Uses causal (left-to-right) mask
- If provided: Uses custom mask for specialized attention patterns

This design allows easy experimentation with different attention mechanisms without modifying the core model.

### 2. Configuration System

Three pre-defined configurations support different use cases:
- **DistilGPT-2**: Production training with official parameters
- **Tiny**: Fast iteration on CPU
- **Nano**: Ultra-fast debugging

### 3. Data Pipeline

Uses Hugging Face's `datasets` library for:
- Automatic Wikipedia dataset downloading
- Streaming support for large-scale training
- Fallback to WikiText-2 if Wikipedia unavailable

## Performance Tips

### CPU Training
- Use `--config tiny` or `--config nano`
- Reduce batch size: `--batch_size 2`
- Limit samples: `--num_train_samples 100`
- Disable workers: `--num_workers 0`

### GPU Training
- Use `--config distilgpt2`
- Increase batch size: `--batch_size 32` (adjust based on GPU memory)
- Enable multiple workers: `--num_workers 4`
- Use mixed precision training (extend trainer with `torch.cuda.amp`)

## Troubleshooting

### Out of Memory (OOM)

- Reduce batch size: `--batch_size 8`
- Use gradient accumulation (extend trainer)
- Reduce sequence length in config
- Use a smaller model config

### Slow Data Loading

- Increase workers: `--num_workers 8`
- Use streaming dataset for very large datasets
- Cache tokenized data (extend dataset class)

### Loss Not Decreasing

- Check learning rate (try `1e-3` for faster initial convergence)
- Ensure data is loading correctly (run sanity checks)
- Increase training samples or epochs
- Check for data preprocessing issues

## Extending the Project

### Custom Attention Mechanisms

1. Create custom mask generation function:
```python
def create_custom_mask(seq_len, pattern="bidirectional"):
    if pattern == "bidirectional":
        return torch.ones(seq_len, seq_len)
    elif pattern == "sliding_window":
        # Implement sliding window attention
        ...
    return mask
```

2. Pass to model:
```python
custom_mask = create_custom_mask(seq_len, pattern="sliding_window")
logits, loss = model(input_ids, attention_mask=custom_mask)
```

### Different Datasets

Modify `train/dataset.py` to load other text datasets:
```python
dataset = load_dataset("your_dataset", split="train")
```

### Evaluation Metrics

Extend `train/loop.py` to add perplexity, BLEU, or other metrics.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{gpt2_pytorch_impl,
  title={GPT-2 PyTorch Implementation from Scratch},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/arbitrary-prob-model}
}
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [DistilGPT-2 on Hugging Face](https://huggingface.co/distilbert/distilgpt2)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

## License

MIT License - Feel free to use this code for research and educational purposes.

## Contact

For questions or issues, please open an issue on GitHub.
