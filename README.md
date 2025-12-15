# Arbitrary Conditional Probability Model

A research platform for studying **arbitrary conditional probability modeling** in transformer-based language models. This project extends standard autoregressive models to compute P(X_e | X_c) where X_c (conditioning set) and X_e (evaluation set) can be any disjoint subsets of a sequence.

## Research Motivation

Standard autoregressive language models (like GPT-2) can only model left-to-right conditional probabilities:

```
P(x_t | x_1, x_2, ..., x_{t-1})
```

This project addresses a fundamental question: **Can transformers learn arbitrary conditional probability distributions?**

We implement models that can compute:

```
P(X_e | X_c)  where X_c ∩ X_e = ∅
```

For example, given a sequence "The cat sat on the mat", we can condition on {"The", "sat", "mat"} and evaluate the probability of {"cat", "on", "the"}.

## Key Features

- **Arbitrary Conditional Probability Modeling**: Custom attention masks enabling conditioning on any subset of tokens
- **Three Model Architectures**: Conditional model, standard autoregressive baseline, and Sigma GPT for comprehensive comparison
- **Unified Training Pipeline**: Single entry point supporting all model types with identical optimization settings
- **Deterministic Evaluation Framework**: Pre-generated fixed splits ensuring all models solve identical tasks
- **Ordering Modes**: Temporal vs. random scramble strategies for systematic capacity testing
- **Visualization Pipeline**: Experiment analysis and comparison tools

## Model Architectures

### 1. Conditional Model (Main Research)
**File**: `model/arbitrary_prob_gpt2.py`

The core research model that uses custom attention masks to enable arbitrary conditioning:
- Tokens in the conditioning set X_c can attend to each other
- Tokens in the evaluation set X_e can attend to X_c and preceding X_e tokens
- Special `[M]` mask token represents unknown positions
- Loss computed only on evaluation set positions

### 2. Baseline Model
**File**: `model/baseline_gpt2.py`

Standard autoregressive GPT-2 implementation for reference:
- Left-to-right causal attention only
- Models P(x_t | x_{<t})
- Serves as performance baseline

### 3. Sigma GPT
**File**: `model/sigmagpt_from_baseline.py`

Random permutation approach for comparison:
- Shuffles entire sequence order
- Trains on permuted sequences
- Two ordering strategies for fair comparison

## Project Structure

```
.
├── model/                          # Model implementations
│   ├── arbitrary_prob_gpt2.py      # Conditional probability model
│   ├── baseline_gpt2.py            # Standard GPT-2 baseline
│   ├── sigmagpt_from_baseline.py   # Sigma GPT implementation
│   ├── config.py                   # Model configurations
│   ├── token_manager.py            # Special token handling ([M] mask)
│   └── order_utils.py              # Order tensor utilities
│
├── train/                          # Training infrastructure (16 modules)
│   ├── conditional_trainer.py      # Conditional model training
│   ├── baseline_trainer.py         # Baseline training
│   ├── sigmagpt_trainer.py         # Sigma GPT training
│   ├── base_trainer.py             # Abstract base trainer
│   ├── dataset.py                  # Wikipedia dataset loading
│   ├── augmentation.py             # Data augmentation with ordering modes
│   ├── mask_utils.py               # Attention mask construction
│   ├── ordering_modes.py           # Temporal vs. scramble strategies
│   ├── deterministic_eval.py       # Reproducible evaluation
│   └── ...
│
├── tests/                          # Test suite (30+ tests)
│   ├── quick_test.py               # Fast validation (~10s)
│   ├── sanity.py                   # Full pipeline test (~3 min)
│   ├── test_ordering_modes.py      # Ordering strategy tests
│   └── ...
│
├── scripts/                        # SLURM and utility scripts (40+ files)
│   ├── submit_comparison_*.sh      # Experiment submission scripts
│   ├── generate_eval_splits.py     # Deterministic split generation
│   └── ...
│
├── visualization/                  # Experiment analysis
│   ├── experiment_loader.py        # Metrics loading
│   ├── plotting.py                 # Visualization functions
│   └── README.md
│
├── utils/                          # Development utilities
│   ├── quickstart_visualization.py
│   └── compare_experiments.py
│
├── experiments/                    # Experiment outputs (gitignored)
├── train.py                        # Unified training entry point
└── requirements.txt
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.12+

### Install Dependencies

```bash
pip install torch transformers datasets matplotlib seaborn pandas
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Tests

Verify the installation:

```bash
# Fast validation (~10 seconds)
python tests/quick_test.py

# Full pipeline test (~3 minutes)
python tests/sanity.py
```

### 2. Train Models

The unified `train.py` supports all three model types:

```bash
# Conditional model (main research)
python train.py --model_type conditional --config tiny --max_steps 1000

# Baseline autoregressive model
python train.py --model_type baseline --config tiny --max_steps 1000

# Sigma GPT model
python train.py --model_type sigmagpt --config tiny --max_steps 1000
```

### 3. Visualize Results

```bash
# Single experiment
python utils/quickstart_visualization.py experiments/your_experiment

# Compare multiple experiments
python utils/quickstart_visualization.py exp1 exp2 --compare
```

## Training

### Unified Training Interface

All models use the same training interface for fair comparison:

```bash
python train.py \
    --model_type conditional \
    --config distilgpt2 \
    --max_steps 50000 \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4 \
    --device cuda
```

**Model Types:**
- `conditional` - Arbitrary conditional probability model
- `baseline` - Standard autoregressive GPT-2
- `sigmagpt` - Sigma GPT with random permutation

### SLURM Cluster Training

For large-scale experiments:

```bash
# Conditional model comparison
sbatch scripts/submit_comparison_conditional.sh

# Sigma GPT with temporal ordering
sbatch scripts/submit_comparison_sigmagpt_temporal.sh

# Sigma GPT with random scrambling
sbatch scripts/submit_comparison_sigmagpt_scramble.sh
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_type` | Model architecture | `conditional` |
| `--config` | Size configuration | `distilgpt2` |
| `--max_steps` | Training steps | `50000` |
| `--batch_size` | Batch size | `8` |
| `--gradient_accumulation_steps` | Gradient accumulation | `16` |
| `--learning_rate` | Learning rate | `5e-4` |
| `--ordering_mode` | Ordering strategy | `temporal` |

## Evaluation

### Deterministic Evaluation Framework

To ensure fair comparison, all models are evaluated on identical tasks:

```bash
# Generate fixed evaluation splits
python scripts/generate_eval_splits.py --num_splits 1000 --output eval_splits.json

# Train with deterministic evaluation
python train.py --model_type conditional --eval_splits eval_splits.json
```

### Ordering Modes

Two ordering strategies for systematic capacity testing:

1. **Temporal** (`--ordering_mode temporal`): Maintain chronological order within blocks
2. **Random Scramble** (`--ordering_mode random_scramble`): Shuffle tokens within blocks

## Model Configurations

### DistilGPT-2 (Production)

```python
n_layer: 6
n_head: 12
n_embd: 768
vocab_size: 50257
max_seq_len: 1024
Parameters: ~82M
```

### Tiny (CPU Testing)

```python
n_layer: 2
n_head: 4
n_embd: 128
max_seq_len: 256
Parameters: ~7M
```

### Nano (Quick Debugging)

```python
n_layer: 1
n_head: 2
n_embd: 64
max_seq_len: 128
Parameters: ~3M
```

## Key Technical Details

### Custom Attention Masks

The conditional model uses custom attention masks:

```python
# Conditioning tokens (X_c) can see each other
# Evaluation tokens (X_e) can see X_c and preceding X_e tokens
attention_mask = create_conditional_mask(
    conditioning_indices=c_indices,
    evaluation_indices=e_indices,
    seq_len=seq_len
)
```

### [M] Mask Token

A special `[M]` token represents unknown positions:

```python
from model.token_manager import TokenManager

token_manager = TokenManager(tokenizer)
# Input: "The [M] sat on the [M]"
# Model predicts tokens at [M] positions given visible tokens
```

### Loss Computation

Loss is computed only on evaluation set positions:

```python
# Only evaluate probability of X_e tokens
loss = cross_entropy(logits[e_indices], targets[e_indices])
```

## Visualization

### Training Curves

```bash
# Plot single experiment
python utils/quickstart_visualization.py experiments/exp_name

# Compare experiments
python utils/quickstart_visualization.py exp1 exp2 exp3 --compare
```

### Experiment Analysis

```python
from visualization import quick_plot

# Single experiment analysis
quick_plot("experiments/your_experiment")

# Multi-experiment comparison
from visualization.plotting import plot_comparison
plot_comparison(["exp1", "exp2"], metrics=["loss", "perplexity"])
```

## Testing

| Test | Description | Runtime |
|------|-------------|---------|
| `quick_test.py` | Fast model validation | ~10s |
| `sanity.py` | Full training pipeline | ~3 min |
| `test_ordering_modes.py` | Ordering strategy verification | ~30s |
| `test_checkpoint_loading.py` | Checkpoint I/O | ~30s |

Run all tests:

```bash
# From project root
python tests/quick_test.py
python tests/sanity.py
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 4 --gradient_accumulation_steps 32

# Use smaller config
python train.py --config tiny
```

### Slow Data Loading

```bash
# Increase workers
python train.py --num_workers 8
```

### Loss Not Decreasing

- Check learning rate (try `1e-3` for faster convergence)
- Verify data loading with `python tests/sanity.py`
- Increase training steps

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [DistilGPT-2 on Hugging Face](https://huggingface.co/distilbert/distilgpt2)

## License

MIT License - For research and educational purposes.

## Author

Yinhan Lu - McGill University

For questions or issues, please open an issue on GitHub.
