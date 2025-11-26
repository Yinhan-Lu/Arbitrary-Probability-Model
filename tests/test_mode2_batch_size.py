"""
Test Mode 2 performance with different batch sizes

This script tests whether batch size affects Mode 2 evaluation results.
It compares:
1. Batch processing (bs=8, like Main pipeline)
2. Sequential processing (bs=1, like Legacy pipeline)

Run from project root: python tests/test_mode2_batch_size.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import math

# Import directly to avoid train module initialization
from model.config import GPT2Config
from model.arbitrary_prob_gpt2 import GPT2Model

# Import sampling functions directly from file to avoid __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("blockwise_sampling", "train/blockwise_sampling.py")
blockwise_sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blockwise_sampling)

generate_boundary_conditioning_split = blockwise_sampling.generate_boundary_conditioning_split
uniform_boundary_block_sizes_distribution = blockwise_sampling.uniform_boundary_block_sizes_distribution

print("=" * 100)
print("MODE 2 BATCH SIZE ABLATION TEST")
print("=" * 100)

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✓ Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ Using CUDA device")
else:
    device = torch.device("cpu")
    print(f"✓ Using CPU device")

print("=" * 100)

# Test configuration
num_samples = 20  # Test with 20 samples
seq_len = 128
vocab_size = 50257
mask_token_id = vocab_size
bos_token_id = vocab_size - 1

# Create model
config = GPT2Config(
    n_layer=4,
    n_head=4,
    n_embd=128,
    vocab_size=vocab_size + 1,
    max_seq_len=256,
    detach_augmentation=False
)

model = GPT2Model(
    config,
    mask_token_id=mask_token_id,
    bos_token_id=bos_token_id
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"\n📋 Test Configuration:")
print(f"  Model: {num_params / 1e6:.2f}M parameters")
print(f"  Samples: {num_samples}")
print(f"  Sequence length: {seq_len}")
print(f"  Device: {device}")

# Create test data (random sequences)
torch.manual_seed(42)
test_sequences = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)

# Mode 2 boundary distribution
def boundary_dist_fn(seq_len):
    return uniform_boundary_block_sizes_distribution(
        seq_len,
        boundary_cond_percentage_range=(0.1, 0.3)
    )

print("\n" + "=" * 100)
print("TEST 1: BATCH PROCESSING (batch_size=8, Main pipeline)")
print("=" * 100)

model.eval()
batch_size = 8
total_loss_batched = 0.0
total_tokens_batched = 0

with torch.no_grad():
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        input_ids_batch = test_sequences[i:batch_end]
        current_batch_size = batch_end - i

        # Sample indices for each sample
        batch_cond_idx = []
        batch_eval_idx = []
        batch_unseen_idx = []

        for j in range(current_batch_size):
            cond_idx, eval_idx, unknown_idx = generate_boundary_conditioning_split(
                seq_len,
                boundary_block_sizes_distribution=boundary_dist_fn,
                valid_positions=list(range(seq_len))
            )
            batch_cond_idx.append(cond_idx)
            batch_eval_idx.append(eval_idx)
            batch_unseen_idx.append(unknown_idx)

        # Forward pass - batched
        logits, loss = model(
            input_ids=input_ids_batch,
            conditional_idx=batch_cond_idx,
            evaluation_idx=batch_eval_idx,
            unseen_idx=batch_unseen_idx
        )

        # Accumulate loss
        if loss is not None:
            total_eval_tokens = sum(len(eval_idx) for eval_idx in batch_eval_idx)
            total_loss_batched += loss.item() * total_eval_tokens
            total_tokens_batched += total_eval_tokens

avg_loss_batched = total_loss_batched / total_tokens_batched if total_tokens_batched > 0 else 0.0
ppl_batched = math.exp(avg_loss_batched) if avg_loss_batched < 20 else float('inf')

print(f"\n✓ Batch processing completed")
print(f"  Total tokens: {total_tokens_batched}")
print(f"  Average loss: {avg_loss_batched:.4f}")
print(f"  Perplexity: {ppl_batched:.2f}")

print("\n" + "=" * 100)
print("TEST 2: SEQUENTIAL PROCESSING (batch_size=1, Legacy pipeline)")
print("=" * 100)

model.eval()
total_loss_sequential = 0.0
total_tokens_sequential = 0

# Use SAME indices as batched version for fair comparison
# Re-sample with same seed
torch.manual_seed(42)

with torch.no_grad():
    for i in range(num_samples):
        input_ids_single = test_sequences[i:i+1]  # Shape: (1, seq_len)

        # Sample indices (using same seed, should match batched version)
        cond_idx, eval_idx, unknown_idx = generate_boundary_conditioning_split(
            seq_len,
            boundary_block_sizes_distribution=boundary_dist_fn,
            valid_positions=list(range(seq_len))
        )

        # Forward pass - single sample
        logits, loss = model(
            input_ids=input_ids_single,
            conditional_idx=[cond_idx],
            evaluation_idx=[eval_idx],
            unseen_idx=[unknown_idx]
        )

        # Accumulate loss
        if loss is not None:
            total_eval_tokens = len(eval_idx)
            total_loss_sequential += loss.item() * total_eval_tokens
            total_tokens_sequential += total_eval_tokens

avg_loss_sequential = total_loss_sequential / total_tokens_sequential if total_tokens_sequential > 0 else 0.0
ppl_sequential = math.exp(avg_loss_sequential) if avg_loss_sequential < 20 else float('inf')

print(f"\n✓ Sequential processing completed")
print(f"  Total tokens: {total_tokens_sequential}")
print(f"  Average loss: {avg_loss_sequential:.4f}")
print(f"  Perplexity: {ppl_sequential:.2f}")

print("\n" + "=" * 100)
print("COMPARISON")
print("=" * 100)

print(f"\n{'Metric':<20} {'Batched (bs=8)':<20} {'Sequential (bs=1)':<20} {'Difference':<15}")
print("-" * 75)
print(f"{'Loss':<20} {avg_loss_batched:<20.4f} {avg_loss_sequential:<20.4f} {abs(avg_loss_batched - avg_loss_sequential):<15.4f}")
print(f"{'Perplexity':<20} {ppl_batched:<20.2f} {ppl_sequential:<20.2f} {abs(ppl_batched - ppl_sequential):<15.2f}")
print(f"{'Tokens':<20} {total_tokens_batched:<20} {total_tokens_sequential:<20} {abs(total_tokens_batched - total_tokens_sequential):<15}")

# Check if results are similar
loss_diff_pct = abs(avg_loss_batched - avg_loss_sequential) / avg_loss_batched * 100 if avg_loss_batched > 0 else 0
ppl_diff_pct = abs(ppl_batched - ppl_sequential) / ppl_batched * 100 if ppl_batched > 0 else 0

print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)

if loss_diff_pct < 1.0:
    print(f"\n✅ Results are VERY SIMILAR (loss diff: {loss_diff_pct:.2f}%)")
    print("\nBatch size does NOT significantly affect Mode 2 performance.")
    print("The Legacy vs Main difference must be due to other factors:")
    print("  1. Random seed control (already found in Experiment 1)")
    print("  2. Tensor construction method (.item() vs pure tensor ops)")
    print("  3. CPU vs GPU augmentation execution")
elif loss_diff_pct < 5.0:
    print(f"\n⚠️  Results have MINOR DIFFERENCES (loss diff: {loss_diff_pct:.2f}%)")
    print("\nBatch size has a small effect, but likely not the main cause of")
    print("the large Legacy (ppl=120) vs Main (ppl=7) difference.")
else:
    print(f"\n❌ Results are SIGNIFICANTLY DIFFERENT (loss diff: {loss_diff_pct:.2f}%)")
    print("\nBatch size DOES significantly affect Mode 2 performance!")
    print("This could explain the Legacy vs Main difference.")

print("\n" + "=" * 100)
print("TEST COMPLETE")
print("=" * 100)
