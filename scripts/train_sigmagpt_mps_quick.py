"""
Sigma GPT MPS (Apple Silicon) Quick Training Test

Quick test with synthetic data (no dataset loading).
Tests Sigma GPT training on Apple Silicon MPS device.

Run: python train_sigmagpt_mps_quick.py
"""

import sys
from pathlib import Path
import torch
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model.sigmagpt_model import SigmaGPT
from model.config import GPT2Config
from train.sigmagpt_adapter import SigmaGPTDataAdapter
from model.order_utils import prepare_sigmagpt_batch

print("=" * 80)
print("  Sigma GPT MPS Quick Training Test")
print("=" * 80)

# ========== Device Setup ==========
print("\n[1/5] Setting up device...")
if not torch.backends.mps.is_available():
    print("❌ MPS not available. Falling back to CPU.")
    device = torch.device('cpu')
else:
    print("✓ MPS is available")
    device = torch.device('mps')

print(f"Using device: {device}")

# ========== Configuration ==========
print("\n[2/5] Creating configuration...")

# Small config for fast testing
config = GPT2Config(
    vocab_size=1000,    # Small vocab for testing
    max_seq_len=128,    # Short sequences
    n_layer=2,          # Minimal layers
    n_head=2,           # Minimal heads
    n_embd=128,         # Small embedding
    dropout=0.1
)

print(f"Model config:")
print(f"  - Layers: {config.n_layer}")
print(f"  - Heads: {config.n_head}")
print(f"  - Embedding dim: {config.n_embd}")
print(f"  - Max seq len: {config.max_seq_len}")
print(f"  - Vocab size: {config.vocab_size}")

# ========== Initialize Model ==========
print("\n[3/5] Initializing Sigma GPT model...")
model = SigmaGPT(config)
model = model.to(device)
print(f"✓ Model moved to {device}")

# Optimizer
optimizer = model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=3e-4,
    betas=(0.9, 0.95),
    device_type=device.type
)
print(f"✓ Optimizer configured (AdamW, lr=3e-4)")

# ========== Training Test ==========
print("\n[4/5] Running training test with synthetic data...")
print("Training for 20 steps...\n")

model.train()
total_loss = 0
total_tokens = 0
step_times = []

num_steps = 20
batch_size = 4
seq_len = 64

try:
    for step in range(num_steps):
        step_start = time.time()

        # Generate synthetic data
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Generate random conditioning/evaluation indices (ensure no overlap)
        num_cond = torch.randint(5, 15, (1,)).item()
        num_eval = torch.randint(5, 15, (1,)).item()

        cond_indices_list = []
        eval_indices_list = []

        for b in range(batch_size):
            # Sample conditioning indices
            cond_sample = torch.randperm(seq_len)[:num_cond]
            cond_set = set(cond_sample.tolist())

            # Sample evaluation indices (ensure no overlap with conditioning)
            available = [i for i in range(seq_len) if i not in cond_set]
            eval_sample = torch.tensor(available)[torch.randperm(len(available))[:num_eval]]

            cond_indices_list.append(cond_sample)
            eval_indices_list.append(eval_sample)

        cond_indices = torch.stack(cond_indices_list)
        eval_indices = torch.stack(eval_indices_list)

        # Prepare Sigma GPT batch (Fair mode)
        inputs, order, targets = prepare_sigmagpt_batch(
            tokens, cond_indices, eval_indices, mode='fair'
        )

        # Move to device
        inputs = inputs.to(device)
        order = order.to(device)
        targets = targets.to(device)

        # Forward pass
        logits, loss = model(idx=inputs, order=order, targets=targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Statistics
        valid_tokens = (targets != -1).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

        step_time = time.time() - step_start
        step_times.append(step_time)

        # Log every 5 steps
        if (step + 1) % 5 == 0:
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            avg_time = sum(step_times) / len(step_times)
            print(f"Step {step + 1:3d}/{num_steps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Time: {step_time:.3f}s | "
                  f"Avg Time: {avg_time:.3f}s")

    print("\n✓ Training test completed successfully!")

except Exception as e:
    print(f"\n❌ Training failed with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== Results ==========
print("\n[5/5] Training results:")
avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
avg_step_time = sum(step_times) / len(step_times) if step_times else 0
throughput = total_tokens / sum(step_times) if sum(step_times) > 0 else 0

print(f"  - Steps completed: {len(step_times)}")
print(f"  - Total tokens processed: {total_tokens:,}")
print(f"  - Average loss: {avg_loss:.4f}")
print(f"  - Average step time: {avg_step_time:.3f}s")
print(f"  - Throughput: {throughput:.1f} tokens/sec")

# ========== Inference Test ==========
print("\nTesting model inference...")
model.eval()

with torch.no_grad():
    # Generate test batch
    test_tokens = torch.randint(0, config.vocab_size, (2, 32))
    test_cond = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    test_eval = torch.tensor([[4, 5], [4, 5]])

    test_inputs, test_order, _ = prepare_sigmagpt_batch(
        test_tokens, test_cond, test_eval, mode='fair'
    )

    # Inference
    logits, _ = model(
        idx=test_inputs.to(device),
        order=test_order.to(device),
        targets=None
    )

    print(f"✓ Inference successful")
    print(f"  - Input shape: {test_inputs.shape}")
    print(f"  - Output shape: {logits.shape}")
    print(f"  - Order shape: {test_order.shape}")

# ========== Summary ==========
print("\n" + "=" * 80)
print("  TEST SUMMARY")
print("=" * 80)
print(f"Device:              {device}")
print(f"Model:               Sigma GPT ({config.n_layer} layers, {config.n_embd} dims)")
print(f"Training mode:       Fair mode (~40% learning efficiency)")
print(f"Data:                Synthetic (random tokens)")
print(f"Steps completed:     {len(step_times)}/{num_steps}")
print(f"Average loss:        {avg_loss:.4f}")
print(f"Average step time:   {avg_step_time:.3f}s")
print(f"Throughput:          {throughput:.1f} tokens/sec")
print(f"Status:              ✓ ALL TESTS PASSED")
print("=" * 80)

print("\n🎉 Sigma GPT MPS quick test completed successfully!")
print(f"\nThe model trains correctly on {device}.")
print("You can now use this for full-scale training with real data.")
