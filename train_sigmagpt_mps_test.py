"""
Sigma GPT MPS (Apple Silicon) Training Test Script

Tests Sigma GPT training on Apple Silicon MPS device.
Uses small model and limited data for quick verification.

Run: python train_sigmagpt_mps_test.py
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model.sigmagpt_model import SigmaGPT
from model.config import GPT2Config
from model.token_manager import TokenManager
from train.sigmagpt_adapter import SigmaGPTDataAdapter
from train.augmentation import ConditionalAugmenter
from train.blockwise_sampling import (
    uniform_num_conditioning_distribution,
    uniform_num_blocks_distribution,
    uniform_block_sizes_distribution,
    uniform_num_evaluation_distribution
)
from train.dataset import get_dataloader

print("=" * 80)
print("  Sigma GPT MPS Training Test")
print("=" * 80)

# ========== Device Setup ==========
print("\n[1/7] Setting up device...")
if not torch.backends.mps.is_available():
    print("❌ MPS not available. Falling back to CPU.")
    device = torch.device('cpu')
else:
    print("✓ MPS is available")
    device = torch.device('mps')

print(f"Using device: {device}")

# ========== Configuration ==========
print("\n[2/7] Creating configuration...")

# Small config for fast testing
config = GPT2Config(
    vocab_size=50257,
    max_seq_len=512,   # Reduced from 1024
    n_layer=4,         # Reduced from 6-12
    n_head=4,          # Reduced from 6-12
    n_embd=256,        # Reduced from 384-768
    dropout=0.1
)

print(f"Model config:")
print(f"  - Layers: {config.n_layer}")
print(f"  - Heads: {config.n_head}")
print(f"  - Embedding dim: {config.n_embd}")
print(f"  - Max seq len: {config.max_seq_len}")
print(f"  - Vocab size: {config.vocab_size}")

# ========== Initialize Components ==========
print("\n[3/7] Initializing components...")

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Token manager
token_manager = TokenManager(
    tokenizer=tokenizer,
    add_mask_token=True,
    add_bos_token=False  # Use EOS as BOS
)
print(f"✓ Token manager initialized")
print(f"  - Mask token ID: {token_manager.mask_token_id}")
print(f"  - BOS token ID: {token_manager.bos_token_id}")

# DataLoader (num_workers=0 for MPS compatibility)
print("Loading dataset...")
dataloader = get_dataloader(
    config=config,
    split='train',
    batch_size=4,      # Small batch size for testing
    num_workers=0,     # IMPORTANT: 0 for MPS to avoid multiprocessing issues
    streaming=False,
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    primary_dataset_only=True
)
print(f"✓ DataLoader created (batch_size=4, num_workers=0)")

# Augmenter
augmenter = ConditionalAugmenter(
    mask_token_id=token_manager.mask_token_id,
    bos_token_id=token_manager.bos_token_id,
    max_seq_len=config.max_seq_len,
    num_conditioning_distribution=uniform_num_conditioning_distribution,
    num_blocks_distribution=uniform_num_blocks_distribution,
    block_sizes_distribution=uniform_block_sizes_distribution,
    num_evaluation_distribution=uniform_num_evaluation_distribution,
    num_eval_blocks_distribution=uniform_num_blocks_distribution,
    eval_block_sizes_distribution=uniform_block_sizes_distribution,
    conditioning_sampling='blockwise',
    evaluation_sampling='blockwise',
    max_cond_blocks=2,
    max_eval_blocks=1,
    tokenizer_pad_token_id=tokenizer.pad_token_id
)
print(f"✓ Augmenter initialized")

# Adapter (Fair mode for testing)
adapter = SigmaGPTDataAdapter(mode='fair')
print(f"✓ Adapter initialized (mode='fair')")

# ========== Initialize Model ==========
print("\n[4/7] Initializing Sigma GPT model...")
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
print("\n[5/7] Running training test...")
print("Training for 20 steps (to verify everything works)...\n")

model.train()
total_loss = 0
total_tokens = 0
step_times = []

num_steps = 20

try:
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        step_start = time.time()

        # Get input_ids
        input_ids = batch['input_ids'].to(device)

        # Data augmentation (on CPU)
        # Note: Must use augment_sequence (not augment_batch) to preserve conditioning/evaluation indices
        input_ids_cpu = input_ids.cpu()
        batch_size = input_ids_cpu.size(0)
        aug_batch = []
        for i in range(batch_size):
            result = augmenter.augment_sequence(input_ids_cpu[i], device='cpu')
            aug_batch.append(result)

        # Convert to Sigma GPT format
        sigmagpt_batch = adapter.convert_batch(aug_batch, input_ids_cpu)

        # Move to device
        inputs = sigmagpt_batch['inputs'].to(device)
        order = sigmagpt_batch['order'].to(device)
        targets = sigmagpt_batch['targets'].to(device)

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
print("\n[6/7] Training results:")
avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
avg_step_time = sum(step_times) / len(step_times) if step_times else 0

print(f"  - Steps completed: {len(step_times)}")
print(f"  - Total tokens processed: {total_tokens:,}")
print(f"  - Average loss: {avg_loss:.4f}")
print(f"  - Average step time: {avg_step_time:.3f}s")
print(f"  - Throughput: {total_tokens / sum(step_times):.1f} tokens/sec")

# ========== Model Test ==========
print("\n[7/7] Testing model inference...")
model.eval()

with torch.no_grad():
    # Get a test batch
    test_batch = next(iter(dataloader))
    input_ids = test_batch['input_ids'][:2].to(device)  # Take 2 samples

    # Augment
    # Note: Must use augment_sequence (not augment_batch) to preserve conditioning/evaluation indices
    input_ids_cpu = input_ids.cpu()
    batch_size = input_ids_cpu.size(0)
    aug_batch = []
    for i in range(batch_size):
        result = augmenter.augment_sequence(input_ids_cpu[i], device='cpu')
        aug_batch.append(result)
    sigmagpt_batch = adapter.convert_batch(aug_batch, input_ids_cpu)

    # Inference
    inputs = sigmagpt_batch['inputs'].to(device)
    order = sigmagpt_batch['order'].to(device)

    logits, _ = model(idx=inputs, order=order, targets=None)

    print(f"✓ Inference successful")
    print(f"  - Input shape: {inputs.shape}")
    print(f"  - Output shape: {logits.shape}")
    print(f"  - Order shape: {order.shape}")

# ========== Summary ==========
print("\n" + "=" * 80)
print("  TEST SUMMARY")
print("=" * 80)
print(f"Device:              {device}")
print(f"Model:               Sigma GPT ({config.n_layer} layers, {config.n_embd} dims)")
print(f"Training mode:       Fair mode (~40% learning efficiency)")
print(f"Steps completed:     {len(step_times)}/{num_steps}")
print(f"Average loss:        {avg_loss:.4f}")
print(f"Average step time:   {avg_step_time:.3f}s")
print(f"Throughput:          {total_tokens / sum(step_times):.1f} tokens/sec")
print(f"Status:              ✓ ALL TESTS PASSED")
print("=" * 80)

print("\n🎉 Sigma GPT MPS test completed successfully!")
print(f"\nThe model trains correctly on {device}.")
print("You can now use this for full-scale training.")
