"""
Test script for OLD Sigma GPT model (Double Position Encoding)

This script tests the original nanoGPT-style Sigma GPT implementation
to compare its evaluation behavior with the current implementation.

Run from project root: python tests/test_sigmagpt_old.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random
from transformers import GPT2Tokenizer
import time

from model.sigmagpt_model_old import SigmaGPTOld
from model.config import get_config
from train.augmentation import ConditionalAugmenter
from train.sigmagpt_adapter import SigmaGPTDataAdapter
from train.dataset import get_dataloader
from train.blockwise_sampling import (
    uniform_num_blocks_distribution,
    uniform_block_sizes_distribution,
)


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_conditioning_distribution(cond_pct_min=0.0, cond_pct_max=0.4):
    """Create conditioning distribution function"""
    def distribution(seq_len):
        min_cond = max(0, int(seq_len * cond_pct_min))
        max_cond = max(min_cond, int(seq_len * cond_pct_max))
        max_cond = min(max_cond, seq_len - 1)
        return random.randint(min_cond, max_cond)
    return distribution


def create_evaluation_distribution(eval_pct_min=1.0, eval_pct_max=1.0):
    """Create evaluation distribution function"""
    def distribution(available_len):
        if available_len == 0:
            return 0
        min_eval = max(1, int(available_len * eval_pct_min))
        max_eval = max(min_eval, int(available_len * eval_pct_max))
        max_eval = min(max_eval, available_len)
        return random.randint(min_eval, max_eval)
    return distribution


def main():
    print("=" * 80)
    print("Testing OLD Sigma GPT Model (Double Position Encoding)")
    print("=" * 80)

    # Set random seed
    set_seed(42)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Configuration
    config = get_config('tiny')  # Use tiny config for quick test
    print(f"\nModel config: tiny")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_head: {config.n_head}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  max_seq_len: {config.max_seq_len}")

    # Initialize model
    print("\n[1] Initializing OLD Sigma GPT model...")
    model = SigmaGPTOld(config).to(device)
    print(f"  Model parameters: {model.get_num_params() / 1e6:.2f}M")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize augmenter
    print("\n[2] Creating augmenter...")
    augmenter = ConditionalAugmenter(
        mask_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.eos_token_id,
        max_seq_len=config.max_seq_len,
        tokenizer_pad_token_id=tokenizer.pad_token_id,
        num_conditioning_distribution=create_conditioning_distribution(0.0, 0.4),
        num_blocks_distribution=uniform_num_blocks_distribution,
        block_sizes_distribution=uniform_block_sizes_distribution,
        num_evaluation_distribution=create_evaluation_distribution(1.0, 1.0),
        num_eval_blocks_distribution=uniform_num_blocks_distribution,
        eval_block_sizes_distribution=uniform_block_sizes_distribution,
        conditioning_sampling='blockwise',
        evaluation_sampling='blockwise',
        max_cond_blocks=3,
        max_eval_blocks=2,
        ordering_mode='temporal',
    )

    # Initialize adapter
    adapter = SigmaGPTDataAdapter(mode='fair')
    print("  Augmenter and adapter initialized")

    # Create dataloaders
    print("\n[3] Loading data...")
    train_loader = get_dataloader(
        config=config,
        split='train',
        batch_size=4,
        num_workers=0,
        streaming=False,
        dataset_name='wikitext',
        dataset_config='wikitext-103-raw-v1',
        primary_dataset_only=True,
        num_samples=100
    )

    val_loader = get_dataloader(
        config=config,
        split='validation',
        batch_size=4,
        num_workers=0,
        streaming=False,
        dataset_name='wikitext',
        dataset_config='wikitext-103-raw-v1',
        primary_dataset_only=True,
        num_samples=50
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type=device.type
    )

    # Training loop
    print("\n[4] Training for 50 steps...")
    model.train()
    train_losses = []

    for step, batch in enumerate(train_loader):
        if step >= 50:
            break

        input_ids = batch['input_ids'].to(device)
        input_ids_cpu = input_ids.cpu()
        batch_size = input_ids_cpu.size(0)

        # Augment
        aug_batch = []
        for i in range(batch_size):
            result = augmenter.augment_sequence(input_ids_cpu[i], device='cpu')
            aug_batch.append(result)

        # Convert to Sigma GPT format
        sigmagpt_batch = adapter.convert_batch(aug_batch, input_ids_cpu)

        inputs = sigmagpt_batch['inputs'].to(device)
        order = sigmagpt_batch['order'].to(device)
        targets = sigmagpt_batch['targets'].to(device)

        # Forward pass
        logits, loss = model(idx=inputs, order=order, targets=targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}: loss = {loss.item():.4f}")

    avg_train_loss = np.mean(train_losses[-10:])
    print(f"\n  Average train loss (last 10 steps): {avg_train_loss:.4f}")

    # Evaluation
    print("\n[5] Running evaluation...")
    model.eval()
    eval_losses = []
    eval_tokens = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 10:  # Only 10 batches for quick test
                break

            input_ids = batch['input_ids'].to(device)
            input_ids_cpu = input_ids.cpu()
            batch_size = input_ids_cpu.size(0)

            # Augment (same as training - random augmentation)
            aug_batch = []
            for i in range(batch_size):
                result = augmenter.augment_sequence(input_ids_cpu[i], device='cpu')
                aug_batch.append(result)

            sigmagpt_batch = adapter.convert_batch(aug_batch, input_ids_cpu)

            inputs = sigmagpt_batch['inputs'].to(device)
            order = sigmagpt_batch['order'].to(device)
            targets = sigmagpt_batch['targets'].to(device)

            logits, loss = model(idx=inputs, order=order, targets=targets)

            valid_tokens = (targets != -1).sum().item()
            eval_losses.append(loss.item() * valid_tokens)
            eval_tokens.append(valid_tokens)

    avg_eval_loss = sum(eval_losses) / sum(eval_tokens)
    perplexity = np.exp(avg_eval_loss)

    print(f"\n  Evaluation loss: {avg_eval_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model: SigmaGPTOld (Double Position Encoding)")
    print(f"Position embedding dim: n_embd // 2 = {config.n_embd // 2}")
    print(f"Train loss (last 10): {avg_train_loss:.4f}")
    print(f"Eval loss: {avg_eval_loss:.4f}")
    print(f"Eval perplexity: {perplexity:.2f}")
    print(f"Train-Eval diff: {avg_eval_loss - avg_train_loss:.4f}")
    print("=" * 80)

    # Key observation
    print("\nKEY OBSERVATION:")
    print("  Both training and evaluation use the SAME random augmentation.")
    print("  Therefore, train_loss and eval_loss should be very similar.")
    print("  The only difference is the underlying data (train vs validation set).")

    if abs(avg_eval_loss - avg_train_loss) < 0.5:
        print("\n  ✓ As expected: Train loss ≈ Eval loss (same augmentation distribution)")
    else:
        print("\n  ! Unexpected: Train and eval losses differ significantly")


if __name__ == "__main__":
    main()
