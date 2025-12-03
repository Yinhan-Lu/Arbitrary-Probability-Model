#!/usr/bin/env python3
"""
Debug script to compare Legacy vs New Pipeline Mode 2 Evaluation

This script runs Mode 2 evaluation on the same sample using both legacy
(external augmentation) and new (internal augmentation) pipelines, printing
all intermediate steps to identify where they diverge.

Usage:
    python debug_mode2_comparison.py
"""

import sys
import torch
import random
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train.blockwise_sampling import generate_boundary_conditioning_split
from model.token_manager import TokenManager
import torch.nn.functional as F


def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subsection(title):
    """Print a subsection header"""
    print(f"\n--- {title} ---")


def visualize_sequence(tokens, labels=None, eval_idx=None, cond_idx=None, mask_token_id=None):
    """Visualize a sequence with annotations"""
    print(f"  Sequence length: {len(tokens)}")
    print(f"  Tokens: {tokens[:20]}..." if len(tokens) > 20 else f"  Tokens: {tokens}")

    if mask_token_id is not None:
        mask_positions = [i for i, t in enumerate(tokens) if t == mask_token_id]
        print(f"  [M] mask positions: {mask_positions}")

    if eval_idx is not None:
        print(f"  Evaluation indices: {eval_idx}")

    if cond_idx is not None:
        print(f"  Conditioning indices: {cond_idx}")

    if labels is not None:
        ignore_positions = [i for i, l in enumerate(labels) if l == -100]
        eval_positions = [i for i, l in enumerate(labels) if l != -100]
        print(f"  Labels -100 positions: {ignore_positions}")
        print(f"  Labels eval positions: {eval_positions}")


def legacy_mode2_evaluation(input_ids, tokenizer, mask_token_id, model=None):
    """
    Simulate Legacy Pipeline Mode 2 evaluation with external augmentation.

    This mimics the logic from legacy branch's evaluation_modes.py
    """
    print_section("LEGACY PIPELINE - Mode 2 Evaluation")

    seq_len = len(input_ids)
    device = torch.device('cpu')

    # Step 1: Sample indices
    print_subsection("Step 1: Index Sampling")
    cond_idx, eval_idx, unknown_idx = generate_boundary_conditioning_split(
        seq_len=seq_len,
        boundary_block_sizes_distribution=None,  # Use defaults
        valid_positions=None
    )

    print(f"  Conditioning indices: {cond_idx}")
    print(f"  Evaluation indices: {eval_idx}")
    print(f"  Unknown indices: {unknown_idx}")

    # Step 2: Calculate truly unseen
    print_subsection("Step 2: Calculate Truly Unseen")
    eval_set = set(eval_idx)
    unknown_set = set(unknown_idx)
    truly_unseen_set = unknown_set - eval_set
    print(f"  Truly unseen = unknown - eval = {truly_unseen_set}")
    print(f"  Number of truly unseen tokens: {len(truly_unseen_set)}")

    # Step 3: Build augmented sequence (EXTERNAL)
    print_subsection("Step 3: Build Augmented Sequence (External)")

    # Build prefix (conditioning tokens)
    sorted_cond = sorted(cond_idx)
    prefix_tokens = [input_ids[i] for i in sorted_cond]
    print(f"  Prefix tokens: {prefix_tokens}")
    print(f"  Prefix decoded: {tokenizer.decode(prefix_tokens)}")

    # Build body (BOS + masked sequence)
    body_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    for i in range(seq_len):
        if i in truly_unseen_set:
            body_tokens.append(mask_token_id)
        else:
            body_tokens.append(input_ids[i])

    print(f"  Body tokens: {body_tokens[:20]}..." if len(body_tokens) > 20 else f"  Body tokens: {body_tokens}")
    masked_positions = [i for i, t in enumerate(body_tokens) if t == mask_token_id]
    print(f"  Masked positions in body: {masked_positions}")

    # Full augmented sequence
    aug_tokens = prefix_tokens + body_tokens
    print(f"  Full augmented sequence length: {len(aug_tokens)}")
    print(f"  = Prefix ({len(prefix_tokens)}) + Body ({len(body_tokens)})")

    # Step 4: Build attention mask
    print_subsection("Step 4: Build Attention Mask")
    prefix_len = len(prefix_tokens)
    total_len = len(aug_tokens)

    # Prefix conditional + causal attention
    attention_mask = torch.zeros((total_len, total_len), dtype=torch.bool)

    # All positions can see the prefix
    attention_mask[:, :prefix_len] = True

    # Body uses causal attention
    for i in range(prefix_len, total_len):
        attention_mask[i, :i+1] = True

    print(f"  Attention mask shape: {attention_mask.shape}")
    print(f"  Prefix visible to all: columns 0-{prefix_len-1}")
    print(f"  Body causal: columns {prefix_len}-{total_len-1}")

    # Step 5: Build labels
    print_subsection("Step 5: Build Labels")

    # Labels: -100 for prefix, then shifted body
    labels = [-100] * prefix_len  # Ignore prefix

    # For body, only evaluate on evaluation indices
    for i in range(seq_len):
        if i in eval_idx:
            labels.append(input_ids[i])
        else:
            labels.append(-100)

    eval_label_positions = [i for i, l in enumerate(labels) if l != -100]
    print(f"  Labels length: {len(labels)}")
    print(f"  Evaluation positions (label != -100): {eval_label_positions}")
    print(f"  Number of evaluation tokens: {len(eval_label_positions)}")

    # Step 6: Compute loss (simulated)
    print_subsection("Step 6: Loss Computation")

    if model is not None:
        # TODO: Actual forward pass if model is provided
        print("  [Model forward pass would happen here]")
        loss = None
    else:
        # Simulated loss for demonstration
        print("  [Simulated loss - no actual model inference]")
        loss = None

    print(f"  Loss: {loss}")

    # Summary
    print_subsection("Summary")
    print(f"  Conditioning: {len(cond_idx)} tokens ({len(cond_idx)/seq_len*100:.1f}%)")
    print(f"  Evaluation: {len(eval_idx)} tokens ({len(eval_idx)/seq_len*100:.1f}%)")
    print(f"  Truly unseen: {len(truly_unseen_set)} tokens")
    print(f"  Masked in body: {len(masked_positions)} positions")

    return {
        'cond_idx': cond_idx,
        'eval_idx': eval_idx,
        'unknown_idx': unknown_idx,
        'truly_unseen': truly_unseen_set,
        'aug_tokens': aug_tokens,
        'prefix_tokens': prefix_tokens,
        'body_tokens': body_tokens,
        'labels': labels,
        'attention_mask': attention_mask,
        'loss': loss,
        'masked_positions': masked_positions
    }


def new_mode2_evaluation(input_ids, tokenizer, mask_token_id, model=None):
    """
    Simulate New Pipeline Mode 2 evaluation with internal augmentation.

    This mimics the logic from main branch's evaluation_modes.py + model internal augmentation
    """
    print_section("NEW PIPELINE - Mode 2 Evaluation (Internal Augmentation)")

    seq_len = len(input_ids)
    device = torch.device('cpu')

    # Step 1: Sample indices (SAME as legacy)
    print_subsection("Step 1: Index Sampling")
    set_seeds(42)  # Reset seed to get same sampling
    cond_idx, eval_idx, unknown_idx = generate_boundary_conditioning_split(
        seq_len=seq_len,
        boundary_block_sizes_distribution=None,  # Use defaults
        valid_positions=None
    )

    print(f"  Conditioning indices: {cond_idx}")
    print(f"  Evaluation indices: {eval_idx}")
    print(f"  Unknown indices: {unknown_idx}")

    # Step 2: Model internal augmentation simulation
    print_subsection("Step 2: Internal Augmentation (in model.forward)")
    print("  [Model receives: input_ids, conditional_idx, evaluation_idx, unseen_idx]")
    print("  [Model builds augmented sequence internally...]")

    # Simulate what model does internally
    # From model/arbitrary_prob_gpt2.py:_build_augmented_sequence_single
    print_subsection("Step 2a: Calculate Truly Unseen (inside model)")
    eval_set = set(eval_idx)
    unseen_set = set(unknown_idx)
    truly_unseen_set = unseen_set - eval_set
    print(f"  eval_idx_set: {eval_set}")
    print(f"  unseen_idx_set: {unseen_set}")
    print(f"  truly_unseen_set = unseen - eval = {truly_unseen_set}")
    print(f"  Number of truly unseen tokens: {len(truly_unseen_set)}")

    # Build prefix
    print_subsection("Step 2b: Build Prefix (inside model)")
    sorted_cond = sorted(cond_idx)
    prefix_tokens = [input_ids[i] for i in sorted_cond]
    print(f"  Prefix tokens: {prefix_tokens}")

    # Build body
    print_subsection("Step 2c: Build Body (inside model)")
    body_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    # Clone and mask
    for i in range(seq_len):
        if i in truly_unseen_set:
            body_tokens.append(mask_token_id)
        else:
            body_tokens.append(input_ids[i])

    print(f"  Body tokens: {body_tokens[:20]}..." if len(body_tokens) > 20 else f"  Body tokens: {body_tokens}")
    masked_positions = [i for i, t in enumerate(body_tokens) if t == mask_token_id]
    print(f"  Masked positions in body: {masked_positions}")

    # Full sequence
    aug_tokens = prefix_tokens + body_tokens
    print(f"  Full augmented sequence length: {len(aug_tokens)}")

    # Step 3: Build attention mask (SAME logic)
    print_subsection("Step 3: Build Attention Mask (inside model)")
    prefix_len = len(prefix_tokens)
    total_len = len(aug_tokens)

    attention_mask = torch.zeros((total_len, total_len), dtype=torch.bool)
    attention_mask[:, :prefix_len] = True
    for i in range(prefix_len, total_len):
        attention_mask[i, :i+1] = True

    print(f"  Attention mask shape: {attention_mask.shape}")

    # Step 4: Build labels (SAME logic)
    print_subsection("Step 4: Build Labels (inside model)")
    labels = [-100] * prefix_len
    for i in range(seq_len):
        if i in eval_idx:
            labels.append(input_ids[i])
        else:
            labels.append(-100)

    eval_label_positions = [i for i, l in enumerate(labels) if l != -100]
    print(f"  Evaluation positions (label != -100): {eval_label_positions}")
    print(f"  Number of evaluation tokens: {len(eval_label_positions)}")

    # Step 5: Loss computation
    print_subsection("Step 5: Loss Computation")

    if model is not None:
        print("  [Model forward pass would happen here]")
        loss = None
    else:
        print("  [Simulated loss - no actual model inference]")
        loss = None

    print(f"  Loss: {loss}")

    # Summary
    print_subsection("Summary")
    print(f"  Conditioning: {len(cond_idx)} tokens ({len(cond_idx)/seq_len*100:.1f}%)")
    print(f"  Evaluation: {len(eval_idx)} tokens ({len(eval_idx)/seq_len*100:.1f}%)")
    print(f"  Truly unseen: {len(truly_unseen_set)} tokens")
    print(f"  Masked in body: {len(masked_positions)} positions")

    return {
        'cond_idx': cond_idx,
        'eval_idx': eval_idx,
        'unknown_idx': unknown_idx,
        'truly_unseen': truly_unseen_set,
        'aug_tokens': aug_tokens,
        'prefix_tokens': prefix_tokens,
        'body_tokens': body_tokens,
        'labels': labels,
        'attention_mask': attention_mask,
        'loss': loss,
        'masked_positions': masked_positions
    }


def compare_results(legacy_result, new_result, tokenizer):
    """Compare results from legacy and new pipelines"""
    print_section("COMPARISON: Legacy vs New")

    def compare_field(field_name, legacy_val, new_val):
        if isinstance(legacy_val, (list, set)):
            match = set(legacy_val) == set(new_val)
        elif isinstance(legacy_val, torch.Tensor):
            match = torch.equal(legacy_val, new_val)
        else:
            match = legacy_val == new_val

        status = "✓" if match else "✗"
        print(f"  {status} {field_name}: {'IDENTICAL' if match else 'DIFFERENT'}")

        if not match and field_name in ['aug_tokens', 'body_tokens', 'prefix_tokens']:
            print(f"      Legacy: {legacy_val[:15]}..." if len(legacy_val) > 15 else f"      Legacy: {legacy_val}")
            print(f"      New:    {new_val[:15]}..." if len(new_val) > 15 else f"      New:    {new_val}")

            if isinstance(legacy_val, list) and isinstance(new_val, list):
                for i, (l, n) in enumerate(zip(legacy_val, new_val)):
                    if l != n:
                        print(f"      First difference at position {i}: {l} vs {n}")
                        if l < len(tokenizer) and n < len(tokenizer):
                            print(f"        Decoded: '{tokenizer.decode([l])}' vs '{tokenizer.decode([n])}'")
                        break

        return match

    # Compare all fields
    all_match = True
    all_match &= compare_field("cond_idx", legacy_result['cond_idx'], new_result['cond_idx'])
    all_match &= compare_field("eval_idx", legacy_result['eval_idx'], new_result['eval_idx'])
    all_match &= compare_field("unknown_idx", legacy_result['unknown_idx'], new_result['unknown_idx'])
    all_match &= compare_field("truly_unseen", legacy_result['truly_unseen'], new_result['truly_unseen'])
    all_match &= compare_field("prefix_tokens", legacy_result['prefix_tokens'], new_result['prefix_tokens'])
    all_match &= compare_field("body_tokens", legacy_result['body_tokens'], new_result['body_tokens'])
    all_match &= compare_field("aug_tokens", legacy_result['aug_tokens'], new_result['aug_tokens'])
    all_match &= compare_field("masked_positions", legacy_result['masked_positions'], new_result['masked_positions'])
    all_match &= compare_field("labels", legacy_result['labels'], new_result['labels'])
    all_match &= compare_field("attention_mask", legacy_result['attention_mask'], new_result['attention_mask'])

    if all_match:
        print("\n✓ ALL STEPS IDENTICAL!")
        print("  → The augmentation logic is the same in both pipelines.")
        print("  → The difference must be elsewhere (loss calculation, model implementation, etc.)")
    else:
        print("\n✗ DIFFERENCES FOUND!")
        print("  → Check the specific differences above to identify the root cause.")

    return all_match


def main():
    print_section("Mode 2 Evaluation: Legacy vs New Pipeline Comparison")
    print("This script compares how Legacy (external augmentation) and New (internal augmentation)")
    print("pipelines process the same sample in Mode 2 evaluation.")

    # Setup
    set_seeds(42)

    # Create a test sample
    print_subsection("Test Sample Setup")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Add mask token if not present
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[M]'})
    mask_token_id = tokenizer.mask_token_id

    # Create a sample sentence
    text = "The quick brown fox jumps over the lazy dog and runs through the forest."
    input_ids = tokenizer.encode(text)

    print(f"  Text: {text}")
    print(f"  Token IDs: {input_ids}")
    print(f"  Sequence length: {len(input_ids)}")
    print(f"  Mask token ID: {mask_token_id}")

    # Run legacy evaluation
    legacy_result = legacy_mode2_evaluation(input_ids, tokenizer, mask_token_id, model=None)

    # Reset seed and run new evaluation
    set_seeds(42)
    new_result = new_mode2_evaluation(input_ids, tokenizer, mask_token_id, model=None)

    # Compare results
    compare_results(legacy_result, new_result, tokenizer)

    print_section("Debug Complete")
    print("Check the output above to identify where Legacy and New pipelines diverge.")


if __name__ == "__main__":
    main()
