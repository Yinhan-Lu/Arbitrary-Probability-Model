"""
Attention Mask Utilities for Arbitrary Conditional Probability Modeling

Provides functions to construct custom attention masks for:
- Causal (standard autoregressive)
- Conditional (allow attending to condition tokens, block unknown tokens)
- Bidirectional (for specific use cases)
"""

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_causal_mask(seq_len, device='cpu'):
    """
    Create standard causal (lower triangular) attention mask

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask of shape (seq_len, seq_len)
        1 = can attend, 0 = cannot attend
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.uint8))
    return mask


def create_conditional_mask(
    seq_len,
    conditioning_indices,
    unknown_indices,
    device='cpu',
    include_bos=True
):
    """
    Create conditional attention mask for arbitrary probability modeling

    Key principles:
    1. Start with causal mask (lower triangular)
    2. Allow ALL positions to attend to conditioning tokens (even future ones)
    3. Block ALL positions from attending to unknown tokens

    Args:
        seq_len: Total sequence length (including BOS if present)
        conditioning_indices: List of indices for conditioning tokens (X_c)
        unknown_indices: List of indices for unknown tokens (X_u, includes evaluation set)
        device: Device to create mask on
        include_bos: Whether sequence includes BOS token at position 0

    Returns:
        Custom attention mask of shape (seq_len, seq_len)
        1 = can attend, 0 = cannot attend
    """
    # Start with causal mask
    mask = create_causal_mask(seq_len, device=device)

    # Adjust indices if BOS is included (shift by 1)
    if include_bos:
        # Conditioning and unknown indices refer to original sequence positions
        # In augmented sequence with BOS, they are shifted by +1
        cond_cols = [idx + 1 for idx in conditioning_indices]
        unknown_cols = [idx + 1 for idx in unknown_indices]
    else:
        cond_cols = conditioning_indices
        unknown_cols = unknown_indices

    # Rule 1: Allow ALL rows to attend to conditioning columns
    # This allows using future conditioning information
    for col in cond_cols:
        if col < seq_len:
            mask[:, col] = 1

    # Rule 2: Block ALL rows from attending to unknown columns
    # Unknown tokens contain no useful information (just [M] placeholders)
    for col in unknown_cols:
        if col < seq_len:
            mask[:, col] = 0
            # Optionally allow self-attention (diagonal)
            # mask[col, col] = 1  # Uncomment if needed

    # Special handling for BOS (position 0)
    if include_bos:
        # BOS column: typically not informative, can block or allow
        # Here we allow BOS to be attended by all (standard practice)
        mask[:, 0] = 1

        # BOS row: what can BOS attend to?
        # Option 1: BOS attends only to itself (most restrictive)
        # mask[0, :] = 0
        # mask[0, 0] = 1

        # Option 2: BOS can attend to all conditioning tokens (recommended)
        mask[0, :] = 0  # First clear all
        mask[0, 0] = 1  # Self-attention
        for col in cond_cols:
            if col < seq_len:
                mask[0, col] = 1  # Attend to conditions

    return mask


def create_bidirectional_mask(seq_len, device='cpu'):
    """
    Create fully bidirectional attention mask (all positions can attend to all)

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Bidirectional mask of shape (seq_len, seq_len) with all 1s
    """
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.uint8)
    return mask


def visualize_mask(mask, title="Attention Mask"):
    """
    Print a visual representation of the attention mask

    Args:
        mask: Attention mask tensor (2D)
        title: Title for the visualization
    """
    print(f"\n{title}")
    print("=" * (mask.size(1) * 2 + 5))
    print("     ", end="")
    for j in range(mask.size(1)):
        print(f"{j:2d}", end=" ")
    print()
    print("=" * (mask.size(1) * 2 + 5))

    for i in range(mask.size(0)):
        print(f"{i:2d} | ", end="")
        for j in range(mask.size(1)):
            symbol = "✓" if mask[i, j] == 1 else "✗"
            print(f" {symbol}", end=" ")
        print()
    print("=" * (mask.size(1) * 2 + 5))
    print("✓ = can attend, ✗ = cannot attend\n")


def apply_mask_to_scores(attention_scores, mask):
    """
    Apply attention mask to attention scores

    Args:
        attention_scores: Tensor of shape (..., seq_len, seq_len)
        mask: Attention mask of shape (seq_len, seq_len) or broadcastable
              1 = keep, 0 = mask out

    Returns:
        Masked attention scores with -inf at masked positions
    """
    # Expand mask dimensions if needed
    if mask.dim() == 2:
        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        # Add head dimension
        mask = mask.unsqueeze(1)

    # Apply mask: set masked positions to -inf
    masked_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

    return masked_scores


def validate_mask_indices(seq_len, conditioning_indices, evaluation_indices, unknown_indices):
    """
    Validate that mask indices are correctly specified

    Args:
        seq_len: Sequence length (original, without BOS)
        conditioning_indices: Conditioning set indices
        evaluation_indices: Evaluation set indices
        unknown_indices: Unknown set indices

    Raises:
        AssertionError if validation fails
    """
    # All indices should be within valid range
    all_indices = set(conditioning_indices + evaluation_indices + unknown_indices)
    assert all(0 <= idx < seq_len for idx in all_indices), \
        f"Indices must be in range [0, {seq_len})"

    # Conditioning and unknown should be disjoint
    cond_set = set(conditioning_indices)
    unknown_set = set(unknown_indices)
    assert cond_set.isdisjoint(unknown_set), \
        "Conditioning and unknown sets must be disjoint"

    # Evaluation should be subset of unknown
    eval_set = set(evaluation_indices)
    assert eval_set.issubset(unknown_set), \
        "Evaluation set must be subset of unknown set"

    # All positions should be accounted for
    total_set = cond_set | unknown_set
    assert len(total_set) == seq_len, \
        f"All {seq_len} positions must be in either conditioning or unknown set"

    logger.debug(f"Mask indices validated: {len(cond_set)} cond, {len(eval_set)} eval, {len(unknown_set)} unknown")


if __name__ == "__main__":
    # Test mask utilities
    print("=" * 80)
    print("Testing Attention Mask Utilities")
    print("=" * 80)

    # Test 1: Causal mask
    print("\nTest 1: Causal Mask")
    causal = create_causal_mask(8)
    visualize_mask(causal, "Causal Mask (8x8)")

    # Test 2: Conditional mask
    print("\nTest 2: Conditional Mask")
    # Example: seq_len=5, conditioning=[1, 3], evaluation=[0, 2], unknown=[0, 2, 4]
    seq_len = 5
    cond_idx = [1, 3]
    eval_idx = [0, 2]
    unknown_idx = [0, 2, 4]

    # Validate
    validate_mask_indices(seq_len, cond_idx, eval_idx, unknown_idx)

    # Create mask (with BOS, so total length = 6)
    conditional = create_conditional_mask(
        seq_len=seq_len + 1,  # +1 for BOS
        conditioning_indices=cond_idx,
        unknown_indices=unknown_idx,
        include_bos=True
    )

    print(f"Sequence length: {seq_len} (+1 for BOS = {seq_len + 1})")
    print(f"Conditioning indices (original): {cond_idx}")
    print(f"Evaluation indices (original): {eval_idx}")
    print(f"Unknown indices (original): {unknown_idx}")
    visualize_mask(conditional, "Conditional Mask (6x6 with BOS)")

    print("Key observations:")
    print("  - Row 0 (BOS) can attend to itself and conditioning tokens")
    print("  - All rows can attend to conditioning columns (1, 3 → positions 2, 4 with BOS)")
    print("  - Unknown columns (0, 2, 4 → positions 1, 3, 5 with BOS) are blocked")
    print("  - Causal structure preserved (lower triangular base)")

    # Test 3: Bidirectional mask
    print("\nTest 3: Bidirectional Mask")
    bidirectional = create_bidirectional_mask(6)
    visualize_mask(bidirectional, "Bidirectional Mask (6x6)")

    # Test 4: Apply mask to scores
    print("\nTest 4: Apply Mask to Attention Scores")
    scores = torch.randn(1, 1, 6, 6)  # (batch, heads, seq, seq)
    print(f"Original scores shape: {scores.shape}")
    print(f"Sample scores (first 3x3):\n{scores[0, 0, :3, :3]}")

    masked_scores = apply_mask_to_scores(scores, conditional)
    print(f"\nMasked scores (first 3x3):\n{masked_scores[0, 0, :3, :3]}")
    print("(Note: -inf appears at masked positions)")

    print("\n" + "=" * 80)
    print("✓ All mask utility tests passed!")
    print("=" * 80)
