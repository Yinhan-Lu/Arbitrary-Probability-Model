"""
Debug script for visualizing attention masks with short sequences.

This script helps you understand and verify attention mask behavior
by using very short sequences (8 tokens) instead of 1024.

Usage:
    # Just run to see mask visualizations
    python tests/debug_attention_mask.py

    # Use debugger to step through
    python -m pdb tests/debug_attention_mask.py

    # VSCode: set breakpoints and press F5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from train.mask_utils import (
    create_causal_mask,
    create_conditional_mask,
    create_prefix_conditional_mask,
    visualize_mask
)
from model.config import get_config
from model.arbitrary_prob_gpt2 import GPT2Model

# Use very short sequence for debugging
SEQ_LEN = 8
BATCH_SIZE = 1


def debug_masks():
    """Visualize different mask types."""
    print("=" * 60)
    print("PART 1: VISUALIZE DIFFERENT MASK TYPES")
    print("=" * 60)

    # 1. Causal mask (standard autoregressive)
    print("\n" + "=" * 50)
    print("1. CAUSAL MASK (Standard Autoregressive)")
    print("=" * 50)
    print("Rule: Each position can only see itself and previous positions")
    causal = create_causal_mask(SEQ_LEN)
    visualize_mask(causal, f"Causal Mask (seq_len={SEQ_LEN})")

    # 2. Conditional mask
    print("\n" + "=" * 50)
    print("2. CONDITIONAL MASK")
    print("=" * 50)
    print("Conditioning on positions [1, 2], unknown positions [5, 6]")
    print("Rule: All positions can see conditioning tokens; no one sees unknown tokens")
    cond = create_conditional_mask(
        seq_len=SEQ_LEN,
        conditioning_indices=[1, 2],
        unknown_indices=[5, 6]
    )
    visualize_mask(cond, "Conditional Mask (cond=[1,2], unknown=[5,6])")

    # 3. Prefix conditional mask
    print("\n" + "=" * 50)
    print("3. PREFIX CONDITIONAL MASK")
    print("=" * 50)
    print("N_cond=3 conditioning tokens, N_seq=5 sequence tokens")
    print("Total length = 8 = 3 + 5")
    print("Rule: Prefix (rows 0-2) only sees prefix; Body (rows 3-7) sees prefix + causal body")
    prefix_cond = create_prefix_conditional_mask(N_cond=3, N_seq=5)
    visualize_mask(prefix_cond, "Prefix Conditional (N_cond=3, N_seq=5)")


def debug_model_forward():
    """Debug model forward pass - single test case with no unseen tokens."""
    print("\n" + "=" * 60)
    print("CONDITIONAL FORWARD (unseen_idx=[] empty)")
    print("=" * 60)

    # Setup
    config = get_config("nano")
    config.max_seq_len = 32
    BOS_TOKEN_ID = 50256
    MASK_TOKEN_ID = 50257

    model = GPT2Model(config, mask_token_id=MASK_TOKEN_ID, bos_token_id=BOS_TOKEN_ID)
    model.eval()

    # Input: 8 tokens at positions 0-7
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN))

    print(f"Input tokens: {input_ids[0].tolist()}")
    print(f"Positions: 0, 1, 2, 3, 4, 5, 6, 7")
    print(f"  - conditional_idx: [1, 2, 7] (some AFTER eval positions!)")
    print(f"  - unseen_idx: [0, 3, 4, 5, 6] (all positions NOT in conditional)")
    print(f"  - evaluation_idx: [3, 4] (subset of unseen, where loss is computed)")

    conditional_idx = [[1, 2, 7]]  # Position 7 is AFTER eval positions 3,4
    unseen_idx = [[0, 3, 4, 5, 6]]  # All positions not in conditional_idx
    evaluation_idx = [[3, 4]]  # Subset of unseen_idx

    with torch.no_grad():
        output = model(
            input_ids,
            conditional_idx=conditional_idx,
            evaluation_idx=evaluation_idx,
            unseen_idx=unseen_idx
        )
        print(f"\nOutput logits shape: {output['logits'].shape}")
        print(f"Loss: {output.get('loss', 'N/A')}")


def debug_with_breakpoint():
    """Interactive debugging with breakpoint."""
    print("\n" + "=" * 60)
    print("PART 3: INTERACTIVE DEBUGGING")
    print("=" * 60)
    print("\nA breakpoint will be set. When it triggers, you can:")
    print("  - Inspect 'mask' variable")
    print("  - Try: visualize_mask(mask, 'Current Mask')")
    print("  - Type 'c' to continue, 'q' to quit")
    print("\nPress Enter to continue to breakpoint...")
    input()

    # Create a sample mask
    mask = create_prefix_conditional_mask(N_cond=3, N_seq=5)

    # Set breakpoint here - inspect 'mask' variable
    breakpoint()  # <-- DEBUGGER STOPS HERE

    print("\nBreakpoint section completed.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ATTENTION MASK DEBUG SCRIPT")
    print("=" * 60)
    print(f"Using short sequence length: {SEQ_LEN} tokens")
    print("This makes it easy to visualize and understand masks\n")

    # Run visualization
    debug_masks()

    # Run model forward pass
    debug_model_forward()

    # Optional: interactive debugging
    print("\n" + "-" * 40)
    response = input("Run interactive breakpoint debugging? (y/n): ")
    if response.lower() == 'y':
        debug_with_breakpoint()

    print("\n" + "=" * 60)
    print("DEBUG SESSION COMPLETE")
    print("=" * 60)
