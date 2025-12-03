#!/usr/bin/env python3
"""
Generate Deterministic Evaluation Splits

This script pre-generates fixed conditioning/evaluation splits for the
evaluation dataset. All models can then use the same splits file to ensure
fair comparison.

IMPORTANT: The distribution parameters (cond_pct_min/max, eval_pct_min/max)
should match your training configuration for fair evaluation.

Usage:
    # Match submit_conditional_moderate_cond.sh configuration:
    python scripts/generate_eval_splits.py \\
        --dataset wikitext \\
        --dataset_config wikitext-103-raw-v1 \\
        --output utils/evaluation_splits/wikitext103_valid_seed42.pt \\
        --cond_pct_min 0.0 \\
        --cond_pct_max 0.4 \\
        --eval_pct_min 1.0 \\
        --eval_pct_max 1.0 \\
        --max_cond_blocks 3 \\
        --max_eval_blocks 2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from train.augmentation import ConditionalAugmenter
from train.deterministic_eval import generate_evaluation_splits
from train.blockwise_sampling import (
    uniform_num_blocks_distribution,
    uniform_block_sizes_distribution,
)


def create_conditioning_distribution(cond_pct_min: float, cond_pct_max: float):
    """
    Create a conditioning distribution function with custom percentage range.

    Args:
        cond_pct_min: Minimum conditioning percentage (e.g., 0.0 for 0%)
        cond_pct_max: Maximum conditioning percentage (e.g., 0.4 for 40%)

    Returns:
        Distribution function that takes seq_len and returns num_conditioning
    """
    def distribution(seq_len: int) -> int:
        min_cond = max(0, int(seq_len * cond_pct_min))
        max_cond = max(min_cond, int(seq_len * cond_pct_max))
        max_cond = min(max_cond, seq_len - 1)  # Leave at least 1 for unknown
        return random.randint(min_cond, max_cond)

    return distribution


def create_evaluation_distribution(eval_pct_min: float, eval_pct_max: float):
    """
    Create an evaluation distribution function with custom percentage range.

    Args:
        eval_pct_min: Minimum evaluation percentage of unknown set (e.g., 1.0 for 100%)
        eval_pct_max: Maximum evaluation percentage of unknown set (e.g., 1.0 for 100%)

    Returns:
        Distribution function that takes available_len and returns num_evaluation
    """
    def distribution(available_len: int) -> int:
        if available_len == 0:
            return 0
        min_eval = max(1, int(available_len * eval_pct_min))
        max_eval = max(min_eval, int(available_len * eval_pct_max))
        max_eval = min(max_eval, available_len)
        return random.randint(min_eval, max_eval)

    return distribution


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate deterministic evaluation splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (matches submit_conditional_moderate_cond.sh):
  python scripts/generate_eval_splits.py \\
      --dataset wikitext \\
      --output utils/evaluation_splits/wikitext103_valid_seed42.pt

  # Custom distribution:
  python scripts/generate_eval_splits.py \\
      --dataset wikitext \\
      --output splits.pt \\
      --cond_pct_min 0.1 --cond_pct_max 0.3 \\
      --eval_pct_min 0.5 --eval_pct_max 1.0
"""
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1",
                       help="Dataset configuration")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples (None = all)")

    # Output
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path (.pt)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Distribution parameters (should match training config!)
    parser.add_argument("--cond_pct_min", type=float, default=0.0,
                       help="Minimum conditioning percentage (default: 0.0 = 0%%)")
    parser.add_argument("--cond_pct_max", type=float, default=0.4,
                       help="Maximum conditioning percentage (default: 0.4 = 40%%)")
    parser.add_argument("--eval_pct_min", type=float, default=1.0,
                       help="Minimum evaluation percentage of unknown (default: 1.0 = 100%%)")
    parser.add_argument("--eval_pct_max", type=float, default=1.0,
                       help="Maximum evaluation percentage of unknown (default: 1.0 = 100%%)")

    # Block configuration
    parser.add_argument("--max_cond_blocks", type=int, default=3,
                       help="Maximum conditioning blocks (default: 3)")
    parser.add_argument("--max_eval_blocks", type=int, default=2,
                       help="Maximum evaluation blocks (default: 2)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Generating Deterministic Evaluation Splits")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset} ({args.dataset_config}, {args.split})")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    print(f"Dataset size: {len(dataset)} samples")

    # Load tokenizer (for token management)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get mask token (assume we're using custom token)
    mask_token_id = tokenizer.convert_tokens_to_ids("[M]")
    if mask_token_id == tokenizer.unk_token_id:
        mask_token_id = 50256  # Use pad token as placeholder
        print(f"Warning: [M] token not found, using {mask_token_id} as placeholder")

    # Create custom distribution functions based on args
    print("\nDistribution configuration:")
    print(f"  Conditioning: {args.cond_pct_min*100:.0f}%-{args.cond_pct_max*100:.0f}%")
    print(f"  Evaluation: {args.eval_pct_min*100:.0f}%-{args.eval_pct_max*100:.0f}% of unknown")
    print(f"  Max cond blocks: {args.max_cond_blocks}")
    print(f"  Max eval blocks: {args.max_eval_blocks}")

    num_conditioning_distribution = create_conditioning_distribution(
        args.cond_pct_min, args.cond_pct_max
    )
    num_evaluation_distribution = create_evaluation_distribution(
        args.eval_pct_min, args.eval_pct_max
    )

    # Create augmenter with custom distributions
    print("\nCreating augmenter...")
    augmenter = ConditionalAugmenter(
        mask_token_id=mask_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        max_seq_len=1024,
        num_conditioning_distribution=num_conditioning_distribution,
        num_blocks_distribution=uniform_num_blocks_distribution,
        block_sizes_distribution=uniform_block_sizes_distribution,
        num_evaluation_distribution=num_evaluation_distribution,
        num_eval_blocks_distribution=uniform_num_blocks_distribution,
        eval_block_sizes_distribution=uniform_block_sizes_distribution,
        conditioning_sampling='blockwise',
        evaluation_sampling='blockwise',
        max_cond_blocks=args.max_cond_blocks,
        max_eval_blocks=args.max_eval_blocks,
        tokenizer_pad_token_id=tokenizer.pad_token_id,
        ordering_mode='temporal'  # Doesn't matter for split generation
    )

    # Generate splits
    output_file = generate_evaluation_splits(
        dataset=dataset,
        augmenter=augmenter,
        output_file=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("✓ Done!")
    print("\nConfiguration used:")
    print(f"  --cond_pct_min {args.cond_pct_min} --cond_pct_max {args.cond_pct_max}")
    print(f"  --eval_pct_min {args.eval_pct_min} --eval_pct_max {args.eval_pct_max}")
    print("\nNext steps:")
    print("  1. Use this file for training:")
    print(f"     python train_sigmagpt.py --eval_splits_file {output_file}")
    print(f"     python train.py --eval_splits_file {output_file}")
    print("\n  2. This ensures all models are evaluated on identical tasks")
    print("=" * 80)


if __name__ == "__main__":
    main()
