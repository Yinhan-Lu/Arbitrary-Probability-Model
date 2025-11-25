#!/usr/bin/env python3
"""
Generate Deterministic Evaluation Splits

This script pre-generates fixed conditioning/evaluation splits for the
evaluation dataset. All models can then use the same splits file to ensure
fair comparison.

Usage:
    python scripts/generate_eval_splits.py \\
        --dataset wikitext \\
        --dataset_config wikitext-103-raw-v1 \\
        --output utils/evaluation_splits/wikitext103_valid_seed42.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from train.augmentation import ConditionalAugmenter
from train.deterministic_eval import generate_evaluation_splits
from train.blockwise_sampling import (
    uniform_num_conditioning_distribution,
    uniform_num_blocks_distribution,
    uniform_block_sizes_distribution,
    uniform_num_evaluation_distribution,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate deterministic evaluation splits")

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

    # Augmenter config (should match training)
    parser.add_argument("--max_cond_blocks", type=int, default=2,
                       help="Maximum conditioning blocks")
    parser.add_argument("--max_eval_blocks", type=int, default=1,
                       help="Maximum evaluation blocks")

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
    # This is a simplified version - real implementation should match training
    mask_token_id = tokenizer.convert_tokens_to_ids("[M]")
    if mask_token_id == tokenizer.unk_token_id:
        # Token doesn't exist, use a placeholder
        mask_token_id = 50256  # Use pad token as placeholder
        print(f"Warning: [M] token not found, using {mask_token_id} as placeholder")

    # Create augmenter with same config as training
    print("\nCreating augmenter...")
    augmenter = ConditionalAugmenter(
        mask_token_id=mask_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        max_seq_len=1024,
        num_conditioning_distribution=uniform_num_conditioning_distribution,
        num_blocks_distribution=uniform_num_blocks_distribution,
        block_sizes_distribution=uniform_block_sizes_distribution,
        num_evaluation_distribution=uniform_num_evaluation_distribution,
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
    print("\nNext steps:")
    print("  1. Use this file for training:")
    print(f"     python train_sigmagpt.py --eval_splits_file {output_file}")
    print(f"     python train_conditional.py --eval_splits_file {output_file}")
    print("\n  2. This ensures all models are evaluated on identical tasks")
    print("=" * 80)


if __name__ == "__main__":
    main()
