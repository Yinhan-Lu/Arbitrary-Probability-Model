"""
Deterministic Evaluation with Fixed Conditioning/Evaluation Splits

This module ensures fair comparison between different models by evaluating
them on exactly the same conditional probability tasks.

Problem:
- Without deterministic splits, each evaluation run uses different random
  conditioning/evaluation splits
- This makes comparison between models unfair (they're solving different tasks)
- Results are non-reproducible across runs

Solution:
- Pre-generate fixed splits for the evaluation dataset
- All models use the same splits file
- Ensures fair, reproducible comparison

Usage:
    # Step 1: Generate splits once
    python scripts/generate_eval_splits.py --output splits.pt

    # Step 2: Use for all models
    python train_sigmagpt.py --eval_splits_file splits.pt
    python train_conditional.py --eval_splits_file splits.pt
"""

import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class DeterministicEvaluationSplits:
    """
    Manager for deterministic evaluation splits.

    This class provides two modes:
    1. Pre-loaded splits: Load from a pre-generated file
    2. On-the-fly deterministic: Generate splits deterministically using sample index as seed
    """

    def __init__(self, splits_file: Optional[str] = None):
        """
        Initialize splits manager.

        Args:
            splits_file: Path to pre-generated splits file (.pt file)
                        If None, will generate splits on-the-fly (deterministic with seed)
                        If provided but file doesn't exist, will warn and fall back to on-the-fly
        """
        self.splits_file = splits_file
        self.splits = None

        if splits_file:
            splits_path = Path(splits_file)
            if splits_path.exists():
                self.splits = torch.load(splits_file)
                print(f"✓ Loaded {len(self.splits)} pre-generated evaluation splits")
                print(f"  from {splits_file}")
            else:
                print(f"⚠ Warning: Splits file not found: {splits_file}")
                print(f"  Will generate splits deterministically on-the-fly")

    def get_split(
        self,
        sample_idx: int,
        seq_len: int,
        augmenter
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Get deterministic split for a specific sample.

        Args:
            sample_idx: Index of sample in evaluation dataset
            seq_len: Sequence length
            augmenter: ConditionalAugmenter instance (used for split generation)

        Returns:
            Tuple of (cond_indices, eval_indices, unseen_indices)

        Notes:
            - If using pre-loaded splits, returns the stored split
            - If generating on-the-fly, uses sample_idx as seed for determinism
            - Same sample_idx always produces same split (deterministic)
        """
        if self.splits and sample_idx in self.splits:
            # Use pre-generated split
            split = self.splits[sample_idx]
            return split['cond'], split['eval'], split['unseen']
        else:
            # Generate deterministically using sample index as seed
            # This ensures same split for same sample across runs
            old_random_state = random.getstate()
            old_np_state = np.random.get_state()
            old_torch_state = torch.get_rng_state()

            try:
                # Deterministic seed based on sample index
                seed = 42 + sample_idx
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                # Generate split
                cond_idx, eval_idx, unseen_idx = augmenter.split_indices(seq_len)

                return cond_idx, eval_idx, unseen_idx

            finally:
                # Restore random state
                random.setstate(old_random_state)
                np.random.set_state(old_np_state)
                torch.set_rng_state(old_torch_state)


def generate_evaluation_splits(
    dataset,
    augmenter,
    output_file: str,
    num_samples: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True
) -> str:
    """
    Pre-generate fixed evaluation splits for entire dataset.

    This should be run ONCE to generate splits, then all models
    use the same splits file for fair comparison.

    Args:
        dataset: Evaluation dataset (e.g., WikiText-103 validation set)
        augmenter: ConditionalAugmenter instance with desired configuration
        output_file: Where to save splits (.pt file)
        num_samples: Number of samples to generate (None = all)
        seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        Path to saved splits file

    Example:
        >>> from datasets import load_dataset
        >>> from train.augmentation import ConditionalAugmenter
        >>>
        >>> dataset = load_dataset('wikitext', 'wikitext-103-v1', split='validation')
        >>> augmenter = ConditionalAugmenter(
        ...     conditioning_ratio=0.6,
        ...     conditioning_sampling='blockwise',
        ...     evaluation_sampling='blockwise'
        ... )
        >>>
        >>> splits_file = generate_evaluation_splits(
        ...     dataset=dataset,
        ...     augmenter=augmenter,
        ...     output_file='eval_splits.pt'
        ... )
    """
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print("=" * 80)
        print("Generating Deterministic Evaluation Splits")
        print("=" * 80)
        print(f"\nSeed: {seed}")
        print(f"Dataset: {len(dataset)} samples")
        print(f"Augmenter configuration:")
        print(f"  - conditioning_sampling: {augmenter.conditioning_sampling}")
        print(f"  - evaluation_sampling: {augmenter.evaluation_sampling}")
        print(f"  - conditioning_ratio: {augmenter.conditioning_ratio}")
        print(f"  - max_cond_blocks: {augmenter.max_cond_blocks}")
        print(f"  - max_eval_blocks: {augmenter.max_eval_blocks}")

    splits = {}
    num_samples = num_samples or len(dataset)

    if verbose:
        print(f"\nGenerating {num_samples} splits...")
        iterator = tqdm(range(num_samples), desc="Generating splits")
    else:
        iterator = range(num_samples)

    for idx in iterator:
        # Get sample
        sample = dataset[idx]

        # Extract sequence length
        if isinstance(sample, dict):
            if 'input_ids' in sample:
                seq_len = len(sample['input_ids'])
            elif 'text' in sample:
                # Need to tokenize
                # This is a simplified version - real implementation should use tokenizer
                seq_len = len(sample['text'].split())
            else:
                raise ValueError(f"Unknown sample format: {sample.keys()}")
        else:
            seq_len = len(sample)

        # Generate split
        cond_idx, eval_idx, unseen_idx = augmenter.split_indices(seq_len)

        # Store split
        splits[idx] = {
            'cond': cond_idx,
            'eval': eval_idx,
            'unseen': unseen_idx,
            'seq_len': seq_len
        }

    # Save splits
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(splits, output_file)

    if verbose:
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"\n✓ Saved {len(splits)} splits to {output_file}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print("\nUsage:")
        print(f"  python train_sigmagpt.py --eval_splits_file {output_file}")
        print(f"  python train_conditional.py --eval_splits_file {output_file}")
        print("=" * 80)

    return output_file


def verify_splits_file(splits_file: str, verbose: bool = True) -> Dict:
    """
    Verify and inspect a splits file.

    Args:
        splits_file: Path to splits file
        verbose: Whether to print detailed information

    Returns:
        Dictionary with statistics about the splits

    Example:
        >>> stats = verify_splits_file('eval_splits.pt')
        >>> print(f"Number of splits: {stats['num_splits']}")
    """
    if not Path(splits_file).exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")

    splits = torch.load(splits_file)

    stats = {
        'num_splits': len(splits),
        'avg_seq_len': np.mean([s['seq_len'] for s in splits.values()]),
        'avg_cond_size': np.mean([len(s['cond']) for s in splits.values()]),
        'avg_eval_size': np.mean([len(s['eval']) for s in splits.values()]),
        'avg_unseen_size': np.mean([len(s['unseen']) for s in splits.values()]),
    }

    if verbose:
        print("=" * 80)
        print(f"Splits File: {splits_file}")
        print("=" * 80)
        print(f"Number of splits: {stats['num_splits']}")
        print(f"Average sequence length: {stats['avg_seq_len']:.1f}")
        print(f"Average conditioning size: {stats['avg_cond_size']:.1f}")
        print(f"Average evaluation size: {stats['avg_eval_size']:.1f}")
        print(f"Average unseen size: {stats['avg_unseen_size']:.1f}")

        # Sample a few splits
        print("\nSample splits:")
        for idx in list(splits.keys())[:3]:
            split = splits[idx]
            print(f"  Sample {idx}:")
            print(f"    Cond: {split['cond'][:10]}{'...' if len(split['cond']) > 10 else ''}")
            print(f"    Eval: {split['eval'][:10]}{'...' if len(split['eval']) > 10 else ''}")
        print("=" * 80)

    return stats


# Example usage
if __name__ == "__main__":
    print("Deterministic Evaluation Splits Module")
    print("=" * 80)
    print("\nThis module provides tools for fair model comparison.")
    print("\nTo generate splits:")
    print("  python scripts/generate_eval_splits.py --output splits.pt")
    print("\nTo use in training:")
    print("  python train_sigmagpt.py --eval_splits_file splits.pt")
    print("=" * 80)
