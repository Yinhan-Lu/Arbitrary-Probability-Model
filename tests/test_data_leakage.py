"""
Test for data leakage between train and validation splits

This script checks:
1. Train and validation datasets are completely separate
2. No overlapping samples between train and val
3. Evaluation uses different data than training

Run from project root: python tests/test_data_leakage.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.config import get_config
from train.dataset import get_dataloader

def test_data_separation():
    """Test that train and validation datasets are completely separate"""

    print("=" * 80)
    print("DATA LEAKAGE TEST")
    print("=" * 80)

    # Get config
    config = get_config("tiny")

    # Load small samples from both splits
    print("\n[1] Loading train and validation datasets...")
    print("    Loading 100 samples from each split...")

    train_loader = get_dataloader(
        config=config,
        split="train",
        batch_size=1,
        num_workers=0,
        streaming=False,
        num_samples=100,
        collate_fn=None
    )

    val_loader = get_dataloader(
        config=config,
        split="validation",
        batch_size=1,
        num_workers=0,
        streaming=False,
        num_samples=100,
        collate_fn=None
    )

    print(f"    ✓ Train dataset loaded: {len(train_loader)} batches")
    print(f"    ✓ Val dataset loaded: {len(val_loader)} batches")

    # Collect samples
    print("\n[2] Collecting samples from train split...")
    train_samples = []
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 100:
            break
        train_samples.append(batch['input_ids'].squeeze(0))

    print(f"    ✓ Collected {len(train_samples)} train samples")

    print("\n[3] Collecting samples from validation split...")
    val_samples = []
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= 100:
            break
        val_samples.append(batch['input_ids'].squeeze(0))

    print(f"    ✓ Collected {len(val_samples)} validation samples")

    # Check for exact matches
    print("\n[4] Checking for exact duplicates between train and val...")
    num_duplicates = 0
    duplicate_examples = []

    for i, train_sample in enumerate(train_samples):
        for j, val_sample in enumerate(val_samples):
            if train_sample.shape == val_sample.shape:
                if torch.equal(train_sample, val_sample):
                    num_duplicates += 1
                    if len(duplicate_examples) < 3:  # Store first 3 examples
                        duplicate_examples.append((i, j, train_sample[:20]))

    if num_duplicates > 0:
        print(f"    ✗ FOUND {num_duplicates} EXACT DUPLICATES!")
        print(f"    This indicates DATA LEAKAGE!")
        print(f"\n    First few duplicates:")
        for train_idx, val_idx, tokens in duplicate_examples:
            print(f"      Train[{train_idx}] == Val[{val_idx}]")
            print(f"      First 20 tokens: {tokens.tolist()}")
        return False
    else:
        print(f"    ✓ No exact duplicates found (good!)")

    # Check for partial overlaps (first 50 tokens)
    print("\n[5] Checking for partial overlaps (first 50 tokens)...")
    num_partial_overlaps = 0
    overlap_examples = []

    for i, train_sample in enumerate(train_samples):
        train_prefix = train_sample[:50]
        for j, val_sample in enumerate(val_samples):
            val_prefix = val_sample[:50]
            if train_prefix.shape == val_prefix.shape:
                if torch.equal(train_prefix, val_prefix):
                    num_partial_overlaps += 1
                    if len(overlap_examples) < 3:
                        overlap_examples.append((i, j, train_prefix[:20]))

    if num_partial_overlaps > 0:
        print(f"    ⚠ FOUND {num_partial_overlaps} PARTIAL OVERLAPS")
        print(f"    This might indicate data leakage!")
        print(f"\n    First few partial overlaps:")
        for train_idx, val_idx, tokens in overlap_examples:
            print(f"      Train[{train_idx}] prefix == Val[{val_idx}] prefix")
            print(f"      First 20 tokens: {tokens.tolist()}")
        return False
    else:
        print(f"    ✓ No partial overlaps found (good!)")

    # Sample statistics
    print("\n[6] Dataset statistics:")
    print(f"    Train samples checked: {len(train_samples)}")
    print(f"    Val samples checked: {len(val_samples)}")
    print(f"    Avg train length: {sum(s.shape[0] for s in train_samples) / len(train_samples):.1f}")
    print(f"    Avg val length: {sum(s.shape[0] for s in val_samples) / len(val_samples):.1f}")

    # Token distribution check
    print("\n[7] Token distribution check:")
    train_vocab = set()
    val_vocab = set()

    for sample in train_samples:
        train_vocab.update(sample.tolist())

    for sample in val_samples:
        val_vocab.update(sample.tolist())

    print(f"    Train unique tokens: {len(train_vocab)}")
    print(f"    Val unique tokens: {len(val_vocab)}")
    print(f"    Overlap: {len(train_vocab & val_vocab)} tokens")
    print(f"    Train-only: {len(train_vocab - val_vocab)} tokens")
    print(f"    Val-only: {len(val_vocab - train_vocab)} tokens")

    # Print first few tokens from each split
    print("\n[8] Sample previews:")
    print(f"\n    Train sample 0 (first 30 tokens):")
    print(f"      {train_samples[0][:30].tolist()}")
    print(f"\n    Val sample 0 (first 30 tokens):")
    print(f"      {val_samples[0][:30].tolist()}")

    print("\n" + "=" * 80)
    print("DATA LEAKAGE TEST RESULT")
    print("=" * 80)
    print("✓ No data leakage detected!")
    print("  - No exact duplicates between train and validation")
    print("  - No partial overlaps between splits")
    print("  - Datasets are properly separated")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_data_separation()
    sys.exit(0 if success else 1)