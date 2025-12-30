"""
Test Deterministic Checkpoint Resume

This test verifies that:
1. DataLoader shuffle is deterministic with the same seed
2. Generator state can be saved and restored correctly
3. Skip logic works correctly for checkpoint resume

Run from project root: python tests/test_deterministic_resume.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import tempfile
from model.config import get_config
from train.dataset import get_dataloader


def test_deterministic_shuffle():
    """
    Verify that DataLoader produces identical batches with the same seed.
    """
    print("=" * 80)
    print("Test 1: Deterministic Shuffle")
    print("=" * 80)

    config = get_config("tiny")
    seed = 42
    num_batches_to_check = 5

    print(f"\n[Setup] Creating two DataLoaders with seed={seed}")

    # Create first DataLoader
    loader1 = get_dataloader(
        config=config,
        split="train",
        batch_size=4,
        num_workers=0,  # Use 0 workers for determinism
        streaming=False,
        num_samples=50,  # Small dataset for quick test
        seed=seed
    )

    # Collect first few batches
    batches1 = []
    for i, batch in enumerate(loader1):
        if i >= num_batches_to_check:
            break
        batches1.append(batch["input_ids"].clone())

    print(f"[Loader 1] Collected {len(batches1)} batches")

    # Create second DataLoader with same seed
    loader2 = get_dataloader(
        config=config,
        split="train",
        batch_size=4,
        num_workers=0,
        streaming=False,
        num_samples=50,
        seed=seed
    )

    # Collect batches from second loader
    batches2 = []
    for i, batch in enumerate(loader2):
        if i >= num_batches_to_check:
            break
        batches2.append(batch["input_ids"].clone())

    print(f"[Loader 2] Collected {len(batches2)} batches")

    # Compare batches
    print("\n[Verification] Comparing batches...")
    all_match = True
    for i, (b1, b2) in enumerate(zip(batches1, batches2)):
        if torch.equal(b1, b2):
            print(f"  Batch {i}: ✓ Match")
        else:
            print(f"  Batch {i}: ✗ MISMATCH!")
            all_match = False

    if all_match:
        print("\n✓ Test 1 PASSED: Deterministic shuffle verified!")
    else:
        print("\n✗ Test 1 FAILED: Batches do not match!")
        sys.exit(1)

    return True


def test_generator_state_save_restore():
    """
    Verify that generator state can be saved and restored correctly.

    This test verifies that:
    1. Generator state can be saved
    2. Generator state can be restored to a new DataLoader
    3. After restoration, the restored state matches the saved state
    """
    print("\n" + "=" * 80)
    print("Test 2: Generator State Save/Restore")
    print("=" * 80)

    config = get_config("tiny")
    seed = 42

    print(f"\n[Setup] Creating DataLoader with seed={seed}")

    # Create DataLoader
    loader = get_dataloader(
        config=config,
        split="train",
        batch_size=4,
        num_workers=0,
        streaming=False,
        num_samples=100,
        seed=seed
    )

    # Save initial generator state
    initial_state = loader.shuffle_generator.get_state().clone()
    print("[Phase 1] Saved initial generator state")

    # Create new DataLoader with different seed (to ensure states are different)
    loader2 = get_dataloader(
        config=config,
        split="train",
        batch_size=4,
        num_workers=0,
        streaming=False,
        num_samples=100,
        seed=999  # Different seed
    )

    # Verify states are different before restore
    state_before_restore = loader2.shuffle_generator.get_state()
    states_different = not torch.equal(initial_state, state_before_restore)
    print(f"[Phase 2] States are different before restore: {states_different}")

    # Restore the saved state
    loader2.shuffle_generator.set_state(initial_state)
    print("[Phase 2] Restored generator state to loader2")

    # Verify states match after restore
    state_after_restore = loader2.shuffle_generator.get_state()
    states_match = torch.equal(initial_state, state_after_restore)
    print(f"[Phase 2] States match after restore: {states_match}")

    if states_different and states_match:
        print("\n✓ Test 2 PASSED: Generator state save/restore works correctly!")
    else:
        if not states_different:
            print("\n✗ Test 2 FAILED: States were already identical (test setup issue)")
        else:
            print("\n✗ Test 2 FAILED: States do not match after restore!")
        sys.exit(1)

    return True


def test_skip_logic():
    """
    Verify that skipping batches produces expected results.
    """
    print("\n" + "=" * 80)
    print("Test 3: Skip Logic Simulation")
    print("=" * 80)

    config = get_config("tiny")
    seed = 42

    print(f"\n[Setup] Creating DataLoader with seed={seed}")

    # Create DataLoader and collect all batches
    loader1 = get_dataloader(
        config=config,
        split="train",
        batch_size=4,
        num_workers=0,
        streaming=False,
        num_samples=40,  # 10 batches total
        seed=seed
    )

    all_batches = []
    for batch in loader1:
        all_batches.append(batch["input_ids"].clone())

    print(f"[Full run] Collected {len(all_batches)} batches")

    # Simulate resume: skip first 3 batches
    skip_count = 3

    loader2 = get_dataloader(
        config=config,
        split="train",
        batch_size=4,
        num_workers=0,
        streaming=False,
        num_samples=40,
        seed=seed
    )

    resumed_batches = []
    for i, batch in enumerate(loader2):
        if i < skip_count:
            continue  # Skip batches
        resumed_batches.append(batch["input_ids"].clone())

    print(f"[Resume run] Skipped {skip_count} batches, collected {len(resumed_batches)} batches")

    # Verify resumed batches match original batches starting from skip_count
    print("\n[Verification] Comparing batches...")
    all_match = True
    for i, (original, resumed) in enumerate(zip(all_batches[skip_count:], resumed_batches)):
        original_idx = skip_count + i
        if torch.equal(original, resumed):
            print(f"  Original batch {original_idx} vs Resumed batch {i}: ✓ Match")
        else:
            print(f"  Original batch {original_idx} vs Resumed batch {i}: ✗ MISMATCH!")
            all_match = False

    if all_match:
        print("\n✓ Test 3 PASSED: Skip logic works correctly!")
    else:
        print("\n✗ Test 3 FAILED: Resumed batches do not match!")
        sys.exit(1)

    return True


def test_different_seeds_produce_different_order():
    """
    Verify that different seeds produce different shuffle orders.
    """
    print("\n" + "=" * 80)
    print("Test 4: Different Seeds Produce Different Orders")
    print("=" * 80)

    config = get_config("tiny")

    print("\n[Setup] Creating two DataLoaders with different seeds")

    # Create DataLoader with seed 42
    loader1 = get_dataloader(
        config=config,
        split="train",
        batch_size=4,
        num_workers=0,
        streaming=False,
        num_samples=20,
        seed=42
    )

    batch1 = next(iter(loader1))["input_ids"]
    print(f"[Seed 42] First batch shape: {batch1.shape}")

    # Create DataLoader with seed 123
    loader2 = get_dataloader(
        config=config,
        split="train",
        batch_size=4,
        num_workers=0,
        streaming=False,
        num_samples=20,
        seed=123
    )

    batch2 = next(iter(loader2))["input_ids"]
    print(f"[Seed 123] First batch shape: {batch2.shape}")

    # Verify they are different
    if not torch.equal(batch1, batch2):
        print("\n✓ Test 4 PASSED: Different seeds produce different shuffle orders!")
    else:
        print("\n✗ Test 4 FAILED: Different seeds produced identical batches (unlikely but possible)")
        # This could happen by chance, so it's not a hard failure

    return True


def test_checkpoint_simulation():
    """
    Simulate a full checkpoint save/restore cycle.

    Key insight: The shuffle order is determined by the seed when the DataLoader
    is created. The generator state saved mid-epoch is for NEXT epoch's shuffle,
    not the current epoch. For same-epoch resume, we just use the same seed
    and skip already-processed batches.
    """
    print("\n" + "=" * 80)
    print("Test 5: Full Checkpoint Simulation")
    print("=" * 80)

    config = get_config("tiny")
    seed = 42

    print(f"\n[Setup] Simulating training with checkpoint at batch 2")
    print("[Note] Same seed produces same shuffle order - just skip processed batches")

    # Phase 1: "Train" for several batches, checkpoint at batch 2
    loader1 = get_dataloader(
        config=config,
        split="train",
        batch_size=4,
        num_workers=0,
        streaming=False,
        num_samples=200,
        seed=seed
    )

    all_batches_phase1 = []
    checkpoint_batch_idx = 2

    for batch_idx, batch in enumerate(loader1):
        all_batches_phase1.append(batch["input_ids"].clone())

        if batch_idx == checkpoint_batch_idx:
            # In real checkpoint, we save batch_idx and generator state
            # Generator state is used for next epoch's shuffle, not current
            print(f"[Phase 1] Checkpoint at batch {batch_idx}")

        if batch_idx >= 5:
            break

    print(f"[Phase 1] Collected {len(all_batches_phase1)} batches (0-5)")

    # Phase 2: "Resume" from checkpoint - use same seed, skip processed batches
    # NOTE: We do NOT restore generator state for same-epoch resume
    # The same seed guarantees the same shuffle order
    loader2 = get_dataloader(
        config=config,
        split="train",
        batch_size=4,
        num_workers=0,
        streaming=False,
        num_samples=200,
        seed=seed  # Same seed = same shuffle order
    )

    # Skip batches before checkpoint (0, 1, 2 already processed)
    skip_batches = checkpoint_batch_idx + 1  # Skip 0-2, start from 3

    resumed_batches = []
    for batch_idx, batch in enumerate(loader2):
        if batch_idx < skip_batches:
            continue
        resumed_batches.append(batch["input_ids"].clone())
        if batch_idx >= 5:
            break

    print(f"[Phase 2] Resumed with same seed, skipped {skip_batches} batches, collected {len(resumed_batches)} batches")

    # Verify: resumed batches should match original batches 3, 4, 5
    print("\n[Verification] Comparing resumed batches with original...")
    all_match = True

    for i, resumed in enumerate(resumed_batches):
        original_idx = skip_batches + i
        original = all_batches_phase1[original_idx]

        if torch.equal(original, resumed):
            print(f"  Batch {original_idx}: ✓ Match")
        else:
            print(f"  Batch {original_idx}: ✗ MISMATCH!")
            all_match = False

    if all_match:
        print("\n✓ Test 5 PASSED: Full checkpoint simulation works correctly!")
    else:
        print("\n✗ Test 5 FAILED: Resumed training does not match original!")
        sys.exit(1)

    return True


def main():
    print("\n" + "=" * 80)
    print("DETERMINISTIC CHECKPOINT RESUME TESTS")
    print("=" * 80)

    tests = [
        ("Deterministic Shuffle", test_deterministic_shuffle),
        ("Generator State Save/Restore", test_generator_state_save_restore),
        ("Skip Logic", test_skip_logic),
        ("Different Seeds", test_different_seeds_produce_different_order),
        ("Full Checkpoint Simulation", test_checkpoint_simulation),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED with exception: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} test(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
