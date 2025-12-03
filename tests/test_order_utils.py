"""
Unit tests for order utilities

Tests verify correct behavior of:
- indices_to_order: Convert indices to order tensor
- apply_order: Shuffle sequences and create targets
- create_labels_*: Create loss masks for fair/full modes
- prepare_sigmagpt_batch: End-to-end batch preparation

Run from project root: python tests/test_order_utils.py
Expected runtime: ~20 seconds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.order_utils import (
    indices_to_order,
    apply_order,
    create_labels_fair,
    create_labels_full,
    apply_labels_mask,
    prepare_sigmagpt_batch
)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_indices_to_order():
    """Test 1: Convert conditioning/evaluation indices to order tensor"""
    print_section("Test 1: indices_to_order")

    B, seq_len = 2, 10

    # Test case 1: Simple sequential order
    cond_indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
    eval_indices = torch.tensor([[3, 4], [3, 4]])

    order = indices_to_order(cond_indices, eval_indices, seq_len)

    # Expected: [0, 1, 2, 3, 4, 10] for both batches
    expected = torch.tensor([[0, 1, 2, 3, 4, 10], [0, 1, 2, 3, 4, 10]])

    assert torch.equal(order, expected), \
        f"Order mismatch:\nGot:      {order}\nExpected: {expected}"

    # Verify shape
    assert order.shape == (B, 6), f"Order shape should be (2, 6), got {order.shape}"

    # Test case 2: Non-contiguous indices
    cond_indices = torch.tensor([[0, 2, 4]])
    eval_indices = torch.tensor([[1, 3, 5]])

    order = indices_to_order(cond_indices, eval_indices, seq_len)
    expected = torch.tensor([[0, 2, 4, 1, 3, 5, 10]])

    assert torch.equal(order, expected), \
        f"Non-contiguous order failed:\nGot:      {order}\nExpected: {expected}"

    print("✓ indices_to_order works correctly")
    print(f"  - Sequential indices: ✓")
    print(f"  - Non-contiguous indices: ✓")
    print(f"  - Correct shape (B, num_cond + num_eval + 1): ✓")


def test_apply_order():
    """Test 2: Apply order to create inputs and targets"""
    print_section("Test 2: apply_order")

    # Test case 1: Simple sequential order
    tokens = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
    order = torch.tensor([[0, 1, 2, 3, 4, 10]])

    inputs, targets = apply_order(tokens, order)

    # Expected inputs: reordered tokens [10, 20, 30, 40, 50]
    expected_inputs = torch.tensor([[10, 20, 30, 40, 50]])

    # Expected targets: next tokens in order [20, 30, 40, 50, -1]
    # Last position: order[5] = 10 (seq_len), so target = -1
    expected_targets = torch.tensor([[20, 30, 40, 50, -1]])

    assert torch.equal(inputs, expected_inputs), \
        f"Inputs mismatch:\nGot:      {inputs}\nExpected: {expected_inputs}"

    assert torch.equal(targets, expected_targets), \
        f"Targets mismatch:\nGot:      {targets}\nExpected: {expected_targets}"

    # Test case 2: Non-sequential order
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    order = torch.tensor([[0, 2, 4, 1, 3, 10]])

    inputs, targets = apply_order(tokens, order)

    # Inputs: tokens at [0, 2, 4, 1, 3] = [0, 2, 4, 1, 3]
    expected_inputs = torch.tensor([[0, 2, 4, 1, 3]])

    # Targets: tokens at [2, 4, 1, 3, 10]
    # 10 is beyond sequence, so -1
    expected_targets = torch.tensor([[2, 4, 1, 3, -1]])

    assert torch.equal(inputs, expected_inputs), \
        f"Non-sequential inputs failed:\nGot:      {inputs}\nExpected: {expected_inputs}"

    assert torch.equal(targets, expected_targets), \
        f"Non-sequential targets failed:\nGot:      {targets}\nExpected: {expected_targets}"

    print("✓ apply_order works correctly")
    print(f"  - Sequential order: ✓")
    print(f"  - Non-sequential order: ✓")
    print(f"  - Handles seq_len marker (-1 target): ✓")


def test_create_labels_fair():
    """Test 3: Create labels for fair mode (only eval positions)"""
    print_section("Test 3: create_labels_fair")

    seq_len = 10
    order = torch.tensor([[0, 1, 2, 3, 4, 10]])

    # Fair mode: only evaluation positions compute loss
    # Conditioning: positions 0, 1, 2 (first 3)
    # Evaluation: positions 3, 4 (next 2)
    cond_size = 3

    mask = create_labels_fair(order, cond_size, seq_len)

    # Expected: [False, False, False, True, False]
    # First 3 positions (conditioning) are ignored
    # Position 3 (evaluation) computes loss
    # Position 4 predicts position 10 (seq_len) which doesn't exist, so ignored
    expected_mask = torch.tensor([[False, False, False, True, False]])

    assert torch.equal(mask, expected_mask), \
        f"Fair mode mask failed:\nGot:      {mask}\nExpected: {expected_mask}"

    # Test with position beyond seq_len
    order_with_invalid = torch.tensor([[0, 1, 2, 3, 15]])
    cond_size = 2

    mask = create_labels_fair(order_with_invalid, cond_size, seq_len)

    # Expected: [False, False, True, False]
    # Positions 0, 1 are conditioning (ignored)
    # Position 2 is evaluation (compute loss)
    # Position 3 predicts position 15 which is beyond seq_len (ignored)
    expected_mask = torch.tensor([[False, False, True, False]])

    assert torch.equal(mask, expected_mask), \
        f"Fair mode with invalid position failed:\nGot:      {mask}\nExpected: {expected_mask}"

    print("✓ create_labels_fair works correctly")
    print(f"  - Ignores conditioning positions: ✓")
    print(f"  - Computes loss on evaluation positions: ✓")
    print(f"  - Handles invalid positions: ✓")


def test_create_labels_full():
    """Test 4: Create labels for full mode (all positions)"""
    print_section("Test 4: create_labels_full")

    seq_len = 10
    order = torch.tensor([[0, 1, 2, 3, 4, 10]])

    # Full mode: all positions compute loss (except invalid)
    mask = create_labels_full(order, seq_len)

    # Expected: [True, True, True, True, False]
    # All positions compute loss except the last one (predicts seq_len)
    expected_mask = torch.tensor([[True, True, True, True, False]])

    assert torch.equal(mask, expected_mask), \
        f"Full mode mask failed:\nGot:      {mask}\nExpected: {expected_mask}"

    # Test with multiple invalid positions
    order_with_invalids = torch.tensor([[0, 1, 10, 15, 20]])

    mask = create_labels_full(order_with_invalids, seq_len)

    # Expected: [True, False, False, False]
    # Position 0 -> 1 (valid)
    # Position 1 -> 10 (invalid, >= seq_len)
    # Position 2 -> 15 (invalid)
    # Position 3 -> 20 (invalid)
    expected_mask = torch.tensor([[True, False, False, False]])

    assert torch.equal(mask, expected_mask), \
        f"Full mode with invalids failed:\nGot:      {mask}\nExpected: {expected_mask}"

    print("✓ create_labels_full works correctly")
    print(f"  - Computes loss on all valid positions: ✓")
    print(f"  - Handles invalid positions: ✓")


def test_apply_labels_mask():
    """Test 5: Apply mask to targets"""
    print_section("Test 5: apply_labels_mask")

    targets = torch.tensor([[10, 20, 30, 40, 50]])
    mask = torch.tensor([[True, False, True, False, True]])

    masked_targets = apply_labels_mask(targets, mask)

    # Expected: [10, -1, 30, -1, 50]
    expected = torch.tensor([[10, -1, 30, -1, 50]])

    assert torch.equal(masked_targets, expected), \
        f"Masked targets failed:\nGot:      {masked_targets}\nExpected: {expected}"

    print("✓ apply_labels_mask works correctly")
    print(f"  - Sets masked positions to -1: ✓")
    print(f"  - Preserves unmasked positions: ✓")


def test_prepare_sigmagpt_batch_fair():
    """Test 6: End-to-end batch preparation (fair mode)"""
    print_section("Test 6: prepare_sigmagpt_batch (fair mode)")

    tokens = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
    cond_indices = torch.tensor([[0, 1, 2]])
    eval_indices = torch.tensor([[3, 4]])

    inputs, order, targets = prepare_sigmagpt_batch(
        tokens, cond_indices, eval_indices, mode='fair'
    )

    # Expected inputs: [10, 20, 30, 40, 50]
    expected_inputs = torch.tensor([[10, 20, 30, 40, 50]])

    # Expected order: [0, 1, 2, 3, 4, 10]
    expected_order = torch.tensor([[0, 1, 2, 3, 4, 10]])

    # Expected targets (fair mode):
    # [-1, -1, -1, 40, 50]
    #  ^^^^^^^^^^^ conditioning (ignored)
    #              ^^^^^^^^ evaluation (compute loss)
    # But wait, targets should be next tokens: [20, 30, 40, 50, -1] (raw)
    # After masking: [-1, -1, -1, 50, -1]
    # Actually, let me recalculate:
    # Raw targets from apply_order: [20, 30, 40, 50, -1]
    # Fair mask: [False, False, False, True, True]
    # But position 4 predicts seq_len (10), so it should already be -1
    # Fair mask should be: [False, False, False, True, False]
    # After applying mask: [-1, -1, -1, 50, -1]
    expected_targets = torch.tensor([[-1, -1, -1, 50, -1]])

    assert torch.equal(inputs, expected_inputs), \
        f"Inputs failed:\nGot:      {inputs}\nExpected: {expected_inputs}"

    assert torch.equal(order, expected_order), \
        f"Order failed:\nGot:      {order}\nExpected: {expected_order}"

    assert torch.equal(targets, expected_targets), \
        f"Targets failed:\nGot:      {targets}\nExpected: {expected_targets}"

    print("✓ prepare_sigmagpt_batch (fair) works correctly")
    print(f"  - Inputs: {inputs.tolist()}")
    print(f"  - Order:  {order.tolist()}")
    print(f"  - Targets: {targets.tolist()}")
    print(f"  - Conditioning positions ignored: ✓")
    print(f"  - Evaluation positions have valid targets: ✓")


def test_prepare_sigmagpt_batch_full():
    """Test 7: End-to-end batch preparation (full mode)"""
    print_section("Test 7: prepare_sigmagpt_batch (full mode)")

    tokens = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
    cond_indices = torch.tensor([[0, 1, 2]])
    eval_indices = torch.tensor([[3, 4]])

    inputs, order, targets = prepare_sigmagpt_batch(
        tokens, cond_indices, eval_indices, mode='full'
    )

    # Expected inputs: [10, 20, 30, 40, 50]
    expected_inputs = torch.tensor([[10, 20, 30, 40, 50]])

    # Expected order: [0, 1, 2, 3, 4, 10]
    expected_order = torch.tensor([[0, 1, 2, 3, 4, 10]])

    # Expected targets (full mode):
    # Raw targets: [20, 30, 40, 50, -1]
    # Full mask: [True, True, True, True, False]
    # After masking: [20, 30, 40, 50, -1]
    expected_targets = torch.tensor([[20, 30, 40, 50, -1]])

    assert torch.equal(inputs, expected_inputs), \
        f"Inputs failed:\nGot:      {inputs}\nExpected: {expected_inputs}"

    assert torch.equal(order, expected_order), \
        f"Order failed:\nGot:      {order}\nExpected: {expected_order}"

    assert torch.equal(targets, expected_targets), \
        f"Targets failed:\nGot:      {targets}\nExpected: {expected_targets}"

    print("✓ prepare_sigmagpt_batch (full) works correctly")
    print(f"  - Inputs: {inputs.tolist()}")
    print(f"  - Order:  {order.tolist()}")
    print(f"  - Targets: {targets.tolist()}")
    print(f"  - All valid positions compute loss: ✓")
    print(f"  - Invalid position marked as -1: ✓")


def test_batch_dimensions():
    """Test 8: Verify batch handling with multiple samples"""
    print_section("Test 8: Batch Dimensions")

    B = 4
    seq_len = 16

    tokens = torch.randint(0, 100, (B, seq_len))
    cond_indices = torch.tensor([[0, 1, 2, 3]] * B)
    eval_indices = torch.tensor([[4, 5, 6]] * B)

    inputs, order, targets = prepare_sigmagpt_batch(
        tokens, cond_indices, eval_indices, mode='fair'
    )

    # Verify shapes
    expected_T = 7  # 4 cond + 3 eval
    assert inputs.shape == (B, expected_T), \
        f"Inputs shape should be ({B}, {expected_T}), got {inputs.shape}"

    assert order.shape == (B, expected_T + 1), \
        f"Order shape should be ({B}, {expected_T + 1}), got {order.shape}"

    assert targets.shape == (B, expected_T), \
        f"Targets shape should be ({B}, {expected_T}), got {targets.shape}"

    # Verify all batches have seq_len at the end of order
    assert (order[:, -1] == seq_len).all(), \
        "All batches should have seq_len as last element of order"

    print("✓ Batch dimensions handled correctly")
    print(f"  - Batch size: {B}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Inputs shape: {inputs.shape}")
    print(f"  - Order shape: {order.shape}")
    print(f"  - Targets shape: {targets.shape}")


def test_non_contiguous_indices():
    """Test 9: Handle non-contiguous and random indices"""
    print_section("Test 9: Non-Contiguous Indices")

    tokens = torch.arange(0, 20).unsqueeze(0)  # [0, 1, 2, ..., 19]
    cond_indices = torch.tensor([[0, 5, 10, 15]])  # Non-contiguous
    eval_indices = torch.tensor([[2, 7, 12]])      # Non-contiguous

    inputs, order, targets = prepare_sigmagpt_batch(
        tokens, cond_indices, eval_indices, mode='fair'
    )

    # Expected order: [0, 5, 10, 15, 2, 7, 12, 20]
    expected_order = torch.tensor([[0, 5, 10, 15, 2, 7, 12, 20]])

    # Expected inputs: tokens at [0, 5, 10, 15, 2, 7, 12]
    expected_inputs = torch.tensor([[0, 5, 10, 15, 2, 7, 12]])

    # Expected targets (fair mode):
    # Raw: [5, 10, 15, 2, 7, 12, -1]
    # Fair mask: [F, F, F, F, T, T, T]
    # But position 6 predicts 20 (beyond seq_len=20), so it's -1
    # Fair mask should be: [F, F, F, F, T, T, F]
    # After mask: [-1, -1, -1, -1, 7, 12, -1]
    expected_targets = torch.tensor([[-1, -1, -1, -1, 7, 12, -1]])

    assert torch.equal(order, expected_order), \
        f"Order failed:\nGot:      {order}\nExpected: {expected_order}"

    assert torch.equal(inputs, expected_inputs), \
        f"Inputs failed:\nGot:      {inputs}\nExpected: {expected_inputs}"

    assert torch.equal(targets, expected_targets), \
        f"Targets failed:\nGot:      {targets}\nExpected: {expected_targets}"

    print("✓ Non-contiguous indices handled correctly")
    print(f"  - Order: {order.tolist()}")
    print(f"  - Inputs: {inputs.tolist()}")
    print(f"  - Targets: {targets.tolist()}")


def run_all_tests():
    """Run all order utility tests"""
    print("\n" + "=" * 80)
    print("  ORDER UTILITIES TESTS")
    print("=" * 80)

    tests = [
        ("indices_to_order", test_indices_to_order),
        ("apply_order", test_apply_order),
        ("create_labels_fair", test_create_labels_fair),
        ("create_labels_full", test_create_labels_full),
        ("apply_labels_mask", test_apply_labels_mask),
        ("prepare_sigmagpt_batch (fair)", test_prepare_sigmagpt_batch_fair),
        ("prepare_sigmagpt_batch (full)", test_prepare_sigmagpt_batch_full),
        ("Batch Dimensions", test_batch_dimensions),
        ("Non-Contiguous Indices", test_non_contiguous_indices),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {test_name}")
            print(f"   Error: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}")
            print(f"   {type(e).__name__}: {str(e)}")
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)
    print(f"  Total tests: {len(tests)}")
    print(f"  Passed: {passed} ✓")
    print(f"  Failed: {failed} {'❌' if failed > 0 else ''}")
    print("=" * 80)

    if failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
