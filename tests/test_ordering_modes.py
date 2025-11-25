"""
Test Ordering Modes Implementation

Verifies that Eric's two training methods work correctly.

Run from project root: python tests/test_ordering_modes.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.ordering_modes import OrderingMode, apply_ordering_mode, get_ordering_mode


def test_temporal_ordering():
    """Test that temporal ordering maintains sorted order"""
    print("\n[Test 1] Temporal Ordering")
    print("-" * 80)

    cond = [2, 0, 1, 5]
    eval = [4, 3, 6]

    cond_out, eval_out = apply_ordering_mode(cond, eval, OrderingMode.TEMPORAL)

    assert cond_out == [0, 1, 2, 5], f"Expected sorted [0,1,2,5], got {cond_out}"
    assert eval_out == [3, 4, 6], f"Expected sorted [3,4,6], got {eval_out}"

    print(f"  Input cond: {cond}")
    print(f"  Output cond: {cond_out} (sorted)")
    print(f"  Input eval: {eval}")
    print(f"  Output eval: {eval_out} (sorted)")
    print("✓ Temporal ordering test passed")


def test_random_scrambling():
    """Test that random scrambling produces different orders"""
    print("\n[Test 2] Random Scrambling")
    print("-" * 80)

    cond = [0, 1, 2, 3, 4]
    eval = [5, 6, 7]

    # Test multiple times
    results = []
    for i in range(20):
        cond_out, eval_out = apply_ordering_mode(cond, eval, OrderingMode.RANDOM_SCRAMBLE)
        results.append((tuple(cond_out), tuple(eval_out)))

        if i < 3:
            print(f"  Run {i+1}: cond={cond_out}, eval={eval_out}")

    # Should have multiple different orderings
    unique_results = len(set(results))
    print(f"\n  Unique orderings: {unique_results}/20")
    assert unique_results > 1, f"Random scrambling not random: only {unique_results} unique results"
    print("✓ Random scrambling test passed")


def test_get_ordering_mode():
    """Test string to enum conversion"""
    print("\n[Test 3] String to Enum Conversion")
    print("-" * 80)

    mode1 = get_ordering_mode("temporal")
    assert mode1 == OrderingMode.TEMPORAL
    print("  'temporal' → OrderingMode.TEMPORAL ✓")

    mode2 = get_ordering_mode("RANDOM_SCRAMBLE")
    assert mode2 == OrderingMode.RANDOM_SCRAMBLE
    print("  'RANDOM_SCRAMBLE' → OrderingMode.RANDOM_SCRAMBLE ✓")

    try:
        get_ordering_mode("invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  'invalid' → ValueError ✓")

    print("✓ String conversion test passed")


def test_integration():
    """Test integration with typical usage"""
    print("\n[Test 4] Integration Test")
    print("-" * 80)

    # Simulate typical conditioning/evaluation split
    cond_indices = [0, 1, 2, 10, 15, 20]
    eval_indices = [5, 8, 12, 18]

    print(f"  Original conditioning: {cond_indices}")
    print(f"  Original evaluation: {eval_indices}")

    # Method 1: Temporal
    cond_temp, eval_temp = apply_ordering_mode(cond_indices, eval_indices, OrderingMode.TEMPORAL)
    order_temp = cond_temp + eval_temp + [25]
    print(f"\n  Method 1 (Temporal):")
    print(f"    Conditioning: {cond_temp}")
    print(f"    Evaluation: {eval_temp}")
    print(f"    Order tensor: {order_temp}")

    # Method 2: Random
    cond_rand, eval_rand = apply_ordering_mode(cond_indices, eval_indices, OrderingMode.RANDOM_SCRAMBLE)
    order_rand = cond_rand + eval_rand + [25]
    print(f"\n  Method 2 (Random Scramble):")
    print(f"    Conditioning: {cond_rand}")
    print(f"    Evaluation: {eval_rand}")
    print(f"    Order tensor: {order_rand}")

    # Verify semantic integrity
    assert set(cond_temp) == set(cond_indices), "Temporal shouldn't change token set"
    assert set(eval_temp) == set(eval_indices), "Temporal shouldn't change token set"
    assert set(cond_rand) == set(cond_indices), "Random shouldn't change token set"
    assert set(eval_rand) == set(eval_indices), "Random shouldn't change token set"

    print("\n  ✓ Semantic integrity preserved (same tokens, different order)")
    print("✓ Integration test passed")


if __name__ == "__main__":
    print("=" * 80)
    print("Ordering Modes Test Suite")
    print("=" * 80)

    try:
        test_temporal_ordering()
        test_random_scrambling()
        test_get_ordering_mode()
        test_integration()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nOrdering modes implementation is correct.")
        print("Eric's two methods are ready for experimentation.")

    except AssertionError as e:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
