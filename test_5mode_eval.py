"""
Quick test for 5-mode evaluation system
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train.blockwise_sampling import generate_boundary_conditioning_split

print("=" * 80)
print("Testing 5-Mode Evaluation System")
print("=" * 80)

# Test 1: Boundary sampling
print("\nTest 1: Boundary Conditioning Split")
print("-" * 80)

for i in range(5):
    seq_len = 10
    cond_idx, eval_idx, unknown_idx = generate_boundary_conditioning_split(seq_len)

    print(f"\nSample {i+1}:")
    print(f"  Conditioning (start+end): {sorted(cond_idx)}")
    print(f"  Evaluation (middle):      {sorted(eval_idx)}")
    print(f"  Unknown:                  {sorted(unknown_idx)}")

    # Validate constraints
    assert len(set(cond_idx) & set(eval_idx)) == 0, "Conditioning and evaluation must be disjoint"
    assert len(cond_idx) + len(eval_idx) == seq_len, "Must cover full sequence"
    assert eval_idx == unknown_idx, "For Mode 2, unknown = evaluation"

    # Check boundary constraint
    if len(cond_idx) > 0:
        assert 0 in cond_idx, "Must include start"
        assert (seq_len - 1) in cond_idx, "Must include end"

    # Check evaluation is continuous middle part
    if len(eval_idx) > 0:
        eval_sorted = sorted(eval_idx)
        expected_middle = list(range(eval_sorted[0], eval_sorted[-1] + 1))
        assert eval_sorted == expected_middle, "Evaluation must be continuous middle part"

    print("  ✓ Validation passed")

print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)

# Test 2: Import check
print("\nTest 2: Import Check")
print("-" * 80)

try:
    from train.evaluation_modes import (
        evaluate_mode1_autoregressive,
        evaluate_mode2_boundary_filling,
        evaluate_mode3_training_dist,
        evaluate_mode4_cross_boundary,
        evaluate_mode5_cross_training,
        evaluate_all_modes
    )
    print("✓ All evaluation mode functions imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ 5-mode evaluation system is ready!")
print("=" * 80)
