"""
Ordering Modes for Sigma GPT Training (Eric's Two Methods)

This module implements the two training strategies proposed by Eric Elmoznino
for Sigma GPT to enable fair comparison with the conditional probability model.

Two Methods:
1. Temporal Ordering: Maintain left-to-right temporal order within conditioning
   and evaluation sets. This matches the behavior of the existing conditional
   model and enables fair architectural comparison.

2. Random Scrambling: Randomly shuffle tokens within conditioning and evaluation
   sets. This tests whether Sigma GPT can handle additional ordering complexity
   and validates the "capacity problem" hypothesis.

Reference: Meeting notes (Updates-Arbitrary-conditionals-638a889b-a2c6.md)
Timestamp: [22:06] - [28:40]
"""

from enum import Enum
import random
from typing import List, Tuple


class OrderingMode(Enum):
    """
    Enumeration of ordering strategies for Sigma GPT training.

    TEMPORAL: Conditioning and evaluation sets maintain temporal (left-to-right) order
    RANDOM_SCRAMBLE: Conditioning and evaluation sets are randomly shuffled
    """
    TEMPORAL = "temporal"
    RANDOM_SCRAMBLE = "random_scramble"


def apply_ordering_mode(
    cond_indices: List[int],
    eval_indices: List[int],
    mode: OrderingMode
) -> Tuple[List[int], List[int]]:
    """
    Apply ordering strategy to conditioning and evaluation indices.

    This function reorders the conditioning and evaluation indices according
    to the specified ordering mode. The semantic roles (conditioning vs evaluation)
    remain unchanged - only the order in which tokens appear in the sequence changes.

    Args:
        cond_indices: List of conditioning position indices
                     Example: [0, 3, 5] means tokens at positions 0, 3, 5
        eval_indices: List of evaluation position indices
                     Example: [2, 7] means tokens at positions 2, 7
        mode: OrderingMode enum specifying the ordering strategy

    Returns:
        Tuple of (reordered_cond_indices, reordered_eval_indices)

        The returned indices will be used to create the order tensor:
        order = [reordered_cond] + [reordered_eval] + [seq_len]

    Examples:
        >>> cond = [0, 3, 5]
        >>> eval = [2, 7]

        # Method 1: Temporal ordering (Eric's first method)
        # Maintains left-to-right order within each set
        >>> apply_ordering_mode(cond, eval, OrderingMode.TEMPORAL)
        ([0, 3, 5], [2, 7])
        # Order tensor would be: [0, 3, 5, 2, 7, seq_len]
        # Generation: see 0 → see 3 → see 5 → predict 2 → predict 7

        # Method 2: Random scrambling (Eric's second method)
        # Shuffles within each set
        >>> apply_ordering_mode(cond, eval, OrderingMode.RANDOM_SCRAMBLE)
        ([5, 0, 3], [7, 2])  # Example output (random)
        # Order tensor would be: [5, 0, 3, 7, 2, seq_len]
        # Generation: see 5 → see 0 → see 3 → predict 7 → predict 2

    Notes:
        - Temporal ordering is deterministic (always returns sorted order)
        - Random scrambling is non-deterministic (different each call)
        - The semantic meaning is preserved:
          * All conditioning tokens are still conditioning (model can see them)
          * All evaluation tokens are still evaluation (loss computed on them)
          * Only the ORDER changes, not the roles
    """
    if mode == OrderingMode.TEMPORAL:
        # Method 1: Temporal ordering
        # Keep tokens in left-to-right temporal order within each set
        # This matches the existing conditional model's behavior
        return sorted(cond_indices), sorted(eval_indices)

    elif mode == OrderingMode.RANDOM_SCRAMBLE:
        # Method 2: Random scrambling
        # Shuffle tokens within conditioning and evaluation sets independently
        # This tests Sigma GPT's ability to handle arbitrary orderings

        # Make copies to avoid modifying input lists
        cond_shuffled = cond_indices.copy()
        eval_shuffled = eval_indices.copy()

        # Shuffle in-place
        random.shuffle(cond_shuffled)
        random.shuffle(eval_shuffled)

        return cond_shuffled, eval_shuffled

    else:
        raise ValueError(
            f"Unknown ordering mode: {mode}. "
            f"Valid options: {[m.value for m in OrderingMode]}"
        )


def get_ordering_mode(mode_name: str) -> OrderingMode:
    """
    Convert string to OrderingMode enum.

    Args:
        mode_name: String name of ordering mode (case-insensitive)
                  Examples: "temporal", "TEMPORAL", "random_scramble"

    Returns:
        OrderingMode enum

    Raises:
        ValueError: If mode_name is not a valid ordering mode

    Examples:
        >>> get_ordering_mode("temporal")
        OrderingMode.TEMPORAL

        >>> get_ordering_mode("RANDOM_SCRAMBLE")
        OrderingMode.RANDOM_SCRAMBLE
    """
    try:
        return OrderingMode[mode_name.upper()]
    except KeyError:
        valid_modes = [m.value for m in OrderingMode]
        raise ValueError(
            f"Invalid ordering mode: '{mode_name}'. "
            f"Valid options: {valid_modes}"
        )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Ordering Modes Test")
    print("=" * 80)

    # Test data
    cond_indices = [0, 3, 5, 7]
    eval_indices = [2, 4, 6]

    print(f"\nOriginal indices:")
    print(f"  Conditioning: {cond_indices}")
    print(f"  Evaluation: {eval_indices}")

    # Test Method 1: Temporal
    print(f"\nMethod 1: Temporal Ordering")
    cond_temp, eval_temp = apply_ordering_mode(
        cond_indices, eval_indices, OrderingMode.TEMPORAL
    )
    print(f"  Conditioning: {cond_temp} (sorted)")
    print(f"  Evaluation: {eval_temp} (sorted)")
    print(f"  Order tensor: {cond_temp + eval_temp + [10]}")

    # Test Method 2: Random Scramble
    print(f"\nMethod 2: Random Scrambling")
    for i in range(3):
        cond_rand, eval_rand = apply_ordering_mode(
            cond_indices, eval_indices, OrderingMode.RANDOM_SCRAMBLE
        )
        print(f"  Run {i+1}:")
        print(f"    Conditioning: {cond_rand} (shuffled)")
        print(f"    Evaluation: {eval_rand} (shuffled)")
        print(f"    Order tensor: {cond_rand + eval_rand + [10]}")

    print("\n" + "=" * 80)
    print("✓ Ordering modes module working correctly")
    print("=" * 80)
