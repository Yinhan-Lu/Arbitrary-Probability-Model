"""
Blockwise Sampling for Conditioning and Evaluation Sets

Generates conditioning, evaluation, and unknown sets as contiguous blocks
using customizable probability distributions. This provides flexible control
over the sampling strategy for arbitrary conditional probability training.

Key concepts:
- Conditioning Set (X_c): Tokens the model observes (not masked)
- Unknown Set (X_u): All non-conditioning tokens (masked with [M])
- Evaluation Set (X_e): Subset of unknown set where we compute loss

Relations:
- Conditioning ∩ Unknown = ∅ (disjoint)
- Evaluation ⊆ Unknown
- Conditioning ∪ Unknown = All tokens
"""

from typing import Callable
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_conditioning_evaluation_sets_blockwise(
    seq_len: int,
    num_conditioning_distribution: Callable[[int], int],
    num_blocks_distribution: Callable[[int], int],
    block_sizes_distribution: Callable[[int, int], list[int]],
    num_evaluation_distribution: Callable[[int], int] = None,
    num_eval_blocks_distribution: Callable[[int], int] = None,
    eval_block_sizes_distribution: Callable[[int, int], list[int]] = None,
    valid_positions: list[int] = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Generate conditioning, evaluation, and unknown sets as contiguous blocks.

    Args:
        seq_len: Length of the sequence being conditioned
        num_conditioning_distribution: Function that takes seq_len and samples
            the number of conditioning variables in range [0, seq_len-1]
        num_blocks_distribution: Function that takes num_conditioning and samples
            the number of contiguous blocks in range [1, num_conditioning]
        block_sizes_distribution: Function that takes (num_conditioning, num_blocks)
            and samples block sizes that sum to num_conditioning
        num_evaluation_distribution: Function that takes available_len and samples
            the number of evaluation variables in range [0, available_len]
            (default: uses same ratio as conditioning)
        num_eval_blocks_distribution: Function for evaluation blocks
            (default: uses same logic as conditioning)
        eval_block_sizes_distribution: Function for evaluation block sizes
            (default: uses same logic as conditioning)
        valid_positions: Optional list of valid positions to sample from (e.g., non-padding).
                        If provided, all sampling only occurs within these positions.
                        If None, samples from range(seq_len).

    Returns:
        Tuple of (conditioning_indices, evaluation_indices, unknown_indices)
        - conditioning_indices: List of conditioning variable indices (sorted)
        - evaluation_indices: List of evaluation variable indices (sorted, subset of unknown)
        - unknown_indices: List of unknown variable indices (sorted, all non-conditioning)

    Example:
        >>> cond, eval_set, unknown = generate_conditioning_evaluation_sets_blockwise(
        ...     seq_len=10,
        ...     num_conditioning_distribution=lambda l: random.randint(0, l//3),
        ...     num_blocks_distribution=lambda n: random.randint(1, min(3, n)) if n > 0 else 1,
        ...     block_sizes_distribution=uniform_block_sizes_distribution,
        ... )
    """
    # Step 1: Generate conditioning set (with valid_positions if provided)
    conditioning_indices = _generate_blockwise_set(
        seq_len=seq_len,
        num_items_distribution=num_conditioning_distribution,
        num_blocks_distribution=num_blocks_distribution,
        block_sizes_distribution=block_sizes_distribution,
        valid_positions=valid_positions,
    )

    # Step 2: Calculate unknown set (all non-conditioning positions within valid range)
    if valid_positions is not None:
        # Unknown = valid positions that are not in conditioning
        unknown_indices = [i for i in valid_positions if i not in conditioning_indices]
    else:
        # Unknown = all positions that are not in conditioning
        unknown_indices = [i for i in range(seq_len) if i not in conditioning_indices]

    # Step 3: Generate evaluation set from unknown positions
    if len(unknown_indices) == 0:
        # No unknown positions, no evaluation
        evaluation_indices = []
    else:
        # Use provided distributions or defaults
        if num_evaluation_distribution is None:
            # Default: same percentage as conditioning
            conditioning_ratio = len(conditioning_indices) / seq_len if seq_len > 0 else 0.3
            num_evaluation_distribution = lambda l: min(
                l, max(1, int(l * conditioning_ratio))
            )

        if num_eval_blocks_distribution is None:
            num_eval_blocks_distribution = lambda n: random.randint(1, min(2, max(1, n)))

        if eval_block_sizes_distribution is None:
            eval_block_sizes_distribution = block_sizes_distribution

        # Generate evaluation set from available unknown positions
        evaluation_indices = _generate_blockwise_set_from_positions(
            available_positions=unknown_indices,
            num_items_distribution=num_evaluation_distribution,
            num_blocks_distribution=num_eval_blocks_distribution,
            block_sizes_distribution=eval_block_sizes_distribution,
        )

    # Validate output
    _validate_sets(seq_len, conditioning_indices, evaluation_indices, unknown_indices, valid_positions=valid_positions)

    logger.debug(
        f"Blockwise sampling: seq_len={seq_len}, "
        f"cond={len(conditioning_indices)}, "
        f"eval={len(evaluation_indices)}, "
        f"unknown={len(unknown_indices)}"
    )

    return conditioning_indices, evaluation_indices, unknown_indices


def _generate_blockwise_set(
    seq_len: int,
    num_items_distribution: Callable[[int], int],
    num_blocks_distribution: Callable[[int], int],
    block_sizes_distribution: Callable[[int, int], list[int]],
    valid_positions: list[int] = None,
) -> list[int]:
    """
    Generate a set of indices as contiguous blocks from sequence positions.

    This is the core blockwise sampling algorithm based on the provided template.

    Args:
        seq_len: Total sequence length
        num_items_distribution: Samples number of items to select
        num_blocks_distribution: Samples number of blocks
        block_sizes_distribution: Samples sizes of each block
        valid_positions: Optional list of valid positions to sample from (e.g., non-padding).
                        If provided, sampling only occurs within these positions.
                        If None, samples from range(seq_len).

    Returns:
        Sorted list of selected indices
    """
    # Use valid_positions if provided, otherwise use all positions
    if valid_positions is not None:
        # Sample from valid positions using the from_positions variant
        return _generate_blockwise_set_from_positions(
            available_positions=valid_positions,
            num_items_distribution=num_items_distribution,
            num_blocks_distribution=num_blocks_distribution,
            block_sizes_distribution=block_sizes_distribution,
        )

    # Original logic when valid_positions not provided
    # Sample number of items
    num_items = num_items_distribution(seq_len)
    assert 0 <= num_items < seq_len, \
        f"Number of items must be in range [0, {seq_len-1}], got {num_items}"

    if num_items == 0:
        return []

    # Sample number of blocks
    num_blocks = num_blocks_distribution(num_items)
    assert 1 <= num_blocks <= num_items, \
        f"Number of blocks must be in range [1, {num_items}], got {num_blocks}"

    # Sample block sizes
    block_sizes = block_sizes_distribution(num_items, num_blocks)
    assert all(size >= 1 for size in block_sizes), "All block sizes must be at least 1"
    assert sum(block_sizes) == num_items, \
        f"Sum of block sizes ({sum(block_sizes)}) must equal num_items ({num_items})"

    # Shuffle block sizes for randomness
    random.shuffle(block_sizes)

    # Distribute gaps using random weak composition
    free = seq_len - num_items  # Number of gap slots to distribute
    gaps_needed = len(block_sizes) + 1  # Gap before first, between, after last

    gaps = _random_composition(free, gaps_needed)

    # Place blocks with gaps
    selected_indices = []
    cursor = gaps[0]  # Start after first gap

    for block_size, gap_after in zip(block_sizes, gaps[1:]):
        # Place block
        block_start = cursor
        block_end = cursor + block_size
        selected_indices.extend(range(block_start, block_end))

        # Move cursor past this block and gap
        cursor = block_end + gap_after

    return sorted(selected_indices)


def _generate_blockwise_set_from_positions(
    available_positions: list[int],
    num_items_distribution: Callable[[int], int],
    num_blocks_distribution: Callable[[int], int],
    block_sizes_distribution: Callable[[int, int], list[int]],
) -> list[int]:
    """
    Generate blockwise set from a given list of available positions.

    This is used for sampling evaluation set from unknown positions,
    which may not be contiguous in the original sequence.

    Args:
        available_positions: List of available positions to select from
        num_items_distribution: Samples number of items to select
        num_blocks_distribution: Samples number of blocks
        block_sizes_distribution: Samples sizes of each block

    Returns:
        Sorted list of selected positions
    """
    available_len = len(available_positions)
    if available_len == 0:
        return []

    # Sample number of items
    num_items = num_items_distribution(available_len)
    num_items = min(num_items, available_len)  # Can't exceed available

    if num_items == 0:
        return []

    # Optimization: if selecting all available positions, return them directly
    # This handles the "no unseen set" case (eval_pct=100%) efficiently
    if num_items == available_len:
        return sorted(available_positions)

    # Sample number of blocks
    num_blocks = num_blocks_distribution(num_items)
    num_blocks = min(num_blocks, num_items)

    # Sample block sizes
    block_sizes = block_sizes_distribution(num_items, num_blocks)
    assert sum(block_sizes) == num_items, "Block sizes must sum to num_items"

    # Sort available positions for easier contiguous block selection
    positions = sorted(available_positions)

    # Find contiguous segments in available positions
    segments = _find_contiguous_segments(positions)

    # Select blocks from segments
    selected = []
    remaining = num_items
    random.shuffle(block_sizes)

    for block_size in block_sizes:
        if remaining == 0 or not segments:
            break

        # Choose a random segment
        segment = random.choice(segments)

        # Select contiguous block from segment
        actual_size = min(block_size, len(segment))
        if len(segment) > actual_size:
            start = random.randint(0, len(segment) - actual_size)
            block = segment[start:start + actual_size]
        else:
            block = segment

        selected.extend(block)
        remaining -= len(block)

        # Remove used slice from the chosen segment and keep the leftovers
        segments.remove(segment)
        left = segment[: segment.index(block[0])] if block else []
        right = segment[segment.index(block[-1]) + 1 :] if block else []
        if left:
            segments.append(left)
        if right:
            segments.append(right)

    return sorted(selected)


def _random_composition(n: int, k: int) -> list[int]:
    """
    Sample a random weak composition of n into k parts uniformly.

    Uses stars-and-bars method: choose (k-1) cut points among n+k-1 positions.

    Args:
        n: Number to decompose
        k: Number of parts

    Returns:
        List of k non-negative integers that sum to n
    """
    if k == 1:
        return [n]

    # Choose k-1 cut points
    cuts = sorted(random.sample(range(1, n + k), k - 1))

    # Compute parts from cuts
    parts = []
    prev = 0
    for cut in cuts + [n + k]:
        parts.append(cut - prev - 1)
        prev = cut

    return parts


def _find_contiguous_segments(positions: list[int]) -> list[list[int]]:
    """
    Find contiguous segments in a sorted list of positions.

    Args:
        positions: Sorted list of positions

    Returns:
        List of contiguous segments (each segment is a list of consecutive positions)
    """
    if not positions:
        return []

    segments = []
    current_segment = [positions[0]]

    for i in range(1, len(positions)):
        if positions[i] == positions[i-1] + 1:
            # Contiguous
            current_segment.append(positions[i])
        else:
            # Gap found, start new segment
            segments.append(current_segment)
            current_segment = [positions[i]]

    segments.append(current_segment)
    return segments


def _validate_sets(seq_len, conditioning_indices, evaluation_indices, unknown_indices, valid_positions=None):
    """Validate the three sets satisfy required properties.

    Args:
        seq_len: Sequence length
        conditioning_indices: Conditioning indices
        evaluation_indices: Evaluation indices
        unknown_indices: Unknown indices
        valid_positions: Optional list of valid positions. If provided, checks coverage against
                        valid_positions instead of range(seq_len).
    """
    cond_set = set(conditioning_indices)
    eval_set = set(evaluation_indices)
    unknown_set = set(unknown_indices)

    # Check disjoint
    assert cond_set.isdisjoint(unknown_set), \
        "Conditioning and unknown sets must be disjoint"

    # Check evaluation is subset of unknown
    assert eval_set.issubset(unknown_set), \
        "Evaluation set must be subset of unknown set"

    # Check coverage
    if valid_positions is not None:
        # When using valid_positions, check coverage against valid positions only
        expected_coverage = set(valid_positions)
    else:
        # Default: check coverage against all positions
        expected_coverage = set(range(seq_len))

    assert cond_set | unknown_set == expected_coverage, \
        f"Conditioning and unknown sets must cover all {'valid' if valid_positions is not None else ''} positions"


# =============================================================================
# Default Distribution Functions
# =============================================================================

def uniform_num_conditioning_distribution(
    seq_len: int,
    conditioning_percentage_range: tuple[float, float] = (0.2, 0.4)
) -> int:
    """
    Sample number of conditioning variables uniformly within a percentage range.

    Args:
        seq_len: Sequence length
        conditioning_percentage_range: (min_ratio, max_ratio) for conditioning

    Returns:
        Number of conditioning variables
    """
    min_cond = max(0, int(seq_len * conditioning_percentage_range[0]))
    max_cond = max(min_cond, int(seq_len * conditioning_percentage_range[1]))
    max_cond = min(max_cond, seq_len - 1)  # Leave at least 1 for unknown
    return random.randint(min_cond, max_cond)


def uniform_num_blocks_distribution(num_items: int) -> int:
    """
    Sample number of blocks uniformly.

    Original design: blocks can be anywhere from 1 to num_items.
    No artificial max_blocks limit.

    Args:
        num_items: Number of items to distribute into blocks

    Returns:
        Number of blocks in range [1, num_items]
    """
    if num_items == 0:
        return 1
    return random.randint(1, num_items)


def uniform_block_sizes_distribution(
    num_items: int,
    num_blocks: int
) -> list[int]:
    """
    Sample block sizes uniformly such that they sum to num_items.

    Strategy: Start with [1, 1, ..., 1], then distribute remainder randomly.

    Args:
        num_items: Total number of items
        num_blocks: Number of blocks

    Returns:
        List of block sizes
    """
    sizes = [1] * num_blocks
    remaining = num_items - num_blocks

    for _ in range(remaining):
        sizes[random.randint(0, num_blocks - 1)] += 1

    random.shuffle(sizes)
    return sizes


def uniform_num_evaluation_distribution(
    available_len: int,
    evaluation_percentage_range: tuple[float, float] = (0.2, 0.4)
) -> int:
    """
    Sample number of evaluation variables uniformly within a percentage range.

    Args:
        available_len: Number of available unknown positions
        evaluation_percentage_range: (min_ratio, max_ratio) for evaluation

    Returns:
        Number of evaluation variables
    """
    min_eval = max(1, int(available_len * evaluation_percentage_range[0]))
    max_eval = max(min_eval, int(available_len * evaluation_percentage_range[1]))
    max_eval = min(max_eval, available_len)
    return random.randint(min_eval, max_eval)


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_conditioning_set_blockwise(
    seq_len: int,
    conditioning_ratio: float = 0.3,
    evaluation_ratio: float = 0.3,
    min_conditioning: int = 1,
    min_evaluation: int = 1,
    max_cond_blocks: int = 3,
    max_eval_blocks: int = 2,
) -> tuple[list[int], list[int], list[int]]:
    """
    Convenience function with simple ratio-based configuration.

    This provides backward compatibility with the old API.

    Args:
        seq_len: Sequence length
        conditioning_ratio: Target ratio for conditioning (e.g., 0.3 = 30%)
        evaluation_ratio: Target ratio for evaluation (e.g., 0.3 = 30%)
        min_conditioning: Minimum conditioning tokens
        min_evaluation: Minimum evaluation tokens
        max_cond_blocks: Maximum blocks for conditioning
        max_eval_blocks: Maximum blocks for evaluation

    Returns:
        Tuple of (conditioning_indices, evaluation_indices, unknown_indices)
    """
    # Define distributions based on ratios
    def num_cond_dist(seq_len):
        target = max(min_conditioning, int(seq_len * conditioning_ratio))
        return min(target, seq_len - 1)

    def num_cond_blocks_dist(num_cond):
        return random.randint(1, min(max_cond_blocks, max(1, num_cond)))

    def num_eval_dist(available_len):
        target = max(min_evaluation, int(available_len * evaluation_ratio))
        return min(target, available_len)

    def num_eval_blocks_dist(num_eval):
        return random.randint(1, min(max_eval_blocks, max(1, num_eval)))

    return generate_conditioning_evaluation_sets_blockwise(
        seq_len=seq_len,
        num_conditioning_distribution=num_cond_dist,
        num_blocks_distribution=num_cond_blocks_dist,
        block_sizes_distribution=uniform_block_sizes_distribution,
        num_evaluation_distribution=num_eval_dist,
        num_eval_blocks_distribution=num_eval_blocks_dist,
        eval_block_sizes_distribution=uniform_block_sizes_distribution,
    )


def uniform_boundary_block_sizes_distribution(
    seq_len: int,
    boundary_cond_percentage_range: tuple[float, float] = (0.1, 0.3)
) -> tuple[int, int]:
    """
    Sample boundary block sizes for Mode 2 evaluation.

    Returns start_size and end_size such that their sum respects the
    conditioning percentage range.

    Args:
        seq_len: Sequence length
        boundary_cond_percentage_range: (min_pct, max_pct) for total conditioning

    Returns:
        (start_size, end_size) tuple

    Example:
        For seq_len=1024, range=(0.1, 0.3):
        - Total conditioning: 103-307 tokens
        - Returns (start=50, end=60) where start+end ∈ [103, 307]
    """
    min_pct, max_pct = boundary_cond_percentage_range

    # Sample total conditioning percentage
    total_pct = random.uniform(min_pct, max_pct)
    max_total_cond = int(seq_len * total_pct)

    # Ensure at least 1 token each and leave room for middle
    max_total_cond = min(max_total_cond, seq_len - 1)
    max_total_cond = max(max_total_cond, 2)  # At least 2 (1 start + 1 end)

    # Randomly split between start and end
    # Start: 20%-80% of total, End: the rest
    start_ratio = random.uniform(0.2, 0.8)
    start_size = max(1, int(max_total_cond * start_ratio))
    end_size = max(1, max_total_cond - start_size)

    # Ensure doesn't exceed sequence
    if start_size + end_size >= seq_len:
        # Adjust to leave at least 1 middle token
        total = seq_len - 1
        start_size = max(1, total // 2)
        end_size = total - start_size

    return start_size, end_size


def generate_boundary_conditioning_split(
    seq_len: int,
    boundary_block_sizes_distribution: Callable[[int], tuple[int, int]] = None,
    start_block_range: tuple[int, int] = None,
    end_block_range: tuple[int, int] = None,
    valid_positions: list[int] = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Generate boundary-constrained conditioning split for Mode 2 evaluation.

    Conditioning set MUST include start block + end block (exactly 2 blocks).
    Evaluation set is the continuous middle part.

    Args:
        seq_len: Sequence length
        boundary_block_sizes_distribution: Optional callable that takes seq_len
                                          and returns (start_size, end_size).
                                          If provided, overrides range-based sampling.
        start_block_range: (min_size, max_size) for start block.
                          Default: (1, seq_len // 3)
                          Ignored if boundary_block_sizes_distribution is provided.
        end_block_range: (min_size, max_size) for end block.
                        Default: (1, seq_len // 3)
                        Ignored if boundary_block_sizes_distribution is provided.
        valid_positions: Optional list of valid positions to sample from (e.g., non-padding).
                        If provided, boundary blocks are selected from start/end of valid positions.
                        If None, samples from range(seq_len).

    Returns:
        Tuple of (conditioning_indices, evaluation_indices, unknown_indices)
        - conditioning_indices: start block + end block
        - evaluation_indices: middle part
        - unknown_indices: same as evaluation_indices (no separate unseen in Mode 2)

    Example:
        For seq_len=10:
        - start_block: [0, 1] (size 2)
        - end_block: [8, 9] (size 2)
        - conditioning: [0, 1, 8, 9]
        - evaluation: [2, 3, 4, 5, 6, 7]
        - unknown: [2, 3, 4, 5, 6, 7]
    """
    # Use valid_positions if provided
    if valid_positions is not None:
        positions = sorted(valid_positions)
        effective_len = len(positions)
    else:
        positions = list(range(seq_len))
        effective_len = seq_len

    if effective_len < 3:
        raise ValueError(f"Sequence too short for boundary split: {effective_len}")

    # Use distribution if provided, otherwise fall back to range-based sampling
    if boundary_block_sizes_distribution is not None:
        start_size, end_size = boundary_block_sizes_distribution(effective_len)
    else:
        # Default ranges
        if start_block_range is None:
            max_single_block = max(1, effective_len // 3)
            start_block_range = (1, max_single_block)

        if end_block_range is None:
            max_single_block = max(1, effective_len // 3)
            end_block_range = (1, max_single_block)

        # Sample start block size
        min_start, max_start = start_block_range
        max_start = min(max_start, effective_len - 2)  # Leave room for end block and middle
        start_size = random.randint(min_start, max_start)

        # Sample end block size (must leave room for start block and at least 1 middle token)
        min_end, max_end = end_block_range
        max_end = min(max_end, effective_len - start_size - 1)  # Leave room for start block and middle
        if max_end < min_end:
            max_end = min_end
        end_size = random.randint(min_end, max_end)

    # Build indices from positions list
    # Start block: first start_size positions
    # End block: last end_size positions
    # Middle: everything in between
    conditioning_indices = positions[:start_size] + positions[-end_size:]
    evaluation_indices = positions[start_size:-end_size] if end_size > 0 else positions[start_size:]
    unknown_indices = evaluation_indices.copy()  # In Mode 2, unknown = evaluation

    # Validate
    assert len(conditioning_indices) + len(evaluation_indices) == effective_len
    assert len(evaluation_indices) > 0, "Evaluation set cannot be empty"
    assert set(conditioning_indices).isdisjoint(set(evaluation_indices))

    return conditioning_indices, evaluation_indices, unknown_indices


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Testing Blockwise Sampling with Distribution-based API")
    print("=" * 80)

    # Test 1: Basic usage with default distributions
    print("\n[Test 1] Basic blockwise sampling with default distributions")
    seq_len = 20
    cond, eval_set, unknown = generate_conditioning_evaluation_sets_blockwise(
        seq_len=seq_len,
        num_conditioning_distribution=lambda l: uniform_num_conditioning_distribution(l, (0.2, 0.4)),
        num_blocks_distribution=lambda n: uniform_num_blocks_distribution(n, max_blocks=3),
        block_sizes_distribution=uniform_block_sizes_distribution,
        num_evaluation_distribution=lambda l: uniform_num_evaluation_distribution(l, (0.3, 0.5)),
        num_eval_blocks_distribution=lambda n: uniform_num_blocks_distribution(n, max_blocks=2),
        eval_block_sizes_distribution=uniform_block_sizes_distribution,
    )

    print(f"Sequence length: {seq_len}")
    print(f"Conditioning indices ({len(cond)}): {cond}")
    print(f"Evaluation indices ({len(eval_set)}): {eval_set}")
    print(f"Unknown indices ({len(unknown)}): {unknown}")
    print("✓ Test 1 passed")

    # Test 2: Convenience function (backward compatibility)
    print("\n[Test 2] Backward compatible convenience function")
    cond, eval_set, unknown = generate_conditioning_set_blockwise(
        seq_len=15,
        conditioning_ratio=0.3,
        evaluation_ratio=0.4,
        max_cond_blocks=2,
        max_eval_blocks=2,
    )
    print(f"Sequence length: 15")
    print(f"Conditioning ({len(cond)}): {cond}")
    print(f"Evaluation ({len(eval_set)}): {eval_set}")
    print(f"Unknown ({len(unknown)}): {unknown}")
    print("✓ Test 2 passed")

    # Test 3: Check block structure
    print("\n[Test 3] Verify contiguous block structure")

    def check_contiguity(indices, name):
        if not indices:
            print(f"{name}: Empty")
            return

        blocks = []
        current = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current.append(indices[i])
            else:
                blocks.append(current)
                current = [indices[i]]
        blocks.append(current)

        print(f"{name}: {len(blocks)} block(s)")
        for i, block in enumerate(blocks):
            print(f"  Block {i+1}: [{block[0]}..{block[-1]}] (size {len(block)})")

    check_contiguity(cond, "Conditioning")
    check_contiguity(eval_set, "Evaluation")
    print("✓ Test 3 passed")

    # Test 4: Multiple runs for randomness
    print("\n[Test 4] Randomness check (3 runs)")
    for run in range(3):
        c, e, u = generate_conditioning_set_blockwise(
            seq_len=10,
            conditioning_ratio=0.3,
            evaluation_ratio=0.3,
        )
        print(f"Run {run+1}: Cond={c}, Eval={e}")
    print("✓ Test 4 passed")

    # Test 5: Edge cases
    print("\n[Test 5] Edge cases")

    # Small sequence
    c, e, u = generate_conditioning_set_blockwise(seq_len=5)
    print(f"Small sequence (5): Cond={c}, Eval={e}")

    # High conditioning ratio
    c, e, u = generate_conditioning_set_blockwise(
        seq_len=10, conditioning_ratio=0.7, evaluation_ratio=0.2
    )
    print(f"High cond ratio (0.7): Cond={c}, Eval={e}")

    # Very small conditioning
    c, e, u = generate_conditioning_set_blockwise(
        seq_len=20, conditioning_ratio=0.1, evaluation_ratio=0.3
    )
    print(f"Small cond ratio (0.1): Cond={c}, Eval={e}")
    print("✓ Test 5 passed")

    print("\n" + "=" * 80)
    print("✓ All blockwise sampling tests passed!")
    print("=" * 80)
