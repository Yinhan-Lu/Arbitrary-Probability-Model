"""
Order utilities for Sigma GPT

Key functions for converting conditioning/evaluation indices into order tensors
and applying them to create shuffled sequences with proper targets.

Reference: sigma-gpt/text/order.py
"""

import torch


def indices_to_order(cond_indices, eval_indices, seq_len):
    """
    Convert conditioning and evaluation indices to Sigma GPT order tensor.

    Order format: [cond_positions, eval_positions, unseen_positions, seq_len]
    - Conditioning positions come first (model can attend to them)
    - Evaluation positions come second (positions where we compute loss)
    - Unseen positions come last (positions the model doesn't see during training)

    Args:
        cond_indices: (B, num_cond) indices of conditioning positions
        eval_indices: (B, num_eval) indices of evaluation positions
        seq_len: int, total sequence length

    Returns:
        order: (B, T+1) order tensor where T = len(cond) + len(eval)
              Last element is always seq_len (Sigma GPT convention)

    Example:
        cond_indices = [[0, 1, 2]]  # First 3 positions
        eval_indices = [[3, 4]]     # Next 2 positions
        seq_len = 10
        -> order = [[0, 1, 2, 3, 4, 10]]
                    ^^^^^^^  ^^^^  ^^
                    cond     eval  seq_len (unseen: 5-9)
    """
    B = cond_indices.shape[0]
    device = cond_indices.device

    # Concatenate: [cond_positions, eval_positions, seq_len]
    # Total length: num_cond + num_eval + 1
    order_parts = [
        cond_indices,  # (B, num_cond)
        eval_indices,  # (B, num_eval)
        torch.full((B, 1), seq_len, device=device, dtype=torch.long)  # (B, 1)
    ]

    order = torch.cat(order_parts, dim=1)  # (B, num_cond + num_eval + 1)

    return order


def apply_order(tokens, order):
    """
    Apply order to token sequence, creating inputs and targets.

    This function:
    1. Reorders tokens according to order tensor
    2. Creates input sequence (all but last position)
    3. Creates target sequence (all but first position)

    The model will see:
    - Input: tokens at positions order[:-1] (conditioning + evaluation)
    - Target: tokens at positions order[1:] (what to predict next)

    Reference: sigma-gpt/text/order.py line 152-170

    Args:
        tokens: (B, seq_len) original token sequence
        order: (B, T+1) order tensor

    Returns:
        inputs: (B, T) reordered input tokens
        targets: (B, T) target tokens to predict

    Example:
        tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        order = [0, 1, 2, 3, 4, 10]  # Conditioning: 0-2, Eval: 3-4, Unseen: 5-9

        Reordered: [tok0, tok1, tok2, tok3, tok4]
        inputs  = [tok0, tok1, tok2, tok3, tok4]  # What model sees
        targets = [tok1, tok2, tok3, tok4, ???]   # What model predicts
                                           ^-- This will be -1 (no next token)
    """
    # Gather tokens according to order
    # order[:, :-1] gives us positions to take (excluding final seq_len marker)
    order_indices = order[:, :-1]  # (B, T)

    # Reorder tokens: gather tokens at positions specified by order
    reordered = tokens.gather(1, order_indices)  # (B, T)

    # Create inputs and targets (shifted by 1)
    # But order[1:] might reference seq_len (which is out of bounds)
    # We need to be careful here

    # Get target positions (what comes after each input position in the order)
    order_next = order[:, 1:]  # (B, T)

    # Gather target tokens
    # Clamp to avoid out of bounds (seq_len -> last token)
    seq_len = tokens.shape[1]
    order_next_clamped = order_next.clamp(max=seq_len - 1)
    targets = tokens.gather(1, order_next_clamped)  # (B, T)

    # Mark positions where order[i+1] == seq_len as invalid (target = -1)
    # These are positions where there's no next token in the sequence
    invalid_mask = (order_next >= seq_len)
    targets = targets.masked_fill(invalid_mask, -1)

    inputs = reordered

    return inputs, targets


def create_labels(order, seq_len, mode='fair'):
    """
    Create label tensor with -1 for positions to ignore in loss.

    In 'fair' mode (matching original Sigma GPT):
    - Only evaluation positions compute loss
    - Conditioning positions: label = -1 (ignored)
    - Evaluation positions: label = target token
    - Positions predicting unseen tokens: label = -1 (ignored)

    In 'full' mode (all positions learn):
    - All valid positions compute loss
    - Only positions predicting beyond seq_len are ignored

    This is used in conjunction with apply_order to create the final targets.

    Args:
        order: (B, T+1) order tensor
        seq_len: int, total sequence length
        mode: 'fair' or 'full'
              - 'fair': only evaluation positions compute loss (~20-30% learning)
              - 'full': all positions compute loss (100% learning)

    Returns:
        mask: (B, T) boolean tensor
              True = compute loss, False = ignore (will be set to -1)

    Example ('fair' mode):
        order = [[0, 1, 2, 3, 4, 10]]
                 ^^^^^^^  ^^^^
                 cond     eval
        cond_size = 3, eval_size = 2

        mask = [[False, False, False, True, True]]
                 ^^^^^^^^^^^^^^^^^^^^^^ ^^^^ ^^^^
                 ignore (cond)          compute loss (eval)
    """
    B, T_plus_1 = order.shape
    T = T_plus_1 - 1

    if mode == 'fair':
        # Need to determine which positions are conditioning vs evaluation
        # Convention: assume first positions are conditioning
        # But we don't have this info here... we need to pass it
        # Actually, let's return this information differently
        # Return a function that takes cond_size
        raise NotImplementedError(
            "create_labels needs cond_size parameter. "
            "Use create_labels_fair(order, cond_size, seq_len) or "
            "create_labels_full(order, seq_len) instead."
        )
    elif mode == 'full':
        return create_labels_full(order, seq_len)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_labels_fair(order, cond_size, seq_len):
    """
    Create labels for 'fair' mode: only evaluation positions compute loss.

    Args:
        order: (B, T+1) order tensor
        cond_size: int, number of conditioning positions
        seq_len: int, total sequence length

    Returns:
        mask: (B, T) boolean tensor (True = compute loss)
    """
    B, T_plus_1 = order.shape
    T = T_plus_1 - 1
    device = order.device

    # Create mask: False for first cond_size positions, True for rest
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    # Evaluation positions start at cond_size
    if cond_size < T:
        mask[:, cond_size:] = True

    # Also mask out positions that predict beyond seq_len
    order_next = order[:, 1:]  # (B, T)
    invalid = (order_next >= seq_len)
    mask = mask & ~invalid

    return mask


def create_labels_full(order, seq_len):
    """
    Create labels for 'full' mode: all positions compute loss.

    Args:
        order: (B, T+1) order tensor
        seq_len: int, total sequence length

    Returns:
        mask: (B, T) boolean tensor (True = compute loss)
    """
    B, T_plus_1 = order.shape
    T = T_plus_1 - 1
    device = order.device

    # All positions compute loss, except those predicting beyond seq_len
    order_next = order[:, 1:]  # (B, T)
    mask = order_next < seq_len  # (B, T)

    return mask


def apply_labels_mask(targets, mask):
    """
    Apply mask to targets, setting ignored positions to -1.

    Args:
        targets: (B, T) target tokens
        mask: (B, T) boolean mask (True = keep, False = ignore)

    Returns:
        targets: (B, T) targets with -1 for ignored positions
    """
    return targets.masked_fill(~mask, -1)


# Simplified API for end-to-end usage
def prepare_sigmagpt_batch(tokens, cond_indices, eval_indices, mode='fair'):
    """
    Prepare a batch for Sigma GPT training.

    This is the main API function that combines all steps:
    1. Create order tensor from indices
    2. Apply order to create inputs and initial targets
    3. Create label mask based on mode
    4. Apply mask to targets (set ignored positions to -1)

    Args:
        tokens: (B, seq_len) original token sequence
        cond_indices: (B, num_cond) conditioning position indices
        eval_indices: (B, num_eval) evaluation position indices
        mode: 'fair' or 'full'

    Returns:
        inputs: (B, T) input tokens for model
        order: (B, T+1) order tensor for model
        targets: (B, T) target tokens with -1 for ignored positions

    Example:
        tokens = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
        cond_indices = torch.tensor([[0, 1, 2]])
        eval_indices = torch.tensor([[3, 4]])
        mode = 'fair'

        inputs, order, targets = prepare_sigmagpt_batch(tokens, cond_indices, eval_indices, mode)

        inputs  = [[10, 20, 30, 40, 50]]
        order   = [[0, 1, 2, 3, 4, 10]]
        targets = [[-1, -1, -1, 40, 50]]  # Only eval positions have valid targets
                   ^^^^^^^^^^^ conditioning (ignored)
                               ^^^^^^^^^ evaluation (compute loss)
    """
    B, seq_len = tokens.shape

    # Step 1: Create order tensor
    order = indices_to_order(cond_indices, eval_indices, seq_len)

    # Step 2: Apply order to get inputs and targets
    inputs, targets = apply_order(tokens, order)

    # Step 3: Create and apply label mask
    cond_size = cond_indices.shape[1]

    if mode == 'fair':
        mask = create_labels_fair(order, cond_size, seq_len)
    elif mode == 'full':
        mask = create_labels_full(order, seq_len)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    targets = apply_labels_mask(targets, mask)

    return inputs, order, targets
