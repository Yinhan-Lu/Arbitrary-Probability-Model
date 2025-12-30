"""
Thinking Token Module for SigmaGPT

Provides learnable "thinking tokens" that are prepended to input sequences,
allowing the model to perform latent computation before generating.

This provides a fair comparison to conditioning tokens in the conditional model,
giving SigmaGPT equivalent "extra computation" at the start of sequences.

Key Components:
- ThinkingTokenPrepender: Module that prepends thinking tokens and adjusts positions
- compute_thinking_token_count: Helper to calculate n based on conditioning config

Usage:
    # Compute number of thinking tokens based on config
    n = compute_thinking_token_count(cond_pct_max=0.4, mode='expectation')  # -> 205

    # Create prepender with token IDs from TokenManager
    prepender = ThinkingTokenPrepender(thinking_token_ids, n_embd=768)

    # Apply in model forward
    new_inputs, new_order, new_targets = prepender(inputs, order, targets)
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def compute_thinking_token_count(cond_pct_max: float, max_seq_len: int = 1024,
                                  mode: str = "expectation") -> int:
    """
    Compute number of thinking tokens based on conditioning percentage configuration.

    The number of thinking tokens is designed to match the conditioning tokens
    in the conditional model for fair comparison:

    - Expectation mode: n = 0.5 * cond_pct_max * max_seq_len
      (Expected number of conditioning tokens for uniform 0-x% distribution)
    - Upper-bound mode: n = cond_pct_max * max_seq_len
      (Maximum possible conditioning tokens)

    Args:
        cond_pct_max: Maximum conditioning percentage (e.g., 0.4 for 40%)
        max_seq_len: Maximum sequence length (default: 1024)
        mode: "expectation" or "upper_bound"

    Returns:
        Number of thinking tokens to use

    Examples:
        >>> compute_thinking_token_count(0.4, 1024, "expectation")
        204
        >>> compute_thinking_token_count(0.4, 1024, "upper_bound")
        409
    """
    if mode == "expectation":
        # E[cond_tokens] for uniform(0, cond_pct_max) = 0.5 * cond_pct_max
        n = int(0.5 * cond_pct_max * max_seq_len)
    elif mode == "upper_bound":
        # Maximum conditioning tokens
        n = int(cond_pct_max * max_seq_len)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'expectation' or 'upper_bound'")

    logger.info(f"Computed thinking token count: n={n} "
                f"(cond_pct_max={cond_pct_max}, mode={mode})")
    return n


class ThinkingTokenPrepender(nn.Module):
    """
    Prepends thinking tokens to input sequences and adjusts order/targets.

    Thinking tokens are unique learnable tokens [think_1], [think_2], ..., [think_n]
    that are prepended to the input sequence. They allow the model to perform
    "latent computation" before generating the actual sequence.

    Transformations:
        inputs:  (B, T) -> (B, n + T)      # Prepend thinking token IDs
        order:   (B, T+1) -> (B, n + T + 1) # Shift body positions by n
        targets: (B, T) -> (B, n + T)      # Prepend -1 for no-loss positions

    Position Encoding:
        - Thinking tokens: positions 0 to n-1 (sequential)
        - Body tokens: original positions + n (shifted)

    Attention Pattern:
        - Standard causal mask extended to include thinking prefix
        - Thinking tokens attend causally to previous thinking tokens
        - Body tokens can attend to all thinking tokens + causal body

    Args:
        thinking_token_ids: List of token IDs for [think_1], ..., [think_n]
        n_embd: Embedding dimension (for future extensions)
    """

    def __init__(self, thinking_token_ids: list, n_embd: int):
        super().__init__()
        self.n_thinking = len(thinking_token_ids)
        self.n_embd = n_embd

        # Register thinking token IDs as buffer (moves with model to device)
        self.register_buffer(
            "thinking_ids",
            torch.tensor(thinking_token_ids, dtype=torch.long)
        )

        logger.info(f"ThinkingTokenPrepender initialized with {self.n_thinking} tokens "
                    f"(IDs: {thinking_token_ids[0]} to {thinking_token_ids[-1]})")

    def forward(self, inputs: torch.Tensor, order: torch.Tensor,
                targets: torch.Tensor = None) -> tuple:
        """
        Prepend thinking tokens and adjust order/targets.

        Args:
            inputs: (B, T) input token IDs
            order: (B, T+1) order tensor (positions + seq_len marker)
            targets: (B, T) target tokens, optional

        Returns:
            new_inputs: (B, n + T) with thinking tokens prepended
            new_order: (B, n + T + 1) with positions shifted by n
            new_targets: (B, n + T) with -1 for thinking positions (or None)

        Order Tensor Format:
            Original: [cond_pos_1, ..., cond_pos_k, eval_pos_1, ..., eval_m, seq_len]
            New: [0, 1, ..., n-1, n+cond_pos_1, ..., n+eval_m, n+seq_len]
        """
        B, T = inputs.shape
        n = self.n_thinking
        device = inputs.device

        # 1. Prepend thinking tokens to inputs
        # thinking_ids: (n,) -> (1, n) -> (B, n)
        think_prefix = self.thinking_ids.unsqueeze(0).expand(B, -1)
        new_inputs = torch.cat([think_prefix, inputs], dim=1)  # (B, n + T)

        # 2. Create order for thinking + shifted body
        # Thinking order: positions 0 to n-1, with "next" positions 1 to n
        # For SigmaGPT order format: [current_1, current_2, ..., next_of_last]
        # Thinking: [0, 1, 2, ..., n-1] as current positions
        # Body: original positions shifted by +n

        # Thinking order prefix: [0, 1, 2, ..., n-1]
        think_order_prefix = torch.arange(n, device=device, dtype=torch.long)
        think_order_prefix = think_order_prefix.unsqueeze(0).expand(B, -1)  # (B, n)

        # Body order: shift all positions by n
        # Original order[-1] is seq_len end marker -> becomes seq_len + n
        body_order = order + n  # (B, T+1)

        # Combine: [think_positions, shifted_body_order]
        # Full order: [0, 1, ..., n-1, n+cond_1, ..., n+seq_len]
        new_order = torch.cat([think_order_prefix, body_order], dim=1)  # (B, n + T + 1)

        # 3. Adjust targets: prepend -1 for thinking tokens (no loss)
        if targets is not None:
            think_targets = torch.full((B, n), -1, dtype=torch.long, device=device)
            new_targets = torch.cat([think_targets, targets], dim=1)  # (B, n + T)
        else:
            new_targets = None

        return new_inputs, new_order, new_targets

    def get_num_thinking_tokens(self) -> int:
        """Get the number of thinking tokens."""
        return self.n_thinking

    def extra_repr(self) -> str:
        return f"n_thinking={self.n_thinking}"


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("Testing ThinkingTokenPrepender")
    print("=" * 60)

    # Test compute function
    print("\n[Test 1] Token count calculation:")
    for cond_max in [0.2, 0.4, 0.6, 0.8, 1.0]:
        exp = compute_thinking_token_count(cond_max, 1024, "expectation")
        ub = compute_thinking_token_count(cond_max, 1024, "upper_bound")
        print(f"  cond_max={cond_max}: expectation={exp}, upper_bound={ub}")

    # Test prepender
    print("\n[Test 2] ThinkingTokenPrepender shapes:")
    n_thinking = 10
    thinking_ids = list(range(50258, 50258 + n_thinking))
    prepender = ThinkingTokenPrepender(thinking_ids, n_embd=768)

    B, T = 4, 20
    inputs = torch.randint(0, 1000, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)
    targets = torch.randint(0, 1000, (B, T))

    new_inputs, new_order, new_targets = prepender(inputs, order, targets)

    print(f"  inputs: ({B}, {T}) -> {new_inputs.shape}")
    print(f"  order: ({B}, {T+1}) -> {new_order.shape}")
    print(f"  targets: ({B}, {T}) -> {new_targets.shape}")

    assert new_inputs.shape == (B, n_thinking + T)
    assert new_order.shape == (B, n_thinking + T + 1)
    assert new_targets.shape == (B, n_thinking + T)
    print("  Shapes: OK")

    # Test order values
    print("\n[Test 3] Order tensor values:")
    print(f"  Original order[0]: {order[0][:5].tolist()}...{order[0][-3:].tolist()}")
    print(f"  New order[0]: {new_order[0][:5].tolist()}...{new_order[0][-3:].tolist()}")

    # Test target masking
    print("\n[Test 4] Target masking:")
    print(f"  First {n_thinking} targets should be -1: {new_targets[0, :n_thinking].tolist()}")
    assert (new_targets[:, :n_thinking] == -1).all()
    print("  Masking: OK")

    # Test thinking tokens prepended
    print("\n[Test 5] Thinking tokens prepended:")
    print(f"  First {n_thinking} inputs: {new_inputs[0, :n_thinking].tolist()}")
    print(f"  Expected: {thinking_ids}")
    assert new_inputs[0, :n_thinking].tolist() == thinking_ids
    print("  Prepending: OK")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
