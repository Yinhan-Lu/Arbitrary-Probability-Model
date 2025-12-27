"""
Rotary Position Embedding (RoPE) Implementation

RoPE encodes position information by rotating query and key vectors
in the attention mechanism, rather than adding position embeddings
at the input layer.

Key benefits:
- No learnable position parameters
- Better extrapolation to longer sequences
- Encodes relative position naturally

Reference: RoFormer (Su et al., 2021)
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding

    Applies rotation to query and key vectors based on their positions.
    Supports custom position_ids for conditional probability modeling.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Args:
            dim: Dimension per attention head (head_dim), e.g., 64
            max_seq_len: Maximum sequence length for precomputation
            base: Base frequency for rotation angles (higher = lower frequency)
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies: theta_i = 1 / (base^(2i/dim))
        # Lower dimensions get higher frequencies, higher dimensions get lower frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # (dim/2,)

        # Precompute cos/sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin values for positions 0 to seq_len-1"""
        # Positions: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)

        # Angle matrix: positions x inv_freq -> (seq_len, dim/2)
        # angles[i, j] = position_i * theta_j
        angles = torch.outer(positions, self.inv_freq)

        # Duplicate to match dim: (seq_len, dim/2) -> (seq_len, dim)
        # Each pair of dimensions shares the same angle
        angles = torch.cat([angles, angles], dim=-1)

        # Cache cos and sin
        self.register_buffer("cos_cached", angles.cos())  # (seq_len, dim)
        self.register_buffer("sin_cached", angles.sin())  # (seq_len, dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor = None
    ) -> tuple:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, n_head, seq_len, head_dim)
            k: Key tensor of shape (batch, n_head, seq_len, head_dim)
            position_ids: Position indices of shape (batch, seq_len) or None
                - None: Use default sequential positions [0, 1, 2, ...]
                - Tensor: Use custom positions (for conditional probability modeling)

        Returns:
            q_rotated, k_rotated: Rotated tensors with same shape as input
        """
        seq_len = q.shape[2]

        if position_ids is None:
            # Default: sequential positions [0, 1, 2, ..., seq_len-1]
            cos = self.cos_cached[:seq_len]  # (seq_len, dim)
            sin = self.sin_cached[:seq_len]
        else:
            # Custom positions: index into cache using position_ids
            # position_ids: (batch, seq_len) -> cos: (batch, seq_len, dim)
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]

        # Apply rotation to q and k
        q_rotated = self._apply_rotation(q, cos, sin)
        k_rotated = self._apply_rotation(k, cos, sin)

        return q_rotated, k_rotated

    def _apply_rotation(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply 2D rotation to each pair of dimensions.

        Rotation formula for each (x1, x2) pair:
            x1' = x1 * cos - x2 * sin
            x2' = x1 * sin + x2 * cos

        Args:
            x: Input tensor of shape (batch, n_head, seq_len, head_dim)
            cos: Cosine values, shape (seq_len, dim) or (batch, seq_len, dim)
            sin: Sine values, same shape as cos

        Returns:
            Rotated tensor with same shape as x
        """
        # Adjust cos/sin shape for broadcasting with x
        if cos.dim() == 2:
            # (seq_len, dim) -> (1, 1, seq_len, dim)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            # (batch, seq_len, dim) -> (batch, 1, seq_len, dim)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # Split into two halves for rotation
        # x: (batch, n_head, seq_len, head_dim)
        half_dim = self.dim // 2
        x1 = x[..., :half_dim]       # First half: (batch, n_head, seq_len, half_dim)
        x2 = x[..., half_dim:]       # Second half

        cos1 = cos[..., :half_dim]
        sin1 = sin[..., :half_dim]

        # Apply 2D rotation formula
        rotated_x1 = x1 * cos1 - x2 * sin1
        rotated_x2 = x1 * sin1 + x2 * cos1

        # Concatenate back to original dimension
        return torch.cat([rotated_x1, rotated_x2], dim=-1)
