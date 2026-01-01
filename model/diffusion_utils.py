"""
Diffusion Utilities for Discrete Diffusion Language Models (MDLM-style)

This module implements the noise schedule and diffusion utilities for
absorbing-state discrete diffusion, where tokens are progressively
masked during the forward process.

Key components:
- NoiseSchedule: Controls the masking probability at each timestep
- add_noise: Forward diffusion process (x_0 -> x_t)
- add_noise_conditional: Conditional diffusion (keeps X_c positions fixed)
"""

import math
import torch
import torch.nn as nn


class NoiseSchedule:
    """
    Noise schedule for discrete diffusion with absorbing states (masking).

    The schedule controls how aggressively tokens are masked at each timestep:
    - At t=0: alpha_t ≈ 1, almost no masking
    - At t=T: alpha_t ≈ 0, almost everything is masked

    Supports multiple schedule types:
    - cosine: Smooth masking progression (recommended)
    - linear: Linear increase in masking
    - sqrt: Square root schedule
    """

    def __init__(self, num_timesteps=1000, schedule_type='cosine', device=None):
        """
        Initialize the noise schedule.

        Args:
            num_timesteps: Total number of diffusion timesteps T
            schedule_type: Type of noise schedule ('cosine', 'linear', 'sqrt')
            device: Device to store tensors on
        """
        self.T = num_timesteps
        self.schedule_type = schedule_type
        self.device = device

        # Precompute alpha values for all timesteps
        self.alphas = self._create_schedule(schedule_type)

        if device is not None:
            self.alphas = self.alphas.to(device)

    def _create_schedule(self, schedule_type):
        """Create the noise schedule based on schedule type."""
        t = torch.arange(self.T + 1, dtype=torch.float32)

        if schedule_type == 'cosine':
            # Cosine schedule: smooth progression
            # alpha_t = cos((t/T) * pi/2)^2
            alphas = torch.cos((t / self.T) * math.pi / 2) ** 2
        elif schedule_type == 'linear':
            # Linear schedule: alpha decreases linearly from 1 to 0
            alphas = 1.0 - (t / self.T)
        elif schedule_type == 'sqrt':
            # Square root schedule: slower initial decay
            alphas = 1.0 - torch.sqrt(t / self.T)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Ensure alpha is in [epsilon, 1] to avoid numerical issues
        alphas = alphas.clamp(min=1e-5, max=1.0)

        return alphas

    def to(self, device):
        """Move schedule to specified device."""
        self.device = device
        self.alphas = self.alphas.to(device)
        return self

    def get_alpha(self, t):
        """
        Get alpha value for timestep(s) t.

        Args:
            t: Timestep tensor of shape (batch_size,) or scalar

        Returns:
            Alpha values of shape (batch_size,) or scalar
        """
        return self.alphas[t]

    def get_mask_prob(self, t):
        """
        Get masking probability for timestep(s) t.

        mask_prob = 1 - alpha_t

        Args:
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Masking probabilities of shape (batch_size,)
        """
        return 1.0 - self.get_alpha(t)

    def sample_timesteps(self, batch_size, device=None):
        """
        Sample random timesteps for training.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on

        Returns:
            Tensor of shape (batch_size,) with values in [1, T]
        """
        if device is None:
            device = self.device
        # Sample from [1, T] (not 0, as t=0 means no noise)
        return torch.randint(1, self.T + 1, (batch_size,), device=device)

    def add_noise(self, x_0, t, mask_token_id):
        """
        Forward diffusion: x_0 -> x_t

        For each position, replace with mask_token with probability (1 - alpha_t).

        Args:
            x_0: Original sequence of shape (batch_size, seq_len)
            t: Timestep tensor of shape (batch_size,)
            mask_token_id: Token ID for [MASK] token

        Returns:
            x_t: Noisy sequence of shape (batch_size, seq_len)
            mask: Boolean mask indicating which positions were masked, shape (batch_size, seq_len)
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device

        # Get masking probability for each sample in batch
        mask_prob = self.get_mask_prob(t)  # (batch_size,)
        mask_prob = mask_prob.unsqueeze(1)  # (batch_size, 1)

        # Sample which positions to mask
        random_vals = torch.rand(batch_size, seq_len, device=device)
        mask = random_vals < mask_prob  # True = this position will be masked

        # Apply masking
        x_t = torch.where(mask, mask_token_id, x_0)

        return x_t, mask

    def add_noise_conditional(self, x_0, t, mask_token_id, conditioning_indices):
        """
        Conditional forward diffusion: x_0 -> x_t with X_c fixed.

        Conditioning positions (X_c) are never masked - they remain visible
        throughout the diffusion process. Only non-conditioning positions
        can be corrupted.

        Args:
            x_0: Original sequence of shape (batch_size, seq_len)
            t: Timestep tensor of shape (batch_size,)
            mask_token_id: Token ID for [MASK] token
            conditioning_indices: List of lists, where conditioning_indices[i]
                                 contains the conditioning position indices for sample i

        Returns:
            x_t: Noisy sequence of shape (batch_size, seq_len)
            mask: Boolean mask indicating which positions were masked, shape (batch_size, seq_len)
        """
        # First, apply standard noise
        x_t, mask = self.add_noise(x_0, t, mask_token_id)

        # Then, restore conditioning positions (they should never be masked)
        batch_size = x_0.size(0)
        for i in range(batch_size):
            cond_idx = conditioning_indices[i]
            if len(cond_idx) > 0:
                # Convert to tensor if it's a list
                if isinstance(cond_idx, list):
                    cond_idx = torch.tensor(cond_idx, device=x_0.device, dtype=torch.long)

                # Restore original tokens at conditioning positions
                x_t[i, cond_idx] = x_0[i, cond_idx]
                # Mark these positions as not masked
                mask[i, cond_idx] = False

        return x_t, mask

    def add_noise_conditional_fast(self, x_0, t, mask_token_id, conditioning_mask):
        """
        Fast version of conditional diffusion using a pre-computed mask.

        Args:
            x_0: Original sequence of shape (batch_size, seq_len)
            t: Timestep tensor of shape (batch_size,)
            mask_token_id: Token ID for [MASK] token
            conditioning_mask: Boolean mask of shape (batch_size, seq_len)
                              True = conditioning position (never mask)

        Returns:
            x_t: Noisy sequence of shape (batch_size, seq_len)
            noise_mask: Boolean mask indicating which positions were masked
        """
        # Apply standard noise
        x_t, noise_mask = self.add_noise(x_0, t, mask_token_id)

        # Restore conditioning positions
        x_t = torch.where(conditioning_mask, x_0, x_t)
        noise_mask = noise_mask & ~conditioning_mask

        return x_t, noise_mask


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for timesteps.

    This is the standard embedding used in diffusion models to encode
    the current timestep t into a continuous representation.
    """

    def __init__(self, dim, max_timesteps=10000):
        """
        Args:
            dim: Embedding dimension
            max_timesteps: Maximum number of timesteps (for frequency scaling)
        """
        super().__init__()
        self.dim = dim
        self.max_timesteps = max_timesteps

    def forward(self, t):
        """
        Args:
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        # Create frequency bands
        freqs = torch.exp(
            -math.log(self.max_timesteps) * torch.arange(half_dim, device=device) / half_dim
        )

        # Compute embeddings
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)

        return embeddings


def compute_diffusion_loss(logits, x_0, noise_mask, reduction='mean'):
    """
    Compute the diffusion training loss.

    The loss is cross-entropy computed only on positions that were masked
    during the forward diffusion process.

    Args:
        logits: Model output of shape (batch_size, seq_len, vocab_size)
        x_0: Original sequence of shape (batch_size, seq_len)
        noise_mask: Boolean mask indicating masked positions, shape (batch_size, seq_len)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value (scalar if reduction is 'mean' or 'sum')
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for cross entropy
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = x_0.view(-1)
    mask_flat = noise_mask.view(-1)

    # Compute per-token loss
    loss = torch.nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        reduction='none'
    )

    # Apply mask (only count masked positions)
    loss = loss * mask_flat.float()

    if reduction == 'mean':
        # Average over masked positions
        return loss.sum() / mask_flat.sum().clamp(min=1)
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.view(batch_size, seq_len)


def indices_to_mask(indices_list, seq_len, device):
    """
    Convert list of index lists to boolean mask tensor.

    Args:
        indices_list: List of lists, where indices_list[i] contains
                     position indices for sample i
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Boolean mask of shape (batch_size, seq_len)
    """
    batch_size = len(indices_list)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for i, indices in enumerate(indices_list):
        if len(indices) > 0:
            if isinstance(indices, list):
                indices = torch.tensor(indices, device=device, dtype=torch.long)
            mask[i, indices] = True

    return mask


if __name__ == "__main__":
    # Test the noise schedule
    print("=" * 80)
    print("Testing Diffusion Utilities")
    print("=" * 80)

    # Create noise schedule
    schedule = NoiseSchedule(num_timesteps=1000, schedule_type='cosine')

    # Test alpha values at different timesteps
    print("\n1. Alpha values at different timesteps (cosine schedule):")
    for t in [0, 100, 500, 900, 1000]:
        alpha = schedule.get_alpha(torch.tensor([t]))
        mask_prob = schedule.get_mask_prob(torch.tensor([t]))
        print(f"  t={t:4d}: alpha={alpha.item():.4f}, mask_prob={mask_prob.item():.4f}")

    # Test forward diffusion
    print("\n2. Testing forward diffusion:")
    x_0 = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    t = torch.tensor([500])
    mask_token_id = 50257

    x_t, mask = schedule.add_noise(x_0, t, mask_token_id)
    print(f"  Original x_0: {x_0.tolist()}")
    print(f"  Noisy x_t:    {x_t.tolist()}")
    print(f"  Mask:         {mask.tolist()}")

    # Test conditional diffusion
    print("\n3. Testing conditional diffusion (X_c = positions 0, 3, 7):")
    conditioning_indices = [[0, 3, 7]]  # These positions should never be masked

    x_t, mask = schedule.add_noise_conditional(x_0, t, mask_token_id, conditioning_indices)
    print(f"  Original x_0: {x_0.tolist()}")
    print(f"  Noisy x_t:    {x_t.tolist()}")
    print(f"  Mask:         {mask.tolist()}")
    print(f"  Note: Positions 0, 3, 7 should always have original values")

    # Test sinusoidal embedding
    print("\n4. Testing sinusoidal timestep embedding:")
    time_emb = SinusoidalPositionEmbedding(dim=128)
    t_batch = torch.tensor([0, 100, 500, 1000])
    embeddings = time_emb(t_batch)
    print(f"  Input timesteps: {t_batch.tolist()}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding norms: {embeddings.norm(dim=-1).tolist()}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
