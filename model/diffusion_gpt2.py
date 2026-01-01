"""
Diffusion GPT-2 Model for Discrete Diffusion Language Modeling

This module implements a GPT-2 based model adapted for discrete diffusion
(MDLM-style absorbing state diffusion). The key differences from standard
autoregressive GPT-2 are:

1. Timestep embedding: Model is conditioned on diffusion timestep t
2. Bidirectional attention: No causal masking (all positions attend to all)
3. Prediction target: Predict original tokens x_0 from noisy x_t

The architecture reuses the same TransformerBlock components from GPT-2
to ensure fair comparison with autoregressive baselines.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

# Import shared components from arbitrary_prob_gpt2
from model.arbitrary_prob_gpt2 import (
    GPT2Config,
    NewGELU,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
)
from model.diffusion_utils import SinusoidalPositionEmbedding


class DiffusionTransformerBlock(nn.Module):
    """
    Transformer block for diffusion model.

    This is nearly identical to the standard TransformerBlock, but uses
    bidirectional attention by default (no causal mask).
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x, attention_mask=None, position_ids=None):
        """
        Forward pass with bidirectional attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional mask of shape (batch_size, 1, seq_len, seq_len)
                          If None, uses full bidirectional attention
            position_ids: Optional position indices for RoPE

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Pre-LayerNorm architecture (GPT-2 style)
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask, position_ids=position_ids)
        x = x + self.mlp(self.ln_2(x))
        return x


class DiffusionGPT2Model(nn.Module):
    """
    GPT-2 model adapted for discrete diffusion.

    This model takes a noisy sequence x_t and timestep t, and predicts
    the original clean sequence x_0. It uses the same transformer backbone
    as GPT-2 but with bidirectional attention.

    Key components:
    - Token embedding (shared with output projection)
    - Position embedding (learned or RoPE)
    - Timestep embedding (sinusoidal or learned)
    - N transformer blocks with bidirectional attention
    - Output projection to vocabulary

    Architecture: [Token + Pos + Time] → N × TransformerBlock → LN → LM Head
    """

    def __init__(self, config, mask_token_id, num_timesteps=1000,
                 time_emb_type='sinusoidal'):
        """
        Initialize the diffusion GPT-2 model.

        Args:
            config: GPT2Config object with model hyperparameters
            mask_token_id: Token ID for [MASK] token
            num_timesteps: Total number of diffusion timesteps T
            time_emb_type: Type of timestep embedding ('sinusoidal' or 'learned')
        """
        super().__init__()

        self.config = config
        self.mask_token_id = mask_token_id
        self.num_timesteps = num_timesteps

        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embeddings (if not using RoPE)
        self.position_encoding_type = getattr(config, 'position_encoding_type', 'learned')
        if self.position_encoding_type != 'rope':
            self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)

        # Timestep embedding
        self.time_emb_type = time_emb_type
        if time_emb_type == 'sinusoidal':
            self.time_emb = SinusoidalPositionEmbedding(config.n_embd, max_timesteps=num_timesteps)
            # Project sinusoidal embedding to model dimension
            self.time_proj = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd * 4),
                NewGELU(),
                nn.Linear(config.n_embd * 4, config.n_embd),
            )
        else:
            # Learned timestep embedding
            self.time_emb = nn.Embedding(num_timesteps + 1, config.n_embd)
            self.time_proj = nn.Identity()

        # Embedding dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks (using diffusion-specific blocks)
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)

        # Output projection (language modeling head)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output projection
        self.lm_head.weight = self.wte.weight

        # Gradient checkpointing
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)

        # Initialize weights
        self.apply(self._init_weights)

        # Register buffer for bidirectional attention mask
        self.register_buffer(
            "bidirectional_mask",
            torch.ones(config.max_seq_len, config.max_seq_len).view(
                1, 1, config.max_seq_len, config.max_seq_len
            )
        )

    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x_t, t, attention_mask=None, position_ids=None):
        """
        Forward pass: predict x_0 from noisy x_t conditioned on timestep t.

        Args:
            x_t: Noisy input sequence of shape (batch_size, seq_len)
                 Contains original tokens at conditioning positions and
                 [MASK] tokens at corrupted positions
            t: Timestep tensor of shape (batch_size,)
            attention_mask: Optional attention mask of shape (batch_size, 1, seq_len, seq_len)
                          If None, uses full bidirectional attention (default for diffusion)
            position_ids: Optional position indices of shape (batch_size, seq_len)
                         Required for RoPE, optional for learned positions

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
                   Represents predictions for original tokens at all positions
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device

        # Token embeddings
        tok_emb = self.wte(x_t)  # (batch_size, seq_len, n_embd)

        # Position embeddings
        if self.position_encoding_type != 'rope':
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_emb = self.wpe(position_ids)  # (batch_size, seq_len, n_embd)
        else:
            pos_emb = 0  # RoPE handles positions in attention

        # Timestep embedding
        if self.time_emb_type == 'sinusoidal':
            time_emb = self.time_emb(t)  # (batch_size, n_embd)
            time_emb = self.time_proj(time_emb)  # (batch_size, n_embd)
        else:
            time_emb = self.time_emb(t)  # (batch_size, n_embd)
            time_emb = self.time_proj(time_emb)

        # Broadcast time embedding to all positions
        time_emb = time_emb.unsqueeze(1)  # (batch_size, 1, n_embd)

        # Combine embeddings
        x = tok_emb + pos_emb + time_emb
        x = self.drop(x)

        # Create bidirectional attention mask if not provided
        if attention_mask is None:
            attention_mask = self.bidirectional_mask[:, :, :seq_len, :seq_len]
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)

        # Apply transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = gradient_checkpoint(
                    block,
                    x,
                    attention_mask,
                    position_ids if self.position_encoding_type == 'rope' else None,
                    use_reentrant=False
                )
            else:
                x = block(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids if self.position_encoding_type == 'rope' else None
                )

        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        return logits

    def compute_loss(self, x_t, t, x_0, noise_mask, attention_mask=None, position_ids=None):
        """
        Compute the diffusion training loss.

        Args:
            x_t: Noisy input sequence of shape (batch_size, seq_len)
            t: Timestep tensor of shape (batch_size,)
            x_0: Original clean sequence of shape (batch_size, seq_len)
            noise_mask: Boolean mask of masked positions, shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            position_ids: Optional position indices

        Returns:
            loss: Scalar loss value
        """
        # Forward pass
        logits = self.forward(x_t, t, attention_mask, position_ids)

        # Compute loss only on masked positions
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = x_0.view(-1)
        mask_flat = noise_mask.view(-1).float()

        # Cross entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)

        return loss

    @torch.no_grad()
    def sample(self, noise_schedule, batch_size=1, seq_len=None,
               conditioning_tokens=None, conditioning_positions=None,
               num_steps=None, temperature=1.0, top_k=None, device=None):
        """
        Sample from the diffusion model using iterative denoising.

        This is the generation process: start from pure noise (all [MASK])
        and iteratively denoise to get the final sequence.

        Args:
            noise_schedule: NoiseSchedule object
            batch_size: Number of sequences to generate
            seq_len: Sequence length (uses config.max_seq_len if None)
            conditioning_tokens: Optional tensor of conditioning tokens
            conditioning_positions: Optional list of conditioning positions
            num_steps: Number of denoising steps (uses T if None)
            temperature: Sampling temperature
            top_k: Top-k sampling (None for standard sampling)
            device: Device to generate on

        Returns:
            Generated sequences of shape (batch_size, seq_len)
        """
        if seq_len is None:
            seq_len = self.config.max_seq_len
        if device is None:
            device = next(self.parameters()).device
        if num_steps is None:
            num_steps = noise_schedule.T

        # Start with all [MASK] tokens
        x_t = torch.full((batch_size, seq_len), self.mask_token_id,
                        dtype=torch.long, device=device)

        # Set conditioning tokens if provided
        if conditioning_tokens is not None and conditioning_positions is not None:
            for i, pos in enumerate(conditioning_positions):
                x_t[:, pos] = conditioning_tokens[:, i]

        # Create conditioning mask
        cond_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        if conditioning_positions is not None:
            for pos in conditioning_positions:
                cond_mask[:, pos] = True

        # Iterative denoising
        timesteps = torch.linspace(noise_schedule.T, 1, num_steps, dtype=torch.long, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)

            # Get model predictions
            logits = self.forward(x_t, t)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k if specified
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, :, [-1]]] = -float('inf')

            # Sample from predictions
            probs = F.softmax(logits, dim=-1)
            predictions = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
            predictions = predictions.view(batch_size, seq_len)

            # Determine which positions to update (unmask)
            # More positions get unmasked as t decreases
            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1]
                unmask_prob = (t_val - next_t) / noise_schedule.T
                unmask = torch.rand(batch_size, seq_len, device=device) < unmask_prob
                unmask = unmask & (x_t == self.mask_token_id)
            else:
                # Last step: unmask everything
                unmask = (x_t == self.mask_token_id)

            # Update positions
            x_t = torch.where(unmask, predictions, x_t)

            # Restore conditioning tokens
            if conditioning_tokens is not None:
                x_t = torch.where(cond_mask, conditioning_tokens.expand(batch_size, -1), x_t)

        return x_t

    def get_num_params(self, non_embedding=True):
        """Count the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if hasattr(self, 'wpe'):
                n_params -= self.wpe.weight.numel()
        return n_params


if __name__ == "__main__":
    # Test the diffusion model
    print("=" * 80)
    print("Testing DiffusionGPT2Model")
    print("=" * 80)

    # Create a small config for testing
    config = GPT2Config(
        vocab_size=50258,  # +1 for mask token
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=64,
        dropout=0.1
    )

    mask_token_id = 50257
    model = DiffusionGPT2Model(config, mask_token_id=mask_token_id, num_timesteps=1000)

    print(f"\n1. Model created with {model.get_num_params():,} parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    x_t = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    t = torch.randint(1, 1001, (batch_size,))

    print(f"\n2. Testing forward pass:")
    print(f"   Input shape: {x_t.shape}")
    print(f"   Timesteps: {t.tolist()}")

    logits = model(x_t, t)
    print(f"   Output shape: {logits.shape}")

    # Test loss computation
    print(f"\n3. Testing loss computation:")
    x_0 = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    noise_mask = torch.rand(batch_size, seq_len) > 0.5

    loss = model.compute_loss(x_t, t, x_0, noise_mask)
    print(f"   Loss value: {loss.item():.4f}")

    # Test with noise schedule
    print(f"\n4. Testing with noise schedule:")
    from model.diffusion_utils import NoiseSchedule

    schedule = NoiseSchedule(num_timesteps=1000)
    t = schedule.sample_timesteps(batch_size)
    x_t, mask = schedule.add_noise(x_0, t, mask_token_id)

    logits = model(x_t, t)
    loss = model.compute_loss(x_t, t, x_0, mask)
    print(f"   Generated timesteps: {t.tolist()}")
    print(f"   Masked positions per sample: {mask.sum(dim=1).tolist()}")
    print(f"   Loss: {loss.item():.4f}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
