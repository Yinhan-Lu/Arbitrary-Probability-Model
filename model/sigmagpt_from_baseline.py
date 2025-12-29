"""
Sigma GPT Implementation Based on baseline_gpt2.py

This implementation adapts the clean baseline_gpt2.py architecture to implement
Sigma GPT's double position encoding mechanism.

Key differences from baseline_gpt2.py:
- Position embedding size: n_embd // 2 (instead of n_embd)
- Double position encoding: encodes both current and next positions
- Forward signature: forward(idx, order, targets) instead of forward(input_ids, attention_mask, labels)
- Loss computation: direct (no shift), ignore_index=-1 (Sigma GPT convention)

Key features of Sigma GPT:
- Arbitrary generation order via order tensor
- Teacher forcing training only (no generation methods)
- Compatible with existing GPT2Config

Reference:
- Sigma GPT paper: https://arxiv.org/abs/2404.09562
- Reference implementation: sigma-gpt/text/sigmagpt.py
- Base architecture: model/baseline_gpt2.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use shared config from the project (instead of defining our own)
from model.arbitrary_prob_gpt2 import GPT2Config


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism with causal masking"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Combined QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask (lower triangular)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            )
        )

        # Dual-axis RoPE for Sigma GPT (optional)
        self.position_encoding_type = getattr(config, 'position_encoding_type', 'learned')
        if self.position_encoding_type == "dual_rope":
            from model.rope import DualAxisRotaryEmbedding
            self.rotary_emb = DualAxisRotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=config.max_seq_len,
                base_axis1=getattr(config, 'rope_base_axis1', 10000.0),
                base_axis2=getattr(config, 'rope_base_axis2', 10000.0)
            )

    def forward(self, x, current_positions=None, next_positions=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            current_positions: (batch_size, seq_len) - for dual_rope mode
            next_positions: (batch_size, seq_len) - for dual_rope mode

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)

        Note: Sigma GPT uses standard causal masking (no custom attention masks)
        """
        B, T, C = x.size()  # batch_size, seq_len, n_embd

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply dual-axis RoPE if enabled
        if self.position_encoding_type == "dual_rope" and current_positions is not None:
            q, k = self.rotary_emb(q, k, current_positions, next_positions)

        # Compute attention scores: (B, n_head, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (always use default causal mask for Sigma GPT)
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf')
        )

        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Apply attention to values
        attn_output = attn_probs @ v  # (B, n_head, T, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        output = self.c_proj(attn_output)
        output = self.resid_dropout(output)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.mlp_hidden_size)
        self.c_proj = nn.Linear(config.mlp_hidden_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        if config.activation_function == "gelu_new":
            self.act = NewGELU()
        else:
            self.act = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and feed-forward layers"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x, current_positions=None, next_positions=None):
        # Pre-LayerNorm architecture (GPT-2 style)
        x = x + self.attn(self.ln_1(x), current_positions, next_positions)
        x = x + self.mlp(self.ln_2(x))
        return x


class SigmaGPTModel(nn.Module):
    """
    Sigma GPT Language Model - Based on baseline_gpt2.py architecture

    Architecture: Token Embedding + Double Position Encoding → N × TransformerBlock → LayerNorm → LM Head

    Key Innovation: Double Position Encoding
    - Standard GPT-2: Encodes only current position (n_embd dimensions)
    - Sigma GPT: Encodes both current AND next position (n_embd/2 each, concatenated to n_embd)

    This enables arbitrary generation orders by explicitly representing:
    - "Where we are" (current position in generation order)
    - "Where we're going" (next position in generation order)

    Usage Example:
        from model.sigmagpt_from_baseline import SigmaGPTModel, GPT2Config

        config = GPT2Config(vocab_size=50257, n_layer=6, n_head=12, n_embd=768)
        model = SigmaGPTModel(config)

        # Prepare inputs
        idx = torch.randint(0, config.vocab_size, (B, T))  # Token indices
        order = torch.cat([torch.randperm(T), torch.tensor([T])]).unsqueeze(0).expand(B, -1)  # Order tensor
        targets = torch.randint(0, config.vocab_size, (B, T))  # Target tokens

        # Forward pass
        logits, loss = model(idx, order, targets)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings (standard)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Position encoding type: 'learned' (default) or 'dual_rope'
        self.position_encoding_type = getattr(config, 'position_encoding_type', 'learned')

        # Position embeddings (only for learned mode)
        if self.position_encoding_type == "dual_rope":
            # RoPE mode: no learned position embeddings
            self.wpe = None
        else:
            # Learned mode: CRITICAL - n_embd // 2 for double encoding
            # This is the key architectural difference from standard GPT-2
            self.wpe = nn.Embedding(config.max_seq_len, config.n_embd // 2)

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)

        # Language model head (tied with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight initialization
        self.apply(self._init_weights)

        # Tie weights between token embeddings and lm_head (GPT-2 standard)
        self.lm_head.weight = self.wte.weight

        pos_enc_info = f"position_encoding={self.position_encoding_type}"
        print(f"Sigma GPT Model (from baseline) initialized with {self.get_num_params()/1e6:.2f}M parameters, {pos_enc_info}")

    def _init_weights(self, module):
        """Initialize weights following GPT-2 paper"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())

    def _pos_emb(self, idx, order):
        """
        CRITICAL: Double position encoding implementation

        This is the core innovation of Sigma GPT. Instead of encoding just
        the current position, we encode both:
        - Current position (where we are in the generation order)
        - Next position (where we're going next)

        Each gets n_embd/2 dimensions, concatenated to n_embd total.

        Mathematical formulation:
            pos_emb = concat(embed(order[t]), embed(order[t+1]))
        where:
            - order[t] is the current position in generation sequence
            - order[t+1] is the next position to generate
            - Each embed produces n_embd/2 dimensions

        Example:
            If generating in order [2, 0, 1, 3] with seq_len=4:
            - At step 0: encode positions (2, 0) - "at pos 2, next generate pos 0"
            - At step 1: encode positions (0, 1) - "at pos 0, next generate pos 1"
            - At step 2: encode positions (1, 3) - "at pos 1, next generate pos 3"
            - At step 3: encode positions (3, 4) - "at pos 3, done (4 = seq_len)"

        Reference: sigma-gpt/text/sigmagpt.py line 346-363

        Args:
            idx: (B, T) token indices
            order: (B, T+1) order tensor
                  - order[:, :-1] are current positions
                  - order[:, 1:] are next positions
                  - Last element (order[:, T]) is typically seq_len (end marker)

        Returns:
            (B, T, n_embd) position embeddings
        """
        t = idx.size(1)
        assert t <= self.config.max_seq_len, \
            f"Sequence length {t} exceeds max {self.config.max_seq_len}"

        # Order should be length t + 1
        # If longer, truncate (shouldn't happen in normal usage)
        if order.size(1) > t + 1:
            order = order[:, :t + 1]

        # Extract current and next positions
        order_input = order[:, :-1, None]   # (B, T, 1) - current position
        order_target = order[:, 1:, None]   # (B, T, 1) - next position

        # Concatenate to (B, T, 2)
        order_cat = torch.cat((order_input, order_target), dim=2)

        # Clamp order values to valid position embedding range [0, max_seq_len-1]
        # Order tensor uses seq_len as "end-of-sequence" marker, but embedding table
        # only has indices [0, max_seq_len-1], so we need to clamp
        order_cat = order_cat.clamp(max=self.config.max_seq_len - 1)

        # Embed: (B, T, 2) -> (B, T, 2, n_embd/2)
        pos_emb = self.wpe(order_cat)

        # Flatten to (B, T, n_embd)
        # This concatenates current_pos_emb and next_pos_emb along the feature dimension
        return pos_emb.flatten(2)

    def forward(self, idx, order, targets=None):
        """
        Sigma GPT forward pass with double position encoding

        Args:
            idx: (B, T) token indices (tokens already arranged according to order)
                Note: For Sigma GPT, input tokens are typically shuffled according to
                the generation order specified by the order tensor

            order: (B, T+1) order tensor specifying generation sequence
                Example for T=4:
                    - Left-to-right: [0, 1, 2, 3, 4]
                    - Right-to-left: [3, 2, 1, 0, 4]
                    - Random: [2, 0, 3, 1, 4]
                Last element (order[:, T]) is typically T (end marker)

            targets: (B, T) target tokens for loss computation (optional)
                Use -1 for positions to ignore in loss (Sigma GPT convention)
                Note: Unlike standard GPT-2, NO shifting is needed - targets are
                already aligned with predictions via the order tensor

        Returns:
            logits: (B, T, vocab_size) - predicted logits for each position
            loss: scalar loss (or None if targets not provided)

        Key Differences from Standard GPT-2:
            1. Requires order tensor (not optional)
            2. Position encoding is double encoding (current + next)
            3. Loss computation is direct (no shift needed)
            4. ignore_index is -1 (not -100)
        """
        B, T = idx.size()
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds maximum {self.config.max_seq_len}"

        # Token embeddings (standard)
        tok_emb = self.wte(idx)  # (B, T, n_embd)

        # Handle position encoding based on mode
        if self.position_encoding_type == "dual_rope":
            # RoPE mode: no position embeddings added to input
            x = self.drop(tok_emb)

            # Extract positions for attention layers
            current_positions = order[:, :-1]  # (B, T) - current positions
            next_positions = order[:, 1:]      # (B, T) - next positions

            # Clamp to valid range (order tensor may have seq_len as end marker)
            current_positions = current_positions.clamp(max=self.config.max_seq_len - 1)
            next_positions = next_positions.clamp(max=self.config.max_seq_len - 1)
        else:
            # Learned mode: position embeddings (double encoding - Sigma GPT's key innovation)
            pos_emb = self._pos_emb(idx, order)  # (B, T, n_embd)

            # Combine and apply dropout
            x = self.drop(tok_emb + pos_emb)

            # No position IDs needed for learned mode
            current_positions = None
            next_positions = None

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, current_positions, next_positions)

        # Final layer norm
        x = self.ln_f(x)

        # Language model head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Sigma GPT: Direct prediction (no shifting)
            # The order tensor already handles the alignment between predictions and targets
            #
            # Standard GPT-2:
            #   shift_logits = logits[..., :-1, :]
            #   shift_labels = labels[..., 1:]
            #   loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            #
            # Sigma GPT:
            #   loss = cross_entropy(logits, targets, ignore_index=-1)
            #
            # Why no shift? Because input tokens and targets are already arranged
            # according to the generation order specified by the order tensor
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Sigma GPT convention (not PyTorch default -100)
            )

        return logits, loss


def create_causal_mask(seq_len, device='cpu'):
    """
    Create a causal (lower triangular) attention mask

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask of shape (1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0)


if __name__ == "__main__":
    # Test the Sigma GPT model
    print("Testing Sigma GPT Model (from baseline)...")
    print("=" * 80)

    # Create a tiny config for testing
    config = GPT2Config(
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=256,
        dropout=0.1
    )

    model = SigmaGPTModel(config)

    # Random input
    batch_size = 2
    seq_len = 10
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Create order tensor (left-to-right order for testing)
    order = torch.arange(seq_len + 1).unsqueeze(0).expand(batch_size, -1)

    # Create targets
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nInput shapes:")
    print(f"  idx: {idx.shape}")
    print(f"  order: {order.shape}")
    print(f"  targets: {targets.shape}")

    # Forward pass
    logits, loss = model(idx, order, targets)
    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  loss: {loss.item():.4f}")

    # Test with random order
    print("\n" + "=" * 80)
    print("Testing with random generation order...")
    random_order = torch.cat([
        torch.randperm(seq_len),
        torch.tensor([seq_len])
    ]).unsqueeze(0).expand(batch_size, -1)

    print(f"Random order (first batch): {random_order[0].tolist()}")
    logits_random, loss_random = model(idx, random_order, targets)
    print(f"Loss with random order: {loss_random.item():.4f}")

    # Test with some positions ignored (use -1)
    print("\n" + "=" * 80)
    print("Testing with ignored positions (target=-1)...")
    targets_with_ignore = targets.clone()
    targets_with_ignore[:, :3] = -1  # Ignore first 3 positions
    logits_ignore, loss_ignore = model(idx, order, targets_with_ignore)
    print(f"Loss with ignored positions: {loss_ignore.item():.4f}")

    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)
