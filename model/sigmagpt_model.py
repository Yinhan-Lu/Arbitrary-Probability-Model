"""
Sigma GPT Model - Adapted from sigma-gpt/text/sigmagpt.py

This implementation is simplified for teacher forcing training only:
- Removed KV cache support (only needed for generation)
- Removed generation methods (generate, sample_and_evaluate, etc.)
- Removed burst mode
- Adapted to work with GPT2Config (compatible with existing codebase)

Key features:
- Double position encoding (n_embd // 2 per position)
- Order tensor support for arbitrary generation order
- Ignore index = -1 (Sigma GPT convention)

Reference: https://arxiv.org/abs/2404.09562
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPT2Config


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (GPT-2 style)"""

    def __init__(self, ndim, bias=True, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention
    Simplified from sigma-gpt (removed KV cache and burst mode)
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)),
            persistent=False
        )

    def forward(self, x):
        """
        Forward pass (simplified - no KV cache, no burst mode)

        Args:
            x: (B, T, C) input tensor

        Returns:
            (B, T, C) output tensor
        """
        B, T, C = x.size()

        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Attention: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v  # (B, n_head, T, head_dim)

        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network (GPT-2 style)"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block (simplified - no KV cache)"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=True)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SigmaGPT(nn.Module):
    """
    Sigma GPT model with double position encoding

    Key difference from standard GPT-2:
    - Position embedding is n_embd // 2 (not n_embd)
    - Uses double position encoding: current position + next position
    - Forward requires order tensor to specify generation order

    Reference implementation: sigma-gpt/text/sigmagpt.py (line 299-389)
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_seq_len is not None
        self.config = config

        self.transformer = nn.ModuleDict({
            # Token embedding (standard)
            "wte": nn.Embedding(config.vocab_size, config.n_embd),

            # Position embedding (CRITICAL: n_embd // 2, not n_embd)
            # This is the key architectural change in Sigma GPT
            "wpe": nn.Embedding(config.max_seq_len, config.n_embd // 2),

            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": LayerNorm(config.n_embd, bias=True),
        })

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (GPT-2 standard)
        self.transformer["wte"].weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Special scaled init for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"SigmaGPT initialized with {self.get_num_params() / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def _pos_emb(self, idx, order):
        """
        CRITICAL: Double position encoding implementation

        This is the core innovation of Sigma GPT. Instead of encoding just
        the current position, we encode both:
        - Current position (where we are)
        - Next position (where we're going)

        Each gets n_embd/2 dimensions, concatenated to n_embd total.

        Reference: sigma-gpt/text/sigmagpt.py line 346-363

        Args:
            idx: (B, T) token indices
            order: (B, T+1) order tensor
                  - order[:, :-1] are current positions
                  - order[:, 1:] are next positions

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

        # Embed: (B, T, 2) -> (B, T, 2, n_embd/2)
        pos_emb = self.transformer["wpe"](order_cat)

        # Flatten to (B, T, n_embd)
        return pos_emb.flatten(2)

    def forward(self, idx, order, targets=None):
        """
        Forward pass

        Args:
            idx: (B, T) token indices (already shuffled according to order)
            order: (B, T+1) order tensor specifying generation sequence
            targets: (B, T) target tokens for loss computation (optional)
                    Use -1 for positions to ignore in loss

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar loss (or None if targets not provided)
        """
        # Token embeddings (standard)
        tok_emb = self.transformer["wte"](idx)  # (B, T, n_embd)

        # Position embeddings (double encoding - Sigma GPT's key innovation)
        pos_emb = self._pos_emb(idx, order)     # (B, T, n_embd)

        # Combine and apply dropout
        x = self.transformer["drop"](tok_emb + pos_emb)

        # Transformer blocks (standard)
        for block in self.transformer["h"]:
            x = block(x)

        # Final layer norm
        x = self.transformer["ln_f"](x)

        # Language model head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        if targets is None:
            return logits, None

        # CRITICAL: Use ignore_index=-1 (Sigma GPT convention, not PyTorch default -100)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer with weight decay
        (Standard GPT-2 optimization setup)
        """
        # Separate parameters that should and shouldn't have weight decay
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Validate
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters in both decay and no_decay: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Parameters not in either set: {param_dict.keys() - union_params}"

        # Create optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        # Use fused AdamW if available (faster on CUDA)
        use_fused = (device_type == 'cuda')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)

        return optimizer
