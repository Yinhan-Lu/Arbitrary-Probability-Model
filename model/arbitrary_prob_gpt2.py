"""
GPT-2 PyTorch Implementation from Scratch
Decoder-only Transformer with configurable attention masks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2Config:
    """Configuration class for GPT-2 model parameters"""

    def __init__(
        self,
        vocab_size=50257,
        n_layer=6,
        n_head=12,
        n_embd=768,
        max_seq_len=1024,
        dropout=0.1,
        layer_norm_eps=1e-5,
        ffn_mult=4,
        activation_function="gelu_new",
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.ffn_mult = ffn_mult
        self.mlp_hidden_size = n_embd * ffn_mult
        self.activation_function = activation_function


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism with support for custom attention masks"""

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

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                          or (batch_size, 1, seq_len, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()  # batch_size, seq_len, n_embd

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Compute attention scores: (B, n_head, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (default left-to-right)
        if attention_mask is None:
            # Use default causal mask
            attn_scores = attn_scores.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float('-inf')
            )
        else:
            # Use custom attention mask
            # Ensure mask has correct shape: (B, 1, T, T) or (B, n_head, T, T)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # (B, 1, T, T)

            # Apply mask: 0 means mask out (attend to), 1 means keep
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

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

    def forward(self, x, attention_mask=None):
        # Pre-LayerNorm architecture (GPT-2 style)
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    """
    GPT-2 Language Model (Decoder-only Transformer)

    Architecture: Embedding → N × TransformerBlock → LayerNorm → LM Head
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token + Position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embeddings
        self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)  # position embeddings
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)

        # Language model head (tied with token embeddings in original GPT-2)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight initialization
        self.apply(self._init_weights)

        # Tie weights between token embeddings and lm_head
        self.lm_head.weight = self.wte.weight

        print(f"GPT-2 Model initialized with {self.get_num_params()/1e6:.2f}M parameters")

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

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None):
        """
        Args:
            input_ids: Input token indices of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            labels: Optional labels for language modeling loss of shape (batch_size, seq_len)
            position_ids: Optional custom position indices of shape (batch_size, seq_len) or (seq_len,)
                         If None, uses sequential positions 0, 1, 2, ..., seq_len-1

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Optional language modeling loss if labels are provided
        """
        B, T = input_ids.size()

        # Get token and position embeddings
        if position_ids is None:
            # Default: sequential positions 0, 1, 2, ..., T-1
            # In this case, check total sequence length
            assert T <= self.config.max_seq_len, \
                f"Sequence length {T} exceeds maximum {self.config.max_seq_len}"
            pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)  # (1, T)
        else:
            # Custom positions for prefix conditioning
            # In this case, check that position IDs are within range
            # (total sequence length can exceed max_seq_len)
            max_pos = position_ids.max().item()
            assert max_pos < self.config.max_seq_len, \
                f"Position ID {max_pos} exceeds maximum {self.config.max_seq_len - 1}"
            pos = position_ids
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)  # (T,) -> (1, T)

        tok_emb = self.wte(input_ids)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (1, T, n_embd) or (B, T, n_embd)

        x = self.drop(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        # Final layer norm
        x = self.ln_f(x)

        # Language model head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Compute language modeling loss (cross-entropy)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
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
    # Test the model with random input
    print("Testing GPT-2 Model Implementation...")

    # Create a tiny config for testing
    config = GPT2Config(
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=256,
        dropout=0.1
    )

    model = GPT2Model(config)

    # Random input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")

    # Forward pass with default causal mask
    logits, _ = model(input_ids)
    print(f"Output logits shape: {logits.shape}")

    # Forward pass with custom attention mask
    custom_mask = create_causal_mask(seq_len)
    logits_custom, _ = model(input_ids, attention_mask=custom_mask)
    print(f"Output logits shape (custom mask): {logits_custom.shape}")

    # Forward pass with labels to compute loss
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(input_ids, labels=labels)
    print(f"Loss: {loss.item():.4f}")

    print("\nModel test passed successfully!")
