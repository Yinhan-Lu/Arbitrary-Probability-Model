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

    def __init__(self, config, mask_token_id=None, bos_token_id=None):
        super().__init__()
        self.config = config

        # For conditional probability modeling
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id

        # Auto-expand vocab_size if special tokens exceed current size
        effective_vocab_size = config.vocab_size
        if mask_token_id is not None and mask_token_id >= effective_vocab_size:
            effective_vocab_size = mask_token_id + 1
        if bos_token_id is not None and bos_token_id >= effective_vocab_size:
            effective_vocab_size = bos_token_id + 1

        # Store effective vocab size for later use
        self.effective_vocab_size = effective_vocab_size

        # Update config to match actual vocab size (important for token_manager compatibility)
        self.config.vocab_size = effective_vocab_size

        # Token + Position embeddings (use effective vocab_size)
        self.wte = nn.Embedding(effective_vocab_size, config.n_embd)  # token embeddings
        self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)  # position embeddings
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)

        # Language model head (tied with token embeddings in original GPT-2)
        # Use effective_vocab_size to match wte
        self.lm_head = nn.Linear(config.n_embd, effective_vocab_size, bias=False)

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

    def _build_augmented_sequence_single(
        self,
        input_ids,       # (L,) single sequence
        cond_idx,        # List[int]
        eval_idx,        # List[int]
        unseen_idx,      # List[int]
        device='cpu'
    ):
        """
        Build augmented sequence for a single sample using prefix conditioning

        Args:
            input_ids: (L,) original sequence
            cond_idx: List of conditioning position indices
            eval_idx: List of evaluation position indices
            unseen_idx: List of unseen position indices (includes eval_idx)
            device: Device to create tensors on

        Returns:
            aug_input_ids: (aug_len,) augmented sequence [cond_tokens] + [BOS + body]
            position_ids: (aug_len,) custom position encodings
            N_cond: Number of conditioning tokens
            N_seq: Number of sequence tokens (BOS + body)
        """
        seq_len = input_ids.size(0)

        # 1. Build conditioning prefix
        cond_tokens = [input_ids[idx].item() for idx in sorted(cond_idx)]
        cond_position_ids = [idx + 1 for idx in sorted(cond_idx)]  # Body uses 1-seq_len
        N_cond = len(cond_tokens)

        # 2. Build sequence (BOS + body tokens)
        seq_tokens = [self.bos_token_id]
        seq_position_ids = [0]  # BOS uses position 0

        # Only mask truly unseen tokens (unseen - eval)
        # Keep eval tokens visible so model can use previous eval tokens
        truly_unseen = set(unseen_idx) - set(eval_idx)

        for i in range(seq_len):
            if i in truly_unseen:
                seq_tokens.append(self.mask_token_id)
            else:
                seq_tokens.append(input_ids[i].item())
            seq_position_ids.append(i + 1)  # Body uses positions 1 to seq_len

        N_seq = len(seq_tokens)

        # 3. Concatenate [Cond] + [Seq]
        aug_input_ids = torch.tensor(
            cond_tokens + seq_tokens,
            dtype=torch.long,
            device=device
        )
        position_ids = torch.tensor(
            cond_position_ids + seq_position_ids,
            dtype=torch.long,
            device=device
        )

        return aug_input_ids, position_ids, N_cond, N_seq

    def _create_prefix_conditional_mask(self, N_cond, N_seq, device='cpu'):
        """
        Create prefix conditional attention mask

        Structure:
        - Conditioning rows: Can attend to all positions (fully visible)
        - Sequence rows: Can attend to all conditioning + causal sequence

        Args:
            N_cond: Number of conditioning tokens
            N_seq: Number of sequence tokens (BOS + body)
            device: Device to create mask on

        Returns:
            mask: (N_cond + N_seq, N_cond + N_seq)
                  1 = can attend, 0 = cannot attend
        """
        # Conditioning rows: Fully visible (can attend to all)
        cond_rows = torch.ones(N_cond, N_cond + N_seq, device=device, dtype=torch.uint8)

        # Sequence rows: Conditioning visible + causal sequence
        cond_visible = torch.ones(N_seq, N_cond, device=device, dtype=torch.uint8)
        causal_mask = torch.tril(torch.ones(N_seq, N_seq, device=device, dtype=torch.uint8))
        seq_rows = torch.cat([cond_visible, causal_mask], dim=1)

        # Concatenate to form full mask
        full_mask = torch.cat([cond_rows, seq_rows], dim=0)

        return full_mask

    def _create_labels(
        self,
        input_ids,       # (L,) original sequence
        eval_idx,        # List[int]
        aug_input_ids,   # (aug_len,) augmented sequence
        N_cond           # int
    ):
        """
        Create labels for loss computation (only evaluation positions)

        Args:
            input_ids: (L,) original sequence
            eval_idx: List of evaluation position indices
            aug_input_ids: (aug_len,) augmented sequence
            N_cond: Number of conditioning tokens in prefix

        Returns:
            labels: (aug_len,) with -100 for non-eval positions
        """
        labels = torch.full_like(aug_input_ids, -100)

        # Set labels at evaluation positions
        for eval_pos in eval_idx:
            # eval_pos is original sequence index (0-based)
            # In augmented sequence: N_cond (prefix) + 1 (BOS) + eval_pos (body)
            new_pos = N_cond + 1 + eval_pos
            if new_pos < len(labels):
                labels[new_pos] = input_ids[eval_pos].item()

        return labels

    def _augment_batch(
        self,
        input_ids,        # (B, L)
        conditional_idx,  # List[List[int]]
        evaluation_idx,   # List[List[int]]
        unseen_idx,       # List[List[int]]
        pad_token_id=50256,  # Tokenizer's pad token ID
        device='cpu'
    ):
        """
        Augment entire batch with dynamic padding

        Args:
            input_ids: (B, L) batch of original sequences
            conditional_idx: List of conditioning index lists for each sample
            evaluation_idx: List of evaluation index lists for each sample
            unseen_idx: List of unseen index lists for each sample
            pad_token_id: Token ID for padding (default: 50256 for GPT-2)
            device: Device to create tensors on

        Returns:
            aug_input_ids: (B, max_aug_len) augmented and padded batch
            position_ids: (B, max_aug_len) position encodings
            attention_mask: (B, 1, max_aug_len, max_aug_len) 2D attention masks
            labels: (B, max_aug_len) labels for loss computation
        """
        batch_size = input_ids.size(0)

        # Step 1: Build augmented sequence for each sample
        aug_samples = []
        for i in range(batch_size):
            aug_ids, pos_ids, N_cond, N_seq = self._build_augmented_sequence_single(
                input_ids[i],
                conditional_idx[i],
                evaluation_idx[i],
                unseen_idx[i],
                device=device
            )

            # Create attention mask
            attn_mask = self._create_prefix_conditional_mask(N_cond, N_seq, device=device)

            # Create labels
            labels = self._create_labels(
                input_ids[i],
                evaluation_idx[i],
                aug_ids,
                N_cond
            )

            aug_samples.append({
                'input_ids': aug_ids,
                'position_ids': pos_ids,
                'attention_mask': attn_mask,
                'labels': labels
            })

        # Step 2: Find max length in batch
        max_len = max(s['input_ids'].size(0) for s in aug_samples)

        # Step 3: Pad to max_len
        padded_input_ids = []
        padded_position_ids = []
        padded_attention_masks = []
        padded_labels = []

        for sample in aug_samples:
            current_len = sample['input_ids'].size(0)
            pad_len = max_len - current_len

            if pad_len > 0:
                # Pad input_ids
                padded_ids = torch.cat([
                    sample['input_ids'],
                    torch.full((pad_len,), pad_token_id, dtype=torch.long, device=device)
                ])

                # Pad position_ids
                padded_pos = torch.cat([
                    sample['position_ids'],
                    torch.zeros(pad_len, dtype=torch.long, device=device)
                ])

                # Pad labels
                padded_lab = torch.cat([
                    sample['labels'],
                    torch.full((pad_len,), -100, dtype=torch.long, device=device)
                ])

                # Pad attention mask (2D)
                padded_mask = torch.zeros(max_len, max_len, dtype=torch.uint8, device=device)
                padded_mask[:current_len, :current_len] = sample['attention_mask']
                # Let padded positions attend to valid content (avoid softmax NaN)
                padded_mask[current_len:, :current_len] = 1

            else:
                padded_ids = sample['input_ids']
                padded_pos = sample['position_ids']
                padded_lab = sample['labels']
                padded_mask = sample['attention_mask']

            padded_input_ids.append(padded_ids)
            padded_position_ids.append(padded_pos)
            padded_attention_masks.append(padded_mask)
            padded_labels.append(padded_lab)

        # Step 4: Stack into batch
        batch_input_ids = torch.stack(padded_input_ids, dim=0)
        batch_position_ids = torch.stack(padded_position_ids, dim=0)
        batch_labels = torch.stack(padded_labels, dim=0)
        batch_attention_mask = torch.stack(padded_attention_masks, dim=0).unsqueeze(1)  # (B, 1, L, L)

        return batch_input_ids, batch_position_ids, batch_attention_mask, batch_labels

    def forward(
        self,
        input_ids,
        conditional_idx=None,
        evaluation_idx=None,
        unseen_idx=None,
        attention_mask=None,
        labels=None,
        position_ids=None
    ):
        """
        Forward pass with support for two modes:
        1. Conditional mode: Provide conditional_idx, evaluation_idx, unseen_idx
           - Model internally constructs augmented sequences
           - Creates prefix conditional attention masks
           - Builds custom position IDs and labels
        2. Standard mode: Do not provide indices
           - Standard causal LM forward pass
           - Uses provided attention_mask or default causal mask

        Args:
            input_ids: (B, L) input token indices
            conditional_idx: List[List[int]] or None - conditioning positions for each sample
            evaluation_idx: List[List[int]] or None - evaluation positions for each sample
            unseen_idx: List[List[int]] or None - unseen positions for each sample
            attention_mask: Optional attention mask (standard mode only)
            labels: Optional labels (standard mode only, conditional mode auto-generates)
            position_ids: Optional position IDs (standard mode only, conditional mode auto-generates)

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Optional language modeling loss if labels are provided
        """
        # Determine which mode to use
        if conditional_idx is not None:
            # === CONDITIONAL MODE ===
            assert evaluation_idx is not None, "Must provide evaluation_idx in conditional mode"
            assert unseen_idx is not None, "Must provide unseen_idx in conditional mode"
            assert self.mask_token_id is not None, "mask_token_id required for conditional mode"
            assert self.bos_token_id is not None, "bos_token_id required for conditional mode"

            # Internal augmentation
            aug_input_ids, aug_position_ids, aug_attention_mask, aug_labels = self._augment_batch(
                input_ids=input_ids,
                conditional_idx=conditional_idx,
                evaluation_idx=evaluation_idx,
                unseen_idx=unseen_idx,
                pad_token_id=50256,  # TODO: Should get from config or parameter
                device=input_ids.device
            )

            # Use augmented inputs
            input_ids = aug_input_ids
            position_ids = aug_position_ids
            attention_mask = aug_attention_mask
            labels = aug_labels  # Auto-generated labels

        else:
            # === STANDARD MODE ===
            # Use original inputs, no augmentation
            pass
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
