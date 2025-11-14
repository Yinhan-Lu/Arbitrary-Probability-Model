import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistilBertConfig:
    def __init__(self,vocab_size=30522,dim=768,n_layers=6,n_heads=12,hidden_dim=3072,
                max_position_embeddings=1024,dropout=0.1,attention_dropout=0.1,layer_norm_eps=1e-5):
        assert dim % n_heads == 0
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.max_seq_len = max_position_embeddings


class DistilMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention layer used in DistilBERT.

    - Computes Q, K, V in one linear layer
    - Applies scaled dot-product attention
    - Projects output back to hidden dimension
    """
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim//config.n_heads

        # Single linear layer producing concatenated Q,K, V
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.out_proj = nn.Linear(config.dim, config.dim)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        B, T, D = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        ctx = attn_probs @ v
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)

        out = self.out_proj(ctx)
        out = self.resid_dropout(out)
        return out


class DistilFeedForward(nn.Module):
    """
    Standard Transformer feed-forward network:

        FFN(x) = Linear -> GELU -> Linear-> Dropout
    """
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.lin1 = nn.Linear(config.dim, config.hidden_dim)
        self.lin2 = nn.Linear(config.hidden_dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class DistilTransformerBlock(nn.Module):
    """
    DistilBERT block = LayerNorm -> Self-Attention -> Residual+ LayerNorm ->FFN -> Residual
    """
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.attn = DistilMultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.ffn = DistilFeedForward(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DistilBertForMaskedLM(nn.Module):
    """
    Encoder-only Transformer trained with Masked Language Modeling.
    """
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.dim
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [DistilTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_final = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)

        self.mlm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.mlm_head.weight = self.word_embeddings.weight  

        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def build_attention_mask(self, attention_mask):
        if attention_mask is None:
            return None
        mask = (1.0 - attention_mask.float()) * -1e4  
        return mask.unsqueeze(1).unsqueeze(1)    

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.size()
        device = input_ids.device

        pos_ids = torch.arange(0, T, device=device).unsqueeze(0) 

        x = self.word_embeddings(input_ids) + self.position_embeddings(pos_ids)
        x = self.dropout(x)

        attn_mask = self.build_attention_mask(attention_mask)

        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask)

        x = self.ln_final(x)
        logits = self.mlm_head(x)  

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss
