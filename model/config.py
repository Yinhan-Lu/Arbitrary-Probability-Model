"""
Configuration presets for GPT-2 model
Includes DistilGPT-2 (official parameters from Hugging Face) and Tiny (for CPU testing)
"""

from .arbitrary_prob_gpt2 import GPT2Config


# DistilGPT-2 official configuration from Hugging Face
# Source: https://huggingface.co/distilbert/distilgpt2/raw/main/config.json
# Retrieved: 2024
DISTILGPT2_CONFIG = GPT2Config(
    vocab_size=50257,           # Vocabulary size
    n_layer=6,                  # Number of transformer layers (distilled from 12)
    n_head=12,                  # Number of attention heads
    n_embd=768,                 # Embedding dimension
    max_seq_len=1024,           # Maximum sequence length (positions 0-1023)
    dropout=0.1,                # Dropout rate (embd_pdrop, attn_pdrop, resid_pdrop)
    layer_norm_eps=1e-5,        # Layer normalization epsilon
    ffn_mult=4,                 # FFN hidden size = n_embd * 4 = 3072
    activation_function="gelu_new"  # Activation function
)


# Tiny configuration for CPU testing and quick validation
# Significantly reduced parameters for fast iteration on CPU
TINY_CONFIG = GPT2Config(
    vocab_size=50257,           # Keep same vocab to use GPT-2 tokenizer
    n_layer=2,                  # Only 2 layers for fast training
    n_head=4,                   # Reduced attention heads
    n_embd=128,                 # Much smaller embedding dimension
    max_seq_len=256,            # Shorter sequences
    dropout=0.1,                # Same dropout rate
    layer_norm_eps=1e-5,        # Same layer norm epsilon
    ffn_mult=4,                 # FFN hidden size = 128 * 4 = 512
    activation_function="gelu_new"
)


# Small configuration for GPU training (10 min sessions)
# Balanced size for quick GPU experiments with meaningful training
SMALL_CONFIG = GPT2Config(
    vocab_size=50257,           # Same vocab as GPT-2
    n_layer=4,                  # 4 layers (between tiny and distilgpt2)
    n_head=8,                   # 8 attention heads
    n_embd=512,                 # Medium embedding dimension
    max_seq_len=512,            # Medium sequence length
    dropout=0.1,                # Standard dropout rate
    layer_norm_eps=1e-5,        # Standard layer norm epsilon
    ffn_mult=4,                 # FFN hidden size = 512 * 4 = 2048
    activation_function="gelu_new"
)


# Nano configuration - even smaller for debugging
NANO_CONFIG = GPT2Config(
    vocab_size=50257,
    n_layer=1,                  # Just 1 layer
    n_head=2,                   # Minimal heads
    n_embd=64,                  # Tiny embedding
    max_seq_len=128,            # Very short sequences
    dropout=0.0,                # No dropout for faster convergence
    layer_norm_eps=1e-5,
    ffn_mult=2,                 # Smaller FFN
    activation_function="gelu_new"
)


# Configuration registry
CONFIGS = {
    "distilgpt2": DISTILGPT2_CONFIG,
    "small": SMALL_CONFIG,
    "tiny": TINY_CONFIG,
    "nano": NANO_CONFIG
}


def get_config(config_name):
    """
    Get a configuration by name

    Args:
        config_name: One of "distilgpt2", "small", "tiny", "nano"

    Returns:
        GPT2Config instance
    """
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]


def print_config_summary():
    """Print a summary of all available configurations"""
    print("Available GPT-2 Configurations:")
    print("=" * 80)

    for name, config in CONFIGS.items():
        param_count = (
            config.vocab_size * config.n_embd +  # Token embeddings
            config.max_seq_len * config.n_embd +  # Position embeddings
            config.n_layer * (
                3 * config.n_embd * config.n_embd +  # QKV projection
                config.n_embd * config.n_embd +  # Output projection
                config.n_embd * config.mlp_hidden_size +  # FFN up
                config.mlp_hidden_size * config.n_embd +  # FFN down
                4 * config.n_embd  # Layer norms (2 per block, gamma and beta)
            ) +
            config.n_embd * 2  # Final layer norm
        )

        print(f"\n{name.upper()}:")
        print(f"  Layers: {config.n_layer}")
        print(f"  Heads: {config.n_head}")
        print(f"  Embedding dim: {config.n_embd}")
        print(f"  FFN hidden: {config.mlp_hidden_size}")
        print(f"  Max seq len: {config.max_seq_len}")
        print(f"  Vocab size: {config.vocab_size}")
        print(f"  Dropout: {config.dropout}")
        print(f"  Est. parameters: ~{param_count/1e6:.1f}M")


if __name__ == "__main__":
    print_config_summary()

    print("\n" + "=" * 80)
    print("\nDistilGPT-2 Official Parameters (from Hugging Face):")
    print("-" * 80)

    config = get_config("distilgpt2")
    for attr in dir(config):
        if not attr.startswith("_"):
            print(f"  {attr}: {getattr(config, attr)}")
