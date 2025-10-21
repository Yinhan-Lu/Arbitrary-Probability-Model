"""
Quick test to verify the GPT-2 implementation works
Tests model instantiation and forward pass

Run from project root: python utils/quick_test.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.config import get_config, print_config_summary
from model.arbitrary_prob_gpt2 import GPT2Model, create_causal_mask

print("=" * 70)
print("GPT-2 Implementation Quick Test")
print("=" * 70)

# Print available configurations
print("\n1. Available Configurations:")
print("-" * 70)
print_config_summary()

# Test with nano config (fastest)
print("\n2. Testing Model Instantiation (Nano Config):")
print("-" * 70)
config = get_config("nano")
model = GPT2Model(config)
print(f"✓ Model created successfully")
print(f"  Total parameters: {model.get_num_params()/1e6:.2f}M")

# Test forward pass
print("\n3. Testing Forward Pass:")
print("-" * 70)
batch_size = 2
seq_len = 16
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

print(f"Input shape: {input_ids.shape}")

# Forward with default causal mask
logits, _ = model(input_ids)
print(f"✓ Forward pass (default causal mask): {logits.shape}")
assert logits.shape == (batch_size, seq_len, config.vocab_size)

# Forward with custom mask
custom_mask = create_causal_mask(seq_len)
logits, _ = model(input_ids, attention_mask=custom_mask)
print(f"✓ Forward pass (custom mask): {logits.shape}")

# Forward with loss
labels = input_ids.clone()
logits, loss = model(input_ids, labels=labels)
print(f"✓ Loss computation: {loss.item():.4f}")
assert loss.item() > 0

print("\n4. Testing All Configurations:")
print("-" * 70)
for config_name in ["nano", "tiny", "distilgpt2"]:
    cfg = get_config(config_name)
    mdl = GPT2Model(cfg)
    params = mdl.get_num_params()
    print(f"✓ {config_name:12s}: {params/1e6:6.1f}M parameters")

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nYour GPT-2 implementation is working correctly.")
print("\nNext steps:")
print("  1. Run full sanity check: ./scripts/sanity_run.sh")
print("  2. Try training: python scripts/train.py --config tiny --max_epochs 1")
print()
