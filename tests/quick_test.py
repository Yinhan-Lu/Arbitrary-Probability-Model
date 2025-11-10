"""
Quick test to verify the GPT-2 implementation works
Tests model instantiation and forward pass

Run from project root: python tests/quick_test.py
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

# Test conditional probability mode (NEW)
print("\n5. Testing Conditional Probability Mode (NEW INTERFACE):")
print("-" * 70)

# Create model with special tokens for conditional mode
mask_token_id = 50257
bos_token_id = 50256
model_cond = GPT2Model(config, mask_token_id=mask_token_id, bos_token_id=bos_token_id)
print(f"✓ Model created with mask_token_id={mask_token_id}, bos_token_id={bos_token_id}")

# Test conditional forward pass
batch_size_test = 2
seq_len_test = 10
input_ids_test = torch.randint(0, config.vocab_size, (batch_size_test, seq_len_test))

# Define conditioning and evaluation indices
conditional_idx = [
    [1, 3, 5],      # Sample 0: condition on positions 1, 3, 5
    [0, 2, 4, 6]    # Sample 1: condition on positions 0, 2, 4, 6
]
evaluation_idx = [
    [2, 4],         # Sample 0: evaluate positions 2, 4
    [1, 3, 5]       # Sample 1: evaluate positions 1, 3, 5
]
unseen_idx = [
    [2, 4, 6, 7],   # Sample 0: unseen positions (includes eval)
    [1, 3, 5, 7, 8] # Sample 1: unseen positions (includes eval)
]

print(f"Input shape: {input_ids_test.shape}")
print(f"Conditional indices sample 0: {conditional_idx[0]}")
print(f"Evaluation indices sample 0: {evaluation_idx[0]}")

# Forward pass with conditional mode
logits_cond, loss_cond = model_cond(
    input_ids=input_ids_test,
    conditional_idx=conditional_idx,
    evaluation_idx=evaluation_idx,
    unseen_idx=unseen_idx
)

print(f"✓ Conditional forward pass successful")
print(f"  Output logits shape: {logits_cond.shape}")
print(f"  Loss (on evaluation positions only): {loss_cond.item():.4f}")
assert loss_cond.item() > 0, "Loss should be positive"

# Test standard mode still works (backward compatibility)
print("\n6. Testing Backward Compatibility (Standard Mode):")
print("-" * 70)
logits_std, _ = model_cond(input_ids=input_ids_test)
print(f"✓ Standard mode forward pass: {logits_std.shape}")
print(f"  Model can still be used for standard LM")

# Test with labels in standard mode
labels_std = input_ids_test.clone()
logits_std, loss_std = model_cond(input_ids=input_ids_test, labels=labels_std)
print(f"✓ Standard mode with labels: loss={loss_std.item():.4f}")

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED (Including New Conditional Interface)!")
print("=" * 70)
print("\nYour GPT-2 implementation is working correctly.")
print("\nNew features:")
print("  • Conditional probability mode: P(X_e | X_c)")
print("  • Backward compatible: Standard LM mode still works")
print("\nNext steps:")
print("  1. Run full sanity check: ./scripts/sanity_run.sh")
print("  2. Try conditional training: python train_conditional.py --config nano")
print()
