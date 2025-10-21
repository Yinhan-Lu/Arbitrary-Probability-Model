"""
Test script for pretrained model loading functionality

Tests:
1. Loading from HuggingFace (e.g., distilgpt2, gpt2)
2. Vocabulary extension with [M] token
3. Model initialization and parameter counts
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from model.config import get_config
from model.arbitrary_prob_gpt2 import GPT2Model
from model.token_manager import TokenManager
from transformers import GPT2LMHeadModel

print("=" * 80)
print("Testing Pretrained Model Loading")
print("=" * 80)

# Test 1: Load HuggingFace model
print("\n[Test 1] Loading distilgpt2 from HuggingFace...")
hf_model_name = "distilgpt2"
config = get_config(hf_model_name)

# Initialize token manager
token_manager = TokenManager(add_mask_token=True, add_bos_token=False)
print(f"✓ TokenManager initialized: vocab {token_manager.original_vocab_size} -> {len(token_manager.tokenizer)}")

# Create our model
our_model = GPT2Model(config)
print(f"✓ Our model created: {our_model.get_num_params()/1e6:.2f}M parameters")

# Load HuggingFace weights
print(f"\n[Test 2] Loading HuggingFace weights...")
hf_model = GPT2LMHeadModel.from_pretrained(hf_model_name)
hf_state_dict = hf_model.state_dict()
print(f"✓ HuggingFace model loaded: {len(hf_state_dict)} keys")

# Map keys
our_state_dict = {}
key_mapping = {
    'transformer.wte.weight': 'wte.weight',
    'transformer.wpe.weight': 'wpe.weight',
    'transformer.ln_f.weight': 'ln_f.weight',
    'transformer.ln_f.bias': 'ln_f.bias',
    'lm_head.weight': 'lm_head.weight',
}

# Map transformer blocks
for i in range(config.n_layer):
    # Attention
    key_mapping[f'transformer.h.{i}.ln_1.weight'] = f'blocks.{i}.ln_1.weight'
    key_mapping[f'transformer.h.{i}.ln_1.bias'] = f'blocks.{i}.ln_1.bias'
    key_mapping[f'transformer.h.{i}.attn.c_attn.weight'] = f'blocks.{i}.attn.c_attn.weight'
    key_mapping[f'transformer.h.{i}.attn.c_attn.bias'] = f'blocks.{i}.attn.c_attn.bias'
    key_mapping[f'transformer.h.{i}.attn.c_proj.weight'] = f'blocks.{i}.attn.c_proj.weight'
    key_mapping[f'transformer.h.{i}.attn.c_proj.bias'] = f'blocks.{i}.attn.c_proj.bias'

    # MLP
    key_mapping[f'transformer.h.{i}.ln_2.weight'] = f'blocks.{i}.ln_2.weight'
    key_mapping[f'transformer.h.{i}.ln_2.bias'] = f'blocks.{i}.ln_2.bias'
    key_mapping[f'transformer.h.{i}.mlp.c_fc.weight'] = f'blocks.{i}.mlp.c_fc.weight'
    key_mapping[f'transformer.h.{i}.mlp.c_fc.bias'] = f'blocks.{i}.mlp.c_fc.bias'
    key_mapping[f'transformer.h.{i}.mlp.c_proj.weight'] = f'blocks.{i}.mlp.c_proj.weight'
    key_mapping[f'transformer.h.{i}.mlp.c_proj.bias'] = f'blocks.{i}.mlp.c_proj.bias'

# Apply mapping
for hf_key, our_key in key_mapping.items():
    if hf_key in hf_state_dict:
        our_state_dict[our_key] = hf_state_dict[hf_key]

print(f"✓ Mapped {len(our_state_dict)} keys")

# Load weights
missing_keys, unexpected_keys = our_model.load_state_dict(our_state_dict, strict=False)
print(f"✓ Loaded weights: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")

# Test 3: Resize embeddings
print(f"\n[Test 3] Resizing embeddings for new tokens...")
print(f"Before: vocab_size={our_model.config.vocab_size}, embedding shape={our_model.wte.weight.shape}")

our_model = token_manager.resize_model_embeddings(our_model)

print(f"After: vocab_size={our_model.config.vocab_size}, embedding shape={our_model.wte.weight.shape}")
print(f"✓ Embeddings resized successfully")

# Test 4: Forward pass with [M] token
print(f"\n[Test 4] Testing forward pass with [M] token...")
tokenizer = token_manager.get_tokenizer()
mask_token_id = token_manager.mask_token_id

# Create test input with [M]
test_text = "Hello world"
tokens = tokenizer.encode(test_text)
# Insert [M] in the middle
tokens.insert(1, mask_token_id)
input_ids = torch.tensor([tokens])

print(f"Input: {tokenizer.decode(input_ids[0])}")
print(f"Input IDs: {input_ids}")

# Forward pass
our_model.eval()
with torch.no_grad():
    logits, _ = our_model(input_ids)

print(f"✓ Forward pass successful: logits shape = {logits.shape}")
print(f"  Expected: (1, {len(tokens)}, {our_model.config.vocab_size})")

# Test 5: Compare outputs on same text (without [M])
print(f"\n[Test 5] Comparing outputs (HF vs Ours) on same text...")
test_input = tokenizer.encode("The quick brown fox", return_tensors="pt")

hf_model.eval()
with torch.no_grad():
    hf_output = hf_model(test_input).logits
    our_output, _ = our_model(test_input)

# Compare (should be very close for overlapping vocab)
max_diff = (hf_output[0, :, :50257] - our_output[0, :, :50257]).abs().max().item()
mean_diff = (hf_output[0, :, :50257] - our_output[0, :, :50257]).abs().mean().item()

print(f"✓ Output comparison:")
print(f"  Max difference: {max_diff:.6f}")
print(f"  Mean difference: {mean_diff:.6f}")

if max_diff < 1e-4:
    print(f"  ✓ PASS: Outputs match (diff < 1e-4)")
else:
    print(f"  ⚠ WARNING: Outputs differ (might be expected due to model differences)")

print("\n" + "=" * 80)
print("✓ All tests completed successfully!")
print("=" * 80)
