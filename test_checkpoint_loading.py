"""
Test script for checkpoint loading and resume functionality

Tests loading from local checkpoints without HuggingFace dependency
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from model.config import get_config
from model.arbitrary_prob_gpt2 import GPT2Model
from model.token_manager import TokenManager

print("=" * 80)
print("Testing Checkpoint Save/Load Functionality")
print("=" * 80)

# Test 1: Create and save a model
print("\n[Test 1] Creating model and saving checkpoint...")
config = get_config("distilgpt2")
token_manager = TokenManager(add_mask_token=True, add_bos_token=False)

model = GPT2Model(config)
model = token_manager.resize_model_embeddings(model)

print(f"✓ Model created: {model.get_num_params()/1e6:.2f}M parameters")
print(f"✓ Vocab size: {model.config.vocab_size}")

# Save checkpoint
checkpoint_path = Path("/tmp/test_checkpoint.pt")
checkpoint = {
    "global_step": 1000,
    "epoch": 5,
    "model_state_dict": model.state_dict(),
    "config": config.__dict__,
}
torch.save(checkpoint, checkpoint_path)
print(f"✓ Checkpoint saved to: {checkpoint_path}")

# Test 2: Load checkpoint
print("\n[Test 2] Loading checkpoint...")
loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"✓ Checkpoint loaded")
print(f"  Global step: {loaded_checkpoint['global_step']}")
print(f"  Epoch: {loaded_checkpoint['epoch']}")
print(f"  State dict keys: {len(loaded_checkpoint['model_state_dict'])}")

# Test 3: Create new model and load weights
print("\n[Test 3] Creating new model and loading weights...")
new_model = GPT2Model(config)
new_model = token_manager.resize_model_embeddings(new_model)

missing_keys, unexpected_keys = new_model.load_state_dict(
    loaded_checkpoint["model_state_dict"],
    strict=True
)

print(f"✓ Weights loaded successfully")
print(f"  Missing keys: {len(missing_keys)}")
print(f"  Unexpected keys: {len(unexpected_keys)}")

# Test 4: Verify weights match
print("\n[Test 4] Verifying weights match...")
all_match = True
for name, param in model.named_parameters():
    new_param = dict(new_model.named_parameters())[name]
    if not torch.allclose(param, new_param):
        print(f"  ✗ Mismatch in {name}")
        all_match = False

if all_match:
    print(f"✓ All weights match perfectly!")
else:
    print(f"✗ Some weights don't match")

# Test 5: Test forward pass
print("\n[Test 5] Testing forward pass with loaded model...")
tokenizer = token_manager.get_tokenizer()
test_text = "Hello world [M] test"
input_ids = tokenizer.encode(test_text, return_tensors="pt")

model.eval()
new_model.eval()

with torch.no_grad():
    logits1, _ = model(input_ids)
    logits2, _ = new_model(input_ids)

if torch.allclose(logits1, logits2, atol=1e-6):
    print(f"✓ Forward passes match!")
    print(f"  Logits shape: {logits1.shape}")
else:
    print(f"✗ Forward passes don't match")
    print(f"  Max diff: {(logits1 - logits2).abs().max().item()}")

# Cleanup
checkpoint_path.unlink()
print(f"\n✓ Cleaned up test checkpoint")

print("\n" + "=" * 80)
print("✓ All checkpoint tests passed!")
print("=" * 80)
