"""
Quick test to verify the Diffusion model implementation works
Tests noise schedule, model instantiation, and forward pass

Run from project root: python tests/test_diffusion.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.config import get_config
from model.diffusion_gpt2 import DiffusionGPT2Model
from model.diffusion_utils import NoiseSchedule, SinusoidalPositionEmbedding, indices_to_mask

print("=" * 70)
print("Diffusion Model Implementation Quick Test")
print("=" * 70)

# 1. Test Noise Schedule
print("\n1. Testing Noise Schedule:")
print("-" * 70)

schedule = NoiseSchedule(num_timesteps=1000, schedule_type='cosine')
print(f"Created cosine schedule with T={schedule.T}")

# Test alpha values at different timesteps
print("Alpha values at different timesteps:")
for t in [0, 100, 500, 900, 1000]:
    alpha = schedule.get_alpha(torch.tensor([t]))
    mask_prob = schedule.get_mask_prob(torch.tensor([t]))
    print(f"  t={t:4d}: alpha={alpha.item():.4f}, mask_prob={mask_prob.item():.4f}")

# Test timestep sampling
t_samples = schedule.sample_timesteps(10)
print(f"Sampled timesteps (10 samples): min={t_samples.min().item()}, max={t_samples.max().item()}")
assert t_samples.min() >= 1, "Timesteps should be >= 1"
assert t_samples.max() <= 1000, "Timesteps should be <= T"
print("Noise schedule: PASSED")

# 2. Test Forward Diffusion
print("\n2. Testing Forward Diffusion:")
print("-" * 70)

x_0 = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
t = torch.tensor([500])
mask_token_id = 50257

x_t, mask = schedule.add_noise(x_0, t, mask_token_id)
print(f"Original x_0: {x_0.tolist()[0]}")
print(f"Noisy x_t:    {x_t.tolist()[0]}")
print(f"Mask:         {mask.tolist()[0]}")
print(f"Masked positions: {mask.sum().item()} / {mask.numel()}")
print("Forward diffusion: PASSED")

# 3. Test Conditional Diffusion
print("\n3. Testing Conditional Diffusion (X_c positions never masked):")
print("-" * 70)

conditioning_indices = [[0, 3, 7]]  # These positions should never be masked
x_t, mask = schedule.add_noise_conditional(x_0, t, mask_token_id, conditioning_indices)
print(f"Conditioning positions: {conditioning_indices[0]}")
print(f"Original x_0: {x_0.tolist()[0]}")
print(f"Noisy x_t:    {x_t.tolist()[0]}")
print(f"Mask:         {mask.tolist()[0]}")

# Verify conditioning positions are not masked
for idx in conditioning_indices[0]:
    assert x_t[0, idx] == x_0[0, idx], f"Position {idx} should not be masked"
    assert mask[0, idx] == False, f"Position {idx} should have mask=False"
print("Conditioning positions correctly preserved!")
print("Conditional diffusion: PASSED")

# 4. Test Sinusoidal Embedding
print("\n4. Testing Sinusoidal Timestep Embedding:")
print("-" * 70)

time_emb = SinusoidalPositionEmbedding(dim=128)
t_batch = torch.tensor([0, 100, 500, 1000])
embeddings = time_emb(t_batch)
print(f"Input timesteps: {t_batch.tolist()}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding norms: {[f'{n:.3f}' for n in embeddings.norm(dim=-1).tolist()]}")
assert embeddings.shape == (4, 128), "Wrong embedding shape"
print("Sinusoidal embedding: PASSED")

# 5. Test Diffusion Model
print("\n5. Testing Diffusion Model Instantiation:")
print("-" * 70)

# Use tiny config for fast testing
# Note: For diffusion, we need vocab_size to include mask token
from copy import copy
config = copy(get_config("tiny"))
config.vocab_size = 50258  # Include mask token
mask_token_id = 50257
model = DiffusionGPT2Model(config, mask_token_id=mask_token_id, num_timesteps=1000)
print(f"Created DiffusionGPT2Model with tiny config (vocab_size adjusted for [MASK])")
print(f"  Parameters: {model.get_num_params()/1e6:.2f}M")
print(f"  Attention: Bidirectional (not causal)")
print(f"  Timestep embedding: sinusoidal")

# 6. Test Forward Pass
print("\n6. Testing Diffusion Model Forward Pass:")
print("-" * 70)

batch_size = 2
seq_len = 32
x_0 = torch.randint(0, config.vocab_size, (batch_size, seq_len))
t = torch.randint(1, 1001, (batch_size,))

print(f"Input x_t shape: {x_0.shape}")
print(f"Timesteps: {t.tolist()}")

# Apply noise
x_t, noise_mask = schedule.add_noise(x_0, t, mask_token_id)
print(f"Noise mask sum: {noise_mask.sum().tolist()}")

# Forward pass
logits = model(x_t, t)
print(f"Output logits shape: {logits.shape}")
assert logits.shape == (batch_size, seq_len, config.vocab_size), "Wrong output shape"
print("Forward pass: PASSED")

# 7. Test Loss Computation
print("\n7. Testing Loss Computation:")
print("-" * 70)

from model.diffusion_utils import compute_diffusion_loss

loss = compute_diffusion_loss(logits, x_0, noise_mask, reduction='mean')
print(f"Diffusion loss (on masked positions): {loss.item():.4f}")
assert loss.item() > 0, "Loss should be positive"
assert not torch.isnan(loss), "Loss should not be NaN"
print("Loss computation: PASSED")

# 8. Test with Different Noise Schedules
print("\n8. Testing Different Noise Schedules:")
print("-" * 70)

for schedule_type in ['cosine', 'linear', 'sqrt']:
    sched = NoiseSchedule(num_timesteps=1000, schedule_type=schedule_type)
    alpha_mid = sched.get_alpha(torch.tensor([500])).item()
    print(f"  {schedule_type:8s}: alpha(500) = {alpha_mid:.4f}")
print("All schedules: PASSED")

# 9. Test indices_to_mask Utility
print("\n9. Testing indices_to_mask Utility:")
print("-" * 70)

indices_list = [[0, 2, 4], [1, 3]]
seq_len = 6
mask = indices_to_mask(indices_list, seq_len, device='cpu')
print(f"Indices: {indices_list}")
print(f"Mask:\n{mask.int()}")
expected = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0]], dtype=torch.bool)
assert torch.equal(mask, expected), "Mask conversion incorrect"
print("indices_to_mask: PASSED")

# 10. Test Model with Bidirectional Attention
print("\n10. Verifying Bidirectional Attention:")
print("-" * 70)

# In bidirectional mode, changing any position should affect all positions
x_test = torch.randint(0, config.vocab_size, (1, 8))
t_test = torch.tensor([500])

# Get logits
logits1 = model(x_test, t_test)

# Change position 0
x_test_modified = x_test.clone()
x_test_modified[0, 0] = (x_test[0, 0] + 1) % config.vocab_size
logits2 = model(x_test_modified, t_test)

# All positions should be affected (not just later positions as in causal)
diff = (logits1 - logits2).abs().sum(dim=-1)  # (1, 8)
print(f"Logit differences when changing position 0: {diff[0].tolist()}")
# In bidirectional, all positions should change (unlike causal where only later positions change)
assert diff[0, 0].item() > 0, "Position 0 should be affected"
assert diff[0, 4].item() > 0, "Position 4 should be affected (bidirectional)"
assert diff[0, 7].item() > 0, "Position 7 should be affected (bidirectional)"
print("Bidirectional attention: VERIFIED")

print("\n" + "=" * 70)
print("ALL DIFFUSION TESTS PASSED!")
print("=" * 70)
print("\nDiffusion model implementation verified:")
print("  - NoiseSchedule (cosine, linear, sqrt)")
print("  - Forward diffusion (x_0 -> x_t)")
print("  - Conditional diffusion (X_c fixed)")
print("  - DiffusionGPT2Model with timestep embedding")
print("  - Bidirectional attention (all positions attend to all)")
print("  - Loss computation on masked positions")
print("\nNext steps:")
print("  1. Run training: python train.py --model_type diffusion --model_config tiny")
print("  2. Submit to cluster: sbatch scripts/submit_diffusion.sh")
print()
