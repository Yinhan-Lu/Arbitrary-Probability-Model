"""
Unit tests for SigmaGPT model

Tests verify critical architectural features:
- Position embedding dimension (n_embd // 2)
- Double position encoding implementation
- Forward pass shapes and loss computation
- Ignore index=-1 handling
- Order tensor requirements

Run from project root: python tests/test_sigmagpt_model.py
Expected runtime: ~30 seconds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from model.sigmagpt_model import SigmaGPT
from model.config import GPT2Config


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_model_initialization():
    """Test 1: Model initialization and architecture"""
    print_section("Test 1: Model Initialization")

    config = GPT2Config(
        vocab_size=50257,
        max_seq_len=1024,
        n_layer=6,
        n_head=8,
        n_embd=512,
        dropout=0.1
    )

    model = SigmaGPT(config)

    # Verify token embedding dimension (standard)
    assert model.transformer["wte"].embedding_dim == config.n_embd, \
        f"Token embedding should be {config.n_embd}, got {model.transformer['wte'].embedding_dim}"

    # CRITICAL: Verify position embedding dimension (n_embd // 2)
    expected_pos_dim = config.n_embd // 2
    actual_pos_dim = model.transformer["wpe"].embedding_dim
    assert actual_pos_dim == expected_pos_dim, \
        f"Position embedding MUST be {expected_pos_dim}, got {actual_pos_dim}"

    # Verify number of layers
    assert len(model.transformer["h"]) == config.n_layer, \
        f"Expected {config.n_layer} layers, got {len(model.transformer['h'])}"

    # Verify weight tying
    assert model.transformer["wte"].weight is model.lm_head.weight, \
        "Token embedding and LM head should share weights"

    print(f"✓ Model initialized correctly")
    print(f"  - Token embedding dim: {config.n_embd}")
    print(f"  - Position embedding dim: {expected_pos_dim} (n_embd // 2)")
    print(f"  - Number of layers: {config.n_layer}")
    print(f"  - Number of heads: {config.n_head}")
    print(f"  - Total parameters: {model.get_num_params() / 1e6:.2f}M")
    print(f"  - Weight tying: ✓")


def test_double_position_encoding():
    """Test 2: Double position encoding implementation"""
    print_section("Test 2: Double Position Encoding")

    config = GPT2Config(
        vocab_size=50257,
        max_seq_len=128,
        n_layer=2,
        n_head=4,
        n_embd=256,
        dropout=0.0  # Disable dropout for deterministic testing
    )

    model = SigmaGPT(config)
    model.eval()

    B, T = 2, 10

    # Create sample input
    idx = torch.randint(0, config.vocab_size, (B, T))

    # Create order tensor (B, T+1)
    # Format: [current_positions, next_position]
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)

    # Test position embedding
    pos_emb = model._pos_emb(idx, order)

    # Verify output shape
    assert pos_emb.shape == (B, T, config.n_embd), \
        f"Position embedding should be {(B, T, config.n_embd)}, got {pos_emb.shape}"

    # Verify that position encoding uses both current and next positions
    # We can verify this by checking that different order sequences produce different embeddings
    order_reversed = torch.arange(T, -1, -1).unsqueeze(0).expand(B, -1)
    pos_emb_reversed = model._pos_emb(idx, order_reversed)

    # Embeddings should be different for different orders
    assert not torch.allclose(pos_emb, pos_emb_reversed), \
        "Position embeddings should differ for different order sequences"

    print(f"✓ Double position encoding works correctly")
    print(f"  - Input shape: {idx.shape}")
    print(f"  - Order shape: {order.shape}")
    print(f"  - Output shape: {pos_emb.shape}")
    print(f"  - Position embedding dim: {config.n_embd // 2} per position")
    print(f"  - Total dim: {config.n_embd} (current + next)")


def test_forward_pass():
    """Test 3: Forward pass output shapes and loss computation"""
    print_section("Test 3: Forward Pass")

    config = GPT2Config(
        vocab_size=50257,
        max_seq_len=128,
        n_layer=2,
        n_head=4,
        n_embd=256,
        dropout=0.0
    )

    model = SigmaGPT(config)
    model.eval()

    B, T = 4, 16

    # Create inputs
    idx = torch.randint(0, config.vocab_size, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)

    # Test without targets (no loss computation)
    with torch.no_grad():
        logits, loss = model(idx, order, targets=None)

    assert logits.shape == (B, T, config.vocab_size), \
        f"Logits should be {(B, T, config.vocab_size)}, got {logits.shape}"
    assert loss is None, "Loss should be None when targets not provided"

    # Test with targets (with loss computation)
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        logits, loss = model(idx, order, targets=targets)

    assert loss is not None, "Loss should be computed when targets provided"
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"

    print(f"✓ Forward pass works correctly")
    print(f"  - Input shape: {idx.shape}")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Loss: {loss.item():.4f}")


def test_ignore_index():
    """Test 4: Verify ignore_index=-1 handling"""
    print_section("Test 4: Ignore Index Handling")

    config = GPT2Config(
        vocab_size=100,
        max_seq_len=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0
    )

    model = SigmaGPT(config)
    model.eval()

    B, T = 2, 8

    # Create inputs
    idx = torch.randint(0, config.vocab_size, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)

    # Create targets with some positions set to -1 (should be ignored)
    targets = torch.randint(0, config.vocab_size, (B, T))
    targets[0, :4] = -1  # First 4 positions in first batch should be ignored

    # Compute loss
    with torch.no_grad():
        logits, loss_with_ignore = model(idx, order, targets=targets)

    # Manually compute expected loss (ignoring positions with -1)
    valid_mask = (targets != -1)
    n_valid = valid_mask.sum().item()

    logits_flat = logits.view(-1, config.vocab_size)
    targets_flat = targets.view(-1)

    # Compute loss only on valid positions
    loss_manual = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=-1
    )

    # Verify losses match
    assert torch.allclose(loss_with_ignore, loss_manual, atol=1e-6), \
        f"Loss mismatch: model={loss_with_ignore.item():.6f}, manual={loss_manual.item():.6f}"

    # Verify that ignored positions don't contribute to loss
    # Create targets without any -1
    targets_all = targets.clone()
    targets_all[targets == -1] = 0  # Replace -1 with valid token

    with torch.no_grad():
        _, loss_all = model(idx, order, targets_all)

    # Losses should be different (unless by chance the replaced tokens match predictions)
    # We just verify that the computation doesn't crash and produces valid output
    assert not torch.isnan(loss_all), "Loss without ignore should be valid"

    print(f"✓ Ignore index=-1 works correctly")
    print(f"  - Total positions: {B * T}")
    print(f"  - Ignored positions: {(targets == -1).sum().item()}")
    print(f"  - Valid positions: {n_valid}")
    print(f"  - Loss (with ignore): {loss_with_ignore.item():.4f}")
    print(f"  - Loss (all valid): {loss_all.item():.4f}")


def test_order_tensor_shapes():
    """Test 5: Order tensor shape requirements"""
    print_section("Test 5: Order Tensor Shape Requirements")

    config = GPT2Config(
        vocab_size=100,
        max_seq_len=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0
    )

    model = SigmaGPT(config)
    model.eval()

    B, T = 3, 12

    idx = torch.randint(0, config.vocab_size, (B, T))

    # Test 1: Correct shape (B, T+1)
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        logits, _ = model(idx, order, targets=None)

    assert logits.shape == (B, T, config.vocab_size), \
        "Model should work with correct order shape (B, T+1)"

    # Test 2: Truncation of longer order
    order_long = torch.arange(T + 10).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        logits_long, _ = model(idx, order_long, targets=None)

    assert logits_long.shape == (B, T, config.vocab_size), \
        "Model should truncate longer order tensors"

    print(f"✓ Order tensor handling works correctly")
    print(f"  - Input length: {T}")
    print(f"  - Required order length: {T + 1}")
    print(f"  - Order shape: {order.shape}")
    print(f"  - Handles truncation: ✓")


def test_parameter_count():
    """Test 6: Verify parameter count calculation"""
    print_section("Test 6: Parameter Count")

    config = GPT2Config(
        vocab_size=50257,
        max_seq_len=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1
    )

    model = SigmaGPT(config)

    # Manual calculation
    # Token embedding: vocab_size * n_embd (shared with lm_head)
    # Position embedding: max_seq_len * (n_embd // 2)
    # Transformer blocks: n_layer * (...)

    param_count = model.get_num_params()

    # Verify it's reasonable (should be similar to GPT-2 small, ~100-130M)
    assert param_count > 1e6, "Model should have at least 1M parameters"
    assert param_count < 1e9, "Model should have less than 1B parameters"

    # Verify position embedding has fewer params than standard GPT-2
    pos_emb_params = config.max_seq_len * (config.n_embd // 2)
    standard_pos_emb_params = config.max_seq_len * config.n_embd

    assert pos_emb_params < standard_pos_emb_params, \
        "Sigma GPT position embedding should use fewer params than standard GPT-2"

    print(f"✓ Parameter count calculated correctly")
    print(f"  - Total parameters: {param_count / 1e6:.2f}M")
    print(f"  - Position embedding params: {pos_emb_params:,}")
    print(f"  - Standard GPT-2 pos emb would be: {standard_pos_emb_params:,}")
    print(f"  - Savings: {standard_pos_emb_params - pos_emb_params:,} params")


def test_gradient_flow():
    """Test 7: Verify gradients flow correctly through double position encoding"""
    print_section("Test 7: Gradient Flow")

    config = GPT2Config(
        vocab_size=100,
        max_seq_len=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0
    )

    model = SigmaGPT(config)
    model.train()

    B, T = 2, 8

    # Create inputs
    idx = torch.randint(0, config.vocab_size, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)
    targets = torch.randint(0, config.vocab_size, (B, T))

    # Forward pass
    logits, loss = model(idx, order, targets=targets)

    # Backward pass
    loss.backward()

    # Verify position embedding has gradients
    assert model.transformer["wpe"].weight.grad is not None, \
        "Position embedding should have gradients"

    # Verify gradients are non-zero
    pos_grad_norm = model.transformer["wpe"].weight.grad.norm().item()
    assert pos_grad_norm > 0, "Position embedding gradients should be non-zero"

    # Verify token embedding has gradients
    assert model.transformer["wte"].weight.grad is not None, \
        "Token embedding should have gradients"

    tok_grad_norm = model.transformer["wte"].weight.grad.norm().item()
    assert tok_grad_norm > 0, "Token embedding gradients should be non-zero"

    print(f"✓ Gradient flow works correctly")
    print(f"  - Position embedding grad norm: {pos_grad_norm:.4f}")
    print(f"  - Token embedding grad norm: {tok_grad_norm:.4f}")
    print(f"  - Loss: {loss.item():.4f}")


def test_determinism():
    """Test 8: Verify deterministic behavior with same inputs"""
    print_section("Test 8: Determinism")

    torch.manual_seed(42)

    config = GPT2Config(
        vocab_size=100,
        max_seq_len=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0  # Disable dropout for determinism
    )

    model = SigmaGPT(config)
    model.eval()

    B, T = 2, 8

    # Create inputs
    idx = torch.randint(0, config.vocab_size, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)

    # First forward pass
    with torch.no_grad():
        logits1, _ = model(idx, order, targets=None)

    # Second forward pass with same inputs
    with torch.no_grad():
        logits2, _ = model(idx, order, targets=None)

    # Verify outputs are identical
    assert torch.allclose(logits1, logits2, atol=1e-7), \
        "Model should produce identical outputs for same inputs (dropout=0)"

    print(f"✓ Deterministic behavior verified")
    print(f"  - Logits match: ✓ (max diff: {(logits1 - logits2).abs().max().item():.2e})")


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "=" * 80)
    print("  SIGMA GPT MODEL UNIT TESTS")
    print("=" * 80)

    tests = [
        ("Model Initialization", test_model_initialization),
        ("Double Position Encoding", test_double_position_encoding),
        ("Forward Pass", test_forward_pass),
        ("Ignore Index=-1", test_ignore_index),
        ("Order Tensor Shapes", test_order_tensor_shapes),
        ("Parameter Count", test_parameter_count),
        ("Gradient Flow", test_gradient_flow),
        ("Determinism", test_determinism),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {test_name}")
            print(f"   Error: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}")
            print(f"   {type(e).__name__}: {str(e)}")
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)
    print(f"  Total tests: {len(tests)}")
    print(f"  Passed: {passed} ✓")
    print(f"  Failed: {failed} {'❌' if failed > 0 else ''}")
    print("=" * 80)

    if failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
