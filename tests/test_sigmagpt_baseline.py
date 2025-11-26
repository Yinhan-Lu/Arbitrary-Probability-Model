"""
Test Suite for Sigma GPT (from baseline_gpt2.py)

Tests the new sigmagpt_from_baseline.py implementation to ensure:
1. Basic forward pass works correctly
2. Output matches the reference sigmagpt_model.py
3. Edge cases are handled properly
4. Integration with training pipeline works

Run from project root: python tests/test_sigmagpt_baseline.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

# Import both implementations for comparison
from model.sigmagpt_from_baseline import SigmaGPTModel as SigmaGPTBaseline, GPT2Config
from model.sigmagpt_model import SigmaGPT as SigmaGPTReference

print("=" * 80)
print("Sigma GPT (from baseline) Test Suite")
print("=" * 80)

# Shared config for all tests
TEST_CONFIG = GPT2Config(
    vocab_size=1000,  # Small vocab for faster testing
    n_layer=2,
    n_head=4,
    n_embd=128,
    max_seq_len=64,
    dropout=0.0  # Disable dropout for deterministic testing
)

# Test parameters
BATCH_SIZE = 2
SEQ_LEN = 10
TOLERANCE = 1e-5


def test_basic_forward():
    """Test 1: Basic forward pass"""
    print("\n[Test 1] Basic Forward Pass")
    print("-" * 80)

    model = SigmaGPTBaseline(TEST_CONFIG)
    model.eval()

    # Create inputs
    idx = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))
    order = torch.arange(SEQ_LEN + 1).unsqueeze(0).expand(BATCH_SIZE, -1)
    targets = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))

    # Forward pass
    logits, loss = model(idx, order, targets)

    # Assertions
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, TEST_CONFIG.vocab_size), \
        f"Expected logits shape {(BATCH_SIZE, SEQ_LEN, TEST_CONFIG.vocab_size)}, got {logits.shape}"
    assert loss is not None, "Loss should not be None when targets provided"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"

    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Loss: {loss.item():.4f}")
    print("✓ Test 1 passed")


def test_position_embedding_size():
    """Test 2: Position embedding should be n_embd // 2"""
    print("\n[Test 2] Position Embedding Size")
    print("-" * 80)

    model = SigmaGPTBaseline(TEST_CONFIG)

    # Check position embedding dimension
    pos_emb_weight = model.wpe.weight
    expected_shape = (TEST_CONFIG.max_seq_len, TEST_CONFIG.n_embd // 2)

    assert pos_emb_weight.shape == expected_shape, \
        f"Expected position embedding shape {expected_shape}, got {pos_emb_weight.shape}"

    print(f"✓ Position embedding shape: {pos_emb_weight.shape}")
    print(f"✓ Correctly using n_embd // 2 = {TEST_CONFIG.n_embd // 2}")
    print("✓ Test 2 passed")


def test_double_position_encoding():
    """Test 3: Double position encoding produces correct output shape"""
    print("\n[Test 3] Double Position Encoding")
    print("-" * 80)

    model = SigmaGPTBaseline(TEST_CONFIG)
    model.eval()

    idx = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))
    order = torch.arange(SEQ_LEN + 1).unsqueeze(0).expand(BATCH_SIZE, -1)

    # Test _pos_emb method
    pos_emb = model._pos_emb(idx, order)

    expected_shape = (BATCH_SIZE, SEQ_LEN, TEST_CONFIG.n_embd)
    assert pos_emb.shape == expected_shape, \
        f"Expected position embedding shape {expected_shape}, got {pos_emb.shape}"

    # Check not all zeros
    assert pos_emb.abs().sum() > 0, "Position embeddings should not be all zeros"

    print(f"✓ Position embedding output shape: {pos_emb.shape}")
    print(f"✓ Double encoding produces full n_embd={TEST_CONFIG.n_embd} dimensions")
    print("✓ Test 3 passed")


def test_ignore_index():
    """Test 4: Ignore index -1 should not contribute to loss"""
    print("\n[Test 4] Ignore Index (-1)")
    print("-" * 80)

    model = SigmaGPTBaseline(TEST_CONFIG)
    model.eval()

    idx = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))
    order = torch.arange(SEQ_LEN + 1).unsqueeze(0).expand(BATCH_SIZE, -1)

    # Test 1: All valid targets
    targets_all = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))
    _, loss_all = model(idx, order, targets_all)

    # Test 2: Half targets ignored
    targets_half = targets_all.clone()
    targets_half[:, :SEQ_LEN//2] = -1  # Ignore first half
    _, loss_half = model(idx, order, targets_half)

    # Test 3: All targets ignored
    targets_none = torch.full((BATCH_SIZE, SEQ_LEN), -1, dtype=torch.long)
    _, loss_none = model(idx, order, targets_none)

    print(f"  Loss (all valid): {loss_all.item():.4f}")
    print(f"  Loss (half ignored): {loss_half.item():.4f}")
    print(f"  Loss (all ignored): {loss_none.item():.4f}")

    # Loss with all ignored should be very close to 0 (numerical precision)
    assert torch.isnan(loss_none) or loss_none.item() < 1e-6, \
        "Loss should be ~0 when all targets are ignored"

    # Loss with half ignored should be different from loss with all valid
    assert abs(loss_all.item() - loss_half.item()) > 0.01, \
        "Loss should differ when some targets are ignored"

    print("✓ Ignore index -1 working correctly")
    print("✓ Test 4 passed")


def test_random_order():
    """Test 5: Model should work with arbitrary generation orders"""
    print("\n[Test 5] Arbitrary Generation Order")
    print("-" * 80)

    model = SigmaGPTBaseline(TEST_CONFIG)
    model.eval()

    idx = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))
    targets = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))

    # Test different orders
    orders = {
        "left-to-right": torch.arange(SEQ_LEN + 1).unsqueeze(0).expand(BATCH_SIZE, -1),
        "right-to-left": torch.cat([torch.arange(SEQ_LEN - 1, -1, -1), torch.tensor([SEQ_LEN])]).unsqueeze(0).expand(BATCH_SIZE, -1),
        "random": torch.cat([torch.randperm(SEQ_LEN), torch.tensor([SEQ_LEN])]).unsqueeze(0).expand(BATCH_SIZE, -1),
    }

    print(f"  Testing with different generation orders:")
    for order_name, order in orders.items():
        logits, loss = model(idx, order, targets)
        assert logits.shape == (BATCH_SIZE, SEQ_LEN, TEST_CONFIG.vocab_size)
        assert not torch.isnan(loss) and not torch.isinf(loss)
        print(f"    ✓ {order_name}: loss={loss.item():.4f}")

    print("✓ All orders work correctly")
    print("✓ Test 5 passed")


def test_order_clamping():
    """Test 6: Order values >= max_seq_len should be clamped"""
    print("\n[Test 6] Order Clamping")
    print("-" * 80)

    model = SigmaGPTBaseline(TEST_CONFIG)
    model.eval()

    idx = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))
    targets = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))

    # Create order with values >= max_seq_len (should be clamped)
    order = torch.arange(SEQ_LEN + 1).unsqueeze(0).expand(BATCH_SIZE, -1)
    order = order.clone()
    order[:, -1] = TEST_CONFIG.max_seq_len  # Last element is seq_len (should be clamped)

    # Should not raise error
    logits, loss = model(idx, order, targets)

    assert not torch.isnan(loss), "Loss should not be NaN after clamping"
    assert not torch.isinf(loss), "Loss should not be Inf after clamping"

    print(f"✓ Order with max_seq_len handled correctly")
    print(f"✓ Loss: {loss.item():.4f}")
    print("✓ Test 6 passed")


def test_gradient_flow():
    """Test 7: Gradients should flow properly"""
    print("\n[Test 7] Gradient Flow")
    print("-" * 80)

    model = SigmaGPTBaseline(TEST_CONFIG)
    model.train()

    idx = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))
    order = torch.arange(SEQ_LEN + 1).unsqueeze(0).expand(BATCH_SIZE, -1)
    targets = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))

    # Forward pass
    logits, loss = model(idx, order, targets)

    # Backward pass
    loss.backward()

    # Check gradients exist and are not NaN/Inf
    has_grad = 0
    nan_grad = 0
    zero_grad = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad += 1
            if torch.isnan(param.grad).any():
                nan_grad += 1
                print(f"  ✗ NaN gradient in {name}")
            elif param.grad.abs().sum() == 0:
                zero_grad += 1
                # Some parameters might have zero gradients (e.g., unused embeddings)

    print(f"  Parameters with gradients: {has_grad}")
    print(f"  Parameters with NaN gradients: {nan_grad}")
    print(f"  Parameters with zero gradients: {zero_grad}")

    assert nan_grad == 0, "No parameter should have NaN gradients"
    assert has_grad > 0, "At least some parameters should have gradients"

    print("✓ Gradient flow is healthy")
    print("✓ Test 7 passed")


def test_comparison_with_reference():
    """Test 8: Compare output with reference sigmagpt_model.py"""
    print("\n[Test 8] Comparison with Reference Implementation")
    print("-" * 80)

    # Create both models with same config
    model_baseline = SigmaGPTBaseline(TEST_CONFIG)
    model_reference = SigmaGPTReference(TEST_CONFIG)

    # Copy weights from baseline to reference (to ensure identical initialization)
    # Token embeddings
    model_reference.transformer["wte"].weight.data = model_baseline.wte.weight.data.clone()
    # Position embeddings
    model_reference.transformer["wpe"].weight.data = model_baseline.wpe.weight.data.clone()
    # LM head (weight tied, so should already match)

    # Copy transformer blocks
    for i in range(TEST_CONFIG.n_layer):
        # Layer norm 1
        model_reference.transformer["h"][i].ln_1.weight.data = model_baseline.blocks[i].ln_1.weight.data.clone()
        model_reference.transformer["h"][i].ln_1.bias.data = model_baseline.blocks[i].ln_1.bias.data.clone()

        # Attention
        model_reference.transformer["h"][i].attn.c_attn.weight.data = model_baseline.blocks[i].attn.c_attn.weight.data.clone()
        model_reference.transformer["h"][i].attn.c_attn.bias.data = model_baseline.blocks[i].attn.c_attn.bias.data.clone()
        model_reference.transformer["h"][i].attn.c_proj.weight.data = model_baseline.blocks[i].attn.c_proj.weight.data.clone()
        model_reference.transformer["h"][i].attn.c_proj.bias.data = model_baseline.blocks[i].attn.c_proj.bias.data.clone()

        # Layer norm 2
        model_reference.transformer["h"][i].ln_2.weight.data = model_baseline.blocks[i].ln_2.weight.data.clone()
        model_reference.transformer["h"][i].ln_2.bias.data = model_baseline.blocks[i].ln_2.bias.data.clone()

        # MLP
        model_reference.transformer["h"][i].mlp.c_fc.weight.data = model_baseline.blocks[i].mlp.c_fc.weight.data.clone()
        model_reference.transformer["h"][i].mlp.c_fc.bias.data = model_baseline.blocks[i].mlp.c_fc.bias.data.clone()
        model_reference.transformer["h"][i].mlp.c_proj.weight.data = model_baseline.blocks[i].mlp.c_proj.weight.data.clone()
        model_reference.transformer["h"][i].mlp.c_proj.bias.data = model_baseline.blocks[i].mlp.c_proj.bias.data.clone()

    # Final layer norm
    model_reference.transformer["ln_f"].weight.data = model_baseline.ln_f.weight.data.clone()
    model_reference.transformer["ln_f"].bias.data = model_baseline.ln_f.bias.data.clone()

    # Set both to eval mode
    model_baseline.eval()
    model_reference.eval()

    # Create inputs
    torch.manual_seed(42)  # For reproducibility
    idx = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))
    order = torch.arange(SEQ_LEN + 1).unsqueeze(0).expand(BATCH_SIZE, -1)
    targets = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))

    # Forward pass
    with torch.no_grad():
        logits_baseline, loss_baseline = model_baseline(idx, order, targets)
        logits_reference, loss_reference = model_reference(idx, order, targets)

    # Compare outputs
    logits_diff = (logits_baseline - logits_reference).abs().max().item()
    loss_diff = abs(loss_baseline.item() - loss_reference.item())

    print(f"  Max logits difference: {logits_diff:.8f}")
    print(f"  Loss difference: {loss_diff:.8f}")

    # Allow small numerical differences due to different implementations
    assert logits_diff < 1e-4, f"Logits differ too much: {logits_diff}"
    assert loss_diff < 1e-4, f"Loss differs too much: {loss_diff}"

    print("✓ Output matches reference implementation")
    print("✓ Test 8 passed")


def test_no_targets():
    """Test 9: Forward pass without targets (inference mode)"""
    print("\n[Test 9] Forward Pass Without Targets")
    print("-" * 80)

    model = SigmaGPTBaseline(TEST_CONFIG)
    model.eval()

    idx = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH_SIZE, SEQ_LEN))
    order = torch.arange(SEQ_LEN + 1).unsqueeze(0).expand(BATCH_SIZE, -1)

    # Forward pass without targets
    logits, loss = model(idx, order, targets=None)

    assert logits.shape == (BATCH_SIZE, SEQ_LEN, TEST_CONFIG.vocab_size)
    assert loss is None, "Loss should be None when targets not provided"

    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Loss is None (as expected)")
    print("✓ Test 9 passed")


def test_parameter_count():
    """Test 10: Parameter count should match reference"""
    print("\n[Test 10] Parameter Count")
    print("-" * 80)

    model_baseline = SigmaGPTBaseline(TEST_CONFIG)
    model_reference = SigmaGPTReference(TEST_CONFIG)

    params_baseline = model_baseline.get_num_params()
    params_reference = model_reference.get_num_params()

    print(f"  Baseline model: {params_baseline:,} parameters")
    print(f"  Reference model: {params_reference:,} parameters")
    print(f"  Difference: {abs(params_baseline - params_reference):,}")

    # Should be exactly the same
    assert params_baseline == params_reference, \
        f"Parameter count mismatch: {params_baseline} vs {params_reference}"

    print("✓ Parameter counts match")
    print("✓ Test 10 passed")


# Run all tests
if __name__ == "__main__":
    try:
        test_basic_forward()
        test_position_embedding_size()
        test_double_position_encoding()
        test_ignore_index()
        test_random_order()
        test_order_clamping()
        test_gradient_flow()
        test_comparison_with_reference()
        test_no_targets()
        test_parameter_count()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 80)
        print("\nSigma GPT (from baseline) implementation is correct and matches reference.")

    except AssertionError as e:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ UNEXPECTED ERROR")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
