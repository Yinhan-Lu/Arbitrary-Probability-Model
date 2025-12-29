"""
Tests for DualAxisRotaryEmbedding and SigmaGPT with dual_rope mode

Run from project root: python tests/test_dual_axis_rope.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def test_basic_shapes():
    """Test DualAxisRotaryEmbedding output shapes are correct"""
    from model.rope import DualAxisRotaryEmbedding

    rope = DualAxisRotaryEmbedding(dim=64, max_seq_len=128)

    B, n_head, T, head_dim = 2, 4, 10, 64
    q = torch.randn(B, n_head, T, head_dim)
    k = torch.randn(B, n_head, T, head_dim)

    curr_pos = torch.arange(T).unsqueeze(0).expand(B, -1)
    next_pos = torch.arange(1, T + 1).unsqueeze(0).expand(B, -1)

    q_rot, k_rot = rope(q, k, curr_pos, next_pos)

    assert q_rot.shape == q.shape, f"Expected {q.shape}, got {q_rot.shape}"
    assert k_rot.shape == k.shape, f"Expected {k.shape}, got {k_rot.shape}"
    print("✓ test_basic_shapes passed")


def test_different_positions_different_output():
    """Same input with different positions produces different output"""
    from model.rope import DualAxisRotaryEmbedding

    rope = DualAxisRotaryEmbedding(dim=64, max_seq_len=128)

    q = torch.randn(1, 1, 1, 64)
    k = q.clone()

    q1, _ = rope(q, k, torch.tensor([[0]]), torch.tensor([[1]]))
    q2, _ = rope(q, k, torch.tensor([[5]]), torch.tensor([[6]]))

    diff = (q1 - q2).abs().sum().item()
    assert diff > 0.01, f"Different positions should produce different outputs, diff={diff}"
    print("✓ test_different_positions_different_output passed")


def test_same_position_same_output():
    """Same position always produces same rotation"""
    from model.rope import DualAxisRotaryEmbedding

    rope = DualAxisRotaryEmbedding(dim=64, max_seq_len=128)

    q = torch.randn(1, 1, 1, 64)
    k = q.clone()

    q1, k1 = rope(q, k, torch.tensor([[3]]), torch.tensor([[4]]))
    q2, k2 = rope(q, k, torch.tensor([[3]]), torch.tensor([[4]]))

    assert torch.allclose(q1, q2), "Same position should produce identical output for q"
    assert torch.allclose(k1, k2), "Same position should produce identical output for k"
    print("✓ test_same_position_same_output passed")


def test_position_clamping():
    """Positions beyond max_seq_len are clamped"""
    from model.rope import DualAxisRotaryEmbedding

    rope = DualAxisRotaryEmbedding(dim=64, max_seq_len=128)

    q = torch.randn(1, 1, 1, 64)
    k = q.clone()

    # Position 200 should be clamped to 127
    q1, _ = rope(q, k, torch.tensor([[200]]), torch.tensor([[201]]))
    q2, _ = rope(q, k, torch.tensor([[127]]), torch.tensor([[127]]))

    assert torch.allclose(q1, q2), "Clamped positions should match max position"
    print("✓ test_position_clamping passed")


def test_different_bases():
    """Test with different base frequencies for each axis"""
    from model.rope import DualAxisRotaryEmbedding

    rope = DualAxisRotaryEmbedding(
        dim=64, max_seq_len=128,
        base_axis1=10000.0, base_axis2=5000.0
    )

    q = torch.randn(2, 4, 10, 64)
    k = torch.randn(2, 4, 10, 64)

    curr_pos = torch.arange(10).unsqueeze(0).expand(2, -1)
    next_pos = torch.arange(1, 11).unsqueeze(0).expand(2, -1)

    q_rot, k_rot = rope(q, k, curr_pos, next_pos)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print("✓ test_different_bases passed")


def test_two_axes_independent():
    """Test that two axes rotate independently based on their positions"""
    from model.rope import DualAxisRotaryEmbedding

    rope = DualAxisRotaryEmbedding(dim=64, max_seq_len=128)

    q = torch.randn(1, 1, 1, 64)
    k = q.clone()

    # Same current position, different next position
    q1, _ = rope(q, k, torch.tensor([[5]]), torch.tensor([[6]]))
    q2, _ = rope(q, k, torch.tensor([[5]]), torch.tensor([[10]]))

    # First halves should be the same (same current position)
    half = 32
    first_half_diff = (q1[..., :half] - q2[..., :half]).abs().sum().item()
    # Second halves should differ (different next position)
    second_half_diff = (q1[..., half:] - q2[..., half:]).abs().sum().item()

    assert first_half_diff < 1e-5, f"First half should be same, diff={first_half_diff}"
    assert second_half_diff > 0.01, f"Second half should differ, diff={second_half_diff}"
    print("✓ test_two_axes_independent passed")


def test_sigmagpt_with_dual_rope():
    """Test SigmaGPT forward pass with dual_rope"""
    from model.sigmagpt_from_baseline import SigmaGPTModel
    from model.arbitrary_prob_gpt2 import GPT2Config

    config = GPT2Config(
        vocab_size=1000,
        n_embd=128,
        n_head=4,
        n_layer=2,
        max_seq_len=64,
        position_encoding_type="dual_rope"
    )

    model = SigmaGPTModel(config)

    # Verify no wpe
    assert model.wpe is None, "dual_rope mode should not have wpe"

    B, T = 2, 10
    idx = torch.randint(0, 1000, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)
    targets = torch.randint(0, 1000, (B, T))

    logits, loss = model(idx, order, targets)

    assert logits.shape == (B, T, 1000), f"Expected logits shape (2, 10, 1000), got {logits.shape}"
    assert loss is not None, "Loss should not be None when targets provided"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("✓ test_sigmagpt_with_dual_rope passed")


def test_sigmagpt_learned_vs_rope():
    """Compare SigmaGPT with learned vs dual_rope position encoding"""
    from model.sigmagpt_from_baseline import SigmaGPTModel
    from model.arbitrary_prob_gpt2 import GPT2Config

    # Same config except position encoding
    base_kwargs = dict(
        vocab_size=1000,
        n_embd=128,
        n_head=4,
        n_layer=2,
        max_seq_len=64,
    )

    config_learned = GPT2Config(**base_kwargs, position_encoding_type="learned")
    config_rope = GPT2Config(**base_kwargs, position_encoding_type="dual_rope")

    model_learned = SigmaGPTModel(config_learned)
    model_rope = SigmaGPTModel(config_rope)

    # Check wpe presence
    assert model_learned.wpe is not None, "learned mode should have wpe"
    assert model_rope.wpe is None, "dual_rope mode should not have wpe"

    # Both should work with same input
    B, T = 2, 10
    idx = torch.randint(0, 1000, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)
    targets = torch.randint(0, 1000, (B, T))

    logits_learned, loss_learned = model_learned(idx, order, targets)
    logits_rope, loss_rope = model_rope(idx, order, targets)

    assert logits_learned.shape == logits_rope.shape, "Output shapes should match"
    assert not torch.isnan(loss_learned), "Learned loss should not be NaN"
    assert not torch.isnan(loss_rope), "RoPE loss should not be NaN"

    print("✓ test_sigmagpt_learned_vs_rope passed")


def test_sigmagpt_with_random_order():
    """Test SigmaGPT with random generation order"""
    from model.sigmagpt_from_baseline import SigmaGPTModel
    from model.arbitrary_prob_gpt2 import GPT2Config

    config = GPT2Config(
        vocab_size=1000,
        n_embd=128,
        n_head=4,
        n_layer=2,
        max_seq_len=64,
        position_encoding_type="dual_rope"
    )

    model = SigmaGPTModel(config)

    B, T = 2, 10
    idx = torch.randint(0, 1000, (B, T))

    # Create random order for each batch
    orders = []
    for _ in range(B):
        perm = torch.randperm(T)
        order = torch.cat([perm, torch.tensor([T])])
        orders.append(order)
    order = torch.stack(orders)

    targets = torch.randint(0, 1000, (B, T))

    logits, loss = model(idx, order, targets)

    assert logits.shape == (B, T, 1000)
    assert not torch.isnan(loss), "Loss should not be NaN with random order"
    print("✓ test_sigmagpt_with_random_order passed")


def test_backward_pass():
    """Test that gradients flow correctly"""
    from model.sigmagpt_from_baseline import SigmaGPTModel
    from model.arbitrary_prob_gpt2 import GPT2Config

    config = GPT2Config(
        vocab_size=1000,
        n_embd=128,
        n_head=4,
        n_layer=2,
        max_seq_len=64,
        position_encoding_type="dual_rope"
    )

    model = SigmaGPTModel(config)

    B, T = 2, 10
    idx = torch.randint(0, 1000, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)
    targets = torch.randint(0, 1000, (B, T))

    logits, loss = model(idx, order, targets)
    loss.backward()

    # Check that gradients exist for key parameters
    assert model.wte.weight.grad is not None, "Token embedding should have gradients"
    assert model.blocks[0].attn.c_attn.weight.grad is not None, "Attention should have gradients"

    print("✓ test_backward_pass passed")


def test_rope_parameter_efficiency():
    """Test that dual_rope mode has fewer parameters (no position embedding)"""
    from model.sigmagpt_from_baseline import SigmaGPTModel
    from model.arbitrary_prob_gpt2 import GPT2Config

    base_kwargs = dict(
        vocab_size=1000,
        n_embd=128,
        n_head=4,
        n_layer=2,
        max_seq_len=64,
    )

    config_learned = GPT2Config(**base_kwargs, position_encoding_type="learned")
    config_rope = GPT2Config(**base_kwargs, position_encoding_type="dual_rope")

    model_learned = SigmaGPTModel(config_learned)
    model_rope = SigmaGPTModel(config_rope)

    params_learned = model_learned.get_num_params()
    params_rope = model_rope.get_num_params()

    # RoPE should have fewer parameters (no wpe)
    # wpe has max_seq_len * (n_embd // 2) = 64 * 64 = 4096 parameters
    expected_diff = 64 * 64  # max_seq_len * (n_embd // 2)
    actual_diff = params_learned - params_rope

    assert actual_diff == expected_diff, \
        f"Parameter difference should be {expected_diff}, got {actual_diff}"
    print(f"✓ test_rope_parameter_efficiency passed (saved {actual_diff} parameters)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DualAxisRotaryEmbedding and SigmaGPT dual_rope mode")
    print("=" * 60)

    # DualAxisRotaryEmbedding unit tests
    print("\n[DualAxisRotaryEmbedding Unit Tests]")
    test_basic_shapes()
    test_different_positions_different_output()
    test_same_position_same_output()
    test_position_clamping()
    test_different_bases()
    test_two_axes_independent()

    # SigmaGPT integration tests
    print("\n[SigmaGPT Integration Tests]")
    test_sigmagpt_with_dual_rope()
    test_sigmagpt_learned_vs_rope()
    test_sigmagpt_with_random_order()
    test_backward_pass()
    test_rope_parameter_efficiency()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
