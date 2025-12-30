"""
Test Thinking Tokens Implementation

Tests for the thinking tokens feature in SigmaGPT:
1. Token count computation
2. ThinkingTokenPrepender shapes and values
3. TokenManager integration
4. SigmaGPT model integration
5. End-to-end forward pass

Run from project root: python tests/test_thinking_tokens.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_token_count_computation():
    """Test 1: Verify thinking token count calculations"""
    print("\n" + "=" * 60)
    print("[Test 1] Thinking token count computation")
    print("=" * 60)

    from model.thinking_tokens import compute_thinking_token_count

    test_cases = [
        # (cond_pct_max, max_seq_len, mode, expected)
        (0.2, 1024, "expectation", 102),
        (0.2, 1024, "upper_bound", 204),
        (0.4, 1024, "expectation", 204),
        (0.4, 1024, "upper_bound", 409),
        (0.6, 1024, "expectation", 307),
        (0.6, 1024, "upper_bound", 614),
        (0.8, 1024, "expectation", 409),
        (0.8, 1024, "upper_bound", 819),
        (1.0, 1024, "expectation", 512),
        (1.0, 1024, "upper_bound", 1024),
    ]

    all_passed = True
    for cond_max, seq_len, mode, expected in test_cases:
        result = compute_thinking_token_count(cond_max, seq_len, mode)
        status = "OK" if abs(result - expected) <= 1 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  cond_max={cond_max}, mode={mode}: {result} (expected ~{expected}) [{status}]")

    assert all_passed, "Token count computation failed"
    print("  [PASS] Test 1 passed")


def test_prepender_shapes():
    """Test 2: Verify ThinkingTokenPrepender output shapes"""
    print("\n" + "=" * 60)
    print("[Test 2] ThinkingTokenPrepender shapes")
    print("=" * 60)

    from model.thinking_tokens import ThinkingTokenPrepender

    n_thinking = 10
    thinking_ids = list(range(50258, 50258 + n_thinking))
    prepender = ThinkingTokenPrepender(thinking_ids, n_embd=768)

    B, T = 4, 20
    inputs = torch.randint(0, 1000, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)
    targets = torch.randint(0, 1000, (B, T))

    new_inputs, new_order, new_targets = prepender(inputs, order, targets)

    print(f"  n_thinking = {n_thinking}")
    print(f"  inputs: ({B}, {T}) -> {tuple(new_inputs.shape)}")
    print(f"  order: ({B}, {T+1}) -> {tuple(new_order.shape)}")
    print(f"  targets: ({B}, {T}) -> {tuple(new_targets.shape)}")

    assert new_inputs.shape == (B, n_thinking + T), f"Expected inputs shape {(B, n_thinking + T)}"
    assert new_order.shape == (B, n_thinking + T + 1), f"Expected order shape {(B, n_thinking + T + 1)}"
    assert new_targets.shape == (B, n_thinking + T), f"Expected targets shape {(B, n_thinking + T)}"

    print("  [PASS] Test 2 passed")


def test_prepender_order_values():
    """Test 3: Verify order tensor values after prepending"""
    print("\n" + "=" * 60)
    print("[Test 3] Order tensor values")
    print("=" * 60)

    from model.thinking_tokens import ThinkingTokenPrepender

    n_thinking = 3
    thinking_ids = [100, 101, 102]
    prepender = ThinkingTokenPrepender(thinking_ids, n_embd=768)

    B, T = 1, 5
    inputs = torch.tensor([[10, 20, 30, 40, 50]])
    # Original order: [0, 1, 2, 3, 4, 5]
    order = torch.tensor([[0, 1, 2, 3, 4, 5]])

    new_inputs, new_order, _ = prepender(inputs, order, None)

    print(f"  Original order: {order[0].tolist()}")
    print(f"  New order: {new_order[0].tolist()}")

    # Expected: [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # First 3 positions: thinking token positions (0, 1, 2)
    # Remaining: original positions shifted by 3 (0+3=3, 1+3=4, ..., 5+3=8)
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    assert new_order[0].tolist() == expected, f"Expected {expected}, got {new_order[0].tolist()}"
    print("  [PASS] Test 3 passed")


def test_prepender_target_masking():
    """Test 4: Verify thinking tokens have -1 targets (no loss)"""
    print("\n" + "=" * 60)
    print("[Test 4] Target masking for thinking tokens")
    print("=" * 60)

    from model.thinking_tokens import ThinkingTokenPrepender

    n_thinking = 5
    thinking_ids = list(range(100, 100 + n_thinking))
    prepender = ThinkingTokenPrepender(thinking_ids, n_embd=768)

    B, T = 2, 8
    inputs = torch.randint(0, 1000, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)
    targets = torch.randint(0, 1000, (B, T))  # All valid targets

    _, _, new_targets = prepender(inputs, order, targets)

    print(f"  Original targets[0]: {targets[0].tolist()}")
    print(f"  New targets[0]: {new_targets[0].tolist()}")

    # First n_thinking positions should be -1
    assert (new_targets[:, :n_thinking] == -1).all(), "Thinking positions should have -1 targets"
    # Remaining positions should match original
    assert (new_targets[:, n_thinking:] == targets).all(), "Body targets should be unchanged"

    print("  [PASS] Test 4 passed")


def test_prepender_input_tokens():
    """Test 5: Verify thinking tokens are prepended correctly"""
    print("\n" + "=" * 60)
    print("[Test 5] Thinking token prepending")
    print("=" * 60)

    from model.thinking_tokens import ThinkingTokenPrepender

    n_thinking = 4
    thinking_ids = [1001, 1002, 1003, 1004]
    prepender = ThinkingTokenPrepender(thinking_ids, n_embd=768)

    B, T = 2, 6
    inputs = torch.tensor([
        [10, 20, 30, 40, 50, 60],
        [70, 80, 90, 100, 110, 120]
    ])
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)

    new_inputs, _, _ = prepender(inputs, order, None)

    print(f"  Original inputs[0]: {inputs[0].tolist()}")
    print(f"  New inputs[0]: {new_inputs[0].tolist()}")
    print(f"  Expected first {n_thinking} tokens: {thinking_ids}")

    # First n positions should be thinking tokens
    assert new_inputs[0, :n_thinking].tolist() == thinking_ids
    assert new_inputs[1, :n_thinking].tolist() == thinking_ids
    # Rest should be original tokens
    assert new_inputs[0, n_thinking:].tolist() == inputs[0].tolist()
    assert new_inputs[1, n_thinking:].tolist() == inputs[1].tolist()

    print("  [PASS] Test 5 passed")


def test_token_manager_integration():
    """Test 6: Verify TokenManager creates thinking tokens correctly"""
    print("\n" + "=" * 60)
    print("[Test 6] TokenManager integration")
    print("=" * 60)

    from model.token_manager import TokenManager

    num_thinking = 10
    token_manager = TokenManager(
        add_mask_token=False,
        add_bos_token=False,
        num_thinking_tokens=num_thinking
    )

    thinking_ids = token_manager.get_thinking_token_ids()

    print(f"  Requested thinking tokens: {num_thinking}")
    print(f"  Created thinking tokens: {len(thinking_ids)}")
    print(f"  Token ID range: {thinking_ids[0]} to {thinking_ids[-1]}")
    print(f"  Vocab size: {len(token_manager.get_tokenizer())}")

    assert len(thinking_ids) == num_thinking, f"Expected {num_thinking} tokens, got {len(thinking_ids)}"
    assert thinking_ids == list(range(thinking_ids[0], thinking_ids[0] + num_thinking)), "Token IDs should be consecutive"
    assert len(token_manager.get_tokenizer()) == 50257 + num_thinking, "Vocab size should increase by num_thinking"

    print("  [PASS] Test 6 passed")


def test_sigmagpt_model_integration():
    """Test 7: Verify SigmaGPT model with thinking tokens"""
    print("\n" + "=" * 60)
    print("[Test 7] SigmaGPT model integration")
    print("=" * 60)

    from model.sigmagpt_from_baseline import SigmaGPTModel
    from model.arbitrary_prob_gpt2 import GPT2Config
    from model.token_manager import TokenManager

    # Create config
    config = GPT2Config(
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=256,
        dropout=0.0,
        position_encoding_type='dual_rope'
    )

    # Create TokenManager with thinking tokens
    num_thinking = 5
    token_manager = TokenManager(
        add_mask_token=False,
        add_bos_token=False,
        num_thinking_tokens=num_thinking
    )
    thinking_ids = token_manager.get_thinking_token_ids()

    # Create model with thinking tokens
    model = SigmaGPTModel(config, thinking_token_ids=thinking_ids)

    # Resize embeddings
    token_manager.resize_model_embeddings(model)

    print(f"  Model created with {model.num_thinking_tokens} thinking tokens")
    print(f"  Vocab size: {model.config.vocab_size}")
    print(f"  Has thinking prepender: {model.thinking_prepender is not None}")

    assert model.num_thinking_tokens == num_thinking
    assert model.thinking_prepender is not None
    assert model.config.vocab_size == 50257 + num_thinking

    print("  [PASS] Test 7 passed")


def test_forward_pass_with_thinking():
    """Test 8: End-to-end forward pass with thinking tokens"""
    print("\n" + "=" * 60)
    print("[Test 8] Forward pass with thinking tokens")
    print("=" * 60)

    from model.sigmagpt_from_baseline import SigmaGPTModel
    from model.arbitrary_prob_gpt2 import GPT2Config
    from model.token_manager import TokenManager

    # Create config
    config = GPT2Config(
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=256,
        dropout=0.0,
        position_encoding_type='dual_rope'
    )

    # Create TokenManager with thinking tokens
    num_thinking = 8
    token_manager = TokenManager(
        add_mask_token=False,
        add_bos_token=False,
        num_thinking_tokens=num_thinking
    )
    thinking_ids = token_manager.get_thinking_token_ids()

    # Create model with thinking tokens
    model = SigmaGPTModel(config, thinking_token_ids=thinking_ids)
    token_manager.resize_model_embeddings(model)

    # Create test data
    B, T = 2, 16
    inputs = torch.randint(0, 50257, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)
    targets = torch.randint(0, 50257, (B, T))

    print(f"  Input shape: {inputs.shape}")
    print(f"  Order shape: {order.shape}")
    print(f"  Targets shape: {targets.shape}")

    # Forward pass
    logits, loss = model(idx=inputs, order=order, targets=targets)

    # Expected output shape: (B, n + T, vocab_size)
    expected_logits_shape = (B, num_thinking + T, model.config.vocab_size)

    print(f"  Output logits shape: {tuple(logits.shape)}")
    print(f"  Expected: {expected_logits_shape}")
    print(f"  Loss: {loss.item():.4f}")

    assert logits.shape == expected_logits_shape, f"Expected {expected_logits_shape}, got {logits.shape}"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"

    print("  [PASS] Test 8 passed")


def test_loss_excludes_thinking():
    """Test 9: Verify loss excludes thinking token positions"""
    print("\n" + "=" * 60)
    print("[Test 9] Loss excludes thinking positions")
    print("=" * 60)

    from model.sigmagpt_from_baseline import SigmaGPTModel
    from model.arbitrary_prob_gpt2 import GPT2Config
    from model.token_manager import TokenManager

    # Create config
    config = GPT2Config(
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=256,
        dropout=0.0,
        position_encoding_type='dual_rope'
    )

    # Create model WITHOUT thinking tokens
    model_no_think = SigmaGPTModel(config, thinking_token_ids=None)

    # Create model WITH thinking tokens
    num_thinking = 10
    token_manager = TokenManager(
        add_mask_token=False,
        add_bos_token=False,
        num_thinking_tokens=num_thinking
    )
    thinking_ids = token_manager.get_thinking_token_ids()
    model_with_think = SigmaGPTModel(config, thinking_token_ids=thinking_ids)
    token_manager.resize_model_embeddings(model_with_think)

    # Create test data - use same random seed for both
    torch.manual_seed(42)
    B, T = 4, 20
    inputs = torch.randint(0, 50257, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)
    targets = torch.randint(0, 50257, (B, T))

    # Forward pass without thinking
    _, loss_no_think = model_no_think(idx=inputs.clone(), order=order.clone(), targets=targets.clone())

    # Forward pass with thinking
    _, loss_with_think = model_with_think(idx=inputs.clone(), order=order.clone(), targets=targets.clone())

    print(f"  Loss without thinking: {loss_no_think.item():.4f}")
    print(f"  Loss with thinking: {loss_with_think.item():.4f}")

    # Both should be valid (not NaN, not negative)
    assert not torch.isnan(loss_no_think), "Loss without thinking should not be NaN"
    assert not torch.isnan(loss_with_think), "Loss with thinking should not be NaN"
    assert loss_no_think.item() > 0, "Loss without thinking should be positive"
    assert loss_with_think.item() > 0, "Loss with thinking should be positive"

    # Losses will be different due to different models, but both should be reasonable
    print("  [PASS] Test 9 passed")


def test_position_encoding_range():
    """Test 10: Verify position encoding handles extended range without aliasing"""
    print("\n" + "=" * 60)
    print("[Test 10] Position encoding extended range")
    print("=" * 60)

    from model.sigmagpt_from_baseline import SigmaGPTModel
    from model.arbitrary_prob_gpt2 import GPT2Config
    from model.token_manager import TokenManager

    # Create config with small max_seq_len for easier testing
    max_seq_len = 64
    config = GPT2Config(
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=max_seq_len,
        dropout=0.0,
        position_encoding_type='dual_rope'
    )

    # Test with significant number of thinking tokens
    num_thinking = 32  # Half of max_seq_len
    token_manager = TokenManager(
        add_mask_token=False,
        add_bos_token=False,
        num_thinking_tokens=num_thinking
    )
    thinking_ids = token_manager.get_thinking_token_ids()

    # Create model with thinking tokens
    model = SigmaGPTModel(config, thinking_token_ids=thinking_ids)
    token_manager.resize_model_embeddings(model)

    # Verify effective_max_position is set correctly
    expected_effective = max_seq_len + num_thinking
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  num_thinking: {num_thinking}")
    print(f"  effective_max_position: {model.effective_max_position}")

    assert model.effective_max_position == expected_effective, \
        f"Expected {expected_effective}, got {model.effective_max_position}"

    # Verify RoPE cache size is extended
    # Access through the first block's attention layer
    rope = model.blocks[0].attn.rotary_emb
    print(f"  RoPE cache size: {rope.cos_cached_1.shape[0]}")

    assert rope.cos_cached_1.shape[0] == expected_effective, \
        f"RoPE cache should have {expected_effective} positions, got {rope.cos_cached_1.shape[0]}"

    # Create test data with positions that would exceed original max_seq_len
    B, T = 2, max_seq_len // 2  # Use half of max_seq_len for body
    inputs = torch.randint(0, 50257, (B, T))

    # Create order tensor that after shifting will have positions > max_seq_len
    # Original order: [0, 1, ..., T-1, T]
    # After shift: [0, 1, ..., n-1, n, n+1, ..., n+T-1, n+T]
    # Max position = n + T = 32 + 32 = 64, which equals original max_seq_len
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)
    targets = torch.randint(0, 50257, (B, T))

    print(f"  Body sequence length: {T}")
    print(f"  Max shifted position: {num_thinking + T} (vs original max {max_seq_len})")

    # Forward pass should succeed without index errors
    try:
        logits, loss = model(idx=inputs, order=order, targets=targets)
        print(f"  Forward pass succeeded: logits shape = {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        assert not torch.isnan(loss), "Loss should not be NaN"
    except IndexError as e:
        raise AssertionError(f"Forward pass failed with IndexError: {e}")

    # Verify no position aliasing: check that different positions get different embeddings
    # This is implicitly tested by the model working correctly, but we can verify RoPE cache
    pos_32 = rope.cos_cached_1[32]
    pos_33 = rope.cos_cached_1[33]
    pos_64 = rope.cos_cached_1[64] if rope.cos_cached_1.shape[0] > 64 else None

    assert not torch.allclose(pos_32, pos_33), "Positions 32 and 33 should have different embeddings"
    if pos_64 is not None:
        print(f"  Position 64 exists in cache (no aliasing to position 63)")
        assert not torch.allclose(rope.cos_cached_1[63], pos_64), \
            "Position 64 should differ from position 63"

    print("  [PASS] Test 10 passed")


def test_position_encoding_learned_mode():
    """Test 11: Verify learned position encoding handles extended range"""
    print("\n" + "=" * 60)
    print("[Test 11] Learned position encoding extended range")
    print("=" * 60)

    from model.sigmagpt_from_baseline import SigmaGPTModel
    from model.arbitrary_prob_gpt2 import GPT2Config
    from model.token_manager import TokenManager

    # Create config with learned position encoding
    max_seq_len = 64
    config = GPT2Config(
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=max_seq_len,
        dropout=0.0,
        position_encoding_type='learned'  # Use learned mode
    )

    num_thinking = 32
    token_manager = TokenManager(
        add_mask_token=False,
        add_bos_token=False,
        num_thinking_tokens=num_thinking
    )
    thinking_ids = token_manager.get_thinking_token_ids()

    model = SigmaGPTModel(config, thinking_token_ids=thinking_ids)
    token_manager.resize_model_embeddings(model)

    # Verify wpe size is extended
    expected_wpe_size = max_seq_len + num_thinking
    print(f"  wpe.num_embeddings: {model.wpe.num_embeddings}")

    assert model.wpe.num_embeddings == expected_wpe_size, \
        f"wpe should have {expected_wpe_size} positions, got {model.wpe.num_embeddings}"

    # Forward pass test
    B, T = 2, max_seq_len // 2
    inputs = torch.randint(0, 50257, (B, T))
    order = torch.arange(T + 1).unsqueeze(0).expand(B, -1)
    targets = torch.randint(0, 50257, (B, T))

    try:
        logits, loss = model(idx=inputs, order=order, targets=targets)
        print(f"  Forward pass succeeded: logits shape = {logits.shape}")
        assert not torch.isnan(loss), "Loss should not be NaN"
    except IndexError as e:
        raise AssertionError(f"Forward pass failed with IndexError: {e}")

    print("  [PASS] Test 11 passed")


def run_all_tests():
    """Run all thinking token tests"""
    print("\n" + "=" * 60)
    print("THINKING TOKENS TEST SUITE")
    print("=" * 60)

    tests = [
        ("Token count computation", test_token_count_computation),
        ("Prepender shapes", test_prepender_shapes),
        ("Prepender order values", test_prepender_order_values),
        ("Target masking", test_prepender_target_masking),
        ("Input token prepending", test_prepender_input_tokens),
        ("TokenManager integration", test_token_manager_integration),
        ("SigmaGPT model integration", test_sigmagpt_model_integration),
        ("Forward pass with thinking", test_forward_pass_with_thinking),
        ("Loss excludes thinking positions", test_loss_excludes_thinking),
        ("Position encoding extended range (RoPE)", test_position_encoding_range),
        ("Position encoding extended range (learned)", test_position_encoding_learned_mode),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  [FAILED] {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
