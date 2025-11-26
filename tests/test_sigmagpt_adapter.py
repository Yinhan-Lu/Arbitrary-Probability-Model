"""
Unit tests for SigmaGPTDataAdapter

Tests verify correct conversion from ConditionalAugmenter format to Sigma GPT format.

Run from project root: python tests/test_sigmagpt_adapter.py
Expected runtime: ~30 seconds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from train.sigmagpt_adapter import SigmaGPTDataAdapter, create_sigmagpt_adapter


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_adapter_initialization():
    """Test 1: Adapter initialization"""
    print_section("Test 1: Adapter Initialization")

    # Test fair mode
    adapter_fair = SigmaGPTDataAdapter(mode='fair')
    assert adapter_fair.mode == 'fair', "Fair mode not set correctly"

    # Test full mode
    adapter_full = SigmaGPTDataAdapter(mode='full')
    assert adapter_full.mode == 'full', "Full mode not set correctly"

    # Test invalid mode
    try:
        SigmaGPTDataAdapter(mode='invalid')
        assert False, "Should raise ValueError for invalid mode"
    except ValueError:
        pass  # Expected

    # Test factory function
    adapter = create_sigmagpt_adapter(mode='fair')
    assert isinstance(adapter, SigmaGPTDataAdapter), "Factory function failed"
    assert adapter.mode == 'fair', "Factory function mode not set"

    print("✓ Adapter initialization works correctly")
    print(f"  - Fair mode: ✓")
    print(f"  - Full mode: ✓")
    print(f"  - Invalid mode raises error: ✓")
    print(f"  - Factory function: ✓")


def test_convert_sequence_fair():
    """Test 2: Convert single sequence (fair mode)"""
    print_section("Test 2: Convert Sequence (Fair Mode)")

    adapter = SigmaGPTDataAdapter(mode='fair')

    # Create mock augmenter result
    aug_result = {
        'conditioning_indices': [0, 1, 2],
        'evaluation_indices': [3, 4],
        # Other fields not needed for adapter
    }

    # Original tokens
    original_tokens = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Convert
    result = adapter.convert_sequence(aug_result, original_tokens)

    # Expected:
    # inputs: [10, 20, 30, 40, 50] (tokens at positions 0, 1, 2, 3, 4)
    # order: [0, 1, 2, 3, 4, 10] (cond: 0-2, eval: 3-4, seq_len: 10)
    # targets (fair): [-1, -1, -1, 50, -1] (only eval position 3 computes loss)

    expected_inputs = torch.tensor([10, 20, 30, 40, 50])
    expected_order = torch.tensor([0, 1, 2, 3, 4, 10])
    expected_targets = torch.tensor([-1, -1, -1, 50, -1])

    assert torch.equal(result['inputs'], expected_inputs), \
        f"Inputs mismatch:\nGot:      {result['inputs']}\nExpected: {expected_inputs}"

    assert torch.equal(result['order'], expected_order), \
        f"Order mismatch:\nGot:      {result['order']}\nExpected: {expected_order}"

    assert torch.equal(result['targets'], expected_targets), \
        f"Targets mismatch:\nGot:      {result['targets']}\nExpected: {expected_targets}"

    print("✓ Single sequence conversion (fair) works correctly")
    print(f"  - Inputs: {result['inputs'].tolist()}")
    print(f"  - Order:  {result['order'].tolist()}")
    print(f"  - Targets: {result['targets'].tolist()}")


def test_convert_sequence_full():
    """Test 3: Convert single sequence (full mode)"""
    print_section("Test 3: Convert Sequence (Full Mode)")

    adapter = SigmaGPTDataAdapter(mode='full')

    aug_result = {
        'conditioning_indices': [0, 1, 2],
        'evaluation_indices': [3, 4],
    }

    original_tokens = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    result = adapter.convert_sequence(aug_result, original_tokens)

    # Expected (full mode):
    # inputs: [10, 20, 30, 40, 50]
    # order: [0, 1, 2, 3, 4, 10]
    # targets (full): [20, 30, 40, 50, -1] (all positions compute loss except last)

    expected_inputs = torch.tensor([10, 20, 30, 40, 50])
    expected_order = torch.tensor([0, 1, 2, 3, 4, 10])
    expected_targets = torch.tensor([20, 30, 40, 50, -1])

    assert torch.equal(result['inputs'], expected_inputs), \
        f"Inputs mismatch:\nGot:      {result['inputs']}\nExpected: {expected_inputs}"

    assert torch.equal(result['order'], expected_order), \
        f"Order mismatch:\nGot:      {result['order']}\nExpected: {expected_order}"

    assert torch.equal(result['targets'], expected_targets), \
        f"Targets mismatch:\nGot:      {result['targets']}\nExpected: {expected_targets}"

    print("✓ Single sequence conversion (full) works correctly")
    print(f"  - Inputs: {result['inputs'].tolist()}")
    print(f"  - Order:  {result['order'].tolist()}")
    print(f"  - Targets: {result['targets'].tolist()}")


def test_convert_batch():
    """Test 4: Convert batch with multiple sequences"""
    print_section("Test 4: Convert Batch")

    adapter = SigmaGPTDataAdapter(mode='fair')

    # Create batch with different sizes
    aug_batch = [
        {'conditioning_indices': [0, 1], 'evaluation_indices': [2, 3]},  # 4 total
        {'conditioning_indices': [0, 1, 2], 'evaluation_indices': [3]},  # 4 total
        {'conditioning_indices': [0], 'evaluation_indices': [1, 2, 3]},  # 4 total
    ]

    original_tokens = torch.tensor([
        [10, 20, 30, 40, 50],
        [11, 21, 31, 41, 51],
        [12, 22, 32, 42, 52],
    ])

    result = adapter.convert_batch(aug_batch, original_tokens)

    # All have same total size (4), so no padding needed
    B, T = result['inputs'].shape
    assert B == 3, f"Batch size should be 3, got {B}"
    assert T == 4, f"Sequence length should be 4, got {T}"

    # Order shape should be (B, T+1)
    assert result['order'].shape == (3, 5), \
        f"Order shape should be (3, 5), got {result['order'].shape}"

    # Verify all batches end with seq_len=5
    assert (result['order'][:, -1] == 5).all(), \
        "All orders should end with seq_len=5"

    # Verify batch 0
    expected_inputs_0 = torch.tensor([10, 20, 30, 40])
    assert torch.equal(result['inputs'][0], expected_inputs_0), \
        f"Batch 0 inputs failed:\nGot: {result['inputs'][0]}\nExpected: {expected_inputs_0}"

    print("✓ Batch conversion works correctly")
    print(f"  - Batch shape: {result['inputs'].shape}")
    print(f"  - Order shape: {result['order'].shape}")
    print(f"  - Targets shape: {result['targets'].shape}")


def test_convert_batch_with_padding():
    """Test 5: Convert batch with different sizes (requires padding)"""
    print_section("Test 5: Batch Conversion with Padding")

    adapter = SigmaGPTDataAdapter(mode='fair')

    # Create batch with different sizes
    aug_batch = [
        {'conditioning_indices': [0, 1, 2], 'evaluation_indices': [3, 4, 5]},  # 6 total
        {'conditioning_indices': [0, 1], 'evaluation_indices': [2]},           # 3 total
        {'conditioning_indices': [0], 'evaluation_indices': [1, 2, 3, 4]},     # 5 total
    ]

    original_tokens = torch.tensor([
        [10, 20, 30, 40, 50, 60, 70, 80],
        [11, 21, 31, 41, 51, 61, 71, 81],
        [12, 22, 32, 42, 52, 62, 72, 82],
    ])

    result = adapter.convert_batch(aug_batch, original_tokens)

    # Max size is 6, so should be padded to 6
    B, T = result['inputs'].shape
    assert B == 3, f"Batch size should be 3, got {B}"
    assert T == 6, f"Max sequence length should be 6, got {T}"

    # Verify padding in batch 1 (size 3, needs 3 padding tokens)
    # Last 3 positions should be padded
    assert result['inputs'][1, 3:].sum() == 0, \
        f"Batch 1 should have 0-padded inputs, got {result['inputs'][1, 3:]}"

    # Padded targets should be -1
    assert (result['targets'][1, 3:] == -1).all(), \
        f"Batch 1 should have -1 padded targets, got {result['targets'][1, 3:]}"

    # Padded order positions should be seq_len (8)
    seq_len = original_tokens.shape[1]
    assert (result['order'][1, 3:] == seq_len).all(), \
        f"Batch 1 should have seq_len-padded order, got {result['order'][1, 3:]}"

    print("✓ Batch conversion with padding works correctly")
    print(f"  - Batch shape: {result['inputs'].shape}")
    print(f"  - Inputs padding: ✓ (zeros)")
    print(f"  - Targets padding: ✓ (-1)")
    print(f"  - Order padding: ✓ (seq_len)")


def test_get_stats():
    """Test 6: Get statistics from augmented batch"""
    print_section("Test 6: Get Statistics")

    adapter_fair = SigmaGPTDataAdapter(mode='fair')
    adapter_full = SigmaGPTDataAdapter(mode='full')

    aug_batch = [
        {'conditioning_indices': [0, 1, 2], 'evaluation_indices': [3, 4]},     # 3 cond, 2 eval
        {'conditioning_indices': [0, 1], 'evaluation_indices': [2, 3, 4]},     # 2 cond, 3 eval
        {'conditioning_indices': [0, 1, 2, 3], 'evaluation_indices': [4, 5]},  # 4 cond, 2 eval
    ]

    # Fair mode stats
    stats_fair = adapter_fair.get_stats(aug_batch)

    expected_avg_cond = (3 + 2 + 4) / 3  # 3.0
    expected_avg_eval = (2 + 3 + 2) / 3  # 2.33...
    expected_avg_total = (5 + 5 + 6) / 3  # 5.33...

    assert abs(stats_fair['avg_cond_size'] - expected_avg_cond) < 0.01, \
        f"avg_cond_size mismatch: {stats_fair['avg_cond_size']}"

    assert abs(stats_fair['avg_eval_size'] - expected_avg_eval) < 0.01, \
        f"avg_eval_size mismatch: {stats_fair['avg_eval_size']}"

    # Fair mode: learning efficiency = eval_pct
    expected_eval_pct = expected_avg_eval / expected_avg_total
    assert abs(stats_fair['learning_efficiency'] - expected_eval_pct) < 0.01, \
        f"Fair mode learning efficiency should be ~{expected_eval_pct:.2f}, got {stats_fair['learning_efficiency']}"

    # Full mode stats
    stats_full = adapter_full.get_stats(aug_batch)

    # Full mode: learning efficiency = 1.0 (all positions)
    assert abs(stats_full['learning_efficiency'] - 1.0) < 0.01, \
        f"Full mode learning efficiency should be 1.0, got {stats_full['learning_efficiency']}"

    print("✓ Statistics calculation works correctly")
    print(f"  - Average cond size: {stats_fair['avg_cond_size']:.2f}")
    print(f"  - Average eval size: {stats_fair['avg_eval_size']:.2f}")
    print(f"  - Average total size: {stats_fair['avg_total_size']:.2f}")
    print(f"  - Fair mode learning efficiency: {stats_fair['learning_efficiency']:.2%}")
    print(f"  - Full mode learning efficiency: {stats_full['learning_efficiency']:.2%}")


def test_non_contiguous_indices():
    """Test 7: Handle non-contiguous indices"""
    print_section("Test 7: Non-Contiguous Indices")

    adapter = SigmaGPTDataAdapter(mode='fair')

    # Non-contiguous indices
    aug_result = {
        'conditioning_indices': [0, 5, 10],  # Non-contiguous
        'evaluation_indices': [2, 7],         # Non-contiguous
    }

    original_tokens = torch.arange(0, 20)  # [0, 1, 2, ..., 19]

    result = adapter.convert_sequence(aug_result, original_tokens)

    # Expected order: [0, 5, 10, 2, 7, 20]
    expected_order = torch.tensor([0, 5, 10, 2, 7, 20])

    assert torch.equal(result['order'], expected_order), \
        f"Order mismatch:\nGot:      {result['order']}\nExpected: {expected_order}"

    # Expected inputs: [0, 5, 10, 2, 7]
    expected_inputs = torch.tensor([0, 5, 10, 2, 7])

    assert torch.equal(result['inputs'], expected_inputs), \
        f"Inputs mismatch:\nGot:      {result['inputs']}\nExpected: {expected_inputs}"

    print("✓ Non-contiguous indices handled correctly")
    print(f"  - Order: {result['order'].tolist()}")
    print(f"  - Inputs: {result['inputs'].tolist()}")


def test_device_handling():
    """Test 8: Verify device handling"""
    print_section("Test 8: Device Handling")

    adapter = SigmaGPTDataAdapter(mode='fair')

    aug_batch = [
        {'conditioning_indices': [0, 1], 'evaluation_indices': [2]},
    ]

    # Test with CPU
    original_tokens_cpu = torch.tensor([[10, 20, 30, 40]])

    result_cpu = adapter.convert_batch(aug_batch, original_tokens_cpu)

    assert result_cpu['inputs'].device.type == 'cpu', \
        f"CPU device not preserved: {result_cpu['inputs'].device}"

    # Test with CUDA (if available)
    if torch.cuda.is_available():
        original_tokens_cuda = original_tokens_cpu.cuda()
        result_cuda = adapter.convert_batch(aug_batch, original_tokens_cuda)

        assert result_cuda['inputs'].device.type == 'cuda', \
            f"CUDA device not preserved: {result_cuda['inputs'].device}"

        print("✓ Device handling works correctly (CPU + CUDA)")
    else:
        print("✓ Device handling works correctly (CPU only)")


def run_all_tests():
    """Run all adapter tests"""
    print("\n" + "=" * 80)
    print("  SIGMA GPT DATA ADAPTER TESTS")
    print("=" * 80)

    tests = [
        ("Adapter Initialization", test_adapter_initialization),
        ("Convert Sequence (Fair)", test_convert_sequence_fair),
        ("Convert Sequence (Full)", test_convert_sequence_full),
        ("Convert Batch", test_convert_batch),
        ("Batch with Padding", test_convert_batch_with_padding),
        ("Get Statistics", test_get_stats),
        ("Non-Contiguous Indices", test_non_contiguous_indices),
        ("Device Handling", test_device_handling),
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
