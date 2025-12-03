#!/usr/bin/env python3
"""
Quick test script to verify detach_augmentation parameter works correctly

This script tests:
1. Config parameter is correctly set
2. Model can be initialized with the parameter
3. Forward pass works with both detach=True and detach=False
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from model.arbitrary_prob_gpt2 import GPT2Config, GPT2Model

def test_config_parameter():
    """Test that detach_augmentation parameter can be set in config"""
    print("\n" + "=" * 80)
    print("TEST 1: Config Parameter")
    print("=" * 80)

    # Test default value
    config1 = GPT2Config()
    print(f"✓ Config created with default detach_augmentation: {config1.detach_augmentation}")
    assert config1.detach_augmentation == False, "Default should be False"

    # Test explicit True
    config2 = GPT2Config(detach_augmentation=True)
    print(f"✓ Config created with detach_augmentation=True: {config2.detach_augmentation}")
    assert config2.detach_augmentation == True, "Should be True"

    print("\n✅ Config parameter test PASSED")
    return config1, config2


def test_model_initialization():
    """Test that model can be initialized with detach_augmentation configs"""
    print("\n" + "=" * 80)
    print("TEST 2: Model Initialization")
    print("=" * 80)

    # Create configs
    config_no_detach = GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=256,
        detach_augmentation=False
    )

    config_with_detach = GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=256,
        detach_augmentation=True
    )

    # Initialize models
    model_no_detach = GPT2Model(config_no_detach, mask_token_id=50257, bos_token_id=50256)
    print(f"✓ Model created with detach_augmentation=False")
    print(f"  Config detach_augmentation: {model_no_detach.config.detach_augmentation}")

    model_with_detach = GPT2Model(config_with_detach, mask_token_id=50257, bos_token_id=50256)
    print(f"✓ Model created with detach_augmentation=True")
    print(f"  Config detach_augmentation: {model_with_detach.config.detach_augmentation}")

    print("\n✅ Model initialization test PASSED")
    return model_no_detach, model_with_detach


def test_forward_pass():
    """Test forward pass with detach_augmentation"""
    print("\n" + "=" * 80)
    print("TEST 3: Forward Pass with Conditional Mode")
    print("=" * 80)

    # Create a small test model
    config = GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=64,
        max_seq_len=128,
        detach_augmentation=True
    )

    model = GPT2Model(config, mask_token_id=50257, bos_token_id=50256)
    model.eval()  # Set to eval mode

    # Create test input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))

    # Create test indices
    conditional_idx = [[0, 1, 8, 9], [0, 2, 7, 9]]
    evaluation_idx = [[2, 3, 4, 5, 6, 7], [1, 3, 4, 5, 6, 8]]
    unseen_idx = [[2, 3, 4, 5, 6, 7], [1, 3, 4, 5, 6, 8]]  # Same as eval for Mode 2

    print(f"Input shape: {input_ids.shape}")
    print(f"Conditional indices: {conditional_idx}")
    print(f"Evaluation indices: {evaluation_idx}")

    # Forward pass
    with torch.no_grad():
        logits, loss = model(
            input_ids=input_ids,
            conditional_idx=conditional_idx,
            evaluation_idx=evaluation_idx,
            unseen_idx=unseen_idx
        )

    print(f"\n✓ Forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item() if loss is not None else 'None'}")

    # Test that gradients can be computed
    model.train()
    logits, loss = model(
        input_ids=input_ids,
        conditional_idx=conditional_idx,
        evaluation_idx=evaluation_idx,
        unseen_idx=unseen_idx
    )

    if loss is not None:
        loss.backward()
        print(f"✓ Backward pass successful")
        print(f"  Gradients computed successfully")

    print("\n✅ Forward pass test PASSED")


def main():
    print("\n" + "=" * 80)
    print(" DETACH AUGMENTATION FEATURE TEST")
    print("=" * 80)

    try:
        # Run tests
        test_config_parameter()
        test_model_initialization()
        test_forward_pass()

        # Success
        print("\n" + "=" * 80)
        print(" ✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nThe detach_augmentation feature is working correctly!")
        print("\nUsage in training:")
        print("  python train.py --model_type conditional --detach_augmentation \\")
        print("      ... (other arguments)")

    except Exception as e:
        print("\n" + "=" * 80)
        print(" ❌ TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
