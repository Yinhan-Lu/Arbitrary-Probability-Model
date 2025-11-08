"""
Test the new concatenate + chunk pipeline

Verifies:
1. No placeholder text in chunks
2. All chunks have correct length
3. EOS tokens separate documents
4. Augmenter works with chunks
5. Collate functions work correctly

Run from project root: python tests/test_chunk_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import GPT2Tokenizer
from train.dataset import WikipediaDataset
from train.augmentation import ConditionalAugmenter
from train.dataset import create_augment_collate_fn
from model.token_manager import TokenManager


def test_no_placeholder_text():
    """Test that no placeholder text exists in chunks"""
    print("\n" + "=" * 80)
    print("[Test 1] Verifying no placeholder text in chunks")
    print("=" * 80)

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset with small sample
    dataset = WikipediaDataset(
        tokenizer=tokenizer,
        max_length=512,  # Smaller for faster testing
        split="train",
        num_samples=100,  # Only process 100 documents
        dataset_name="wikitext",
        dataset_config="wikitext-103-raw-v1"
    )

    print(f"Created dataset with {len(dataset)} chunks")

    # Check first 50 chunks for placeholder
    placeholder_found = False
    for i in range(min(50, len(dataset))):
        sample = dataset[i]
        text = tokenizer.decode(sample['input_ids'])

        if "placeholder" in text.lower():
            print(f"❌ FAILED: Found placeholder in chunk {i}")
            print(f"Text preview: {text[:200]}")
            placeholder_found = True
            break

    if not placeholder_found:
        print("✓ Test 1 PASSED: No placeholder text found in 50 chunks")
    else:
        raise AssertionError("Placeholder text found in chunks!")


def test_chunk_length():
    """Test that all chunks have exactly max_length"""
    print("\n" + "=" * 80)
    print("[Test 2] Verifying all chunks have correct length")
    print("=" * 80)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = 512
    dataset = WikipediaDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        num_samples=100
    )

    # Check all chunks
    wrong_length_count = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        actual_length = len(sample['input_ids'])

        if actual_length != max_length:
            print(f"❌ Chunk {i} has wrong length: {actual_length} (expected {max_length})")
            wrong_length_count += 1

        if wrong_length_count > 5:  # Stop after finding 5 wrong lengths
            break

    if wrong_length_count == 0:
        print(f"✓ Test 2 PASSED: All {len(dataset)} chunks have length {max_length}")
    else:
        raise AssertionError(f"Found {wrong_length_count} chunks with wrong length!")


def test_eos_separation():
    """Test that documents are separated by EOS tokens"""
    print("\n" + "=" * 80)
    print("[Test 3] Verifying EOS tokens separate documents")
    print("=" * 80)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = WikipediaDataset(
        tokenizer=tokenizer,
        max_length=512,
        split="train",
        num_samples=100
    )

    # Count EOS tokens in first 10 chunks
    eos_count = 0
    eos_token_id = tokenizer.eos_token_id

    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        eos_in_chunk = (sample['input_ids'] == eos_token_id).sum().item()
        eos_count += eos_in_chunk

    print(f"Found {eos_count} EOS tokens in 10 chunks")
    print(f"Average: {eos_count / 10:.1f} EOS tokens per chunk")

    if eos_count > 0:
        print("✓ Test 3 PASSED: EOS tokens present (document boundaries preserved)")
    else:
        raise AssertionError("No EOS tokens found - documents not separated!")


def test_attention_mask():
    """Test that attention masks are all 1s (no padding)"""
    print("\n" + "=" * 80)
    print("[Test 4] Verifying attention masks (no padding in chunks)")
    print("=" * 80)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = WikipediaDataset(
        tokenizer=tokenizer,
        max_length=512,
        split="train",
        num_samples=50
    )

    # Check all chunks
    for i in range(len(dataset)):
        sample = dataset[i]
        attention_mask = sample['attention_mask']

        # All should be 1 (no padding)
        if not (attention_mask == 1).all():
            zeros = (attention_mask == 0).sum().item()
            print(f"❌ Chunk {i} has {zeros} padding tokens")
            raise AssertionError("Found padding in chunks!")

    print(f"✓ Test 4 PASSED: All {len(dataset)} chunks have no padding")


def test_augmenter_compatibility():
    """Test that ConditionalAugmenter works with chunks"""
    print("\n" + "=" * 80)
    print("[Test 5] Verifying ConditionalAugmenter compatibility")
    print("=" * 80)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = WikipediaDataset(
        tokenizer=tokenizer,
        max_length=512,
        split="train",
        num_samples=50
    )

    # Create token manager and augmenter
    token_manager = TokenManager(tokenizer)
    mask_token_id = token_manager.mask_token_id
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    from functools import partial
    from train.augmentation import (
        uniform_num_conditioning_distribution,
        uniform_num_evaluation_distribution
    )
    from train.blockwise_sampling import (
        uniform_num_blocks_distribution,
        uniform_block_sizes_distribution
    )

    num_cond_dist = partial(
        uniform_num_conditioning_distribution,
        conditioning_percentage_range=(0.2, 0.4)
    )

    num_eval_dist = partial(
        uniform_num_evaluation_distribution,
        evaluation_percentage_range=(0.2, 0.4)
    )

    # uniform_num_blocks_distribution doesn't need configuration
    num_blocks_dist = uniform_num_blocks_distribution

    # uniform_block_sizes_distribution doesn't need configuration
    block_sizes_dist = uniform_block_sizes_distribution

    augmenter = ConditionalAugmenter(
        mask_token_id=mask_token_id,
        bos_token_id=bos_token_id,
        max_seq_len=512,
        cond_pct_max=0.4,
        tokenizer_pad_token_id=tokenizer.pad_token_id,
        num_conditioning_distribution=num_cond_dist,
        num_evaluation_distribution=num_eval_dist,
        num_blocks_distribution=num_blocks_dist,
        num_eval_blocks_distribution=num_blocks_dist,
        block_sizes_distribution=block_sizes_dist,
        eval_block_sizes_distribution=block_sizes_dist,
        conditioning_sampling='blockwise',
        evaluation_sampling='blockwise'
    )

    # Test augmentation on first chunk
    sample = dataset[0]
    input_ids = sample['input_ids']
    attention_mask = sample['attention_mask']

    # Extract valid positions
    valid_positions = [i for i in range(len(input_ids)) if attention_mask[i] == 1]

    print(f"Chunk length: {len(input_ids)}")
    print(f"Valid positions: {len(valid_positions)}")

    # Augment
    try:
        aug_result = augmenter.augment_sequence(
            input_ids,
            device='cpu',
            valid_positions=valid_positions
        )

        print(f"Augmented sequence length: {aug_result['aug_input_ids'].size(0)}")
        print(f"Conditioning positions: {len([i for i in range(len(aug_result['labels'])) if aug_result['labels'][i] == -100])}")
        print(f"Evaluation positions: {len([i for i in range(len(aug_result['labels'])) if aug_result['labels'][i] != -100])}")

        print("✓ Test 5 PASSED: Augmenter works with chunks")

    except Exception as e:
        print(f"❌ FAILED: Augmentation error: {e}")
        raise


def test_collate_function():
    """Test that collate function works with chunked data"""
    print("\n" + "=" * 80)
    print("[Test 6] Verifying collate function works with chunks")
    print("=" * 80)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = WikipediaDataset(
        tokenizer=tokenizer,
        max_length=512,
        split="train",
        num_samples=50
    )

    # Create augmenter
    token_manager = TokenManager(tokenizer)
    mask_token_id = token_manager.mask_token_id
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    from functools import partial
    from train.augmentation import (
        uniform_num_conditioning_distribution,
        uniform_num_evaluation_distribution
    )
    from train.blockwise_sampling import (
        uniform_num_blocks_distribution,
        uniform_block_sizes_distribution
    )

    num_cond_dist = partial(
        uniform_num_conditioning_distribution,
        conditioning_percentage_range=(0.2, 0.4)
    )

    num_eval_dist = partial(
        uniform_num_evaluation_distribution,
        evaluation_percentage_range=(0.2, 0.4)
    )

    # uniform_num_blocks_distribution doesn't need configuration
    num_blocks_dist = uniform_num_blocks_distribution

    # uniform_block_sizes_distribution doesn't need configuration
    block_sizes_dist = uniform_block_sizes_distribution

    augmenter = ConditionalAugmenter(
        mask_token_id=mask_token_id,
        bos_token_id=bos_token_id,
        max_seq_len=512,
        cond_pct_max=0.4,
        tokenizer_pad_token_id=tokenizer.pad_token_id,
        num_conditioning_distribution=num_cond_dist,
        num_evaluation_distribution=num_eval_dist,
        num_blocks_distribution=num_blocks_dist,
        num_eval_blocks_distribution=num_blocks_dist,
        block_sizes_distribution=block_sizes_dist,
        eval_block_sizes_distribution=block_sizes_dist,
        conditioning_sampling='blockwise',
        evaluation_sampling='blockwise'
    )

    # Create collate function
    collate_fn = create_augment_collate_fn(augmenter, device='cpu')

    # Create a batch
    batch = [dataset[i] for i in range(4)]

    # Collate
    try:
        collated = collate_fn(batch)

        print(f"Batch input_ids shape: {collated['input_ids'].shape}")
        print(f"Batch position_ids shape: {collated['position_ids'].shape}")
        print(f"Batch labels shape: {collated['labels'].shape}")
        print(f"Batch attention_mask shape: {collated['attention_mask'].shape}")

        print("✓ Test 6 PASSED: Collate function works correctly")

    except Exception as e:
        print(f"❌ FAILED: Collate error: {e}")
        raise


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("CONCATENATE + CHUNK PIPELINE TESTS")
    print("=" * 80)

    try:
        # Run all tests
        test_no_placeholder_text()
        test_chunk_length()
        test_eos_separation()
        test_attention_mask()
        test_augmenter_compatibility()
        test_collate_function()

        # Summary
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nThe new concatenate + chunk pipeline is working correctly:")
        print("  ✓ No placeholder text")
        print("  ✓ All chunks have correct length")
        print("  ✓ Documents separated by EOS tokens")
        print("  ✓ No padding in chunks")
        print("  ✓ Augmenter compatible")
        print("  ✓ Collate function compatible")
        print("\nYou can now train with the new pipeline!")

    except Exception as e:
        print("\n" + "=" * 80)
        print("TESTS FAILED ❌")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
