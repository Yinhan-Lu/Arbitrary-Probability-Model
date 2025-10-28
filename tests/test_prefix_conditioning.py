"""
Unit Tests for Prefix Conditioning Implementation

Tests the core functionality of prefix conditioning for arbitrary conditional
probability modeling, including:
- Attention mask structure
- Labels positioning (only evaluation set)
- Position IDs correctness
- Body masking strategy (only unknown set)
"""

import sys
from pathlib import Path
import torch
import unittest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.augmentation import ConditionalAugmenter
from train.mask_utils import create_prefix_conditional_mask


class TestPrefixConditioning(unittest.TestCase):
    """Test suite for prefix conditioning implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.mask_token_id = 50257  # [M] token
        self.bos_token_id = 50256   # [BOS] token
        self.pad_token_id = -100

        # Create augmenter with fixed ratios for testing
        self.augmenter = ConditionalAugmenter(
            mask_token_id=self.mask_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            conditioning_ratio=0.4,
            evaluation_ratio=0.3,
            min_conditioning=1,
            min_evaluation=1,
            include_bos=True,
            conditioning_sampling='random'
        )

    def test_mask_structure(self):
        """Test 1: Verify attention mask structure"""
        print("\n" + "=" * 80)
        print("Test 1: Attention Mask Structure")
        print("=" * 80)

        N_cond = 3
        N_seq = 6

        mask = create_prefix_conditional_mask(N_cond, N_seq, device='cpu')

        # Check shape
        expected_shape = (N_cond + N_seq, N_cond + N_seq)
        self.assertEqual(mask.shape, expected_shape,
                        f"Mask shape should be {expected_shape}")

        # Check conditioning rows (should see everything)
        for i in range(N_cond):
            self.assertTrue(torch.all(mask[i, :] == 1),
                           f"Conditioning row {i} should see all positions")

        # Check sequence rows (should see cond + causal seq)
        for i in range(N_seq):
            seq_row_idx = N_cond + i

            # Should see all conditioning positions
            self.assertTrue(torch.all(mask[seq_row_idx, :N_cond] == 1),
                           f"Sequence row {i} should see all conditioning positions")

            # Within sequence part, should be causal
            for j in range(N_seq):
                seq_col_idx = N_cond + j
                if j <= i:
                    # Can see previous and current positions
                    self.assertEqual(mask[seq_row_idx, seq_col_idx].item(), 1,
                                   f"Causal mask violated at seq position ({i}, {j})")
                else:
                    # Cannot see future positions
                    self.assertEqual(mask[seq_row_idx, seq_col_idx].item(), 0,
                                   f"Causal mask violated at seq position ({i}, {j})")

        print(f"✓ Mask structure correct: {mask.shape}")
        print(f"  - Conditioning rows: fully visible")
        print(f"  - Sequence rows: cond visible + causal")

    def test_labels_only_evaluation(self):
        """Test 2: Verify labels are only set for evaluation positions"""
        print("\n" + "=" * 80)
        print("Test 2: Labels Only for Evaluation Set")
        print("=" * 80)

        # Create a simple test sequence
        original_seq = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])
        seq_len = len(original_seq)

        # Augment sequence
        result = self.augmenter.augment_sequence(original_seq)

        aug_input_ids = result["aug_input_ids"]
        labels = result["labels"]
        cond_idx = result["conditioning_indices"]
        eval_idx = result["evaluation_indices"]
        N_cond = result["N_cond"]

        # Count non-padding labels
        non_pad_labels = (labels != self.pad_token_id).sum().item()
        self.assertEqual(non_pad_labels, len(eval_idx),
                        f"Should have exactly {len(eval_idx)} non-padding labels")

        # Verify labels are only set at evaluation positions
        for eval_pos in eval_idx:
            # In augmented sequence: Cond (N_cond) + BOS (1) + Body (eval_pos)
            new_pos = N_cond + 1 + eval_pos

            # Label should match original token
            expected_label = original_seq[eval_pos].item()
            actual_label = labels[new_pos].item()

            self.assertEqual(actual_label, expected_label,
                           f"Label at position {new_pos} should be {expected_label}")
            self.assertNotEqual(actual_label, self.pad_token_id,
                              f"Evaluation position {new_pos} should not be padding")

        # Verify conditioning positions have padding labels
        for i in range(N_cond):
            self.assertEqual(labels[i].item(), self.pad_token_id,
                           f"Conditioning position {i} should have padding label")

        # Verify BOS has padding label
        self.assertEqual(labels[N_cond].item(), self.pad_token_id,
                        "BOS position should have padding label")

        print(f"✓ Labels correctly set only for evaluation positions")
        print(f"  - Evaluation indices: {sorted(eval_idx)}")
        print(f"  - Non-padding labels: {non_pad_labels}")
        print(f"  - Total labels: {len(labels)}")

    def test_position_ids(self):
        """Test 3: Verify position IDs use original positions"""
        print("\n" + "=" * 80)
        print("Test 3: Position IDs Correctness")
        print("=" * 80)

        # Create a simple test sequence
        original_seq = torch.tensor([100, 200, 300, 400, 500])
        seq_len = len(original_seq)

        # Augment sequence
        result = self.augmenter.augment_sequence(original_seq)

        position_ids = result["position_ids"]
        cond_idx = result["conditioning_indices"]
        N_cond = result["N_cond"]
        N_seq = result["N_seq"]

        # Check total length
        expected_len = N_cond + N_seq
        self.assertEqual(len(position_ids), expected_len,
                        f"Position IDs length should be {expected_len}")

        # Check conditioning positions (should use original positions)
        sorted_cond_idx = sorted(cond_idx)
        for i, orig_pos in enumerate(sorted_cond_idx):
            # Position encoding should be original position + 1 (1-indexed)
            expected_pos = orig_pos + 1
            actual_pos = position_ids[i].item()
            self.assertEqual(actual_pos, expected_pos,
                           f"Conditioning token at position {i} should have position ID {expected_pos}")

        # Check BOS position (should be 0)
        bos_pos_idx = N_cond
        self.assertEqual(position_ids[bos_pos_idx].item(), 0,
                        "BOS token should have position ID 0")

        # Check body positions (should use 1, 2, 3, ..., seq_len)
        for i in range(seq_len):
            body_pos_idx = N_cond + 1 + i
            expected_pos = i + 1  # 1-indexed
            actual_pos = position_ids[body_pos_idx].item()
            self.assertEqual(actual_pos, expected_pos,
                           f"Body token at index {i} should have position ID {expected_pos}")

        print(f"✓ Position IDs correctly use original positions")
        print(f"  - Conditioning positions: {position_ids[:N_cond].tolist()}")
        print(f"  - BOS position: {position_ids[N_cond].item()}")
        print(f"  - Body positions: {position_ids[N_cond+1:].tolist()}")

    def test_body_masking(self):
        """Test 4: Verify only unknown set is masked in body"""
        print("\n" + "=" * 80)
        print("Test 4: Body Masking Strategy")
        print("=" * 80)

        # Create a test sequence with known values
        original_seq = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        seq_len = len(original_seq)

        # Augment sequence
        result = self.augmenter.augment_sequence(original_seq)

        aug_input_ids = result["aug_input_ids"]
        cond_idx = set(result["conditioning_indices"])
        eval_idx = set(result["evaluation_indices"])
        unknown_idx = set(result["unknown_indices"])
        N_cond = result["N_cond"]

        # Body starts at position N_cond + 1 (after conditioning and BOS)
        body_start = N_cond + 1

        # Check each position in body
        for i in range(seq_len):
            body_pos = body_start + i
            token_id = aug_input_ids[body_pos].item()

            if i in unknown_idx:
                # Unknown positions should be masked
                self.assertEqual(token_id, self.mask_token_id,
                               f"Position {i} is in unknown set, should be masked")
            else:
                # Conditioning positions should NOT be masked
                self.assertEqual(token_id, original_seq[i].item(),
                               f"Position {i} is in conditioning set, should not be masked")

        # Count masked tokens in body
        body_tokens = aug_input_ids[body_start:]
        num_masked = (body_tokens == self.mask_token_id).sum().item()

        # Should equal size of unknown set
        self.assertEqual(num_masked, len(unknown_idx),
                        f"Number of masked tokens ({num_masked}) should equal unknown set size ({len(unknown_idx)})")

        print(f"✓ Body masking correct: only unknown set masked")
        print(f"  - Conditioning indices: {sorted(cond_idx)}")
        print(f"  - Evaluation indices: {sorted(eval_idx)}")
        print(f"  - Unknown indices: {sorted(unknown_idx)}")
        print(f"  - Masked tokens in body: {num_masked}")

    def test_blockwise_mode(self):
        """Test 5: Verify blockwise sampling mode"""
        print("\n" + "=" * 80)
        print("Test 5: Blockwise Sampling Mode")
        print("=" * 80)

        # Create augmenter with blockwise mode
        blockwise_augmenter = ConditionalAugmenter(
            mask_token_id=self.mask_token_id,
            bos_token_id=self.bos_token_id,
            conditioning_ratio=0.4,
            evaluation_ratio=0.3,
            conditioning_sampling='blockwise',
            evaluation_sampling='blockwise',
            max_cond_blocks=2,
            max_eval_blocks=2
        )

        original_seq = torch.tensor([i for i in range(20)])
        result = blockwise_augmenter.augment_sequence(original_seq)

        cond_idx = sorted(result["conditioning_indices"])
        eval_idx = sorted(result["evaluation_indices"])

        # Check if indices form contiguous blocks
        def count_blocks(indices):
            if not indices:
                return 0
            blocks = 1
            for i in range(1, len(indices)):
                if indices[i] != indices[i-1] + 1:
                    blocks += 1
            return blocks

        cond_blocks = count_blocks(cond_idx)
        eval_blocks = count_blocks(eval_idx)

        print(f"✓ Blockwise mode working")
        print(f"  - Conditioning indices: {cond_idx} ({cond_blocks} block(s))")
        print(f"  - Evaluation indices: {eval_idx} ({eval_blocks} block(s))")

        # Blockwise should produce fewer blocks than random
        # (typically 1-3 blocks vs scattered positions)
        self.assertLessEqual(cond_blocks, 3,
                            "Blockwise mode should produce few blocks")

    def test_batch_augmentation(self):
        """Test 6: Verify batch augmentation works correctly"""
        print("\n" + "=" * 80)
        print("Test 6: Batch Augmentation")
        print("=" * 80)

        # Create a batch of sequences
        batch = torch.tensor([
            [10, 20, 30, 40, 50],
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52]
        ])
        batch_size = batch.size(0)

        # Augment batch
        batch_result = self.augmenter.augment_batch(batch)

        # Check shapes
        self.assertEqual(batch_result["input_ids"].size(0), batch_size,
                        "Batch size should be preserved")
        self.assertEqual(batch_result["position_ids"].size(0), batch_size,
                        "Position IDs should have correct batch size")
        self.assertEqual(batch_result["labels"].size(0), batch_size,
                        "Labels should have correct batch size")

        # Check attention mask shape (batch_size, 1, seq_len, seq_len)
        self.assertEqual(batch_result["attention_mask"].dim(), 4,
                        "Attention mask should be 4D")
        self.assertEqual(batch_result["attention_mask"].size(0), batch_size,
                        "Attention mask batch size should match")

        # All sequences in batch should have same length after augmentation
        seq_len = batch_result["input_ids"].size(1)
        self.assertEqual(batch_result["position_ids"].size(1), seq_len,
                        "Position IDs length should match input")
        self.assertEqual(batch_result["labels"].size(1), seq_len,
                        "Labels length should match input")

        print(f"✓ Batch augmentation working correctly")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Augmented sequence length: {seq_len}")
        print(f"  - Input IDs shape: {batch_result['input_ids'].shape}")
        print(f"  - Position IDs shape: {batch_result['position_ids'].shape}")
        print(f"  - Attention mask shape: {batch_result['attention_mask'].shape}")
        print(f"  - Labels shape: {batch_result['labels'].shape}")


def run_tests():
    """Run all tests"""
    print("=" * 80)
    print("RUNNING PREFIX CONDITIONING UNIT TESTS")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPrefixConditioning)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
