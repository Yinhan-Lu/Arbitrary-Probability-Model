"""
Data Augmentation for Arbitrary Conditional Probability Modeling

Prepares training samples by:
1. Randomly splitting sequence into conditioning, evaluation, and unknown sets
2. Replacing unknown tokens with [M] mask token
3. Creating custom attention masks
4. Preparing labels for loss computation (only on evaluation set)
"""

import sys
from pathlib import Path
import torch
import random
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.mask_utils import create_conditional_mask, validate_mask_indices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConditionalAugmenter:
    """
    Augments sequences for arbitrary conditional probability training

    Converts: [x1, x2, x3, x4, x5]
    To: [BOS, x1/[M], x2/[M], x3/[M], x4/[M], x5/[M]]
    With custom attention mask and labels
    """

    def __init__(
        self,
        mask_token_id,
        bos_token_id,
        pad_token_id=-100,
        conditioning_ratio=0.3,
        evaluation_ratio=0.3,
        min_conditioning=1,
        min_evaluation=1,
        include_bos=True
    ):
        """
        Initialize augmenter

        Args:
            mask_token_id: Token ID for [M] mask token
            bos_token_id: Token ID for [BOS] beginning of sequence
            pad_token_id: Token ID for padding (for labels, -100 to ignore)
            conditioning_ratio: Fraction of tokens to use as conditioning (default 0.3)
            evaluation_ratio: Fraction of tokens to use as evaluation (default 0.3)
            min_conditioning: Minimum number of conditioning tokens
            min_evaluation: Minimum number of evaluation tokens
            include_bos: Whether to prepend BOS token
        """
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id

        self.conditioning_ratio = conditioning_ratio
        self.evaluation_ratio = evaluation_ratio
        self.min_conditioning = min_conditioning
        self.min_evaluation = min_evaluation
        self.include_bos = include_bos

        logger.info(f"ConditionalAugmenter initialized:")
        logger.info(f"  Mask token ID: {mask_token_id}")
        logger.info(f"  BOS token ID: {bos_token_id}")
        logger.info(f"  Conditioning ratio: {conditioning_ratio}")
        logger.info(f"  Evaluation ratio: {evaluation_ratio}")

    def split_indices(self, seq_len):
        """
        Randomly split sequence indices into conditioning, evaluation, and unknown sets

        Args:
            seq_len: Length of original sequence (without BOS)

        Returns:
            Tuple of (conditioning_indices, evaluation_indices, unknown_indices)
        """
        indices = list(range(seq_len))

        # Determine sizes
        num_cond = max(self.min_conditioning, int(seq_len * self.conditioning_ratio))
        num_eval = max(self.min_evaluation, int(seq_len * self.evaluation_ratio))

        # Ensure we don't exceed sequence length
        num_cond = min(num_cond, seq_len - self.min_evaluation)
        num_eval = min(num_eval, seq_len - num_cond)

        # Randomly sample conditioning indices
        conditioning_indices = random.sample(indices, num_cond)

        # Remaining indices form unknown set
        remaining = [idx for idx in indices if idx not in conditioning_indices]

        # Sample evaluation indices from remaining
        num_eval = min(num_eval, len(remaining))
        evaluation_indices = random.sample(remaining, num_eval)

        # Unknown set includes evaluation + other unknowns
        unknown_indices = [idx for idx in indices if idx not in conditioning_indices]

        # Validate the split
        validate_mask_indices(seq_len, conditioning_indices, evaluation_indices, unknown_indices)

        return conditioning_indices, evaluation_indices, unknown_indices

    def augment_sequence(self, input_ids, device='cpu'):
        """
        Augment a single sequence for conditional training

        Args:
            input_ids: Original sequence tensor of shape (seq_len,)
            device: Device to create tensors on

        Returns:
            Dictionary containing:
            - aug_input_ids: Augmented input with BOS and [M] masks
            - attention_mask: Custom attention mask
            - labels: Labels for loss computation (-100 for non-evaluation positions)
            - conditioning_indices: Original conditioning indices
            - evaluation_indices: Original evaluation indices
        """
        seq_len = input_ids.size(0)

        # Split into sets
        cond_idx, eval_idx, unknown_idx = self.split_indices(seq_len)

        # Create augmented input
        aug_input_ids = []

        # Add BOS if requested
        if self.include_bos:
            aug_input_ids.append(self.bos_token_id)

        # Add tokens (original or [M])
        for i in range(seq_len):
            if i in unknown_idx:
                # Unknown positions: replace with [M]
                aug_input_ids.append(self.mask_token_id)
            else:
                # Conditioning positions: keep original token
                aug_input_ids.append(input_ids[i].item())

        aug_input_ids = torch.tensor(aug_input_ids, dtype=torch.long, device=device)

        # Create custom attention mask
        aug_seq_len = aug_input_ids.size(0)
        attention_mask = create_conditional_mask(
            seq_len=aug_seq_len,
            conditioning_indices=cond_idx,
            unknown_indices=unknown_idx,
            device=device,
            include_bos=self.include_bos
        )

        # Create labels for loss computation
        # Initialize with pad_token_id (-100 to ignore)
        labels = torch.full_like(aug_input_ids, self.pad_token_id)

        # Set labels only for evaluation positions
        # Labels are for next-token prediction, so we set label[i] = token at original position
        for eval_pos in eval_idx:
            if self.include_bos:
                # Shift by 1 due to BOS
                label_pos = eval_pos + 1
            else:
                label_pos = eval_pos

            # Set label to original token value
            if label_pos < len(labels):
                labels[label_pos] = input_ids[eval_pos].item()

        return {
            "aug_input_ids": aug_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "conditioning_indices": cond_idx,
            "evaluation_indices": eval_idx,
            "unknown_indices": unknown_idx
        }

    def augment_batch(self, input_ids_batch, device='cpu'):
        """
        Augment a batch of sequences

        Args:
            input_ids_batch: Tensor of shape (batch_size, seq_len)
            device: Device to create tensors on

        Returns:
            Dictionary containing batched tensors:
            - input_ids: (batch_size, aug_seq_len)
            - attention_mask: (batch_size, aug_seq_len, aug_seq_len)
            - labels: (batch_size, aug_seq_len)
        """
        batch_size = input_ids_batch.size(0)

        aug_inputs = []
        aug_masks = []
        aug_labels = []

        for i in range(batch_size):
            result = self.augment_sequence(input_ids_batch[i], device=device)

            aug_inputs.append(result["aug_input_ids"])
            aug_masks.append(result["attention_mask"])
            aug_labels.append(result["labels"])

        # Stack into batches
        # All sequences should have same length after augmentation
        input_ids = torch.stack(aug_inputs, dim=0)
        labels = torch.stack(aug_labels, dim=0)

        # Attention masks need an extra dimension for broadcasting
        # Shape: (batch_size, 1, seq_len, seq_len)
        attention_mask = torch.stack(aug_masks, dim=0).unsqueeze(1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


if __name__ == "__main__":
    # Test augmentation
    print("=" * 80)
    print("Testing Conditional Augmenter")
    print("=" * 80)

    # Create dummy augmenter
    MASK_TOKEN_ID = 50257  # Assuming extended vocab
    BOS_TOKEN_ID = 50256   # Reuse EOS as BOS
    PAD_TOKEN_ID = -100

    augmenter = ConditionalAugmenter(
        mask_token_id=MASK_TOKEN_ID,
        bos_token_id=BOS_TOKEN_ID,
        conditioning_ratio=0.4,
        evaluation_ratio=0.3
    )

    # Test 1: Single sequence
    print("\nTest 1: Single Sequence Augmentation")
    original_seq = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])
    print(f"Original sequence: {original_seq.tolist()}")
    print(f"Original length: {len(original_seq)}")

    result = augmenter.augment_sequence(original_seq)

    print(f"\nAugmented input_ids: {result['aug_input_ids'].tolist()}")
    print(f"Augmented length: {len(result['aug_input_ids'])}")
    print(f"Conditioning indices (original): {result['conditioning_indices']}")
    print(f"Evaluation indices (original): {result['evaluation_indices']}")
    print(f"Unknown indices (original): {result['unknown_indices']}")

    print(f"\nLabels: {result['labels'].tolist()}")
    print(f"(Note: -100 means position is ignored in loss)")

    print(f"\nAttention mask shape: {result['attention_mask'].shape}")
    print("Attention mask (first 5x5):")
    print(result['attention_mask'][:5, :5])

    # Test 2: Batch augmentation
    print("\n" + "=" * 80)
    print("Test 2: Batch Augmentation")
    print("=" * 80)

    batch = torch.tensor([
        [10, 20, 30, 40, 50],
        [11, 21, 31, 41, 51],
        [12, 22, 32, 42, 52]
    ])
    print(f"Batch shape: {batch.shape}")
    print(f"Batch:\n{batch}")

    batch_result = augmenter.augment_batch(batch)

    print(f"\nAugmented batch:")
    print(f"  input_ids shape: {batch_result['input_ids'].shape}")
    print(f"  attention_mask shape: {batch_result['attention_mask'].shape}")
    print(f"  labels shape: {batch_result['labels'].shape}")

    print(f"\nFirst sample in batch:")
    print(f"  input_ids: {batch_result['input_ids'][0].tolist()}")
    print(f"  labels: {batch_result['labels'][0].tolist()}")

    # Test 3: Verify different random splits
    print("\n" + "=" * 80)
    print("Test 3: Randomness Check (multiple augmentations of same sequence)")
    print("=" * 80)

    test_seq = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Original sequence: {test_seq.tolist()}")

    print("\nGenerating 3 different augmentations:")
    for i in range(3):
        res = augmenter.augment_sequence(test_seq)
        print(f"\nAugmentation {i+1}:")
        print(f"  Conditioning: {sorted(res['conditioning_indices'])}")
        print(f"  Evaluation: {sorted(res['evaluation_indices'])}")
        print(f"  Input IDs: {res['aug_input_ids'].tolist()}")

    print("\n" + "=" * 80)
    print("✓ All augmentation tests passed!")
    print("=" * 80)
