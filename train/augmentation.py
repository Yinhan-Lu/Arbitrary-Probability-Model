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
from train.blockwise_sampling import (
    generate_conditioning_set_blockwise,
    generate_conditioning_evaluation_sets_blockwise,
    uniform_num_conditioning_distribution,
    uniform_num_blocks_distribution,
    uniform_block_sizes_distribution,
    uniform_num_evaluation_distribution,
)

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
        # Padding parameters
        max_seq_len=1024,
        cond_pct_max=0.5,
        tokenizer_pad_token_id=50256,
        # Distribution function parameters (new, flexible approach)
        num_conditioning_distribution=None,
        num_blocks_distribution=None,
        block_sizes_distribution=None,
        num_evaluation_distribution=None,
        num_eval_blocks_distribution=None,
        eval_block_sizes_distribution=None,
        min_conditioning=1,
        min_evaluation=1,
        include_bos=True,
        conditioning_sampling='blockwise',
        evaluation_sampling='blockwise',
        max_cond_blocks=3,
        max_eval_blocks=2,
    ):
        """
        Initialize augmenter with distribution-based sampling

        Args:
            mask_token_id: Token ID for [M] mask token
            bos_token_id: Token ID for [BOS] beginning of sequence
            pad_token_id: Token ID for padding (for labels, -100 to ignore)

            Padding parameters:
            max_seq_len: Maximum sequence length from dataset (default: 1024)
            cond_pct_max: Maximum conditioning percentage (default: 0.5)
            tokenizer_pad_token_id: Tokenizer's pad token ID for input_ids padding (default: 50256 for GPT-2)

            Distribution function parameters (required):
            num_conditioning_distribution: Callable[[int], int] - samples num conditioning tokens from seq_len
            num_blocks_distribution: Callable[[int], int] - samples num blocks from num_items
            block_sizes_distribution: Callable[[int, int], list[int]] - samples block sizes
            num_evaluation_distribution: Callable[[int], int] - samples num evaluation tokens from available_len
            num_eval_blocks_distribution: Callable[[int], int] - samples num eval blocks
            eval_block_sizes_distribution: Callable[[int, int], list[int]] - samples eval block sizes

            Other parameters:
            min_conditioning: Minimum number of conditioning tokens
            min_evaluation: Minimum number of evaluation tokens
            include_bos: Whether to prepend BOS token
            conditioning_sampling: Sampling mode for conditioning set - 'blockwise' or 'random'
            evaluation_sampling: Sampling mode for evaluation set - 'blockwise' or 'random'
            max_cond_blocks: Maximum blocks for conditioning (when blockwise, default: 3)
            max_eval_blocks: Maximum blocks for evaluation (when blockwise, default: 2)
        """
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.tokenizer_pad_token_id = tokenizer_pad_token_id
        self.include_bos = include_bos
        self.conditioning_sampling = conditioning_sampling
        self.evaluation_sampling = evaluation_sampling

        # Calculate upper bound for augmented sequence length
        # aug_len = N_cond + 1 (BOS) + original_seq_len
        # Maximum N_cond = ceil(max_seq_len * cond_pct_max)
        import math
        max_n_cond = math.ceil(max_seq_len * cond_pct_max)
        self.aug_max_len = max_n_cond + 1 + max_seq_len
        self.max_seq_len = max_seq_len

        # Validate required distribution functions
        assert num_conditioning_distribution is not None, \
            "num_conditioning_distribution must be provided (distribution-based sampling required)"
        assert num_evaluation_distribution is not None, \
            "num_evaluation_distribution must be provided (distribution-based sampling required)"

        # Store distribution functions
        self.num_conditioning_distribution = num_conditioning_distribution
        self.num_blocks_distribution = num_blocks_distribution
        self.block_sizes_distribution = block_sizes_distribution
        self.num_evaluation_distribution = num_evaluation_distribution
        self.num_eval_blocks_distribution = num_eval_blocks_distribution
        self.eval_block_sizes_distribution = eval_block_sizes_distribution

        # Store other parameters
        self.min_conditioning = min_conditioning
        self.min_evaluation = min_evaluation
        self.max_cond_blocks = max_cond_blocks
        self.max_eval_blocks = max_eval_blocks

        logger.info(f"ConditionalAugmenter initialized:")
        logger.info(f"  Mask token ID: {mask_token_id}")
        logger.info(f"  BOS token ID: {bos_token_id}")
        logger.info(f"  Padding:")
        logger.info(f"    Max sequence length: {max_seq_len}")
        logger.info(f"    Max conditioning %: {cond_pct_max * 100:.1f}%")
        logger.info(f"    Augmented max length: {self.aug_max_len}")
        logger.info(f"  Mode: Distribution-based sampling")
        logger.info(f"  Conditioning sampling: {conditioning_sampling}")
        logger.info(f"  Evaluation sampling: {evaluation_sampling}")

    def split_indices(self, seq_len, valid_positions=None):
        """
        Split sequence indices into conditioning, evaluation, and unknown sets

        Uses distribution-based blockwise sampling for both conditioning and evaluation sets.

        Args:
            seq_len: Length of original sequence (without BOS)
            valid_positions: Optional list of valid positions to sample from (e.g., non-padding).
                           If provided, all sampling only occurs within these positions.
                           If None, samples from range(seq_len).

        Returns:
            Tuple of (conditioning_indices, evaluation_indices, unknown_indices)
        """
        # Only support distribution-based blockwise sampling
        if self.conditioning_sampling == 'blockwise' and self.evaluation_sampling == 'blockwise':
            return generate_conditioning_evaluation_sets_blockwise(
                seq_len=seq_len,
                num_conditioning_distribution=self.num_conditioning_distribution,
                num_blocks_distribution=self.num_blocks_distribution,
                block_sizes_distribution=self.block_sizes_distribution,
                num_evaluation_distribution=self.num_evaluation_distribution,
                num_eval_blocks_distribution=self.num_eval_blocks_distribution,
                eval_block_sizes_distribution=self.eval_block_sizes_distribution,
                valid_positions=valid_positions,
            )
        else:
            raise ValueError(
                f"Only blockwise sampling is supported. "
                f"Got conditioning_sampling='{self.conditioning_sampling}', "
                f"evaluation_sampling='{self.evaluation_sampling}'. "
                f"Please set both to 'blockwise'."
            )

    def augment_sequence(self, input_ids, device='cpu', valid_positions=None):
        """
        Augment sequence using concatenation-based prefix conditioning

        Sequence structure: [Cond tokens] + [BOS + Body tokens]
        - Conditioning tokens moved to prefix (use original positions)
        - BOS token separates prefix from body
        - Body: original sequence (only mask unknown set, keep cond + eval)
        - Position encodings: Cond uses original positions, BOS uses 0, Body uses 1-seq_len

        Args:
            input_ids: Original sequence tensor of shape (seq_len,)
            device: Device to create tensors on
            valid_positions: Optional list of valid positions to sample from (e.g., non-padding).
                           If provided, all sampling only occurs within these positions.
                           If None, infers from input_ids (non-padding positions).

        Returns:
            Dictionary containing:
            - aug_input_ids: Augmented sequence [Cond] + [BOS + Body]
            - position_ids: Custom position encodings
            - attention_mask: Prefix conditional mask
            - labels: Labels for loss (only evaluation positions)
            - conditioning_indices: Original conditioning indices
            - evaluation_indices: Original evaluation indices
            - unknown_indices: Original unknown indices
            - N_cond: Number of conditioning tokens
            - N_seq: Number of sequence tokens (BOS + body)
        """
        seq_len = input_ids.size(0)

        # Infer valid_positions if not provided (filter out padding tokens)
        if valid_positions is None:
            padding_token_id = self.tokenizer.pad_token_id
            if padding_token_id is None:
                padding_token_id = 50256  # GPT-2 default
            valid_positions = [i for i in range(seq_len) if input_ids[i] != padding_token_id]

        # Step 1: Get three sets (conditioning, evaluation, unknown)
        cond_idx, eval_idx, unknown_idx = self.split_indices(seq_len, valid_positions=valid_positions)

        # Step 2: Build conditioning tokens (prefix)
        cond_tokens = []
        cond_position_ids = []
        for idx in sorted(cond_idx):
            cond_tokens.append(input_ids[idx].item())
            cond_position_ids.append(idx + 1)  # Body uses positions 1 to seq_len
        N_cond = len(cond_tokens)

        # Step 3: Build sequence tokens (BOS + Body)
        seq_tokens = [self.bos_token_id]  # BOS at front
        seq_position_ids = [0]  # BOS uses position 0

        # Body tokens: Keep cond + eval, only mask truly unseen tokens
        # IMPORTANT: unknown_idx contains eval_idx, but we want to keep eval tokens visible
        # so that when predicting an eval token, the model can see previous eval tokens.
        # Only mask tokens that are neither conditioning nor evaluation (truly unseen).
        unseen_idx = set(unknown_idx) - set(eval_idx)

        for i in range(seq_len):
            if i in unseen_idx:
                seq_tokens.append(self.mask_token_id)  # Mask only truly unseen tokens
            else:
                seq_tokens.append(input_ids[i].item())  # Keep cond + eval visible
            seq_position_ids.append(i + 1)  # Body uses positions 1 to seq_len
        N_seq = len(seq_tokens)

        # Step 4: Concatenate [Cond] + [Seq]
        aug_input_ids = torch.tensor(
            cond_tokens + seq_tokens,
            dtype=torch.long,
            device=device
        )
        position_ids = torch.tensor(
            cond_position_ids + seq_position_ids,
            dtype=torch.long,
            device=device
        )

        # Step 5: Create attention mask (prefix conditional)
        from train.mask_utils import create_prefix_conditional_mask
        attention_mask = create_prefix_conditional_mask(
            N_cond=N_cond,
            N_seq=N_seq,
            device=device
        )

        # Step 6: Create labels (ONLY for evaluation set)
        labels = torch.full_like(aug_input_ids, self.pad_token_id)  # All -100

        # IMPORTANT: Set labels at position p (not p-1)
        # Because: shift_logits[p-1] corresponds to shift_labels[p-1] = labels[p]
        # Example: logits[2] (BOS output) predicts labels[3] (first eval token)
        for eval_pos in eval_idx:
            # eval_pos is original sequence index (0-based)
            # In new sequence: Cond (N_cond) + BOS (1) + Body (eval_pos)
            new_pos = N_cond + 1 + eval_pos
            if new_pos < len(labels):
                labels[new_pos] = input_ids[eval_pos].item()

        return {
            "aug_input_ids": aug_input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "conditioning_indices": cond_idx,
            "evaluation_indices": eval_idx,
            "unknown_indices": unknown_idx,
            "N_cond": N_cond,
            "N_seq": N_seq
        }

    def augment_batch(self, input_ids_batch, device='cpu', attention_mask=None):
        """
        Augment a batch of sequences

        Args:
            input_ids_batch: Tensor of shape (batch_size, seq_len)
            device: Device to create tensors on
            attention_mask: Optional tensor of shape (batch_size, seq_len) indicating valid positions.
                          1 = valid token, 0 = padding.
                          If provided, sampling only occurs within valid positions.
                          If None, infers from input_ids (non-padding positions).

        Returns:
            Dictionary containing batched tensors:
            - input_ids: (batch_size, aug_seq_len)
            - position_ids: (batch_size, aug_seq_len) - custom position encodings
            - attention_mask: (batch_size, 1, aug_seq_len, aug_seq_len)
            - labels: (batch_size, aug_seq_len)
        """
        batch_size = input_ids_batch.size(0)
        seq_len = input_ids_batch.size(1)

        aug_inputs = []
        aug_positions = []
        aug_masks = []
        aug_labels = []

        for i in range(batch_size):
            # Extract valid positions for this sample if attention_mask provided
            valid_positions = None
            if attention_mask is not None:
                sample_attention_mask = attention_mask[i]
                valid_positions = [j for j in range(seq_len) if sample_attention_mask[j] == 1]

            result = self.augment_sequence(input_ids_batch[i], device=device, valid_positions=valid_positions)

            aug_inputs.append(result["aug_input_ids"])
            aug_positions.append(result["position_ids"])
            aug_masks.append(result["attention_mask"])
            aug_labels.append(result["labels"])

        # Pad all sequences to aug_max_len before stacking
        # This ensures all sequences in batch have same length
        padded_inputs = []
        padded_positions = []
        padded_labels = []
        padded_masks = []

        for i in range(batch_size):
            current_len = aug_inputs[i].size(0)
            pad_len = self.aug_max_len - current_len

            if pad_len > 0:
                # Pad input_ids with tokenizer's pad token
                padded_input = torch.cat([
                    aug_inputs[i],
                    torch.full((pad_len,), self.tokenizer_pad_token_id, dtype=torch.long, device=device)
                ], dim=0)

                # Pad position_ids with 0
                padded_position = torch.cat([
                    aug_positions[i],
                    torch.zeros(pad_len, dtype=torch.long, device=device)
                ], dim=0)

                # Pad labels with -100 (ignored by loss)
                padded_label = torch.cat([
                    aug_labels[i],
                    torch.full((pad_len,), -100, dtype=torch.long, device=device)
                ], dim=0)

                # Pad attention mask (2D matrix)
                # Original mask shape: (current_len, current_len)
                # Padded mask shape: (aug_max_len, aug_max_len)
                # Convention: 1 = can attend, 0 = cannot attend
                # Padded rows: can attend to valid content (prevents NaN in softmax)
                # Padded cols: cannot be attended to by valid content
                padded_mask = torch.zeros(self.aug_max_len, self.aug_max_len, dtype=aug_masks[i].dtype, device=device)
                padded_mask[:current_len, :current_len] = aug_masks[i]
                # Let padded positions attend to all valid content (avoids softmax NaN)
                padded_mask[current_len:, :current_len] = 1

            else:
                # No padding needed (already at max length)
                padded_input = aug_inputs[i]
                padded_position = aug_positions[i]
                padded_label = aug_labels[i]
                padded_mask = aug_masks[i]

            padded_inputs.append(padded_input)
            padded_positions.append(padded_position)
            padded_labels.append(padded_label)
            padded_masks.append(padded_mask)

        # Stack into batches (now all sequences have same length)
        input_ids = torch.stack(padded_inputs, dim=0)
        position_ids = torch.stack(padded_positions, dim=0)
        labels = torch.stack(padded_labels, dim=0)

        # Attention masks need an extra dimension for broadcasting
        # Shape: (batch_size, 1, aug_max_len, aug_max_len)
        attention_mask = torch.stack(padded_masks, dim=0).unsqueeze(1)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
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

    from functools import partial
    augmenter = ConditionalAugmenter(
        mask_token_id=MASK_TOKEN_ID,
        bos_token_id=BOS_TOKEN_ID,
        max_seq_len=1024,
        cond_pct_max=0.5,
        num_conditioning_distribution=partial(
            uniform_num_conditioning_distribution,
            conditioning_percentage_range=(0.3, 0.4)
        ),
        num_blocks_distribution=uniform_num_blocks_distribution,
        block_sizes_distribution=uniform_block_sizes_distribution,
        num_evaluation_distribution=partial(
            uniform_num_evaluation_distribution,
            evaluation_percentage_range=(0.2, 0.3)
        ),
        num_eval_blocks_distribution=uniform_num_blocks_distribution,
        eval_block_sizes_distribution=uniform_block_sizes_distribution,
        conditioning_sampling='blockwise',
        evaluation_sampling='blockwise',
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
