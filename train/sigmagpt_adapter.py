"""
Sigma GPT Data Adapter

Converts augmented batches from ConditionalAugmenter to Sigma GPT format.

The adapter takes conditioning/evaluation indices from the augmenter and:
1. Creates order tensors in Sigma GPT format
2. Applies order to shuffle sequences
3. Creates targets with proper masking (fair or full mode)

This allows using the same augmentation logic for both models.
"""

import torch
from model.order_utils import prepare_sigmagpt_batch


class SigmaGPTDataAdapter:
    """
    Adapter to convert ConditionalAugmenter output to Sigma GPT format.

    The adapter works in two modes:
    - 'fair': Only evaluation positions compute loss (~20-30% learning efficiency)
    - 'full': All positions compute loss (100% learning efficiency)

    Usage:
        augmenter = ConditionalAugmenter(...)
        adapter = SigmaGPTDataAdapter(mode='fair')

        # Get augmented batch from augmenter
        aug_batch = augmenter.augment_batch(input_ids_batch)

        # Convert to Sigma GPT format
        sigmagpt_batch = adapter.convert_batch(aug_batch, original_tokens=input_ids_batch)

        # Feed to Sigma GPT model
        logits, loss = model(
            idx=sigmagpt_batch['inputs'],
            order=sigmagpt_batch['order'],
            targets=sigmagpt_batch['targets']
        )
    """

    def __init__(self, mode='fair'):
        """
        Initialize adapter.

        Args:
            mode: 'fair' or 'full'
                  - 'fair': Only evaluation positions compute loss (matching original Sigma GPT)
                  - 'full': All positions compute loss (maximum learning efficiency)
        """
        if mode not in ['fair', 'full']:
            raise ValueError(f"mode must be 'fair' or 'full', got {mode}")

        self.mode = mode

    def convert_sequence(self, aug_result, original_tokens):
        """
        Convert a single augmented sequence to Sigma GPT format.

        Args:
            aug_result: Dictionary from augmenter.augment_sequence()
                       Must contain: 'conditioning_indices', 'evaluation_indices'
            original_tokens: (seq_len,) original token sequence

        Returns:
            Dictionary containing:
            - inputs: (T,) input tokens for Sigma GPT
            - order: (T+1,) order tensor
            - targets: (T,) target tokens with -1 for ignored positions

        where T = len(conditioning_indices) + len(evaluation_indices)
        """
        # Extract indices from augmenter result
        cond_indices = torch.tensor(aug_result['conditioning_indices'], dtype=torch.long)
        eval_indices = torch.tensor(aug_result['evaluation_indices'], dtype=torch.long)

        # Add batch dimension
        cond_indices = cond_indices.unsqueeze(0)  # (1, num_cond)
        eval_indices = eval_indices.unsqueeze(0)  # (1, num_eval)
        original_tokens = original_tokens.unsqueeze(0)  # (1, seq_len)

        # Use order utilities to create Sigma GPT batch
        inputs, order, targets = prepare_sigmagpt_batch(
            original_tokens, cond_indices, eval_indices, mode=self.mode
        )

        # Remove batch dimension
        return {
            'inputs': inputs.squeeze(0),
            'order': order.squeeze(0),
            'targets': targets.squeeze(0),
        }

    def convert_batch(self, aug_batch, original_tokens):
        """
        Convert a batch of augmented sequences to Sigma GPT format.

        Args:
            aug_batch: List of dictionaries from augmenter.augment_batch()
                      Each dict must contain: 'conditioning_indices', 'evaluation_indices'
            original_tokens: (B, seq_len) original token sequences

        Returns:
            Dictionary containing:
            - inputs: (B, T) input tokens for Sigma GPT (padded)
            - order: (B, T+1) order tensors (padded)
            - targets: (B, T) target tokens with -1 for ignored positions (padded)

        Note: T may vary per batch item, so we pad to max_T in the batch.
        """
        B = len(aug_batch)
        device = original_tokens.device

        # Convert each sequence
        converted = []
        for i in range(B):
            result = self.convert_sequence(aug_batch[i], original_tokens[i])
            converted.append(result)

        # Find maximum T in batch
        max_T = max(c['inputs'].shape[0] for c in converted)

        # Pad all sequences to max_T
        inputs_list = []
        order_list = []
        targets_list = []

        for c in converted:
            T = c['inputs'].shape[0]
            pad_len = max_T - T

            if pad_len > 0:
                # Pad inputs with 0 (will be masked anyway)
                inputs_padded = torch.cat([
                    c['inputs'],
                    torch.zeros(pad_len, dtype=torch.long, device=device)
                ])

                # Pad order with seq_len (indicates end of sequence)
                seq_len = original_tokens.shape[1]
                order_padded = torch.cat([
                    c['order'],
                    torch.full((pad_len,), seq_len, dtype=torch.long, device=device)
                ])

                # Pad targets with -1 (ignored in loss)
                targets_padded = torch.cat([
                    c['targets'],
                    torch.full((pad_len,), -1, dtype=torch.long, device=device)
                ])
            else:
                inputs_padded = c['inputs']
                order_padded = c['order']
                targets_padded = c['targets']

            inputs_list.append(inputs_padded)
            order_list.append(order_padded)
            targets_list.append(targets_padded)

        # Stack into batch tensors
        inputs = torch.stack(inputs_list)  # (B, max_T)
        order = torch.stack(order_list)    # (B, max_T + 1)
        targets = torch.stack(targets_list)  # (B, max_T)

        return {
            'inputs': inputs,
            'order': order,
            'targets': targets,
        }

    def get_stats(self, aug_batch):
        """
        Get statistics about conditioning/evaluation split.

        Args:
            aug_batch: List of dictionaries from augmenter

        Returns:
            Dictionary with statistics:
            - avg_cond_size: Average number of conditioning tokens
            - avg_eval_size: Average number of evaluation tokens
            - avg_total_size: Average total size (cond + eval)
            - avg_cond_pct: Average conditioning percentage
            - avg_eval_pct: Average evaluation percentage
            - learning_efficiency: Percentage of positions that compute loss
        """
        cond_sizes = [len(aug['conditioning_indices']) for aug in aug_batch]
        eval_sizes = [len(aug['evaluation_indices']) for aug in aug_batch]
        total_sizes = [c + e for c, e in zip(cond_sizes, eval_sizes)]

        avg_cond = sum(cond_sizes) / len(cond_sizes)
        avg_eval = sum(eval_sizes) / len(eval_sizes)
        avg_total = sum(total_sizes) / len(total_sizes)

        avg_cond_pct = avg_cond / avg_total if avg_total > 0 else 0
        avg_eval_pct = avg_eval / avg_total if avg_total > 0 else 0

        # Learning efficiency depends on mode
        if self.mode == 'fair':
            # Only evaluation positions learn
            learning_efficiency = avg_eval_pct
        else:  # full
            # All positions learn (except invalid ones, ~100%)
            learning_efficiency = 1.0

        return {
            'avg_cond_size': avg_cond,
            'avg_eval_size': avg_eval,
            'avg_total_size': avg_total,
            'avg_cond_pct': avg_cond_pct,
            'avg_eval_pct': avg_eval_pct,
            'learning_efficiency': learning_efficiency,
        }


def create_sigmagpt_adapter(mode='fair'):
    """
    Factory function to create SigmaGPTDataAdapter.

    Args:
        mode: 'fair' or 'full'

    Returns:
        SigmaGPTDataAdapter instance

    Example:
        adapter = create_sigmagpt_adapter(mode='fair')
    """
    return SigmaGPTDataAdapter(mode=mode)
