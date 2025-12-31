"""
Evaluation Modes for Sigma GPT Model

Implements 5 evaluation modes for fair comparison with Conditional Model:
- Mode 1: Standard autoregressive (left-to-right, no conditioning)
- Mode 2: Boundary filling (condition on boundaries, evaluate middle)
- Mode 3: Training distribution (same random split as training)
- Mode 4: Autoregressive on Mode 2's evaluation positions
- Mode 5: Autoregressive on Mode 3's evaluation positions

Key difference from Conditional Model:
- Sigma GPT uses order tensor and double position encoding
- Loss is computed with ignore_index=-1
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import math
import logging
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.order_utils import indices_to_order, apply_order, create_labels_fair, apply_labels_mask
from train.blockwise_sampling import (
    generate_boundary_conditioning_split,
    uniform_boundary_block_sizes_distribution
)

logger = logging.getLogger(__name__)


def create_autoregressive_order(seq_len, device='cpu'):
    """
    Create order tensor for standard autoregressive (Mode 1)

    Autoregressive order: [0, 1, 2, ..., seq_len-1, seq_len]
    - Position 0 is conditioning (model sees first token)
    - Positions 1 to seq_len-1 are evaluation
    - seq_len is the end-of-sequence marker

    Args:
        seq_len: Sequence length
        device: Device for tensor

    Returns:
        order: (seq_len+1,) order tensor
        cond_size: Number of conditioning positions (1 for autoregressive)
    """
    # Order: all positions in sequential order
    order = torch.arange(seq_len + 1, device=device, dtype=torch.long)
    cond_size = 1  # Only first position is conditioning
    return order, cond_size


def create_boundary_order(seq_len, boundary_cond_pct_range=(0.1, 0.3), valid_positions=None, device='cpu'):
    """
    Create order tensor for boundary filling (Mode 2)

    Boundary order: [start_block, end_block, middle_block, seq_len]
    - Start and end blocks are conditioning
    - Middle block is evaluation

    Args:
        seq_len: Sequence length
        boundary_cond_pct_range: (min, max) percentage range for boundary conditioning
        valid_positions: List of valid (non-padding) positions
        device: Device for tensor

    Returns:
        order: (T+1,) order tensor where T = cond_size + eval_size
        cond_size: Number of conditioning positions
        eval_indices: List of evaluation position indices
    """
    # Generate boundary split
    cond_idx, eval_idx, _ = generate_boundary_conditioning_split(
        seq_len,
        boundary_block_sizes_distribution=lambda s: uniform_boundary_block_sizes_distribution(
            s, boundary_cond_percentage_range=boundary_cond_pct_range
        ),
        valid_positions=valid_positions
    )

    # Create order: [conditioning_positions, evaluation_positions, seq_len]
    cond_tensor = torch.tensor(cond_idx, device=device, dtype=torch.long)
    eval_tensor = torch.tensor(eval_idx, device=device, dtype=torch.long)
    end_marker = torch.tensor([seq_len], device=device, dtype=torch.long)

    order = torch.cat([cond_tensor, eval_tensor, end_marker])
    cond_size = len(cond_idx)

    return order, cond_size, eval_idx


def sigmagpt_evaluate_mode1_autoregressive(model, dataloader, device, max_batches=None):
    """
    Mode 1: Standard autoregressive evaluation for Sigma GPT

    - Order: [0, 1, 2, ..., seq_len-1, seq_len] (standard left-to-right)
    - Conditioning: position 0 only
    - Evaluation: positions 1 to seq_len-1
    - Uses Sigma GPT's order tensor format

    Args:
        model: Sigma GPT model
        dataloader: Validation dataloader
        device: Device to run on
        max_batches: Maximum number of batches to evaluate

    Returns:
        dict with keys: loss, perplexity, logits_list, labels_list
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_logits = []
    all_labels = []
    all_attention_masks = []
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))
            batch_size, seq_len = input_ids.shape

            # Create autoregressive order for each sample
            # Order: [0, 1, 2, ..., seq_len-1, seq_len]
            order = torch.arange(seq_len + 1, device=device, dtype=torch.long)
            order = order.unsqueeze(0).expand(batch_size, -1)  # (B, seq_len+1)

            # Apply order to get inputs and targets
            inputs, targets = apply_order(input_ids, order)

            # In autoregressive mode, all positions except first are evaluation
            # Create targets with ignore_index=-1 for conditioning position
            cond_size = 1
            mask = create_labels_fair(order, cond_size, seq_len)
            targets = apply_labels_mask(targets, mask)

            # Forward pass
            logits, loss = model(idx=inputs, order=order, targets=targets)

            # Count valid tokens (not -1)
            valid_tokens = (targets != -1).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            num_batches += 1

            # Strip thinking token logits if present (for Mode 4/5 compatibility)
            # With thinking tokens, logits shape is (B, n_thinking + seq_len, vocab)
            # But labels/attention_mask are (B, seq_len), so we need to strip thinking portion
            n_thinking = 0
            if hasattr(model, 'thinking_prepender') and model.thinking_prepender is not None:
                n_thinking = model.thinking_prepender.get_num_thinking_tokens()

            if n_thinking > 0:
                # Strip thinking token logits: keep only body logits
                logits_to_save = logits[:, n_thinking:, :]
            else:
                logits_to_save = logits

            # Store for Mode 4/5 (CPU, half precision to save memory)
            all_logits.append(logits_to_save.cpu().half())
            all_labels.append(input_ids.cpu())
            all_attention_masks.append(attention_mask.cpu())
            del logits, logits_to_save
            torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': num_batches,
        'total_tokens': total_tokens,
        'logits_list': all_logits,
        'labels_list': all_labels,
        'attention_mask_list': all_attention_masks,
    }


def sigmagpt_evaluate_mode2_boundary_filling(model, dataloader, device, max_batches=None,
                                              boundary_cond_pct_range=(0.1, 0.3)):
    """
    Mode 2: Boundary-constrained conditional evaluation for Sigma GPT

    - Conditioning: start block + end block
    - Evaluation: middle continuous part
    - Uses Sigma GPT's order tensor format

    Args:
        model: Sigma GPT model
        dataloader: Validation dataloader
        device: Device to run on
        max_batches: Maximum number of batches to evaluate
        boundary_cond_pct_range: (min, max) percentage for boundary blocks

    Returns:
        dict with keys: loss, perplexity, eval_indices_list
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_eval_indices = []
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))
            batch_size, seq_len = input_ids.shape

            batch_inputs = []
            batch_orders = []
            batch_targets = []
            batch_eval_indices = []

            for i in range(batch_size):
                # Get valid positions (non-padding)
                valid_pos = [j for j in range(seq_len) if attention_mask[i, j] == 1]

                # Generate boundary split
                order, cond_size, eval_idx = create_boundary_order(
                    seq_len,
                    boundary_cond_pct_range=boundary_cond_pct_range,
                    valid_positions=valid_pos,
                    device=device
                )

                # Apply order to single sequence
                tokens = input_ids[i:i+1]  # (1, seq_len)
                order_unsqueeze = order.unsqueeze(0)  # (1, T+1)

                inputs_i, targets_i = apply_order(tokens, order_unsqueeze)

                # Create targets with ignore_index=-1 for conditioning positions
                mask = create_labels_fair(order_unsqueeze, cond_size, seq_len)
                targets_i = apply_labels_mask(targets_i, mask)

                batch_inputs.append(inputs_i.squeeze(0))
                batch_orders.append(order)
                batch_targets.append(targets_i.squeeze(0))
                batch_eval_indices.append(eval_idx)

            # Pad to same length and stack
            max_len = max(inp.shape[0] for inp in batch_inputs)
            max_order_len = max(ord.shape[0] for ord in batch_orders)

            inputs_padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            orders_padded = torch.full((batch_size, max_order_len), seq_len, dtype=torch.long, device=device)
            targets_padded = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)

            for i, (inp, ord, tgt) in enumerate(zip(batch_inputs, batch_orders, batch_targets)):
                inputs_padded[i, :inp.shape[0]] = inp
                orders_padded[i, :ord.shape[0]] = ord
                targets_padded[i, :tgt.shape[0]] = tgt

            # Forward pass
            logits, loss = model(idx=inputs_padded, order=orders_padded, targets=targets_padded)

            # Count valid tokens
            valid_tokens = (targets_padded != -1).sum().item()
            if valid_tokens > 0:
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

            all_eval_indices.extend(batch_eval_indices)
            num_batches += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': num_batches,
        'total_tokens': total_tokens,
        'eval_indices_list': all_eval_indices,
    }


def sigmagpt_evaluate_mode3_training_dist(model, dataloader, device, augmenter, adapter, max_batches=None):
    """
    Mode 3: Training-distribution conditional evaluation for Sigma GPT

    - Uses same augmenter as training (blockwise sampling)
    - Same as current evaluate() method
    - Uses Sigma GPT's order tensor format

    Args:
        model: Sigma GPT model
        dataloader: Validation dataloader
        device: Device to run on
        augmenter: Training augmenter
        adapter: SigmaGPTDataAdapter
        max_batches: Maximum number of batches to evaluate

    Returns:
        dict with keys: loss, perplexity, eval_indices_list
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_eval_indices = []
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            input_ids_cpu = input_ids.cpu()
            batch_size = input_ids_cpu.size(0)

            aug_batch = []
            batch_eval_indices = []
            for i in range(batch_size):
                result = augmenter.augment_sequence(input_ids_cpu[i], device='cpu')
                aug_batch.append(result)
                batch_eval_indices.append(result['evaluation_indices'])

            sigmagpt_batch = adapter.convert_batch(aug_batch, input_ids_cpu)

            inputs = sigmagpt_batch['inputs'].to(device)
            order = sigmagpt_batch['order'].to(device)
            targets = sigmagpt_batch['targets'].to(device)

            logits, loss = model(idx=inputs, order=order, targets=targets)

            valid_tokens = (targets != -1).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

            all_eval_indices.extend(batch_eval_indices)
            num_batches += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': num_batches,
        'total_tokens': total_tokens,
        'eval_indices_list': all_eval_indices,
    }


def sigmagpt_evaluate_mode4_cross_boundary(logits_list, labels_list, eval_indices_list, attention_mask_list):
    """
    Mode 4: Autoregressive evaluation on Mode 2's evaluation positions

    Reuses Mode 1's logits, computes loss only on Mode 2's evaluation positions.
    No forward pass needed.

    Args:
        logits_list: List of logits from Mode 1
        labels_list: List of labels from Mode 1
        eval_indices_list: Flat list of evaluation indices from Mode 2
        attention_mask_list: List of attention masks from Mode 1 for padding filtering

    Returns:
        dict with keys: loss, perplexity
    """
    total_loss = 0.0
    total_tokens = 0

    # Flatten index into batches
    sample_idx = 0
    for logits_batch, labels_batch, attention_mask_batch in zip(logits_list, labels_list, attention_mask_list):
        batch_size = logits_batch.shape[0]

        for i in range(batch_size):
            if sample_idx >= len(eval_indices_list):
                break

            logits = logits_batch[i]  # (seq_len, vocab_size)
            labels = labels_batch[i]  # (seq_len,)
            attention_mask = attention_mask_batch[i]  # (seq_len,)
            eval_indices = eval_indices_list[sample_idx]

            # Autoregressive: shift for next-token prediction
            shift_logits = logits[:-1]  # (seq_len-1, vocab_size)
            shift_labels = labels[1:]   # (seq_len-1,)

            # Create mask for evaluation positions
            eval_mask = torch.zeros(shift_labels.shape[0], dtype=torch.bool, device=shift_labels.device)
            for eval_pos in eval_indices:
                # To predict labels[eval_pos], we use shift_logits[eval_pos-1]
                if 0 <= eval_pos - 1 < eval_mask.shape[0]:
                    eval_mask[eval_pos - 1] = True

            # Filter out padding tokens using attention_mask (not token ID)
            # This correctly includes EOS tokens (50256) which are valid content tokens
            attention_mask_shifted = attention_mask[1:]  # Shift to match labels
            non_padding_mask = (attention_mask_shifted == 1)
            valid_mask = eval_mask & non_padding_mask

            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    shift_logits[valid_mask].float(),
                    shift_labels[valid_mask],
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += valid_mask.sum().item()

            sample_idx += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
    }


def sigmagpt_evaluate_mode5_cross_training(logits_list, labels_list, eval_indices_list, attention_mask_list):
    """
    Mode 5: Autoregressive evaluation on Mode 3's evaluation positions

    Reuses Mode 1's logits, computes loss only on Mode 3's evaluation positions.
    No forward pass needed.

    Args:
        logits_list: List of logits from Mode 1
        labels_list: List of labels from Mode 1
        eval_indices_list: Flat list of evaluation indices from Mode 3
        attention_mask_list: List of attention masks from Mode 1 for padding filtering

    Returns:
        dict with keys: loss, perplexity
    """
    # Implementation is identical to Mode 4, just uses different indices
    return sigmagpt_evaluate_mode4_cross_boundary(logits_list, labels_list, eval_indices_list, attention_mask_list)


def sigmagpt_evaluate_all_modes(model, dataloader, device, augmenter, adapter, max_batches=None,
                                  boundary_cond_pct_range=(0.1, 0.3)):
    """
    Orchestrator: Run all 5 evaluation modes for Sigma GPT

    Execution order:
    1. Mode 2 (record eval_indices_mode2)
    2. Mode 3 (record eval_indices_mode3)
    3. Mode 1 (record logits and labels)
    4. Mode 4 (use Mode 1 logits + Mode 2 indices)
    5. Mode 5 (use Mode 1 logits + Mode 3 indices)

    Args:
        model: Sigma GPT model
        dataloader: Validation dataloader
        device: Device to run on
        augmenter: Training augmenter
        adapter: SigmaGPTDataAdapter
        max_batches: Maximum number of batches
        boundary_cond_pct_range: (min, max) for Mode 2 boundary conditioning

    Returns:
        dict with all metrics for 5 modes
    """
    logger.info("Running 5-mode evaluation for Sigma GPT...")

    # Mode 2: Boundary filling
    logger.info("  Mode 2: Boundary filling...")
    metrics_mode2 = sigmagpt_evaluate_mode2_boundary_filling(
        model, dataloader, device, max_batches, boundary_cond_pct_range
    )

    # Mode 3: Training distribution
    logger.info("  Mode 3: Training distribution...")
    metrics_mode3 = sigmagpt_evaluate_mode3_training_dist(
        model, dataloader, device, augmenter, adapter, max_batches
    )

    # Mode 1: Autoregressive
    logger.info("  Mode 1: Autoregressive...")
    metrics_mode1 = sigmagpt_evaluate_mode1_autoregressive(model, dataloader, device, max_batches)

    # Mode 4: Cross-boundary (reuse Mode 1 logits)
    logger.info("  Mode 4: Cross-boundary...")
    metrics_mode4 = sigmagpt_evaluate_mode4_cross_boundary(
        metrics_mode1['logits_list'],
        metrics_mode1['labels_list'],
        metrics_mode2['eval_indices_list'],
        metrics_mode1['attention_mask_list']
    )

    # Mode 5: Cross-training (reuse Mode 1 logits)
    logger.info("  Mode 5: Cross-training...")
    metrics_mode5 = sigmagpt_evaluate_mode5_cross_training(
        metrics_mode1['logits_list'],
        metrics_mode1['labels_list'],
        metrics_mode3['eval_indices_list'],
        metrics_mode1['attention_mask_list']
    )

    # Aggregate results
    return {
        'mode1_loss': metrics_mode1['loss'],
        'mode1_ppl': metrics_mode1['perplexity'],
        'mode1_tokens': metrics_mode1['total_tokens'],
        'mode2_loss': metrics_mode2['loss'],
        'mode2_ppl': metrics_mode2['perplexity'],
        'mode2_tokens': metrics_mode2['total_tokens'],
        'mode3_loss': metrics_mode3['loss'],
        'mode3_ppl': metrics_mode3['perplexity'],
        'mode3_tokens': metrics_mode3['total_tokens'],
        'mode4_loss': metrics_mode4['loss'],
        'mode4_ppl': metrics_mode4['perplexity'],
        'mode4_tokens': metrics_mode4['total_tokens'],
        'mode5_loss': metrics_mode5['loss'],
        'mode5_ppl': metrics_mode5['perplexity'],
        'mode5_tokens': metrics_mode5['total_tokens'],
        'num_batches': metrics_mode1['num_batches'],
    }
