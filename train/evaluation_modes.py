"""
Evaluation Modes for Conditional Probability Model

Implements 5 evaluation modes as described in docs/Evaluation设计.md:
- Mode 1: Standard autoregressive evaluation
- Mode 2: Boundary-constrained conditional evaluation
- Mode 3: Training-distribution conditional evaluation
- Mode 4: Autoregressive on Mode 2's evaluation set
- Mode 5: Autoregressive on Mode 3's evaluation set
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import math
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.blockwise_sampling import (
    generate_boundary_conditioning_split,
    uniform_boundary_block_sizes_distribution
)
from train.mask_utils import create_prefix_conditional_mask

logger = logging.getLogger(__name__)


def evaluate_mode1_autoregressive(model, dataloader, device, max_batches=None):
    """
    Mode 1: Standard autoregressive evaluation

    - No conditioning
    - Evaluate loss on ALL tokens
    - Standard causal attention mask

    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        device: Device to run on
        max_batches: Maximum number of batches to evaluate (None = all)

    Returns:
        dict with keys: loss, perplexity, logits_list, labels_list
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_logits = []
    all_labels = []
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask_1d = batch['attention_mask'].to(device)
            batch_size, seq_len = input_ids.shape

            # Standard causal mask
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.uint8))
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)

            # Forward pass
            # Model returns (logits, loss) tuple
            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Compute loss on all positions (shift for next-token prediction)
            # Only compute loss on non-padding tokens using attention_mask
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Use attention_mask to filter out padding tokens
            # attention_mask_1d shape: [batch_size, seq_len], 1=real token, 0=padding
            attention_mask_shifted = attention_mask_1d[:, 1:]  # Shift to match labels
            valid_mask = (attention_mask_shifted == 1)

            loss = F.cross_entropy(
                shift_logits[valid_mask],
                shift_labels[valid_mask],
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += valid_mask.sum().item()
            num_batches += 1

            # Store for Mode 4/5 (move to CPU and free GPU memory immediately)
            all_logits.append(logits.cpu())
            all_labels.append(input_ids.cpu())
            del logits, input_ids  # Free GPU memory
            torch.cuda.empty_cache()  # Clear cache

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': num_batches,
        'total_tokens': total_tokens,
        'logits_list': all_logits,
        'labels_list': all_labels,
    }


def evaluate_mode2_boundary_filling(model, dataloader, device, augmenter, max_batches=None, trainer_args=None):
    """
    Mode 2: Boundary-constrained conditional evaluation

    - Conditioning = start block + end block
    - Evaluation = middle continuous part
    - Prefix conditional attention mask

    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        device: Device to run on
        augmenter: Augmenter with boundary sampling (used for augment_sequence API)
        max_batches: Maximum number of batches to evaluate
        trainer_args: Trainer arguments containing Mode 2 boundary distribution parameters

    Returns:
        dict with keys: loss, perplexity, eval_indices_list
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_eval_indices = []
    num_batches = 0

    # Create boundary distribution function from trainer_args
    boundary_distribution = None
    if trainer_args is not None:
        boundary_cond_pct_min = getattr(trainer_args, 'mode2_boundary_cond_pct_min', 0.1)
        boundary_cond_pct_max = getattr(trainer_args, 'mode2_boundary_cond_pct_max', 0.3)

        # Create partial function with the percentage range
        def boundary_dist_fn(seq_len):
            return uniform_boundary_block_sizes_distribution(
                seq_len,
                boundary_cond_percentage_range=(boundary_cond_pct_min, boundary_cond_pct_max)
            )
        boundary_distribution = boundary_dist_fn

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask_1d = batch['attention_mask'].to(device)
            batch_size, seq_len = input_ids.shape

            batch_loss = 0.0
            batch_tokens = 0
            batch_eval_indices = []

            # Process each sample in batch
            for i in range(batch_size):
                sample_ids = input_ids[i]
                sample_attention_mask = attention_mask_1d[i]

                # Get valid (non-padding) positions
                valid_positions = [j for j in range(seq_len) if sample_attention_mask[j] == 1]

                # Generate boundary split with distribution (only from valid positions)
                cond_idx, eval_idx, unknown_idx = generate_boundary_conditioning_split(
                    seq_len,
                    boundary_block_sizes_distribution=boundary_distribution,
                    valid_positions=valid_positions
                )
                batch_eval_indices.append(eval_idx)

                # Augment sequence (manually, similar to augmenter.augment_sequence)
                # Build prefix (conditioning tokens)
                cond_tokens = [sample_ids[idx].item() for idx in sorted(cond_idx)]
                cond_position_ids = [idx + 1 for idx in sorted(cond_idx)]  # Body uses positions 1-seq_len
                N_cond = len(cond_tokens)

                # Build body (BOS + masked body)
                bos_token_id = augmenter.bos_token_id
                mask_token_id = augmenter.mask_token_id

                seq_tokens = [bos_token_id]
                seq_position_ids = [0]

                for j in range(seq_len):
                    if j in unknown_idx:
                        seq_tokens.append(mask_token_id)  # Mask unknown (which = eval for Mode 2)
                    else:
                        seq_tokens.append(sample_ids[j].item())  # Keep conditioning
                    seq_position_ids.append(j + 1)  # Body uses positions 1-seq_len

                N_seq = len(seq_tokens)

                # Create tensors
                aug_input_ids = torch.tensor(cond_tokens + seq_tokens, dtype=torch.long, device=device)
                position_ids = torch.tensor(cond_position_ids + seq_position_ids, dtype=torch.long, device=device)

                # Create attention mask
                attention_mask = create_prefix_conditional_mask(N_cond, N_seq, device=device)
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

                # Create labels (only for evaluation positions)
                labels = torch.full_like(aug_input_ids, -100)
                for eval_pos in eval_idx:
                    new_pos = N_cond + 1 + eval_pos
                    if new_pos < len(labels):
                        labels[new_pos] = sample_ids[eval_pos].item()

                # Forward pass
                # Model returns (logits, loss) tuple
                logits, _ = model(
                    input_ids=aug_input_ids.unsqueeze(0),
                    position_ids=position_ids.unsqueeze(0),
                    attention_mask=attention_mask,
                )
                logits = logits.squeeze(0)

                # Compute loss (with shift)
                shift_logits = logits[:-1]
                shift_labels = labels[1:]

                valid_mask = (shift_labels != -100)
                if valid_mask.sum() > 0:
                    loss = F.cross_entropy(
                        shift_logits[valid_mask],
                        shift_labels[valid_mask],
                        reduction='sum'
                    )
                    batch_loss += loss.item()
                    batch_tokens += valid_mask.sum().item()

            total_loss += batch_loss
            total_tokens += batch_tokens
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


def evaluate_mode3_training_dist(model, dataloader, device, augmenter, max_batches=None):
    """
    Mode 3: Training-distribution conditional evaluation

    - Uses same augmenter as training (blockwise sampling)
    - Evaluation on blockwise-sampled evaluation set
    - Prefix conditional attention mask

    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        device: Device to run on
        augmenter: Training augmenter (with blockwise sampling)
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
            attention_mask_1d = batch['attention_mask'].to(device)
            batch_size, seq_len = input_ids.shape

            batch_loss = 0.0
            batch_tokens = 0
            batch_eval_indices = []

            # Process each sample individually (like Mode 2)
            # This avoids the need for padding different-length sequences
            for i in range(batch_size):
                sample_ids = input_ids[i]
                sample_attention_mask = attention_mask_1d[i]

                # Get valid (non-padding) positions
                valid_positions = [j for j in range(seq_len) if sample_attention_mask[j] == 1]

                # Augment single sequence (with valid positions)
                result = augmenter.augment_sequence(sample_ids, device=device, valid_positions=valid_positions)

                # Collect evaluation indices (with valid positions)
                cond_idx, eval_idx, unknown_idx = augmenter.split_indices(seq_len, valid_positions=valid_positions)
                batch_eval_indices.append(eval_idx)

                # Forward pass for this single sample
                # Model returns (logits, loss) tuple
                logits, _ = model(
                    input_ids=result['aug_input_ids'].unsqueeze(0),
                    position_ids=result['position_ids'].unsqueeze(0),
                    attention_mask=result['attention_mask'].unsqueeze(0).unsqueeze(0),
                )
                logits = logits.squeeze(0)
                labels = result['labels']

                # Compute loss (with shift)
                shift_logits = logits[:-1]
                shift_labels = labels[1:]

                valid_mask = (shift_labels != -100)
                if valid_mask.sum() > 0:
                    loss = F.cross_entropy(
                        shift_logits[valid_mask],
                        shift_labels[valid_mask],
                        reduction='sum'
                    )
                    batch_loss += loss.item()
                    batch_tokens += valid_mask.sum().item()

            total_loss += batch_loss
            total_tokens += batch_tokens
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


def evaluate_mode4_cross_boundary(logits_list, labels_list, eval_indices_list):
    """
    Mode 4: Autoregressive evaluation on Mode 2's evaluation set

    Reuses Mode 1's logits, computes loss only on Mode 2's evaluation positions.
    No forward pass needed.

    Args:
        logits_list: List of logits from Mode 1 [batch_size, seq_len, vocab_size]
        labels_list: List of labels from Mode 1 [batch_size, seq_len]
        eval_indices_list: List of evaluation indices from Mode 2

    Returns:
        dict with keys: loss, perplexity
    """
    total_loss = 0.0
    total_tokens = 0

    for logits_batch, labels_batch, eval_indices_batch in zip(logits_list, labels_list, eval_indices_list):
        batch_size = logits_batch.shape[0]

        # Process each sample in batch
        for i in range(batch_size):
            logits = logits_batch[i]  # [seq_len, vocab_size]
            labels = labels_batch[i]  # [seq_len]
            eval_indices = eval_indices_batch[i] if isinstance(eval_indices_batch[0], list) else eval_indices_batch

            # Shift for next-token prediction
            shift_logits = logits[:-1]  # [seq_len-1, vocab_size]
            shift_labels = labels[1:]   # [seq_len-1]

            # Create mask for evaluation positions
            # eval_indices are original positions, need to map to shifted positions
            eval_mask = torch.zeros(shift_labels.shape[0], dtype=torch.bool, device=shift_labels.device)
            for eval_pos in eval_indices:
                # Position eval_pos in original sequence corresponds to shift position eval_pos
                # because shift_logits[t] predicts shift_labels[t] = labels[t+1]
                # So to evaluate labels[eval_pos], we use shift_logits[eval_pos-1]
                if 0 <= eval_pos - 1 < eval_mask.shape[0]:
                    eval_mask[eval_pos - 1] = True

            # Filter out padding tokens (padding token ID = 50256 for GPT-2)
            # Only compute loss on non-padding evaluation tokens
            padding_token_id = 50256
            non_padding_mask = (shift_labels != padding_token_id)
            valid_mask = eval_mask & non_padding_mask

            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    shift_logits[valid_mask],
                    shift_labels[valid_mask],
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += valid_mask.sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
    }


def evaluate_mode5_cross_training(logits_list, labels_list, eval_indices_list):
    """
    Mode 5: Autoregressive evaluation on Mode 3's evaluation set

    Reuses Mode 1's logits, computes loss only on Mode 3's evaluation positions.
    No forward pass needed.

    Args:
        logits_list: List of logits from Mode 1 [batch_size, seq_len, vocab_size]
        labels_list: List of labels from Mode 1 [batch_size, seq_len]
        eval_indices_list: List of evaluation indices from Mode 3

    Returns:
        dict with keys: loss, perplexity
    """
    total_loss = 0.0
    total_tokens = 0

    for logits_batch, labels_batch, eval_indices_batch in zip(logits_list, labels_list, eval_indices_list):
        batch_size = logits_batch.shape[0]

        # Process each sample in batch
        for i in range(batch_size):
            logits = logits_batch[i]  # [seq_len, vocab_size]
            labels = labels_batch[i]  # [seq_len]
            eval_indices = eval_indices_batch[i] if isinstance(eval_indices_batch[0], list) else eval_indices_batch

            # Shift for next-token prediction
            shift_logits = logits[:-1]  # [seq_len-1, vocab_size]
            shift_labels = labels[1:]   # [seq_len-1]

            # Create mask for evaluation positions
            eval_mask = torch.zeros(shift_labels.shape[0], dtype=torch.bool, device=shift_labels.device)
            for eval_pos in eval_indices:
                if 0 <= eval_pos - 1 < eval_mask.shape[0]:
                    eval_mask[eval_pos - 1] = True

            # Filter out padding tokens (padding token ID = 50256 for GPT-2)
            # Only compute loss on non-padding evaluation tokens
            padding_token_id = 50256
            non_padding_mask = (shift_labels != padding_token_id)
            valid_mask = eval_mask & non_padding_mask

            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    shift_logits[valid_mask],
                    shift_labels[valid_mask],
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += valid_mask.sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
    }


def evaluate_all_modes(model, dataloader, device, augmenter, max_batches=None, trainer_args=None):
    """
    Orchestrator: Run all 5 evaluation modes

    Execution order:
    1. Mode 2 (record eval_indices_mode2)
    2. Mode 3 (record eval_indices_mode3)
    3. Mode 1 (record logits and labels)
    4. Mode 4 (use Mode 1 logits + Mode 2 indices)
    5. Mode 5 (use Mode 1 logits + Mode 3 indices)

    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        device: Device to run on
        augmenter: Training augmenter
        max_batches: Maximum number of batches to evaluate
        trainer_args: Trainer arguments containing Mode 2 boundary distribution parameters

    Returns:
        dict with all metrics for 5 modes
    """
    logger.info("Running 5-mode evaluation...")

    # Mode 2: Boundary filling
    logger.info("  Mode 2: Boundary filling...")
    metrics_mode2 = evaluate_mode2_boundary_filling(model, dataloader, device, augmenter, max_batches, trainer_args)

    # Mode 3: Training distribution
    logger.info("  Mode 3: Training distribution...")
    metrics_mode3 = evaluate_mode3_training_dist(model, dataloader, device, augmenter, max_batches)

    # Mode 1: Autoregressive
    logger.info("  Mode 1: Autoregressive...")
    metrics_mode1 = evaluate_mode1_autoregressive(model, dataloader, device, max_batches)

    # Mode 4: Cross-boundary (reuse Mode 1 logits)
    logger.info("  Mode 4: Cross-boundary...")
    # Group eval_indices by batch for Mode 2
    eval_indices_mode2_grouped = []
    batch_size = metrics_mode1['labels_list'][0].shape[0] if len(metrics_mode1['labels_list']) > 0 else 0
    for i in range(0, len(metrics_mode2['eval_indices_list']), batch_size):
        eval_indices_mode2_grouped.append(metrics_mode2['eval_indices_list'][i:i+batch_size])

    metrics_mode4 = evaluate_mode4_cross_boundary(
        metrics_mode1['logits_list'],
        metrics_mode1['labels_list'],
        eval_indices_mode2_grouped
    )

    # Mode 5: Cross-training (reuse Mode 1 logits)
    logger.info("  Mode 5: Cross-training...")
    # Group eval_indices by batch for Mode 3
    eval_indices_mode3_grouped = []
    for i in range(0, len(metrics_mode3['eval_indices_list']), batch_size):
        eval_indices_mode3_grouped.append(metrics_mode3['eval_indices_list'][i:i+batch_size])

    metrics_mode5 = evaluate_mode5_cross_training(
        metrics_mode1['logits_list'],
        metrics_mode1['labels_list'],
        eval_indices_mode3_grouped
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
