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
    all_attention_masks = []  # Store attention masks for Mode 4/5
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
            # Convert to float16 to save memory (31GB → 15.5GB)
            all_logits.append(logits.cpu().half())
            all_labels.append(input_ids.cpu())
            all_attention_masks.append(attention_mask_1d.cpu())  # Store attention mask for Mode 4/5
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
        'attention_mask_list': all_attention_masks,  # For Mode 4/5 padding filtering
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

            batch_eval_indices = []

            # Sample indices for each sample in batch
            batch_cond_idx = []
            batch_eval_idx = []
            batch_unseen_idx = []

            for i in range(batch_size):
                sample_attention_mask = attention_mask_1d[i]

                # Get valid (non-padding) positions
                valid_positions = [j for j in range(seq_len) if sample_attention_mask[j] == 1]

                # Generate boundary split with distribution (only from valid positions)
                cond_idx, eval_idx, unknown_idx = generate_boundary_conditioning_split(
                    seq_len,
                    boundary_block_sizes_distribution=boundary_distribution,
                    valid_positions=valid_positions
                )

                batch_cond_idx.append(cond_idx)
                batch_eval_idx.append(eval_idx)
                batch_unseen_idx.append(unknown_idx)
                batch_eval_indices.append(eval_idx)

            # Forward pass - model does augmentation internally
            logits, loss = model(
                input_ids=input_ids,
                conditional_idx=batch_cond_idx,
                evaluation_idx=batch_eval_idx,
                unseen_idx=batch_unseen_idx
            )

            # Accumulate loss (model returns mean over all eval tokens in batch)
            # Count actual evaluation tokens to properly weight the loss
            if loss is not None:
                total_eval_tokens = sum(len(eval_idx) for eval_idx in batch_eval_idx)
                total_loss += loss.item() * total_eval_tokens
                total_tokens += total_eval_tokens

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

            batch_eval_indices = []

            # Sample indices for each sample in batch
            batch_cond_idx = []
            batch_eval_idx = []
            batch_unseen_idx = []

            for i in range(batch_size):
                sample_attention_mask = attention_mask_1d[i]

                # Get valid (non-padding) positions
                valid_positions = [j for j in range(seq_len) if sample_attention_mask[j] == 1]

                # Use augmenter to sample indices (same as training)
                cond_idx, eval_idx, unknown_idx = augmenter.split_indices(seq_len, valid_positions=valid_positions)

                batch_cond_idx.append(cond_idx)
                batch_eval_idx.append(eval_idx)
                batch_unseen_idx.append(unknown_idx)
                batch_eval_indices.append(eval_idx)

            # Forward pass - model does augmentation internally
            logits, loss = model(
                input_ids=input_ids,
                conditional_idx=batch_cond_idx,
                evaluation_idx=batch_eval_idx,
                unseen_idx=batch_unseen_idx
            )

            # Accumulate loss (model returns mean over all eval tokens in batch)
            # Count actual evaluation tokens to properly weight the loss
            if loss is not None:
                total_eval_tokens = sum(len(eval_idx) for eval_idx in batch_eval_idx)
                total_loss += loss.item() * total_eval_tokens
                total_tokens += total_eval_tokens

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


def evaluate_mode4_cross_boundary(logits_list, labels_list, eval_indices_list, attention_mask_list):
    """
    Mode 4: Autoregressive evaluation on Mode 2's evaluation set

    Reuses Mode 1's logits, computes loss only on Mode 2's evaluation positions.
    No forward pass needed.

    Args:
        logits_list: List of logits from Mode 1 [batch_size, seq_len, vocab_size]
        labels_list: List of labels from Mode 1 [batch_size, seq_len]
        eval_indices_list: List of evaluation indices from Mode 2
        attention_mask_list: List of attention masks from Mode 1 for padding filtering

    Returns:
        dict with keys: loss, perplexity
    """
    total_loss = 0.0
    total_tokens = 0

    for logits_batch, labels_batch, eval_indices_batch, attention_mask_batch in zip(logits_list, labels_list, eval_indices_list, attention_mask_list):
        batch_size = logits_batch.shape[0]

        # Process each sample in batch
        for i in range(batch_size):
            logits = logits_batch[i]  # [seq_len, vocab_size]
            labels = labels_batch[i]  # [seq_len]
            attention_mask = attention_mask_batch[i]  # [seq_len]
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

            # Filter out padding tokens using attention_mask (not token ID)
            # This correctly includes EOS tokens (50256) which are valid content tokens
            # Old buggy code: non_padding_mask = (shift_labels != 50256)
            # This excluded EOS tokens since GPT-2's pad_token_id == eos_token_id == 50256
            attention_mask_shifted = attention_mask[1:]  # Shift to match labels
            non_padding_mask = (attention_mask_shifted == 1)
            valid_mask = eval_mask & non_padding_mask

            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    shift_logits[valid_mask].float(),  # Convert float16 back to float32 for computation
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


def evaluate_mode5_cross_training(logits_list, labels_list, eval_indices_list, attention_mask_list):
    """
    Mode 5: Autoregressive evaluation on Mode 3's evaluation set

    Reuses Mode 1's logits, computes loss only on Mode 3's evaluation positions.
    No forward pass needed.

    Args:
        logits_list: List of logits from Mode 1 [batch_size, seq_len, vocab_size]
        labels_list: List of labels from Mode 1 [batch_size, seq_len]
        eval_indices_list: List of evaluation indices from Mode 3
        attention_mask_list: List of attention masks from Mode 1 for padding filtering

    Returns:
        dict with keys: loss, perplexity
    """
    total_loss = 0.0
    total_tokens = 0

    for logits_batch, labels_batch, eval_indices_batch, attention_mask_batch in zip(logits_list, labels_list, eval_indices_list, attention_mask_list):
        batch_size = logits_batch.shape[0]

        # Process each sample in batch
        for i in range(batch_size):
            logits = logits_batch[i]  # [seq_len, vocab_size]
            labels = labels_batch[i]  # [seq_len]
            attention_mask = attention_mask_batch[i]  # [seq_len]
            eval_indices = eval_indices_batch[i] if isinstance(eval_indices_batch[0], list) else eval_indices_batch

            # Shift for next-token prediction
            shift_logits = logits[:-1]  # [seq_len-1, vocab_size]
            shift_labels = labels[1:]   # [seq_len-1]

            # Create mask for evaluation positions
            eval_mask = torch.zeros(shift_labels.shape[0], dtype=torch.bool, device=shift_labels.device)
            for eval_pos in eval_indices:
                if 0 <= eval_pos - 1 < eval_mask.shape[0]:
                    eval_mask[eval_pos - 1] = True

            # Filter out padding tokens using attention_mask (not token ID)
            # This correctly includes EOS tokens (50256) which are valid content tokens
            # Old buggy code: non_padding_mask = (shift_labels != 50256)
            # This excluded EOS tokens since GPT-2's pad_token_id == eos_token_id == 50256
            attention_mask_shifted = attention_mask[1:]  # Shift to match labels
            non_padding_mask = (attention_mask_shifted == 1)
            valid_mask = eval_mask & non_padding_mask

            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    shift_logits[valid_mask].float(),  # Convert float16 back to float32 for computation
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
        eval_indices_mode2_grouped,
        metrics_mode1['attention_mask_list']  # Pass attention_mask for correct padding filtering
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
        eval_indices_mode3_grouped,
        metrics_mode1['attention_mask_list']  # Pass attention_mask for correct padding filtering
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
