"""
BERT-specific Evaluation Modes

For BERT/DistilBERT, we use non-autoregressive masking where:
- At training: random 15% of tokens are masked
- At inference: we mask the tokens we want to predict and unmask them one at a time

This implements 5 evaluation modes adapted for BERT:
- Mode 1: Standard MLM (parallel prediction, random 15% masking)
- Mode 2: Boundary-constrained (iterative left-to-right unmasking)
- Mode 3: Training-distribution (iterative left-to-right unmasking)
- Mode 4: Boundary-constrained (parallel prediction - all masked at once)
- Mode 5: Training-distribution (parallel prediction - all masked at once)

Key differences from GPT-2 modes:
- BERT is bidirectional, not autoregressive
- Modes 2 vs 4: Same eval set, but iterative (2) vs parallel (4) prediction
- Modes 3 vs 5: Same eval set, but iterative (3) vs parallel (5) prediction
- Modes 4&5 test BERT's native parallel prediction capability
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

logger = logging.getLogger(__name__)


def evaluate_bert_mode1_joint_probability(model, dataloader, device, tokenizer, max_batches=None):
    """
    Mode 1: Non-autoregressive joint probability evaluation
    
    For BERT, we evaluate P(X_eval | X_cond) by:
    1. Masking all tokens in X_eval
    2. Running inference multiple times, unmasking left-to-right one token at a time
    3. Computing the joint probability as product of conditionals
    
    For simplicity in this baseline, we'll evaluate the standard MLM objective:
    - Randomly mask 15% of tokens (same as training)
    - Predict all masked tokens in parallel
    - This gives us a proxy for the model's joint prediction capability
    
    Args:
        model: BERT/DistilBERT model
        dataloader: Validation dataloader (should use MLMDataCollator)
        device: Device to run on
        tokenizer: Tokenizer (needed for mask_token_id)
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        dict with keys: loss, perplexity, num_batches, total_tokens
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    mask_token_id = tokenizer.mask_token_id
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass - model handles masking via labels
            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Count non-padding tokens that were masked (labels != -100)
            valid_mask = (labels != -100)
            num_tokens = valid_mask.sum().item()
            
            if num_tokens > 0:
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
            num_batches += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': num_batches,
        'total_tokens': total_tokens,
    }


def evaluate_bert_mode2_boundary_filling(model, dataloader, device, tokenizer, max_batches=None, trainer_args=None):
    """
    Mode 2: Boundary-constrained conditional evaluation for BERT
    
    - Conditioning: start block + end block (unmasked)
    - Evaluation: middle continuous part (masked, then unmasked left-to-right)
    - Non-autoregressive attention (full bidirectional)
    
    For each sequence:
    1. Sample boundary conditioning (start + end blocks)
    2. Mask the middle evaluation tokens
    3. Run inference iteratively, unmasking left-to-right
    4. Compute loss on each unmasked token given previous predictions
    
    Args:
        model: BERT/DistilBERT model
        dataloader: Validation dataloader
        device: Device to run on
        tokenizer: Tokenizer
        max_batches: Maximum number of batches to evaluate
        trainer_args: Trainer arguments containing boundary distribution parameters
        
    Returns:
        dict with keys: loss, perplexity, num_batches, total_tokens
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Create boundary distribution
    boundary_distribution = None
    if trainer_args is not None:
        boundary_cond_pct_min = getattr(trainer_args, 'mode2_boundary_cond_pct_min', 0.1)
        boundary_cond_pct_max = getattr(trainer_args, 'mode2_boundary_cond_pct_max', 0.3)
        
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
                
            # Get original input_ids (not masked version from collator)
            # We need to re-create clean input_ids from the batch
            input_ids = batch['input_ids'].to(device).clone()
            attention_mask = batch['attention_mask'].to(device)
            
            # Unmask all tokens first (in case collator already masked them)
            # This is a bit hacky - we assume labels contain the original tokens where they were masked
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(device)
                masked_positions = (labels != -100)
                input_ids[masked_positions] = labels[masked_positions]
            
            batch_size, seq_len = input_ids.shape
            
            # For each sample, do boundary-constrained evaluation
            for i in range(batch_size):
                sample_input_ids = input_ids[i:i+1].clone()
                sample_attention_mask = attention_mask[i:i+1]
                
                # Get valid positions (non-padding)
                valid_positions = [j for j in range(seq_len) if sample_attention_mask[0, j] == 1]
                
                if len(valid_positions) == 0:
                    continue
                
                # Generate boundary split
                cond_idx, eval_idx, _ = generate_boundary_conditioning_split(
                    seq_len,
                    boundary_block_sizes_distribution=boundary_distribution,
                    valid_positions=valid_positions
                )
                
                if len(eval_idx) == 0:
                    continue
                
                # Sort eval_idx for left-to-right unmasking
                eval_idx_sorted = sorted(eval_idx)


                
                # Save ground truth for evaluation tokens
                ground_truth = sample_input_ids[0, eval_idx_sorted].clone()
                
                # Mask all evaluation tokens
                sample_input_ids[0, eval_idx_sorted] = mask_token_id
                
                # Unmask left-to-right and compute loss
                sample_loss = 0.0
                for unmask_step, pos in enumerate(eval_idx_sorted):
                    # Forward pass
                    logits, _ = model(
                        input_ids=sample_input_ids,
                        attention_mask=sample_attention_mask,
                        labels=None
                    )
                    
                    # Get logits for this position
                    token_logits = logits[0, pos, :]  # [vocab_size]
                    true_token = ground_truth[unmask_step]
                    
                    # Compute loss
                    loss = F.cross_entropy(
                        token_logits.unsqueeze(0),
                        true_token.unsqueeze(0),
                        reduction='sum'
                    )
                    sample_loss += loss.item()
                    
                    # Unmask this token for next iteration
                    sample_input_ids[0, pos] = true_token
                
                total_loss += sample_loss
                total_tokens += len(eval_idx_sorted)
            
            num_batches += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': num_batches,
        'total_tokens': total_tokens,
    }


def evaluate_bert_mode3_training_dist(model, dataloader, device, tokenizer, augmenter, max_batches=None):
    """
    Mode 3: Training-distribution conditional evaluation for BERT
    
    Uses the same masking distribution as training (blockwise sampling).
    Similar to Mode 2 but with blockwise-sampled conditioning instead of boundary.
    
    Args:
        model: BERT/DistilBERT model
        dataloader: Validation dataloader
        device: Device to run on
        tokenizer: Tokenizer
        augmenter: Training augmenter (with blockwise sampling)
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        dict with keys: loss, perplexity, num_batches, total_tokens
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            # Get original input_ids (unmask if needed)
            input_ids = batch['input_ids'].to(device).clone()
            attention_mask = batch['attention_mask'].to(device)
            
            # Unmask all tokens first
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(device)
                masked_positions = (labels != -100)
                input_ids[masked_positions] = labels[masked_positions]
            
            batch_size, seq_len = input_ids.shape
            
            # For each sample, do blockwise-sampled evaluation
            for i in range(batch_size):
                sample_input_ids = input_ids[i:i+1].clone()
                sample_attention_mask = attention_mask[i:i+1]
                
                # Get valid positions
                valid_positions = [j for j in range(seq_len) if sample_attention_mask[0, j] == 1]
                
                if len(valid_positions) == 0:
                    continue
                
                # Use augmenter to sample indices (same as training)
                cond_idx, eval_idx, _ = augmenter.split_indices(seq_len, valid_positions=valid_positions)
                
                if len(eval_idx) == 0:
                    continue
                
                # Sort eval_idx for left-to-right unmasking
                eval_idx_sorted = sorted(eval_idx)

                # Save ground truth
                ground_truth = sample_input_ids[0, eval_idx_sorted].clone()
                
                # Mask all evaluation tokens
                sample_input_ids[0, eval_idx_sorted] = mask_token_id
                
                # Unmask left-to-right and compute loss
                sample_loss = 0.0
                for unmask_step, pos in enumerate(eval_idx_sorted):
                    # Forward pass
                    logits, _ = model(
                        input_ids=sample_input_ids,
                        attention_mask=sample_attention_mask,
                        labels=None
                    )
                    
                    # Get logits for this position
                    token_logits = logits[0, pos, :]
                    true_token = ground_truth[unmask_step]
                    
                    # Compute loss
                    loss = F.cross_entropy(
                        token_logits.unsqueeze(0),
                        true_token.unsqueeze(0),
                        reduction='sum'
                    )
                    sample_loss += loss.item()
                    
                    # Unmask this token
                    sample_input_ids[0, pos] = true_token
                
                total_loss += sample_loss
                total_tokens += len(eval_idx_sorted)
            
            num_batches += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': num_batches,
        'total_tokens': total_tokens,
    }


def evaluate_bert_mode4_parallel_boundary(model, dataloader, device, tokenizer, max_batches=None, trainer_args=None):
    """
    Mode 4: Parallel prediction on Mode 2's boundary-constrained evaluation set
    
    Same evaluation set as Mode 2 (boundary-constrained), but:
    - Mode 2: Iterative left-to-right unmasking (sequential)
    - Mode 4: Parallel prediction of all masked tokens (non-autoregressive)
    
    This tests if BERT can predict all evaluation tokens simultaneously
    when conditioned on boundary blocks.
    
    Args:
        model: BERT/DistilBERT model
        dataloader: Validation dataloader
        device: Device to run on
        tokenizer: Tokenizer
        max_batches: Maximum number of batches to evaluate
        trainer_args: Trainer arguments containing boundary distribution parameters
        
    Returns:
        dict with keys: loss, perplexity, num_batches, total_tokens
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Create boundary distribution
    boundary_distribution = None
    if trainer_args is not None:
        boundary_cond_pct_min = getattr(trainer_args, 'mode2_boundary_cond_pct_min', 0.1)
        boundary_cond_pct_max = getattr(trainer_args, 'mode2_boundary_cond_pct_max', 0.3)
        
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
                
            # Get original input_ids (unmask if needed)
            input_ids = batch['input_ids'].to(device).clone()
            attention_mask = batch['attention_mask'].to(device)
            
            # Unmask all tokens first
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(device)
                masked_positions = (labels != -100)
                input_ids[masked_positions] = labels[masked_positions]
            
            batch_size, seq_len = input_ids.shape
            
            # For each sample, do boundary-constrained evaluation (parallel)
            for i in range(batch_size):
                sample_input_ids = input_ids[i:i+1].clone()
                sample_attention_mask = attention_mask[i:i+1]
                
                # Get valid positions (non-padding)
                valid_positions = [j for j in range(seq_len) if sample_attention_mask[0, j] == 1]
                
                if len(valid_positions) == 0:
                    continue
                
                # Generate boundary split (same as Mode 2)
                cond_idx, eval_idx, _ = generate_boundary_conditioning_split(
                    seq_len,
                    boundary_block_sizes_distribution=boundary_distribution,
                    valid_positions=valid_positions
                )
                
                if len(eval_idx) == 0:
                    continue
                
                # Save ground truth for evaluation tokens
                ground_truth = sample_input_ids[0, eval_idx].clone()
                
                # Mask all evaluation tokens (PARALLEL - all at once)
                sample_input_ids[0, eval_idx] = mask_token_id
                
                # Single forward pass - predict all masked tokens in parallel
                logits, _ = model(
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    labels=None
                )
                
                # Compute loss for all evaluation tokens
                sample_loss = 0.0
                for idx, pos in enumerate(eval_idx):
                    token_logits = logits[0, pos, :]  # [vocab_size]
                    true_token = ground_truth[idx]
                    
                    loss = F.cross_entropy(
                        token_logits.unsqueeze(0),
                        true_token.unsqueeze(0),
                        reduction='sum'
                    )
                    sample_loss += loss.item()
                
                total_loss += sample_loss
                total_tokens += len(eval_idx)
            
            num_batches += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': num_batches,
        'total_tokens': total_tokens,
    }


def evaluate_bert_mode5_parallel_training(model, dataloader, device, tokenizer, augmenter, max_batches=None):
    """
    Mode 5: Parallel prediction on Mode 3's training-distribution evaluation set
    
    Same evaluation set as Mode 3 (training distribution), but:
    - Mode 3: Iterative left-to-right unmasking (sequential)
    - Mode 5: Parallel prediction of all masked tokens (non-autoregressive)
    
    This tests if BERT can predict all evaluation tokens simultaneously
    when conditioned on training-distribution blocks.
    
    Args:
        model: BERT/DistilBERT model
        dataloader: Validation dataloader
        device: Device to run on
        tokenizer: Tokenizer
        augmenter: Training augmenter (with blockwise sampling)
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        dict with keys: loss, perplexity, num_batches, total_tokens
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            # Get original input_ids (unmask if needed)
            input_ids = batch['input_ids'].to(device).clone()
            attention_mask = batch['attention_mask'].to(device)
            
            # Unmask all tokens first
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(device)
                masked_positions = (labels != -100)
                input_ids[masked_positions] = labels[masked_positions]
            
            batch_size, seq_len = input_ids.shape
            
            # For each sample, do blockwise-sampled evaluation (parallel)
            for i in range(batch_size):
                sample_input_ids = input_ids[i:i+1].clone()
                sample_attention_mask = attention_mask[i:i+1]
                
                # Get valid positions
                valid_positions = [j for j in range(seq_len) if sample_attention_mask[0, j] == 1]
                
                if len(valid_positions) == 0:
                    continue
                
                # Use augmenter to sample indices (same as Mode 3 and training)
                cond_idx, eval_idx, _ = augmenter.split_indices(seq_len, valid_positions=valid_positions)
                
                if len(eval_idx) == 0:
                    continue
                
                # Save ground truth
                ground_truth = sample_input_ids[0, eval_idx].clone()
                
                # Mask all evaluation tokens (PARALLEL - all at once)
                sample_input_ids[0, eval_idx] = mask_token_id
                
                # Single forward pass - predict all masked tokens in parallel
                logits, _ = model(
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    labels=None
                )
                
                # Compute loss for all evaluation tokens
                sample_loss = 0.0
                for idx, pos in enumerate(eval_idx):
                    token_logits = logits[0, pos, :]
                    true_token = ground_truth[idx]
                    
                    loss = F.cross_entropy(
                        token_logits.unsqueeze(0),
                        true_token.unsqueeze(0),
                        reduction='sum'
                    )
                    sample_loss += loss.item()
                
                total_loss += sample_loss
                total_tokens += len(eval_idx)
            
            num_batches += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': num_batches,
        'total_tokens': total_tokens,
    }


def evaluate_bert_all_modes(model, dataloader, device, tokenizer, augmenter, max_batches=None, trainer_args=None):
    """
    Run all 5 BERT evaluation modes
    
    BERT-adapted modes:
    - Mode 1: Standard MLM (parallel prediction, random 15% masking)
    - Mode 2: Boundary filling (iterative left-to-right unmasking)
    - Mode 3: Training distribution (iterative left-to-right unmasking)
    - Mode 4: Boundary filling (parallel prediction - all masked at once)
    - Mode 5: Training distribution (parallel prediction - all masked at once)
    
    Modes 2 vs 4, and 3 vs 5 use the same evaluation sets but differ in prediction strategy:
    - Modes 2&3: Sequential unmasking (iterative, context-dependent)
    - Modes 4&5: Parallel prediction (non-autoregressive, independent)
    
    Args:
        model: BERT/DistilBERT model
        dataloader: Validation dataloader
        device: Device to run on
        tokenizer: Tokenizer
        augmenter: Training augmenter
        max_batches: Maximum number of batches to evaluate
        trainer_args: Trainer arguments
        
    Returns:
        dict with metrics for 5 modes
    """
    logger.info("Running BERT 5-mode evaluation...")
    
    # Mode 1: Standard MLM evaluation (parallel masking)
    logger.info("  Mode 1: Joint probability (MLM baseline)...")
    metrics_mode1 = evaluate_bert_mode1_joint_probability(
        model, dataloader, device, tokenizer, max_batches
    )
    
    # Mode 2: Boundary filling (iterative unmasking)
    logger.info("  Mode 2: Boundary filling (iterative)...")
    metrics_mode2 = evaluate_bert_mode2_boundary_filling(
        model, dataloader, device, tokenizer, max_batches, trainer_args
    )
    
    # Mode 3: Training distribution (iterative unmasking)
    logger.info("  Mode 3: Training distribution (iterative)...")
    metrics_mode3 = evaluate_bert_mode3_training_dist(
        model, dataloader, device, tokenizer, augmenter, max_batches
    )
    
    # Mode 4: Boundary filling (parallel prediction)
    logger.info("  Mode 4: Boundary filling (parallel)...")
    metrics_mode4 = evaluate_bert_mode4_parallel_boundary(
        model, dataloader, device, tokenizer, max_batches, trainer_args
    )
    
    # Mode 5: Training distribution (parallel prediction)
    logger.info("  Mode 5: Training distribution (parallel)...")
    metrics_mode5 = evaluate_bert_mode5_parallel_training(
        model, dataloader, device, tokenizer, augmenter, max_batches
    )
    
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
