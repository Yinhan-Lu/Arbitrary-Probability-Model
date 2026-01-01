"""
Evaluation Modes for Diffusion Model

Implements 5-mode evaluation adapted for discrete diffusion models:
- Mode 1: Standard evaluation (all positions may be masked)
- Mode 2: Boundary-constrained conditional evaluation
- Mode 3: Training-distribution conditional evaluation
- Mode 4: Cross-comparison with Mode 2
- Mode 5: Cross-comparison with Mode 3

Key difference from autoregressive evaluation:
- Uses importance sampling over noise levels to estimate NLL
- Conditioning is achieved by fixing X_c positions during diffusion
- Multiple forward passes at different timesteps are averaged
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


def compute_diffusion_nll(model, noise_schedule, x_0, conditioning_mask, eval_mask,
                          mask_token_id, device, num_samples=10):
    """
    Estimate negative log-likelihood for diffusion model.

    Uses importance sampling over noise levels:
    - Sample multiple timesteps t
    - Add noise to non-conditioning positions
    - Get model predictions
    - Compute NLL on evaluation positions

    Args:
        model: Diffusion model
        noise_schedule: NoiseSchedule object
        x_0: Original clean sequence (batch_size, seq_len)
        conditioning_mask: Boolean mask of conditioning positions (batch_size, seq_len)
                          True = conditioning position (never masked)
        eval_mask: Boolean mask of evaluation positions (batch_size, seq_len)
                  True = position where we compute NLL
        mask_token_id: Token ID for [MASK]
        device: Computing device
        num_samples: Number of timesteps to sample for averaging

    Returns:
        total_nll: Total negative log-likelihood (sum over eval positions)
        num_tokens: Number of evaluation tokens
    """
    batch_size, seq_len = x_0.shape
    total_nll = 0.0
    num_tokens = eval_mask.sum().item()

    if num_tokens == 0:
        return 0.0, 0

    with torch.no_grad():
        for _ in range(num_samples):
            # Sample random timesteps
            t = noise_schedule.sample_timesteps(batch_size, device=device)

            # Apply noise with conditioning fixed
            x_t, noise_mask = noise_schedule.add_noise_conditional_fast(
                x_0, t, mask_token_id, conditioning_mask
            )

            # Forward pass
            logits = model(x_t, t)  # (batch_size, seq_len, vocab_size)

            # Compute NLL only on evaluation positions
            nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                x_0.view(-1),
                reduction='none'
            ).view(batch_size, seq_len)

            # Sum over evaluation positions
            eval_nll = (nll * eval_mask.float()).sum()
            total_nll += eval_nll.item()

    # Average over samples
    avg_nll = total_nll / num_samples

    return avg_nll, num_tokens


def evaluate_diffusion_mode1(model, noise_schedule, dataloader, device,
                             mask_token_id, max_batches=None, num_samples=10):
    """
    Mode 1: Standard diffusion evaluation

    - All positions may be masked during diffusion
    - No explicit conditioning (except padding)
    - NLL computed on all valid tokens

    Args:
        model: Diffusion model
        noise_schedule: NoiseSchedule object
        dataloader: Validation dataloader
        device: Computing device
        mask_token_id: Token ID for [MASK]
        max_batches: Maximum batches to evaluate
        num_samples: Number of timesteps for NLL estimation

    Returns:
        dict with keys: loss, perplexity, total_tokens
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size, seq_len = input_ids.shape

            # No conditioning - empty mask
            conditioning_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

            # Evaluate on all valid (non-padding) positions
            eval_mask = attention_mask.bool()

            # Compute NLL
            nll, n_tokens = compute_diffusion_nll(
                model, noise_schedule, input_ids, conditioning_mask, eval_mask,
                mask_token_id, device, num_samples
            )

            total_nll += nll
            total_tokens += n_tokens
            num_batches += 1

    avg_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
        'num_batches': num_batches
    }


def evaluate_diffusion_mode2(model, noise_schedule, dataloader, device, augmenter,
                             mask_token_id, max_batches=None, num_samples=10,
                             trainer_args=None):
    """
    Mode 2: Boundary-constrained conditional evaluation

    - Conditioning = start block + end block (boundaries)
    - Evaluation = middle continuous part
    - Boundaries are fixed during diffusion

    Args:
        model: Diffusion model
        noise_schedule: NoiseSchedule object
        dataloader: Validation dataloader
        device: Computing device
        augmenter: Augmenter (for split_indices method signature)
        mask_token_id: Token ID for [MASK]
        max_batches: Maximum batches to evaluate
        num_samples: Number of timesteps for NLL estimation
        trainer_args: Trainer arguments containing boundary distribution parameters

    Returns:
        dict with keys: loss, perplexity, total_tokens, eval_indices_list
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    all_eval_indices = []
    num_batches = 0

    # Create boundary distribution function
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

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size, seq_len = input_ids.shape

            # Generate boundary split for each sample
            batch_cond_idx = []
            batch_eval_idx = []

            for i in range(batch_size):
                sample_attention_mask = attention_mask[i]
                valid_positions = [j for j in range(seq_len) if sample_attention_mask[j] == 1]

                # Generate boundary split
                cond_idx, eval_idx, _ = generate_boundary_conditioning_split(
                    seq_len,
                    boundary_block_sizes_distribution=boundary_distribution,
                    valid_positions=valid_positions
                )
                batch_cond_idx.append(cond_idx)
                batch_eval_idx.append(eval_idx)

            # Create masks from indices
            conditioning_mask = _indices_to_mask(batch_cond_idx, seq_len, device)
            eval_mask = _indices_to_mask(batch_eval_idx, seq_len, device)

            # Compute NLL
            nll, n_tokens = compute_diffusion_nll(
                model, noise_schedule, input_ids, conditioning_mask, eval_mask,
                mask_token_id, device, num_samples
            )

            total_nll += nll
            total_tokens += n_tokens
            all_eval_indices.extend(batch_eval_idx)
            num_batches += 1

    avg_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
        'num_batches': num_batches,
        'eval_indices_list': all_eval_indices
    }


def evaluate_diffusion_mode3(model, noise_schedule, dataloader, device, augmenter,
                             mask_token_id, max_batches=None, num_samples=10):
    """
    Mode 3: Training-distribution conditional evaluation

    - Uses same augmenter as training (blockwise sampling)
    - Conditioning = randomly sampled blocks
    - Evaluation = randomly sampled evaluation blocks

    Args:
        model: Diffusion model
        noise_schedule: NoiseSchedule object
        dataloader: Validation dataloader
        device: Computing device
        augmenter: Training augmenter (with blockwise sampling)
        mask_token_id: Token ID for [MASK]
        max_batches: Maximum batches to evaluate
        num_samples: Number of timesteps for NLL estimation

    Returns:
        dict with keys: loss, perplexity, total_tokens, eval_indices_list
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    all_eval_indices = []
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size, seq_len = input_ids.shape

            # Generate splits using training augmenter
            batch_cond_idx = []
            batch_eval_idx = []

            for i in range(batch_size):
                sample_attention_mask = attention_mask[i]
                valid_positions = [j for j in range(seq_len) if sample_attention_mask[j] == 1]

                # Use augmenter to sample indices (same as training)
                cond_idx, eval_idx, _ = augmenter.split_indices(seq_len, valid_positions=valid_positions)
                batch_cond_idx.append(cond_idx)
                batch_eval_idx.append(eval_idx)

            # Create masks from indices
            conditioning_mask = _indices_to_mask(batch_cond_idx, seq_len, device)
            eval_mask = _indices_to_mask(batch_eval_idx, seq_len, device)

            # Compute NLL
            nll, n_tokens = compute_diffusion_nll(
                model, noise_schedule, input_ids, conditioning_mask, eval_mask,
                mask_token_id, device, num_samples
            )

            total_nll += nll
            total_tokens += n_tokens
            all_eval_indices.extend(batch_eval_idx)
            num_batches += 1

    avg_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
        'num_batches': num_batches,
        'eval_indices_list': all_eval_indices
    }


def _indices_to_mask(indices_list, seq_len, device):
    """
    Convert list of index lists to boolean mask tensor.

    Args:
        indices_list: List of lists of indices
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Boolean mask of shape (batch_size, seq_len)
    """
    batch_size = len(indices_list)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for i, indices in enumerate(indices_list):
        if len(indices) > 0:
            if isinstance(indices, list):
                indices = torch.tensor(indices, device=device, dtype=torch.long)
            mask[i, indices] = True

    return mask


def diffusion_evaluate_all_modes(model, noise_schedule, dataloader, device, augmenter,
                                  mask_token_id, max_batches=None, num_nll_samples=10,
                                  trainer_args=None, modes=None):
    """
    Run all evaluation modes for diffusion model.

    Args:
        model: Diffusion model
        noise_schedule: NoiseSchedule object
        dataloader: Validation dataloader
        device: Computing device
        augmenter: Training augmenter
        mask_token_id: Token ID for [MASK]
        max_batches: Maximum batches per mode
        num_nll_samples: Number of timesteps for NLL estimation
        trainer_args: Trainer arguments
        modes: List of modes to run (default: [1, 2, 3])

    Returns:
        dict with metrics for all requested modes
    """
    if modes is None:
        modes = [1, 2, 3]  # Diffusion only supports modes 1-3 directly
        # Modes 4-5 would require autoregressive baseline comparison

    results = {}

    # Mode 1: Standard evaluation
    if 1 in modes:
        logger.info("  Mode 1: Standard diffusion evaluation...")
        mode1_results = evaluate_diffusion_mode1(
            model, noise_schedule, dataloader, device, mask_token_id,
            max_batches, num_nll_samples
        )
        results['mode1_loss'] = mode1_results['loss']
        results['mode1_ppl'] = mode1_results['perplexity']
        results['mode1_tokens'] = mode1_results['total_tokens']

    # Mode 2: Boundary filling
    if 2 in modes:
        logger.info("  Mode 2: Boundary-constrained evaluation...")
        mode2_results = evaluate_diffusion_mode2(
            model, noise_schedule, dataloader, device, augmenter, mask_token_id,
            max_batches, num_nll_samples, trainer_args
        )
        results['mode2_loss'] = mode2_results['loss']
        results['mode2_ppl'] = mode2_results['perplexity']
        results['mode2_tokens'] = mode2_results['total_tokens']

    # Mode 3: Training distribution
    if 3 in modes:
        logger.info("  Mode 3: Training distribution evaluation...")
        mode3_results = evaluate_diffusion_mode3(
            model, noise_schedule, dataloader, device, augmenter, mask_token_id,
            max_batches, num_nll_samples
        )
        results['mode3_loss'] = mode3_results['loss']
        results['mode3_ppl'] = mode3_results['perplexity']
        results['mode3_tokens'] = mode3_results['total_tokens']

    # Modes 4 and 5 require autoregressive baseline logits
    # For fair comparison, these would need to run the autoregressive model first
    # and then compute diffusion NLL on the same positions
    # For now, we leave these empty for diffusion
    if 4 in modes:
        results['mode4_loss'] = ''
        results['mode4_ppl'] = ''
        results['mode4_tokens'] = 0

    if 5 in modes:
        results['mode5_loss'] = ''
        results['mode5_ppl'] = ''
        results['mode5_tokens'] = 0

    return results


if __name__ == "__main__":
    # Test evaluation modes
    print("=" * 80)
    print("Testing Diffusion Evaluation Modes")
    print("=" * 80)

    import torch
    from functools import partial

    # Create mock components for testing
    from model.diffusion_gpt2 import DiffusionGPT2Model
    from model.diffusion_utils import NoiseSchedule
    from model.arbitrary_prob_gpt2 import GPT2Config
    from train.augmentation import ConditionalAugmenter
    from train.blockwise_sampling import (
        uniform_num_conditioning_distribution,
        uniform_num_blocks_distribution,
        uniform_block_sizes_distribution,
        uniform_num_evaluation_distribution,
    )

    device = torch.device('cpu')

    # Create small model
    config = GPT2Config(
        vocab_size=50258,
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=64
    )
    mask_token_id = 50257
    model = DiffusionGPT2Model(config, mask_token_id=mask_token_id, num_timesteps=100)
    noise_schedule = NoiseSchedule(num_timesteps=100)

    # Create augmenter
    augmenter = ConditionalAugmenter(
        mask_token_id=mask_token_id,
        bos_token_id=50256,
        max_seq_len=64,
        cond_pct_max=0.3,
        num_conditioning_distribution=partial(
            uniform_num_conditioning_distribution,
            conditioning_percentage_range=(0.1, 0.3)
        ),
        num_blocks_distribution=uniform_num_blocks_distribution,
        block_sizes_distribution=uniform_block_sizes_distribution,
        num_evaluation_distribution=partial(
            uniform_num_evaluation_distribution,
            evaluation_percentage_range=(0.2, 0.3)
        ),
        num_eval_blocks_distribution=uniform_num_blocks_distribution,
        eval_block_sizes_distribution=uniform_block_sizes_distribution,
    )

    # Create mock batch
    batch_size = 4
    seq_len = 32
    x_0 = torch.randint(0, 50257, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    print("\n1. Testing compute_diffusion_nll...")
    conditioning_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    conditioning_mask[:, :5] = True  # First 5 positions are conditioning
    eval_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    eval_mask[:, 10:20] = True  # Positions 10-19 are evaluation

    nll, n_tokens = compute_diffusion_nll(
        model, noise_schedule, x_0, conditioning_mask, eval_mask,
        mask_token_id, device, num_samples=3
    )
    print(f"  NLL: {nll:.4f}")
    print(f"  Tokens: {n_tokens}")

    print("\n2. Testing _indices_to_mask...")
    indices = [[0, 1, 5], [2, 3], [4], []]
    mask = _indices_to_mask(indices, seq_len, device)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Sum per sample: {mask.sum(dim=1).tolist()}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
