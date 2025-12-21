"""
Sigma GPT Trainer

Trainer for Sigma GPT model with arbitrary order generation.
Implements BaseTrainer interface for unified training pipeline.

Key Differences from Conditional Model:
- No special tokens needed ([M] mask token, BOS token)
- Uses SigmaGPTDataAdapter to convert data format
- Supports fair mode (~40% learning) and full mode (100% learning)
- Different forward signature: forward(idx, order, targets)
- Augmentation performed in train_step (not in DataLoader)

Usage:
    python train.py --model_type sigmagpt --sigmagpt_mode fair ...
"""

import sys
from pathlib import Path
import logging
import torch
import numpy as np
from transformers import GPT2Tokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.base_trainer import BaseTrainer
from model.sigmagpt_from_baseline import SigmaGPTModel
from model.sigmagpt_model_old import SigmaGPTOld
from model.config import get_config
from model.order_utils import apply_order, create_labels_fair, apply_labels_mask
from train.dataset import get_dataloader
from train.augmentation import ConditionalAugmenter
from train.sigmagpt_adapter import SigmaGPTDataAdapter
from train.blockwise_sampling import (
    uniform_num_blocks_distribution,
    uniform_block_sizes_distribution,
)
import random

logger = logging.getLogger(__name__)


def create_conditioning_distribution(cond_pct_min: float, cond_pct_max: float):
    """
    Create a conditioning distribution function with custom percentage range.

    Args:
        cond_pct_min: Minimum conditioning percentage (e.g., 0.0 for 0%)
        cond_pct_max: Maximum conditioning percentage (e.g., 0.4 for 40%)

    Returns:
        Distribution function that takes seq_len and returns num_conditioning
    """
    def distribution(seq_len: int) -> int:
        min_cond = max(0, int(seq_len * cond_pct_min))
        max_cond = max(min_cond, int(seq_len * cond_pct_max))
        max_cond = min(max_cond, seq_len - 1)  # Leave at least 1 for unknown
        return random.randint(min_cond, max_cond)

    return distribution


def create_evaluation_distribution(eval_pct_min: float, eval_pct_max: float):
    """
    Create an evaluation distribution function with custom percentage range.

    Args:
        eval_pct_min: Minimum evaluation percentage of unknown set (e.g., 1.0 for 100%)
        eval_pct_max: Maximum evaluation percentage of unknown set (e.g., 1.0 for 100%)

    Returns:
        Distribution function that takes available_len and returns num_evaluation
    """
    def distribution(available_len: int) -> int:
        if available_len == 0:
            return 0
        min_eval = max(1, int(available_len * eval_pct_min))
        max_eval = max(min_eval, int(available_len * eval_pct_max))
        max_eval = min(max_eval, available_len)
        return random.randint(min_eval, max_eval)

    return distribution


class SigmaGPTTrainer(BaseTrainer):
    """
    Trainer for Sigma GPT model

    Extends BaseTrainer with:
    - SigmaGPT model with double position encoding
    - SigmaGPTDataAdapter for data format conversion
    - ConditionalAugmenter for conditioning/evaluation splits
    - Support for fair and full training modes
    """

    def setup_model(self):
        """
        Setup Sigma GPT model

        Sigma GPT model does NOT require:
        - TokenManager (no special tokens)
        - Embedding resizing
        - Pretrained weight loading (simplified for now)

        This is simpler than conditional model setup.
        """
        logger.info("Setting up Sigma GPT model...")

        # Get model configuration
        self.config = get_config(self.args.model_config)

        # Simple tokenizer (no special tokens needed)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Tokenizer initialized (vocab_size: {len(self.tokenizer)})")

        # Get architecture type (default to 'new' if not specified)
        self.sigmagpt_arch = getattr(self.args, 'sigmagpt_arch', 'new')
        self.sigmagpt_eval_mode = getattr(self.args, 'sigmagpt_eval_mode', 'autoregressive')

        # Initialize Sigma GPT model based on architecture choice
        if self.sigmagpt_arch == 'old':
            # Old architecture: double position encoding (paper's original)
            self.model = SigmaGPTOld(self.config).to(self.device)
            logger.info(f"Sigma GPT OLD (double position encoding) initialized with {self.model.get_num_params() / 1e6:.2f}M parameters")
            logger.info(f"  Position embedding dim: n_embd // 2 = {self.config.n_embd // 2}")
        else:
            # New architecture: from baseline (standard position encoding)
            self.model = SigmaGPTModel(self.config).to(self.device)
            logger.info(f"Sigma GPT NEW (from baseline) initialized with {self.model.get_num_params() / 1e6:.2f}M parameters")

        # Store mode for logging
        self.sigmagpt_mode = self.args.sigmagpt_mode
        logger.info(f"Training mode: {self.sigmagpt_mode} "
                   f"({'~40% learning efficiency' if self.sigmagpt_mode == 'fair' else '100% learning efficiency'})")
        logger.info(f"Evaluation mode: {self.sigmagpt_eval_mode}")

    def setup_data(self):
        """
        Setup data loaders and augmentation components

        Creates:
        - ConditionalAugmenter: Samples conditioning/evaluation splits
        - SigmaGPTDataAdapter: Converts augmenter output to Sigma GPT format
        - Training and validation dataloaders (simple padding, no custom collate)
        """
        logger.info("Setting up data components...")

        # Sigma GPT uses full sequence length (no BOS token needed in augmenter)
        # Unlike conditional model, we don't need to reserve space for BOS
        # because Sigma GPT doesn't use the augmented sequence format

        # Create custom distribution functions based on CLI args
        num_conditioning_distribution = create_conditioning_distribution(
            self.args.cond_pct_min, self.args.cond_pct_max
        )
        num_evaluation_distribution = create_evaluation_distribution(
            self.args.eval_pct_min, self.args.eval_pct_max
        )
        logger.info(f"Distribution config: cond={self.args.cond_pct_min*100:.0f}%-{self.args.cond_pct_max*100:.0f}%, "
                   f"eval={self.args.eval_pct_min*100:.0f}%-{self.args.eval_pct_max*100:.0f}%")

        # Create ConditionalAugmenter with custom distribution functions
        from functools import partial
        
        self.augmenter = ConditionalAugmenter(
            mask_token_id=self.tokenizer.eos_token_id,  # Dummy (not actually used by Sigma GPT)
            bos_token_id=self.tokenizer.eos_token_id,    # BOS token for augmenter
            max_seq_len=self.config.max_seq_len,  # Use full sequence length
            tokenizer_pad_token_id=self.tokenizer.pad_token_id,
            num_conditioning_distribution=num_conditioning_distribution,
            num_blocks_distribution=partial(
                uniform_num_blocks_distribution,
                max_blocks=self.args.max_cond_blocks
            ),
            block_sizes_distribution=uniform_block_sizes_distribution,
            num_evaluation_distribution=num_evaluation_distribution,
            num_eval_blocks_distribution=partial(
                uniform_num_blocks_distribution,
                max_blocks=self.args.max_eval_blocks
            ),
            eval_block_sizes_distribution=uniform_block_sizes_distribution,
            conditioning_sampling=self.args.conditioning_sampling,
            evaluation_sampling=self.args.evaluation_sampling,
            ordering_mode=self.args.ordering_mode,  # Eric's ordering modes: temporal or random_scramble
        )
        logger.info(f"ConditionalAugmenter initialized "
                   f"(conditioning: {self.args.conditioning_sampling}, "
                   f"evaluation: {self.args.evaluation_sampling}, "
                   f"ordering: {self.args.ordering_mode})")

        # Create SigmaGPTDataAdapter
        self.adapter = SigmaGPTDataAdapter(mode=self.sigmagpt_mode)
        logger.info(f"SigmaGPTDataAdapter initialized (mode: {self.sigmagpt_mode})")

        # Create training dataloader (simple padding, no custom collate)
        # Note: Augmentation happens in train_step, not in DataLoader
        logger.info(f"Creating training dataloader "
                   f"(batch_size: {self.args.batch_size}, "
                   f"num_workers: {self.args.num_workers})...")

        self.train_loader = get_dataloader(
            config=self.config,
            split='train',
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            streaming=self.args.streaming,
            dataset_name=self.args.dataset_name,
            dataset_config=self.args.dataset_config,
            primary_dataset_only=self.args.primary_dataset_only,
            num_samples=self.args.num_train_samples
        )
        logger.info(f"Training dataloader created with {len(self.train_loader)} batches per epoch")

        # Create validation dataloader
        logger.info(f"Creating validation dataloader "
                   f"(batch_size: {self.args.eval_batch_size}, "
                   f"num_workers: {self.args.num_workers})...")

        self.val_loader = get_dataloader(
            config=self.config,
            split='validation',
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            streaming=False,
            dataset_name=self.args.dataset_name,
            dataset_config=self.args.dataset_config,
            primary_dataset_only=self.args.primary_dataset_only,
            num_samples=self.args.num_eval_samples
        )
        logger.info(f"Validation dataloader created with {len(self.val_loader)} batches")

    def train_step(self, batch):
        """
        Single training step for Sigma GPT

        Process:
        1. Get input_ids from batch
        2. Apply augmentation on CPU (loop over batch samples)
        3. Convert to Sigma GPT format using adapter
        4. Move to device and forward pass
        5. Return raw loss (scaling handled in base_trainer.train())

        Note: Augmentation happens here (not in DataLoader) because:
        - Augmenter output includes indices needed by adapter
        - Keeps implementation simple and matches reference
        """
        # Get input_ids
        input_ids = batch['input_ids'].to(self.device)

        # Data augmentation on CPU (to reduce GPU memory)
        # Must use augment_sequence (not augment_batch) to preserve indices
        input_ids_cpu = input_ids.cpu()
        batch_size = input_ids_cpu.size(0)

        aug_batch = []
        for i in range(batch_size):
            result = self.augmenter.augment_sequence(input_ids_cpu[i], device='cpu')
            aug_batch.append(result)

        # Convert to Sigma GPT format using adapter
        sigmagpt_batch = self.adapter.convert_batch(aug_batch, input_ids_cpu)

        # Move to device
        inputs = sigmagpt_batch['inputs'].to(self.device)
        order = sigmagpt_batch['order'].to(self.device)
        targets = sigmagpt_batch['targets'].to(self.device)

        # Forward pass
        logits, loss = self.model(idx=inputs, order=order, targets=targets)

        # Loss scaling is now handled in base_trainer.py train() method
        return loss

    def evaluate(self):
        """
        Evaluate Sigma GPT model on validation set using 5-mode evaluation

        Runs all 5 evaluation modes for fair comparison with conditional model:
        - Mode 1: Standard autoregressive (left-to-right)
        - Mode 2: Boundary filling (condition on boundaries, evaluate middle)
        - Mode 3: Training distribution (same random split as training)
        - Mode 4: Autoregressive on Mode 2's evaluation positions
        - Mode 5: Autoregressive on Mode 3's evaluation positions

        Returns:
            dict with keys for all 5 modes: mode1_loss, mode1_ppl, ..., mode5_ppl
        """
        from train.sigmagpt_evaluation_modes import sigmagpt_evaluate_all_modes

        # Get boundary conditioning percentage range for Mode 2
        boundary_pct_range = (
            getattr(self.args, 'mode2_boundary_cond_pct_min', 0.1),
            getattr(self.args, 'mode2_boundary_cond_pct_max', 0.3)
        )

        eval_results = sigmagpt_evaluate_all_modes(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            augmenter=self.augmenter,
            adapter=self.adapter,
            max_batches=self.args.max_eval_batches,
            boundary_cond_pct_range=boundary_pct_range
        )

        # Log all 5 modes
        logger.info(f"=" * 80)
        logger.info(f"Evaluation Results (Step {self.global_step})")
        logger.info(f"=" * 80)
        logger.info(f"Mode 1 (Autoregressive)   : loss={eval_results['mode1_loss']:.4f}, ppl={eval_results['mode1_ppl']:.2f}")
        logger.info(f"Mode 2 (Boundary Filling) : loss={eval_results['mode2_loss']:.4f}, ppl={eval_results['mode2_ppl']:.2f}")
        logger.info(f"Mode 3 (Training Dist)    : loss={eval_results['mode3_loss']:.4f}, ppl={eval_results['mode3_ppl']:.2f}")
        logger.info(f"Mode 4 (Auto on Boundary) : loss={eval_results['mode4_loss']:.4f}, ppl={eval_results['mode4_ppl']:.2f}")
        logger.info(f"Mode 5 (Auto on Training) : loss={eval_results['mode5_loss']:.4f}, ppl={eval_results['mode5_ppl']:.2f}")
        logger.info(f"-" * 80)
        logger.info(f"Comparisons:")
        logger.info(f"  Mode 2 vs 4 (Boundary):  Δ={eval_results['mode2_loss'] - eval_results['mode4_loss']:.4f} (negative = SigmaGPT better)")
        logger.info(f"  Mode 3 vs 5 (Training):  Δ={eval_results['mode3_loss'] - eval_results['mode5_loss']:.4f} (negative = SigmaGPT better)")
        logger.info(f"=" * 80)

        # Use Mode 3 loss as main criterion (same as conditional model)
        eval_results["loss"] = eval_results["mode3_loss"]

        return eval_results

    def _evaluate_autoregressive(self):
        """
        Mode 1: Standard autoregressive evaluation (paper's method)

        - Order: [0, 1, 2, ..., seq_len-1, seq_len] (standard left-to-right)
        - Conditioning: position 0 only
        - Evaluation: positions 1 to seq_len-1
        """
        logger.info("Running autoregressive evaluation (paper's method)...")
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= self.args.max_eval_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                batch_size, seq_len = input_ids.shape

                # Create autoregressive order: [0, 1, 2, ..., seq_len-1, seq_len]
                order = torch.arange(seq_len + 1, device=self.device, dtype=torch.long)
                order = order.unsqueeze(0).expand(batch_size, -1)  # (B, seq_len+1)

                # Apply order to get inputs and targets
                inputs, targets = apply_order(input_ids, order)

                # In autoregressive mode, all positions except first are evaluation
                # Create targets with ignore_index=-1 for conditioning position
                cond_size = 1
                mask = create_labels_fair(order, cond_size, seq_len)
                targets = apply_labels_mask(targets, mask)

                # Forward pass
                logits, loss = self.model(idx=inputs, order=order, targets=targets)

                # Count valid tokens (not -1)
                valid_tokens = (targets != -1).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

        # Compute average loss and perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss) if avg_loss < 20 else float('inf')

        logger.info(f"Autoregressive eval complete: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}")

        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }

    def _evaluate_training_dist(self):
        """
        Training distribution evaluation (same augmentation as training)

        Uses the same random augmentation as training.
        Expected result: train_loss ≈ eval_loss
        """
        logger.info("Running training-distribution evaluation...")
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= self.args.max_eval_batches:
                    break

                # Same augmentation process as training
                input_ids = batch['input_ids'].to(self.device)
                input_ids_cpu = input_ids.cpu()
                batch_size = input_ids_cpu.size(0)

                aug_batch = []
                for i in range(batch_size):
                    result = self.augmenter.augment_sequence(input_ids_cpu[i], device='cpu')
                    aug_batch.append(result)

                sigmagpt_batch = self.adapter.convert_batch(aug_batch, input_ids_cpu)

                inputs = sigmagpt_batch['inputs'].to(self.device)
                order = sigmagpt_batch['order'].to(self.device)
                targets = sigmagpt_batch['targets'].to(self.device)

                # Forward pass
                logits, loss = self.model(idx=inputs, order=order, targets=targets)

                # Accumulate loss (weighted by number of valid tokens)
                valid_tokens = (targets != -1).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

        # Compute average loss and perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')

        logger.info(f"Training-dist eval complete: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}")

        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }

    def get_csv_header(self):
        """
        Get CSV header for logging

        Format matches conditional model for fair comparison:
        - step, epoch
        - train_loss, train_perplexity
        - mode1_loss, mode1_ppl, ..., mode5_loss, mode5_ppl
        - learning_rate
        - sigmagpt_mode, ordering_mode (SigmaGPT-specific)
        """
        return [
            "step",
            "epoch",
            "train_loss",
            "train_perplexity",
            "mode1_loss",
            "mode1_ppl",
            "mode2_loss",
            "mode2_ppl",
            "mode3_loss",
            "mode3_ppl",
            "mode4_loss",
            "mode4_ppl",
            "mode5_loss",
            "mode5_ppl",
            "learning_rate",
            "sigmagpt_mode",
            "ordering_mode"
        ]

    def format_train_metrics(self, avg_loss, perplexity, lr):
        """
        Format training metrics for CSV logging

        During training step, eval metrics are empty.
        Matches conditional model's 5-mode format for fair comparison.
        """
        return {
            'train_loss': avg_loss,
            'train_perplexity': perplexity,
            'mode1_loss': '',
            'mode1_ppl': '',
            'mode2_loss': '',
            'mode2_ppl': '',
            'mode3_loss': '',
            'mode3_ppl': '',
            'mode4_loss': '',
            'mode4_ppl': '',
            'mode5_loss': '',
            'mode5_ppl': '',
            'learning_rate': lr,
            'sigmagpt_mode': self.sigmagpt_mode,
            'ordering_mode': getattr(self.args, 'ordering_mode', 'temporal')
        }

    def format_eval_metrics(self, eval_results):
        """
        Format evaluation metrics for CSV logging

        During evaluation step, train metrics are empty.
        Matches conditional model's 5-mode format for fair comparison.
        """
        return {
            'train_loss': '',
            'train_perplexity': '',
            'mode1_loss': eval_results['mode1_loss'],
            'mode1_ppl': eval_results['mode1_ppl'],
            'mode2_loss': eval_results['mode2_loss'],
            'mode2_ppl': eval_results['mode2_ppl'],
            'mode3_loss': eval_results['mode3_loss'],
            'mode3_ppl': eval_results['mode3_ppl'],
            'mode4_loss': eval_results['mode4_loss'],
            'mode4_ppl': eval_results['mode4_ppl'],
            'mode5_loss': eval_results['mode5_loss'],
            'mode5_ppl': eval_results['mode5_ppl'],
            'learning_rate': self.optimizer.param_groups[0]["lr"],
            'sigmagpt_mode': self.sigmagpt_mode,
            'ordering_mode': getattr(self.args, 'ordering_mode', 'temporal')
        }
