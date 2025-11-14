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
from model.sigmagpt_model import SigmaGPT
from model.config import get_config
from train.dataset import get_dataloader
from train.augmentation import ConditionalAugmenter
from train.sigmagpt_adapter import SigmaGPTDataAdapter
from train.blockwise_sampling import (
    uniform_num_conditioning_distribution,
    uniform_num_blocks_distribution,
    uniform_block_sizes_distribution,
    uniform_num_evaluation_distribution,
)

logger = logging.getLogger(__name__)


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

        # Initialize Sigma GPT model
        self.model = SigmaGPT(self.config).to(self.device)
        logger.info(f"Sigma GPT initialized with {self.model.get_num_params() / 1e6:.2f}M parameters")

        # Store mode for logging
        self.sigmagpt_mode = self.args.sigmagpt_mode
        logger.info(f"Training mode: {self.sigmagpt_mode} "
                   f"({'~40% learning efficiency' if self.sigmagpt_mode == 'fair' else '100% learning efficiency'})")

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

        # Create ConditionalAugmenter with distribution functions
        self.augmenter = ConditionalAugmenter(
            mask_token_id=self.tokenizer.eos_token_id,  # Dummy (not actually used by Sigma GPT)
            bos_token_id=self.tokenizer.eos_token_id,    # BOS token for augmenter
            max_seq_len=self.config.max_seq_len,  # Use full sequence length
            tokenizer_pad_token_id=self.tokenizer.pad_token_id,
            num_conditioning_distribution=uniform_num_conditioning_distribution,
            num_blocks_distribution=uniform_num_blocks_distribution,
            block_sizes_distribution=uniform_block_sizes_distribution,
            num_evaluation_distribution=uniform_num_evaluation_distribution,
            num_eval_blocks_distribution=uniform_num_blocks_distribution,
            eval_block_sizes_distribution=uniform_block_sizes_distribution,
            conditioning_sampling=self.args.conditioning_sampling,
            evaluation_sampling=self.args.evaluation_sampling,
            max_cond_blocks=self.args.max_cond_blocks,
            max_eval_blocks=self.args.max_eval_blocks,
        )
        logger.info(f"ConditionalAugmenter initialized "
                   f"(conditioning: {self.args.conditioning_sampling}, "
                   f"evaluation: {self.args.evaluation_sampling})")

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
        5. Return scaled loss

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

        # Scale loss for gradient accumulation
        loss = loss / self.args.gradient_accumulation_steps

        return loss

    def evaluate(self):
        """
        Evaluate Sigma GPT model on validation set

        Uses simple single-mode evaluation (not 5-mode like conditional model).
        Computes average loss and perplexity over validation batches.

        Returns:
            dict with keys:
            - "loss": float - average validation loss
            - "perplexity": float - perplexity (exp(loss))
        """
        logger.info("Running evaluation...")
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

        logger.info(f"Evaluation complete: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}")

        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }

    def get_csv_header(self):
        """
        Get CSV header for logging

        Format matches conditional model style but simplified:
        - step, epoch
        - train_loss, train_perplexity
        - eval_loss, eval_perplexity
        - learning_rate
        - sigmagpt_mode
        """
        return [
            "step",
            "epoch",
            "train_loss",
            "train_perplexity",
            "eval_loss",
            "eval_perplexity",
            "learning_rate",
            "sigmagpt_mode"
        ]

    def format_train_metrics(self, avg_loss, perplexity, lr):
        """
        Format training metrics for CSV logging

        During training step, eval metrics are empty.
        """
        return {
            'train_loss': f'{avg_loss:.6f}',
            'train_perplexity': f'{perplexity:.4f}',
            'eval_loss': '',
            'eval_perplexity': '',
            'learning_rate': f'{lr:.2e}',
            'sigmagpt_mode': self.sigmagpt_mode
        }

    def format_eval_metrics(self, eval_results):
        """
        Format evaluation metrics for CSV logging

        During evaluation step, train metrics are empty.
        """
        return {
            'train_loss': '',
            'train_perplexity': '',
            'eval_loss': f'{eval_results["loss"]:.6f}',
            'eval_perplexity': f'{eval_results["perplexity"]:.4f}',
            'learning_rate': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
            'sigmagpt_mode': self.sigmagpt_mode
        }
