"""
Baseline Trainer for Standard Autoregressive Language Modeling

Implements standard autoregressive language modeling (e.g., DistilGPT-2) as a baseline
for comparison with conditional probability models.

Key features:
- Standard left-to-right autoregressive training
- FP16 support via BaseTrainer (enabled with --fp16)
- Standard evaluation (single loss and perplexity)
- No special tokens or augmentation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
from torch.amp import autocast

from train.base_trainer import BaseTrainer
from model.config import get_config
from model.arbitrary_prob_gpt2 import GPT2Model
from train.dataset import get_dataloader

logger = logging.getLogger(__name__)


class BaselineTrainer(BaseTrainer):
    """
    Trainer for standard autoregressive baseline models

    Extends BaseTrainer with:
    - Standard model initialization (no special tokens)
    - Standard data loading (no augmentation)
    - Standard forward pass (FP16 handled by BaseTrainer)
    - Standard evaluation (single loss and perplexity)
    """

    def setup_model(self):
        """
        Setup standard model (no special tokens)

        Baseline model uses:
        - Standard GPT-2 tokenizer (via BaseTrainer helper)
        - Standard model initialization (no special tokens)
        """
        logger.info("Setting up baseline model...")

        # Get model configuration
        self.config = get_config(self.args.model_config)

        # Set gradient checkpointing from args
        if hasattr(self.args, 'gradient_checkpointing'):
            self.config.gradient_checkpointing = self.args.gradient_checkpointing
            if self.config.gradient_checkpointing:
                logger.info("Gradient checkpointing: ENABLED (memory optimization)")

        # Set position encoding type from args (learned or rope)
        if hasattr(self.args, 'position_encoding_type'):
            self.config.position_encoding_type = self.args.position_encoding_type
            logger.info(f"Position encoding type: {self.config.position_encoding_type}")
        if hasattr(self.args, 'rope_base'):
            self.config.rope_base = self.args.rope_base

        # Create standard model (no special tokens)
        self.model = GPT2Model(self.config).to(self.device)

        # Standard tokenizer (using BaseTrainer helper)
        self.tokenizer = self._setup_tokenizer()
        logger.info(f"Pad token ID: {self.tokenizer.pad_token_id}")

        total_params = self.model.get_num_params()
        logger.info(f"Model parameters: {total_params/1e6:.2f}M")

    def setup_data(self):
        """
        Setup standard data loaders (no augmentation)

        Uses standard Wikipedia data loading without any conditional augmentation.
        """
        logger.info("Loading dataset...")

        # Training dataloader
        self.train_loader = get_dataloader(
            config=self.config,
            split="train",
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            streaming=getattr(self.args, 'streaming', False),
            num_samples=self.args.num_train_samples,
            seed=getattr(self.args, 'seed', 42)  # For deterministic checkpoint resume
        )

        logger.info(f"Training batches: {len(self.train_loader) if hasattr(self.train_loader, '__len__') else 'streaming'}")

        # Validation dataloader
        if self.args.do_eval:
            self.val_loader = get_dataloader(
                config=self.config,
                split="validation",
                batch_size=self.args.eval_batch_size,
                num_workers=self.args.num_workers,
                streaming=False,
                num_samples=self.args.num_eval_samples
            )

            logger.info(f"Validation batches: {len(self.val_loader)}")
        else:
            self.val_loader = None

    def train_step(self, batch):
        """
        Single training step with standard autoregressive forward pass

        Supports optional mixed precision training (FP16).

        Args:
            batch: Batch from dataloader containing input_ids and attention_mask

        Returns:
            loss: Raw training loss (scaling handled in base_trainer.train())
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Prepare labels: set padding tokens to -100 so they're ignored in loss
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Forward pass with optional mixed precision
        if self.use_amp:
            with autocast('cuda'):
                logits, loss = self.model(input_ids, labels=labels)
        else:
            logits, loss = self.model(input_ids, labels=labels)

        # Loss scaling is now handled in base_trainer.py train() method
        return loss

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluation using Mode 1 (autoregressive) from unified evaluation framework

        Uses evaluate_all_modes with modes=[1] for consistency with other trainers.

        Returns:
            dict: Evaluation results containing mode1_loss, mode1_ppl, and "loss" for compatibility
        """
        from train.evaluation_modes import evaluate_all_modes

        eval_results = evaluate_all_modes(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            augmenter=None,  # Baseline doesn't need augmenter
            modes=[1],       # Only Mode 1 (autoregressive)
            max_batches=getattr(self.args, 'max_eval_batches', None)
        )

        # Log results
        logger.info(f"Mode 1 (Autoregressive): loss={eval_results['mode1_loss']:.4f}, ppl={eval_results['mode1_ppl']:.2f}")

        # Set "loss" for best model selection (used by BaseTrainer)
        eval_results["loss"] = eval_results["mode1_loss"]
        eval_results["perplexity"] = eval_results["mode1_ppl"]

        self.model.train()
        return eval_results
