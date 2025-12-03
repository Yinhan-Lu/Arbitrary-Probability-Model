"""
Baseline Trainer for Standard Autoregressive Language Modeling

Implements standard autoregressive language modeling (e.g., DistilGPT-2) as a baseline
for comparison with conditional probability models.

Key features:
- Standard left-to-right autoregressive training
- Optional mixed precision training (FP16)
- Standard evaluation (single loss and perplexity)
- No special tokens or augmentation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from transformers import GPT2Tokenizer

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
    - Standard forward pass with optional FP16
    - Standard evaluation (single loss and perplexity)
    """

    def setup_model(self):
        """
        Setup standard model (no special tokens)

        Baseline model uses:
        - Standard GPT-2 tokenizer
        - Standard model initialization (no special tokens)
        - Optional mixed precision training (FP16)
        """
        logger.info("Setting up baseline model...")

        # Get model configuration
        self.config = get_config(self.args.model_config)

        # Create standard model (no special tokens)
        self.model = GPT2Model(self.config).to(self.device)

        # Standard tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Tokenizer vocabulary size: {len(self.tokenizer)}")
        logger.info(f"Pad token ID: {self.tokenizer.pad_token_id}")

        # Setup mixed precision training
        self.use_amp = hasattr(self.args, 'fp16') and self.args.fp16 and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Using mixed precision training (FP16)")
        else:
            logger.info("Using full precision training (FP32)")

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
            num_samples=self.args.num_train_samples
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
        Standard evaluation on validation set

        Computes average loss and perplexity over validation batches.

        Returns:
            dict: Evaluation results containing "loss" and "perplexity"
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        logger.info("Running evaluation...")

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Prepare labels: set padding tokens to -100 so they're ignored in loss
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            # Forward pass
            logits, loss = self.model(input_ids, labels=labels)

            total_loss += loss.item()
            num_batches += 1

            # Limit evaluation batches if specified
            if self.args.max_eval_batches > 0 and num_batches >= self.args.max_eval_batches:
                break

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        logger.info(f"Validation Loss: {avg_loss:.4f} | Validation PPL: {perplexity:.2f}")

        self.model.train()

        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }

    def get_csv_header(self):
        """
        Get CSV header for baseline model logging

        Returns:
            list: Column names for standard baseline logging
        """
        return [
            "step",
            "epoch",
            "train_loss",
            "train_perplexity",
            "val_loss",
            "val_perplexity",
            "learning_rate"
        ]

    def format_train_metrics(self, avg_loss, perplexity, lr):
        """
        Format training metrics for CSV logging

        Args:
            avg_loss: Average training loss
            perplexity: Training perplexity
            lr: Current learning rate

        Returns:
            dict: Metrics dictionary with empty validation columns
        """
        return {
            'train_loss': avg_loss,
            'train_perplexity': perplexity,
            'val_loss': '',
            'val_perplexity': '',
            'learning_rate': lr
        }

    def format_eval_metrics(self, eval_results):
        """
        Format evaluation metrics for CSV logging

        Args:
            eval_results: Results from evaluate() containing loss and perplexity

        Returns:
            dict: Metrics dictionary with validation results
        """
        return {
            'train_loss': '',
            'train_perplexity': '',
            'val_loss': eval_results['loss'],
            'val_perplexity': eval_results['perplexity'],
            'learning_rate': self.optimizer.param_groups[0]["lr"]
        }

    def train(self):
        """
        Override train method to handle FP16-specific gradient operations

        This override is needed because FP16 requires special handling:
        - Gradient scaling for backward pass
        - Unscaling before gradient clipping
        - Scaler update after optimizer step
        """
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)
        logger.info(f"Model parameters: {self.model.get_num_params() / 1e6:.2f}M")
        logger.info(f"Training samples: {len(self.train_loader.dataset) if hasattr(self.train_loader.dataset, '__len__') else 'unknown'}")
        logger.info(f"Batch size: {self.args.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps}")
        logger.info(f"Number of epochs: {self.args.num_epochs}")
        logger.info(f"Total training steps: {self.total_steps}")
        logger.info("=" * 80)

        self.model.train()
        running_loss = 0
        running_batch_count = 0  # Track actual batch count for accurate averaging
        self.optimizer.zero_grad()

        for epoch in range(self.args.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")

            for batch_idx, batch in enumerate(self.train_loader):
                # Calculate accumulation flags FIRST (needed for loss scaling)
                is_accum_step = (batch_idx + 1) % self.args.gradient_accumulation_steps == 0
                is_last_batch = (batch_idx + 1) == len(self.train_loader)

                # Calculate actual number of accumulated batches for this step
                # (handles partial accumulation at epoch boundaries)
                if is_last_batch and not is_accum_step:
                    num_accumulated = (batch_idx % self.args.gradient_accumulation_steps) + 1
                else:
                    num_accumulated = self.args.gradient_accumulation_steps

                # Training step (model-specific implementation)
                loss = self.train_step(batch)

                # Scale loss by actual accumulated steps (HuggingFace standard)
                scaled_loss = loss / num_accumulated

                # Backward pass (FP16-aware)
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Accumulate unscaled loss for logging
                running_loss += loss.item()
                running_batch_count += 1

                if is_accum_step or is_last_batch:
                    # Gradient clipping (FP16-aware)
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )

                    # Optimizer step (FP16-aware)
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.args.logging_steps == 0:
                        self._log_training_metrics(running_loss, running_batch_count)
                        running_loss = 0
                        running_batch_count = 0

                    # Evaluation
                    if self.args.do_eval and self.global_step % self.args.eval_steps == 0:
                        logger.info(f"\nEvaluating at step {self.global_step}...")
                        eval_results = self.evaluate()
                        self._log_evaluation_metrics(eval_results)

                        # Save best model
                        if eval_results["loss"] < self.best_val_loss:
                            self.best_val_loss = eval_results["loss"]
                            logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                            self._save_checkpoint("best_model")

                        self.model.train()

                    # Save checkpoint
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(f"checkpoint_step_{self.global_step}")

        # Final checkpoint
        logger.info("\nTraining completed!")
        self._save_checkpoint("final_model")
        logger.info(f"Final model saved to {self.checkpoint_dir / 'final_model.pt'}")
