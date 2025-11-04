"""
Training Script for Arbitrary Conditional Probability Model

Trains a GPT-2 model to handle arbitrary conditional probability queries:
P(X_e | X_c) where X_e is evaluation set and X_c is conditioning set

Key features:
- Random conditioning/evaluation split during training
- Custom attention masks (attend to all conditions, block unknowns)
- Loss only on evaluation positions
- Support for fine-tuning from pretrained models
"""

import os
import sys
import argparse
import logging
import csv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model.config import get_config
from model.arbitrary_prob_gpt2 import GPT2Model
from model.token_manager import TokenManager
from train.dataset import get_dataloader, create_augment_collate_fn, create_simple_collate_fn
from train.augmentation import ConditionalAugmenter
from train.blockwise_sampling import (
    uniform_num_conditioning_distribution,
    uniform_num_blocks_distribution,
    uniform_block_sizes_distribution,
    uniform_num_evaluation_distribution,
    generate_boundary_conditioning_split,
)
from train.evaluation_modes import evaluate_all_modes
from functools import partial
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ConditionalTrainer:
    """
    Trainer for arbitrary conditional probability modeling
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Create experiment directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{args.exp_name}_{self.timestamp}"
        self.exp_dir = Path(args.output_dir) / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        logger.info(f"Experiment directory: {self.exp_dir}")
        logger.info(f"Device: {self.device}")

        # Initialize token manager and model
        self._setup_model()

        # Setup data
        self._setup_data()

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Initialize CSV logger
        self.csv_log_file = self.log_dir / "metrics.csv"
        self._init_csv_logger()

    def _setup_model(self):
        """Initialize model with special tokens"""
        logger.info("Setting up model with special tokens...")

        # Get model configuration
        self.config = get_config(self.args.model_config)

        # Initialize token manager
        self.token_manager = TokenManager(
            add_mask_token=True,
            add_bos_token=False  # Reuse EOS as BOS
        )

        # Get tokenizer and special token IDs
        self.tokenizer = self.token_manager.get_tokenizer()
        special_tokens = self.token_manager.get_special_token_ids()

        self.mask_token_id = special_tokens["mask_token_id"]
        self.bos_token_id = special_tokens["bos_token_id"]

        logger.info(f"Special tokens: Mask={self.mask_token_id}, BOS={self.bos_token_id}")

        # Create model
        self.model = GPT2Model(self.config).to(self.device)

        # Load pretrained weights if specified
        if self.args.pretrained_model_path:
            self._load_pretrained_weights(self.args.pretrained_model_path)

        # Resize embeddings to include new tokens
        self.model = self.token_manager.resize_model_embeddings(self.model)

        total_params = self.model.get_num_params()
        logger.info(f"Model parameters: {total_params/1e6:.2f}M")

    def _setup_data(self):
        """Setup data loaders"""
        logger.info("Loading dataset...")

        # Dataset configuration for prefix conditioning
        # Design: Total 1024 positions (0-1023), including BOS at position 0
        # BOS (1 token) + Body (1023 tokens) = 1024 total positions
        # Conditioning tokens are extra prefix: total_len = N_cond + 1024
        # Position IDs: BOS uses 0, Body uses 1-1023, Conditioning reuses 1-1023

        import math
        self.body_seq_len = self.config.max_seq_len - 1  # Reserve position 0 for BOS
        logger.info(f"Dataset configuration for prefix conditioning:")
        logger.info(f"  Model max_seq_len: {self.config.max_seq_len} (positions 0-{self.config.max_seq_len-1})")
        logger.info(f"  Dataset max_length: {self.body_seq_len} (body tokens, positions 1-{self.config.max_seq_len-1})")
        logger.info(f"  Max conditioning percentage: {self.args.cond_pct_max}")

        # Calculate expected augmented sequence length
        max_n_cond = math.ceil(self.body_seq_len * self.args.cond_pct_max)
        aug_max_len = max_n_cond + self.config.max_seq_len  # prefix + (BOS + body)
        logger.info(f"  Expected max augmented length: {aug_max_len}")
        logger.info(f"    = {max_n_cond} (prefix) + {self.config.max_seq_len} (BOS + body)")

        # Create dataset config with body_seq_len
        from copy import copy
        dataset_config = copy(self.config)
        dataset_config.max_seq_len = self.body_seq_len  # Dataloader provides body tokens only

        # Create conditional augmenter BEFORE dataloaders
        # This allows us to use augment-first, then dynamic padding approach
        logger.info("Creating augmenter with distribution-based sampling")
        logger.info(f"  Conditioning percentage range: [{self.args.cond_pct_min}, {self.args.cond_pct_max}]")
        logger.info(f"  Evaluation percentage range: [{self.args.eval_pct_min}, {self.args.eval_pct_max}]")

        # Create distribution functions using partial
        num_cond_dist = partial(
            uniform_num_conditioning_distribution,
            conditioning_percentage_range=(self.args.cond_pct_min, self.args.cond_pct_max)
        )
        num_cond_blocks_dist = uniform_num_blocks_distribution

        num_eval_dist = partial(
            uniform_num_evaluation_distribution,
            evaluation_percentage_range=(self.args.eval_pct_min, self.args.eval_pct_max)
        )
        num_eval_blocks_dist = uniform_num_blocks_distribution

        self.augmenter = ConditionalAugmenter(
            mask_token_id=self.mask_token_id,
            bos_token_id=self.bos_token_id,
            max_seq_len=self.body_seq_len,  # Body tokens only (1023)
            cond_pct_max=self.args.cond_pct_max,
            tokenizer_pad_token_id=self.tokenizer.pad_token_id,
            num_conditioning_distribution=num_cond_dist,
            num_blocks_distribution=num_cond_blocks_dist,
            block_sizes_distribution=uniform_block_sizes_distribution,
            num_evaluation_distribution=num_eval_dist,
            num_eval_blocks_distribution=num_eval_blocks_dist,
            eval_block_sizes_distribution=uniform_block_sizes_distribution,
            min_conditioning=self.args.min_conditioning,
            min_evaluation=self.args.min_evaluation,
            conditioning_sampling=self.args.conditioning_sampling,
            evaluation_sampling=self.args.evaluation_sampling,
        )

        # Create collate function that does augmentation + dynamic padding
        # This is OPTIMAL: augment first, then single dynamic padding to max in batch
        logger.info("Creating augment collate function for dynamic padding optimization")
        train_collate_fn = create_augment_collate_fn(self.augmenter, device='cpu')

        # Note: Using augment collate function for optimal single-pass dynamic padding
        self.train_loader = get_dataloader(
            config=dataset_config,
            split="train",
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            streaming=False,
            num_samples=self.args.num_train_samples,
            collate_fn=train_collate_fn
        )

        if self.args.do_eval:
            # Validation dataloader with simple dynamic padding (no augmentation)
            # Evaluation modes apply their own augmentation strategies
            val_collate_fn = create_simple_collate_fn(pad_token_id=self.tokenizer.pad_token_id)
            self.val_loader = get_dataloader(
                config=dataset_config,
                split="validation",
                batch_size=self.args.eval_batch_size,
                num_workers=self.args.num_workers,
                streaming=False,
                num_samples=self.args.num_eval_samples,
                collate_fn=val_collate_fn  # Use simple collate (dynamic padding only)
            )
        else:
            self.val_loader = None

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay
        )

        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // self.args.gradient_accumulation_steps
        self.total_steps = steps_per_epoch * self.args.num_epochs

        # Learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.args.warmup_steps
        )

        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - self.args.warmup_steps,
            eta_min=self.args.learning_rate * 0.1
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.args.warmup_steps]
        )

    def train_step(self, batch):
        """Single training step with conditional augmentation

        Note: Augmentation is now done in collate_fn for optimal single-pass dynamic padding
        """
        # Move augmented batch to device (augmentation already done in collate_fn)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        position_ids = batch["position_ids"].to(self.device)

        # Forward pass with augmented inputs
        logits, loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids
        )

        # Scale loss for gradient accumulation
        loss = loss / self.args.gradient_accumulation_steps

        return loss

    def train(self):
        """Main training loop"""
        logger.info("=" * 80)
        logger.info("Starting Conditional Training")
        logger.info("=" * 80)
        logger.info(f"Total epochs: {self.args.num_epochs}")
        logger.info(f"Total steps: {self.total_steps}")
        logger.info(f"Conditioning percentage range: [{self.args.cond_pct_min}, {self.args.cond_pct_max}]")
        logger.info(f"Evaluation percentage range: [{self.args.eval_pct_min}, {self.args.eval_pct_max}]")
        logger.info("=" * 80)

        self.model.train()
        running_loss = 0

        for epoch in range(self.args.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")

            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                loss = self.train_step(batch)
                loss.backward()

                running_loss += loss.item() * self.args.gradient_accumulation_steps

                # Gradient accumulation
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.args.logging_steps == 0:
                        # Average over actual number of batches (logging_steps * gradient_accumulation_steps)
                        avg_loss = running_loss / (self.args.logging_steps * self.args.gradient_accumulation_steps)
                        perplexity = torch.exp(torch.tensor(avg_loss)).item()
                        lr = self.optimizer.param_groups[0]["lr"]

                        logger.info(
                            f"Step {self.global_step}/{self.total_steps} | "
                            f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | LR: {lr:.2e}"
                        )

                        # Log to CSV
                        metrics = {
                            'step': self.global_step,
                            'epoch': epoch + 1,
                            'train_loss': avg_loss,
                            'train_perplexity': perplexity,
                            'learning_rate': lr
                        }
                        self._log_to_csv(metrics)

                        running_loss = 0

                    # Evaluation
                    if self.args.do_eval and self.global_step % self.args.eval_steps == 0:
                        eval_results = evaluate_all_modes(
                            model=self.model,
                            dataloader=self.val_loader,
                            device=self.device,
                            augmenter=self.augmenter,
                            max_batches=self.args.max_eval_batches,
                            trainer_args=self.args
                        )

                        # Log all 5 modes
                        logger.info(f"=" * 80)
                        logger.info(f"Evaluation Results (Step {self.global_step})")
                        logger.info(f"=" * 80)
                        logger.info(f"Mode 1 (Autoregressive)   : loss={eval_results['mode1_loss']:.4f}, ppl={eval_results['mode1_ppl']:.2f}, tokens={eval_results.get('mode1_tokens', 'N/A')}")
                        logger.info(f"Mode 2 (Boundary Filling) : loss={eval_results['mode2_loss']:.4f}, ppl={eval_results['mode2_ppl']:.2f}, tokens={eval_results.get('mode2_tokens', 'N/A')}")
                        logger.info(f"Mode 3 (Training Dist)    : loss={eval_results['mode3_loss']:.4f}, ppl={eval_results['mode3_ppl']:.2f}, tokens={eval_results.get('mode3_tokens', 'N/A')}")
                        logger.info(f"Mode 4 (Auto on Boundary) : loss={eval_results['mode4_loss']:.4f}, ppl={eval_results['mode4_ppl']:.2f}, tokens={eval_results.get('mode4_tokens', 'N/A')}")
                        logger.info(f"Mode 5 (Auto on Training) : loss={eval_results['mode5_loss']:.4f}, ppl={eval_results['mode5_ppl']:.2f}, tokens={eval_results.get('mode5_tokens', 'N/A')}")
                        logger.info(f"-" * 80)
                        logger.info(f"Comparisons:")
                        logger.info(f"  Mode 2 vs 4 (Boundary):  Δ={eval_results['mode2_loss'] - eval_results['mode4_loss']:.4f} (negative = conditional better)")
                        logger.info(f"  Mode 3 vs 5 (Training):  Δ={eval_results['mode3_loss'] - eval_results['mode5_loss']:.4f} (negative = conditional better)")
                        logger.info(f"=" * 80)

                        # Create evaluation metrics entry
                        eval_metrics = {
                            'step': self.global_step,
                            'epoch': epoch + 1,
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
                            'learning_rate': self.optimizer.param_groups[0]["lr"]
                        }
                        self._log_to_csv(eval_metrics)

                        # Save best model (using Mode 3 loss as criterion)
                        if eval_results["mode3_loss"] < self.best_val_loss:
                            self.best_val_loss = eval_results["mode3_loss"]
                            logger.info(f"New best model saved! Val Loss (Mode 3): {self.best_val_loss:.4f}")
                            self._save_checkpoint("best_model")

                    # Save checkpoint
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(f"checkpoint_step_{self.global_step}")

        logger.info("=" * 80)
        logger.info("Training completed!")
        if self.args.do_eval:
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)

        # Save final model
        self._save_checkpoint("final_model")

    def _load_pretrained_weights(self, pretrained_path):
        """
        Load pretrained weights from HuggingFace or local checkpoint

        Args:
            pretrained_path: Either HuggingFace model name (e.g., 'gpt2', 'distilgpt2')
                           or path to local checkpoint file
        """
        logger.info(f"Loading pretrained weights from: {pretrained_path}")

        # Check if it's a HuggingFace model name or local path
        if pretrained_path in ['gpt2', 'distilgpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            # Load from HuggingFace
            self._load_from_huggingface(pretrained_path)
        else:
            # Load from local checkpoint
            self._load_from_checkpoint(pretrained_path)

    def _load_from_huggingface(self, model_name):
        """Load pretrained weights from HuggingFace transformers"""
        from transformers import GPT2LMHeadModel

        logger.info(f"Loading HuggingFace model: {model_name}")
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        hf_state_dict = hf_model.state_dict()

        # Map HuggingFace state dict to our model
        our_state_dict = {}

        # Mapping table: HuggingFace -> Our model
        key_mapping = {
            'transformer.wte.weight': 'wte.weight',
            'transformer.wpe.weight': 'wpe.weight',
            'transformer.ln_f.weight': 'ln_f.weight',
            'transformer.ln_f.bias': 'ln_f.bias',
            'lm_head.weight': 'lm_head.weight',
        }

        # Map transformer blocks
        for i in range(self.config.n_layer):
            # Attention
            key_mapping[f'transformer.h.{i}.ln_1.weight'] = f'blocks.{i}.ln_1.weight'
            key_mapping[f'transformer.h.{i}.ln_1.bias'] = f'blocks.{i}.ln_1.bias'
            key_mapping[f'transformer.h.{i}.attn.c_attn.weight'] = f'blocks.{i}.attn.c_attn.weight'
            key_mapping[f'transformer.h.{i}.attn.c_attn.bias'] = f'blocks.{i}.attn.c_attn.bias'
            key_mapping[f'transformer.h.{i}.attn.c_proj.weight'] = f'blocks.{i}.attn.c_proj.weight'
            key_mapping[f'transformer.h.{i}.attn.c_proj.bias'] = f'blocks.{i}.attn.c_proj.bias'

            # MLP
            key_mapping[f'transformer.h.{i}.ln_2.weight'] = f'blocks.{i}.ln_2.weight'
            key_mapping[f'transformer.h.{i}.ln_2.bias'] = f'blocks.{i}.ln_2.bias'
            key_mapping[f'transformer.h.{i}.mlp.c_fc.weight'] = f'blocks.{i}.mlp.c_fc.weight'
            key_mapping[f'transformer.h.{i}.mlp.c_fc.bias'] = f'blocks.{i}.mlp.c_fc.bias'
            key_mapping[f'transformer.h.{i}.mlp.c_proj.weight'] = f'blocks.{i}.mlp.c_proj.weight'
            key_mapping[f'transformer.h.{i}.mlp.c_proj.bias'] = f'blocks.{i}.mlp.c_proj.bias'

        # Apply mapping
        for hf_key, our_key in key_mapping.items():
            if hf_key in hf_state_dict:
                our_state_dict[our_key] = hf_state_dict[hf_key]

        # Load mapped weights (strict=False because vocab size will be different)
        missing_keys, unexpected_keys = self.model.load_state_dict(our_state_dict, strict=False)

        logger.info(f"✓ Loaded pretrained weights from {model_name}")
        logger.info(f"  Missing keys: {len(missing_keys)} (expected if vocab size differs)")
        logger.info(f"  Unexpected keys: {len(unexpected_keys)}")

        if len(missing_keys) > 0:
            logger.debug(f"  Missing: {missing_keys[:5]}...")  # Show first 5
        if len(unexpected_keys) > 0:
            logger.debug(f"  Unexpected: {unexpected_keys[:5]}...")

    def _load_from_checkpoint(self, checkpoint_path):
        """Load from local checkpoint file"""
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # Optionally restore training state
        if self.args.resume_training:
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "epoch" in checkpoint:
                self.epoch = checkpoint["epoch"]
            if "global_step" in checkpoint:
                self.global_step = checkpoint["global_step"]
            if "best_val_loss" in checkpoint:
                self.best_val_loss = checkpoint["best_val_loss"]

            logger.info(f"✓ Resumed training from epoch {self.epoch}, step {self.global_step}")
        else:
            logger.info(f"✓ Loaded model weights only (not resuming training state)")

    def _save_checkpoint(self, name):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
            "args": vars(self.args)
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _init_csv_logger(self):
        """Initialize CSV log file"""
        with open(self.csv_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
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
                "learning_rate"
            ])

    def _log_to_csv(self, metrics):
        """Log metrics to CSV"""
        with open(self.csv_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get('step', ''),
                metrics.get('epoch', ''),
                metrics.get('train_loss', ''),
                metrics.get('train_perplexity', ''),
                metrics.get('mode1_loss', ''),
                metrics.get('mode1_ppl', ''),
                metrics.get('mode2_loss', ''),
                metrics.get('mode2_ppl', ''),
                metrics.get('mode3_loss', ''),
                metrics.get('mode3_ppl', ''),
                metrics.get('mode4_loss', ''),
                metrics.get('mode4_ppl', ''),
                metrics.get('mode5_loss', ''),
                metrics.get('mode5_ppl', ''),
                metrics.get('learning_rate', '')
            ])


def parse_args():
    parser = argparse.ArgumentParser(description="Train Arbitrary Conditional Probability Model")

    # Model arguments
    parser.add_argument("--model_config", type=str, default="distilgpt2",
                        help="Model configuration")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path to pretrained model or HuggingFace model name (e.g., 'gpt2', 'distilgpt2')")
    parser.add_argument("--resume_training", action="store_true",
                        help="Resume training state (optimizer, scheduler, epoch) from checkpoint")

    # Data arguments
    parser.add_argument("--num_train_samples", type=int, default=10000,
                        help="Number of training samples")
    parser.add_argument("--num_eval_samples", type=int, default=1000,
                        help="Number of evaluation samples")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of warmup steps")

    # Conditional modeling arguments (distribution-based sampling)
    parser.add_argument("--cond_pct_min", type=float, default=0.2,
                        help="Minimum percentage for conditioning tokens")
    parser.add_argument("--cond_pct_max", type=float, default=0.4,
                        help="Maximum percentage for conditioning tokens")
    parser.add_argument("--eval_pct_min", type=float, default=0.2,
                        help="Minimum percentage for evaluation tokens")
    parser.add_argument("--eval_pct_max", type=float, default=0.4,
                        help="Maximum percentage for evaluation tokens")

    # Common parameters
    parser.add_argument("--min_conditioning", type=int, default=1,
                        help="Minimum number of conditioning tokens")
    parser.add_argument("--min_evaluation", type=int, default=1,
                        help="Minimum number of evaluation tokens")
    parser.add_argument("--conditioning_sampling", type=str, default="blockwise",
                        choices=["random", "blockwise"],
                        help="Sampling mode for conditioning set: 'random' or 'blockwise'")
    parser.add_argument("--evaluation_sampling", type=str, default="blockwise",
                        choices=["random", "blockwise"],
                        help="Sampling mode for evaluation set: 'random' or 'blockwise'")
    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--max_eval_batches", type=int, default=100,
                        help="Maximum evaluation batches")
    parser.add_argument("--do_eval", action="store_true",
                        help="Run evaluation during training")

    # Mode 2 (Boundary filling) evaluation parameters
    parser.add_argument("--mode2_boundary_cond_pct_min", type=float, default=0.1,
                        help="Minimum conditioning percentage for Mode 2 boundary evaluation")
    parser.add_argument("--mode2_boundary_cond_pct_max", type=float, default=0.3,
                        help="Maximum conditioning percentage for Mode 2 boundary evaluation")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default="conditional_gpt2")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()

    # Print configuration
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info("=" * 80)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("=" * 80)

    # Create trainer
    trainer = ConditionalTrainer(args)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
