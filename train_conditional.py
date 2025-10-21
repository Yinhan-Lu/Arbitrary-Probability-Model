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
from train.dataset import get_dataloader
from train.augmentation import ConditionalAugmenter

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

        # Resize embeddings to include new tokens
        self.model = self.token_manager.resize_model_embeddings(self.model)

        total_params = self.model.get_num_params()
        logger.info(f"Model parameters: {total_params/1e6:.2f}M")

    def _setup_data(self):
        """Setup data loaders"""
        logger.info("Loading dataset...")

        # Note: Using standard dataloader, will augment in training loop
        self.train_loader = get_dataloader(
            config=self.config,
            split="train",
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            streaming=False,
            num_samples=self.args.num_train_samples
        )

        if self.args.do_eval:
            self.val_loader = get_dataloader(
                config=self.config,
                split="validation",
                batch_size=self.args.eval_batch_size,
                num_workers=self.args.num_workers,
                streaming=False,
                num_samples=self.args.num_eval_samples
            )
        else:
            self.val_loader = None

        # Create conditional augmenter
        self.augmenter = ConditionalAugmenter(
            mask_token_id=self.mask_token_id,
            bos_token_id=self.bos_token_id,
            conditioning_ratio=self.args.conditioning_ratio,
            evaluation_ratio=self.args.evaluation_ratio,
            min_conditioning=self.args.min_conditioning,
            min_evaluation=self.args.min_evaluation
        )

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
        """Single training step with conditional augmentation"""
        # Get original input
        input_ids = batch["input_ids"].to(self.device)

        # Apply conditional augmentation
        aug_batch = self.augmenter.augment_batch(input_ids, device=self.device)

        # Forward pass with custom attention mask
        logits, loss = self.model(
            input_ids=aug_batch["input_ids"],
            attention_mask=aug_batch["attention_mask"],
            labels=aug_batch["labels"]
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
        logger.info(f"Conditioning ratio: {self.args.conditioning_ratio}")
        logger.info(f"Evaluation ratio: {self.args.evaluation_ratio}")
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
                        avg_loss = running_loss / self.args.logging_steps
                        lr = self.optimizer.param_groups[0]["lr"]

                        logger.info(
                            f"Step {self.global_step}/{self.total_steps} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                        )

                        running_loss = 0

                    # Save checkpoint
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(f"checkpoint_step_{self.global_step}")

        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)

        # Save final model
        self._save_checkpoint("final_model")

    def _save_checkpoint(self, name):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
            "args": vars(self.args)
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Arbitrary Conditional Probability Model")

    # Model arguments
    parser.add_argument("--model_config", type=str, default="distilgpt2",
                        help="Model configuration")

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

    # Conditional modeling arguments
    parser.add_argument("--conditioning_ratio", type=float, default=0.3,
                        help="Fraction of tokens to use as conditioning")
    parser.add_argument("--evaluation_ratio", type=float, default=0.3,
                        help="Fraction of tokens to use as evaluation")
    parser.add_argument("--min_conditioning", type=int, default=1,
                        help="Minimum number of conditioning tokens")
    parser.add_argument("--min_evaluation", type=int, default=1,
                        help="Minimum number of evaluation tokens")

    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--do_eval", action="store_true",
                        help="Run evaluation during training")

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
