"""
Complete DistilGPT-2 Training Script on Wikipedia Dataset

This script trains a DistilGPT-2 model from scratch using the same configuration
as HuggingFace's distilgpt2 model, with comprehensive logging and tracking.

Features:
- Official DistilGPT-2 configuration (6 layers, 82M parameters)
- Wikipedia dataset from HuggingFace
- Comprehensive logging to CSV, TensorBoard, and WandB (optional)
- Automatic checkpointing with best model tracking
- Gradient accumulation for large effective batch sizes
- Mixed precision training (FP16)
- Detailed metrics: loss, perplexity, learning rate, gradient norms, etc.
- Resume from checkpoint support
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# Setup logging first (before optional imports)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model.config import get_config
from model.arbitrary_prob_gpt2 import GPT2Model
from train.dataset import get_dataloader


class ComprehensiveTrainer:
    """
    Comprehensive trainer for DistilGPT-2 with extensive logging and tracking
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.start_time = time.time()

        # Create experiment directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{args.exp_name}_{self.timestamp}"
        self.exp_dir = Path(args.output_dir) / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Setup subdirectories
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        logger.info(f"Experiment directory: {self.exp_dir}")
        logger.info(f"Device: {self.device}")

        # Save configuration
        self._save_config()

        # Initialize model
        logger.info("Initializing DistilGPT-2 model...")
        self.config = get_config("distilgpt2")
        self.model = GPT2Model(self.config).to(self.device)

        total_params = self.model.get_num_params()
        logger.info(f"Model parameters: {total_params/1e6:.2f}M")

        # Setup data loaders
        logger.info("Loading Wikipedia dataset...")
        self.train_loader = get_dataloader(
            config=self.config,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            streaming=args.streaming,
            num_samples=args.num_train_samples
        )

        if args.do_eval:
            self.val_loader = get_dataloader(
                config=self.config,
                split="validation",
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                streaming=False,
                num_samples=args.num_eval_samples
            )
        else:
            self.val_loader = None

        # Calculate total training steps
        if args.max_steps > 0:
            self.total_steps = args.max_steps
        else:
            steps_per_epoch = len(self.train_loader) // args.gradient_accumulation_steps
            self.total_steps = steps_per_epoch * args.num_epochs

        logger.info(f"Total training steps: {self.total_steps}")

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay
        )

        # Setup learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=args.warmup_steps
        )

        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - args.warmup_steps,
            eta_min=args.learning_rate * args.min_lr_ratio
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[args.warmup_steps]
        )

        # Setup mixed precision training
        self.use_amp = args.fp16 and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Using mixed precision training (FP16)")

        # Setup logging
        if TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=self.log_dir / "tensorboard")
        else:
            self.tb_writer = None
            logger.warning("TensorBoard logging disabled (not installed)")

        # WandB integration (optional)
        self.use_wandb = args.use_wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=self.exp_name,
                    config=vars(args)
                )
                logger.info("WandB logging enabled")
            except ImportError:
                logger.warning("WandB not installed, skipping WandB logging")
                self.use_wandb = False

        # Initialize CSV logger
        self.csv_log_file = self.log_dir / "metrics.csv"
        self._init_csv_logger()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # Load from checkpoint if specified
        if args.resume_from_checkpoint:
            self._load_checkpoint(args.resume_from_checkpoint)

    def _save_config(self):
        """Save training configuration to JSON"""
        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        logger.info(f"Configuration saved to {config_path}")

    def _init_csv_logger(self):
        """Initialize CSV log file"""
        with open(self.csv_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "epoch",
                "train_loss",
                "train_perplexity",
                "val_loss",
                "val_perplexity",
                "learning_rate",
                "grad_norm",
                "time_elapsed_seconds",
                "tokens_per_second"
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
                metrics.get('val_loss', ''),
                metrics.get('val_perplexity', ''),
                metrics.get('learning_rate', ''),
                metrics.get('grad_norm', ''),
                metrics.get('time_elapsed', ''),
                metrics.get('tokens_per_sec', '')
            ])

    def train_step(self, batch):
        """Single training step"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Prepare labels: set padding tokens to -100 so they're ignored in loss
        labels = input_ids.clone()
        labels[labels == 50256] = -100  # 50256 is GPT-2's pad_token_id
        
        if self.use_amp:
            with autocast('cuda'):
                logits, loss = self.model(input_ids, labels=labels)
                loss = loss / self.args.gradient_accumulation_steps
        else:
            logits, loss = self.model(input_ids, labels=labels)
            loss = loss / self.args.gradient_accumulation_steps

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.args.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation on validation set"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        num_batches = 0

        logger.info("Running evaluation...")

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Prepare labels: set padding tokens to -100 so they're ignored in loss
            labels = input_ids.clone()
            labels[labels == 50256] = -100  # 50256 is GPT-2's pad_token_id
            
            logits, loss = self.model(input_ids, labels=labels)

            total_loss += loss.item()
            total_tokens += input_ids.numel()
            num_batches += 1
            
            # Limit evaluation batches if specified
            if self.args.max_eval_batches > 0 and num_batches >= self.args.max_eval_batches:
                break

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.model.train()

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "num_batches": num_batches
        }

    def train(self):
        """Main training loop"""
        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)
        logger.info(f"Total epochs: {self.args.num_epochs}")
        logger.info(f"Batch size per device: {self.args.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps}")
        logger.info(f"Total training steps: {self.total_steps}")
        logger.info(f"Warmup steps: {self.args.warmup_steps}")
        logger.info("=" * 80)

        self.model.train()
        running_loss = 0
        tokens_processed = 0
        step_start_time = time.time()

        for epoch in range(self.epoch, self.args.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")

            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                loss = self.train_step(batch)
                running_loss += loss
                tokens_processed += batch["input_ids"].numel()

                # Gradient accumulation
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )

                    # Optimizer step
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
                        # Fix: Account for gradient accumulation when computing average loss
                        # We accumulate loss from (logging_steps * gradient_accumulation_steps) batches
                        avg_loss = running_loss / (self.args.logging_steps * self.args.gradient_accumulation_steps)
                        perplexity = torch.exp(torch.tensor(avg_loss)).item()
                        lr = self.optimizer.param_groups[0]["lr"]
                        elapsed = time.time() - step_start_time
                        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

                        metrics = {
                            "step": self.global_step,
                            "epoch": epoch + 1,
                            "train_loss": avg_loss,
                            "train_perplexity": perplexity,
                            "learning_rate": lr,
                            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "time_elapsed": time.time() - self.start_time,
                            "tokens_per_sec": tokens_per_sec
                        }

                        logger.info(
                            f"Step {self.global_step}/{self.total_steps} | "
                            f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                            f"LR: {lr:.2e} | Grad Norm: {grad_norm:.4f} | "
                            f"Tokens/s: {tokens_per_sec:.0f}"
                        )

                        # TensorBoard logging
                        if self.tb_writer:
                            self.tb_writer.add_scalar("train/loss", avg_loss, self.global_step)
                            self.tb_writer.add_scalar("train/perplexity", perplexity, self.global_step)
                            self.tb_writer.add_scalar("train/learning_rate", lr, self.global_step)
                            self.tb_writer.add_scalar("train/grad_norm", grad_norm, self.global_step)

                        # WandB logging
                        if self.use_wandb:
                            import wandb
                            wandb.log(metrics, step=self.global_step)

                        running_loss = 0
                        tokens_processed = 0
                        step_start_time = time.time()

                    # Evaluation
                    if self.args.do_eval and self.global_step % self.args.eval_steps == 0:
                        eval_results = self.evaluate()

                        logger.info(
                            f"Evaluation at step {self.global_step} | "
                            f"Val Loss: {eval_results['loss']:.4f} | "
                            f"Val PPL: {eval_results['perplexity']:.2f}"
                        )

                        metrics["val_loss"] = eval_results["loss"]
                        metrics["val_perplexity"] = eval_results["perplexity"]

                        # TensorBoard logging
                        if self.tb_writer:
                            self.tb_writer.add_scalar("eval/loss", eval_results["loss"], self.global_step)
                            self.tb_writer.add_scalar("eval/perplexity", eval_results["perplexity"], self.global_step)

                        # WandB logging
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "eval/loss": eval_results["loss"],
                                "eval/perplexity": eval_results["perplexity"]
                            }, step=self.global_step)

                        # Save best model
                        if eval_results["loss"] < self.best_val_loss:
                            self.best_val_loss = eval_results["loss"]
                            self._save_checkpoint("best_model")
                            logger.info(f"New best model saved! Val Loss: {self.best_val_loss:.4f}")

                    # Log to CSV
                    if self.global_step % self.args.logging_steps == 0:
                        self._log_to_csv(metrics)

                    # Save checkpoint
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(f"checkpoint_step_{self.global_step}")

                    # Check if max steps reached
                    if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                        logger.info(f"Reached max steps ({self.args.max_steps})")
                        self._save_checkpoint("final_model")
                        return

            # End of epoch - save checkpoint
            self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}")

        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Total time: {(time.time() - self.start_time) / 3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)

        # Save final model
        self._save_checkpoint("final_model")

        # Close loggers
        if self.tb_writer:
            self.tb_writer.close()
        if self.use_wandb:
            import wandb
            wandb.finish()

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

        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DistilGPT-2 on Wikipedia")

    # Model arguments
    parser.add_argument("--model_config", type=str, default="distilgpt2",
                        help="Model configuration to use")

    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="wikipedia",
                        help="Dataset name from HuggingFace")
    parser.add_argument("--dataset_config", type=str, default="20220301.en",
                        help="Dataset configuration")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode for dataset")
    parser.add_argument("--num_train_samples", type=int, default=None,
                        help="Number of training samples (None for all)")
    parser.add_argument("--num_eval_samples", type=int, default=10000,
                        help="Number of evaluation samples")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum training steps (-1 for all epochs)")
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
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of warmup steps")
    parser.add_argument("--warmup_start_factor", type=float, default=0.1,
                        help="Starting warmup factor")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Minimum LR as ratio of peak LR")

    # Mixed precision
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")

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

    # WandB arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="distilgpt2-wikipedia",
                        help="WandB project name")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--exp_name", type=str, default="distilgpt2_train",
                        help="Experiment name")

    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

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
    trainer = ComprehensiveTrainer(args)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
