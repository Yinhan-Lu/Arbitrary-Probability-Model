"""
Training loop for GPT-2 model
Handles training, validation, logging, and checkpointing
"""

import os
import time
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import logging
from pathlib import Path
from typing import Optional, Dict
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Training loop manager for GPT-2 model
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        config=None,
        learning_rate=5e-4,
        weight_decay=0.01,
        max_epochs=10,
        device="cuda",
        log_dir="logs",
        checkpoint_dir="checkpoints",
        log_interval=100,
        eval_interval=1000,
        save_interval=1000,
        warmup_steps=100,
        max_grad_norm=1.0
    ):
        """
        Args:
            model: GPT-2 model instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Model configuration
            learning_rate: Peak learning rate
            weight_decay: Weight decay for AdamW
            max_epochs: Maximum number of training epochs
            device: Device to train on ('cuda' or 'cpu')
            log_dir: Directory for logs and plots
            checkpoint_dir: Directory for model checkpoints
            log_interval: Steps between logging
            eval_interval: Steps between validation
            save_interval: Steps between checkpoints
            warmup_steps: Number of warmup steps for LR scheduler
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_grad_norm = max_grad_norm

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        # Learning rate scheduler with warmup
        self.warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        total_steps = len(train_loader) * max_epochs
        self.main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=learning_rate * 0.1
        )

        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Logging setup
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.log_file = self.log_dir / "training_log.csv"
        self._init_log_file()

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.steps = []

        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Total parameters: {model.get_num_params()/1e6:.2f}M")
        logger.info(f"Training for {max_epochs} epochs")

    def _init_log_file(self):
        """Initialize CSV log file"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "epoch", "train_loss", "val_loss",
                "learning_rate", "time_elapsed"
            ])

    def _log_metrics(self, step, epoch, train_loss, val_loss, lr, elapsed_time):
        """Log metrics to CSV file"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, epoch, train_loss, val_loss, lr, elapsed_time])

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0

        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Forward pass
            logits, loss = self.model(input_ids, labels=input_ids)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer step
            self.optimizer.step()

            # Learning rate scheduling
            if self.current_step < self.warmup_steps:
                self.warmup_scheduler.step()
            else:
                self.main_scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update tracking
            epoch_loss += loss.item()
            num_batches += 1
            self.current_step += 1

            # Logging
            if self.current_step % self.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                elapsed = time.time() - epoch_start_time

                logger.info(
                    f"Epoch {epoch} | Step {self.current_step} | "
                    f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
                )

                self.train_losses.append(avg_loss)
                self.learning_rates.append(current_lr)
                self.steps.append(self.current_step)

            # Validation
            if self.val_loader and self.current_step % self.eval_interval == 0:
                val_loss = self.validate()
                self.val_losses.append(val_loss)

                self._log_metrics(
                    self.current_step, epoch, avg_loss, val_loss,
                    current_lr, time.time() - epoch_start_time
                )

                self.model.train()  # Back to training mode

            # Checkpointing
            if self.current_step % self.save_interval == 0:
                self.save_checkpoint(epoch, self.current_step)

        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch} completed | Avg Loss: {avg_epoch_loss:.4f}")

        return avg_epoch_loss

    @torch.no_grad()
    def validate(self):
        """Run validation"""
        self.model.eval()
        val_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            logits, loss = self.model(input_ids, labels=input_ids)

            val_loss += loss.item()
            num_batches += 1

            # Limit validation batches for speed
            if num_batches >= 50:
                break

        avg_val_loss = val_loss / num_batches
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        return avg_val_loss

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.max_epochs}")
            logger.info(f"{'='*60}")

            epoch_loss = self.train_epoch(epoch)

            # End of epoch validation
            if self.val_loader:
                val_loss = self.validate()
            else:
                val_loss = None

            # Save checkpoint at end of epoch
            self.save_checkpoint(epoch, self.current_step)

            # Plot progress
            self.plot_training_curves()

        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time/3600:.2f} hours")

    def save_checkpoint(self, epoch, step):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__ if self.config else None
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_step = checkpoint["step"]

        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        return checkpoint["epoch"]

    def plot_training_curves(self):
        """Plot and save training curves"""
        if not self.train_losses:
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Loss curve
        axes[0].plot(self.steps, self.train_losses, label="Train Loss", alpha=0.7)
        if self.val_losses:
            val_steps = self.steps[::len(self.steps)//len(self.val_losses)][:len(self.val_losses)]
            axes[0].plot(val_steps, self.val_losses, label="Val Loss", alpha=0.7)
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss over Time")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Learning rate curve
        axes[1].plot(self.steps, self.learning_rates, label="Learning Rate", color='orange')
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.log_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        logger.info(f"Training curves saved: {plot_path}")


if __name__ == "__main__":
    # Test the trainer with a tiny model
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from model.config import get_config
    from model.arbitrary_prob_gpt2 import GPT2Model
    from train.dataset import get_dataloader

    print("Testing Trainer...")

    # Use nano config for quick test
    config = get_config("nano")

    # Create model
    model = GPT2Model(config)

    # Create small dataloaders
    train_loader = get_dataloader(config, split="train", batch_size=2, num_samples=20)
    val_loader = get_dataloader(config, split="validation", batch_size=2, num_samples=10)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        max_epochs=2,
        device="cpu",
        log_interval=5,
        eval_interval=10,
        save_interval=20
    )

    # Train for 1 epoch
    print("\nRunning 1 epoch of training...")
    trainer.train_epoch(epoch=1)

    print("\nTrainer test completed successfully!")
