"""
Base Trainer for Unified Training Pipeline

This abstract base class ensures all baseline models use identical training procedures,
differing only in model architecture and model-specific augmentation.

Key components shared across all trainers:
- Training loop structure (gradient accumulation, optimizer stepping)
- Optimizer configuration (AdamW with identical hyperparameters)
- Learning rate scheduler (linear warmup + cosine annealing)
- Logging infrastructure (CSV logging)
- Checkpointing format
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from abc import ABC, abstractmethod
import logging
import csv
from datetime import datetime
from tqdm import tqdm
import math
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base trainer for fair baseline comparisons

    Ensures all baselines use identical:
    - Training loop structure
    - Optimizer configuration
    - Scheduler configuration
    - Logging infrastructure
    - Checkpointing format
    - FP16/AMP support (optional, enabled via args.fp16)

    Subclasses must implement:
    - setup_model(): Model initialization (model-specific)
    - setup_data(): Data loader creation (model-specific)
    - train_step(batch): Single training step (model-specific)
    - evaluate(): Evaluation procedure (model-specific)

    Subclasses can optionally override:
    - _get_extra_csv_columns(): Add model-specific CSV columns
    - _get_extra_train_metrics(): Add model-specific training metrics
    - _get_extra_eval_metrics(): Add model-specific evaluation metrics
    """

    def __init__(self, args):
        """
        Initialize base trainer

        Args:
            args: Command-line arguments containing all hyperparameters
        """
        self.args = args

        # Setup device
        self.device = self._setup_device(args.device)
        logger.info(f"Using device: {self.device}")

        # Setup experiment directory
        self.exp_dir = self._setup_experiment_dir(args)
        logger.info(f"Experiment directory: {self.exp_dir}")

        # Setup model and data (call abstract methods - implemented by subclasses)
        logger.info("Setting up model...")
        self.setup_model()

        logger.info("Setting up data loaders...")
        self.setup_data()

        # Setup optimizer and scheduler (identical for all baselines)
        logger.info("Setting up optimizer and scheduler...")
        self._setup_optimizer()

        # Setup automatic mixed precision (optional, enabled via args.fp16)
        self._setup_amp()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_triggered = False

        # Setup CSV logger
        # CRITICAL: Use append mode when resuming to preserve existing metrics
        is_resuming = hasattr(args, 'resume_from') and args.resume_from
        self.csv_log_file = self.exp_dir / "logs" / "metrics.csv"
        self._init_csv_logger(append=is_resuming)

    @abstractmethod
    def setup_model(self):
        """
        Setup model (model-specific implementation)

        Must set:
        - self.model: The model instance
        - self.config: Model configuration
        - self.tokenizer: Tokenizer instance
        - Any other model-specific attributes
        """
        pass

    @abstractmethod
    def setup_data(self):
        """
        Setup data loaders (model-specific implementation)

        Must set:
        - self.train_loader: Training data loader
        - self.val_loader: Validation data loader (if args.do_eval is True)
        """
        pass

    @abstractmethod
    def train_step(self, batch):
        """
        Single training step (model-specific implementation)

        Args:
            batch: Batch of data from dataloader

        Returns:
            loss: Raw training loss (scaling handled in train() method)
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluation procedure (model-specific implementation)

        Returns:
            dict: Evaluation results containing at least {"loss": float, ...}
        """
        pass

    # =========================================================================
    # CSV Logging Methods (with default 5-mode format)
    # =========================================================================

    def get_csv_header(self):
        """
        Get CSV header for logging

        Default implementation provides 5-mode evaluation format.
        Override _get_extra_csv_columns() to add model-specific columns.

        Returns:
            list: Column names for CSV file
        """
        base_columns = [
            "step", "epoch", "train_loss", "train_perplexity",
            "mode1_loss", "mode1_ppl", "mode2_loss", "mode2_ppl",
            "mode3_loss", "mode3_ppl", "mode4_loss", "mode4_ppl",
            "mode5_loss", "mode5_ppl", "learning_rate"
        ]
        return base_columns + self._get_extra_csv_columns()

    def _get_extra_csv_columns(self):
        """
        Override to add model-specific CSV columns

        Example: return ["sigmagpt_mode", "ordering_mode"]
        """
        return []

    def format_train_metrics(self, avg_loss, perplexity, lr):
        """
        Format training metrics for CSV logging

        Default implementation provides 5-mode format with empty eval columns.
        Override _get_extra_train_metrics() to add model-specific metrics.

        Args:
            avg_loss: Average training loss
            perplexity: Training perplexity
            lr: Current learning rate

        Returns:
            dict: Metrics dictionary with keys matching CSV header
        """
        metrics = {
            'train_loss': avg_loss,
            'train_perplexity': perplexity,
            'mode1_loss': '', 'mode1_ppl': '',
            'mode2_loss': '', 'mode2_ppl': '',
            'mode3_loss': '', 'mode3_ppl': '',
            'mode4_loss': '', 'mode4_ppl': '',
            'mode5_loss': '', 'mode5_ppl': '',
            'learning_rate': lr
        }
        metrics.update(self._get_extra_train_metrics())
        return metrics

    def _get_extra_train_metrics(self):
        """
        Override to add model-specific training metrics

        Example: return {"sigmagpt_mode": self.sigmagpt_mode}
        """
        return {}

    def format_eval_metrics(self, eval_results):
        """
        Format evaluation metrics for CSV logging

        Default implementation provides 5-mode format.
        Override _get_extra_eval_metrics() to add model-specific metrics.

        Args:
            eval_results: Results from evaluate() method

        Returns:
            dict: Metrics dictionary with keys matching CSV header
        """
        metrics = {
            'train_loss': '', 'train_perplexity': '',
            'mode1_loss': eval_results.get('mode1_loss', ''),
            'mode1_ppl': eval_results.get('mode1_ppl', ''),
            'mode2_loss': eval_results.get('mode2_loss', ''),
            'mode2_ppl': eval_results.get('mode2_ppl', ''),
            'mode3_loss': eval_results.get('mode3_loss', ''),
            'mode3_ppl': eval_results.get('mode3_ppl', ''),
            'mode4_loss': eval_results.get('mode4_loss', ''),
            'mode4_ppl': eval_results.get('mode4_ppl', ''),
            'mode5_loss': eval_results.get('mode5_loss', ''),
            'mode5_ppl': eval_results.get('mode5_ppl', ''),
            'learning_rate': self.optimizer.param_groups[0]["lr"]
        }
        metrics.update(self._get_extra_eval_metrics(eval_results))
        return metrics

    def _get_extra_eval_metrics(self, eval_results):
        """
        Override to add model-specific evaluation metrics

        Example: return {"sigmagpt_mode": self.sigmagpt_mode}
        """
        return {}

    # =========================================================================
    # FP16/AMP Support
    # =========================================================================

    def _setup_amp(self):
        """
        Setup automatic mixed precision training (optional)

        Enabled via args.fp16 flag. Only works on CUDA devices.
        """
        self.use_amp = getattr(self.args, 'fp16', False) and self.device.type == 'cuda'
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            logger.info("Using mixed precision training (FP16)")
        else:
            self.scaler = None

    # =========================================================================
    # Training loop hooks (with FP16 support)
    # =========================================================================

    def _backward(self, scaled_loss):
        """
        Backward pass with optional FP16 gradient scaling

        Args:
            scaled_loss: Loss tensor (already scaled by gradient accumulation)
        """
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def _clip_gradients(self):
        """
        Gradient clipping with FP16 unscaling if needed

        Called after backward, before optimizer step.
        """
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.args.max_grad_norm
        )

    def _optimizer_step(self):
        """
        Optimizer step with FP16 scaler support

        Called after gradient clipping.
        """
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def _setup_device(self, device_arg):
        """
        Setup computing device

        Args:
            device_arg: Device argument from command line ("auto", "cuda", "mps", "cpu")

        Returns:
            torch.device: Selected device
        """
        if device_arg == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using Apple Silicon GPU (MPS)")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(device_arg)

        return device

    def _setup_tokenizer(self, pretrained_name='gpt2', tokenizer_class=None):
        """
        Helper to setup tokenizer with common configuration

        Args:
            pretrained_name: HuggingFace model name (default: 'gpt2')
            tokenizer_class: Optional tokenizer class (default: GPT2Tokenizer)

        Returns:
            Configured tokenizer with pad_token set
        """
        if tokenizer_class is None:
            from transformers import GPT2Tokenizer
            tokenizer_class = GPT2Tokenizer

        tokenizer = tokenizer_class.from_pretrained(pretrained_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Tokenizer: {pretrained_name}, vocab_size: {len(tokenizer)}")
        return tokenizer

    def _setup_experiment_dir(self, args):
        """
        Create experiment directory with timestamp, or use existing dir when resuming

        Args:
            args: Command-line arguments

        Returns:
            Path: Experiment directory path
        """
        # If resuming from checkpoint, use the checkpoint's experiment directory
        if hasattr(args, 'resume_from') and args.resume_from:
            checkpoint_path = Path(args.resume_from)
            # Checkpoint path: experiments/exp_name/checkpoints/checkpoint_step_X.pt
            # We want: experiments/exp_name
            if 'checkpoints' in checkpoint_path.parts:
                exp_dir = checkpoint_path.parent.parent
                logger.info(f"Resuming in existing experiment directory: {exp_dir}")
            else:
                # Fallback: checkpoint is directly in exp dir
                exp_dir = checkpoint_path.parent
        else:
            # New training: create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if hasattr(args, 'exp_name') and args.exp_name:
                # Check if exp_name already ends with a timestamp (YYYYMMDD_HHMMSS pattern)
                # This prevents double timestamps when submission scripts include their own
                has_timestamp = re.search(r'_\d{8}_\d{6}$', args.exp_name) is not None
                if has_timestamp:
                    exp_name = args.exp_name  # Use as-is, already has timestamp
                else:
                    exp_name = f"{args.exp_name}_{timestamp}"
            else:
                exp_name = f"exp_{timestamp}"
            exp_dir = Path(args.output_dir) / exp_name

        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)

        self.checkpoint_dir = exp_dir / "checkpoints"
        self.log_dir = exp_dir / "logs"

        return exp_dir

    def _setup_optimizer(self):
        """
        Setup optimizer and scheduler (IDENTICAL for all baselines)

        This ensures fair comparison by using identical optimization configuration
        across all models.
        """
        # AdamW optimizer with configurable hyperparameters
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay
        )

        # Calculate total training steps
        steps_per_epoch = math.ceil(len(self.train_loader) / self.args.gradient_accumulation_steps)
        self.total_steps = steps_per_epoch * self.args.num_epochs

        logger.info(f"Total training steps: {self.total_steps}")
        logger.info(f"Warmup steps: {self.args.warmup_steps}")

        # Learning rate scheduler: Linear warmup + Cosine annealing
        warmup_start_factor = getattr(self.args, 'warmup_start_factor', 0.1)
        min_lr_ratio = getattr(self.args, 'min_lr_ratio', 0.1)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=self.args.warmup_steps
        )

        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - self.args.warmup_steps,
            eta_min=self.args.learning_rate * min_lr_ratio
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.args.warmup_steps]
        )

    def _init_csv_logger(self, append=False):
        """
        Initialize CSV logger with header

        Args:
            append: If True and CSV exists, don't overwrite (for resume mode)
        """
        if append and self.csv_log_file.exists():
            # Append mode: CSV already exists with header, don't overwrite
            logger.info(f"CSV logger in append mode: {self.csv_log_file}")
            return

        with open(self.csv_log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.get_csv_header())
            writer.writeheader()

    def _log_to_csv(self, metrics):
        """
        Log metrics to CSV file

        Args:
            metrics: Dictionary of metrics to log
        """
        with open(self.csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.get_csv_header())
            writer.writerow(metrics)

    def _truncate_csv_to_step(self, target_step):
        """
        Truncate CSV to only include rows with step <= target_step

        Used when resuming from checkpoint to remove "dirty" rows
        written after the checkpoint was saved (e.g., after preemption).

        Args:
            target_step: Maximum step to keep in CSV
        """
        if not self.csv_log_file.exists():
            logger.info("No CSV file to truncate")
            return

        # Read existing CSV
        rows_to_keep = []
        removed_count = 0

        with open(self.csv_log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            for row in reader:
                step = int(row.get('step', 0))
                if step <= target_step:
                    rows_to_keep.append(row)
                else:
                    removed_count += 1

        # Rewrite CSV with only valid rows
        with open(self.csv_log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_keep)

        logger.info(f"CSV truncated: kept {len(rows_to_keep)} rows (step <= {target_step}), removed {removed_count} rows")

    def train(self):
        """
        Main training loop (IDENTICAL for all baselines)

        This ensures all models are trained with identical procedures.
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

        # Start from resumed epoch (0 if starting fresh)
        start_epoch = self.epoch
        if start_epoch > 0:
            logger.info(f"Resuming from epoch {start_epoch + 1}")

        for epoch in range(start_epoch, self.args.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")

            # Calculate batches to skip for deterministic resume
            # Only skip in the first epoch after resume
            skip_batches = 0
            if hasattr(self, 'resume_batch_idx') and self.resume_batch_idx > 0:
                skip_batches = self.resume_batch_idx + 1  # +1 to start from the batch after checkpoint
                self.resume_batch_idx = 0  # Only skip once
                logger.info(f"Resuming: skipping first {skip_batches} batches (already processed)")

            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch + 1}")

            for batch_idx, batch in progress_bar:
                # Skip already processed batches (for deterministic resume)
                if batch_idx < skip_batches:
                    continue

                # Track current batch index for checkpoint save
                self.current_batch_idx = batch_idx

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
                self._backward(scaled_loss)

                # Accumulate unscaled loss for logging
                running_loss += loss.item()
                running_batch_count += 1

                if is_accum_step or is_last_batch:
                    # Gradient clipping (hook for FP16 unscaling)
                    self._clip_gradients()

                    # Optimizer step (hook for FP16 scaler)
                    self._optimizer_step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Update progress bar
                    current_lr = self.optimizer.param_groups[0]['lr']
                    avg_loss = running_loss / running_batch_count if running_batch_count > 0 else 0
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })

                    # Logging
                    if self.global_step % self.args.logging_steps == 0:
                        self._log_training_metrics(running_loss, running_batch_count)
                        running_loss = 0
                        running_batch_count = 0

                    # Evaluation
                    if self.args.do_eval and self.global_step % self.args.eval_steps == 0:
                        # Log current training loss before evaluation (if we haven't logged at this step yet)
                        if self.global_step % self.args.logging_steps != 0 and running_loss > 0:
                            self._log_training_metrics(running_loss)
                        
                        logger.info(f"\nEvaluating at step {self.global_step}...")
                        eval_results = self.evaluate()
                        self._log_evaluation_metrics(eval_results)

                        # Save best model and check early stopping
                        if eval_results["loss"] < self.best_val_loss:
                            self.best_val_loss = eval_results["loss"]
                            self.patience_counter = 0
                            logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                            self._save_checkpoint("best_model")
                        else:
                            self.patience_counter += 1
                            if self.args.early_stopping_patience > 0:
                                logger.info(f"No improvement. Patience: {self.patience_counter}/{self.args.early_stopping_patience}")
                                if self.patience_counter >= self.args.early_stopping_patience:
                                    logger.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
                                    self.early_stop_triggered = True
                                    self._save_checkpoint("early_stopped_model")
                                    return

                        self.model.train()

                    # Save checkpoint
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(f"checkpoint_step_{self.global_step}")

        # Final checkpoint
        logger.info("\nTraining completed!")
        self._save_checkpoint("final_model")
        logger.info(f"Final model saved to {self.checkpoint_dir / 'final_model.pt'}")

    def _log_training_metrics(self, running_loss, running_batch_count):
        """
        Log training metrics

        Args:
            running_loss: Accumulated loss since last logging
            running_batch_count: Number of batches processed since last logging
        """
        # Calculate average loss and perplexity
        avg_loss = running_loss / running_batch_count if running_batch_count > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        lr = self.optimizer.param_groups[0]["lr"]

        # Log to console
        logger.info(
            f"Step {self.global_step}/{self.total_steps} | "
            f"Epoch {self.epoch + 1} | "
            f"Loss: {avg_loss:.4f} | "
            f"PPL: {perplexity:.2f} | "
            f"LR: {lr:.2e}"
        )

        # Format metrics using model-specific method
        metrics = self.format_train_metrics(avg_loss, perplexity, lr)

        # Add common fields
        metrics['step'] = self.global_step
        metrics['epoch'] = self.epoch + 1

        # Log to CSV
        self._log_to_csv(metrics)

    def _log_evaluation_metrics(self, eval_results):
        """
        Log evaluation metrics (calls model-specific formatting)

        Args:
            eval_results: Results from evaluate() method
        """
        # Format metrics using model-specific method
        metrics = self.format_eval_metrics(eval_results)

        # Add common fields
        metrics['step'] = self.global_step
        metrics['epoch'] = self.epoch + 1

        # Log to CSV
        self._log_to_csv(metrics)

    def _save_checkpoint(self, name):
        """
        Save checkpoint (IDENTICAL structure for all baselines)

        Args:
            name: Checkpoint name (e.g., "best_model", "checkpoint_step_1000")
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        # Get generator state for deterministic resume
        generator_state = None
        if hasattr(self.train_loader, 'shuffle_generator') and self.train_loader.shuffle_generator is not None:
            generator_state = self.train_loader.shuffle_generator.get_state()

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "batch_idx": getattr(self, 'current_batch_idx', 0),  # For deterministic resume
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__ if hasattr(self, 'config') else {},
            "args": vars(self.args),
            "generator_state": generator_state  # For deterministic shuffle resume
        }

        # Save atomically: write to temp file first, then rename
        # This prevents corrupted checkpoints if process is killed during save
        temp_path = str(checkpoint_path) + '.tmp'
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path, resume_csv=True):
        """
        Load checkpoint and optionally handle CSV for resume

        Args:
            checkpoint_path: Path to checkpoint file
            resume_csv: If True, truncate CSV to checkpoint's global_step
                       (removes rows written after checkpoint was saved)
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        # Restore batch_idx for deterministic resume (skip already processed batches)
        self.resume_batch_idx = checkpoint.get("batch_idx", 0)

        # Restore generator state for deterministic shuffle
        generator_state = checkpoint.get("generator_state")
        if generator_state is not None and hasattr(self.train_loader, 'shuffle_generator') and self.train_loader.shuffle_generator is not None:
            # Ensure state is ByteTensor (may be converted during save/load)
            if not isinstance(generator_state, torch.ByteTensor):
                generator_state = generator_state.to(torch.uint8)
            self.train_loader.shuffle_generator.set_state(generator_state)
            logger.info("Restored DataLoader shuffle generator state")

        # Truncate CSV to remove rows written after this checkpoint (preemption recovery)
        if resume_csv:
            self._truncate_csv_to_step(self.global_step)

        logger.info(f"Checkpoint loaded: step={self.global_step}, epoch={self.epoch}, batch_idx={self.resume_batch_idx}, best_val_loss={self.best_val_loss:.4f}")
