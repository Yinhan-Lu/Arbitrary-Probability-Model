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

    Subclasses must implement:
    - setup_model(): Model initialization (model-specific)
    - setup_data(): Data loader creation (model-specific)
    - train_step(batch): Single training step (model-specific)
    - evaluate(): Evaluation procedure (model-specific)
    - get_csv_header(): CSV log column headers (model-specific)
    - format_train_metrics(avg_loss, perplexity, lr): Format training metrics for CSV
    - format_eval_metrics(eval_results): Format evaluation metrics for CSV
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

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Setup CSV logger
        self.csv_log_file = self.exp_dir / "logs" / "metrics.csv"
        self._init_csv_logger()

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
            loss: Training loss (already divided by gradient_accumulation_steps)
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

    @abstractmethod
    def get_csv_header(self):
        """
        Get CSV header for logging (model-specific)

        Returns:
            list: Column names for CSV file
        """
        pass

    @abstractmethod
    def format_train_metrics(self, avg_loss, perplexity, lr):
        """
        Format training metrics for CSV logging (model-specific)

        Args:
            avg_loss: Average training loss
            perplexity: Training perplexity
            lr: Current learning rate

        Returns:
            dict: Metrics dictionary with keys matching CSV header
        """
        pass

    @abstractmethod
    def format_eval_metrics(self, eval_results):
        """
        Format evaluation metrics for CSV logging (model-specific)

        Args:
            eval_results: Results from evaluate() method

        Returns:
            dict: Metrics dictionary with keys matching CSV header
        """
        pass

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

    def _setup_experiment_dir(self, args):
        """
        Create experiment directory with timestamp

        Args:
            args: Command-line arguments

        Returns:
            Path: Experiment directory path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{args.exp_name}_{timestamp}" if hasattr(args, 'exp_name') and args.exp_name else f"exp_{timestamp}"

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
        steps_per_epoch = len(self.train_loader) // self.args.gradient_accumulation_steps
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

    def _init_csv_logger(self):
        """Initialize CSV logger with header"""
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

        for epoch in range(self.args.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")

            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch + 1}")

            for batch_idx, batch in progress_bar:
                # Training step (model-specific implementation)
                loss = self.train_step(batch)

                # Backward pass
                loss.backward()

                # Accumulate loss for logging
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

                    # Update progress bar
                    current_lr = self.optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'loss': f'{running_loss / (self.args.logging_steps * self.args.gradient_accumulation_steps):.4f}',
                        'lr': f'{current_lr:.2e}'
                    })

                    # Logging
                    if self.global_step % self.args.logging_steps == 0:
                        self._log_training_metrics(running_loss)
                        running_loss = 0

                    # Evaluation
                    if self.args.do_eval and self.global_step % self.args.eval_steps == 0:
                        # Log current training loss before evaluation (if we haven't logged at this step yet)
                        if self.global_step % self.args.logging_steps != 0 and running_loss > 0:
                            self._log_training_metrics(running_loss)
                        
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

    def _log_training_metrics(self, running_loss):
        """
        Log training metrics

        Args:
            running_loss: Accumulated loss since last logging
        """
        # Calculate average loss and perplexity
        avg_loss = running_loss / (self.args.logging_steps * self.args.gradient_accumulation_steps)
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

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__ if hasattr(self, 'config') else {},
            "args": vars(self.args)
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(f"Checkpoint loaded: step={self.global_step}, epoch={self.epoch}, best_val_loss={self.best_val_loss:.4f}")
