"""
Main training script for GPT-2 model
Supports both DistilGPT-2 and Tiny configurations
"""

import sys
from pathlib import Path
import argparse
import torch
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import get_config, print_config_summary
from model.arbitrary_prob_gpt2 import GPT2Model
from train.dataset import get_dataloader
from train.loop import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 model on Wikipedia")

    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=["distilgpt2", "tiny", "nano"],
        help="Model configuration to use"
    )

    # Data arguments
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=None,
        help="Number of training samples (None for all)"
    )
    parser.add_argument(
        "--num_val_samples",
        type=int,
        default=None,
        help="Number of validation samples (None for all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    # Training arguments
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Peak learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )

    # Logging and checkpointing
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Steps between logging"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Steps between validation"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Steps between checkpoints"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to train on"
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Print configuration summary
    logger.info("\n" + "=" * 80)
    logger.info("GPT-2 Training Script")
    logger.info("=" * 80)
    print_config_summary()

    # Load model configuration
    logger.info(f"\nLoading model configuration: {args.config}")
    config = get_config(args.config)

    # Create model
    logger.info("\nCreating model...")
    model = GPT2Model(config)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    logger.info(f"Train samples: {args.num_train_samples if args.num_train_samples else 'all'}")
    logger.info(f"Val samples: {args.num_val_samples if args.num_val_samples else 'all'}")
    logger.info(f"Batch size: {args.batch_size}")

    train_loader = get_dataloader(
        config,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_train_samples
    )

    val_loader = get_dataloader(
        config,
        split="validation",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_val_samples
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create trainer
    logger.info("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        device=device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80 + "\n")

    try:
        trainer.train()
        logger.info("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("\n⚠ Training interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.current_step // len(train_loader),
            step=trainer.current_step
        )
    except Exception as e:
        logger.error(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
