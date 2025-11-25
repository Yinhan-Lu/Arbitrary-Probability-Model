"""
Sigma GPT Training Script

Trains Sigma GPT model with arbitrary conditional probability modeling.
Supports both Fair mode (~40% learning) and Full mode (100% learning).

Based on:
- Sigma GPT paper (ArXiv 2404.09562)
- GPT-2 training configuration

Usage:
    # Fair mode (recommended for comparison with existing model)
    python train_sigmagpt.py --mode fair --model_config distilgpt2

    # Full mode (maximum learning efficiency)
    python train_sigmagpt.py --mode full --model_config distilgpt2

Key differences from standard GPT-2 training:
- Uses order tensors for arbitrary generation order
- Supports fair mode (40% learning) and full mode (100% learning)
- Compatible with ConditionalAugmenter for data augmentation
"""

import sys
import argparse
import logging
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
import time
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model.sigmagpt_model import SigmaGPT
from model.config import GPT2Config
from model.token_manager import TokenManager
from train.sigmagpt_adapter import SigmaGPTDataAdapter
from train.augmentation import ConditionalAugmenter
from train.blockwise_sampling import (
    uniform_num_conditioning_distribution,
    uniform_num_blocks_distribution,
    uniform_block_sizes_distribution,
    uniform_num_evaluation_distribution
)
from train.dataset import get_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to: {seed}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Sigma GPT Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========== Model Configuration ==========
    parser.add_argument("--model_config", type=str, default="distilgpt2",
                       choices=["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large", "custom"],
                       help="Model configuration (uses HuggingFace config names)")
    parser.add_argument("--mode", type=str, default="fair",
                       choices=["fair", "full"],
                       help="Training mode: 'fair' (~40%% learning) or 'full' (100%% learning)")

    # Custom model config (only used if model_config="custom")
    parser.add_argument("--vocab_size", type=int, default=50257,
                       help="Vocabulary size (only for custom config)")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--n_layer", type=int, default=6,
                       help="Number of layers (only for custom config)")
    parser.add_argument("--n_head", type=int, default=12,
                       help="Number of attention heads (only for custom config)")
    parser.add_argument("--n_embd", type=int, default=768,
                       help="Embedding dimension (only for custom config)")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")

    # ========== Dataset Configuration ==========
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                       help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1",
                       help="Dataset configuration")
    parser.add_argument("--num_train_samples", type=int, default=None,
                       help="Number of training samples (None = use all)")
    parser.add_argument("--num_eval_samples", type=int, default=10000,
                       help="Number of evaluation samples")
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming dataset")

    # ========== Training Configuration ==========
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")

    # ========== Optimizer Configuration ==========
    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                       help="Peak learning rate (Sigma GPT uses 2.5e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="Weight decay (Sigma GPT uses 0.1)")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.95,
                       help="Adam beta2 (Sigma GPT uses 0.95)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Gradient clipping norm")

    # ========== Learning Rate Schedule ==========
    parser.add_argument("--warmup_steps", type=int, default=2000,
                       help="Learning rate warmup steps")
    parser.add_argument("--lr_decay_steps", type=int, default=None,
                       help="Learning rate decay steps (None = use total steps)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                       help="Minimum learning rate as ratio of peak LR")

    # ========== Conditioning Configuration ==========
    parser.add_argument("--conditioning_sampling", type=str, default="blockwise",
                       choices=["blockwise", "uniform"],
                       help="Conditioning sampling strategy")
    parser.add_argument("--evaluation_sampling", type=str, default="blockwise",
                       choices=["blockwise", "uniform"],
                       help="Evaluation sampling strategy")
    parser.add_argument("--max_cond_blocks", type=int, default=2,
                       help="Maximum number of conditioning blocks")
    parser.add_argument("--max_eval_blocks", type=int, default=1,
                       help="Maximum number of evaluation blocks")

    # NEW: Eric's two training methods
    parser.add_argument("--ordering_mode", type=str, default="temporal",
                       choices=["temporal", "random_scramble"],
                       help="Ordering mode: 'temporal' (Method 1) or 'random_scramble' (Method 2)")

    # ========== Evaluation Configuration ==========
    parser.add_argument("--eval_splits_file", type=str, default=None,
                       help="Pre-generated evaluation splits file for deterministic evaluation")

    # ========== Logging & Checkpointing ==========
    parser.add_argument("--output_dir", type=str, default="./experiments",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--exp_name", type=str, default=None,
                       help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                       help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=5000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--max_eval_batches", type=int, default=50,
                       help="Maximum number of batches for evaluation")

    # ========== System Configuration ==========
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],
                       help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training (FP16)")

    return parser.parse_args()


def get_config(args):
    """Get model configuration"""
    if args.model_config == "custom":
        config = GPT2Config(
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout
        )
    else:
        # Load from HuggingFace config
        from transformers import GPT2Config as HFConfig
        hf_config = HFConfig.from_pretrained(args.model_config)
        config = GPT2Config(
            vocab_size=hf_config.vocab_size,
            max_seq_len=args.max_seq_len,
            n_layer=hf_config.n_layer,
            n_head=hf_config.n_head,
            n_embd=hf_config.n_embd,
            dropout=args.dropout
        )

    return config


def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio):
    """
    Get cosine learning rate scheduler with warmup

    Schedule:
    - Linear warmup from 0 to peak_lr over warmup_steps
    - Cosine decay from peak_lr to min_lr over remaining steps
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, augmenter, adapter, optimizer, scheduler, scaler,
                device, args, epoch, global_step, log_file):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    epoch_start_time = time.time()
    step_times = []

    for batch_idx, batch in enumerate(dataloader):
        step_start = time.time()

        # Get input_ids
        input_ids = batch['input_ids'].to(device)

        # Data augmentation (on CPU to reduce GPU memory)
        # Note: Must use augment_sequence (not augment_batch) to preserve conditioning/evaluation indices
        input_ids_cpu = input_ids.cpu()
        batch_size = input_ids_cpu.size(0)
        aug_batch = []
        for i in range(batch_size):
            result = augmenter.augment_sequence(input_ids_cpu[i], device='cpu')
            aug_batch.append(result)

        # Convert to Sigma GPT format
        sigmagpt_batch = adapter.convert_batch(aug_batch, input_ids_cpu)

        # Move to device
        inputs = sigmagpt_batch['inputs'].to(device)
        order = sigmagpt_batch['order'].to(device)
        targets = sigmagpt_batch['targets'].to(device)

        # Forward pass with mixed precision
        if args.fp16:
            with torch.cuda.amp.autocast():
                logits, loss = model(idx=inputs, order=order, targets=targets)
        else:
            logits, loss = model(idx=inputs, order=order, targets=targets)

        # Scale loss by gradient accumulation
        loss = loss / args.gradient_accumulation_steps

        # Backward pass
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            # Statistics
            valid_tokens = (targets != -1).sum().item()
            total_loss += loss.item() * args.gradient_accumulation_steps * valid_tokens
            total_tokens += valid_tokens

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Logging
            if global_step % args.logging_steps == 0:
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                current_lr = scheduler.get_last_lr()[0]
                throughput = valid_tokens / step_time if step_time > 0 else 0

                log_msg = (
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Throughput: {throughput:.1f} tok/s"
                )
                logger.info(log_msg)

                # Write to log file
                with open(log_file, 'a') as f:
                    f.write(f"{global_step},{avg_loss},{current_lr},{throughput}\n")

            # Save checkpoint
            if global_step % args.save_steps == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, args)

    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

    logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f}")

    return global_step, avg_loss


@torch.no_grad()
def evaluate(model, dataloader, augmenter, adapter, device, args):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_eval_batches:
            break

        # Get input_ids
        input_ids = batch['input_ids'].to(device)

        # Data augmentation
        # Note: Must use augment_sequence (not augment_batch) to preserve conditioning/evaluation indices
        input_ids_cpu = input_ids.cpu()
        batch_size = input_ids_cpu.size(0)
        aug_batch = []
        for i in range(batch_size):
            result = augmenter.augment_sequence(input_ids_cpu[i], device='cpu')
            aug_batch.append(result)

        # Convert to Sigma GPT format
        sigmagpt_batch = adapter.convert_batch(aug_batch, input_ids_cpu)

        # Move to device
        inputs = sigmagpt_batch['inputs'].to(device)
        order = sigmagpt_batch['order'].to(device)
        targets = sigmagpt_batch['targets'].to(device)

        # Forward pass
        logits, loss = model(idx=inputs, order=order, targets=targets)

        # Statistics
        valid_tokens = (targets != -1).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
        num_batches += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')

    return avg_loss, perplexity


def save_checkpoint(model, optimizer, scheduler, global_step, args):
    """Save model checkpoint"""
    exp_dir = Path(args.output_dir) / args.exp_name
    checkpoint_dir = exp_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = checkpoint_dir / "model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        'config': model.config.__dict__
    }, model_path)

    logger.info(f"Checkpoint saved to {checkpoint_dir}")


def main():
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"sigmagpt_{args.mode}_{args.model_config}_{timestamp}"

    # Create output directory
    exp_dir = Path(args.output_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Save configuration
    config_file = exp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.info("=" * 80)
    logger.info(f"Sigma GPT Training - {args.mode.upper()} Mode")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Output directory: {exp_dir}")

    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Get model configuration
    config = get_config(args)
    logger.info(f"Model config: {args.model_config}")
    logger.info(f"  - Layers: {config.n_layer}")
    logger.info(f"  - Heads: {config.n_head}")
    logger.info(f"  - Embedding: {config.n_embd}")
    logger.info(f"  - Max sequence length: {config.max_seq_len}")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize token manager
    token_manager = TokenManager(
        tokenizer=tokenizer,
        add_mask_token=True,
        add_bos_token=False
    )
    logger.info("Token manager initialized")

    # Initialize model
    logger.info("Initializing Sigma GPT model...")
    model = SigmaGPT(config)
    model = model.to(device)
    logger.info(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")

    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        device_type=device.type
    )
    logger.info(f"Optimizer: AdamW (lr={args.learning_rate}, betas=({args.adam_beta1}, {args.adam_beta2}))")

    # Initialize data augmenter
    augmenter = ConditionalAugmenter(
        mask_token_id=token_manager.mask_token_id,
        bos_token_id=token_manager.bos_token_id,
        max_seq_len=config.max_seq_len,
        num_conditioning_distribution=uniform_num_conditioning_distribution,
        num_blocks_distribution=uniform_num_blocks_distribution,
        block_sizes_distribution=uniform_block_sizes_distribution,
        num_evaluation_distribution=uniform_num_evaluation_distribution,
        num_eval_blocks_distribution=uniform_num_blocks_distribution,
        eval_block_sizes_distribution=uniform_block_sizes_distribution,
        conditioning_sampling=args.conditioning_sampling,
        evaluation_sampling=args.evaluation_sampling,
        max_cond_blocks=args.max_cond_blocks,
        max_eval_blocks=args.max_eval_blocks,
        tokenizer_pad_token_id=tokenizer.pad_token_id,
        ordering_mode=args.ordering_mode  # NEW: Eric's two methods
    )
    logger.info(f"Augmenter initialized (conditioning: {args.conditioning_sampling}, evaluation: {args.evaluation_sampling}, ordering: {args.ordering_mode})")

    # NEW: Set eval splits file if provided
    if args.eval_splits_file:
        augmenter.eval_splits_file = args.eval_splits_file
        logger.info(f"Using deterministic evaluation splits from: {args.eval_splits_file}")

    # Initialize adapter
    adapter = SigmaGPTDataAdapter(mode=args.mode)
    logger.info(f"Adapter initialized (mode: {args.mode})")

    # Create dataloaders
    logger.info("Loading datasets...")
    train_dataloader = get_dataloader(
        config=config,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        streaming=args.streaming,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        primary_dataset_only=True,
        num_samples=args.num_train_samples
    )

    eval_dataloader = get_dataloader(
        config=config,
        split='validation',
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        streaming=False,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        primary_dataset_only=True,
        num_samples=args.num_eval_samples
    )
    logger.info(f"Dataloaders created (train workers: {args.num_workers})")

    # Calculate total steps
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    if args.lr_decay_steps is None:
        args.lr_decay_steps = total_steps

    logger.info(f"Training schedule:")
    logger.info(f"  - Steps per epoch: {steps_per_epoch}")
    logger.info(f"  - Total steps: {total_steps}")
    logger.info(f"  - Warmup steps: {args.warmup_steps}")

    # Initialize learning rate scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        args.warmup_steps,
        args.lr_decay_steps,
        args.min_lr_ratio
    )

    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # Create log file
    log_file = log_dir / "training_log.csv"
    with open(log_file, 'w') as f:
        f.write("step,loss,learning_rate,throughput\n")

    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        # Train
        global_step, train_loss = train_epoch(
            model, train_dataloader, augmenter, adapter,
            optimizer, scheduler, scaler, device, args,
            epoch, global_step, log_file
        )

        # Evaluate
        if epoch % 1 == 0:  # Evaluate every epoch
            logger.info("Running evaluation...")
            eval_loss, perplexity = evaluate(
                model, eval_dataloader, augmenter, adapter, device, args
            )
            logger.info(f"Evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(model, optimizer, scheduler, global_step, args)
                logger.info(f"New best model saved (eval loss: {eval_loss:.4f})")

    # Save final model
    save_checkpoint(model, optimizer, scheduler, global_step, args)
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Final model saved to: {exp_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
