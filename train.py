"""
Unified Training Script for Arbitrary Conditional Probability Models and Baselines

This script provides a unified interface for training different model types:
- conditional: Arbitrary conditional probability model P(X_e | X_c)
- baseline: Standard autoregressive baseline (e.g., DistilGPT-2)

Usage:
    # Train conditional model
    python train.py --model_type conditional \
        --model_config small \
        --num_epochs 3 \
        --cond_pct_min 0.2 \
        --cond_pct_max 0.4

    # Train baseline model
    python train.py --model_type baseline \
        --model_config distilgpt2 \
        --num_epochs 3 \
        --fp16

Key features:
- Unified training infrastructure ensures fair baseline comparisons
- All models use identical optimizer, scheduler, and training loop
- Model-specific logic isolated in separate trainer classes
- Easy to add new baselines by implementing 4 abstract methods
"""

import sys
import argparse
import logging
from pathlib import Path
import random
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from train.conditional_trainer import ConditionalTrainer
from train.baseline_trainer import BaselineTrainer
from train.sigmagpt_trainer import SigmaGPTTrainer
from train.distilbert_trainer import DistilBertTrainer
from train.diffusion_trainer import DiffusionTrainer
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Set random seed for reproducibility across all random sources

    This function sets seeds for:
    - Python's built-in random module
    - NumPy random generator
    - PyTorch CPU operations
    - PyTorch CUDA operations (all GPUs)

    Args:
        seed: Integer seed value

    Note:
        - DataLoader workers need separate seed control via worker_init_fn
        - For full determinism, also set CUBLAS and CUDNN environment variables
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional: Enable deterministic mode (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to: {seed}")


def parse_args():
    """
    Parse command-line arguments

    Arguments are divided into:
    1. Model type selection
    2. Common arguments (shared by all models)
    3. Model-specific arguments (conditional or baseline)
    """
    parser = argparse.ArgumentParser(
        description="Unified Training Script for Conditional Models and Baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========== Model Type Selection ==========
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["conditional", "baseline", "sigmagpt", "distilbert", "diffusion"],
        help="Type of model to train: 'conditional', 'baseline', 'sigmagpt', 'distilbert', or 'diffusion'"
    )

    # ========== Common Arguments (Shared by All Models) ==========

    # Model arguments
    parser.add_argument(
        "--model_config",
        type=str,
        default="distilgpt2",
        help="Model configuration (e.g., 'distilgpt2', 'gpt2', 'gpt2_medium', 'small', 'tiny')"
    )
    parser.add_argument(
        "--position_encoding_type",
        type=str,
        default="learned",
        choices=["learned", "rope"],
        help="Position encoding type: 'learned' (default) or 'rope' (Rotary Position Embedding)"
    )
    parser.add_argument(
        "--rope_base",
        type=float,
        default=10000.0,
        help="Base frequency for RoPE (only used when position_encoding_type='rope')"
    )

    # Data arguments
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=10000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=1000,
        help="Number of evaluation samples"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce GPU memory usage (trades compute for memory)"
    )

    # Optimizer arguments
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
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1"
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Adam epsilon"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )

    # Scheduler arguments
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--warmup_start_factor",
        type=float,
        default=0.1,
        help="Starting factor for warmup scheduler"
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="Minimum learning rate ratio for cosine annealing"
    )

    # Logging arguments
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Number of evaluations without improvement before stopping. 0 = disabled"
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=100,
        help="Maximum number of evaluation batches"
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Run evaluation during training"
    )

    # Resume from checkpoint
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (handles CSV truncation automatically)"
    )

    # Reproducibility arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments",
        help="Output directory for experiments"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="unified_training",
        help="Experiment name"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "mps", "cpu", "auto"],
        help="Device to use for training (auto: automatically select best available)"
    )

    # Parse known args first to get model_type
    args, unknown = parser.parse_known_args()

    # ========== Model-Specific Arguments ==========

    if args.model_type == "conditional":
        # Conditional model specific arguments
        parser.add_argument(
            "--pretrained_model_path",
            type=str,
            default=None,
            help="Path to pretrained model or HuggingFace model name (e.g., 'gpt2', 'distilgpt2')"
        )
        parser.add_argument(
            "--resume_training",
            action="store_true",
            help="Resume training state (optimizer, scheduler, epoch) from checkpoint"
        )

        # Conditional modeling arguments
        parser.add_argument(
            "--cond_pct_min",
            type=float,
            default=0.2,
            help="Minimum percentage for conditioning tokens"
        )
        parser.add_argument(
            "--cond_pct_max",
            type=float,
            default=0.4,
            help="Maximum percentage for conditioning tokens"
        )
        parser.add_argument(
            "--eval_pct_min",
            type=float,
            default=0.2,
            help="Minimum percentage for evaluation tokens"
        )
        parser.add_argument(
            "--eval_pct_max",
            type=float,
            default=0.4,
            help="Maximum percentage for evaluation tokens"
        )
        parser.add_argument(
            "--min_conditioning",
            type=int,
            default=1,
            help="Minimum number of conditioning tokens"
        )
        parser.add_argument(
            "--min_evaluation",
            type=int,
            default=1,
            help="Minimum number of evaluation tokens"
        )
        parser.add_argument(
            "--conditioning_sampling",
            type=str,
            default="blockwise",
            choices=["random", "blockwise"],
            help="Sampling mode for conditioning set: 'random' or 'blockwise'"
        )
        parser.add_argument(
            "--evaluation_sampling",
            type=str,
            default="blockwise",
            choices=["random", "blockwise"],
            help="Sampling mode for evaluation set: 'random' or 'blockwise'"
        )
        parser.add_argument(
            "--detach_augmentation",
            action="store_true",
            default=False,
            help="Detach augmentation tensors to prevent gradient flow through augmentation operations. "
                 "This makes internal augmentation behave like legacy external augmentation (for debugging/comparison)"
        )

        # Blockwise sampling parameters
        parser.add_argument(
            "--max_cond_blocks",
            type=int,
            default=None,
            help="Maximum number of conditioning blocks (default: None = no limit, use num_cond_tokens)"
        )

        # Mode 2 (Boundary filling) evaluation parameters
        parser.add_argument(
            "--mode2_boundary_cond_pct_min",
            type=float,
            default=0.1,
            help="Minimum conditioning percentage for Mode 2 boundary evaluation"
        )
        parser.add_argument(
            "--mode2_boundary_cond_pct_max",
            type=float,
            default=0.3,
            help="Maximum conditioning percentage for Mode 2 boundary evaluation"
        )

        # Bug fix ablation switch
        parser.add_argument(
            "--use_attention_mask_for_valid",
            action="store_true",
            default=True,
            help="Use attention_mask to determine valid positions (new correct behavior). "
                 "If False, use pad_token_id (old buggy behavior that excludes EOS tokens)."
        )
        parser.add_argument(
            "--no_use_attention_mask_for_valid",
            dest="use_attention_mask_for_valid",
            action="store_false",
            help="Use pad_token_id to determine valid positions (old buggy behavior)"
        )

    elif args.model_type in ["baseline", "distilbert"]:
        # Baseline model specific arguments
        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Use mixed precision training (FP16)"
        )
        parser.add_argument(
            "--streaming",
            action="store_true",
            help="Use streaming dataset"
        )
        parser.add_argument(
            "--dataset_name",
            type=str,
            default="wikipedia",
            help="Dataset name"
        )
        parser.add_argument(
            "--dataset_config",
            type=str,
            default="20220301.en",
            help="Dataset configuration"
        )
        parser.add_argument(
            "--cond_pct_min",
            type=float,
            default=0.2,
            help="Minimum percentage of conditioning tokens for BERT evaluation mode 3"
        )
        parser.add_argument(
            "--cond_pct_max",
            type=float,
            default=0.4,
            help="Maximum percentage of conditioning tokens for BERT evaluation mode 3"
        )

    elif args.model_type == "sigmagpt":
        # Sigma GPT specific arguments
        parser.add_argument(
            "--sigmagpt_mode",
            type=str,
            default="fair",
            choices=["fair", "full"],
            help="Sigma GPT training mode: 'fair' (~40%% learning) or 'full' (100%% learning)"
        )
        parser.add_argument(
            "--sigmagpt_arch",
            type=str,
            default="new",
            choices=["new", "old"],
            help="Sigma GPT architecture: 'new' (from baseline) or 'old' (double position encoding from paper)"
        )
        parser.add_argument(
            "--sigmagpt_eval_mode",
            type=str,
            default="autoregressive",
            choices=["autoregressive", "training_dist"],
            help="Evaluation mode: 'autoregressive' (paper's left-to-right) or 'training_dist' (same as training)"
        )
        parser.add_argument(
            "--ordering_mode",
            type=str,
            default="temporal",
            choices=["temporal", "random_scramble"],
            help="Ordering mode for Sigma GPT: 'temporal' or 'random_scramble'"
        )
        parser.add_argument(
            "--conditioning_sampling",
            type=str,
            default="blockwise",
            choices=["random", "blockwise"],
            help="Conditioning sampling strategy"
        )
        parser.add_argument(
            "--evaluation_sampling",
            type=str,
            default="blockwise",
            choices=["random", "blockwise"],
            help="Evaluation sampling strategy"
        )
        parser.add_argument(
            "--max_cond_blocks",
            type=int,
            default=None,
            help="Maximum number of conditioning blocks (default: None = no limit, use num_cond_tokens)"
        )
        # Distribution parameters (must match training config for fair comparison)
        parser.add_argument(
            "--cond_pct_min",
            type=float,
            default=0.0,
            help="Minimum conditioning percentage (default: 0.0 = 0%%)"
        )
        parser.add_argument(
            "--cond_pct_max",
            type=float,
            default=0.4,
            help="Maximum conditioning percentage (default: 0.4 = 40%%)"
        )
        parser.add_argument(
            "--eval_pct_min",
            type=float,
            default=1.0,
            help="Minimum evaluation percentage of unknown (default: 1.0 = 100%%)"
        )
        parser.add_argument(
            "--eval_pct_max",
            type=float,
            default=1.0,
            help="Maximum evaluation percentage of unknown (default: 1.0 = 100%%)"
        )
        # Mode 2 (Boundary filling) evaluation parameters
        parser.add_argument(
            "--mode2_boundary_cond_pct_min",
            type=float,
            default=0.1,
            help="Minimum conditioning percentage for Mode 2 boundary evaluation"
        )
        parser.add_argument(
            "--mode2_boundary_cond_pct_max",
            type=float,
            default=0.3,
            help="Maximum conditioning percentage for Mode 2 boundary evaluation"
        )
        # Thinking tokens (latent computation prefix)
        parser.add_argument(
            "--use_thinking_tokens",
            action="store_true",
            help="Enable thinking tokens (learnable latent computation prefix)"
        )
        parser.add_argument(
            "--thinking_token_mode",
            type=str,
            default="expectation",
            choices=["expectation", "upper_bound"],
            help="Mode for computing thinking token count: "
                 "'expectation' (0.5 * cond_max * seq_len) or "
                 "'upper_bound' (cond_max * seq_len)"
        )
        parser.add_argument(
            "--streaming",
            action="store_true",
            help="Use streaming dataset"
        )
        parser.add_argument(
            "--dataset_name",
            type=str,
            default="wikitext",
            help="Dataset name"
        )
        parser.add_argument(
            "--dataset_config",
            type=str,
            default="wikitext-103-raw-v1",
            help="Dataset configuration"
        )
        parser.add_argument(
            "--primary_dataset_only",
            action="store_true",
            default=True,
            help="Use only primary dataset (no secondary augmentation)"
        )
        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Use mixed precision training (FP16)"
        )

    elif args.model_type == "diffusion":
        # Diffusion model specific arguments
        parser.add_argument(
            "--num_diffusion_steps",
            type=int,
            default=1000,
            help="Number of diffusion timesteps (T)"
        )
        parser.add_argument(
            "--noise_schedule",
            type=str,
            default="cosine",
            choices=["cosine", "linear", "sqrt"],
            help="Noise schedule type for diffusion"
        )
        parser.add_argument(
            "--time_emb_type",
            type=str,
            default="sinusoidal",
            choices=["sinusoidal", "learned"],
            help="Timestep embedding type: 'sinusoidal' or 'learned'"
        )
        parser.add_argument(
            "--num_nll_samples",
            type=int,
            default=10,
            help="Number of timestep samples for NLL estimation during evaluation"
        )
        # Conditioning parameters (same as conditional model)
        parser.add_argument(
            "--cond_pct_min",
            type=float,
            default=0.0,
            help="Minimum conditioning percentage"
        )
        parser.add_argument(
            "--cond_pct_max",
            type=float,
            default=0.4,
            help="Maximum conditioning percentage"
        )
        parser.add_argument(
            "--eval_pct_min",
            type=float,
            default=0.2,
            help="Minimum evaluation percentage"
        )
        parser.add_argument(
            "--eval_pct_max",
            type=float,
            default=0.4,
            help="Maximum evaluation percentage"
        )
        parser.add_argument(
            "--conditioning_sampling",
            type=str,
            default="blockwise",
            choices=["random", "blockwise"],
            help="Conditioning sampling strategy"
        )
        parser.add_argument(
            "--evaluation_sampling",
            type=str,
            default="blockwise",
            choices=["random", "blockwise"],
            help="Evaluation sampling strategy"
        )
        # Mode 2 (Boundary filling) evaluation parameters
        parser.add_argument(
            "--mode2_boundary_cond_pct_min",
            type=float,
            default=0.1,
            help="Minimum conditioning percentage for Mode 2 boundary evaluation"
        )
        parser.add_argument(
            "--mode2_boundary_cond_pct_max",
            type=float,
            default=0.3,
            help="Maximum conditioning percentage for Mode 2 boundary evaluation"
        )
        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Use mixed precision training (FP16)"
        )

    # Parse all arguments
    args = parser.parse_args()

    return args


def create_trainer(args):
    """
    Factory function to create appropriate trainer based on model_type

    Args:
        args: Parsed command-line arguments

    Returns:
        trainer: ConditionalTrainer, BaselineTrainer, or SigmaGPTTrainer instance
    """
    if args.model_type == "conditional":
        logger.info("Creating Conditional Trainer...")
        return ConditionalTrainer(args)
    elif args.model_type == "baseline":
        logger.info("Creating Baseline Trainer...")
        return BaselineTrainer(args)
    elif args.model_type == "sigmagpt":
        logger.info("Creating Sigma GPT Trainer...")
        return SigmaGPTTrainer(args)
    elif args.model_type == "distilbert":
        logger.info("Creating DistilBert Trainer...")
        return DistilBertTrainer(args)
    elif args.model_type == "diffusion":
        logger.info("Creating Diffusion Trainer...")
        return DiffusionTrainer(args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")


def main():
    """Main training function"""

    # Parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Print configuration
    logger.info("=" * 80)
    logger.info("Unified Training Script")
    logger.info("=" * 80)
    logger.info(f"Model Type: {args.model_type}")
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info("=" * 80)
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 80)

    # Create appropriate trainer
    trainer = create_trainer(args)

    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from, resume_csv=True)
        # Re-init CSV logger in append mode (don't overwrite)
        trainer._init_csv_logger(append=True)

    # Start training
    try:
        trainer.train()
        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
    except KeyboardInterrupt:
        logger.warning("\n" + "=" * 80)
        logger.warning("Training interrupted by user")
        logger.warning("=" * 80)
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error(f"Training failed with error: {e}")
        logger.error("=" * 80)
        raise


if __name__ == "__main__":
    main()
