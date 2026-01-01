"""
Diffusion Trainer for Discrete Diffusion Language Modeling

Implements model-specific logic for training discrete diffusion models
(MDLM-style absorbing state diffusion) for arbitrary conditional
probability modeling P(X_e | X_c).

Key features:
- Timestep-conditioned noise injection
- Conditional diffusion (X_c positions never masked)
- Bidirectional attention (not causal)
- Loss only on masked positions
- 5-mode evaluation adapted for diffusion
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
from copy import copy
from functools import partial
import math

from train.base_trainer import BaseTrainer
from model.config import get_config
from model.diffusion_gpt2 import DiffusionGPT2Model
from model.diffusion_utils import NoiseSchedule, indices_to_mask
from model.token_manager import TokenManager
from train.dataset import get_dataloader, create_simple_collate_fn, create_indices_sampling_collate_fn
from train.augmentation import ConditionalAugmenter
from train.blockwise_sampling import (
    uniform_num_conditioning_distribution,
    uniform_num_blocks_distribution,
    uniform_block_sizes_distribution,
    uniform_num_evaluation_distribution,
)

logger = logging.getLogger(__name__)


class DiffusionTrainer(BaseTrainer):
    """
    Trainer for discrete diffusion language modeling

    Extends BaseTrainer with:
    - NoiseSchedule for diffusion process
    - Conditional noise injection (X_c positions fixed)
    - DiffusionGPT2Model with timestep conditioning
    - Adapted 5-mode evaluation for diffusion
    """

    def setup_model(self):
        """
        Setup diffusion model with noise schedule

        Diffusion model requires:
        - TokenManager for [M] mask token
        - NoiseSchedule for forward diffusion
        - DiffusionGPT2Model with timestep embedding
        """
        logger.info("Setting up diffusion model...")

        # Get model configuration (use diffusion-specific config if available)
        config_name = self.args.model_config
        # Map to diffusion config if standard config name is provided
        if not config_name.startswith('diffusion_'):
            diffusion_config_name = f'diffusion_{config_name}'
            try:
                self.config = get_config(diffusion_config_name)
                logger.info(f"Using diffusion config: {diffusion_config_name}")
            except ValueError:
                # Fallback to standard config
                self.config = get_config(config_name)
                logger.info(f"Using standard config: {config_name}")
        else:
            self.config = get_config(config_name)

        # Set position encoding type from args
        if hasattr(self.args, 'position_encoding_type'):
            self.config.position_encoding_type = self.args.position_encoding_type
            logger.info(f"Position encoding type: {self.config.position_encoding_type}")
        if hasattr(self.args, 'rope_base'):
            self.config.rope_base = self.args.rope_base

        # Set gradient checkpointing from args
        if hasattr(self.args, 'gradient_checkpointing'):
            self.config.gradient_checkpointing = self.args.gradient_checkpointing
            if self.config.gradient_checkpointing:
                logger.info("Gradient checkpointing: ENABLED")

        # Initialize token manager for [MASK] token
        self.token_manager = TokenManager(
            add_mask_token=True,
            add_bos_token=False
        )

        # Get tokenizer and special token IDs
        self.tokenizer = self.token_manager.get_tokenizer()
        special_tokens = self.token_manager.get_special_token_ids()
        self.mask_token_id = special_tokens["mask_token_id"]
        logger.info(f"Mask token ID: {self.mask_token_id}")

        # Get diffusion hyperparameters
        self.num_diffusion_steps = getattr(self.args, 'num_diffusion_steps', 1000)
        self.noise_schedule_type = getattr(self.args, 'noise_schedule', 'cosine')
        self.time_emb_type = getattr(self.args, 'time_emb_type', 'sinusoidal')

        logger.info(f"Diffusion configuration:")
        logger.info(f"  Number of timesteps: {self.num_diffusion_steps}")
        logger.info(f"  Noise schedule: {self.noise_schedule_type}")
        logger.info(f"  Time embedding: {self.time_emb_type}")

        # Create noise schedule
        self.noise_schedule = NoiseSchedule(
            num_timesteps=self.num_diffusion_steps,
            schedule_type=self.noise_schedule_type,
            device=self.device
        )

        # Create diffusion model
        self.model = DiffusionGPT2Model(
            self.config,
            mask_token_id=self.mask_token_id,
            num_timesteps=self.num_diffusion_steps,
            time_emb_type=self.time_emb_type
        ).to(self.device)

        # Resize embeddings to include mask token
        self.model = self.token_manager.resize_model_embeddings(self.model)

        total_params = self.model.get_num_params()
        logger.info(f"Model parameters: {total_params/1e6:.2f}M")

    def setup_data(self):
        """
        Setup data loaders with conditioning index sampling

        Uses the same dataset and augmenter as conditional model to ensure
        fair comparison with identical X_c/X_e splits.
        """
        logger.info("Loading dataset...")

        # Use same sequence length as config
        self.seq_len = self.config.max_seq_len
        logger.info(f"Sequence length: {self.seq_len}")

        # Get conditioning parameters
        self.cond_pct_min = getattr(self.args, 'cond_pct_min', 0.0)
        self.cond_pct_max = getattr(self.args, 'cond_pct_max', 0.4)
        self.eval_pct_min = getattr(self.args, 'eval_pct_min', 0.2)
        self.eval_pct_max = getattr(self.args, 'eval_pct_max', 0.4)

        logger.info(f"Conditioning percentage range: [{self.cond_pct_min}, {self.cond_pct_max}]")
        logger.info(f"Evaluation percentage range: [{self.eval_pct_min}, {self.eval_pct_max}]")

        # Create conditional augmenter (reuses same code as ConditionalTrainer)
        self.augmenter = ConditionalAugmenter(
            mask_token_id=self.mask_token_id,
            bos_token_id=self.tokenizer.eos_token_id,  # Reuse EOS as BOS
            max_seq_len=self.seq_len,
            cond_pct_max=self.cond_pct_max,
            tokenizer_pad_token_id=self.tokenizer.pad_token_id or 50256,
            num_conditioning_distribution=partial(
                uniform_num_conditioning_distribution,
                conditioning_percentage_range=(self.cond_pct_min, self.cond_pct_max)
            ),
            num_blocks_distribution=uniform_num_blocks_distribution,
            block_sizes_distribution=uniform_block_sizes_distribution,
            num_evaluation_distribution=partial(
                uniform_num_evaluation_distribution,
                evaluation_percentage_range=(self.eval_pct_min, self.eval_pct_max)
            ),
            num_eval_blocks_distribution=uniform_num_blocks_distribution,
            eval_block_sizes_distribution=uniform_block_sizes_distribution,
            conditioning_sampling=getattr(self.args, 'conditioning_sampling', 'blockwise'),
            evaluation_sampling=getattr(self.args, 'evaluation_sampling', 'blockwise'),
        )

        # Create collate function that samples conditioning/evaluation indices
        collate_fn = create_indices_sampling_collate_fn(
            augmenter=self.augmenter,
            use_attention_mask_for_valid=True
        )

        # Create training dataloader
        self.train_loader = get_dataloader(
            tokenizer=self.tokenizer,
            max_length=self.seq_len,
            num_samples=self.args.num_train_samples,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            seed=self.args.seed,
            collate_fn=collate_fn
        )
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")

        # Create validation dataloader
        if self.args.do_eval:
            val_collate_fn = create_simple_collate_fn(pad_token_id=self.tokenizer.pad_token_id or 50256)
            self.val_loader = get_dataloader(
                tokenizer=self.tokenizer,
                max_length=self.seq_len,
                num_samples=self.args.num_eval_samples,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                split='validation',
                collate_fn=val_collate_fn
            )
            logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

    def train_step(self, batch):
        """
        Single diffusion training step

        Process:
        1. Get conditioning/evaluation indices from batch
        2. Sample random timestep t
        3. Apply conditional noise (X_c fixed)
        4. Forward pass with timestep conditioning
        5. Compute loss on masked positions
        """
        # Get data from batch
        input_ids = batch['input_ids'].to(self.device)
        batch_size = input_ids.size(0)

        # Get conditioning indices from collate function
        cond_idx = batch.get('conditional_idx', [[] for _ in range(batch_size)])
        eval_idx = batch.get('evaluation_idx', [[] for _ in range(batch_size)])

        # Sample random timesteps
        t = self.noise_schedule.sample_timesteps(batch_size, device=self.device)

        # Apply conditional noise (X_c positions remain original)
        x_t, noise_mask = self.noise_schedule.add_noise_conditional(
            input_ids, t, self.mask_token_id, cond_idx
        )

        # Forward pass with FP16 support
        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits = self.model(x_t, t)
                loss = self._compute_diffusion_loss(logits, input_ids, noise_mask)
        else:
            logits = self.model(x_t, t)
            loss = self._compute_diffusion_loss(logits, input_ids, noise_mask)

        return loss

    def _compute_diffusion_loss(self, logits, x_0, noise_mask):
        """
        Compute cross-entropy loss on masked positions

        Args:
            logits: Model predictions (batch_size, seq_len, vocab_size)
            x_0: Original clean sequence (batch_size, seq_len)
            noise_mask: Boolean mask of corrupted positions (batch_size, seq_len)

        Returns:
            loss: Scalar loss value
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for cross entropy
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = x_0.view(-1)
        mask_flat = noise_mask.view(-1).float()

        # Compute per-token loss
        loss = torch.nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='none'
        )

        # Average over masked positions only
        loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)

        return loss

    def evaluate(self):
        """
        5-mode evaluation adapted for diffusion

        Uses importance sampling over noise levels to estimate NLL.
        Same evaluation positions (X_c/X_e splits) as conditional model.
        """
        self.model.eval()
        logger.info("Running diffusion evaluation...")

        # Import evaluation function
        from train.diffusion_evaluation_modes import diffusion_evaluate_all_modes

        eval_results = diffusion_evaluate_all_modes(
            model=self.model,
            noise_schedule=self.noise_schedule,
            dataloader=self.val_loader,
            device=self.device,
            augmenter=self.augmenter,
            mask_token_id=self.mask_token_id,
            max_batches=getattr(self.args, 'max_eval_batches', None),
            num_nll_samples=getattr(self.args, 'num_nll_samples', 10),
            trainer_args=self.args
        )

        # Use Mode 3 as primary metric (same as conditional model)
        eval_results["loss"] = eval_results.get("mode3_loss", eval_results.get("mode1_loss", 0))

        return eval_results

    def _get_extra_csv_columns(self):
        """Add diffusion-specific CSV columns"""
        return ["num_diffusion_steps", "noise_schedule"]

    def _get_extra_train_metrics(self):
        """Add diffusion-specific training metrics"""
        return {
            "num_diffusion_steps": self.num_diffusion_steps,
            "noise_schedule": self.noise_schedule_type
        }

    def _get_extra_eval_metrics(self, eval_results):
        """Add diffusion-specific evaluation metrics"""
        return {
            "num_diffusion_steps": self.num_diffusion_steps,
            "noise_schedule": self.noise_schedule_type
        }


if __name__ == "__main__":
    # Quick test of DiffusionTrainer
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default="tiny")
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--noise_schedule", default="cosine")
    parser.add_argument("--num_train_samples", type=int, default=100)
    parser.add_argument("--num_eval_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--do_eval", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="./experiments")
    parser.add_argument("--exp_name", default="diffusion_test")
    parser.add_argument("--cond_pct_min", type=float, default=0.0)
    parser.add_argument("--cond_pct_max", type=float, default=0.3)
    parser.add_argument("--eval_pct_min", type=float, default=0.2)
    parser.add_argument("--eval_pct_max", type=float, default=0.3)
    parser.add_argument("--max_eval_batches", type=int, default=5)

    args = parser.parse_args([])

    print("=" * 80)
    print("Testing DiffusionTrainer")
    print("=" * 80)

    trainer = DiffusionTrainer(args)
    print(f"\nModel created with {trainer.model.get_num_params():,} parameters")
    print(f"Noise schedule: {trainer.noise_schedule_type} with {trainer.num_diffusion_steps} steps")

    # Test a single training step
    print("\nTesting single training step...")
    batch = next(iter(trainer.train_loader))
    loss = trainer.train_step(batch)
    print(f"Training loss: {loss.item():.4f}")

    print("\n" + "=" * 80)
    print("DiffusionTrainer test passed!")
    print("=" * 80)
