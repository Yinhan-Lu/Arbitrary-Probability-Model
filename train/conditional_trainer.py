"""
Conditional Trainer for Arbitrary Conditional Probability Modeling

Implements model-specific logic for training conditional probability models P(X_e | X_c)
where X_c (conditioning set) and X_e (evaluation set) can be any token subsets.

Key features:
- Random conditioning/evaluation split during training
- Custom attention masks (attend to all conditions, block unknowns)
- Loss only on evaluation positions
- Support for fine-tuning from pretrained models
- 5-mode evaluation
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
from model.arbitrary_prob_gpt2 import GPT2Model
from model.token_manager import TokenManager
from train.dataset import get_dataloader, create_simple_collate_fn, create_indices_sampling_collate_fn
from train.augmentation import ConditionalAugmenter
from train.blockwise_sampling import (
    uniform_num_conditioning_distribution,
    uniform_num_blocks_distribution,
    uniform_block_sizes_distribution,
    uniform_num_evaluation_distribution,
)
from train.evaluation_modes import evaluate_all_modes

logger = logging.getLogger(__name__)


class ConditionalTrainer(BaseTrainer):
    """
    Trainer for arbitrary conditional probability modeling

    Extends BaseTrainer with:
    - TokenManager for special tokens ([M] mask token, BOS token)
    - ConditionalAugmenter for random conditioning/evaluation splits
    - Custom forward pass with index-based augmentation
    - 5-mode evaluation (autoregressive, boundary filling, training dist, etc.)
    """

    def setup_model(self):
        """
        Setup model with special tokens

        Conditional model requires:
        - TokenManager to handle [M] mask token and BOS token
        - Model initialization with special token IDs
        - Embedding resizing to accommodate new tokens
        - Optional pretrained weight loading
        """
        logger.info("Setting up conditional model with special tokens...")

        # Get model configuration
        self.config = get_config(self.args.model_config)

        # Set detach_augmentation parameter from args
        if hasattr(self.args, 'detach_augmentation'):
            self.config.detach_augmentation = self.args.detach_augmentation
            logger.info(f"Detach augmentation: {self.config.detach_augmentation}")

        # Set position encoding type from args (learned or rope)
        if hasattr(self.args, 'position_encoding_type'):
            self.config.position_encoding_type = self.args.position_encoding_type
            logger.info(f"Position encoding type: {self.config.position_encoding_type}")
        if hasattr(self.args, 'rope_base'):
            self.config.rope_base = self.args.rope_base

        # Initialize token manager with special tokens
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

        # Create model with special tokens for conditional probability modeling
        self.model = GPT2Model(
            self.config,
            mask_token_id=self.mask_token_id,
            bos_token_id=self.bos_token_id
        ).to(self.device)

        # Load pretrained weights if specified
        if hasattr(self.args, 'pretrained_model_path') and self.args.pretrained_model_path:
            self._load_pretrained_weights(self.args.pretrained_model_path)

        # Resize embeddings to include new tokens
        self.model = self.token_manager.resize_model_embeddings(self.model)

        total_params = self.model.get_num_params()
        logger.info(f"Model parameters: {total_params/1e6:.2f}M")

    def setup_data(self):
        """
        Setup data loaders with conditional augmentation

        Dataset configuration:
        - Total 1024 positions (0-1023), including BOS at position 0
        - BOS (1 token) + Body (1023 tokens) = 1024 total positions
        - Conditioning tokens are extra prefix: total_len = N_cond + 1024
        - Position IDs: BOS uses 0, Body uses 1-1023, Conditioning reuses 1-1023
        """
        logger.info("Loading dataset...")

        # Calculate body sequence length (reserve position 0 for BOS)
        self.body_seq_len = self.config.max_seq_len - 1
        logger.info(f"Dataset configuration for prefix conditioning:")
        logger.info(f"  Model max_seq_len: {self.config.max_seq_len} (positions 0-{self.config.max_seq_len-1})")
        logger.info(f"  Dataset max_length: {self.body_seq_len} (body tokens, positions 1-{self.config.max_seq_len-1})")
        logger.info(f"  Max conditioning percentage: {self.args.cond_pct_max}")

        # Calculate expected augmented sequence length
        max_n_cond = math.ceil(self.body_seq_len * self.args.cond_pct_max)
        aug_max_len = max_n_cond + self.config.max_seq_len
        logger.info(f"  Expected max augmented length: {aug_max_len}")
        logger.info(f"    = {max_n_cond} (prefix) + {self.config.max_seq_len} (BOS + body)")

        # Create dataset config with body_seq_len
        dataset_config = copy(self.config)
        dataset_config.max_seq_len = self.body_seq_len  # Dataloader provides body tokens only

        # Create conditional augmenter
        logger.info("Creating augmenter with distribution-based sampling")
        logger.info(f"  Conditioning percentage range: [{self.args.cond_pct_min}, {self.args.cond_pct_max}]")
        logger.info(f"  Evaluation percentage range: [{self.args.eval_pct_min}, {self.args.eval_pct_max}]")

        # Create distribution functions
        num_cond_dist = partial(
            uniform_num_conditioning_distribution,
            conditioning_percentage_range=(self.args.cond_pct_min, self.args.cond_pct_max)
        )
        num_cond_blocks_dist = partial(
            uniform_num_blocks_distribution,
            max_blocks=self.args.max_cond_blocks
        )

        num_eval_dist = partial(
            uniform_num_evaluation_distribution,
            evaluation_percentage_range=(self.args.eval_pct_min, self.args.eval_pct_max)
        )
        num_eval_blocks_dist = uniform_num_blocks_distribution

        self.augmenter = ConditionalAugmenter(
            mask_token_id=self.mask_token_id,
            bos_token_id=self.bos_token_id,
            max_seq_len=self.body_seq_len,
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

        # Use indices sampling collate function for performance optimization
        # This moves CPU-bound indices sampling from main process to DataLoader workers,
        # achieving 2-4x speedup by parallelizing CPU work (GPU no longer waits for CPU)
        use_attention_mask = getattr(self.args, 'use_attention_mask_for_valid', True)
        logger.info(f"Using indices sampling collate function (parallel indices sampling)")
        logger.info(f"  use_attention_mask_for_valid: {use_attention_mask}")
        train_collate_fn = create_indices_sampling_collate_fn(
            augmenter=self.augmenter,
            pad_token_id=self.tokenizer.pad_token_id,
            use_attention_mask_for_valid=use_attention_mask
        )

        self.train_loader = get_dataloader(
            config=dataset_config,
            split="train",
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            streaming=False,
            num_samples=self.args.num_train_samples,
            collate_fn=train_collate_fn,
            seed=getattr(self.args, 'seed', 42)  # For deterministic checkpoint resume
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
                collate_fn=val_collate_fn
            )
        else:
            self.val_loader = None

    def train_step(self, batch):
        """
        Single training step with conditional augmentation

        Process:
        1. Extract input_ids and pre-sampled indices from batch
        2. Forward pass - model handles augmentation internally
        3. Return scaled loss for gradient accumulation

        Note: Indices are now sampled in parallel DataLoader workers (not here),
              which gives 2-4x speedup by avoiding GPU waiting for CPU.

        Args:
            batch: Batch from dataloader containing:
                - input_ids: (B, L)
                - conditional_idx: List[List[int]] (pre-sampled)
                - evaluation_idx: List[List[int]] (pre-sampled)
                - unseen_idx: List[List[int]] (pre-sampled)

        Returns:
            loss: Training loss (scaled by gradient_accumulation_steps)
        """
        # Extract data from batch (indices already sampled in workers!)
        input_ids = batch["input_ids"].to(self.device)  # (B, L)
        batch_cond_idx = batch["conditional_idx"]  # List[List[int]]
        batch_eval_idx = batch["evaluation_idx"]  # List[List[int]]
        batch_unseen_idx = batch["unseen_idx"]  # List[List[int]]

        # Forward pass - model does augmentation internally
        logits, loss = self.model(
            input_ids=input_ids,
            conditional_idx=batch_cond_idx,
            evaluation_idx=batch_eval_idx,
            unseen_idx=batch_unseen_idx
        )

        # Loss scaling is now handled in base_trainer.py train() method
        return loss

    def evaluate(self):
        """
        5-mode evaluation for conditional modeling

        Evaluation modes:
        1. Autoregressive: Standard left-to-right autoregressive
        2. Boundary Filling: Condition on boundaries, evaluate middle
        3. Training Distribution: Same split as training
        4. Autoregressive on Boundary: Compare autoregressive on boundary filling setup
        5. Autoregressive on Training: Compare autoregressive on training split setup

        Returns:
            dict: Evaluation results containing all 5 modes' losses and perplexities
                  Uses mode3_loss as main criterion for best model selection
        """
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
        logger.info(f"Mode 1 (Autoregressive)   : loss={eval_results['mode1_loss']:.4f}, ppl={eval_results['mode1_ppl']:.2f}")
        logger.info(f"Mode 2 (Boundary Filling) : loss={eval_results['mode2_loss']:.4f}, ppl={eval_results['mode2_ppl']:.2f}")
        logger.info(f"Mode 3 (Training Dist)    : loss={eval_results['mode3_loss']:.4f}, ppl={eval_results['mode3_ppl']:.2f}")
        logger.info(f"Mode 4 (Auto on Boundary) : loss={eval_results['mode4_loss']:.4f}, ppl={eval_results['mode4_ppl']:.2f}")
        logger.info(f"Mode 5 (Auto on Training) : loss={eval_results['mode5_loss']:.4f}, ppl={eval_results['mode5_ppl']:.2f}")
        logger.info(f"-" * 80)
        logger.info(f"Comparisons:")
        logger.info(f"  Mode 2 vs 4 (Boundary):  Δ={eval_results['mode2_loss'] - eval_results['mode4_loss']:.4f} (negative = conditional better)")
        logger.info(f"  Mode 3 vs 5 (Training):  Δ={eval_results['mode3_loss'] - eval_results['mode5_loss']:.4f} (negative = conditional better)")
        logger.info(f"=" * 80)

        # Use Mode 3 loss as main criterion
        eval_results["loss"] = eval_results["mode3_loss"]

        return eval_results

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
        """
        Load pretrained weights from HuggingFace transformers

        Args:
            model_name: HuggingFace model name (e.g., 'gpt2', 'distilgpt2')
        """
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
        """
        Load from local checkpoint file

        Args:
            checkpoint_path: Path to local checkpoint
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # Optionally restore training state
        if hasattr(self.args, 'resume_training') and self.args.resume_training:
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
