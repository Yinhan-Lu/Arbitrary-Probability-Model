"""
DistilBERT Baseline Trainer (Masked Language Modeling)

Trains a DistilBERT-style encoder-only Transformer from scratch
using the same pipeline infrastructure as the autoregressive baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from train.bert_evaluation_modes import evaluate_bert_all_modes


import torch
import logging
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from transformers import GPT2TokenizerFast  

from train.base_trainer import BaseTrainer
from model.distilbert import DistilBertConfig, DistilBertForMaskedLM
from train.mlm_collator import MLMDataCollator
from train.dataset import get_dataloader 

logger = logging.getLogger(__name__)


class DistilBertTrainer(BaseTrainer):
    """Trainer for the DistilBERT-style masked language model."""

    def setup_model(self):
        logger.info("Setting up DistilBERT model...")
        #Load GPT-2 tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # Add PAD token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Add a MASK token if missing (needed for MLM)
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        # Build config 

        vocab_size = len(self.tokenizer)
        logger.info(f"Tokenizer vocab size: {vocab_size}")
        self.config = DistilBertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=1024,
        )
        self.model = DistilBertForMaskedLM(self.config).to(self.device)


        #to make it run faster
        self.use_amp = getattr(self.args, "fp16", False) and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Using mixed precision (FP16)")
        else:
            logger.info("Using FP32")

        logger.info(
            f"Model params (rough): "
            f"{self.model.config.dim * self.model.config.n_layers * self.model.config.n_heads:,}"
        )

    def setup_data(self):
        logger.info("Loading dataset for MLM training...")

        # Reuse dataset loader, but do masking in the collator
        collator = MLMDataCollator(self.tokenizer, mlm_probability=0.15)

        self.train_loader = get_dataloader(
            config=self.config,
            split="train",
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            streaming=getattr(self.args, "streaming", False),
            num_samples=self.args.num_train_samples,
            collate_fn=collator,
        )

        if self.args.do_eval:
            self.val_loader = get_dataloader(
                config=self.config,
                split="validation",
                batch_size=self.args.eval_batch_size,
                num_workers=self.args.num_workers,
                streaming=False,
                num_samples=self.args.num_eval_samples,
                collate_fn=collator,
            )
        else:
            self.val_loader = None

        # Import blockwise sampling functions (matching conditional/sigmagpt trainers)
        from functools import partial
        from train.blockwise_sampling import (
            uniform_num_conditioning_distribution,
            uniform_num_blocks_distribution,
            uniform_block_sizes_distribution,
            uniform_num_evaluation_distribution,
            generate_conditioning_evaluation_sets_blockwise
        )

        # Create distribution functions (matching conditional trainer pattern)
        logger.info("Creating augmenter with distribution-based sampling")
        logger.info(f"  Conditioning percentage range: [{self.args.cond_pct_min}, {self.args.cond_pct_max}]")
        logger.info(f"  Evaluation percentage range: [{self.args.cond_pct_min}, {self.args.cond_pct_max}]")
        
        num_cond_dist = partial(
            uniform_num_conditioning_distribution,
            conditioning_percentage_range=(self.args.cond_pct_min, self.args.cond_pct_max)
        )
        num_eval_dist = partial(
            uniform_num_evaluation_distribution,
            evaluation_percentage_range=(self.args.cond_pct_min, self.args.cond_pct_max)
        )
        
        # Create block distribution functions with limits
        num_cond_blocks_dist = partial(
            uniform_num_blocks_distribution,
            max_blocks=self.args.max_cond_blocks
        )

        class SimpleAugmenter:
            """Simple augmenter wrapper for BERT evaluation modes"""
            def __init__(self, num_cond_dist, num_eval_dist, num_cond_blocks_dist):
                self.num_cond_dist = num_cond_dist
                self.num_eval_dist = num_eval_dist
                self.num_cond_blocks_dist = num_cond_blocks_dist

            def split_indices(self, seq_len, valid_positions=None):
                return generate_conditioning_evaluation_sets_blockwise(
                    seq_len=seq_len,
                    num_conditioning_distribution=self.num_cond_dist,
                    num_blocks_distribution=self.num_cond_blocks_dist,
                    block_sizes_distribution=uniform_block_sizes_distribution,
                    num_evaluation_distribution=self.num_eval_dist,
                    num_eval_blocks_distribution=uniform_num_blocks_distribution,
                    eval_block_sizes_distribution=uniform_block_sizes_distribution,
                    valid_positions=valid_positions,
                )

        self.augmenter = SimpleAugmenter(
            num_cond_dist=num_cond_dist,
            num_eval_dist=num_eval_dist,
            num_cond_blocks_dist=num_cond_blocks_dist
        )



    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.use_amp:
            with autocast("cuda"):
                _, loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
        else:
            _, loss = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # Loss scaling is now handled in base_trainer.py train() method
        return loss

    @torch.no_grad()
    def evaluate(self):
        """
        Run BERT-style multi-mode evaluation (Modes 1, 2, 3).

        We still expose `loss` and `perplexity` (using Mode 1)
        so BaseTrainer and any generic code keep working.
        """

        logger.info("Running BERT multi-mode evaluation (Modes 1, 2, 3)...")

        metrics = evaluate_bert_all_modes(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            tokenizer=self.tokenizer,
            augmenter=self.augmenter,         
            max_batches=self.args.max_eval_batches,
            trainer_args=self.args,
        )

        # Use Mode 3 loss as main criterion (matching conditional/sigmagpt for fair comparison)
        # Mode 3 represents the training distribution, which is the target for early stopping
        metrics["loss"] = metrics["mode3_loss"]
        metrics["perplexity"] = metrics["mode3_ppl"]

        logger.info(f"Mode 1 (MLM baseline):        loss={metrics['mode1_loss']:.4f}, ppl={metrics['mode1_ppl']:.2f}")
        logger.info(f"Mode 2 (Boundary iter):       loss={metrics['mode2_loss']:.4f}, ppl={metrics['mode2_ppl']:.2f}")
        logger.info(f"Mode 3 (Train dist iter):     loss={metrics['mode3_loss']:.4f}, ppl={metrics['mode3_ppl']:.2f}")
        logger.info(f"Mode 4 (Boundary parallel):   loss={metrics['mode4_loss']:.4f}, ppl={metrics['mode4_ppl']:.2f}")
        logger.info(f"Mode 5 (Train dist parallel): loss={metrics['mode5_loss']:.4f}, ppl={metrics['mode5_ppl']:.2f}")

        return metrics


    def get_csv_header(self):
        """
        Get CSV header for logging
        
        Format matches conditional/sigmagpt models for fair comparison:
        - step, epoch
        - train_loss, train_perplexity
        - mode1_loss, mode1_ppl, ..., mode5_loss, mode5_ppl
        - learning_rate
        """
        return [
            "step",
            "epoch",
            "train_loss",
            "train_perplexity",
            "mode1_loss",
            "mode1_ppl",
            "mode2_loss",
            "mode2_ppl",
            "mode3_loss",
            "mode3_ppl",
            "mode4_loss",
            "mode4_ppl",
            "mode5_loss",
            "mode5_ppl",
            "learning_rate"
        ]

    def format_train_metrics(self, avg_loss, perplexity, lr):
        """
        Format training metrics for CSV logging
        
        During training step, eval metrics are empty.
        Matches conditional model's 5-mode format for fair comparison.
        
        Args:
            avg_loss: Average training loss
            perplexity: Training perplexity
            lr: Current learning rate
            
        Returns:
            dict: Metrics dictionary with empty eval columns (filled during evaluation)
        """
        return {
            'train_loss': avg_loss,
            'train_perplexity': perplexity,
            'mode1_loss': '',
            'mode1_ppl': '',
            'mode2_loss': '',
            'mode2_ppl': '',
            'mode3_loss': '',
            'mode3_ppl': '',
            'mode4_loss': '',
            'mode4_ppl': '',
            'mode5_loss': '',
            'mode5_ppl': '',
            'learning_rate': lr
        }

    def format_eval_metrics(self, eval_results):
        """
        Format evaluation metrics for CSV logging
        
        During evaluation step, train metrics are empty.
        Matches conditional model's 5-mode format for fair comparison.
        
        Args:
            eval_results: Results from evaluate() containing all 5 modes
            
        Returns:
            dict: Metrics dictionary with all 5 modes and learning rate
        """
        return {
            'train_loss': '',
            'train_perplexity': '',
            'mode1_loss': eval_results['mode1_loss'],
            'mode1_ppl': eval_results['mode1_ppl'],
            'mode2_loss': eval_results['mode2_loss'],
            'mode2_ppl': eval_results['mode2_ppl'],
            'mode3_loss': eval_results['mode3_loss'],
            'mode3_ppl': eval_results['mode3_ppl'],
            'mode4_loss': eval_results['mode4_loss'],
            'mode4_ppl': eval_results['mode4_ppl'],
            'mode5_loss': eval_results['mode5_loss'],
            'mode5_ppl': eval_results['mode5_ppl'],
            'learning_rate': self.optimizer.param_groups[0]["lr"]
        }

