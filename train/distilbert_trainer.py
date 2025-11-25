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

        from train.blockwise_sampling import (
            uniform_num_conditioning_distribution,
            uniform_num_blocks_distribution,
            uniform_block_sizes_distribution,
            uniform_num_evaluation_distribution,
            generate_conditioning_evaluation_sets_blockwise
        )

        class SimpleAugmenter:
            def __init__(self, cond_pct_min, cond_pct_max):
                self.cond_pct_min = cond_pct_min
                self.cond_pct_max = cond_pct_max

            def split_indices(self, seq_len, valid_positions=None):
                return generate_conditioning_evaluation_sets_blockwise(
                    seq_len=seq_len,
                    num_conditioning_distribution=lambda l: uniform_num_conditioning_distribution(
                        l, (self.cond_pct_min, self.cond_pct_max)
                    ),
                    num_blocks_distribution=uniform_num_blocks_distribution,
                    block_sizes_distribution=uniform_block_sizes_distribution,
                    num_evaluation_distribution=lambda l: uniform_num_evaluation_distribution(
                        l, (self.cond_pct_min, self.cond_pct_max)
                    ),
                    num_eval_blocks_distribution=uniform_num_blocks_distribution,
                    eval_block_sizes_distribution=uniform_block_sizes_distribution,
                    valid_positions=valid_positions,
                )

        self.augmenter = SimpleAugmenter(
            cond_pct_min=self.args.cond_pct_min,
            cond_pct_max=self.args.cond_pct_max,
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
                loss = loss / self.args.gradient_accumulation_steps
        else:
            _, loss = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = loss / self.args.gradient_accumulation_steps

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

        # Mode 1 is the “main” loss (for BaseTrainer)
        metrics["loss"] = metrics["mode1_loss"]
        metrics["perplexity"] = metrics["mode1_ppl"]

        logger.info(f"Mode 1 (MLM baseline): loss={metrics['mode1_loss']:.4f}, ppl={metrics['mode1_ppl']:.2f}")
        logger.info(f"Mode 2 (Boundary):      loss={metrics['mode2_loss']:.4f}, ppl={metrics['mode2_ppl']:.2f}")
        logger.info(f"Mode 3 (Train dist):    loss={metrics['mode3_loss']:.4f}, ppl={metrics['mode3_ppl']:.2f}")

        return metrics


    def get_csv_header(self):
        return [
            "step", "epoch", "split",
            # generic
            "loss", "perplexity",
            "learning_rate", "grad_norm",
            "tokens_per_second", "time_elapsed_seconds",
            # BERT-mode-specific metrics
            "mode1_loss", "mode1_ppl",
            "mode2_loss", "mode2_ppl",
            "mode3_loss", "mode3_ppl",
        ]


    def format_train_metrics(self, avg_loss, perplexity, lr, **extra):
        """
        Called by BaseTrainer as: format_train_metrics(avg_loss, perplexity, lr)

        avg_loss: *real* MLM loss (already averaged over logging window)
        perplexity: exp(avg_loss)
        lr: learning rate
        """
        row = {
            "split": "train",
            "loss": float(avg_loss),
            "perplexity": float(perplexity),
            "learning_rate": float(lr),
            "grad_norm": extra.get("grad_norm"),
            "tokens_per_second": extra.get("tokens_per_second"),
            "time_elapsed_seconds": extra.get("time_elapsed_seconds"),
        }
        row.update(extra)
        return row

    def format_eval_metrics(self, eval_results, **extra):
        """
        Called by BaseTrainer as: format_eval_metrics(eval_results)

        eval_results is whatever `evaluate()` returns, including:
        - loss, perplexity
        - mode1_loss, mode1_ppl
        - mode2_loss, mode2_ppl
        - mode3_loss, mode3_ppl
        """
        loss = eval_results.get("loss")
        perplexity = eval_results.get("perplexity")

        row = {
            "split": "val",
            "loss": float(loss) if loss is not None else None,
            "perplexity": float(perplexity) if perplexity is not None else None,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
                              if hasattr(self, "optimizer") else None,
            "grad_norm": None,
            "tokens_per_second": None,
            "time_elapsed_seconds": None,

            # mode-specific (important for plotting!)
            "mode1_loss": eval_results.get("mode1_loss"),
            "mode1_ppl": eval_results.get("mode1_ppl"),
            "mode2_loss": eval_results.get("mode2_loss"),
            "mode2_ppl": eval_results.get("mode2_ppl"),
            "mode3_loss": eval_results.get("mode3_loss"),
            "mode3_ppl": eval_results.get("mode3_ppl"),
        }

        row.update(extra)
        return row

