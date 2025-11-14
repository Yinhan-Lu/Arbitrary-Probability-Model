"""
DistilBERT Baseline Trainer (Masked Language Modeling)

Trains a DistilBERT-style encoder-only Transformer from scratch
using the same pipeline infrastructure as the autoregressive baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

        logger.info("Setting up DistilBERT model...")

        #Load GPT-2 tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        #Ensure we have a PAD token (use EOS as PAD, like your dataset)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add a MASK token if missing (needed for MLM)
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        # Build config 
        self.config = DistilBertConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=1024,
        )

        # 5) Create model
        self.model = DistilBertForMaskedLM(self.config).to(self.device)

        # Mixed precision
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

        # Use the SAME config as the model so max_seq_len / positions are consistent
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
        self.model.eval()
        total_loss, num_batches = 0.0, 0
        logger.info("Running MLM evaluation...")

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            _, loss = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += loss.item()
            num_batches += 1

            if self.args.max_eval_batches > 0 and num_batches >= self.args.max_eval_batches:
                break

        avg_loss = total_loss / max(1, num_batches)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        logger.info(f"Validation loss {avg_loss:.4f} | PPL {perplexity:.2f}")
        self.model.train()
        return {"loss": avg_loss, "perplexity": perplexity}
