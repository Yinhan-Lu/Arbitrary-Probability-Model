"""
Evaluate DistilBERT model with BERT-specific evaluation (Mode 1 only - quick test)

This runs only Mode 1 (standard MLM evaluation) as a quick test.
For full 3-mode evaluation, use evaluate_bert.py
"""

import sys
from pathlib import Path
import torch
import logging

sys.path.insert(0, str(Path(__file__).parent))

from model.distilbert import DistilBertForMaskedLM
from train.dataset import get_dataloader
from train.mlm_collator import MLMDataCollator
from train.bert_evaluation_modes import evaluate_bert_mode1_joint_probability
from transformers import GPT2TokenizerFast

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def main():
    # Configuration
    checkpoint_path = "experiments/distilbert_full_training_20251120_171222/checkpoints/best_model.pt"
    num_eval_samples = 1000
    eval_batch_size = 8
    max_eval_batches = 50  # Limit for faster evaluation
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle config - might be dict or DistilBertConfig object
    config = checkpoint['config']
    if isinstance(config, dict):
        from model.distilbert import DistilBertConfig
        # Filter out keys that don't match DistilBertConfig constructor
        valid_keys = {'vocab_size', 'dim', 'n_layers', 'n_heads', 'hidden_dim', 
                      'max_position_embeddings', 'dropout', 'attention_dropout', 'layer_norm_eps'}
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        config = DistilBertConfig(**filtered_config)
    
    model = DistilBertForMaskedLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from step {checkpoint.get('step', 'unknown')}")
    logger.info(f"Model parameters: {model.get_num_params():,}")
    
    # Setup tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Setup dataloader
    logger.info("Loading validation dataset...")
    collator = MLMDataCollator(tokenizer, mlm_probability=0.15)
    val_loader = get_dataloader(
        config=config,
        split="validation",
        batch_size=eval_batch_size,
        num_workers=4,
        streaming=False,
        num_samples=num_eval_samples,
        collate_fn=collator,
    )
    
    logger.info("=" * 80)
    logger.info("Running BERT Mode 1 Evaluation (MLM Baseline)")
    logger.info("=" * 80)
    logger.info(f"Evaluating {max_eval_batches} batches...")
    
    # Run evaluation
    metrics = evaluate_bert_mode1_joint_probability(
        model=model,
        dataloader=val_loader,
        device=device,
        tokenizer=tokenizer,
        max_batches=max_eval_batches
    )
    
    # Print results
    logger.info("=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"Mode 1 (MLM Baseline - Parallel Masking):")
    logger.info(f"  Loss: {metrics['loss']:.4f}")
    logger.info(f"  Perplexity: {metrics['perplexity']:.2f}")
    logger.info(f"  Tokens evaluated: {metrics['total_tokens']}")
    logger.info(f"  Batches evaluated: {metrics['num_batches']}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("✓ Mode 1 evaluation completed successfully!")
    logger.info("")
    logger.info("This evaluates the standard MLM objective (15% masked tokens).")
    logger.info("For Modes 2 and 3 (iterative unmasking), use evaluate_bert.py")

if __name__ == '__main__':
    main()
