"""
Evaluate trained DistilBERT model with BERT-specific evaluation modes

Usage:
    python evaluate_bert.py \
        --checkpoint experiments/distilbert_full_training_20251120_171222/checkpoints/checkpoint_step_XXXX.pt \
        --num_eval_samples 1000 \
        --eval_batch_size 8 \
        --max_eval_batches 50
"""

import sys
import argparse
import logging
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from model.distilbert import DistilBertConfig, DistilBertForMaskedLM
from train.dataset import get_dataloader
from train.mlm_collator import MLMDataCollator
from train.bert_evaluation_modes import evaluate_bert_all_modes
from transformers import GPT2TokenizerFast

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    config = checkpoint.get('config')
    if config is None:
        # If config not saved, recreate it
        logger.warning("Config not found in checkpoint, using default config")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        config = DistilBertConfig(
            vocab_size=len(tokenizer),
            max_position_embeddings=1024,
        )
    elif isinstance(config, dict):
        # Config is a dict, convert to DistilBertConfig
        valid_keys = {'vocab_size', 'dim', 'n_layers', 'n_heads', 'hidden_dim', 
                      'max_position_embeddings', 'dropout', 'attention_dropout', 'layer_norm_eps'}
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        config = DistilBertConfig(**filtered_config)
    
    model = DistilBertForMaskedLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    logger.info(f"Model has {model.get_num_params():,} parameters")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate DistilBERT with BERT-specific modes")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset_name', type=str, default='wikitext',
                       help='Dataset name')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-v1',
                       help='Dataset config')
    parser.add_argument('--num_eval_samples', type=int, default=1000,
                       help='Number of evaluation samples')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                       help='Evaluation batch size')
    parser.add_argument('--max_eval_batches', type=int, default=50,
                       help='Maximum number of batches to evaluate (to save time)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda/mps)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Mode 2 boundary distribution parameters
    parser.add_argument('--mode2_boundary_cond_pct_min', type=float, default=0.1,
                       help='Mode 2: Minimum boundary conditioning percentage')
    parser.add_argument('--mode2_boundary_cond_pct_max', type=float, default=0.3,
                       help='Mode 2: Maximum boundary conditioning percentage')
    
    # Mode 3 training distribution parameters
    parser.add_argument('--cond_pct_min', type=float, default=0.2,
                       help='Mode 3: Minimum conditioning percentage (for augmenter)')
    parser.add_argument('--cond_pct_max', type=float, default=0.4,
                       help='Mode 3: Maximum conditioning percentage (for augmenter)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    model, config = load_checkpoint(args.checkpoint, device)
    
    # Setup tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Setup data collator
    collator = MLMDataCollator(tokenizer, mlm_probability=0.15)
    
    # Setup dataloader
    logger.info("Loading validation dataset...")
    val_loader = get_dataloader(
        config=config,
        split="validation",
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        streaming=False,
        num_samples=args.num_eval_samples,
        collate_fn=collator,
    )
    
    # Setup augmenter for Mode 3  
    # Create distribution functions matching your training setup
    from functools import partial
    from train.blockwise_sampling import (
        uniform_num_conditioning_distribution,
        uniform_num_blocks_distribution,
        uniform_block_sizes_distribution,
        uniform_num_evaluation_distribution,
    )
    
    # Create a simple augmenter that just provides split_indices method
    class SimpleAugmenter:
        def __init__(self, cond_pct_min, cond_pct_max):
            self.cond_pct_min = cond_pct_min
            self.cond_pct_max = cond_pct_max
            
        def split_indices(self, seq_len, valid_positions=None):
            from train.blockwise_sampling import generate_conditioning_evaluation_sets_blockwise
            
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
    
    augmenter = SimpleAugmenter(
        cond_pct_min=args.cond_pct_min,
        cond_pct_max=args.cond_pct_max,
    )
    
    # Run evaluation
    logger.info("=" * 80)
    logger.info("Starting BERT 5-mode evaluation")
    logger.info("=" * 80)
    
    metrics = evaluate_bert_all_modes(
        model=model,
        dataloader=val_loader,
        device=device,
        tokenizer=tokenizer,
        augmenter=augmenter,
        max_batches=args.max_eval_batches,
        trainer_args=args
    )
    
    # Print results
    logger.info("=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"Number of batches evaluated: {metrics['num_batches']}")
    logger.info("")
    logger.info("Mode 1 (MLM Baseline - Parallel Masking):")
    logger.info(f"  Loss: {metrics['mode1_loss']:.4f}")
    logger.info(f"  Perplexity: {metrics['mode1_ppl']:.2f}")
    logger.info(f"  Tokens: {metrics['mode1_tokens']}")
    logger.info("")
    logger.info("Mode 2 (Boundary Filling - Iterative Unmasking):")
    logger.info(f"  Loss: {metrics['mode2_loss']:.4f}")
    logger.info(f"  Perplexity: {metrics['mode2_ppl']:.2f}")
    logger.info(f"  Tokens: {metrics['mode2_tokens']}")
    logger.info("")
    logger.info("Mode 3 (Training Distribution - Iterative Unmasking):")
    logger.info(f"  Loss: {metrics['mode3_loss']:.4f}")
    logger.info(f"  Perplexity: {metrics['mode3_ppl']:.2f}")
    logger.info(f"  Tokens: {metrics['mode3_tokens']}")
    logger.info("")
    logger.info("Mode 4 (Boundary Filling - Parallel Prediction):")
    logger.info(f"  Loss: {metrics['mode4_loss']:.4f}")
    logger.info(f"  Perplexity: {metrics['mode4_ppl']:.2f}")
    logger.info(f"  Tokens: {metrics['mode4_tokens']}")
    logger.info("")
    logger.info("Mode 5 (Training Distribution - Parallel Prediction):")
    logger.info(f"  Loss: {metrics['mode5_loss']:.4f}")
    logger.info(f"  Perplexity: {metrics['mode5_ppl']:.2f}")
    logger.info(f"  Tokens: {metrics['mode5_tokens']}")
    logger.info("=" * 80)
    
    # Save results
    checkpoint_path = Path(args.checkpoint)
    results_dir = checkpoint_path.parent.parent / "evaluation_results"
    results_dir.mkdir(exist_ok=True)
    
    # Save text summary
    results_file = results_dir / f"bert_eval_modes_{checkpoint_path.stem}.txt"
    with open(results_file, 'w') as f:
        f.write("BERT 5-Mode Evaluation Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Batches evaluated: {metrics['num_batches']}\n\n")
        f.write(f"Mode 1 (MLM Baseline): Loss={metrics['mode1_loss']:.4f}, PPL={metrics['mode1_ppl']:.2f}, Tokens={metrics['mode1_tokens']}\n")
        f.write(f"Mode 2 (Boundary Iterative): Loss={metrics['mode2_loss']:.4f}, PPL={metrics['mode2_ppl']:.2f}, Tokens={metrics['mode2_tokens']}\n")
        f.write(f"Mode 3 (Training Iterative): Loss={metrics['mode3_loss']:.4f}, PPL={metrics['mode3_ppl']:.2f}, Tokens={metrics['mode3_tokens']}\n")
        f.write(f"Mode 4 (Boundary Parallel): Loss={metrics['mode4_loss']:.4f}, PPL={metrics['mode4_ppl']:.2f}, Tokens={metrics['mode4_tokens']}\n")
        f.write(f"Mode 5 (Training Parallel): Loss={metrics['mode5_loss']:.4f}, PPL={metrics['mode5_ppl']:.2f}, Tokens={metrics['mode5_tokens']}\n")
    
    logger.info(f"Results saved to: {results_file}")
    
    # Save CSV for plotting (compatible with your training metrics format)
    import csv
    csv_file = results_dir / f"bert_eval_modes_{checkpoint_path.stem}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Use same header as training metrics
        writer.writerow(['step', 'epoch', 'split', 'loss', 'perplexity', 'learning_rate', 'grad_norm', 'tokens_per_second', 'time_elapsed_seconds'])
        # Mode 1-5 - use step=1 for mode1, step=2 for mode2, etc.
        writer.writerow([1, 1, 'eval_mode1', metrics['mode1_loss'], metrics['mode1_ppl'], None, None, None, None])
        writer.writerow([2, 1, 'eval_mode2', metrics['mode2_loss'], metrics['mode2_ppl'], None, None, None, None])
        writer.writerow([3, 1, 'eval_mode3', metrics['mode3_loss'], metrics['mode3_ppl'], None, None, None, None])
        writer.writerow([4, 1, 'eval_mode4', metrics['mode4_loss'], metrics['mode4_ppl'], None, None, None, None])
        writer.writerow([5, 1, 'eval_mode5', metrics['mode5_loss'], metrics['mode5_ppl'], None, None, None, None])
    
    logger.info(f"CSV saved to: {csv_file}")
    logger.info("")
    logger.info("To plot results:")
    logger.info(f"  python plot_loss.py  # (update exp_dir to evaluation_results folder)")
    logger.info(f"  Or: df = pd.read_csv('{csv_file}')")


if __name__ == '__main__':
    main()
