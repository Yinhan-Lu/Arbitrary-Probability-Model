"""
Debug script for Mode 2 vs Mode 4 loss comparison

This script analyzes why Mode 2 loss > Mode 4 loss when Mode 2 theoretically
has MORE information (sees end block in addition to what Mode 4 sees).

Run from project root: python tests/debug_mode2_vs_mode4.py [checkpoint_path]
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.arbitrary_prob_gpt2 import GPT2Model, GPT2Config
from train.blockwise_sampling import generate_boundary_conditioning_split
from train.mask_utils import create_prefix_conditional_mask


def create_test_model(device='cpu'):
    """Create a small test model (random init)"""
    config = GPT2Config(
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=128,
        max_seq_len=256,
    )
    # mask_token_id=50257, bos_token_id=50258 (following project convention)
    model = GPT2Model(config, mask_token_id=50257, bos_token_id=50258)
    model.to(device)
    model.eval()
    return model


def load_trained_model(checkpoint_path, device='cpu'):
    """Load a trained model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = GPT2Config(**config_dict)
    else:
        # Default DistilGPT2 config
        config = GPT2Config(
            vocab_size=50258,  # 50257 + mask
            n_layer=6,
            n_head=12,
            n_embd=768,
            max_seq_len=1024,
        )

    # Determine bos_token_id based on vocab_size
    # vocab_size=50258: only mask token added, use mask (50257) as both mask and bos
    # vocab_size=50259: both mask and bos added
    if config.vocab_size == 50258:
        bos_token_id = 50257  # Model uses mask token as BOS separator
    else:
        bos_token_id = 50258

    model = GPT2Model(config, mask_token_id=50257, bos_token_id=bos_token_id)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"Model loaded successfully: {config.n_layer} layers, {config.n_embd} dim, vocab_size={config.vocab_size}")
    return model


def compute_loss_at_position(logits, labels, position):
    """Compute cross-entropy loss at a specific position"""
    # For next-token prediction: logits[pos-1] predicts labels[pos]
    if position == 0:
        return None  # Cannot predict position 0

    pred_logit = logits[position - 1]  # Shape: (vocab_size,)
    target = labels[position]

    loss = F.cross_entropy(pred_logit.unsqueeze(0).float(), target.unsqueeze(0))
    return loss.item()


def debug_single_sample(model, input_ids, device='cpu'):
    """
    Compare Mode 2 vs Mode 4 on a single sample

    Args:
        model: The model to evaluate
        input_ids: Single sequence tensor (seq_len,)
        device: Device to run on
    """
    # Move input_ids to device
    input_ids = input_ids.to(device)

    seq_len = input_ids.size(0)
    print(f"\n{'='*80}")
    print(f"Debugging Mode 2 vs Mode 4")
    print(f"Sequence length: {seq_len}")
    print(f"{'='*80}")

    # Generate boundary split
    cond_idx, eval_idx, unknown_idx = generate_boundary_conditioning_split(
        seq_len,
        start_block_range=(seq_len // 10, seq_len // 5),
        end_block_range=(seq_len // 10, seq_len // 5),
    )

    print(f"\nConditioning indices ({len(cond_idx)}): {cond_idx[:10]}{'...' if len(cond_idx) > 10 else ''}")
    print(f"Evaluation indices ({len(eval_idx)}): {eval_idx[:10]}{'...' if len(eval_idx) > 10 else ''}")
    print(f"Unknown indices ({len(unknown_idx)}): {unknown_idx[:10]}{'...' if len(unknown_idx) > 10 else ''}")

    # Identify start and end blocks from conditioning
    start_block = sorted([i for i in cond_idx if i < min(eval_idx)])
    end_block = sorted([i for i in cond_idx if i > max(eval_idx)])

    print(f"\nStart block: {start_block}")
    print(f"End block: {end_block}")

    # ==================== Mode 1 (Standard Causal) ====================
    print(f"\n{'='*40}")
    print("Mode 1: Standard Causal Attention")
    print(f"{'='*40}")

    input_ids_batch = input_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        logits_mode1, _ = model(input_ids_batch)
    logits_mode1 = logits_mode1[0]  # Remove batch dim

    print(f"Mode 1 logits shape: {logits_mode1.shape}")

    # ==================== Mode 2 (Prefix Conditional) ====================
    print(f"\n{'='*40}")
    print("Mode 2: Prefix Conditional Attention")
    print(f"{'='*40}")

    with torch.no_grad():
        logits_mode2, loss_mode2 = model(
            input_ids_batch,
            conditional_idx=[[c for c in cond_idx]],
            evaluation_idx=[[e for e in eval_idx]],
            unseen_idx=[[u for u in unknown_idx]]
        )
    logits_mode2 = logits_mode2[0]  # Remove batch dim

    print(f"Mode 2 logits shape: {logits_mode2.shape}")
    print(f"Mode 2 internal loss: {loss_mode2.item():.4f}")

    # ==================== Per-Position Loss Comparison ====================
    print(f"\n{'='*40}")
    print("Per-Position Loss Comparison")
    print(f"{'='*40}")

    # Mode 4 uses Mode 1 logits on eval positions
    # Mode 2 uses its own logits on eval positions

    mode1_losses = []
    mode2_losses = []

    # For Mode 2, we need to map original positions to augmented positions
    # Augmented sequence: [cond_tokens] [BOS] [body]
    # Position t in body maps to: N_cond + 1 + t in augmented
    N_cond = len(cond_idx)

    print(f"\nN_cond = {N_cond}")
    print(f"Mode 2 augmented seq structure: [cond({N_cond})] [BOS(1)] [body({seq_len})]")
    print(f"Total augmented length expected: {N_cond + 1 + seq_len}")
    print(f"Actual Mode 2 logits length: {logits_mode2.shape[0]}")

    print(f"\n{'Pos':<6} {'M1 Loss':<12} {'M2 Loss':<12} {'Diff':<12} {'Status'}")
    print("-" * 60)

    sample_positions = eval_idx[::max(1, len(eval_idx)//20)]  # Sample ~20 positions

    for pos in sample_positions:
        # Mode 1/4: standard position mapping
        # logits_mode1[pos-1] predicts token at pos
        loss_m1 = compute_loss_at_position(logits_mode1, input_ids, pos)

        # Mode 2: position in augmented sequence
        # Original position pos in body is at augmented position N_cond + 1 + pos
        aug_pos = N_cond + 1 + pos
        if aug_pos > 0 and aug_pos < logits_mode2.shape[0]:
            # logits_mode2[aug_pos - 1] predicts token at aug_pos
            # But we need to predict the ORIGINAL token at position pos
            pred_logit = logits_mode2[aug_pos - 1]
            target = input_ids[pos]
            loss_m2 = F.cross_entropy(pred_logit.unsqueeze(0).float(), target.unsqueeze(0)).item()
        else:
            loss_m2 = None

        if loss_m1 is not None and loss_m2 is not None:
            mode1_losses.append(loss_m1)
            mode2_losses.append(loss_m2)
            diff = loss_m2 - loss_m1
            status = "M2 better" if diff < -0.01 else ("M1 better" if diff > 0.01 else "~Same")
            print(f"{pos:<6} {loss_m1:<12.4f} {loss_m2:<12.4f} {diff:<12.4f} {status}")

    # ==================== Summary ====================
    print(f"\n{'='*40}")
    print("Summary Statistics")
    print(f"{'='*40}")

    if mode1_losses and mode2_losses:
        avg_m1 = sum(mode1_losses) / len(mode1_losses)
        avg_m2 = sum(mode2_losses) / len(mode2_losses)

        print(f"\nAverage Mode 1/4 loss: {avg_m1:.4f}")
        print(f"Average Mode 2 loss: {avg_m2:.4f}")
        print(f"Difference (M2 - M1): {avg_m2 - avg_m1:.4f}")

        m2_better_count = sum(1 for m1, m2 in zip(mode1_losses, mode2_losses) if m2 < m1 - 0.01)
        m1_better_count = sum(1 for m1, m2 in zip(mode1_losses, mode2_losses) if m1 < m2 - 0.01)
        same_count = len(mode1_losses) - m2_better_count - m1_better_count

        print(f"\nMode 2 better: {m2_better_count} positions")
        print(f"Mode 1 better: {m1_better_count} positions")
        print(f"~Same: {same_count} positions")

        if avg_m2 > avg_m1:
            print(f"\n⚠️ UNEXPECTED: Mode 2 loss > Mode 1 loss!")
            print("Mode 2 should have MORE information (sees end block)")
            print("\nPossible causes:")
            print("1. Position encoding mismatch in prefix conditioning")
            print("2. Model not trained to use prefix conditional attention")
            print("3. Bug in loss computation")
        else:
            print(f"\n✓ Expected: Mode 2 loss ≤ Mode 1 loss")

    # ==================== Position Encoding Analysis ====================
    print(f"\n{'='*40}")
    print("Position Encoding Analysis")
    print(f"{'='*40}")

    # Analyze position IDs in Mode 2's augmented sequence
    # Based on model code: cond_position_ids = cond_idx_t + 1
    #                     body_position_ids = [1, 2, ..., seq_len]
    cond_position_ids = [pos + 1 for pos in sorted(cond_idx)]
    body_position_ids = list(range(1, seq_len + 1))  # BOS at 0, body at 1..seq_len
    bos_position = [0]

    full_position_ids = cond_position_ids + bos_position + body_position_ids

    print(f"\nMode 2 position IDs in augmented sequence:")
    print(f"  Conditioning positions (first 5): {cond_position_ids[:5]}...")
    print(f"  Conditioning positions (last 5): ...{cond_position_ids[-5:]}")
    print(f"  BOS position: {bos_position}")
    print(f"  Body positions: [1, 2, ..., {seq_len}]")

    # Check for issues
    unique_pos = len(set(full_position_ids))
    total_pos = len(full_position_ids)
    duplicates = total_pos - unique_pos
    print(f"\n⚠️  Position ID analysis:")
    print(f"  - Total positions: {total_pos}")
    print(f"  - Unique positions: {unique_pos}")
    print(f"  - Duplicates: {duplicates}")

    if duplicates > 0:
        from collections import Counter
        pos_counts = Counter(full_position_ids)
        dup_positions = [pos for pos, count in pos_counts.items() if count > 1]
        print(f"  - Duplicated position IDs: {dup_positions[:10]}{'...' if len(dup_positions) > 10 else ''}")

    # Check for non-sequential jumps
    jumps = []
    for i in range(1, len(full_position_ids)):
        diff = full_position_ids[i] - full_position_ids[i-1]
        if abs(diff) > 10:  # Large jump
            jumps.append((i, full_position_ids[i-1], full_position_ids[i], diff))
    if jumps:
        print(f"  - Large position jumps: {jumps[:5]}{'...' if len(jumps) > 5 else ''}")

    # ==================== Attention Analysis ====================
    print(f"\n{'='*40}")
    print("Attention Visibility Analysis")
    print(f"{'='*40}")

    # Analyze what each mode can see for a middle position
    mid_pos = eval_idx[len(eval_idx) // 2]
    print(f"\nFor predicting token at position {mid_pos}:")

    print(f"\nMode 1/4 can see:")
    print(f"  - BOS (position 0)")
    print(f"  - Tokens at positions 1 to {mid_pos - 1}")
    print(f"  - Total: {mid_pos} tokens")

    aug_mid_pos = N_cond + 1 + mid_pos
    print(f"\nMode 2 can see (at augmented position {aug_mid_pos}):")
    print(f"  - All conditioning tokens ({N_cond}): positions {start_block[:3]}...{end_block[-3:]}")
    print(f"  - BOS (augmented position {N_cond})")
    print(f"  - Body tokens 0 to {mid_pos - 1} (augmented positions {N_cond + 1} to {aug_mid_pos - 1})")
    print(f"  - Total: {N_cond + 1 + mid_pos} tokens")
    print(f"  - Extra info: END BLOCK ({len(end_block)} tokens)")

    return {
        'mode1_avg_loss': sum(mode1_losses) / len(mode1_losses) if mode1_losses else None,
        'mode2_avg_loss': sum(mode2_losses) / len(mode2_losses) if mode2_losses else None,
        'mode2_internal_loss': loss_mode2.item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Debug Mode 2 vs Mode 4 loss comparison")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to model checkpoint (optional, uses random model if not provided)"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length for test input"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Mode 2 vs Mode 4 Debug Analysis")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load or create model
    if args.checkpoint_path:
        print(f"\nLoading trained model from: {args.checkpoint_path}")
        model = load_trained_model(args.checkpoint_path, device)
    else:
        print("\nCreating random test model (no checkpoint provided)...")
        model = create_test_model(device)

    # Create synthetic input
    print(f"Creating synthetic input (seq_len={args.seq_len})...")
    torch.manual_seed(42)
    input_ids = torch.randint(0, 50257, (args.seq_len,))

    # Run debug
    results = debug_single_sample(model, input_ids, device)

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Mode 1/4 average loss: {results['mode1_avg_loss']:.4f}")
    print(f"Mode 2 average loss: {results['mode2_avg_loss']:.4f}")
    print(f"Mode 2 internal loss: {results['mode2_internal_loss']:.4f}")

    if results['mode2_avg_loss'] > results['mode1_avg_loss']:
        print(f"\n⚠️  Mode 2 loss ({results['mode2_avg_loss']:.4f}) > Mode 4 loss ({results['mode1_avg_loss']:.4f})")
        print("This is unexpected - Mode 2 has MORE information!")
    else:
        print(f"\n✓ Mode 2 loss ({results['mode2_avg_loss']:.4f}) <= Mode 4 loss ({results['mode1_avg_loss']:.4f})")
        print("This is expected - Mode 2 has more information and should perform at least as well.")

    print("\n" + "=" * 80)
    print("✓ Debug analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
