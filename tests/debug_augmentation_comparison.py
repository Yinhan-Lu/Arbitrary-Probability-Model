"""
Debug script: Compare Legacy vs Main augmentation implementations

This script compares the augmentation results between:
- Legacy: External augmentation (CPU, using .item() and Python lists)
- Main: Internal augmentation (GPU, using pure tensor operations)

Run from project root: python tests/debug_augmentation_comparison.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.config import GPT2Config
from model.arbitrary_prob_gpt2 import GPT2Model

print("=" * 100)
print("AUGMENTATION IMPLEMENTATION COMPARISON: Legacy vs Main")
print("=" * 100)

# Check device availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✓ Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ Using CUDA device")
else:
    device = torch.device("cpu")
    print(f"✓ Using CPU device")

print("=" * 100)

# Test configuration
vocab_size = 50257
mask_token_id = vocab_size
bos_token_id = vocab_size - 1

# Create a deterministic test sample
torch.manual_seed(42)
sample_seq = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], device=device)
seq_len = len(sample_seq)

# Fixed conditioning/evaluation/unseen indices for comparison
cond_idx = [0, 2, 5]  # Condition on tokens at positions 0, 2, 5
eval_idx = [1, 3, 6, 8]  # Evaluate tokens at positions 1, 3, 6, 8
unseen_idx = [1, 3, 4, 6, 7, 8, 9]  # Unknown includes eval + some extra

print(f"\n📋 Test Configuration:")
print(f"  Original sequence: {sample_seq.tolist()}")
print(f"  Sequence length: {seq_len}")
print(f"  Conditioning indices: {cond_idx}")
print(f"  Evaluation indices: {eval_idx}")
print(f"  Unseen indices: {unseen_idx}")
print(f"  Truly unseen (unseen - eval): {sorted(set(unseen_idx) - set(eval_idx))}")

print("\n" + "=" * 100)
print("IMPLEMENTATION 1: LEGACY (External Augmentation)")
print("=" * 100)
print("Using .item() and Python lists (mimics legacy implementation)")

# === LEGACY IMPLEMENTATION (MANUAL) ===
def legacy_augment_sequence(input_ids, cond_idx, eval_idx, unseen_idx, device):
    """
    Mimic legacy augmentation implementation

    Key characteristics:
    - Uses .item() to extract values
    - Uses Python lists
    - Creates new tensor with torch.tensor()
    """
    seq_len = input_ids.size(0)

    # Step 1: Build conditioning tokens (prefix) using .item()
    cond_tokens = []
    cond_position_ids = []
    for idx in sorted(cond_idx):
        cond_tokens.append(input_ids[idx].item())  # ❌ .item()!
        cond_position_ids.append(idx + 1)
    N_cond = len(cond_tokens)

    # Step 2: Build sequence tokens (BOS + Body) using Python lists
    seq_tokens = [bos_token_id]  # ❌ Python list!
    seq_position_ids = [0]

    # Only mask truly unseen tokens
    unseen_set = set(unseen_idx)
    eval_set = set(eval_idx)
    truly_unseen = unseen_set - eval_set

    for i in range(seq_len):
        if i in truly_unseen:
            seq_tokens.append(mask_token_id)
        else:
            seq_tokens.append(input_ids[i].item())  # ❌ .item()!
        seq_position_ids.append(i + 1)
    N_seq = len(seq_tokens)

    # Step 3: Create new tensor from Python lists
    aug_input_ids = torch.tensor(
        cond_tokens + seq_tokens,  # ❌ Python list concatenation!
        dtype=torch.long,
        device=device
    )
    position_ids = torch.tensor(
        cond_position_ids + seq_position_ids,
        dtype=torch.long,
        device=device
    )

    # Step 4: Create labels
    labels = torch.full_like(aug_input_ids, -100)
    for eval_pos in eval_idx:
        new_pos = N_cond + 1 + eval_pos
        if new_pos < len(labels):
            labels[new_pos] = input_ids[eval_pos].item()  # ❌ .item()!

    return aug_input_ids, position_ids, labels, N_cond, N_seq


legacy_aug_ids, legacy_pos_ids, legacy_labels, legacy_N_cond, legacy_N_seq = legacy_augment_sequence(
    sample_seq, cond_idx, eval_idx, unseen_idx, device
)

print(f"\nLegacy Results:")
print(f"  Augmented sequence: {legacy_aug_ids.tolist()}")
print(f"  Length: {len(legacy_aug_ids)}")
print(f"  N_cond: {legacy_N_cond}, N_seq: {legacy_N_seq}")
print(f"  Position IDs: {legacy_pos_ids.tolist()}")
print(f"  Labels: {legacy_labels.tolist()}")
print(f"  Tensor dtype: {legacy_aug_ids.dtype}")
print(f"  Tensor device: {legacy_aug_ids.device}")

# Analyze structure
print(f"\n  Structure breakdown:")
print(f"    Conditioning prefix: {legacy_aug_ids[:legacy_N_cond].tolist()}")
print(f"    BOS token: {legacy_aug_ids[legacy_N_cond].item()}")
print(f"    Body sequence: {legacy_aug_ids[legacy_N_cond+1:].tolist()}")

print("\n" + "=" * 100)
print("IMPLEMENTATION 2: MAIN (Internal Augmentation)")
print("=" * 100)
print("Using pure tensor operations (current main branch)")

# === MAIN IMPLEMENTATION (via model) ===
config = GPT2Config(
    n_layer=2,
    n_head=2,
    n_embd=64,
    vocab_size=vocab_size + 1,
    max_seq_len=128,
    detach_augmentation=False  # Test without detach first
)

model = GPT2Model(
    config,
    mask_token_id=mask_token_id,
    bos_token_id=bos_token_id
).to(device)

# Call internal augmentation method directly
main_aug_ids, main_pos_ids, main_N_cond, main_N_seq = model._build_augmented_sequence_single(
    sample_seq,
    cond_idx,
    eval_idx,
    unseen_idx,
    device=device
)

# Create labels using model's method
main_labels = model._create_labels(
    sample_seq,
    eval_idx,
    main_aug_ids,
    main_N_cond
)

print(f"\nMain Results:")
print(f"  Augmented sequence: {main_aug_ids.tolist()}")
print(f"  Length: {len(main_aug_ids)}")
print(f"  N_cond: {main_N_cond}, N_seq: {main_N_seq}")
print(f"  Position IDs: {main_pos_ids.tolist()}")
print(f"  Labels: {main_labels.tolist()}")
print(f"  Tensor dtype: {main_aug_ids.dtype}")
print(f"  Tensor device: {main_aug_ids.device}")

# Analyze structure
print(f"\n  Structure breakdown:")
print(f"    Conditioning prefix: {main_aug_ids[:main_N_cond].tolist()}")
print(f"    BOS token: {main_aug_ids[main_N_cond].item()}")
print(f"    Body sequence: {main_aug_ids[main_N_cond+1:].tolist()}")

print("\n" + "=" * 100)
print("COMPARISON ANALYSIS")
print("=" * 100)

# Compare results
print("\n1. Augmented Input IDs:")
if torch.equal(legacy_aug_ids, main_aug_ids):
    print("  ✅ IDENTICAL")
else:
    print("  ❌ DIFFERENT!")
    print(f"  Legacy: {legacy_aug_ids.tolist()}")
    print(f"  Main:   {main_aug_ids.tolist()}")

    # Find differences
    if len(legacy_aug_ids) == len(main_aug_ids):
        diff_positions = [i for i in range(len(legacy_aug_ids)) if legacy_aug_ids[i] != main_aug_ids[i]]
        if diff_positions:
            print(f"  Differences at positions: {diff_positions}")
            for pos in diff_positions:
                print(f"    Position {pos}: Legacy={legacy_aug_ids[pos].item()}, Main={main_aug_ids[pos].item()}")
    else:
        print(f"  Length mismatch: Legacy={len(legacy_aug_ids)}, Main={len(main_aug_ids)}")

print("\n2. Position IDs:")
if torch.equal(legacy_pos_ids, main_pos_ids):
    print("  ✅ IDENTICAL")
else:
    print("  ❌ DIFFERENT!")
    print(f"  Legacy: {legacy_pos_ids.tolist()}")
    print(f"  Main:   {main_pos_ids.tolist()}")

print("\n3. Labels:")
if torch.equal(legacy_labels, main_labels):
    print("  ✅ IDENTICAL")
else:
    print("  ❌ DIFFERENT!")
    print(f"  Legacy: {legacy_labels.tolist()}")
    print(f"  Main:   {main_labels.tolist()}")

    # Find differences
    if len(legacy_labels) == len(main_labels):
        diff_positions = [i for i in range(len(legacy_labels)) if legacy_labels[i] != main_labels[i]]
        if diff_positions:
            print(f"  Differences at positions: {diff_positions}")
            for pos in diff_positions:
                print(f"    Position {pos}: Legacy={legacy_labels[pos].item()}, Main={main_labels[pos].item()}")

print("\n4. N_cond and N_seq:")
if legacy_N_cond == main_N_cond and legacy_N_seq == main_N_seq:
    print("  ✅ IDENTICAL")
    print(f"  N_cond={legacy_N_cond}, N_seq={legacy_N_seq}")
else:
    print("  ❌ DIFFERENT!")
    print(f"  Legacy: N_cond={legacy_N_cond}, N_seq={legacy_N_seq}")
    print(f"  Main:   N_cond={main_N_cond}, N_seq={main_N_seq}")

print("\n5. Tensor Properties:")
print(f"  Same dtype: {legacy_aug_ids.dtype == main_aug_ids.dtype} (Legacy: {legacy_aug_ids.dtype}, Main: {main_aug_ids.dtype})")
print(f"  Same device: {legacy_aug_ids.device == main_aug_ids.device}")
print(f"  Legacy is contiguous: {legacy_aug_ids.is_contiguous()}")
print(f"  Main is contiguous: {main_aug_ids.is_contiguous()}")

print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)

if (torch.equal(legacy_aug_ids, main_aug_ids) and
    torch.equal(legacy_pos_ids, main_pos_ids) and
    torch.equal(legacy_labels, main_labels)):
    print("\n✅ SUCCESS: Both implementations produce IDENTICAL results!")
    print("\nThis means the difference in training results is NOT due to")
    print("the augmentation implementation itself, but likely due to:")
    print("  1. Different random sampling in DataLoader workers vs Model forward")
    print("  2. Different batching/padding behavior")
    print("  3. CPU vs GPU execution timing/order")
    print("  4. Different tensor memory layouts affecting backpropagation")
else:
    print("\n❌ IMPLEMENTATIONS DIFFER!")
    print("\nThe augmentation implementations produce different results.")
    print("This could explain the training performance differences.")
    print("\nNext steps:")
    print("  1. Identify which specific difference causes the issue")
    print("  2. Decide which implementation is correct")
    print("  3. Fix or standardize the implementation")

print("\n" + "=" * 100)
print("ADDITIONAL TEST: With detach_augmentation=True")
print("=" * 100)

config_detach = GPT2Config(
    n_layer=2,
    n_head=2,
    n_embd=64,
    vocab_size=vocab_size + 1,
    max_seq_len=128,
    detach_augmentation=True  # Enable detach
)

model_detach = GPT2Model(
    config_detach,
    mask_token_id=mask_token_id,
    bos_token_id=bos_token_id
).to(device)

# Test with detach enabled
detach_aug_ids, detach_pos_ids, detach_N_cond, detach_N_seq = model_detach._build_augmented_sequence_single(
    sample_seq,
    cond_idx,
    eval_idx,
    unseen_idx,
    device=device
)

detach_labels = model_detach._create_labels(
    sample_seq,
    eval_idx,
    detach_aug_ids,
    detach_N_cond
)

print(f"\nMain (detach=True) Results:")
print(f"  Augmented sequence: {detach_aug_ids.tolist()}")
print(f"  Position IDs: {detach_pos_ids.tolist()}")
print(f"  Labels: {detach_labels.tolist()}")

print("\nComparison with Legacy:")
print(f"  Input IDs match: {torch.equal(legacy_aug_ids, detach_aug_ids)}")
print(f"  Position IDs match: {torch.equal(legacy_pos_ids, detach_pos_ids)}")
print(f"  Labels match: {torch.equal(legacy_labels, detach_labels)}")

if (torch.equal(legacy_aug_ids, detach_aug_ids) and
    torch.equal(legacy_pos_ids, detach_pos_ids) and
    torch.equal(legacy_labels, detach_labels)):
    print("\n  ✅ With detach=True, main matches legacy perfectly!")
    print("  → This confirms detach_augmentation works as intended")
else:
    print("\n  ❌ Even with detach=True, results still differ from legacy")
    print("  → Suggests differences beyond gradient flow")

print("\n" + "=" * 100)
print("DEBUG COMPLETE")
print("=" * 100)
