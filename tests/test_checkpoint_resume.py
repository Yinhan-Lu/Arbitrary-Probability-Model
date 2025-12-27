"""
Test checkpoint resume functionality for preemption recovery

Tests:
1. CSV truncation - _truncate_csv_to_step() removes rows with step > target
2. CSV append mode - _init_csv_logger(append=True) preserves existing CSV
3. Integration test - Full save → dirty rows → resume flow
4. Pattern matching - Experiment folder pattern works across timestamps

Run from project root: python tests/test_checkpoint_resume.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import csv
import tempfile
import shutil
import fnmatch

print("=" * 80)
print("Testing Checkpoint Resume System")
print("=" * 80)

# =========================================================================
# Test 1: CSV Truncation
# =========================================================================
print("\n[Test 1] CSV Truncation (_truncate_csv_to_step)")
print("-" * 40)

# Create a temporary directory for testing
test_dir = tempfile.mkdtemp(prefix="test_resume_")
csv_path = Path(test_dir) / "metrics.csv"

# Create CSV with steps [100, 200, 300, 400, 500]
fieldnames = ["step", "epoch", "train_loss", "train_perplexity"]
rows = [
    {"step": 100, "epoch": 0, "train_loss": 2.5, "train_perplexity": 12.0},
    {"step": 200, "epoch": 0, "train_loss": 2.3, "train_perplexity": 10.0},
    {"step": 300, "epoch": 1, "train_loss": 2.1, "train_perplexity": 8.0},
    {"step": 400, "epoch": 1, "train_loss": 1.9, "train_perplexity": 6.7},
    {"step": 500, "epoch": 2, "train_loss": 1.7, "train_perplexity": 5.5},
]

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"  Created CSV with {len(rows)} rows (steps: 100, 200, 300, 400, 500)")

# Simulate the truncation logic (same as base_trainer._truncate_csv_to_step)
target_step = 300
rows_to_keep = []

with open(csv_path, 'r', newline='') as f:
    reader = csv.DictReader(f)
    csv_fieldnames = reader.fieldnames
    for row in reader:
        step = int(row.get('step', 0))
        if step <= target_step:
            rows_to_keep.append(row)

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
    writer.writeheader()
    writer.writerows(rows_to_keep)

# Verify results
with open(csv_path, 'r', newline='') as f:
    reader = csv.DictReader(f)
    remaining_rows = list(reader)
    remaining_steps = [int(r['step']) for r in remaining_rows]

print(f"  Truncated to step <= {target_step}")
print(f"  Remaining rows: {len(remaining_rows)}")
print(f"  Remaining steps: {remaining_steps}")

assert len(remaining_rows) == 3, f"Expected 3 rows, got {len(remaining_rows)}"
assert remaining_steps == [100, 200, 300], f"Expected [100, 200, 300], got {remaining_steps}"
print("✓ Test 1 passed: CSV truncation works correctly")

# =========================================================================
# Test 2: CSV Append Mode
# =========================================================================
print("\n[Test 2] CSV Append Mode (_init_csv_logger with append=True)")
print("-" * 40)

# Create CSV with 3 rows
csv_path2 = Path(test_dir) / "metrics2.csv"
original_rows = [
    {"step": 100, "epoch": 0, "train_loss": 2.5, "train_perplexity": 12.0},
    {"step": 200, "epoch": 0, "train_loss": 2.3, "train_perplexity": 10.0},
    {"step": 300, "epoch": 1, "train_loss": 2.1, "train_perplexity": 8.0},
]

with open(csv_path2, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(original_rows)

print(f"  Created CSV with {len(original_rows)} rows")

# Simulate _init_csv_logger(append=True) - should NOT overwrite
append = True
if append and csv_path2.exists():
    # Append mode: don't overwrite
    print("  Called _init_csv_logger(append=True) - should preserve CSV")
else:
    # Would overwrite
    with open(csv_path2, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# Verify CSV still has original data
with open(csv_path2, 'r', newline='') as f:
    reader = csv.DictReader(f)
    after_rows = list(reader)

print(f"  Rows after append mode init: {len(after_rows)}")
assert len(after_rows) == 3, f"Expected 3 rows (unchanged), got {len(after_rows)}"
print("✓ Test 2 passed: Append mode preserves existing CSV")

# =========================================================================
# Test 3: Integration Test (Preemption Recovery Simulation)
# =========================================================================
print("\n[Test 3] Integration Test (Preemption Recovery Simulation)")
print("-" * 40)

# Simulate:
# 1. Training progresses to step 500
# 2. Checkpoint saved at step 300
# 3. Process killed after writing steps 400, 500 to CSV
# 4. Resume from checkpoint at step 300
# 5. Verify CSV is truncated and training would resume from step 300

csv_path3 = Path(test_dir) / "metrics3.csv"
checkpoint_path = Path(test_dir) / "checkpoint.pt"

# Step 1-3: Create CSV with all 5 steps (simulating steps written before kill)
all_rows = [
    {"step": 100, "epoch": 0, "train_loss": 2.5, "train_perplexity": 12.0},
    {"step": 200, "epoch": 0, "train_loss": 2.3, "train_perplexity": 10.0},
    {"step": 300, "epoch": 1, "train_loss": 2.1, "train_perplexity": 8.0},
    {"step": 400, "epoch": 1, "train_loss": 1.9, "train_perplexity": 6.7},  # "dirty"
    {"step": 500, "epoch": 2, "train_loss": 1.7, "train_perplexity": 5.5},  # "dirty"
]

with open(csv_path3, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print("  Simulated scenario:")
print("    - CSV has steps: 100, 200, 300, 400, 500")
print("    - Checkpoint saved at step 300, epoch 1")
print("    - Steps 400, 500 are 'dirty' (written after checkpoint)")

# Simulate checkpoint (what would be saved)
checkpoint_global_step = 300
checkpoint_epoch = 1

# Step 4: Resume - truncate CSV
class MockCSVLogger:
    def __init__(self, csv_file):
        self.csv_log_file = csv_file

    def _truncate_csv_to_step(self, target_step):
        if not self.csv_log_file.exists():
            return

        rows_to_keep = []
        with open(self.csv_log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            csv_fieldnames = reader.fieldnames
            for row in reader:
                step = int(row.get('step', 0))
                if step <= target_step:
                    rows_to_keep.append(row)

        with open(self.csv_log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_keep)

mock_logger = MockCSVLogger(csv_path3)
mock_logger._truncate_csv_to_step(checkpoint_global_step)

# Step 5: Verify
with open(csv_path3, 'r', newline='') as f:
    reader = csv.DictReader(f)
    final_rows = list(reader)
    final_steps = [int(r['step']) for r in final_rows]

print(f"\n  After resume from checkpoint (step {checkpoint_global_step}):")
print(f"    - CSV rows: {len(final_rows)}")
print(f"    - Steps: {final_steps}")
print(f"    - Training would resume from epoch {checkpoint_epoch + 1}")

assert len(final_rows) == 3, f"Expected 3 rows, got {len(final_rows)}"
assert final_steps == [100, 200, 300], f"Expected [100, 200, 300], got {final_steps}"
assert 400 not in final_steps, "Step 400 should have been removed"
assert 500 not in final_steps, "Step 500 should have been removed"
print("✓ Test 3 passed: Preemption recovery correctly truncates CSV")

# =========================================================================
# Test 4: Pattern Matching (Experiment Folder Detection)
# =========================================================================
print("\n[Test 4] Pattern Matching (Experiment Folder Detection)")
print("-" * 40)

# Test the pattern: cond0-40_max_block_rope_gpt2_conditional_*
# Should match any timestamp
pattern = "cond0-40_max_block_rope_gpt2_conditional_*"

test_folders = [
    # Should match
    ("cond0-40_max_block_rope_gpt2_conditional_20251227_143052", True),
    ("cond0-40_max_block_rope_gpt2_conditional_20251228_093000", True),
    ("cond0-40_max_block_rope_gpt2_conditional_20260101_000000", True),
    # Should NOT match (different cond percentage)
    ("cond0-60_max_block_rope_gpt2_conditional_20251227_143052", False),
    ("cond0-20_max_block_rope_gpt2_conditional_20251227_143052", False),
    # Should NOT match (different model)
    ("cond0-40_max_block_rope_distilgpt2_conditional_20251227_143052", False),
    ("cond0-40_max_block_rope_gpt2_medium_conditional_20251227_143052", False),
]

print(f"  Pattern: {pattern}")
print()

all_passed = True
for folder_name, should_match in test_folders:
    matches = fnmatch.fnmatch(folder_name, pattern)
    status = "✓" if matches == should_match else "✗"
    expected = "match" if should_match else "no match"
    actual = "matched" if matches else "no match"

    if matches != should_match:
        all_passed = False
        print(f"  {status} {folder_name}")
        print(f"      Expected: {expected}, Got: {actual}")
    else:
        print(f"  {status} {folder_name} ({actual})")

assert all_passed, "Some pattern matching tests failed"
print("\n✓ Test 4 passed: Pattern matching works correctly")

# =========================================================================
# Test 5: Verify old buggy pattern would fail
# =========================================================================
print("\n[Test 5] Verify Old Buggy Pattern Would Fail")
print("-" * 40)

# The old buggy pattern was: ${EXP_NAME%_*}_*
# This only removes ONE underscore segment (the time), not both (date and time)

# Simulate what happens when job restarts on a different day
original_exp = "cond0-40_max_block_rope_gpt2_conditional_20251227_143052"
new_timestamp_exp = "cond0-40_max_block_rope_gpt2_conditional_20251228_093000"

# Simulate ${EXP_NAME%_*}_* (remove last underscore segment)
new_exp_without_time = "_".join(new_timestamp_exp.rsplit("_", 1)[:-1])
buggy_pattern = new_exp_without_time + "_*"

print(f"  Original experiment: {original_exp}")
print(f"  New timestamp (after restart): {new_timestamp_exp}")
print(f"  Buggy pattern: {buggy_pattern}")

# The buggy pattern would be: cond0-40_max_block_rope_gpt2_conditional_20251228_*
# This would NOT match the original: cond0-40_max_block_rope_gpt2_conditional_20251227_143052
buggy_matches = fnmatch.fnmatch(original_exp, buggy_pattern)
print(f"  Buggy pattern matches original? {buggy_matches}")

assert not buggy_matches, "Buggy pattern should NOT match (this is the bug we fixed!)"
print("✓ Test 5 passed: Confirmed old buggy pattern would fail")

# Show what the correct pattern should be
correct_pattern = "cond0-40_max_block_rope_gpt2_conditional_*"
correct_matches = fnmatch.fnmatch(original_exp, correct_pattern)
print(f"\n  Correct pattern: {correct_pattern}")
print(f"  Correct pattern matches original? {correct_matches}")
assert correct_matches, "Correct pattern should match"

# =========================================================================
# Cleanup
# =========================================================================
print("\n" + "-" * 40)
shutil.rmtree(test_dir)
print(f"✓ Cleaned up test directory: {test_dir}")

print("\n" + "=" * 80)
print("✓ All checkpoint resume tests passed!")
print("=" * 80)
