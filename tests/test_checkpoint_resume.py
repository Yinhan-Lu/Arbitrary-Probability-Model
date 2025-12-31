#!/usr/bin/env python3
"""
Quick test for checkpoint resume functionality.

This test verifies:
1. Training starts and saves checkpoints
2. metrics.csv is NOT cleared when resuming
3. Training continues in the SAME folder
4. Step counter continues correctly

Run from project root:
    python tests/test_checkpoint_resume.py

Expected time: ~60 seconds
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd, description):
    """Run a command and return output"""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd[:10])}...")
    print()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Command failed with code {result.returncode}")

    return result.stdout


def count_csv_lines(csv_path):
    """Count non-header lines in CSV"""
    if not csv_path.exists():
        return 0
    with open(csv_path) as f:
        lines = f.readlines()
    return max(0, len(lines) - 1)


def get_last_step(csv_path):
    """Get the last step number from CSV"""
    if not csv_path.exists():
        return -1
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return -1
    last_line = lines[-1].strip()
    if not last_line:
        last_line = lines[-2].strip()
    try:
        return int(last_line.split(',')[0])
    except:
        return -1


def main():
    print("\n" + "="*60)
    print("CHECKPOINT RESUME TEST")
    print("="*60)
    print("This test verifies that checkpoint resume works correctly")
    print("and that metrics.csv is NOT cleared when resuming.\n")

    # Test parameters - use a fixed name without timestamp
    exp_name = "test_checkpoint_resume_20991231_235959"
    exp_dir = Path("experiments") / exp_name
    logs_dir = exp_dir / "logs"
    ckpt_dir = exp_dir / "checkpoints"
    csv_path = logs_dir / "metrics.csv"

    # Clean up any previous test
    if exp_dir.exists():
        print(f"Cleaning up previous test directory: {exp_dir}")
        shutil.rmtree(exp_dir)

    # =========================================================================
    # PHASE 1: Initial training (save checkpoint at step 50)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Initial Training")
    print("="*60)

    # Use very few samples to make training quick
    # Phase 1: train with 64 samples, batch_size=4, grad_accum=2 = 8 steps
    # save_steps=5 will save checkpoint at step 5
    cmd_phase1 = [
        "python", "train.py",
        "--model_type", "sigmagpt",
        "--model_config", "tiny",
        "--position_encoding_type", "rope",
        "--num_epochs", "1",
        "--batch_size", "4",
        "--gradient_accumulation_steps", "2",
        "--num_train_samples", "64",
        "--num_eval_samples", "16",
        "--logging_steps", "2",
        "--eval_steps", "1000",
        "--save_steps", "5",
        "--output_dir", "./experiments",
        "--exp_name", exp_name,
        "--device", "cpu",
        "--num_workers", "0",
    ]

    run_command(cmd_phase1, "Running initial training")

    # Verify experiment directory exists
    if not exp_dir.exists():
        raise RuntimeError(f"Experiment directory not created: {exp_dir}")

    print(f"\nExperiment directory: {exp_dir}")

    # Verify checkpoint exists (either step checkpoint or final model)
    checkpoints = list(ckpt_dir.glob("checkpoint_step_*.pt"))
    if checkpoints:
        latest_ckpt = sorted(checkpoints)[-1]
    else:
        # Fall back to final_model.pt
        final_ckpt = ckpt_dir / "final_model.pt"
        if final_ckpt.exists():
            latest_ckpt = final_ckpt
        else:
            raise RuntimeError("No checkpoint found after Phase 1!")

    print(f"Checkpoint saved: {latest_ckpt.name}")

    # Record metrics before resume
    lines_before = count_csv_lines(csv_path)
    last_step_before = get_last_step(csv_path)
    print(f"metrics.csv lines (before resume): {lines_before}")
    print(f"Last step (before resume): {last_step_before}")

    if lines_before == 0:
        raise RuntimeError("metrics.csv is empty after Phase 1!")

    # Save a copy of metrics.csv for comparison
    csv_backup = logs_dir / "metrics_backup.csv"
    shutil.copy(csv_path, csv_backup)

    # =========================================================================
    # PHASE 2: Resume training (run another epoch)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Resume Training")
    print("="*60)

    # Phase 2: resume and train another epoch
    cmd_phase2 = [
        "python", "train.py",
        "--model_type", "sigmagpt",
        "--model_config", "tiny",
        "--position_encoding_type", "rope",
        "--num_epochs", "2",  # Train 2 epochs total (resume from epoch 1)
        "--batch_size", "4",
        "--gradient_accumulation_steps", "2",
        "--num_train_samples", "64",
        "--num_eval_samples", "16",
        "--logging_steps", "2",
        "--eval_steps", "1000",
        "--save_steps", "5",
        "--output_dir", "./experiments",
        "--exp_name", exp_name,  # Same exp_name!
        "--device", "cpu",
        "--num_workers", "0",
        "--resume_from", str(latest_ckpt),
    ]

    run_command(cmd_phase2, "Resuming training from checkpoint")

    # =========================================================================
    # VERIFICATION
    # =========================================================================
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    # Check metrics.csv
    lines_after = count_csv_lines(csv_path)
    last_step_after = get_last_step(csv_path)

    print(f"\nmetrics.csv lines (after resume): {lines_after}")
    print(f"Last step (after resume): {last_step_after}")

    # Test 1: Lines were added, not replaced
    if lines_after <= lines_before:
        print("\n[FAIL] metrics.csv was cleared or not appended!")
        print(f"  Before: {lines_before} lines")
        print(f"  After:  {lines_after} lines")
        return False
    print(f"[PASS] metrics.csv grew from {lines_before} to {lines_after} lines")

    # Test 2: Step counter continued
    if last_step_after <= last_step_before:
        print("\n[FAIL] Step counter did not continue!")
        print(f"  Before: step {last_step_before}")
        print(f"  After:  step {last_step_after}")
        return False
    print(f"[PASS] Step counter continued from {last_step_before} to {last_step_after}")

    # Test 3: Original data preserved
    with open(csv_backup) as f:
        original_lines = f.readlines()
    with open(csv_path) as f:
        current_lines = f.readlines()

    for i, orig_line in enumerate(original_lines):
        if current_lines[i] != orig_line:
            print(f"\n[FAIL] Original data at line {i} was modified!")
            return False
    print("[PASS] Original metrics data preserved")

    # Test 4: No duplicate folders created
    matching_folders = list(Path("experiments").glob("test_checkpoint_resume*"))
    if len(matching_folders) > 1:
        print(f"\n[FAIL] Multiple folders created: {matching_folders}")
        return False
    print("[PASS] Only one experiment folder exists")

    # Clean up
    print(f"\nCleaning up test directory: {exp_dir}")
    shutil.rmtree(exp_dir)

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nCheckpoint resume is working correctly:")
    print("  - Checkpoints are saved")
    print("  - metrics.csv is appended (not cleared)")
    print("  - Step counter continues correctly")
    print("  - Original data is preserved")
    print("  - No duplicate folders created")
    print()

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
