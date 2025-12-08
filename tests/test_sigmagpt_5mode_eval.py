"""
Test SigmaGPT 5-mode evaluation

This test verifies that SigmaGPT trainer can run all 5 evaluation modes
correctly and produce valid metrics matching the conditional model format.

Run from project root: python tests/test_sigmagpt_5mode_eval.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_args():
    """Create minimal args for testing"""
    args = argparse.Namespace(
        # Model config
        model_config='distilgpt2',
        sigmagpt_mode='fair',
        sigmagpt_arch='new',
        sigmagpt_eval_mode='autoregressive',  # Not used anymore (we now use 5-mode)

        # Data config
        dataset_name='wikitext',
        dataset_config='wikitext-103-raw-v1',
        num_train_samples=100,
        num_eval_samples=50,
        batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        streaming=False,
        primary_dataset_only=False,

        # Augmentation config
        cond_pct_min=0.0,
        cond_pct_max=0.4,
        eval_pct_min=1.0,
        eval_pct_max=1.0,
        conditioning_sampling='blockwise',
        evaluation_sampling='blockwise',
        max_cond_blocks=3,
        max_eval_blocks=2,
        ordering_mode='temporal',

        # Mode 2 boundary config
        mode2_boundary_cond_pct_min=0.1,
        mode2_boundary_cond_pct_max=0.3,

        # Training config
        num_epochs=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        weight_decay=0.1,
        warmup_steps=10,
        warmup_start_factor=0.1,
        min_lr_ratio=0.1,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,

        # Evaluation config
        max_eval_batches=2,
        do_eval=True,
        eval_steps=100,
        logging_steps=10,
        save_steps=1000,

        # Output config
        output_dir='./test_output',
        exp_name='test_sigmagpt_5mode',
        device='cpu',  # Use CPU for testing
    )
    return args


def test_5mode_evaluation():
    """Test that 5-mode evaluation runs and produces valid results"""
    print("=" * 80)
    print("TEST: SigmaGPT 5-Mode Evaluation")
    print("=" * 80)

    from train.sigmagpt_trainer import SigmaGPTTrainer

    # Create args
    args = create_test_args()
    print(f"\n[1] Created test args with ordering_mode={args.ordering_mode}")

    # Create trainer (BaseTrainer.__init__ calls setup_model and setup_data automatically)
    print("\n[2] Creating SigmaGPT trainer...")
    trainer = SigmaGPTTrainer(args)
    print("    ✓ Trainer created successfully")

    # Run evaluation
    print("\n[3] Running 5-mode evaluation...")
    eval_results = trainer.evaluate()

    # Verify results have all 5 modes
    print("\n[4] Verifying results contain all 5 modes...")
    expected_keys = [
        'mode1_loss', 'mode1_ppl',
        'mode2_loss', 'mode2_ppl',
        'mode3_loss', 'mode3_ppl',
        'mode4_loss', 'mode4_ppl',
        'mode5_loss', 'mode5_ppl',
        'loss'  # Main loss (mode3_loss)
    ]

    missing_keys = [k for k in expected_keys if k not in eval_results]
    if missing_keys:
        print(f"    ✗ FAILED: Missing keys: {missing_keys}")
        return False

    print("    ✓ All expected keys present")

    # Verify values are valid numbers
    print("\n[5] Verifying values are valid numbers...")
    for key in expected_keys:
        value = eval_results[key]
        if not isinstance(value, (int, float)):
            print(f"    ✗ FAILED: {key} is not a number: {value}")
            return False
        if value < 0 or value > 1000:  # Sanity check
            print(f"    ⚠ WARNING: {key} has unusual value: {value}")

    print("    ✓ All values are valid numbers")

    # Print results
    print("\n[6] Evaluation Results:")
    print("-" * 60)
    for i in range(1, 6):
        loss_key = f'mode{i}_loss'
        ppl_key = f'mode{i}_ppl'
        print(f"    Mode {i}: loss={eval_results[loss_key]:.4f}, ppl={eval_results[ppl_key]:.2f}")
    print("-" * 60)

    # Verify CSV format
    print("\n[7] Verifying CSV header format...")
    csv_header = trainer.get_csv_header()
    expected_header_keys = [
        'step', 'epoch', 'train_loss', 'train_perplexity',
        'mode1_loss', 'mode1_ppl', 'mode2_loss', 'mode2_ppl',
        'mode3_loss', 'mode3_ppl', 'mode4_loss', 'mode4_ppl',
        'mode5_loss', 'mode5_ppl', 'learning_rate',
        'sigmagpt_mode', 'ordering_mode'
    ]

    if csv_header != expected_header_keys:
        print(f"    ✗ FAILED: CSV header mismatch")
        print(f"    Expected: {expected_header_keys}")
        print(f"    Got: {csv_header}")
        return False

    print("    ✓ CSV header matches expected format")

    # Verify format_eval_metrics
    print("\n[8] Verifying format_eval_metrics()...")
    formatted = trainer.format_eval_metrics(eval_results)

    for key in ['mode1_loss', 'mode2_loss', 'mode3_loss', 'mode4_loss', 'mode5_loss']:
        if key not in formatted:
            print(f"    ✗ FAILED: {key} missing from formatted metrics")
            return False

    print("    ✓ format_eval_metrics() produces correct format")

    print("\n" + "=" * 80)
    print("TEST PASSED: SigmaGPT 5-mode evaluation works correctly!")
    print("=" * 80)

    return True


def test_both_ordering_modes():
    """Test both temporal and scramble ordering modes"""
    print("\n" + "=" * 80)
    print("TEST: Both Ordering Modes (Temporal and Scramble)")
    print("=" * 80)

    from train.sigmagpt_trainer import SigmaGPTTrainer

    for ordering_mode in ['temporal', 'random_scramble']:
        print(f"\n--- Testing ordering_mode={ordering_mode} ---")

        args = create_test_args()
        args.ordering_mode = ordering_mode

        trainer = SigmaGPTTrainer(args)

        eval_results = trainer.evaluate()

        # Quick validation
        if 'mode1_loss' not in eval_results:
            print(f"    ✗ FAILED for {ordering_mode}")
            return False

        print(f"    ✓ {ordering_mode} mode works correctly")
        print(f"      Mode 1 loss: {eval_results['mode1_loss']:.4f}")
        print(f"      Mode 3 loss: {eval_results['mode3_loss']:.4f}")

    print("\n" + "=" * 80)
    print("TEST PASSED: Both ordering modes work correctly!")
    print("=" * 80)

    return True


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("SigmaGPT 5-Mode Evaluation Tests")
    print("=" * 80)

    # Run tests
    test1_passed = test_5mode_evaluation()

    if test1_passed:
        test2_passed = test_both_ordering_modes()
    else:
        test2_passed = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  5-Mode Evaluation: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"  Both Ordering Modes: {'PASSED' if test2_passed else 'FAILED'}")
    print("=" * 80)

    if test1_passed and test2_passed:
        print("\n✓ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED!")
        sys.exit(1)
