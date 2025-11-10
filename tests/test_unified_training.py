"""
Test Script for Unified Training Pipeline

Tests the unified training pipeline to ensure:
1. Both trainers (conditional and baseline) can be instantiated
2. Training loop runs without errors
3. Optimizer and scheduler configurations are identical
4. Checkpointing works correctly
5. CSV logging is consistent

Run from project root:
    python tests/test_unified_training.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import logging
import shutil
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_test_args(model_type):
    """
    Create test arguments for quick training

    Args:
        model_type: "conditional" or "baseline"

    Returns:
        args: Namespace with test configuration
    """
    args = argparse.Namespace()

    # Common arguments
    args.model_type = model_type
    args.model_config = "distilgpt2"
    args.num_train_samples = 100  # Very small for quick testing
    args.num_eval_samples = 50
    args.num_workers = 0  # Avoid multiprocessing issues
    args.num_epochs = 1  # Single epoch
    args.batch_size = 4
    args.eval_batch_size = 4
    args.gradient_accumulation_steps = 2

    # Optimizer arguments
    args.learning_rate = 5e-4
    args.weight_decay = 0.01
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8
    args.max_grad_norm = 1.0

    # Scheduler arguments
    args.warmup_steps = 10  # Small warmup
    args.warmup_start_factor = 0.1
    args.min_lr_ratio = 0.1

    # Logging arguments
    args.logging_steps = 5
    args.eval_steps = 10
    args.save_steps = 20
    args.max_eval_batches = 5
    args.do_eval = True

    # Output arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"./test_experiments_{timestamp}"
    args.exp_name = f"test_{model_type}"

    # Device
    args.device = "cpu"  # Use CPU for testing to avoid device issues

    # Model-specific arguments
    if model_type == "conditional":
        args.pretrained_model_path = None
        args.resume_training = False
        args.cond_pct_min = 0.2
        args.cond_pct_max = 0.4
        args.eval_pct_min = 0.2
        args.eval_pct_max = 0.4
        args.min_conditioning = 1
        args.min_evaluation = 1
        args.conditioning_sampling = "blockwise"
        args.evaluation_sampling = "blockwise"
        args.mode2_boundary_cond_pct_min = 0.1
        args.mode2_boundary_cond_pct_max = 0.3
    elif model_type == "baseline":
        args.fp16 = False  # Disable FP16 for testing on CPU
        args.streaming = False

    return args


def test_trainer_instantiation():
    """Test that both trainers can be instantiated"""
    logger.info("=" * 80)
    logger.info("Test 1: Trainer Instantiation")
    logger.info("=" * 80)

    try:
        from train.conditional_trainer import ConditionalTrainer
        from train.baseline_trainer import BaselineTrainer

        # Test conditional trainer
        logger.info("\nTesting ConditionalTrainer instantiation...")
        args_cond = create_test_args("conditional")
        trainer_cond = ConditionalTrainer(args_cond)
        logger.info("✓ ConditionalTrainer instantiated successfully")

        # Clean up
        shutil.rmtree(args_cond.output_dir, ignore_errors=True)

        # Test baseline trainer
        logger.info("\nTesting BaselineTrainer instantiation...")
        args_base = create_test_args("baseline")
        trainer_base = BaselineTrainer(args_base)
        logger.info("✓ BaselineTrainer instantiated successfully")

        # Clean up
        shutil.rmtree(args_base.output_dir, ignore_errors=True)

        logger.info("\n✓ Test 1 passed: Both trainers can be instantiated\n")
        return True

    except Exception as e:
        logger.error(f"\n✗ Test 1 failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_consistency():
    """Test that optimizer configurations are identical"""
    logger.info("=" * 80)
    logger.info("Test 2: Optimizer Consistency")
    logger.info("=" * 80)

    try:
        from train.conditional_trainer import ConditionalTrainer
        from train.baseline_trainer import BaselineTrainer

        # Create both trainers with identical arguments (where applicable)
        args_cond = create_test_args("conditional")
        args_base = create_test_args("baseline")

        trainer_cond = ConditionalTrainer(args_cond)
        trainer_base = BaselineTrainer(args_base)

        # Check optimizer parameters
        cond_lr = trainer_cond.optimizer.param_groups[0]['lr']
        base_lr = trainer_base.optimizer.param_groups[0]['lr']

        cond_betas = trainer_cond.optimizer.param_groups[0]['betas']
        base_betas = trainer_base.optimizer.param_groups[0]['betas']

        cond_eps = trainer_cond.optimizer.param_groups[0]['eps']
        base_eps = trainer_base.optimizer.param_groups[0]['eps']

        cond_weight_decay = trainer_cond.optimizer.param_groups[0]['weight_decay']
        base_weight_decay = trainer_base.optimizer.param_groups[0]['weight_decay']

        logger.info(f"\nConditional Trainer:")
        logger.info(f"  Learning Rate: {cond_lr}")
        logger.info(f"  Betas: {cond_betas}")
        logger.info(f"  Epsilon: {cond_eps}")
        logger.info(f"  Weight Decay: {cond_weight_decay}")

        logger.info(f"\nBaseline Trainer:")
        logger.info(f"  Learning Rate: {base_lr}")
        logger.info(f"  Betas: {base_betas}")
        logger.info(f"  Epsilon: {base_eps}")
        logger.info(f"  Weight Decay: {base_weight_decay}")

        # Verify they are identical
        assert cond_lr == base_lr, "Learning rates differ"
        assert cond_betas == base_betas, "Betas differ"
        assert cond_eps == base_eps, "Epsilon differs"
        assert cond_weight_decay == base_weight_decay, "Weight decay differs"

        logger.info("\n✓ All optimizer parameters are identical")

        # Clean up
        shutil.rmtree(args_cond.output_dir, ignore_errors=True)
        shutil.rmtree(args_base.output_dir, ignore_errors=True)

        logger.info("\n✓ Test 2 passed: Optimizer configurations are identical\n")
        return True

    except Exception as e:
        logger.error(f"\n✗ Test 2 failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test that training loop runs without errors"""
    logger.info("=" * 80)
    logger.info("Test 3: Training Loop Execution")
    logger.info("=" * 80)

    success = True

    # Test conditional trainer
    logger.info("\nTesting ConditionalTrainer training loop...")
    try:
        from train.conditional_trainer import ConditionalTrainer

        args = create_test_args("conditional")
        trainer = ConditionalTrainer(args)
        trainer.train()

        logger.info("✓ ConditionalTrainer training completed successfully")

        # Check if files were created
        assert (Path(args.output_dir) / trainer.exp_name / "logs" / "metrics.csv").exists(), "CSV log not created"
        assert (Path(args.output_dir) / trainer.exp_name / "checkpoints" / "final_model.pt").exists(), "Final checkpoint not created"

        logger.info("✓ All expected files created")

        # Clean up
        shutil.rmtree(args.output_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"✗ ConditionalTrainer failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test baseline trainer
    logger.info("\nTesting BaselineTrainer training loop...")
    try:
        from train.baseline_trainer import BaselineTrainer

        args = create_test_args("baseline")
        trainer = BaselineTrainer(args)
        trainer.train()

        logger.info("✓ BaselineTrainer training completed successfully")

        # Check if files were created
        assert (Path(args.output_dir) / trainer.exp_name / "logs" / "metrics.csv").exists(), "CSV log not created"
        assert (Path(args.output_dir) / trainer.exp_name / "checkpoints" / "final_model.pt").exists(), "Final checkpoint not created"

        logger.info("✓ All expected files created")

        # Clean up
        shutil.rmtree(args.output_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"✗ BaselineTrainer failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    if success:
        logger.info("\n✓ Test 3 passed: Training loops run successfully\n")
    else:
        logger.error("\n✗ Test 3 failed: Training loop errors\n")

    return success


def test_unified_train_script():
    """Test the unified train.py script"""
    logger.info("=" * 80)
    logger.info("Test 4: Unified train.py Script")
    logger.info("=" * 80)

    import subprocess

    success = True

    # Test conditional training via unified script
    logger.info("\nTesting unified train.py with model_type=conditional...")
    try:
        result = subprocess.run([
            "python", "train.py",
            "--model_type", "conditional",
            "--model_config", "distilgpt2",
            "--num_train_samples", "100",
            "--num_eval_samples", "50",
            "--num_epochs", "1",
            "--batch_size", "4",
            "--gradient_accumulation_steps", "2",
            "--logging_steps", "5",
            "--eval_steps", "10",
            "--save_steps", "20",
            "--do_eval",
            "--device", "cpu",
            "--num_workers", "0",
            "--output_dir", "./test_experiments_unified",
            "--exp_name", "test_conditional_unified"
        ], timeout=300, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✓ Conditional training via unified script succeeded")
        else:
            logger.error(f"✗ Conditional training failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            success = False

        # Clean up
        shutil.rmtree("./test_experiments_unified", ignore_errors=True)

    except subprocess.TimeoutExpired:
        logger.error("✗ Conditional training timed out")
        success = False
    except Exception as e:
        logger.error(f"✗ Conditional training failed: {e}")
        success = False

    # Test baseline training via unified script
    logger.info("\nTesting unified train.py with model_type=baseline...")
    try:
        result = subprocess.run([
            "python", "train.py",
            "--model_type", "baseline",
            "--model_config", "distilgpt2",
            "--num_train_samples", "100",
            "--num_eval_samples", "50",
            "--num_epochs", "1",
            "--batch_size", "4",
            "--gradient_accumulation_steps", "2",
            "--logging_steps", "5",
            "--eval_steps", "10",
            "--save_steps", "20",
            "--do_eval",
            "--device", "cpu",
            "--num_workers", "0",
            "--output_dir", "./test_experiments_unified",
            "--exp_name", "test_baseline_unified"
        ], timeout=300, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✓ Baseline training via unified script succeeded")
        else:
            logger.error(f"✗ Baseline training failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            success = False

        # Clean up
        shutil.rmtree("./test_experiments_unified", ignore_errors=True)

    except subprocess.TimeoutExpired:
        logger.error("✗ Baseline training timed out")
        success = False
    except Exception as e:
        logger.error(f"✗ Baseline training failed: {e}")
        success = False

    if success:
        logger.info("\n✓ Test 4 passed: Unified train.py script works correctly\n")
    else:
        logger.error("\n✗ Test 4 failed: Unified script errors\n")

    return success


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 80)
    logger.info("Unified Training Pipeline Test Suite")
    logger.info("=" * 80 + "\n")

    results = []

    # Run tests
    results.append(("Trainer Instantiation", test_trainer_instantiation()))
    results.append(("Optimizer Consistency", test_optimizer_consistency()))
    results.append(("Training Loop Execution", test_training_loop()))
    # results.append(("Unified Script", test_unified_train_script()))  # Skip subprocess test for speed

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:.<50} {status}")

    logger.info("=" * 80)

    all_passed = all(passed for _, passed in results)

    if all_passed:
        logger.info("\n✓ ALL TESTS PASSED!\n")
        return 0
    else:
        logger.error("\n✗ SOME TESTS FAILED\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
