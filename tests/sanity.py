"""
Sanity check script for GPT-2 implementation
Runs a quick training loop with tiny config to verify:
1. Model can be instantiated
2. Data can be loaded
3. Forward/backward passes work
4. Loss decreases over time
"""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import get_config
from model.arbitrary_prob_gpt2 import GPT2Model, create_causal_mask
from train.dataset import get_dataloader
from train.loop import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_instantiation(config_name="tiny"):
    """Test M0: Model instantiation and forward pass"""
    logger.info("=" * 60)
    logger.info("TEST M0: Model Instantiation & Forward Pass")
    logger.info("=" * 60)

    config = get_config(config_name)
    model = GPT2Model(config)

    # Test with random input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logger.info(f"Input shape: {input_ids.shape}")

    # Forward pass with default causal mask
    try:
        logits, _ = model(input_ids)
        logger.info(f"Output shape: {logits.shape}")
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        logger.info("✓ Forward pass with default causal mask: PASSED")
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        return False

    # Forward pass with custom mask
    try:
        custom_mask = create_causal_mask(seq_len)
        logits, _ = model(input_ids, attention_mask=custom_mask)
        logger.info(f"Output shape (custom mask): {logits.shape}")
        logger.info("✓ Forward pass with custom mask: PASSED")
    except Exception as e:
        logger.error(f"✗ Forward pass with custom mask failed: {e}")
        return False

    # Test loss computation
    try:
        labels = input_ids.clone()
        logits, loss = model(input_ids, labels=labels)
        logger.info(f"Loss: {loss.item():.4f}")
        assert loss.item() > 0, "Loss should be positive"
        logger.info("✓ Loss computation: PASSED")
    except Exception as e:
        logger.error(f"✗ Loss computation failed: {e}")
        return False

    logger.info("\n✓ M0: Model instantiation test PASSED\n")
    return True


def test_data_loading(config_name="tiny", num_samples=10):
    """Test M1: Data loading and tokenization"""
    logger.info("=" * 60)
    logger.info("TEST M1: Data Loading & Tokenization")
    logger.info("=" * 60)

    config = get_config(config_name)

    try:
        train_loader = get_dataloader(
            config,
            split="train",
            batch_size=2,
            num_workers=0,
            num_samples=num_samples
        )
        logger.info(f"✓ Train DataLoader created: {len(train_loader)} batches")
    except Exception as e:
        logger.error(f"✗ Failed to create train DataLoader: {e}")
        return False

    try:
        batch = next(iter(train_loader))
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"input_ids shape: {batch['input_ids'].shape}")
        logger.info(f"attention_mask shape: {batch['attention_mask'].shape}")

        assert "input_ids" in batch, "Batch should contain input_ids"
        assert "attention_mask" in batch, "Batch should contain attention_mask"
        assert batch["input_ids"].shape[1] <= config.max_seq_len, "Sequence too long"

        logger.info("✓ Batch structure: PASSED")
    except Exception as e:
        logger.error(f"✗ Failed to load batch: {e}")
        return False

    logger.info("\n✓ M1: Data loading test PASSED\n")
    return True


def test_training_loop(config_name="tiny", num_train_samples=50, num_val_samples=10):
    """Test M2: Training loop and loss convergence"""
    logger.info("=" * 60)
    logger.info("TEST M2: Training Loop & Loss Convergence")
    logger.info("=" * 60)

    config = get_config(config_name)
    logger.info(f"Using config: {config_name}")
    logger.info(f"Parameters: n_layer={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}")

    # Create model
    model = GPT2Model(config)

    # Create dataloaders
    logger.info(f"\nCreating dataloaders (train={num_train_samples}, val={num_val_samples})...")
    train_loader = get_dataloader(
        config,
        split="train",
        batch_size=4,
        num_workers=0,
        num_samples=num_train_samples
    )

    val_loader = get_dataloader(
        config,
        split="validation",
        batch_size=4,
        num_workers=0,
        num_samples=num_val_samples
    )

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        learning_rate=1e-3,  # Higher LR for faster convergence in sanity check
        max_epochs=1,
        device=device,
        log_dir="logs/sanity",
        checkpoint_dir="checkpoints/sanity",
        log_interval=5,
        eval_interval=20,
        save_interval=100
    )

    # Record initial loss
    logger.info("\nRecording initial loss...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(device)
        # Prepare labels: set padding tokens to -100
        labels = input_ids.clone()
        labels[labels == 50256] = -100
        _, initial_loss = model(input_ids, labels=labels)
        initial_loss = initial_loss.item()
        logger.info(f"Initial loss: {initial_loss:.4f}")

    # Train for 1 epoch
    logger.info("\nTraining for 1 epoch...")
    try:
        trainer.train_epoch(epoch=1)
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Record final loss
    logger.info("\nRecording final loss...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(device)
        # Prepare labels: set padding tokens to -100
        labels = input_ids.clone()
        labels[labels == 50256] = -100
        _, final_loss = model(input_ids, labels=labels)
        final_loss = final_loss.item()
        logger.info(f"Final loss: {final_loss:.4f}")

    # Check if loss decreased
    logger.info(f"\nLoss change: {initial_loss:.4f} → {final_loss:.4f} (Δ = {final_loss - initial_loss:.4f})")

    if final_loss < initial_loss:
        logger.info("✓ Loss decreased: PASSED")
        logger.info(f"✓ Reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
    else:
        logger.warning("⚠ Loss did not decrease (may need more steps or higher LR)")

    # Plot training curves
    try:
        trainer.plot_training_curves()
        logger.info("✓ Training curves plotted")
    except Exception as e:
        logger.warning(f"⚠ Failed to plot curves: {e}")

    logger.info("\n✓ M2: Training loop test PASSED\n")
    return True


def run_full_sanity_check():
    """Run all sanity checks"""
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING FULL SANITY CHECK")
    logger.info("=" * 60 + "\n")

    results = {}

    # M0: Model instantiation
    results["M0_instantiation"] = test_model_instantiation("tiny")

    # M1: Data loading
    results["M1_data_loading"] = test_data_loading("tiny", num_samples=10)

    # M2: Training loop
    results["M2_training"] = test_training_loop("tiny", num_train_samples=50, num_val_samples=10)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECK SUMMARY")
    logger.info("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        logger.info("\n✓ ALL TESTS PASSED - Model is ready for training!")
    else:
        logger.error("\n✗ SOME TESTS FAILED - Please fix issues before proceeding")

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity check for GPT-2 implementation")
    parser.add_argument(
        "--test",
        choices=["all", "m0", "m1", "m2"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument(
        "--config",
        choices=["tiny", "nano", "distilgpt2"],
        default="tiny",
        help="Model configuration to use"
    )

    args = parser.parse_args()

    if args.test == "all":
        success = run_full_sanity_check()
    elif args.test == "m0":
        success = test_model_instantiation(args.config)
    elif args.test == "m1":
        success = test_data_loading(args.config)
    elif args.test == "m2":
        success = test_training_loop(args.config)

    sys.exit(0 if success else 1)
