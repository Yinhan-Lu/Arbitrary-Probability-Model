# Test Suite

This folder contains all test scripts for the Arbitrary Probability Model project.

## 📋 Test Organization

All test scripts are organized in this `tests/` folder for clarity and maintainability.

### Current Tests

| Test File | Purpose | Run Time | Command |
|-----------|---------|----------|---------|
| **quick_test.py** | Fast model validation (instantiation + forward pass) | ~10s | `python tests/quick_test.py` |
| **sanity.py** | Complete training pipeline test | ~3 min | `python tests/sanity.py` |
| **test_checkpoint_loading.py** | Verify checkpoint save/load functionality | ~30s | `python tests/test_checkpoint_loading.py` |
| **test_pretrained_loading.py** | Test HuggingFace model loading | ~1 min | `python tests/test_pretrained_loading.py` |
| **test_detach_augmentation.py** | Test detach_augmentation parameter functionality | ~5s | `python tests/test_detach_augmentation.py` |
| **debug_mode2_comparison.py** | Compare Legacy vs New Mode 2 augmentation (requires GPU) | ~30s | `python tests/debug_mode2_comparison.py` |
| **debug_mode2_simple.py** | Simplified Mode 2 logic comparison (no GPU required) | ~1s | `python tests/debug_mode2_simple.py` |

## 🚀 Quick Start

### Run All Quick Tests

```bash
# From project root
cd /path/to/Arbitrary-Probability-Model

# Quick validation (10 seconds)
python tests/quick_test.py

# Checkpoint loading test
python tests/test_checkpoint_loading.py
```

### Full Sanity Check

```bash
# Complete training pipeline test (~3 minutes)
python tests/sanity.py
```

## 📝 Test Categories

### 1. Unit Tests (Fast: < 30s)
- `quick_test.py` - Model instantiation and forward pass
- `test_checkpoint_loading.py` - Checkpoint I/O

### 2. Integration Tests (Medium: 30s - 5min)
- `test_pretrained_loading.py` - HuggingFace integration
- `sanity.py` - Training pipeline (10 steps)

### 3. Debug Tests (Special Purpose)
- `test_detach_augmentation.py` - Verify detach_augmentation feature
- `debug_mode2_comparison.py` - Deep dive into Mode 2 augmentation differences
- `debug_mode2_simple.py` - Simplified logic comparison (no dependencies)

### 4. System Tests (Slow: > 5min)
- Located in `scripts/` folder for SLURM execution

## 🔧 Writing New Tests

### Test File Template

```python
"""
Brief description of what this test does

Run from project root: python tests/test_name.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.config import get_config
from model.arbitrary_prob_gpt2 import GPT2Model

print("=" * 80)
print("Test Name")
print("=" * 80)

# Test code here
print("\n[Test 1] Description...")
# ... test logic ...
print("✓ Test 1 passed")

print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
```

### Guidelines

1. **Always add project root to sys.path**
   ```python
   sys.path.insert(0, str(Path(__file__).parent.parent))
   ```

2. **Use clear test names**
   - `test_<feature>.py` for feature-specific tests
   - `<component>_test.py` for component tests

3. **Print clear progress**
   ```python
   print("\n[Test 1] Testing feature X...")
   # ... test code ...
   print("✓ Test 1 passed")
   ```

4. **Exit with proper codes**
   ```python
   # On success: implicit exit(0)
   # On failure: raise exception or sys.exit(1)
   ```

5. **Add to this README**
   - Update the test table above
   - Categorize by run time
   - Document expected output

## 🐛 Debugging Tests

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'model'`**

Solution: Run from project root, not from tests/ folder
```bash
# ✗ Wrong
cd tests && python quick_test.py

# ✓ Correct
python tests/quick_test.py
```

**Issue: `CUDA out of memory`**

Solution: Tests should use small models (nano or distilgpt2)
```python
config = get_config("nano")  # Not "gpt2-large"
```

**Issue: Test hangs indefinitely**

Solution: Set reasonable timeouts and small data sizes
```python
num_samples = 100  # Not 100000 for tests
num_epochs = 1     # Not 10 for tests
```

## 📊 Test Coverage

Current coverage:
- ✅ Model instantiation
- ✅ Forward pass
- ✅ Checkpoint save/load
- ✅ Token manager
- ✅ Custom attention masks
- ✅ Data augmentation
- ✅ Training loop (basic)
- ⚠️ HuggingFace loading (PIL dependency issue)
- ❌ Inference/generation (TODO)
- ❌ Evaluation metrics (TODO)

## 🎯 Best Practices

1. **Fast Feedback Loop**
   - Keep quick tests under 30 seconds
   - Run before every commit

2. **Isolation**
   - Tests should not depend on each other
   - Clean up temporary files

3. **Deterministic**
   - Use `torch.manual_seed()` for reproducibility
   - Document expected outputs

4. **Informative**
   - Print what is being tested
   - Show intermediate results
   - Clear success/failure messages

## 🔄 CI/CD Integration (Future)

When setting up CI/CD, run tests in this order:

```bash
# Stage 1: Quick validation (< 1 min)
python tests/quick_test.py
python tests/test_checkpoint_loading.py

# Stage 2: Integration tests (< 5 min)
python tests/sanity.py

# Stage 3: Full training (run nightly)
# ... full experiments ...
```

---

**Last Updated**: 2025-10-21
**Maintained By**: Project team
