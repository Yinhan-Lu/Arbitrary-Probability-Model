"""
Performance benchmark for tensor augmentation operations

Measures augmentation speed on CPU vs GPU to verify .item() elimination speedup.
Expected results:
- GPU augmentation: ~10-50ms per batch (pure tensor ops)
- CPU augmentation: ~50-200ms per batch (acceptable baseline)

Run from project root: python tests/benchmark_augmentation.py
"""

import torch
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.arbitrary_prob_gpt2 import GPT2Model
from model.config import get_config

def benchmark_single_sample(model, input_ids, cond_idx, eval_idx, unseen_idx, device, n_iterations=100):
    """Benchmark single sample augmentation"""
    input_ids = input_ids.to(device)

    # Warmup
    for _ in range(10):
        _ = model._build_augmented_sequence_single(input_ids, cond_idx, eval_idx, unseen_idx, device=device)

    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iterations):
        _ = model._build_augmented_sequence_single(input_ids, cond_idx, eval_idx, unseen_idx, device=device)

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = (elapsed / n_iterations) * 1000  # Convert to ms

    return avg_time

def benchmark_batch_augmentation(model, batch_size=16, seq_len=256, device='cpu', n_iterations=50):
    """Benchmark full batch augmentation (simulating training scenario)"""

    # Generate batch data
    batch_input_ids = []
    batch_cond_idx = []
    batch_eval_idx = []
    batch_unseen_idx = []

    for _ in range(batch_size):
        input_ids = torch.randint(0, 50257, (seq_len,))

        # Typical indices distributions
        n_cond = torch.randint(5, 20, (1,)).item()
        n_eval = torch.randint(10, 30, (1,)).item()
        n_unseen = torch.randint(20, 50, (1,)).item()

        cond_idx = torch.randperm(seq_len)[:n_cond].tolist()
        eval_idx = torch.randperm(seq_len)[:n_eval].tolist()
        unseen_idx = torch.randperm(seq_len)[:n_unseen].tolist()

        batch_input_ids.append(input_ids)
        batch_cond_idx.append(cond_idx)
        batch_eval_idx.append(eval_idx)
        batch_unseen_idx.append(unseen_idx)

    # Move to device
    batch_input_ids = [ids.to(device) for ids in batch_input_ids]

    # Warmup
    for _ in range(5):
        for i in range(batch_size):
            _ = model._build_augmented_sequence_single(
                batch_input_ids[i], batch_cond_idx[i], batch_eval_idx[i],
                batch_unseen_idx[i], device=device
            )
            _ = model._create_labels(
                batch_input_ids[i], batch_eval_idx[i],
                torch.randint(0, 50257, (300,), device=device), 10
            )

    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iterations):
        for i in range(batch_size):
            aug_ids, pos_ids, N_c, N_s = model._build_augmented_sequence_single(
                batch_input_ids[i], batch_cond_idx[i], batch_eval_idx[i],
                batch_unseen_idx[i], device=device
            )
            labels = model._create_labels(
                batch_input_ids[i], batch_eval_idx[i], aug_ids, N_c
            )

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time_per_batch = (elapsed / n_iterations) * 1000  # ms per batch
    avg_time_per_sample = avg_time_per_batch / batch_size

    return avg_time_per_batch, avg_time_per_sample

def run_benchmarks():
    """Run comprehensive performance benchmarks"""
    print("=" * 80)
    print("Performance Benchmark: Tensor Augmentation Operations")
    print("=" * 80)

    config = get_config("tiny")
    model = GPT2Model(config, mask_token_id=50257, bos_token_id=50256)

    # Test data
    input_ids = torch.randint(0, 50257, (256,))
    cond_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    eval_idx = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100, 110, 120]
    unseen_idx = list(range(100, 150))

    print(f"\nTest Configuration:")
    print(f"  Sequence length: {len(input_ids)}")
    print(f"  Conditioning tokens: {len(cond_idx)}")
    print(f"  Evaluation tokens: {len(eval_idx)}")
    print(f"  Unseen tokens: {len(unseen_idx)}")

    # ===== Single Sample Benchmark =====
    print("\n" + "-" * 80)
    print("[Benchmark 1] Single Sample Augmentation")
    print("-" * 80)

    cpu_time = benchmark_single_sample(model, input_ids, cond_idx, eval_idx, unseen_idx, device='cpu')
    print(f"  CPU: {cpu_time:.3f} ms per sample")

    if torch.cuda.is_available():
        gpu_time = benchmark_single_sample(model, input_ids, cond_idx, eval_idx, unseen_idx, device='cuda')
        print(f"  GPU: {gpu_time:.3f} ms per sample")
        print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
    else:
        print("  ⚠ CUDA not available, skipping GPU benchmark")

    # ===== Batch Benchmark =====
    print("\n" + "-" * 80)
    print("[Benchmark 2] Full Batch Augmentation (batch_size=16)")
    print("-" * 80)

    batch_time_cpu, sample_time_cpu = benchmark_batch_augmentation(
        model, batch_size=16, seq_len=256, device='cpu', n_iterations=50
    )
    print(f"  CPU:")
    print(f"    Per batch:  {batch_time_cpu:.1f} ms")
    print(f"    Per sample: {sample_time_cpu:.2f} ms")

    if torch.cuda.is_available():
        batch_time_gpu, sample_time_gpu = benchmark_batch_augmentation(
            model, batch_size=16, seq_len=256, device='cuda', n_iterations=50
        )
        print(f"  GPU:")
        print(f"    Per batch:  {batch_time_gpu:.1f} ms")
        print(f"    Per sample: {sample_time_gpu:.2f} ms")
        print(f"  Speedup: {batch_time_cpu / batch_time_gpu:.2f}x")

        # Analysis
        print("\n" + "-" * 80)
        print("Performance Analysis")
        print("-" * 80)

        if batch_time_gpu < 100:
            print(f"  ✅ GPU augmentation is FAST ({batch_time_gpu:.1f} ms per batch)")
            print(f"     Expected training impact: minimal overhead")
        elif batch_time_gpu < 300:
            print(f"  ⚠ GPU augmentation is acceptable ({batch_time_gpu:.1f} ms per batch)")
            print(f"     Expected training impact: ~10-20% overhead")
        else:
            print(f"  ❌ GPU augmentation is SLOW ({batch_time_gpu:.1f} ms per batch)")
            print(f"     Expected training impact: significant overhead (>30%)")

        # Estimate training impact
        forward_backward_time = 50  # Typical time for forward + backward on small model
        total_time_per_batch = batch_time_gpu + forward_backward_time
        augmentation_fraction = (batch_time_gpu / total_time_per_batch) * 100

        print(f"\n  Estimated augmentation overhead in training:")
        print(f"    Augmentation: {batch_time_gpu:.1f} ms")
        print(f"    Forward+Backward: ~{forward_backward_time} ms (estimated)")
        print(f"    Total per batch: ~{total_time_per_batch:.1f} ms")
        print(f"    Augmentation fraction: {augmentation_fraction:.1f}%")

    print("\n" + "=" * 80)
    print("Benchmark Complete")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmarks()
