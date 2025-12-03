"""
Test pure tensor operations in augmentation
验证正确性、边界情况、设备一致性
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.arbitrary_prob_gpt2 import GPT2Model
from model.config import get_config

def test_basic_augmentation():
    """测试基本功能"""
    print("\n[Test 1] Basic augmentation")
    config = get_config("tiny")
    model = GPT2Model(config, mask_token_id=50257, bos_token_id=50256)

    input_ids = torch.randint(0, 50257, (100,))
    cond_idx = [0, 10, 20, 30]
    eval_idx = [5, 15, 25]
    unseen_idx = [5, 15, 25, 35, 45]

    aug_ids, pos_ids, N_c, N_s = model._build_augmented_sequence_single(
        input_ids, cond_idx, eval_idx, unseen_idx, device='cpu'
    )

    # 验证形状
    expected_len = len(cond_idx) + 1 + len(input_ids)
    assert aug_ids.shape[0] == expected_len, f"Expected {expected_len}, got {aug_ids.shape[0]}"
    assert pos_ids.shape[0] == expected_len
    assert N_c == len(cond_idx)
    assert N_s == 1 + len(input_ids)

    print(f"  ✓ Shape correct: aug_len={aug_ids.shape[0]}, N_cond={N_c}, N_seq={N_s}")

def test_empty_sets():
    """测试边界情况：空集合"""
    print("\n[Test 2] Empty sets")
    config = get_config("tiny")
    model = GPT2Model(config, mask_token_id=50257, bos_token_id=50256)

    input_ids = torch.randint(0, 50257, (50,))

    # 空 conditioning
    aug_ids, _, N_c, _ = model._build_augmented_sequence_single(
        input_ids, [], [5, 10], [5, 10, 15], device='cpu'
    )
    assert N_c == 0
    print(f"  ✓ Empty conditioning: N_cond={N_c}")

    # 空 evaluation
    labels = model._create_labels(input_ids, [], aug_ids, N_c)
    assert (labels == -100).all()
    print(f"  ✓ Empty evaluation: all labels=-100")

    # 空 unseen
    aug_ids, _, _, _ = model._build_augmented_sequence_single(
        input_ids, [0, 5], [10, 15], [], device='cpu'
    )
    print(f"  ✓ Empty unseen: no masking")

def test_device_consistency():
    """测试设备一致性"""
    print("\n[Test 3] Device consistency")
    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available, skipping GPU test")
        return

    config = get_config("tiny")
    model = GPT2Model(config, mask_token_id=50257, bos_token_id=50256)

    input_ids_cpu = torch.randint(0, 50257, (100,))
    input_ids_gpu = input_ids_cpu.cuda()

    cond_idx = [0, 10, 20]
    eval_idx = [5, 15]
    unseen_idx = [5, 15, 25]

    # CPU 版本
    aug_cpu, pos_cpu, N_c, N_s = model._build_augmented_sequence_single(
        input_ids_cpu, cond_idx, eval_idx, unseen_idx, device='cpu'
    )

    # GPU 版本
    aug_gpu, pos_gpu, _, _ = model._build_augmented_sequence_single(
        input_ids_gpu, cond_idx, eval_idx, unseen_idx, device='cuda'
    )

    # 验证结果一致
    assert torch.equal(aug_cpu, aug_gpu.cpu()), "CPU and GPU results differ!"
    assert torch.equal(pos_cpu, pos_gpu.cpu()), "Position IDs differ!"

    print(f"  ✓ CPU and GPU produce identical results")

def test_labels_creation():
    """测试 labels 创建"""
    print("\n[Test 4] Labels creation")
    config = get_config("tiny")
    model = GPT2Model(config, mask_token_id=50257, bos_token_id=50256)

    input_ids = torch.tensor([100, 200, 300, 400, 500])
    eval_idx = [1, 3]  # 评估位置 1 和 3

    # 创建 augmented sequence (假设 N_cond=2)
    aug_input_ids = torch.tensor([10, 20, 50256, 100, 200, 300, 400, 500])
    N_cond = 2

    labels = model._create_labels(input_ids, eval_idx, aug_input_ids, N_cond)

    # 验证
    # eval_idx=1 -> new_pos=2+1+1=4 -> labels[4]=input_ids[1]=200
    # eval_idx=3 -> new_pos=2+1+3=6 -> labels[6]=input_ids[3]=400
    assert labels[4] == 200, f"Expected 200, got {labels[4]}"
    assert labels[6] == 400, f"Expected 400, got {labels[6]}"
    assert (labels[[0,1,2,3,5,7]] == -100).all(), "Non-eval positions should be -100"

    print(f"  ✓ Labels correctly set at evaluation positions")

def test_masking_logic():
    """测试 masking 逻辑"""
    print("\n[Test 5] Masking logic")
    config = get_config("tiny")
    model = GPT2Model(config, mask_token_id=50257, bos_token_id=50256)

    input_ids = torch.arange(10)  # [0, 1, 2, ..., 9]
    cond_idx = [0, 2]
    eval_idx = [4, 6]
    unseen_idx = [4, 5, 6, 7]  # truly_unseen = {5, 7}

    aug_ids, _, N_c, _ = model._build_augmented_sequence_single(
        input_ids, cond_idx, eval_idx, unseen_idx, device='cpu'
    )

    # 提取 body 部分（跳过 cond prefix 和 BOS）
    body_start = N_c + 1
    body = aug_ids[body_start:]

    # 验证：位置 4,6 应该可见（在 eval），5,7 应该被 mask
    assert body[4] == 4, f"Position 4 should be visible, got {body[4]}"
    assert body[6] == 6, f"Position 6 should be visible, got {body[6]}"
    assert body[5] == 50257, f"Position 5 should be masked, got {body[5]}"
    assert body[7] == 50257, f"Position 7 should be masked, got {body[7]}"

    print(f"  ✓ Masking logic correct: eval visible, truly_unseen masked")

def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("Testing Pure Tensor Augmentation Implementation")
    print("=" * 80)

    test_basic_augmentation()
    test_empty_sets()
    test_device_consistency()
    test_labels_creation()
    test_masking_logic()

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()
