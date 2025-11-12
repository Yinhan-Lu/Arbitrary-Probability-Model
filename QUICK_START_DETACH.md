# Quick Start: Detach Augmentation Feature

## 🎯 一句话总结

在训练脚本顶部设置 `DETACH_AUGMENTATION="true"` 来阻止梯度流过augmentation操作，测试是否能解决Mode 2性能反转问题。

---

## 📝 快速使用

### 1. 修改训练脚本

打开 `scripts/submit_conditional_moderate_cond.sh`，找到第72行：

```bash
# === AUGMENTATION DETACH OPTION (for debugging gradient flow) ===
DETACH_AUGMENTATION="false"  # Change to "true" to enable detach
```

**修改为**：

```bash
DETACH_AUGMENTATION="true"  # 启用detach，阻止梯度流
```

### 2. 提交训练

```bash
sbatch scripts/submit_conditional_moderate_cond.sh
```

### 3. 对比实验

建议同时运行两个实验：

**实验A（默认行为）**：
```bash
# 保持 DETACH_AUGMENTATION="false"
sbatch scripts/submit_conditional_moderate_cond.sh
```

**实验B（detach模式）**：
```bash
# 设置 DETACH_AUGMENTATION="true"
sbatch scripts/submit_conditional_moderate_cond.sh
```

---

## 🔬 预期结果

### 如果假设正确

| 实验 | DETACH | Mode 2 PPL | 说明 |
|------|--------|-----------|------|
| Legacy | N/A | ~120 | 基准（external aug） |
| 实验A | false | ~7 | 梯度流artifact |
| 实验B | true | ~120 | 修复，接近legacy |

### 如果假设错误

- 实验B的Mode 2依然是~7（或者~120）
- 说明问题在其他地方

---

## 📊 如何查看结果

训练完成后，查看最终评估结果：

```bash
# 找到最新的实验文件夹
ls -lt experiments/

# 查看Mode 2的ppl
cat experiments/conditional_moderate_cond_*/logs/metrics.csv | grep mode2
```

或者用可视化工具：

```bash
python utils/quickstart_visualization.py experiments/conditional_moderate_cond_*
```

---

## 🧪 测试功能是否正常

在提交大型训练之前，先测试功能：

```bash
# 快速测试（5秒）
python tests/test_detach_augmentation.py

# 应该看到：
# ✅ ALL TESTS PASSED
```

---

## 🔧 详细文档

更多信息请查看：
- **完整指南**：`DETACH_AUGMENTATION_GUIDE.md`
- **测试说明**：`tests/README.md`
- **Debug脚本**：`tests/debug_mode2_*.py`

---

## 💡 背景

- **问题**：Legacy pipeline中Mode 2 ppl=120（最差），New pipeline中Mode 2 ppl=7（最好）
- **假设**：New pipeline的internal augmentation允许梯度流过augmentation操作，导致模型学习到不同的东西
- **解决方案**：添加`.detach()`阻止梯度流，让internal augmentation行为类似external augmentation

---

## ❓ 常见问题

**Q: 会影响训练速度吗？**
A: 不会，detach是轻量级操作

**Q: 会影响其他模式吗？**
A: 可能会，但主要关注Mode 2

**Q: 如果两个实验结果都一样怎么办？**
A: 说明假设错误，需要调查其他原因（random seed, padding, etc.）

**Q: 可以在训练中途改变这个设置吗？**
A: 不行，这是训练时参数，需要重新训练

---

**Last Updated**: 2025-01-12
**实验状态**: 待验证
