# NaN 诊断

## 是什么

NaN（Not a Number）是深度学习训练中最常见的崩溃信号，表现为 loss 在某个时刻变为 `nan`。根据出现方式分为**突发型**（loss 正常训练到某一步突然跳变为 NaN）和**渐变型**（loss 逐渐增大直至溢出为 NaN）两种模式，对应不同的根因和诊断路径。

突发型 NaN 几乎是"坏样本 + 数值不稳定操作"的组合：某条样本包含无效值（NaN/inf），或者交叉熵中碰到了 `log(0)`、除以零、平方根负数。渐变型 NaN 本质是参数数值失控：梯度不停累积导致权重膨胀，最终某个前向或反向计算溢出。

---

## 为什么重要

NaN 是训练流程中最高优先级的阻断信号。不解决 NaN，后续所有调参、评估、实验对比都毫无意义。更重要的是，NaN 的两种模式指向完全不同的根因——用渐变型的方法（降学习率）处理突发型（坏样本），或者用突发型的方法（检查数据）处理梯度爆炸，都会浪费时间。

此外，某些场景下 NaN 被**隐藏性地"修复"**：梯度裁剪强行把 NaN 梯度截断到有限值，训练看似继续，实际参数已经不更新或已损坏。如果模型训完一轮评估结果稳定却偏低，NaN 可能是元凶。

---

## 如何使用

### 突发型 NaN 排查流程

出现 `loss = nan` 且此前 loss 正常：

1. **立即停止训练，固定当前 step**。回放日志，确认 NaN 首次出现的精确 step。
2. **用 `torch.isnan()` 检查输入数据**。在第一个 batch 加载后插入：
   ```python
   assert not torch.isnan(batch["input"]).any(), "输入含 NaN"
   ```
   常见原因：数据文件损坏、预处理除零、padding 后 mask 未正确处理。
3. **检查 loss 函数中的数值操作**。交叉熵的 `log(0)` 是最常见的 NaN 来源：
   ```python
   # 脆弱写法
   loss = -torch.log(probs[range(n), labels])
   # 稳健写法
   loss = -torch.log(probs[range(n), labels] + 1e-8)
   ```
   其他常见数值陷阱：`sqrt()` 的负输入、`1/x` 的零分母、`x - y` 因精度导致的极小负数开方。
4. **运行坏样本检测**。遍历数据集，用固定的模型 checkpoint 跑前向，记录所有产生 NaN 的样本 ID：
   ```python
   for i, batch in enumerate(dataloader):
       out = model(batch["input"])
       if torch.isnan(out).any():
           print(f"坏样本: batch {i}, index {batch['idx']}")
   ```
5. **定位精确操作**。用 `torch.autograd.detect_anomaly()` 包裹前向 + 反向传播：
   ```python
   with torch.autograd.detect_anomaly():
       loss = model(batch)
       loss.backward()
   ```
   PyTorch 会在检测到 NaN 梯度的第一个操作处抛出异常和堆栈信息。

### 渐变型 NaN 排查流程

loss 持续上升一段时间后变为 NaN：

1. **立即降低学习率**。将 lr 降到当前的 1/10 或 1/100，重新运行。如果 loss 恢复正常但收敛过慢，说明梯度爆炸与 lr 相关。
2. **检查学习率预热（warmup）**。无预热的训练初期梯度幅值巨大，容易在第一个 step 就推爆参数。确保前 5%-10% 的 step 使用线性预热。
3. **检查梯度范数**。插入梯度监控：
   ```python
   total_norm = 0.0
   for p in model.parameters():
       if p.grad is not None:
           total_norm += p.grad.norm().item() ** 2
   total_norm = total_norm ** 0.5
   print(f"grad norm: {total_norm:.4f}")
   ```
   正常梯度范数应在 0.01~10 范围。稳定增长超过 100 意味着梯度爆炸。
4. **确认是否有混合精度问题**。fp16/bf16 下梯度下溢也表现为 NaN，检查是否启用了 `GradScaler`。
5. **启用梯度裁剪作为临时修复**（详见 [[../debugging/gradient-clipping]]）。

### detect_anomaly 的最佳实践

- **不要在正常训练中长期开启**。`detect_anomaly` 会大幅降低训练速度并增加显存消耗，仅在排查时使用。
- **和梯度裁剪不兼容**。裁剪后的梯度不再反映真实回传路径，`detect_anomaly` 可能无法捕捉原始问题。先关掉裁剪再排查。
- **和混合精度配合**。在 `GradScaler` 的 `scale` 和 `unscale_` 之间使用时，注意 NaN 可能来自 loss 缩放而非模型本身。

### 梯度裁剪的使用边界

```python
# 按范数裁剪（推荐）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 按值裁剪（仅用于极端情况）
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

详见 [[../debugging/gradient-clipping]]。

---

## 常见误区

- **"NaN 出现了加梯度裁剪就行"**。裁剪掩盖而非修复问题。突发型 NaN 来自坏样本，裁剪无法让坏样本变好；渐变型 NaN 来自学习率或初始化，裁剪降低了有效学习率却让你以为问题已修复。
- **"fp16 的 NaN 是正常的，忽略就行"**。fp16 的梯度下溢确实是特性，但前向 NaN 始终是 bug。区分前向 NaN（模型输出含 NaN）和反向 NaN（仅梯度 NaN，前向正常）。
- **"detect_anomaly 应该默认开启"**。它在正常训练中产生约 2-3 倍开销和额外显存，且与某些算子不兼容。仅在排查阶段短时启用。
- **"只有 loss 变成 NaN 才需要关注"**。有些模型在前向中已产生 NaN（分类概率全为 NaN）但 loss 恰好被兜底为有限值，表现为"训了但没效果"。定期检查模型输出是否含 NaN 是一个容易被忽略但十分有效的检查。
- **"NaN 都是代码问题，和硬件无关"**。极少情况下 GPU 硬件故障（如显存位错）会产生随机 NaN。如果在相同 seed 下每次 NaN 出现的 step 不同，排除法和 TensorCore 一致性检查不可跳过。固定 seed 复现 NaN 比无 seed 随机出现要好诊断得多。

---

## 参见

- [[../debugging/small-data-overfit-test]] — 在小数据上不能过拟合时，NaN 是常见原因
- [[../debugging/gradient-clipping]] — 裁剪的详细使用和阈值选择
- [[../debugging/training-curve-diagnosis]] — loss 曲线的 NaN 模式识别
- [[../optimizer-lr/optimizer-lr]] — 学习率与梯度爆炸的关系
- [[../regularization/overfitting-underfitting]] — 梯度爆炸导致的欠拟合
- [[../data-eval/data-eval]] — 坏样本检测与数据清洗
- [[../../contradictory]] — 不同调试建议的矛盾之处
