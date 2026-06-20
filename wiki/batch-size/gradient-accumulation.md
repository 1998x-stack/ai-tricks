# 梯度累积

## 是什么

梯度累积（Gradient Accumulation）是一种在不增加显存占用的情况下，通过多次前向-反向传播累积梯度后统一更新参数的技术，等效于使用更大的批大小进行训练。

核心机制：

1. 每个小批次（micro-batch）正常做 forward + backward，计算梯度
2. 将梯度累积到 `.grad` 中，不清零
3. 每累积 N 步后执行 `optimizer.step()` 更新参数
4. 更新后调用 `optimizer.zero_grad()` 清零梯度

关键细节：`loss` 必须除以累积步数 N，否则梯度量级会放大 N 倍，破坏训练稳定性。

有效批大小计算公式：

```
effective_batch_size = per_device_batch_size × num_gpus × accumulation_steps
```

这就是 [[../batch-size/batch-size]] 中 BugBuster喵 强调的必须记录完整的有效批大小公式——很多实验无法复现就是因为这里没说清。

## 为什么重要

梯度累积是当代深度学习训练的基础技术，主要原因：

- **突破显存墙**：单卡显存有限（A100 80GB / H100 80GB），大批次（如 4096）无法单次装入，梯度累积是唯一的等效方案。
- **等效批大小扩展**：在对比学习、RLHF/PPO 等场景中，模型对批大小有硬性要求（太小不收敛），梯度累积直接决定训练能否成功。
- **多卡训练的标配**：分布式数据并行（DDP/FSDP）中，每卡计算 gradient accumulation steps 次本地梯度，然后 all-reduce 聚合。这也是 [[../batch-size/batch-size]] 中阿里云提到的 "总批量大小是单个 GPU 上的批量大小乘以 GPU 的数量" 背后的实现机制。
- **实验复现的关键信息**：论文和开源仓库常只说 "batch size = 1024"，但实际由 4 卡 x 8 累积 x 32 单卡 batch 构成。不记录完整公式会导致严重的可复现性问题。

## 如何使用

### PyTorch 标准实现

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps  # 关键：除以累积步数
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 注意事项

- **Batch Normalization**：累积模式下 BN 的均值和方差基于 micro-batch 而非 effective batch 计算，可能引入统计量偏差。可改用 SyncBN（同步所有卡的 BN 统计量）或 LayerNorm / GroupNorm。
- **学习率不变**：梯度累积扩大的是等效批大小，学习率不需要对应调整。如需调整，参见 [[batch-size-lr-scaling]]。
- **Loss 缩放**：`loss / accumulation_steps` 确保每个 micro-batch 对最终参数更新的贡献等价于在 effective batch 中增加一个样本。
- **梯度裁剪**：在 `optimizer.step()` 前执行，基于累积后的完整梯度做裁剪，而不是在每次 micro-batch 后裁剪。
- **混合精度（AMP）**：GradScaler 需在 accumulation 循环外 step，确保缩放正确。同时注意 micro-batch 的 loss 溢出检测——单个 micro-batch 的 loss 可能因除以累积步数而过小。

### 何时使用

- 显存 OOM，但需要更大等效批大小
- 对比学习/表示学习（SimCLR、MoCo、CLIP）——需要大量负样本避免模型坍塌
- PPO / GRPO / RLHF 训练——梯度稳定性要求大批次（参见 [[../batch-size/batch-size]] 中 Tang AI 的观点）
- 推荐系统 CTR 模型——批大小在 32-512 区间内有效

### 何时避免

- 使用 BatchNorm 且无法替换为 SyncBN 时，micro-batch 太小（如 1-2）会导致 BN 统计量严重偏差
- 时间敏感的训练：梯度累积会降低 wall-clock 训练速度（前向+反向多次后才更新一次）

## 常见误区

- **loss 不除以累积步数**：最常见错误。不除则梯度放大 N 倍，轻则训练不稳定，重则直接发散。
- **BN 统计量问题被忽略**：micro-batch 上的 BN 统计量和 effective batch 上的 BN 统计量不同，在大幅度梯度累积时不可忽略。
- **认为梯度累积和增大 batch size 完全等价**：不完全等价。BN 行为不同，优化轨迹也不同（梯度是 N 次独立前向的累积而非一次大前向），有些场景中效果不如直接增大 batch size。
- **记录不完整**：实验记录只写 "batch size = 1024"，不写 "per_device=32, gpus=4, accum=8"，导致不可复现。
- **学习率盲目跟随**：梯度累积等效扩大批大小不需要同步调大学习率。学习率调整需遵循线性缩放规则。

## 参见

- [[../batch-size/batch-size]] — 批大小核心概念与技巧集合
- [[batch-size-generalization]] — 批大小对泛化能力的影响
- [[batch-size-lr-scaling]] — 批大小与学习率的缩放关系
- [[../multi-gpu]] — 多卡分布式训练
- [[../../contradictory]]
