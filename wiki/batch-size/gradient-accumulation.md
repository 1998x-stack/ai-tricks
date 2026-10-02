# 梯度累积

## 是什么

梯度累积在多个 microbatch 上依次执行 forward/backward，再统一更新参数。它减少单次需要保留的激活显存，**不会保证全部显存不增加**。来源：原素材中 LindgeWAI 的实现建议与 BugBuster喵 的 effective batch 记录建议。

对于可逐样本分解的损失，第 j 个 microbatch 有 n_j 个有效监督单位、平均损失 L_j，整组目标为：

`L = sum(n_j * L_j) / sum(n_j)`。

监督单位可以是图像、有效回复 token 或未忽略的像素，必须与损失的 reduction 定义一致。只有 n_j 相同时，才能直接平均各 microbatch 的 mean loss；尾组应使用实际大小。

完整且各卡等大的批次可记录 `effective_batch = micro_batch × world_size × accumulation_steps`；尾组须另外记录真实样本/token 数。

## 为什么重要

- 显存不足时，以更多次前后向换取较少的单次激活存储。
- 将 loss 归一化、裁剪和 optimizer/scheduler 的调用时点统一到参数更新。
- 将普通累积与对比学习负样本池分开记录，防止实验解释错误。

## 如何使用

以下单卡分类示例按真实样本数归一化，包含 epoch 尾组。输入组暂存在 CPU，只把当前 microbatch 放上设备。

```python
from itertools import islice
import torch
import torch.nn.functional as F

accumulation_steps = 4
iterator = iter(dataloader)
while group := list(islice(iterator, accumulation_steps)):
    sample_count = sum(len(labels) for _, labels in group)
    optimizer.zero_grad(set_to_none=True)
    for inputs, labels in group:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="sum") / sample_count
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    # 若 scheduler 的单位是 optimizer step，在此调用 scheduler.step()。
```

变长语言模型使用 `labels[:, 1:] != -100` 统计因果预测的有效 token，按该数量给各 microbatch 的 mean loss 加权。像素 ignore_index、类别权重会改变分母，需要按具体损失定义处理，不能一律除以 batch 长度。

AMP 时先逐次 `scaler.scale(loss).backward()`，整组完成后只 `unscale_` 一次，再裁剪、`scaler.step()` 与 `scaler.update()`；参考 [PyTorch AMP 梯度累积说明](https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)。

A/B 应记录 microbatch、实际有效样本/token、更新步数、峰值显存和耗时。比较“大 batch 与等效累积”时，固定 effective batch 和样本顺序；比较“更大 effective batch”时则把更新次数变化明确写入解释。

## 常见误区

- **普通累积自动扩大对比学习负样本池**：错误。若每次 InfoNCE 只看到当前 microbatch，多个小 softmax 不等于一次大 softmax。Gao 等的[梯度缓存方法](https://arxiv.org/abs/2101.06983)另行处理跨样本依赖。
- **任何情况下都除以固定累积步数**：尾组少于该步数、有效 token 不等长时会缩放错误；以上代码按实际总量计算。
- **不处理尾组**：仅在 `(i + 1) % N == 0` 时更新会丢失尾部梯度。
- **SyncBN 解决跨累积步的 BN 差异**：SyncBN 同步同一前向中的各卡统计量，不合并不同时刻的 microbatch。BN、Dropout 和浮点舍入都会影响与单次大 batch 的一致性。
- **累积改变就必须/不必调学习率**：固定有效批量的实现替换与扩大有效批量是不同实验；学习率调整应另设对照。
- **DDP 自动只在末次通信**：通常需按框架要求使用 `no_sync()`，并保证所有 rank 的步数与全局损失归一化一致。

## 参见

- [[batch-size]] — 来源技巧与有效批大小
- [[batch-size-lr-scaling]] — 批大小与学习率
- [[../infra/distributed-training]] — 多卡训练
- [[../debugging/gradient-clipping]] — 累积完成后裁剪
- [对应实验目录](../../experiments/batch-size/README.md) — 协议与实际驱动覆盖状态
