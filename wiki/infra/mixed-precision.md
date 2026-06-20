# 混合精度训练

## 是什么

混合精度训练（Mixed Precision Training）是指在深度学习训练过程中同时使用半精度浮点数（FP16 或 BF16）和单精度浮点数（FP32）的技术。核心思路是将大部分计算和存储下放到半精度，仅在必要时保留 FP32 精度，从而在不显著影响模型收敛的前提下降低显存占用并提升训练速度。

PyTorch 从 1.6 开始原生支持 `torch.cuda.amp`（Automatic Mixed Precision），提供 `GradScaler` 自动管理 loss scaling 和精度转换。第三方库 `apex.amp` 是 NVIDIA 的早期方案，提供 `O0-O3` 四个优化等级。

## 为什么重要

- **显存减半**：FP16/BF16 的存储占用量是 FP32 的一半，可有效扩大 batch size 或模型规模。
- **计算加速**：支持 Tensor Core 的 GPU（Volta 架构及以上）在半精度下吞吐量可达 FP32 的 2-8 倍。
- **端到端效果接近**：只要正确使用 loss scaling（FP16）或直接使用 BF16（Ampere 及以上），模型精度损失通常可以忽略。

## 如何使用

### PyTorch 原生 AMP（推荐）

```python
scaler = torch.cuda.amp.GradScaler()   # 仅 FP16 需要，BF16 不需要

for data, target in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(dtype=torch.float16):  # 或 bfloat16
        output = model(data)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()       # 缩放 loss 防止 FP16 下溢
    scaler.step(optimizer)              # unscale + step
    scaler.update()                     # 动态调整 loss scale
```

### BF16 vs FP16 选择

| 特性 | FP16 | BF16 |
|------|------|------|
| 硬件支持 | 从 Volta (V100) 开始 | 从 Ampere (A100, 3090) 开始 |
| 指数位 | 5 位 | 8 位（同 FP32） |
| 表示范围 | 窄，易溢出 | 与 FP32 相同，不易溢出 |
| Loss scaling | 必须 | 通常不需要 |
| 精度损失 | 需小心处理 | 更安全 |

**经验法则**：硬件支持 BF16 时优先使用 BF16。FP16 的 loss scaling 和溢出问题有时会让你误判参数效果——明明参数没问题，但 loss scale 爆炸导致训练崩溃。

### apex.amp（旧项目兼容）

```python
from apex import amp
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

`opt_level` 从 `O0`（纯 FP32）到 `O3`（纯 FP16），常用 `O1`（自动混合）。

## 硬件支持速查

| GPU 架构 | FP16 Tensor Core | BF16 Tensor Core | 代表卡 |
|----------|-----------------|------------------|--------|
| Volta    | V               | x                | V100, Titan V |
| Turing   | V               | x                | RTX 20xx, T4 |
| Ampere   | V               | V                | A100, RTX 30xx, A30 |
| Hopper   | V               | V                | H100, H200 |
| Blackwell| V               | V                | B100, B200 |

注意：即使不支持 Tensor Core，FP16 的存储减半效果仍然存在（Pascal 卡训练时显存确实能省一半），但计算不会变快。`torch.cuda.amp` 会自动检测硬件能力选择最佳策略。

## 常见误区

1. **BF16 不需要 loss scaling 就能无脑用**：BF16 确实没有 FP16 的溢出问题，但精度仍然比 FP32 低，少数对精度极度敏感的操作（如 loss 计算、softmax）仍建议留在 FP32 域中。`autocast` 会自动处理这些。

2. **混合精度降精度**：正常使用下混合精度不会导致模型精度明显下降。如果出现精度损失，优先检查：(a) loss scale 是否频繁为 0；(b) 是否有操作被错误地放在了 autocast 作用域外；(c) 模型中有没有数值范围极大的操作（如某些自定义 loss）。

3. **FP16 的 loss scale 越大越好**：不是。`GradScaler` 会自动调整 scale 因子——频繁出现 inf/nan 会减小，连续稳定会增大。手动设一个极大的初始 scale 不会带来好处，反而会延迟对梯度爆炸的响应。

4. **所有 GPU 都支持混合精度加速**：只有 Volta 架构（V100）及以上才支持 Tensor Core 的 FP16 加速。Pascal（P100）及更早的 GPU 虽然可以跑 FP16，但计算不会加速，有时甚至更慢。在购买云 GPU 实例前应确认架构支持情况。

5. **混合精度只在训练时有用**：推理时也可以使用半精度。`model.half()` 或 `torch.autocast()` 在推理时同样可以加速和减少显存，且通常不需要 loss scaling。对于推理部署，量化（INT8/INT4）往往比半精度更激进。

## 参见

- [[../batch-size/gradient-accumulation]] —— 梯度累积与混合精度配合使用，注意累积梯度时 loss scaling 的影响
- [[../optimizer-lr/optimizer-lr]] —— 混合精度不影响学习率和优化器选择，但 loss scale 与优化器状态更新顺序需要留意
- [[../../contradictory]] —— 知识矛盾：社区对 FP16 vs BF16 的优先级存在争议
