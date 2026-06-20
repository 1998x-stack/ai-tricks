# Batch Normalization

## 是什么

Batch Normalization（BN）于 2015 年由 Google 提出，是深度学习中最具影响力的归一化方法。BN 在 **batch 维度**上对每个神经元做标准化：对每个 mini-batch 计算均值和方差，将激活值拉回到零均值单位方差，再通过可学习的仿射变换（gamma/beta）恢复表达能力。

### 核心公式

$$
\hat{x}_i = \frac{x_i - \mu_{\text{batch}}}{\sqrt{\sigma_{\text{batch}}^2 + \epsilon}}
\quad
y_i = \gamma \hat{x}_i + \beta
$$

其中 $\mu_{\text{batch}}$ 和 $\sigma_{\text{batch}}^2$ 是当前 mini-batch 在 **(N, H, W)** 维度上的统计量（对全连接层则是 N 维度），$\gamma$ 和 $\beta$ 是形状为 `[C]` 的可学习参数。

### 训练 vs 推理

| 阶段 | 统计量来源 | 行为差异 |
|------|-----------|---------|
| **训练** | 当前 mini-batch 的均值和方差 | 梯度流经归一化层，统计量随 batch 变化 |
| **推理** | 训练期间累计的滑动平均（running mean/var） | 固定统计量，不依赖 batch 大小，确定性输出 |

推理时必须切换到 running statistics，否则单样本推理时 BN 退化为恒等映射（batch size=1 时方差为 0，数值不稳定）。

## 为什么重要

- **加速收敛**：缓解内部协变量偏移（Internal Covariate Shift），允许使用更大的学习率，训练速度可提升 2-10 倍。
- **轻微正则化**：每个 mini-batch 的随机统计量引入噪声，起到正则化效果，实验表明可部分替代 Dropout。
- **缓解梯度问题**：防止激活值进入饱和区（如 Sigmoid 的高平区域），改善梯度流动。
- **降低对初始化敏感度**：即使初始化不理想，BN 也能快速将激活拉回有效范围。

## 如何使用

- **默认位置**：卷积层之后、激活函数之前（Conv → BN → ReLU 是标准模式）。
- **CNN 架构**：几乎所有现代 CNN（ResNet、DenseNet、EfficientNet）都内置 BN，不建议移除。
- **大 batch**：batch size >= 16 时 BN 效果稳健。推荐 batch size 32-256。
- **学习率调整**：使用 BN 后可尝试将初始学习率提高 5-10 倍（如从 0.01 提到 0.1）。
- **分布式训练**：多卡训练中使用 SyncBN 跨卡同步统计量（PyTorch: `torch.nn.SyncBatchNorm`，TensorFlow: `tf.keras.layers.experimental.SyncBatchNormalization`）。

### BN + Dropout 交互

两者可以共存但有约束：

- **先有 BN 再有 Dropout**：BN 的方差估计在推理时是固定的，Dropout 在推理时关闭。如果 BN 前加 Dropout，训练时 BN 统计量受 Dropout mask 扰动，推理时 Dropout 关闭导致统计量不匹配。
- **BN 已提供正则化**：CNN 中使用 BN 后，全连接层上的 Dropout 可省略或降为 0.2-0.3。
- **过拟合严重时**：可以同时使用 BN + Dropout，但 Dropout rate 不宜过大（0.3-0.5），且放在 BN 之后。

### 代码示例（PyTorch）

```python
self.block = nn.Sequential(
    nn.Conv2d(in_c, out_c, 3, padding=1),
    nn.BatchNorm2d(out_c),      # Conv → BN → ReLU
    nn.ReLU(inplace=True),
)
```

## 常见误区

- **小 batch 不能用 BN**：batch size < 8 时 BN 统计量噪声过大，训练不稳定。此时换 GN 或 LN。
- **BN 在 RNN/Transformer 中效果差**：序列长度变化导致 batch 统计量不稳定，且 RNN 时间步共享参数时 BN 维度不匹配。
- **推理时忘记切换模式**：`model.eval()` 切换到 running statistics，`model.train()` 回训练模式。漏掉此步会导致推理结果错误。
- **BN 不是万能正则化**：在小数据集或极端过拟合场景，BN 的正则化强度不够，仍需 Dropout/Weight Decay。
- **BN 与权重归一化（WeightNorm）可互换**：BN 的归一化方向在 batch 维度，WeightNorm 在权重方向，两者解决不同问题，不可直接替代。

## 参见

- [[../normalization/normalization]] — 归一化层总览与核心技巧
- [[layer-normalization]] — Transformer 中的归一化选择
- [[group-normalization]] — 小 batch 场景替代方案
- [[../regularization/dropout]] — Dropout 原理与 BN 的交互
- [[../cnn-cv/cnn-cv]] — CNN 架构中的标准配置
- [[../initialization/initialization]] — 初始化与 BN 的协作
- [[../../contradictory]]
