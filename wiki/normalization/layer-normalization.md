# Layer Normalization

## 是什么

Layer Normalization（LN）于 2016 年由 Jimmy Lei Ba 等人提出，作为 BN 在序列模型上的替代方案。LN 在 **特征维度** 对单个样本的所有神经元做标准化，完全独立于 batch 维度：

$$
\mu = \frac{1}{H} \sum_{i=1}^{H} x_i, \quad
\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2
$$

其中 $H$ 是当前层的隐藏单元数。LN 同样使用可学习的 $\gamma$ 和 $\beta$ 恢复表征能力。

与 BN 的关键区别：LN 对每个样本单独计算统计量，训练和推理行为完全一致，不存在 BN 的"训练/推理模式转换"问题。

### 在 Transformer 中的位置

现代 Transformer（GPT、BERT、Llama）的标准配置：

- **Post-LN（原始）**：残差连接之后加 LN（LayerNorm → 残差聚合）。
- **Pre-LN（主流）**：每个子层之前加 LN，残差连接之后不加。训练更稳定，无需 warmup 也能收敛。
- **RMS Norm（Llama/GPT-NeoX）**：去掉均值中心化，仅做均方根缩放 $\text{RMS}(x) = \sqrt{\frac{1}{n} \sum x_i^2}$，计算更快且效果相当。

LN 之所以在 Transformer 中不可替代，是因为它解决了两个关键问题：
1. **变长序列**：不同样本的序列长度不同，BN 无法在 batch 维度上做有意义的统计。
2. **固定参数分布**：Self-Attention 和 FFN 的激活值分布各不相同，LN 在每个子层后重新规整分布。

## 为什么重要

- **Transformer 的基石**：没有 LN，深层 Transformer 无法稳定训练。Pre-LN 是让 100+ 层 Transformer 能训练的必备设计。
- **batch 无关性**：训练和推理行为一致，不需要 running statistics。batch size = 1 时效果不变。
- **序列友好**：适合变长输入和自回归生成（逐 token 推理时 BN 不可行，LN 无此问题）。
- **稳定梯度流**：Pre-LN 在残差网络中提供梯度 highway，缓解梯度消失。

## 如何使用

| 架构 | 推荐归一化 | 原因 |
|------|-----------|------|
| Transformer Encoder (BERT) | LN (Pre-LN 或 Post-LN) | 序列特征 + 深层堆叠 |
| Transformer Decoder (GPT) | LN (Pre-LN + RMS Norm) | 自回归生成，逐 token 推理 |
| LSTM/RNN | LN | 变长序列，时间步共享参数 |
| Policy Network (RL) | LN | batch size 小或在线学习 |
| Vision Transformer (ViT) | LN (遵循 Transformer 原始设计) | patch 序列的 Transformer 范式 |

- **默认使用 Pre-LN**：比 Post-LN 更稳定，允许更少 warmup 步数。
- **RMS Norm**：在 7B+ 大模型中逐渐替代完整 LN，提速约 15-30% 无精度损失。
- **对 weight decay 敏感**：LN 的 $\gamma$ 和 $\beta$ 通常不做 weight decay 或单独设置很小的 decay。

### LN vs BN 对比表

| 维度 | Layer Normalization | Batch Normalization |
|------|-------------------|-------------------|
| 归一化方向 | 特征维度（每个样本独立） | batch 维度（跨样本） |
| batch 依赖 | 完全独立 | 依赖 batch 统计量 |
| 小 batch | 效果不变 | 不稳定（< 8 时退化） |
| 训练/推理一致性 | 一致，无 running stats | 不同，需 running_mean/var |
| RNN/Transformer | 最佳选择 | 不适用 |
| CNN | 可用但不如 BN | 最佳选择 |
| 参数量 | 2C（gamma + beta） | 4C（gamma + beta + running_mean + running_var） |
| 计算开销 | O(N × H) | O(N × C × H × W) |
| 推荐学习率 | 正常 | 可比无 BN 时高 5-10 倍 |

## 常见误区

- **LN 可以替代 BN 用于 CNN**：LN 在 CNN 中效果通常不如 BN，因为 CNN 的激活模式依赖局部感受野，LN 对整个通道层做归一化破坏了通道间的相对关系。
- **Post-LN 和 Pre-LN 可以混用**：两者计算相同，只是位置不同。混用不会提升效果，坚持一种模式即可。
- **LN 不需要 running statistics**：正确。LN 训练推理行为一致，不需要滑动平均统计量。
- **LN 本身没有正则化效果**：相比 BN 的 mini-batch 噪声带来的正则化，LN 是确定性计算，正则化效果较弱。需要在 Transformer 中配合 Dropout 使用。
- **RMS Norm 和 LN 完全等价**：RMS Norm 去掉了均值中心化，保留了缩放变换。对 Transformer 训练影响极小但计算更快。

## 参见

- [[../normalization/normalization]] — 归一化层总览与核心技巧
- [[batch-normalization]] — CNN 中的归一化选择与 BN 原理
- [[group-normalization]] — 小 batch 场景及视觉任务替代方案
- [[../llm-sft/llm-sft]] — 大语言模型微调中的 LN 配置
- [[../nlp/nlp]] — NLP 任务中 LN 的统治地位
- [[../regularization/dropout]] — Transformer 中 LN + Dropout 配合
- [[../../contradictory]]
