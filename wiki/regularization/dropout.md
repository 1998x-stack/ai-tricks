# Dropout

## 是什么

Dropout 是一种在训练过程中随机"丢弃"（置零）部分神经元及其连接的正则化技术。训练时每个神经元以概率 p 被保留（或以概率 1-p 被丢弃），每次前向传播采样子网络不同；推理时所有神经元参与计算，权重乘以 p 以保持输出期望一致（Inverted Dropout 由主流框架自动处理）。

### Dropout 机制分解

1. **训练阶段**：对每层输出 $h$ 应用掩码 $m \sim \text{Bernoulli}(p)$，得到 $\tilde{h} = h \odot m / p$（除以 p 是为了保持训练时输出期望与推理时一致）
2. **推理阶段**：直接使用 $h$，不做任何丢弃——此时 $h$ 的期望与训练时 $\tilde{h}$ 的期望相同

### 变体

- **SpatialDropout**：整通道丢弃而非单神经元丢弃，适用于卷积层。当多个卷积层共享同一输入时，输出特征之间的相关性高，整通道丢弃能更有效地打破相关性和共适应
- **Monte Carlo Dropout**：推理时保留 Dropout，多次前向传播取均值和方差，可作为贝叶斯近似的廉价替代

## 为什么重要

- **打破共适应（Co-adaptation）**：每个神经元不能依赖其他特定神经元存活，被迫独立学习有用特征，所学到的表示更加鲁棒和分散
- **隐式 Ensemble**：每次迭代相当于采样一个不同的子网络，推理时所有子网络做平均等价于多个模型的集成，但集成成本几乎为零。这是 Dropout 最精妙之处——以极低代价获得集成效果，参见 [[regularization]]
- **泛化差距的有力工具**：当训练 loss 远低于验证 loss 时，Dropout 是缩小差距最高效的手段之一，仅次于数据增强
- **实现简单、介入成本低**：只需在目标层后加一行代码，无需修改训练流程或损失函数

## 如何使用

### Dropout rate 选择

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 全连接层（通用） | 0.2~0.5 | 不确定时选 0.5，这是经验上最鲁棒的取值 |
| CNN 卷积层 | 0.1~0.3 | 卷积层参数共享，不易过拟合，过高的 rate 反而有害 |
| 预训练模型微调 | 0~0.1 | 预训练权重已高度适应，大幅 dropout 破坏已学特征。有时 reset 到 0 有奇效 |
| LoRA / Adapter | 0~0.1 | 参数高效微调本身已经足够正则化 |
| RNN | 0.2~0.3（只对非循环连接） | 对循环连接加 dropout 会破坏序列记忆能力 |

### Dropout 在模型中的位置

- **全连接层**：Dropout 主要用武之地，通常放在激活函数之后、下一层之前
- **CNN**：普通 Dropout 对卷积层效果有限，推荐 SpatialDropout，尤其在多路特征融合（concatenation）前的低层特征图
- **Transformer**：在 attention 输出和 FFN 输出后加 Dropout，但通常 rate 设得很低（0~0.1），pre-LN 结构本身已有正则化效果
- **Embedding 层**：过拟合严重时，对 Embedding 层加 Dropout 也有奇效——[[../../contradictory]]

### 训练 vs 推理的注意事项

- 训练后推理必须切换到 eval 模式：`model.eval()`，否则 Dropout 会继续随机丢弃，导致输出不稳定
- 推理时做 batch prediction 也要确保 eval 模式
- 若使用 Monte Carlo Dropout 做不确定性估计，则需在推理时手动启用 Dropout

## Dropout 与 BN 的交互

[[../normalization/normalization]] 中的 BN 层本身具有轻微的正则化效果（通过 mini-batch 统计量的噪声）。Dropout 与 BN 的交互是一个长期争议的话题：

- **有 BN 时可以少加 Dropout**：BN 的随机性已提供一定的正则化，全连接层不一定需要额外加 Dropout
- **过拟合越严重，两者叠加越有效**：严重过拟合时，Dropout + BN 加的位置越多效果越好，甚至可以对 Embedding 层加
- **CNN vs RNN 取舍**：CV 任务建议优先加 BN，RNN/序列任务用 LN（因为 BN 对序列长度敏感）
- **小 batch 问题**：BN 在小 batch 下不稳定，此时考虑 GN（Group Normalization）替代，同时 Dropout 仍可正常使用

## 常见误区

1. **一过拟合就猛加 Dropout**：过拟合时的排查顺序应为"数据量 → 标签质量 → weight decay → 早停 → Dropout"。不要跳过前面的检查直接把 Dropout 当首要调节手段
2. **预训练模型 Dropout 越大越好**：预训练权重已经在大数据上学到鲁棒特征，大幅 dropout 反而破坏这些特征，导致欠拟合
3. **Dropout 与 BN 叠加无副作用**：两者都引入随机性，叠加过多可能导致训练不稳定、收敛变慢
4. **推理时忘记关闭 Dropout**：最常见的低级错误，会导致推理结果异常且难以定位
5. **所有层用相同的 Dropout rate**：低层和高层对 Dropout 的敏感度不同，应分别调节
6. **对 RNN 循环连接也加 Dropout**：这会破坏 RNN 的时间序列记忆能力，只应对非循环连接加 dropout

## 什么时候应该跳过 Dropout

- 模型已经高度正则化（预训练 Transformer、LoRA 微调）
- 数据集非常小且已经用了强数据增强
- 使用了大 weight decay 且效果已经很好
- BN 的正则化效果已经足够，模型不过拟合
- 训练时发现 Dropout 导致收敛明显变慢或不稳定

## 参见

- [[regularization]] — 正则化总论与超参数速查
- [[../normalization/normalization]] — BN/LN/GN 的正则化效果与选择
- [[../data-eval/data-eval]] — 过拟合诊断与训练代码验证
- [[../optimizer-lr/optimizer-lr]] — 学习率与正则化的配合
- [[../../contradictory]] — Dropout 是否必要的争议观点
