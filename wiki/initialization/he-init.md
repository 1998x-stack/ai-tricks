# He/Kaiming 初始化

## 是什么

He 初始化（也称 Kaiming 初始化）由 Kaiming He 等人在 2015 年的论文 *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* 中提出。它专门针对 ReLU 及其变体（LeakyReLU、PReLU）设计，解决了 Xavier 初始化在 ReLU 激活函数下失效的问题。

Xavier 初始化假设激活函数关于 0 对称且在 0 附近线性。但 ReLU 在负半轴输出恒为 0，导致**经过激活函数后大约一半的方差被丢弃**。如果直接用 Xavier 初始化，经过 ReLU 后实际输出的方差会减半，信号逐层衰减。He 初始化的推导通过补偿因子解决了这一问题。

直观推导如下：
- Xavier 假设 $\operatorname{Var}(y) = \operatorname{Var}(Wx) = n_{in} \operatorname{Var}(W) \operatorname{Var}(x)$
- 经过 ReLU 后，$x' = \max(0, y)$。由于 y 均值为 0 且对称，$\operatorname{Var}(x') \approx \frac{1}{2} \operatorname{Var}(y)$
- 为了保持 $\operatorname{Var}(x') = \operatorname{Var}(x)$，需要补偿回这个 1/2 因子
- 最终得到 $\operatorname{Var}(W) = \frac{2}{n_{in}}$

对比 Xavier 的 $\frac{2}{n_{in} + n_{out}}$，He 初始化有两个关键差异：
- **分子为 2 而非调和平均**：因为 ReLU 砍掉一半方差，需要翻倍补偿
- **分母只用 $n_{in}$**：反向传播时 ReLU 的梯度同样会砍去一半，但 He 的推导选择只保证前向稳定

对应两种分布实现：

- **均匀分布**: $W \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)$
- **正态分布**: $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$

对于 LeakyReLU，如果负斜率记为 $a$（通常 $a=0.01$），公式修正为：

$$
\operatorname{Var}(W) = \frac{2}{(1 + a^2) n_{in}}
$$

参见 [[../activation/relu]] 了解 ReLU 家族各变体的负半轴行为。

## 为什么重要

He 初始化是 ResNet 等现代深度 CNN 能成功训练的关键因素之一。152 层的 ResNet 能够在 ImageNet 上以简单 SGD 收敛，很大程度上归功于 He 初始化和 Batch Normalization 的组合。具体来说：

- **解决 ReLU 方差坍塌**：通过因子 2 补偿 ReLU 造成的信息损失，让深层网络前向和反向的激活值保持稳定。没有这一补偿，50 层以上的 ReLU 网络几乎必然梯度消失。
- **配合 BN 实现深层训练**：He 初始化 + Batch Normalization 的组合是 2015-2020 年间 CNN 训练的事实标准。BN 稳定训练过程中的分布偏移，He 初始化提供好的起点。
- **简单高效**：不需要逐层预训练、复杂的自适应策略，一行代码即可实现。这降低了深度学习研究者的实验成本。
- **理论价值**：He 初始化是"根据激活函数特性定制初始化"这一思路的代表作，启发了后续 GELU、Swish 等新型激活函数的初始化研究。

## 如何使用

He 初始化是 ReLU 及其变体的**标准初始化方法**。PyTorch 的 `nn.Linear` 和 `nn.Conv2d` 默认使用 Kaiming Uniform 初始化（`mode='fan_in'`, `nonlinearity='leaky_relu'`），因此在大多数场景下不需要手动设置。但了解正确的调用方式仍然重要：

```python
import torch.nn as nn

# 推荐用于 ReLU
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

# 推荐用于 LeakyReLU（自动根据默认负斜率 a=0.01 调整缩放）
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

# bias 全零
nn.init.zeros_(layer.bias)
```

**适用场景**：
- 使用 ReLU、LeakyReLU、PReLU、ELU 的卷积层和全连接层
- ResNet、DenseNet、MobileNet、EfficientNet 等现代 CNN 架构
- 深层网络（50 层以上）中任何使用 ReLU 系激活的线性变换

**不适用场景**：
- tanh 或 sigmoid 激活函数 — 对称激活函数用 Xavier 更优，见 [[xavier-init]]
- Embedding 层 — 推荐截断正态分布
- 需要对输出正交性做精细控制的场景 — 推荐正交初始化
- 无激活函数的线性层 — 此时用 Xavier 或简单的 $1/\sqrt{n}$ 即可

关于不同激活函数和初始化的匹配关系，参见 [[../activation/activation-selection]]。

## 常见误区

1. **以为 He 初始化只适用于 CNN**。实际上它适用于任何使用 ReLU 激活的线性变换层，包括 MLP、Transformer 中的 FFN 层、以及卷积层。关键因素是激活函数而非网络结构。
2. **忽略 mode 参数的选择**。`fan_in` 保持前向方差不变，`fan_out` 保持反向方差不变。CNN 中通常用 `fan_in`，因为输入通道数通常更多，方差估计更稳定。但如果某层反向传播的梯度更重要（如靠近输出层的分类头），可以考虑 `fan_out`。
3. **不知道 LeakyReLU 需要不同的缩放**。LeakyReLU 的负斜率 $a$ 保留了部分负半轴信息（$a=0.01$ 时保留了 0.01%），因此方差缩放因子应从 $2/n$ 修正为 $2/((1 + a^2)n)$。PyTorch 的 `nonlinearity='leaky_relu'` 选项会自动处理这一修正。
4. **认为 He 初始化可以替代 Batch Normalization**。两者是互补关系而非替代关系：He 初始化处理了第一阶的方差问题（初始信号传播），BN 处理了训练过程中的分布偏移（Internal Covariate Shift）。深层网络两者都需要。
5. **在新激活函数上滥用 He 初始化**。任何不对称的激活函数（如 Swish、GELU、Mish）的初始化方差都需要重新推导，不能盲目套用 $2/n$ 公式。盲目套用可能带来意料之外的梯度问题。参见 [[../../contradictory]] 中的相关讨论。
6. **在迁移学习中忽略初始化与预训练的冲突**。微调时如果手动用 He 初始化覆盖了预训练权重，会导致灾难性遗忘。加载预训练模型后不应再执行初始化。

## 参见

- [[initialization]] — 初始化方法总体概览
- [[xavier-init]] — Xavier/Glorot 初始化（tanh/sigmoid 专用）
- [[why-initialization-matters]] — 初始化为什么重要（理论背景）
- [[../activation/relu]] — ReLU 激活函数
- [[../activation/activation-selection]] — 激活函数选择指南
- [[../../contradictory]] — 初始化相关的争议与讨论
