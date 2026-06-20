# Xavier/Glorot 初始化

## 是什么

Xavier 初始化（也称 Glorot 初始化）由 Xavier Glorot 和 Yoshua Bengio 在 2010 年的论文 *Understanding the difficulty of training deep feedforward neural networks* 中提出。其核心思想是：**前向传播时各层的激活值方差保持一致，反向传播时各层的梯度方差保持一致**。如果每层输出的方差逐层放大或缩小，深层网络的信号会迅速爆炸或消失。

公式推导的起点是假设输入 $x$ 和权重 $W$ 均独立同分布且均值为 0。对线性层 $y = Wx$，有：

$$
\operatorname{Var}(y) = n_{in} \cdot \operatorname{Var}(W) \cdot \operatorname{Var}(x)
$$

要保持 $\operatorname{Var}(y) = \operatorname{Var}(x)$，需要 $\operatorname{Var}(W) = 1/n_{in}$。反向传播同理，需要 $\operatorname{Var}(W) = 1/n_{out}$。但 $n_{in}$ 和 $n_{out}$ 通常不相等，于是取两者的调和平均：

$$
\operatorname{Var}(W) = \frac{2}{n_{in} + n_{out}}
$$

调和平均的选择并不是唯一的解，但经验上它同时兼顾了前向和反向的方差需求。Glorot 在论文中通过实验验证了这一选择的稳定性。推导中的关键假设包括：
- 权重和输入相互独立
- 激活函数在 0 附近近似线性（tanh、sigmoid 的饱和区远偏离 0，但 0 附近可近似为 $f(x) \approx x$）
- 权重初始化时均值为 0

对应两种分布实现：

- **均匀分布**: $W \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$
- **正态分布**: $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$

均匀分布的边界 $\sqrt{3\sigma^2}$ 来源于均匀分布方差 $\sigma^2 = (b-a)^2/12$ 的反推。正态分布直接使用 $\sigma^2$ 的平方根。

## 为什么重要

Xavier 初始化是深度学习走向工程化的重要里程碑。在该方法提出之前，深层网络的训练极度依赖经验性的手动调参和逐层无监督预训练（如 DBN 的逐层 RBM 预训练）。Xavier 初始化让**随机初始化 + 有监督训练**可以直接训练 5 层以上的网络，无需预训练。具体来说：

- **方差守恒**：保证信息在前向和反向传播中既不膨胀也不衰减，信号可以稳定地流过数十层。这是后续所有初始化方法的理论基础。
- **简化流程**：移除逐层预训练的需求，让端到端训练成为现实。深度学习从此不再依赖复杂的预训练流程。
- **通用性**：对于以 tanh 和 sigmoid 为激活函数的网络，Xavier 初始化几乎是理论最优的简单初始化策略。
- **历史地位**：Glorot 的论文是第一篇系统分析初始化对深度网络训练影响的论文。后续的 He 初始化、正交初始化等都是在 Xavier 基础上的修正和改进。

## 如何使用

Xavier 初始化的适用条件有两个：**激活函数在 0 附近近似线性**，且**激活函数关于 0 对称**。满足这两个条件的典型激活函数是 tanh，sigmoid（在 0 附近）。对于线性层和卷积层，Xavier 是默认推荐。PyTorch 中直接调用：

```python
import torch.nn as nn

# 均匀分布版（推荐，PyTorch Linear/Conv 默认是其变体 Kaiming Uniform）
nn.init.xavier_uniform_(layer.weight)
# 正态分布版
nn.init.xavier_normal_(layer.weight)
# bias 始终为零
nn.init.zeros_(layer.bias)
```

**不适用场景**：
- ReLU 及其变体 — ReLU 在负半轴输出恒为 0，破坏了线性近似假设，需要使用 He 初始化。参见 [[he-init]]。
- Embedding 层 — 推荐截断正态分布（truncated normal），而不是 Xavier。
- RNN/LSTM — 推荐正交初始化，因 Xavier 无法解决循环连接中的长期依赖问题。
- 输出层 — 输出层的初始化通常不需要特殊处理，跟随前一层激活函数选择即可。

选择 Uniform 还是 Normal：两者方差等价，对于中小规模网络效果几乎一致。大规模网络（参数 > 1 亿）时 Normal 版本的尾部可能产生异常大的权重，此时 Uniform 更安全。但差异通常可忽略不计。

更多激活函数的选择参见 [[../activation/activation-selection]]，关于 ReLU 的初始化方法参见 [[../activation/relu]]。

## 常见误区

1. **认为 Xavier 初始化在所有激活函数上都能工作**。实际上它在 sigmoid 上虽然可用，但 sigmoid 的饱和区问题仍然存在。ReLU 下 Xavier 初始化会导致约一半的神经元死亡，因为负半轴的方差贡献被浪费了。这是 He 初始化提出的直接动机。
2. **混淆 Xavier Uniform 和 Xavier Normal 的选择**。两者的方差相同，只是分布形态不同。对于小规模网络效果几乎一样，不需要过度纠结。框架默认值通常是合理的选择。
3. **忽略 fan_in 和 fan_out 的选择**。卷积层中的 fan_in 是 $输入通道数 \times 卷积核大小$，不是简单的输入神经元数。mode='fan_in' 保证前向方差不变，mode='fan_out' 保证反向方差不变，通常推荐用 fan_in。
4. **将 Xavier 初始化当作万能药**。对于现代网络（ResNet、Transformer），Xavier 只是起点，后续的 LayerNorm、残差连接、学习率调度同样关键。初始化和归一化层是互补的两层防线，不能相互替代。见 [[../../contradictory]] 中的相关讨论。
5. **在迁移学习场景中忽略验证**。加载预训练权重后务必检查参数名称和形状是否匹配（`model.load_state_dict(..., strict=True)`），静默加载失败是初始化相关的最难以排查的 bug。

## 参见

- [[initialization]] — 初始化方法总体概览
- [[he-init]] — He/Kaiming 初始化（ReLU 系列专用）
- [[why-initialization-matters]] — 初始化为什么重要（理论背景）
- [[../activation/activation-selection]] — 激活函数选择
- [[../activation/relu]] — ReLU 激活函数
- [[../../contradictory]] — 初始化相关的争议与讨论
