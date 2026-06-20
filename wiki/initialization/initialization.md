# 初始化方法

## 概述

参数初始化决定了模型的收敛速度与最终性能，合适的初始化可以让参数更接近最优解、防止落入局部极小，不合适的初始化则可能导致梯度消失或梯度爆炸。好的初始化应同时考虑前向和反向两个过程：数值太大会导致前向陷入饱和、反向梯度爆炸；数值太小会导致反向梯度消失。

## 详细知识

- [[xavier-init]] — Xavier/Glorot初始化：方差保持原理、适用场景（tanh/sigmoid）
- [[he-init]] — He/Kaiming初始化：为何ReLU需要2/n、与Xavier对比
- [[why-initialization-matters]] — 初始化为何重要：梯度消失/爆炸、收敛速度、失败案例

## 核心技巧

### Xavier/Glorot 初始化（通用首选）
- **来源**: 爱睡觉的KKY / 炫光 / Jarvix / 匿名用户
- **适用场景**: 线性层、卷积层、tanh/sigmoid 激活函数
- **原文**: "初始化方法，linear / cnn一般选用kaiming uniform 或者normalize" / "无脑用xavier" / "一次惨痛的教训是用初始化cnn的参数，最后acc只能到70%多，仅仅改成xavier，acc可以到98%"
- **要点**: Xavier 初始化适用于 tanh 和 sigmoid 激活函数，是卷积层和线性层的首选默认初始化方法。PyTorch 已内置 Xavier/Kaiming 初始化，不需要手动实现。
- **参见**: [[../activation/activation]]

### He 初始化（ReLU 专用）
- **来源**: 炫光 / 京东白条 / 摘星狐狸
- **适用场景**: ReLU 及变体（LeakyReLU、PReLU）激活函数的网络层
- **原文**: "relu激活函数初始化推荐使用He normal，tanh初始化推荐使用Glorot normal" / "ReLU初始化推荐使用He normal，tanh初始化推荐使用Glorot normal"
- **要点**: He 初始化（也称 Kaiming 初始化）是 ReLU 系列激活函数的标准搭配，标准差设置为 $2/n$，可以避免 ReLU 带来的梯度消失问题。
- **参见**: [[../activation/activation]]

### 截断正态初始化（Truncated Normal）用于 Embedding
- **来源**: 爱睡觉的KKY
- **适用场景**: Embedding 层的权重初始化
- **原文**: "embedding 一般选择截断 normalize"
- **要点**: Embedding 层推荐使用截断正态分布初始化，比均匀分布或标准正态分布更适合词向量的分布特性，能加速收敛并提升效果。

### Uniform vs Normal 初始化的选择
- **来源**: Jarvix / 摘星狐狸
- **适用场景**: 不同网络层类型对应的初始化分布选择
- **原文**: "给word embedding初始化，最开始使用了TensorFlow中默认的initializer（即glorot_uniform_initializer），训练速度慢不说，结果也不好。改为uniform，训练速度飙升，结果也飙升" / "均匀分布初始化：w = np.random.uniform(low=-scale, high=scale, size=[n_in, n_out])，其中scale通常设置为√3/n"
- **要点**: 均匀分布简化实现为 $w = \text{Uniform}(-\sqrt{3/n}, \sqrt{3/n})$；高斯分布为 $w = \mathcal{N}(0, 1/\sqrt{n})$。具体选择取决于激活函数和任务性质，Embedding 用 uniform 可能优于 glorot_uniform。
- **参见**: [[embedding]]

### SVD 初始化用于 RNN
- **来源**: 摘星狐狸
- **适用场景**: 循环神经网络（RNN），特别是处理长期依赖关系时
- **原文**: "SVD（奇异值分解）初始化主要用于循环神经网络（RNN），在处理长期依赖关系时可能会更加有效"
- **要点**: SVD 初始化通过对随机矩阵做奇异值分解来得到正交的权重矩阵，有助于缓解 RNN 中的梯度消失和梯度爆炸问题。
- **参见**: [[rnn]]

### 零初始化的神话与陷阱
- **来源**: 京东白条
- **适用场景**: 理解为什么不能全零初始化（bias 除外）
- **原文**: "将权重矩阵W初始化方法改成了全为0的初始化...在训练完第一个epoch后预测精度就达到了85%以上，最终20个epoch后精度达到92%"
- **要点**: 权重的全零初始化在单层 Softmax 回归中可以工作（凸优化），但在多层网络中会破坏对称性导致每层神经元学到相同特征。权重用随机初始化，**bias 全初始化为 0**。
- **参见**: [[optimizer]]

### TensorFlow 与 PyTorch 初始化差异
- **来源**: 爱睡觉的KKY
- **适用场景**: 跨框架复现实验结果时
- **原文**: "参数初始化用xavier和truncated_normal可以加速收敛，但是，同样是tensorflow和pytorch用同样的初始化，pytorch可能存在多跑一段时间才开始收敛的情况"
- **要点**: 即使使用相同的初始化方法（如 Xavier），PyTorch 可能需要更多 epoch 才开始收敛。若发现 loss 久不下降，可多跑几个 epoch，或用 TensorFlow 实现同一逻辑做对比以排除框架差异导致的误判。

### 权重用随机初始化，Bias 恒为零
- **来源**: 炫光 / 匿名用户
- **适用场景**: 所有网络层的偏置项初始化
- **原文**: "对于weight的初始化我一般都是使用xavier初始化。当然也可以可以尝试何凯明大神的He初始化。对于bias的初始化全置于0" / "weight用高斯分布初始化，bias全初始化为0"
- **要点**: Bias 始终初始化为 0，不需要随机初始化。权重使用与激活函数匹配的初始化策略（Xavier/He/Kaiming）。

### 不同初始化做 Ensemble
- **来源**: 炫光
- **适用场景**: 竞赛、需要提升模型鲁棒性时
- **原文**: "可以使用不同的初始化方式训练出模型，然后做ensemble"
- **要点**: 用不同的初始化方法（Xavier、He、Uniform、Normal）分别训练模型，再对结果做 voting/soft-voting，可以低成本获取多样性。
- **参见**: [[ensemble]]

### 预训练模型加载后必须验证
- **来源**: 微尘-黄含驰
- **适用场景**: 使用预训练模型微调时
- **原文**: "使用 pretrained 模型时记得把 load 过程打印出来，保证模型被 load 了"
- **要点**: 加载预训练权重时务必打印 load 日志，检查参数名称和形状是否匹配。模型未正确加载的情况在静默模式下极难排查。
- **参见**: [[transfer-learning]]

### 长训练中换初始化方法跳出局部最优
- **来源**: 炫光
- **适用场景**: 训练初期 loss 不下降或陷入局部最优
- **原文**: "如果训练一开始不容易收敛或陷入局部最优，可以试试更换网络初始化方法，虽然pytorch默认会进行初始化，但试试别的初始化方法也是一种不错的选择"
- **要点**: PyTorch 默认初始化通常足够好，但当模型训练困难时，更换初始化方法（如 Kaiming → Xavier，或反之）是一种低成本的尝试方向。

### DeepSeek 大模型初始化标准差
- **来源**: Sam聊算法
- **适用场景**: 大语言模型（LLM）预训练的参数初始化
- **原文**: "参数初始化的标准差取0.06"
- **要点**: DeepSeek LLM 的实验表明，在 1e17~2e19 FLOPS 的计算规模下，大部分超参的最优值固定，其中参数初始化标准差固定为 0.06，不需要随计算量调整。
- **参见**: [[llm]]

### 正交初始化用于深层网络
- **来源**: 摘星狐狸
- **适用场景**: 深层网络或需要保持梯度流动的场景
- **原文**: "SVD（奇异值分解）初始化主要用于循环神经网络（RNN）"
- **要点**: 正交初始化（SVD 初始化的特例）能保持输入信号的范数在层间传递时不衰减，适合极深网络和 RNN。

### 初始化对 CNN 的巨大影响（案例）
- **来源**: Jarvix
- **适用场景**: 诊断模型训练失败时首先检查初始化
- **原文**: "一次惨痛的教训是用初始化cnn的参数，最后acc只能到70%多，仅仅改成xavier，acc可以到98%"
- **要点**: 初始化方法不当可能导致结果差距巨大（70% → 98%）。当模型表现异常时，初始化应是排查的首要方向之一。

### 先过拟合再谈初始化
- **来源**: 微尘-黄含驰
- **适用场景**: 新模型开发流程中的初始化策略
- **原文**: "做新模型的时候，最开始不要加激活函数，不要加batchnorm，不要加dropout，先就纯模型。然后再一步一步的实验"
- **要点**: 开发新模型时，先确保模型 capacity 过拟合，再逐步添加正则化和调整初始化策略，逐个区分每个组件的影响。
- **参见**: [[overfitting]]

## 代码示例

### Xavier/Glorot Uniform 初始化

```python
# Glorot Uniform
scale = np.sqrt(6. / (n_in + n_out))
W = np.random.uniform(low=-scale, high=scale, size=[n_in, n_out])
```

### He (Kaiming) Normal 初始化

```python
# He Normal (for ReLU)
stdev = np.sqrt(2. / n_in)
W = np.random.randn(n_in, n_out) * stdev
```

### SVD 正交初始化

```python
# SVD-based orthogonal init
W = np.random.randn(n_in, n_out)
U, S, Vt = np.linalg.svd(W, full_matrices=False)
W_ortho = U if U.shape == (n_in, n_out) else Vt
```

### PyTorch 设置初始化

```python
import torch.nn as nn

# Kaiming Normal (推荐用于 ReLU)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Xavier Uniform (推荐用于 tanh/sigmoid)  
nn.init.xavier_uniform_(layer.weight)

# 截断正态 (推荐用于 Embedding)
nn.init.trunc_normal_(embedding.weight, std=0.02)

# Bias 全零
nn.init.zeros_(layer.bias)
```

## 原则总结

| 激活函数 | 推荐初始化 | 标准差/缩放 |
|----------|-----------|-------------|
| ReLU / LeakyReLU | He (Kaiming) Normal | $\sqrt{2/n}$ |
| Tanh / Sigmoid | Xavier (Glorot) Uniform | $\sqrt{6/(n_{in}+n_{out})}$ |
| Embedding | Truncated Normal | $\sigma \approx 0.02$ |
| RNN / LSTM | Orthogonal (SVD) | 正交矩阵 |
| LLM 预训练 | Normal | $\sigma = 0.06$ |
| Bias | Zero | $0$ |

- **初始化足够好 → 超参都不用调，直接接近最优解。**
- **初始化没选对 → 结果跟模型有 bug 一样，不忍直视。**
- 当 loss 不下降时，耐心多跑几个 epoch（特别是 PyTorch），或换一个初始化方法试试。
- 使用预训练模型时，打印并确认 load 日志，排除静默加载失败。
