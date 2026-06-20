# 归一化层

## 概述

归一化层是深度学习中最重要的训练加速与正则化手段之一，通过在层间对激活值进行标准化，解决梯度弥散/爆炸问题并加速收敛。不同归一化方法在规范化维度上有所区别，选择取决于网络架构、任务类型和 batch size 等因素。

## 详细知识

- [[batch-normalization]] — BN深入解析：训练vs推理、与Dropout的交互
- [[layer-normalization]] — LN机制、为何主导Transformer、与BN对比
- [[group-normalization]] — GN适用小批次场景、与BN/IN的关系
- [[normalization-cheatsheet]] — 归一化速查表：按架构/任务/批次大小快速选型

## 核心技巧

### 搭建新模型先不加归一化层，逐步实验
- **来源**: 匿名用户
- **适用场景**: 新模型开发
- **原文**: > 做新模型的时候，最开始不要加激活函数，不要加batchnorm，不要加dropout，先就纯模型。然后再一步一步的实验，不要过于信赖经典的模型结构（除非它是预训练的），比如加了dropout一定会有效果，或者加了batchnorm一定会有提升所以先加上，首先你要确定你的模型处于什么阶段，到底是欠拟合还是过拟合，然后再确定解决手段。
- **要点**: 从裸模型起步，根据欠拟合/过拟合状态逐个添加组件，而非一次性堆叠。
- **参见**: [[normalization-cheatsheet]], [[../debugging/debugging]], [[../regularization/regularization]]

### 过拟合时先 Dropout 后 BN，严重时对 Embedding 层加
- **来源**: 匿名用户
- **适用场景**: 过拟合处理
- **原文**: > 如果过拟合，首先是dropout，然后batchnorm，过拟合越严重dropout+bn加的地方就越多，有些直接对embedding层加，有奇效。
- **要点**: 按过拟合严重度渐进增加正则：dropout → BN → 更密集的 dropout+BN → 对 embedding 层直接加。
- **参见**: [[batch-normalization]], [[../regularization/regularization]], [[../recsys/recsys]]

### BN 能大幅加速收敛，有 BN 可省略 Dropout
- **来源**: 「人工智能是什么」公众号
- **适用场景**: CNN 分类/回归
- **原文**: > Batch Normalization。这是我一直在使用的技巧，可以很大程度的加快收敛速度。建议搭建自己网络的时候尽量加上BN，如果有BN了全连接层就没必要加Dropout了。
- **要点**: BN 不仅加速训练，本身也提供正则化效果，可替代全连接层上的 Dropout。
- **参见**: [[batch-normalization]], [[../regularization/regularization]], [[../cnn-cv/cnn-cv]]

### 序列特征用 LayerNorm，其他用 BatchNorm
- **来源**: 匿名用户
- **适用场景**: 推荐系统 / NLP
- **原文**: > 序列特征才用layernorm，正常用batchnorm。
- **要点**: LayerNorm 独立于 batch 维度，适合变长序列；BN 依赖 batch 统计量，适合固定长度的非序列特征。
- **参见**: [[layer-normalization]], [[../nlp/nlp]], [[../recsys/recsys]]

### Batch Size 决定归一化方案：大 Batch 用 BN，小 Batch 用 IN 或 GN
- **来源**: Towards Data Science (编译)
- **适用场景**: 不同 batch size 下的训练
- **原文**: > 在你的网络中始终使用归一化层（normalization layers）。如果你使用较大的批处理大小（比如10个或更多）来训练网络，请使用批标准化层（BatchNormalization）。否则，如果你使用较小的批大小（比如1）进行训练，则使用InstanceNormalization层。请注意，大部分作者发现，如果增加批处理大小，那么批处理规范化会提高性能，而当批处理大小较小时，则会降低性能。但是，如果使用较小的批处理大小，InstanceNormalization会略微提高性能。或者你也可以尝试组规范化（）。
- **要点**: BN 依赖 batch 统计量，小 batch 时估计不稳定；IN 在每个样本内做归一化，GN 在通道分组内做归一化，均不依赖 batch 大小。
- **参见**: [[group-normalization]], [[normalization-cheatsheet]], [[../batch-size/batch-size]], [[../cnn-cv/cnn-cv]]

### 多机多卡训练用同步 BN
- **来源**: 匿名用户
- **适用场景**: 分布式训练 / 多 GPU
- **原文**: > batchnorm一定要用，如果你是多机多卡，也可以考虑同步的batchnorm。
- **要点**: 分布式训练中默认各卡独立计算 BN 统计量，Sync BN 跨卡同步均值和方差，得到更准确的统计估计。
- **参见**: [[batch-normalization]], [[../infra/infra]]

### 各种 Norm 是低成本优化方向
- **来源**: zhuzhu
- **适用场景**: 模型调优
- **原文**: > 为网络增加各种norm：batchnorm、weightnorm、layernorm、groupnorm等等，不一定有用，但也是可能的优化方向，最重要的——代码改动不大。
- **要点**: 尝试不同类型的归一化几乎不增加代码复杂度，值得作为调优候选方案。
- **参见**: [[normalization-cheatsheet]], [[../optimizer-lr/optimizer-lr]], [[../debugging/debugging]]

### 不仅输入要归一化，输出也要
- **来源**: 李 adad
- **适用场景**: 回归任务
- **原文**: > 一般来说分类就是Softmax, 回归就是L2的loss. 但是要注意loss的错误范围（主要是回归），你预测一个label是10000的值，模型输出0，你算算这loss多大，这还是单变量的情况下。一般结果都是nan。所以不仅仅输入要做normalization，输出也要这么弄。
- **要点**: 回归任务中预测目标量级过大时 loss 会爆炸为 NaN，需对 label 做归一化到合理范围（如 0~1 或零均值单位方差）。
- **参见**: [[../data-eval/data-eval]]

### BN 是必选组件
- **来源**: 赵一鸣
- **适用场景**: 通用 CNN 训练
- **原文**: > batch normalization我一直没用，虽然我知道这个很好，我不用仅仅是因为我懒。所以要鼓励使用batch normalization。
- **要点**: BN 效果公认优秀，不应因惰性而省略。
- **参见**: [[batch-normalization]], [[../cnn-cv/cnn-cv]]

### BN 是大杀器，能极大加快训练
- **来源**: 罗浩.ZJU
- **适用场景**: 通用训练
- **原文**: > batchnorm也是大杀器，可以大大加快训练速度和模型性能
- **要点**: BN 是训练速度和模型性能双重提升的关键组件之一（与 ReLU、Dropout、Adam 并列）。
- **参见**: [[batch-normalization]], [[../activation/activation]], [[../optimizer-lr/optimizer-lr]], [[../regularization/regularization]]

### Group Normalization 可替代 BN
- **来源**: 吴育昕 / 何恺明 (FAIR)
- **适用场景**: 小 batch 训练 / 检测分割
- **原文**: > Face book AI research（FAIR）吴育昕-恺明联合推出重磅新作Group Normalization（GN），提出使用Group Normalization 替代深度学习里程碑式的工作Batch normalization。一句话概括，Group Normbalization（GN）是一种新的深度学习归一化方式，可以替代BN。
- **要点**: GN 将通道分为若干组，每组内计算均值和方差，不依赖 batch 维度，在小 batch（batch_size=1~8）场景效果优于 BN。
- **参见**: [[group-normalization]], [[../cnn-cv/cnn-cv]]

### Group Normalization 的 TensorFlow 实现
- **来源**: 吴育昕 / 何恺明 (FAIR)
- **适用场景**: 代码参考
- **原文**:
```python
def GroupNorm(x, gamma, beta, G, eps=1e-5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W])
    return x * gamma + beta
```
- **要点**: GN 将 (N, H, W) 和组内通道合并统计，核心操作是 reshape 后在组内维度上计算 moments。
- **参见**: [[../cnn-cv/cnn-cv]]

### BN 的定义与原理
- **来源**: 忍猫
- **适用场景**: 理论学习
- **原文**: > Batch Normalization 于2015年由 Google 提出，开 Normalization 之先河。其规范化针对单个神经元进行，利用网络训练时一个 mini-batch 的数据来计算该神经元的均值和方差，因而称为 Batch Normalization。
- **要点**: BN 沿 batch 维度计算统计量，对每个神经元独立做标准化，从根本上缓解内部协变量偏移。
- **参见**: [[batch-normalization]], [[../activation/activation]], [[../initialization/initialization]]

### LRN 基本可弃用
- **来源**: 匿名用户
- **适用场景**: CNN 训练
- **原文**: > LRN一类的，其实可以不用。不行可以再拿来试试看。
- **要点**: Local Response Normalization 已被 BN 等更优方法取代，仅作为兜底尝试。
- **参见**: [[../cnn-cv/cnn-cv]]

### 适当考虑 BN
- **来源**: 匿名用户
- **适用场景**: 小数据集
- **原文**: > 适当考虑可用Batch Normalization，尽管有些偷懒
- **要点**: 小数据集上 BN 仍然是有效的加速手段，不应省略。
- **参见**: [[batch-normalization]], [[../data-eval/data-eval]], [[../batch-size/batch-size]]

## 经典论文

- **Batch Normalization**: https://arxiv.org/abs/1502.03167
- **Layer Normalization**: https://arxiv.org/abs/1607.06450
- **Instance Normalization**: https://arxiv.org/abs/1607.08022
- **Group Normalization**: https://arxiv.org/abs/1803.08494
- **Weight Normalization**: https://arxiv.org/abs/1602.07868
