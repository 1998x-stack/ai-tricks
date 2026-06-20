# Label Smoothing

## 是什么

Label Smoothing（标签平滑）是一种将 one-hot 硬标签（hard label）转化为软标签（soft label）的正则化技术。它通过对真实标签分布与均匀分布做加权平均，降低模型对训练标签的"过度自信"。

### 数学公式

对于 $K$ 类分类任务，原始 one-hot 标签 $y_{\text{hard}}$ 中正确类为 1、其余为 0。Label Smoothing 将其转换为：

$$y_{\text{soft}} = (1 - \alpha) \cdot y_{\text{hard}} + \frac{\alpha}{K}$$

其中 $\alpha$（通常记为 `label_smoothing` 或 `epsilon`）是控制平滑强度的超参数。当 $\alpha = 0.1$、$K = 1000$ 时：

- 正确类标签从 1 降为 $0.9 + 0.1/1000 = 0.9001$
- 其他类标签从 0 升为 $0.1/1000 = 0.0001$

### 损失函数变化

原始交叉熵损失（标准）：
$$\mathcal{L} = -\sum_{k=1}^{K} y_{\text{hard}, k} \log p_k$$

Label Smoothing 后的损失：
$$\mathcal{L}_{\text{LS}} = -(1 - \alpha) \log p_{\text{correct}} - \frac{\alpha}{K} \sum_{k=1}^{K} \log p_k$$

第二项本质上是对所有类别的 logit 做正则化，惩罚模型在错误类别上输出极小的概率。

## 为什么重要

- **防止过度自信（Overconfidence）**：神经网络在 hard label 下容易输出接近 1 的置信度，即使对于难以分类的样本也是如此。Label Smoothing 让模型学会"给自己留余地"，输出更温和的概率分布
- **提升泛化能力**：硬标签可能导致模型过度关注训练集中的噪声标注。软标签的容错机制让模型对标注错误更鲁棒
- **改善模型校准（Calibration）**：输出概率更好地反映真实正确率——这是所有可靠置信度评估的基础
- **与知识蒸馏互补**：软标签使教师模型输出的分布更易于被学生模型模仿，是所有现代知识蒸馏流程的标配

### 对模型输出的影响

标准模型在 hard label 下训练后，正确类 logit 趋于无穷大，softmax 输出趋于 0.999+；Label Smoothing 迫使正确类与错误类之间维持一个有界的差距，softmax 输出更加温和、信息量更丰富。这些软化的输出正是 [[../normalization/normalization]] 和知识蒸馏所需要的目标分布。

## 如何使用

### 超参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| $\alpha$（epsilon） | 0.1 | 最常用的默认值，多数任务表现良好 |
| 小数据集 | 0.05~0.1 | 平滑强度不宜过大，否则标签信息损失过多 |
| 大规模分类（ImageNet） | 0.1~0.2 | 类别数多时稍微加大平滑有助于提升校准效果 |
| 多标签分类 | 0.01~0.05 | 多标签场景下平滑强度通常更低 |

### 实现方式

```python
# PyTorch 手动实现
def label_smoothing_loss(logits, targets, alpha=0.1):
    K = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    # 标准交叉熵部分
    ce_loss = F.nll_loss(log_probs, targets, reduction='none')
    # KL divergence with uniform distribution
    kl_loss = -log_probs.mean(dim=-1)
    return (1 - alpha) * ce_loss + alpha * (kl_loss * K / (K - 1))
```

多数框架已内置（PyTorch 的 `CrossEntropyLoss(label_smoothing=0.1)`、TensorFlow 的 `CategoricalCrossentropy(label_smoothing=0.1)`）。

### 适用场景

- **分类任务**：图像分类（ImageNet 上已默认使用）、文本分类
- **教师-学生蒸馏**：知识蒸馏前对教师模型应用 Label Smoothing，软化输出分布
- **细粒度分类**：类别间相似度高时，硬标签过度强调区分边界，软标签更符合实际

## Label Smoothing 与知识蒸馏的交互

知识蒸馏中，学生模型同时学习硬标签和教师输出的软标签。Label Smoothing 在这里有两个作用：

1. **改善教师质量**：对教师模型加 Label Smoothing 训练，其输出分布更平滑、信息更丰富，学生更容易模仿
2. **避免过度自信的教师**：无 Label Smoothing 时，教师可能输出接近 1 的置信度，蒸馏损失被极少数样本主导

交互关系较为微妙——[[../../contradictory]] 中有观点认为在某些蒸馏配置下 Label Smoothing 的好处会减弱，需要实验验证。

## 常见误区

1. **Label Smoothing 一直有正面效果**：当训练集无噪声、标注质量极高时，平滑反而会损失信息，降低模型在干净数据上的表现
2. **对所有任务用同一个 $\alpha$**：不同任务的最佳 $\alpha$ 差异大，应通过验证集调优
3. **回归任务也可以用**：Label Smoothing 仅适用于分类任务的交叉熵损失，回归任务中硬标签是连续值，没有 one-hot 形式
4. **$\alpha$ 越大正则化越强**：过大的 $\alpha$ 让所有类别的标签趋于均匀分布，模型无法学到类别区分信息，反而欠拟合
5. **与 Dropout 等价**：两者虽然都是正则化手段，但机制完全不同——Dropout 通过扰乱特征路径，Label Smoothing 通过软化目标分布，应配合使用而非替代
6. **只看 top-1 accuracy 调 $\alpha$**：Label Smoothing 对准确率的影响可能是轻微的，但对模型校准（Expected Calibration Error）的影响更显著。应同时关注校准指标

## 什么时候应该跳过 Label Smoothing

- 训练数据标注绝对正确且无噪声
- 模型在校准指标上已经表现良好，不需要进一步改善
- 蒸馏场景中学生模型较小，软标签包含的信息量已经足够
- 类别数非常少（如二分类），平滑效果有限

## 参见

- [[regularization]] — 正则化总论与超参数速查表
- [[../data-eval/data-eval]] — 泛化诊断与校准评估
- [[../optimizer-lr/optimizer-lr]] — 学习率与 label smoothing 的配合
- [[../../contradictory]] — Label Smoothing 有效性的争议与边界条件
