# 权重衰减与 L2 正则化

## 是什么

L2 正则化是在目标函数中加入 $\lambda\|\theta\|^2/2$，其梯度为 $\lambda\theta$。解耦权重衰减直接缩小参数，不让衰减项进入优化器的梯度动量和二阶矩。

对于**无动量的普通 SGD**，把 L2 梯度加到任务梯度，更新为：

$$\theta_{t+1}=(1-\eta_t\lambda)\theta_t-\eta_t\nabla L(\theta_t).$$

这与使用相同学习率缩放的乘法衰减等价。加入动量或自适应预条件后，需要检查具体更新公式，不能直接推广该等价性。PyTorch SGD 的 `weight_decay` 将衰减项加到梯度中；有 momentum 时该项进入动量缓冲区，**不是独立的解耦更新**。参见 [PyTorch SGD 官方算法](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)。

Adam 的耦合 L2 会改变一、二阶矩估计。AdamW 将参数衰减与任务梯度的自适应更新分开，其简化形式为：

$$\theta_{t+1}=(1-\eta_t\lambda)\theta_t-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

这里的 $\hat m_t,\hat v_t$ 来自任务梯度，不含 L2 项。不要再手工添加同一 L2 惩罚，否则改变了实验处理。

## 为什么重要

同名 `weight_decay` 在不同优化器中的语义可能不同，直接复用数值会混淆正则化强度与算法变化。AdamW 的论文给出了特定图像分类设置中的实验结果，不能据此承诺所有任务固定提升 0.5–2%，或声称所有主流模型采用同一配置。参见 [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)。

仅看一个 weight decay 数值还不够：在忽略梯度的直观分析中，累计缩放因子为 $\prod_t(1-\eta_t\lambda)$。因此学习率调度、更新步数、batch 和训练时长都会改变累计作用。增大 batch 后更新步数减少，即使 epoch 和 decay 数值不变，也不等于相同的累计衰减。

## 如何使用

### 定义一个可以解释的 A/B 对照

1. 固定优化器、学习率调度、训练样本、更新预算与参数分组。
2. 只改变 `optimizer.weight_decay`，例如 A=0、B=0.01。这个例子是探索性候选值，不是跨任务推荐值。
3. 使用相同种子配对，记录训练/验证损失、主任务指标、参数范数及实际学习率。
4. 按验证集选择 checkpoint，最终在固定测试集比较。若继续试多个 decay 数值，应报告搜索预算并保留独立测试集。
5. Adam 和 AdamW 的比较应另设优化器策略实验，避免把算法切换与 decay 数值修改混为一谈。

出现 loss 上升时，先排查标签、数据预处理、数值问题及学习率；不能仅凭曲线断言 decay 太大。小数据是否需要更强正则化取决于预训练、任务偏移和参数子空间，没有单调通用规律。

### 参数分组必须明确

以下示例声明一个可复查的规则：对可训练且维度大于 1 的参数衰减；其余参数不衰减。这是**维度规则**，并不等于对每种架构都完美识别 normalization/bias。特殊归一化参数、向量参数和共享权重需按模型结构核对。

```python
import torch

decay, no_decay = [], []
for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
        continue
    target = decay if parameter.ndim > 1 else no_decay
    target.append(parameter)

optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=1e-4,
)
```

实验日志应保存每组参数名、数量与 decay，而不只保存优化器顶层默认值。LoRA 中衰减的是适配器矩阵；其缩放、分解与合并后的等效权重不同，不能直接当成全量微调同一强度的正则化。

## 常见误区

- **“Adam + L2 = AdamW”**：通常不等价，区别是衰减是否进入动量和二阶矩。
- **“PyTorch SGD 总是解耦 weight decay”**：其公开算法是梯度加项；只有明确条件下才与独立缩放等价。
- **“bias/norm 不衰减因为正则化没有意义”**：是否排除是需要说明和验证的设计选择，不能用参数值小就推导出必然无效。
- **“所有模型从同一个 decay 开始就正确”**：数值只可作为预注册候选，需同时报告学习率、步数和参数组。
- **“训练损失更低证明泛化更好”**：比较独立验证/测试指标及种子差异，保留负结果。

## 参见

- [[optimizer-lr]]：来源技巧与作者经验。
- [[adam]]：Adam / AdamW 与动量估计。
- [[sgd-momentum]]：SGD 与 momentum 的交互。
- [[learning-rate-scheduling]]：学习率调度。
- [[../../contradictory]]：不同任务条件下的相反经验。

本文为编辑校订，原作者引文保留在主页面；示例数值不构成已验证的实验结论。
