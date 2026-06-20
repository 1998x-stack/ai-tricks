# Embedding 对抗扰动（FGM / PGD）

## 是什么

Embedding 对抗扰动是一种在 NLP 模型的 Embedding 层添加微小扰动来提升模型鲁棒性的技术。核心思想是：在训练时找到能让模型损失最大的扰动方向（即"对抗"方向），将扰动叠加到 Embedding 上，强制模型对这些扰动不敏感。这等价于一种数据增强——模型在训练时看到的不是静态的 Embedding，而是被"攻击"过的变体。

两种主流方法：

- **FGM（Fast Gradient Method）**：单步对抗训练。计算损失对 Embedding 的梯度，沿梯度方向添加一个固定幅度的扰动，然后重新计算损失并更新参数。速度快、实现简单，是 NLP 中最常用的方法。
- **PGD（Projected Gradient Descent）**：多步对抗训练。在 FGM 的基础上迭代多步，每步添加小扰动后重新计算梯度，并将扰动投影到允许范围内。效果通常更好，但训练成本是 FGM 的数倍。

它们的数学本质是解决一个 min-max 优化问题：最小化模型在"最难例"上的损失。

## 为什么重要

NLP 模型对输入的微小扰动往往非常敏感。同义词替换、拼写错误甚至不可察觉的噪声都可能导致预测结果翻转。Embedding 对抗扰动从两个层面解决这个问题：

1. **提升泛化能力**：对抗训练可以视为一种特别的 [[../regularization/regularization]]，它在 Embedding 空间中找到模型脆弱的决策边界区域，强迫模型在这些区域也做出正确判断。这在数据量有限时尤其有效——本质上在 Embedding 连续空间上做数据增强。
2. **提升鲁棒性**：经过对抗训练的模型对对抗样本、输入噪声和分布外样本具有更好的抵抗能力。在法律文书、医疗文本等风险敏感场景中，这一点可能比准确率更重要。

与视觉领域的对抗训练不同，NLP 的对抗扰动加在连续的 Embedding 空间而非离散的 Token 空间。这意味着扰动更"自然"——因为 Embedding 本身就是连续表示，添加微小扰动后的表示仍然在合理的语义区域。

## 如何使用

### FGM 实现要点

```python
class FGM:
    def __init__(self, model, epsilon=0.5):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'embedding' in name:
                self.backup[name] = param.data.clone()
                grad = param.grad
                if grad is not None:
                    norm = torch.norm(grad)
                    if norm != 0:
                        param.data.add_(grad / norm, alpha=self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
```

使用时，在常规的前向-反向传播之间插入攻击步骤。注意只扰动 Embedding 层，不扰动其他层（全连接、注意力等），在 Embedding 空间做扰动是最有效的做法。

### PGD 实现要点

PGD 与 FGM 类似但需要多次迭代：

```python
class PGD:
    def __init__(self, model, epsilon=0.5, alpha=0.1, k=3):
        self.model = model
        self.epsilon = epsilon    # 总扰动限制
        self.alpha = alpha        # 每步步长
        self.k = k                # 迭代步数
        self.backup = {}
```

PGD 的核心区别是 `k` 步内每步加 `alpha` 大小的扰动，每步都重新计算梯度，最后限制总扰动不超过 `epsilon`。在每步之间需要将扰动投影回 `epsilon` 球内。

### 超参数选择

- **FGM epsilon**：常见范围 `0.3 ~ 1.0`，`0.5` 是常用默认值。epsilon 过小无效果，过大则扰动太强导致训练不稳定。
- **PGD epsilon**：同 FGM，`0.5`。alpha 通常设为 `epsilon / k` 或 `epsilon / (k * 2)`。k 一般取 `3~5`，再多步收益递减。
- **使用时机**：在训练的后期（如前 20%~30% 步数完成后）再开启对抗训练，让模型先学到基本模式。一开始就用对抗训练可能收敛困难。

### 什么时候有效

- 数据量不大（几千到几万条）时提升明显
- 任务本身对输入噪声敏感（分类、情感分析、文本匹配）
- 模型容量足够（大模型比小模型从对抗训练中获益更多）
- 与 [[./text-data-augmentation]] 结合使用时效果更显著

## 常见误区

1. **扰动所有参数**：只应扰动 Embedding 层。扰动注意力层或全连接层不会带来收益，反而可能导致训练不稳定。
2. **epsilon 选择过大**：FGM 最怕 epsilon 太大。如果训练损失开始震荡或飙升，先降低 epsilon 而不是调整学习率。建议从 0.1 开始往大调。
3. **从第一步开始就用**：推荐先做一定步数的普通训练，再开启对抗训练。这能保证模型先获得基本的判别能力，否则对抗训练容易"学废"。
4. **期望 PGD 总是比 FGM 好**：PGD 确实更强，但 K 步前反向传播意味着训练成本成倍增加。在小数据集或快速实验中，FGM 就足够了。PGD 适合追求极限效果的场景。
5. **误解为"对输入文本加噪声"**：对抗扰动加在 Embedding 空间而非 Token 空间，训练完成后不需要在推理时加扰动。这是训练阶段的技巧，推理时完全不需要修改。

## 参见

- [[./bert-finetuning]] — BERT 微调中如何结合对抗训练
- [[./text-data-augmentation]] — NLP 数据增强方法，与对抗训练互补
- [[../regularization/regularization]] — 正则化技术全景，对抗训练作为一种隐式正则化
- [[../optimizer-lr/optimizer-lr]] — 训练中学习率调度与对抗训练的配合
- [[./nlp]] — NLP 核心技巧概览
