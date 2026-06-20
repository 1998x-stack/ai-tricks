# 权重衰减与L2正则化

## 是什么

权重衰减（Weight Decay）是一种正则化技术，在每一步参数更新时额外减去参数本身的一个小比例：

```math
θ_{t+1} = θ_t - lr * ∇L(θ_t) - lr * λ * θ_t
```

等价于在损失函数中加入L2正则化项 `0.5 * λ * ||θ||²`，因为L2正则化的梯度 `λ * θ` 正好等于权重衰减项 —— **但仅对SGD成立**。

关键区别在于 **L2正则化与自适应优化器的交互**：

- **SGD中**：L2正则化 ≈ 权重衰减，两者完全等价
- **Adam中**：权重衰减项 `λ * θ_t` 被Adam的自适应学习率缩放，使得大梯度参数（对应高频特征）的权重衰减被抑制，小梯度参数（对应稀疏特征）的权重衰减被放大——这并非设计意图

**AdamW**（2019年，Loshchilov & Hutter）修正了这个问题：将权重衰减从梯度更新中解耦，在Adam自适应更新之后独立执行：

```
# AdamW伪码
m_t = β1 * m_{t-1} + (1-β1) * g_t
v_t = β2 * v_{t-1} + (1-β2) * g_t²
θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)  # Adam更新
θ_t = θ_t - lr * λ * θ_{t-1}            # 解耦的权重衰减
```

## 为什么重要

1. **防止过拟合的核心手段**。权重衰减通过限制参数范数防止模型对训练数据中的噪声过拟合。没有weight decay，深层网络几乎必然过拟合。

2. **AdamW是现代训练的标准配置**。HuggingFace Transformers、LLaMA、GPT系列等主流模型全部使用AdamW而非Adam。从Adam切换到AdamW通常能获得0.5-2%的效果提升而不需要额外调参。

3. **不同模型对weight decay的敏感度差异极大**。ResNet类CNN对weight decay相当鲁棒（0.0001~0.1都能工作），但Transformer类模型对weight decay极为敏感——设太大可能使模型完全不收敛，设太小则过拟合。这背后的原因可能在于Transformer的LayerNorm与weight decay存在交互。

4. **与小数据微调的关系**。微调预训练模型时，weight decay过大会损伤预训练学到的知识，过小又不足以防止过拟合——这个平衡点的选择直接影响微调效果（参见[[../optimizer-lr#小数据微调：dropout和weight decay不宜过大]]）。

## 如何使用

### 不同场景的weight decay推荐值

| 场景 | 推荐WD | 说明 |
|------|--------|------|
| CV ResNet + SGD | 0.0001 ~ 0.005 | 不敏感，0.0001是起点 |
| CV + AdamW | 0.01 ~ 0.1 | AdamW通常用比SGD更大的WD |
| Transformer LLM预训练 | 0.1（DeepSeek） | 大模型固定超参，0.1很常见 |
| Transformer微调 | 0.01 ~ 0.1 | 从0.01开始试，不稳定则降低 |
| LoRA微调 | 0.01 | 通常不调rank或alpha后再动 |
| NLP AdamW | 0.01 | HuggingFace默认值 |
| 大模型SFT | 0.25 -> 下探 | 从较大值（0.25）向小调 |
| DPO训练 | 0.01 ~ 0.1 | 与beta配合调整 |
| 小数据微调 | 0.001 ~ 0.01 | 保守开始，防止损伤预训练能力 |

### 基本调参方法

1. **初始化先固定**：所有模型从 `weight_decay=0.01` 开始
2. **判断稳定性**：观察loss曲线是否发散——如果训练早期loss不降反升，WD太高了
3. **粗调方向**：
   - 数据量小 → 增大WD（更多正则化）
   - 数据量大 → 减小WD
   - 模型容量大 → 减小WD
   - 模型容量小 → 增大WD
4. **与其他超参数的协调**：
   - LR越大本身就有正则化效果，此时WD可设小
   - Dropout大时WD可设小
   - 小数据微调时WD和dropout都宜小（参见[[../optimizer-lr#小数据微调：dropout和weight decay不宜过大]]）

### 是否对所有参数施加Weight Decay

实践中通常不对bias和LayerNorm/BatchNorm参数施加weight decay。PyTorch中需要手动将参数分组：

```python
optimizer = AdamW([
    {'params': model.parameters_without_ln_and_bias()},
    {'params': model.ln_and_bias_params(), 'weight_decay': 0.0}
], lr=1e-4, weight_decay=0.01)
```

这种做法在LLaMA、GPT等主流模型中广泛采用，原因是bias和normalization参数的数值范围本就不大，正则化它们没有意义。

## 常见误区

1. **Adam + L2正则化 = AdamW**。这是最常见的误解。AdamW的作者明确证明了Adam的L2实现不等于解耦的权重衰减，并在几乎所有实验中发现AdamW优于Adam+L2。详见AdamW原论文《Decoupled Weight Decay Regularization》。

2. **Weight Decay越大泛化越好**。WD过大会限制模型容量，导致欠拟合。尤其对于Transformer，WD过大会破坏训练稳定性——模型参数被过度约束在零附近，LayerNorm的偏移量无法有效学习。

3. **Weight Decay只关于过拟合**。对Transformer而言，WD更多影响训练稳定性。大batch训练时WD过大可以直接让loss不下降。这种情况下需要先将WD降到0.001甚至0.0001。

4. **不同框架的weight decay等价**。PyTorch的 `weight_decay` 参数在 `optim.SGD` 中实现的是"真实的权重衰减"（解耦的），而在旧的 `optim.Adam` 中实现的是L2正则化（耦合的）。`optim.AdamW` 使用了解耦实现。用TensorFlow/Keras时也需注意框架文档中的实现语义。

5. **微调时weight decay与预训练一致即可**。预训练使用大WD（0.1）是因为有大量数据、长训练周期。微调通常数据更少，直接用0.1可能过度正则化。建议从0.01开始。

## 相关技巧

- 来自[[../optimizer-lr/optimizer-lr]]的核心技巧摘要：
  - AdamW替代Adam，权重衰减更合理（[[../optimizer-lr#AdamW替代Adam权重衰减更合理]]）
  - Weight Decay默认值及调参（[[../optimizer-lr#Weight Decay默认值及调参]]）
  - ResNet对WD不敏感，Transformer很敏感（[[../optimizer-lr#Weight Decay默认值及调参]]）
  - 权重衰减与数据集规模的关系（[[../optimizer-lr#权重衰减与数据集规模的关系]]）
  - 小数据微调：dropout和weight decay不宜过大（[[../optimizer-lr#小数据微调：dropout和weight decay不宜过大]]）
  - 大模型SFT初始LR与weight decay（[[../optimizer-lr#大模型SFT初始LR与weight decay]]）
  - LR太大时的泛化正则化效应（[[../optimizer-lr#LR太大时的泛化正则化效应]]）
- 跨类别链接：[[../regularization/weight-decay]] 从正则化角度讨论权重衰减
- [[../../contradictory]]：大WD vs 小WD哪个更好 的矛盾——这完全取决于数据规模和模型类型，不存在普适最优值

## 参见

- [[../optimizer-lr/optimizer-lr]] 主页面
- [[adam]] Adam/AdamW优化器实现细节
- [[sgd-momentum]] SGD中weight decay的经典设置
- [[learning-rate-scheduling]] 调度策略与weight decay的协同
