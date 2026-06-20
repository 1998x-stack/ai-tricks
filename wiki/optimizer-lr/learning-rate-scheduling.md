# 学习率调度策略

## 是什么

学习率调度（Learning Rate Scheduling）是指在训练过程中按预设规则动态调整学习率的策略。初始学习率决定了训练的起点，而调度策略决定了训练过程中学习率如何变化——两者同样重要。

常见的调度策略分为以下几类：

### 固定衰减类

**Step Decay（阶梯衰减）**：在预设的epoch/step将LR乘以衰减系数。例如每30个epoch将LR乘以0.1。这是最经典的做法，需要手动选择衰减时机。

```python
# PyTorch示例：每30个epoch LR×0.1
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
```

**Multistep Decay（多阶梯衰减）**：在多个自定义节点衰减。比单Step Decay灵活。典型做法如DeepSeek在训练到80%和90% tokens时各衰减一次。

```python
scheduler = MultiStepLR(optimizer, milestones=[80, 90], gamma=0.1)  # 百分比
```

**Exponential Decay（指数衰减）**：每步按指数衰减，LR = lr₀ * γ^step。γ通常取0.99 ~ 0.999，使LR缓慢平滑下降。

### 周期类

**Cosine Decay（余弦退火）**：LR按半个余弦曲线从初始值下降到接近零。公式为 `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))`，其中T是总步数。

**Cosine Annealing with Warm Restarts（带热重启的余弦退火，SGDR）**：余弦退火后突然将LR跳回初始值，模拟重新开始训练。可以收集多个局部最优解用于模型集成，或帮助跳出局部极小值。

**Cyclical LR（循环学习率，CLR）**：LR在规定上下界之间周期性变化，通常使用三角波（线性增减）或三角波+指数衰减。上下界可用LR Range Test确定。与Cyclical Momentum配合使用时，LR增大时momentum减小、LR减小时momentum增大，效果更好。

### 自适应类

**ReduceLROnPlateau（瓶颈降低LR）**：当验证集指标不再改善时自动降低LR。系数通常选0.1 ~ 0.5，patience参数控制等待步数。

```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

**Warmup（预热）**：训练初期从极小LR线性或指数地增长到目标LR。严格来说warmup是一种"启动策略"而非完整调度，但通常与其他调度器串联使用。

```
lr = lr_target * (step / warmup_steps)          # 线性warmup
lr = lr_min + (lr_target - lr_min) * step / wsp  # 从lr_min到lr_target
```

## 为什么重要

1. **LR调度的作用常被低估**。在Adam论文原实验中，甚至使用常数LR也能工作——但这远非最优。一个合适的LR调度可以和修改模型架构一样显著影响最终性能。

2. **决定训练后期的有效步长**。初始LR决定了多快能接近最优区域，而调度策略决定了能否精确收敛到最优值。没有衰减的LR会在最优值附近震荡，始终无法收敛。

3. **缓解优化与泛化的冲突**。训练早期需要大LR快速探索、跳过局部陷阱；训练后期需要小LR精细收敛、避免发散。调度策略就是在这两个需求之间做平滑过渡。

4. **Warmup解决训练初期的不稳定性**。训练开始时参数随机初始化、梯度统计不稳定，尤其是Adam的自适应机制导致二阶矩估计不准确——直接使用目标LR可能导致初始更新过大。Warmup让模型"热身"后再进入主训练阶段。

## 如何使用

### 选择策略的决策树

```
你的任务是什么？
├── 通用/不确定 → AdamW + Linear Warmup + Cosine Decay（万能模板）
├── CV分类从头训 → SGD + Momentum + Step Decay 或 Multistep
├── CV检测 → SGD + Multistep + Warmup
├── NLP预训练 → AdamW + Warmup + Cosine Decay（最终LR降至0）
├── NLP微调 → AdamW + Linear Warmup + Linear Decay
├── LLM预训练 → AdamW + Warmup + Cosine 或 Multistep（80%/90%处衰减）
├── 推荐系统 → Adam/Adagrad + Exponential Decay（γ=0.99）
├── 验证集驱动 → ReduceLROnPlateau + EarlyStopping
└── 需要模型集成 → Cosine Annealing with Warm Restarts / Cyclical LR
```

### 各策略的参数设置参考

| 策略 | 关键参数 | 推荐值 |
|------|---------|--------|
| Step Decay | step_size, gamma | step=30epoch, gamma=0.1 |
| Multistep | milestones | 视总步数，50%/80%处衰减 |
| Exponential | gamma | 0.99 ~ 0.9995 |
| Cosine | T_max, eta_min | T_max=总步数, eta_min=0~1e-5 |
| Cyclical | base_lr, max_lr | base=max/10, 见LR Range Test |
| ReduceLROnPlateau | factor, patience | factor=0.1~0.5, patience=5~10 |
| Warmup | warmup_steps | 总步数1/10~1/20，或固定2000步 |

### Wanrup的具体设置

- **warmup步数**：模型越复杂步数越多，数据集越大步数越少。BERT默认用总步数的1/10。LLM预训练常见2000步固定warmup。
- **warmup方式**：线性warmup最通用；对小规模数据可考虑指数warmup（增长更慢更稳）。
- **结合调度器**：warmup结束后再切换主调度器。PyTorch中通常用LambdaLR实现warmup，或用自定义WarmupCosineScheduler串联（参见[[../optimizer-lr#余弦退火的实现代码]]）。

### Cosine Decay的变体

- **Cosine + Linear Warmup**：最通用的组合，各框架广泛支持（HuggingFace Transformers的 `get_cosine_schedule_with_warmup`）。
- **Cosine + Restart**：SGDR，适合需要跳出局部最优或做模型集成的场景。注意重启时LR跳回初始值可能短暂增加loss，这是正常现象。
- **Half Cosine**：如果训练步数不足一个完整cosine周期，可将T_max设为总步数，cosine只下降至一半即结束。

### 推进式调整（训练中的手动干预）

除了预设调度器，还存在由训练loss/验证指标驱动的"在线调整"：
- **验证集瓶颈手动折半**：当验证loss停止下降时手动将LR除以2或5，继续训练（[[../optimizer-lr#LR按验证集瓶颈手动折半]]）
- **先小后大再衰减**：复杂任务先用小LR确保不爆，loss下降后再升LR加速，最后降LR收敛（[[../optimizer-lr#LR先小后大再衰减策略]]）

## 常见误区

1. **Warmup只对Transformer有用**。实际上，大batch训练（batch size > 1024）和深层网络（>50层）即使使用SGD，warmup仍能显著改善训练稳定性。

2. **ReduceLROnPlateau不需要patience**。没有patience的Plateau会在验证指标波动的第一个瞬间就降低LR，导致训练过早进入低LR阶段。patience通常设5~20个epoch。

3. **Cyclical LR只是花哨的cosine**。CLR的核心区别在于LR在上下界之间来回变化，而不是单调下降。这使得模型能探索不同的局部区域，尤其适合大学习率会破坏已学到特征的任务。

4. **Step Decay已过时被cosine全面取代**。Step Decay的优势在于可解释性：每次衰减后我们可以观察模型反应，判断是否还有继续训练的价值。Cosine Decay的LR全程都在快速下降，在某些任务中训练不足。具体选择需视任务而定。

5. **调度策略可以完全补偿初始LR选错**。如果初始LR偏离最优值超过一个数量级，任何调度策略都无法挽回。第一步永远是找到合理的初始LR范围（见[[lr-range-test]]）。

6. **Warmup Step越长越好**。过长的warmup意味着大量步数在低效LR下运行，浪费计算资源。经验值1/10总步数是平衡点，但如果模型本来就是从预训练权重开始的（微调），warmup步数可显著缩短。

## 相关技巧

- 来自[[../optimizer-lr/optimizer-lr]]的核心技巧摘要：
  - Cosine退火+warmup万能模板（[[../optimizer-lr#Cosine退火+warmup万能模板]]）
  - Warmup步数设置原则（[[../optimizer-lr#Warmup步数设置原则]]）
  - 阶梯下降精调法（[[../optimizer-lr#阶梯下降（Step Decay）精调法]]）
  - Cyclical LR周期性重启（[[../optimizer-lr#Cyclical LR周期性重启]]）
  - ReduceLROnPlateau + EarlyStopping（[[../optimizer-lr#ReduceLROnPlateau + EarlyStopping]]）
  - LR按验证集瓶颈手动折半（[[../optimizer-lr#LR按验证集瓶颈手动折半]]）
  - 指数衰减LR（[[../optimizer-lr#指数衰减LR]]）
  - 多策略配合：CLR + Cyclical Momentum联动（[[../optimizer-lr#循环动量（Cyclical Momentum）]]）
- 跨类别链接：[[../batch-size/gradient-accumulation]] 梯度累积影响有效总步数进而影响调度
- [[../../contradictory]]：Cosine全面优于Step vs Step在某些任务更可解释且效果不差 的矛盾——这取决于是否有足够的训练步数和是否需要手动判定收敛点

## 参见

- [[../optimizer-lr/optimizer-lr]] 主页面
- [[adam]] Adam配合LR调度
- [[sgd-momentum]] SGD搭配LR调度
- [[lr-range-test]] 确定LR上下界后再选择调度策略
- [[weight-decay]] 不同调度策略下weight decay的配合调整
