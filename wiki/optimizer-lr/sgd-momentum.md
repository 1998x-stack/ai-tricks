# SGD + Momentum 优化器

## 是什么

SGD（Stochastic Gradient Descent，随机梯度下降）是最基础的一阶优化算法。给定参数集 `θ` 和损失函数 `L`，SGD沿当前梯度方向更新参数：

```math
θ_{t+1} = θ_t - lr * ∇L(θ_t)
```

**Momentum（动量法）**在SGD基础上引入了历史梯度方向的累积，使更新方向不再单纯取决于当前梯度，而是过去梯度的指数衰减平均：

```math
v_t = α * v_{t-1} + lr * ∇L(θ_t)
θ_{t+1} = θ_t - v_t
```

其中 `α` 是动量系数（通常0.9），控制历史梯度的衰减速度。动量越大，历史梯度影响越持久。

**Nesterov Accelerated Gradient (NAG)**是动量的改进版本，先沿着上次更新的方向"向前看"再计算梯度：

```math
θ_lookahead = θ_t - α * v_{t-1}
v_t = α * v_{t-1} + lr * ∇L(θ_lookahead)
θ_{t+1} = θ_t - v_t
```

NAG的修正效果可以这样理解：标准动量可能因积累的惯性越过谷底再折返，而NAG提前在"预判位置"计算梯度，相当于在即将爬升时踩刹车，收敛更精确。

## 为什么重要

SGD+Momentum虽然在收敛速度上不如Adam，但因其出色的**泛化能力**仍是许多领域的首选优化器：

1. **泛化性能更好**：多项理论和实验表明，SGD倾向于收敛到"平坦最小值"（flat minima），而自适应方法更容易收敛到"尖锐最小值"（sharp minima）。平坦极小值的泛化性能通常更好，因为它对参数扰动的鲁棒性更强。在实践中这通常意味着SGD+Momentum能在测试集上比Adam高1-2个百分点。

2. **超参数物理意义明确**：LR和momentum各自的作用直观可解释——LR控制步长，momentum控制惯性。调参时可以通过loss曲线直接判断哪个参数需要调整。

3. **计算效率高**：相比Adam需要维护和更新两组状态（m和v），SGD+Momentum只有一组动量状态，内存占用约减半。这对大模型训练有明显优势。

4. **CV领域的标准选择**：从AlexNet到ResNet再到ViT，ImageNet上的基准方法几乎都使用SGD+Momentum（0.9）。这种历史惯性意味着大量已有经验和技巧可以直接复用。

## 如何使用

### 默认超参数

| 超参数 | 推荐值 | 范围 |
|--------|--------|------|
| momentum | 0.9 | 0.8 ~ 0.99 |
| 初始LR（CV从头训） | 0.1 | 0.01 ~ 0.1 |
| 初始LR（通用） | 0.01 | 0.001 ~ 0.1 |
| Nesterov | True | 建议开启 |

### 不同场景的初始LR

| 场景 | 推荐初始LR | 说明 |
|------|-----------|------|
| CV分类（ResNet等） | 0.1 | 经典Kaiming初始化搭配 |
| CV检测（YOLO等） | 0.01 | batch size综合决定 |
| 微调预训练模型 | 0.001 ~ 0.01 | 预训练权重已接近最优 |
| 推荐系统 | 0.01 | 从0.01开始2倍递减 |
| 小数据/小模型 | 0.01 ~ 0.1 | 先确保收敛 |

### 与LR调度的配合

SGD+Momentum对LR调度更加敏感，通常需要配合衰减策略：
- **Step Decay**：当loss进入平台期，将LR乘以0.1。这是最经典的做法——"训练不动了就降LR"。
- **Cosine Decay**：LR从初始值余弦下降到0，不需要手动选择衰减节点。配合SGD+Cosine的效果常优于Adam+Cosine。
- **Multistep**：在预设的step位置（如总体步数的50%, 75%）分段降低LR，典型如DeepSeek在80%和90% tokens处各降1次。

### Batch Size与LR的线性缩放

使用SGD+Momentum时，batch size与LR存在近似线性缩放关系（参见[[../optimizer-lr#Batch Size与LR的线性缩放关系]]）：
- batch size从64扩到256 → LR可从0.1扩到0.4
- 梯度累积等效扩大batch size，需同步调整LR
- 这种线性缩放对Adam不完全适用，因为Adam有自适应机制

## 常见误区

1. **"SGD就是没有动量的SGD"**。实践中几乎没有人使用纯SGD（无动量）。"SGD"在大多数框架和论文中指的就是SGD+Momentum（PyTorch的 `torch.optim.SGD` 默认momentum=0），使用前必须确认momentum参数是否已打开。

2. **CV用SGD，NLP用Adam是绝对规则**。Transformer类模型（ViT、BERT）也是CV架构，但基于Attention的模型对LR波动敏感，使用SGD+Momentum时LR需要极其精细的调度。实践中ViT通常也用AdamW。

3. **认为SGD不需要warmup**。大batch训练或深层网络时，SGD同样可以从warmup中受益——初始小LR稳定训练，再逐步增大到目标LR。

4. **SGD训练慢等于效果差**。SGD+Momentum收敛需要更多iteration，但每个iteration的计算成本和Adam差不多。最终的泛化性能往往更好，慢在wall time而非效果天花板。

5. **Momentum越大越好**。momentum=0.99会带来更强的惯性，但也意味着更新对方向变化响应慢，容易震荡或发散。0.9是经过广泛验证的安全值。

## 相关技巧

- 来自[[../optimizer-lr/optimizer-lr]]的核心技巧摘要：
  - SGD+Momentum最终效果常优于Adam（[[../optimizer-lr#SGD+Momentum最终效果常优于Adam]]）
  - 两阶段训练法：Adam前期快速收敛 → SGD后期提升泛化（[[../optimizer-lr#两阶段训练法Adam过渡到SGD]]）
  - 无脑用SGD+Momentum（[[../optimizer-lr#无脑用SGD+Momentum]]）
  - 初始LR：0.1搭配SGD+Momentum（[[../optimizer-lr#初始LR：0.1搭配SGD+Momentum]]）
  - Momentum通用设置范围0.9（[[../optimizer-lr#Momentum通用设置范围]]）
  - 最优配置：SGD + 0.1lr + 0.9momentum + weight decay 0.005（[[../optimizer-lr#最优配置：SGD + 0.1lr + 0.9momentum + weight decay 0.005]]）
- 跨类别链接：[[../batch-size/gradient-accumulation]] 梯度累积与LR缩放
- [[../../contradictory]]：SGD泛化更优 vs Adam收敛更快效果也足够好 的矛盾——最终选择取决于任务对泛化精度的要求和实验预算

## 参见

- [[../optimizer-lr/optimizer-lr]] 主页面
- [[adam]] Adam优化器对照
- [[learning-rate-scheduling]] 配合SGD的LR调度策略
- [[weight-decay]] SGD中的权重衰减设置
- [[lr-range-test]] 为SGD寻找合适的初始LR
