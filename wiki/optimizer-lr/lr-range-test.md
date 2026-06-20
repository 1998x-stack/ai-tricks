# 学习率范围测试（LR Range Test）

## 是什么

学习率范围测试（LR Range Test / Learning Rate Finder）是一种系统性地确定最佳初始学习率范围的方法。核心思想非常简单：**学习率从极小值开始，在若干个batch内线性或指数地增长到极大值，同时记录每个点的loss值，然后观察loss曲线的形态来确定合适的LR区间**。

该方法由Leslie N. Smith在2015年的论文《Cyclical Learning Rates for Training Neural Networks》中提出，最初是为Cyclical LR服务的，后来被fastai框架推广为通用的LR寻找工具。

典型操作流程：

1. 初始化一个极小的LR（如1e-7）
2. 选择LR的下界和上界（如1e-7 ~ 10）
3. 在每个mini-batch后增加LR（指数增长），记录对应的loss
4. 绘制LR vs Loss曲线
5. 从曲线中找到loss下降最快的区间，选择该区间内的某个值作为初始LR

```
Loss
^
|   *  ← loss开始下降
|     * * *
|          * *  ← loss下降最快（最佳LR区间）
|              * *
|                  *  ← loss不再下降或开始震荡
|                      * *
+------------------------------→ LR (log scale)
        ↑                       ↑
     下界 1e-5             上界 0.1
```

## 为什么重要

学习率是深度学习中最关键的单一超参数（参见[[../optimizer-lr#优先调LR再谈其他]]），但寻找合适的初始LR长期以来依赖经验法则或手动trial-and-error。LR Range Test首次提供了一种**系统化、可复现的LR搜索方法**，其重要性体现在以下方面：

1. **替代经验猜测**。经验法则（NLP 1e-5, CV 1e-3, Adam 3e-4）只提供了粗糙的起点，不能覆盖所有模型结构和数据组合。Range Test对每个具体场景给出定制化的LR建议。

2. **效率远高于网格搜索**。网格/随机搜索需要多次完整训练，而Range Test只需要1-2个epoch的前若干个batch，通常在几分钟内完成。这在调参效率上的提升是数量级的。

3. **不仅是寻找初始LR**。Range Test的结果也可用于设置Cyclical LR的上界（max_lr）、Cosine Decay的初始值、以及判断模型结构是否存在问题（如loss完全不下降可能表明模型代码有bug）。

4. **为Batch Size缩放提供定量依据**。当调整batch size时，LR理论上应线性缩放。Range Test可以在新batch size下快速验证缩放后的LR是否合理。

## 如何使用

### 实践步骤

**Step 1：准备工作**
- 准备一小部分训练数据（通常1-2个epoch足够了，不需要完整训练集）
- 将模型和优化器初始化为目标状态
- 设置optimizer的初始LR为极小值（如1e-7）

**Step 2：执行扫描**
- 对每个mini-batch：
  1. 执行前向传播，计算loss
  2. 执行反向传播
  3. 调用optimizer.step()（此时使用的是当前LR）
  4. 按预定规则增大LR（指数增长最常用）

**Step 3：分析曲线**
- 绘制LR（log scale）vs Loss曲线
- 找到loss开始快速下降的点（A）和loss停止下降或开始震荡的点（B）
- 推荐初始LR通常在A和B之间的中部，或者取B的1/10

```
Loss曲线解读：
  区间1（LR过小）：loss几乎不变 → 模型几乎不学习
  区间2（LR合适）：loss快速平滑下降 → 这是目标区域
  区间3（LR过大）：loss震荡或上升 → 溢出最优区域
```

### PyTorch示例

```python
import torch

def lr_range_test(model, train_loader, optimizer, loss_fn,
                  lr_min=1e-7, lr_max=10, num_iter=100):
    """简单的LR Range Test实现"""
    lrs, losses = [], []
    lr = lr_min
    gamma = (lr_max / lr_min) ** (1 / num_iter)

    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= num_iter:
            break

        # 设置当前LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向+反向
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())
        lr *= gamma

    return lrs, losses
```

### 已知的经验参考值

| 模型类型 | 典型LR范围 | 通过Range Test的经验起始值 |
|---------|-----------|--------------------------|
| 简单CNN（CIFAR-10） | 1e-3 ~ 1e-1 | 接近1e-2 |
| ResNet（ImageNet） | 1e-3 ~ 1e-1 | 0.1（SGD）|
| ResNet（CIFAR-10） | 1e-2 ~ 1 | 0.1（SGD）|
| BERT微调 | 1e-6 ~ 1e-4 | 2e-5 ~ 5e-5 |
| Transformer（WMT） | 1e-5 ~ 5e-4 | 1e-4（AdamW）|
| ViT | 1e-5 ~ 1e-3 | 3e-4（AdamW）|
| LLM预训练 | 1e-5 ~ 5e-4 | 3e-4（AdamW）|

### 何时应该信任Range Test的结果

- 在**小数据集的小型模型**上，结果非常可靠——loss曲线平滑，区间边界清晰
- 在**大数据集/大模型**上，由于单次扫描只暴露了数据的一个子集，结果会有一定方差。建议重复2-3次取平均
- 对**LLM**，Range Test的参考价值有限（训练数据分布太广，单次扫描偏差大），更推荐使用scaling law的预测或直接使用业界已验证的范围（参见[[../optimizer-lr#大模型预训练的Scaling Law]]）

### 进阶变体

- **One-Cycle LR**：结合了Range Test和Cyclical LR，训练中LR先升后降（cosine形状），peak LR用Range Test确定。fastai是其有力推广者。
- **Super-Convergence**：使用非常大的max_lr（通过Range Test确定配合One-Cycle），可以在极少的epoch内达到收敛效率。
- **预热扫描法**（[[../optimizer-lr#预热扫描的实用操作]]）：从1e-5开始以10倍为单位离散扫描，观察loss曲线——以10倍为单位跳，找发散前一档。这是Range Test的粗粒度版本，更适合快速估测。

## 常见误区

1. **在完整数据集上跑Range Test浪费资源**。Range Test的核心假设是：loss vs LR的曲线形态主要取决于模型结构和损失函数，与数据量关系不大。1-2个epoch的mini-batch足以反映趋势。在完整训练集上跑Range Test会极大浪费计算资源。

2. **Range Test找到的LR直接就是最佳LR**。Range Test给出的是"loss下降最快的LR区间"，并非训练全周期的最优LR。考虑到warmup、LR调度等因素，最终采用的初始LR通常在Range Test结果的偏下限。

3. **LR Range Test必须指数增长**。指数增长能最均匀地覆盖对数尺度上的各量级，但如果LR范围已知较窄（如微调场景），线性增长更细致。选择哪种取决于任务先验。

4. **忽略loss的平滑处理**。原始的loss曲线可能有震荡，影响判断。建议对loss做指数移动平均（EMA，α=0.9）或使用Savitzky-Golay滤波器后再看曲线。

5. **Range Test不能诊断所有LR问题**。如果模型代码有bug（如梯度未正确传播），Range Test的曲线可能不正常（loss完全不下降或剧烈震荡）。Range Test既是LR搜索工具，也是模型正确性的快速检验手段。

## 相关技巧

- 来自[[../optimizer-lr/optimizer-lr]]的核心技巧摘要：
  - 初始LR二分搜索法（[[../optimizer-lr#初始LR二分搜索法]]）
  - LR范围的土办法：预热扫描（[[../optimizer-lr#LR范围的土办法：预热扫描]]）
  - LR Range Test（[[../optimizer-lr#LR Range Test]]）
  - Cyclical LR的范围测试法（[[../optimizer-lr#Cyclical LR的范围测试法]]）
  - 初始LR的阶梯搜索法（[[../optimizer-lr#初始LR的阶梯搜索法]]）
  - 先锁定LR再调其他参数（[[../optimizer-lr#先锁定LR再调其他参数]]）
  - 仅调LR和batch size：大模型Scaling Law（[[../optimizer-lr#仅调LR和batch size：大模型Scaling Law]]）
  - BN层对LR的加速效应（[[../optimizer-lr#BN层对LR的加速效应]]）
- 跨类别链接：[[../batch-size/gradient-accumulation]] 改变有效batch后需重新Range Test
- [[../../contradictory]]：Range Test作为通用方法 vs 大模型依赖Scaling Law预测 的矛盾——模型规模越大，Range Test的统计效率越低，此时更应依赖理论推导或经验值

## 参见

- [[../optimizer-lr/optimizer-lr]] 主页面
- [[adam]] 适用于Adam的LR范围
- [[sgd-momentum]] 适用于SGD的LR范围
- [[learning-rate-scheduling]] 根据Range Test结果选择调度策略
