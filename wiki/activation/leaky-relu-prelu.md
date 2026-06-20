# LeakyReLU / PReLU 与 ReLU 变体

## 是什么

LeakyReLU 和 PReLU 是 ReLU 的最直接改进，目的是解决 ReLU 的 dying ReLU 问题。此外还有 ELU、SELU、GELU、Swish、Mish 等变体，各自针对不同场景优化。

**LeakyReLU**：在负半轴引入一个固定小斜率 $\alpha$（通常 0.01），公式为 $f(x) = \max(\alpha x, x)$。负值不再被硬截断为 0，而是保留小梯度，允许神经元从死亡状态恢复。$\alpha$ 是超参数，常用值包括 0.01、0.1 和 0.2。

**PReLU（Parametric ReLU）**：负半轴斜率 $\alpha$ 为可学习参数，随网络一起训练。每个通道或每层可以有自己的 $\alpha$。在 CNN 任务中 PReLU 常能比 LeakyReLU 额外提升精度，但也增加了过拟合风险和参数量。

**其他变体**：

| 变体 | 公式 | 特点 |
|------|------|------|
| ELU | $f(x)=x \text{ if } x>0, \alpha(e^x-1) \text{ if } x≤0$ | 负半轴指数饱和，输出均值接近 0，加速收敛 |
| SELU | 缩放版 ELU（$\lambda \cdot \text{ELU}$） | 自动实现自归一化（配合 AlphaDropout） |
| GELU | $x \cdot \Phi(x)$ 其中 $\Phi$ 是标准正态 CDF | Transformer 默认，平滑随机正则化近似 ReLU |
| Swish | $x \cdot \sigma(x)$ 其中 $\sigma$ 是 sigmoid | 无界上界 + 非单调，Google 提出，有时称 SiLU |
| Mish | $x \cdot \tanh(\ln(1+e^x))$ | Swish 的平滑变体，CV 小任务上略有提升 |
| ReLU6 | $\min(\max(0, x), 6)$ | 限制上界为 6，适合量化/移动端部署 |

## 为什么重要

**解决 dying ReLU**：当大量神经元输入为负时，ReLU 梯度恒为 0，这些神经元永久死亡不再更新（反向传播过程中永远不会收到梯度信号）。LeakyReLU 通过保留负梯度让神经元有机会恢复，特别适合大学习率或长序列训练。实验表明，在训练初期使用稍大的学习率时，LeakyReLU 的神经元存活率显著高于 ReLU。参见 [[relu]]。

**PReLU 的自适应能力**：PReLU 让网络自己决定负半轴斜率，适合那些负信息有意义的任务（如细粒度分类、人脸识别）。He 等人在 2015 年的 ImageNet 分类实验表明，将 ReLU 替换为 PReLU 可在 Top-5 错误率上获得约 1.2% 的绝对提升。PReLU 的参数量虽然微小（每通道一个标量），但确实能提升模型容量。

**GELU 是现代架构的基石**：BERT、GPT、ViT 等 Transformer 模型全系列使用 GELU。它不是 ReLU 的变体，但功能等价（非线性的门控机制），且平滑可微，训练更稳定。GELU 可以理解为在 ReLU 的基础上加入了输入的随机正则化——按输入值的大小概率性地保留或归零。

**SELU 的自归一化**：如果网络设计为 SELU + AlphaDropout 的组合，可以自动保持每层输出的零均值和单位方差，无需 [[../normalization/normalization]]。但约束极其苛刻（网络必须是全连接、使用 LeCun 初始化），实际应用有限。在满足条件的场景中，SELU 可以简化训练流程，消除对 BN 的依赖。

## 如何使用

**快速决策流程**：
1. 先用 ReLU 建 baseline → 训练正常？→ 保持 ReLU 不变
2. dying ReLU 出现（激活率持续下降或大量神经元输出恒为 0）→ LeakyReLU(α=0.01) → 还死？→ PReLU
3. 精度已到瓶颈且类别相近 → 尝试 PReLU 获取最后 1-2%
4. 移动端部署 → ReLU6（量化友好，上界约束简化定点运算）
5. 做 Transformer → 直接 GELU，不要用 ReLU，这是行业标准
6. 追求理论优雅或全连接自归一化 → SELU，但确认网络架构满足约束条件

**场景匹配表**：

| 场景 | 推荐 | 原因 |
|------|------|------|
| 通用 CV 基线 | ReLU | 默认，计算最快，足够好用 |
| dying ReLU 出现 | LeakyReLU(0.01) | 最小改动，最大收益 |
| 最后 1-2% 精度提升 | PReLU | 可学习参数，自适应负半轴 |
| 深层 CNN（ResNet） | ReLU + BN | 足够稳定，PReLU 提升有限不值得额外复杂度 |
| Transformer | GELU | 行业标准，平滑可微，预训练权重都用它 |
| 全连接自归一化 | SELU | 可替代 BN，但约束多需验证 |
| 移动端/量化 | ReLU6 | 上界约束降低量化难度 |
| 推荐系统（DLRM 等） | ReLU / LeakyReLU | 宽深网络，LeakyReLU 可防死亡 |
| 音频/信号处理 | LeakyReLU / ELU | 负信息重要，不宜截断 |

**PReLU 的学习率**：PReLU 的 α 参数学习率可以比主网络稍大（约 1.1-1.5 倍），使其快速适应负半斜率的合适值。也可以对 α 使用独立的学习率调度器。

**参数共享**：PReLU 可以每层共享一个 α（参数量少，稳定性好），也可以每个通道独立（灵活性高，适合 CNN）。推荐在 CNN 中按通道独立，在 MLP 中按层共享。

**初始化**：LeakyReLU 仍推荐配合 [[../initialization/initialization]] 中的 He 初始化，但需使用修正版方差 $2/(1+\alpha^2)n$，补偿负半轴梯度贡献。不改初始化可能导致变体表现不如预期。PReLU 的 α 通常初始化为 0.25。

## 常见误区

**LeakyReLU 一定优于 ReLU**：并不。多项实验表明，在 BN 充分使用且学习率合适的条件下，ReLU 和 LeakyReLU 性能相当。LeakyReLU 主要在 dying ReLU 出现的场景有价值，无脑替换可能只增加计算量而无收益。尤其是在现代架构（ResNet + BN + Adam）中，ReLU 几乎不会出现大规模神经元死亡。

**PReLU 的 α 学到 -1 就废了**：PReLU 若不加约束，α 可能变为负值甚至小于 -1，使激活函数变成反转版，破坏训练。实践中通常限制 α ∈ [0, 1] 或在 α 上加入 L2 正则。PyTorch 的 PReLU 默认 α 初始化为 0.25，但未做上界约束，需自行处理。

**变体越多越好**：GELU 的计算量远大于 ReLU（涉及 erf 或近似计算），在大规模部署时成本不容忽视。Transformer 用 GELU 是因为其平滑性质带来的训练收益超过了计算成本，但多数 NCHW 架构的 CV 任务中 ReLU 已足够。不要为了追求"先进"而引入不必要的计算开销。

**SELU 可以完美替代 BN**：有条件——SELU 的自归一化性质只在全连接网络中有效（且序列必须由 SELU + AlphaDropout + LeCun 初始化构成），CNN、RNN、Transformer 等架构不适用。不要在不满足条件时依赖 SELU 做归一化，否则训练可能比不用 BN 还差。

**ReLU6 只是限制了上界，没有副作用**：ReLU6 将输出限制在 6 以内，在大梯度情况下会截断梯度，可能限制模型表达能力。它并非 ReLU 的免费增强版，而是为量化部署做的权衡。训练时若使用 ReLU6 但最终在 GPU 上推理（不量化），其精度通常略低于 ReLU。

## 参见

- [[relu]] — ReLU 基础与 dying ReLU 详解
- [[activation-selection]] — 完整激活函数选型框架
- [[../initialization/initialization]] — 各激活函数对应的初始化方法
- [[../normalization/normalization]] — 激活函数与 BN/LN 的配合
- [[../quantization/]] — ReLU6 与量化部署
