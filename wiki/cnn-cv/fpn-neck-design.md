# FPN与Neck设计

## 是什么

特征金字塔网络（Feature Pyramid Network, FPN）是一种多尺度特征融合架构，通过自顶向下路径和横向连接将深层语义信息传递到浅层高分辨率特征图中。

标准 FPN 包含两条路径：

- **自底向上路径（Bottom-Up Pathway）**：backbone 前向传播，以 ResNet stage 为单位生成不同分辨率特征图 `{C2, C3, C4, C5}`。分辨率逐级减半（1/4, 1/8, 1/16, 1/32），通道数逐级增加（64, 128, 256, 512）。

- **自顶向下路径（Top-Down Pathway）**：从 C5 开始，通过 2× 上采样（最近邻插值）将深层特征放大到上一级分辨率，与横向连接的 1×1 卷积输出逐元素相加。上采样 → 相加 → 3×3 卷积消除混叠效应，生成输出 `{P2, P3, P4, P5}`，每层通道统一为 256。

- **横向连接（Lateral Connections）**：1×1 卷积将自底向上特征降维到 256 通道，与上采样后的深层特征维度匹配。使用 1×1 而非 3×3，是因为降维同时避免引入额外空间混叠。

**主要 Neck 变体**：

| 变体 | 核心改进 | 代表工作 |
|------|----------|----------|
| FPN | 自顶向下 + 横向连接 | Mask R-CNN |
| BiFPN | 双向融合 + 可学习权重 | EfficientDet |
| PAFPN | 自底向上增强路径 | PANet |
| NAS-FPN | NAS 自动搜索融合拓扑 | NAS-FPN |
| Simple-FPN | 简化横向连接 | YOLOX |
| Recursive FPN | 递归深度融合 | RFP |

**BiFPN 的独特设计**：
- 可学习权重：`output = Σ(w_i × input_i) / (ε + Σw_j)`，通过 ReLU 确保非负。
- 双向融合：先自顶向下、再自底向上反复多次（通常 3-5 轮），加深特征交互。
- 节点删除：移除只有单输入边的节点。
- 额外跳跃连接：同尺度输入与输出间添加跳跃连接。

**ASFF（Adaptively Spatial Feature Fusion）**：为每个 FPN 输出层的每个空间位置学习软融合权重，使网络在空间维度上自适应选择各层贡献。遮挡严重、目标密集场景下，ASFF 的逐位置加权比 BiFPN 的全局权重更有效。

**YOLO 系列的 FPN 演化**：YOLOv3 首次引入 FPN；YOLOv4 使用 PANet；YOLOv5 优化 PANet 的 CSP 结构；YOLOX 简化 PANet 并可选项 ASFF；YOLOv8 采用 C2f 模块结合 PAN-FPN。清晰展示了 FPN 从检测标配到持续优化的过程。

**Neck 与 Head 交互**：FPN 各层独立预测，但跨层 head（将所有 FPN 层 concat 后预测或使用 Transformer encoder-decoder）可缓解各层预测不一致。Query-based 检测器（DETR、Deformable DETR）用单个 query 集合从所有 FPN 层采样，是隐式 neck-head 融合。

**特征对齐问题**：不同下采样倍率特征图存在亚像素错位。解决方案包括双线性插值替代最近邻上采样、融合后 3×3 消除混叠、可变形卷积学习偏移。

## 为什么重要

FPN 有效解决了单一特征图难以同时检测大小差异显著目标的经典问题。

**解决尺度变化挑战**：图像中目标尺度可相差数十倍（航拍车辆 vs 建筑物）。浅层分辨率高、感受野小（参见 `[[../cnn-cv/receptive-field]]`），适合小目标；深层语义强，适合大目标。FPN 让每层兼具高分辨率和强语义。

**提升小目标检测**：`[[../cnn-cv/cnn-cv]]` 强调："基于 backbone 构建层次化的 neck 一般都比直接使用最后一层输出要好。" Mask R-CNN 使用 FPN 后小目标 AP 提升超 10 个百分点。

**Neck 性价比高**：好的 neck 设计可显著提升精度，计算开销相对可控。工业级检测系统中 neck 改进比扩大 backbone 更划算。

**语义传播**：深层语义通过上采样逐级传播到浅层，解决浅层特征语义不足问题。对全景分割等需要理解整体场景的任务至关重要。

**FPN 的局限性**：FPN 假设各层特征在融合前已经良好对齐（空间位置），但实际上不同分辨率的感受野不同，融合后特征可能存在空间歧义。BiFPN 的可学习权重和 ASFF 的逐位置加权都在努力缓解这一问题。

## 如何使用

**标准 FPN 使用**：检测任务用 ResNet-50/101 backbone，C3~C5 作为 FPN 输入，输出 P3~P7（P6/P7 从 P5 经 stride=2 卷积生成），通道统一 256。每层独立预测。ResNet-50-FPN 是 COCO 检测经典基线。

**分支保留原则**：`[[../cnn-cv/cnn-cv]]` 警告："目标检测不能盲目去掉 fpn 结构。尽管你分析出某个分支的 Anchor 基本不可能会对你预测的目标起作用，但直接去掉分支很可能会带来漏检。" 即使某分支 anchor 与数据集不匹配也不应移除——该分支为特征融合提供了语义流通路径。实践中发现 P2 对某些数据集的直接贡献小，但去掉后 P3 的 AP 也会下降。

**BiFPN 策略**：`[[../cnn-cv/cnn-cv]]` 明确说："BiFPN 真的有用。" 使用 3-5 次重复融合，更多轮次收益递减。可学习权重在训练前 1/3 轮次波动大，随后稳定。适合中大型模型。轻量版（1-2 次融合）在小模型上可与标准 FPN 相当。

**何时简化 FPN**：
- 单尺度任务（分类）——无需多尺度特征。
- 轻量级检测（边缘部署）——Simple-FPN 或单层预测。
- 目标尺度极窄——单层预测可能足够。
- 计算极度受限——极简 neck。

**层次化 Neck 设计**：reduce function 中"attention 优于简单 pooling"（`[[../cnn-cv/cnn-cv]]`）。使用 SE 模块或轻量注意力跨尺度聚合，效果优于简单拼接或相加。通道统一后，FFN 风格 MLP 先升维再降维优于单纯卷积。attention 机制可动态分配各尺度特征权重，轻量 SE-block 即可，不需要完整 Transformer。

## 常见误区

1. **FPN 层数越多越好**：P2 含大量背景噪声引入误检。P7 对小数据集几乎无意义。基于目标尺寸分布选择有效范围（典型 P3~P6）。

2. **BiFPN 始终优于标准 FPN**：BiFPN 重复融合增加 FLOPs，小 backbone 场景不一定值得。EfficientDet-D0 的 BiFPN 优势比 D7 小得多。

3. **FPN 只能用于检测**：FPN 是多尺度融合通用方案——语义分割（DeepLabv3+）、实例分割（Mask R-CNN）、关键点检测（HigherHRNet）中同样有效。

4. **横向连接必须用 1×1**：1×1 最常用。也可用 3×3（更强变换但更大计算量）或恒等映射（通道已匹配）。

5. **BiFPN 权重自动解决一切**：权重学习依赖初始化和数据量。数据过少可能不收敛。极不均衡权重暗示融合设计有问题。

6. **FPN 与 stronger backbone 收益累加**：backbone 已有多尺度融合时（如 HRNet）叠加 FPN 收益递减。

7. **Neck 可独立于 Head 优化**：Neck 和 Head 高度耦合。BiFPN 与 EfficientDet 的 shared head 配合最佳，迁移到其他 head 效果可能打折扣。

## 参见

- `[[../cnn-cv/cnn-cv]]` — FPN 与 Neck 设计、Anchor 设计章节
- `[[../cnn-cv/receptive-field]]` — 感受野与多尺度设计
- `[[../cnn-cv/skip-connections]]` — 跳跃连接与特征融合
- `[[../cnn-cv/3x3-conv-rule]]` — 3×3 卷积与特征提取
- `[[../cnn-cv/depthwise-separable-conv]]` — 轻量特征提取
- `[[../normalization/batch-normalization]]` — BN 与特征归一化
- `[[../activation/relu]]` — ReLU 激活函数
- `[[../../contradictory]]` — 矛盾观点
