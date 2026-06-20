# Group Normalization

## 是什么

Group Normalization（GN）于 2018 年由吴育昕和何恺明（FAIR）提出，专门解决小 batch size 场景下 BN 失效的问题。GN 将通道分为若干组，在 **每组内的通道和空间维度** 上计算均值和方差，不依赖 batch 维度：

$$
\mu_g = \frac{1}{(C/G)HW} \sum_{c \in \text{group}} \sum_{h,w} x_{g,c,h,w}
\quad
\sigma_g^2 = \frac{1}{(C/G)HW} \sum_{c \in \text{group}} \sum_{h,w} (x_{g,c,h,w} - \mu_g)^2
$$

其中 $G$ 是组数，$C/G$ 是每组通道数。GN 同样使用 $\gamma$ 和 $\beta$ 做仿射变换。

### 与 BN、IN 的关系

从归一化维度角度看，BN、GN、LN、IN 构成了一个连续谱系：

| 方法 | 归一化维度 | batch 依赖 | 组数 G |
|------|-----------|-----------|--------|
| BatchNorm | (N, H, W) | 是 | G = 1（跨样本全通道） |
| GroupNorm | (G×C/G, H, W) = (C, H, W) 分组 | 否 | G 为超参数 |
| LayerNorm | (C, H, W) | 否 | G = 1（单样本全通道） |
| InstanceNorm | (H, W) | 否 | G = C（单样本单通道） |

**当 G = 1 时，GN = LN；当 G = C 时，GN = IN。** GN 通过调节组数 $G$ 在 LN 和 IN 之间平滑插值，默认 G = 32。

## 为什么重要

- **小 batch 的救星**：batch size = 1-8 时 BN 统计量噪声大，GN 完全不受影响。这对视频理解、3D 医学图像、目标检测等显存受限任务至关重要。
- **检测/分割任务首选**：Mask R-CNN 中 BN 的 batch 维度是每张图的 RoI 数（很小的 batch），GN 直接提升 2-3 个点 mAP。
- **不增加推理延迟**：训练和推理完全一致，无 running statistics 开销。
- **何恺明背书**：在 FAIR 的检测/分割框架中，GN 已取代 BN 成为默认归一化层。

## 如何使用

### 组数选择

- **默认 G = 32**：这是原始论文推荐值，适用于通道数较多的深层网络。
- **G = 16**：通道数较小的浅层网络（如第一层卷积只有 3 或 16 通道）。
- **相邻通道的语义相关性**：GN 假设相邻通道有相关性（分组内统计才有意义），通道随机打乱效果下降。
- **实用建议**：如果通道数 C 不能被 G 整除，框架会自动取最近的可整除值。

### 适用场景

| 场景 | 推荐 | 原因 |
|------|------|------|
| 检测/分割（Mask R-CNN, YOLO） | GN | batch size = 1-2 per GPU |
| 视频理解 | GN | 每帧消耗显存大，batch 小 |
| 3D 医学图像 | GN | batch size 往往为 1 |
| 风格迁移 | IN（G = C） | 需要保持实例特征 |
| 通用 CNN，大 batch | BN | BN 在大 batch 时优于 GN |
| Transformer 视觉（ViT） | LN | 序列范式，通道维度规整 |

### 与 BN 的效果对比

- **batch size >= 32**：BN 优于 GN（BN 的 batch 统计量更准确，且有正则化效果）。
- **batch size = 8-16**：GN 与 BN 相当，BN 仍有竞争力。
- **batch size <= 4**：GN 显著优于 BN（BN 误差 > 10%）。
- **batch size = 2（检测典型值）**：GN 比 BN 提升约 2% mAP。

### 代码示例（PyTorch）

```python
# GN: G = 32 groups, 或手动指定
self.conv_block = nn.Sequential(
    nn.Conv2d(in_c, out_c, 3, padding=1),
    nn.GroupNorm(num_groups=32, num_channels=out_c),
    nn.ReLU(inplace=True),
)
```

## 常见误区

- **GN 在所有情况下优于 BN**：大 batch 时 BN 有 batch 噪声带来的正则化优势，且工程生态更成熟（推理优化、ONNX 导出等）。
- **GN 越少组越好**：组数太少（G=1）退化为 LN，在 CNN 中效果下降；组数太多（G=C）退化为 IN，丢失通道间信息。默认 32 组是平衡点。
- **GN 不需要 $\gamma$ 和 $\beta$**：需要。去掉仿射变换后表征能力下降，实验证明有 0.5-1% 精度损失。
- **GN 和 BN 可以无缝替换**：替换后可能需要调整学习率和 weight decay。GN 没有 BN 的 running statistics 问题，训练更稳定但收敛速度可能略慢。
- **GN 可用于 Transformer**：Transformer 的序列结构适合 LN，GN 在视觉 Transformer（ViT）中实验效果不如 LN。

## 参见

- [[../normalization/normalization]] — 归一化层总览与核心技巧
- [[batch-normalization]] — 大 batch 场景的标准选择
- [[layer-normalization]] — Transformer 系列的归一化方案
- [[../cnn-cv/cnn-cv]] — CNN 架构选型参考
- [[../batch-size/batch-size]] — batch size 对训练的影响
- [[../../contradictory]]
