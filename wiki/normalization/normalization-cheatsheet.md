# 归一化方法速查表

## 快速选择

### 按架构选择

| 架构 | 推荐归一化 | 默认配置 | 备选 |
|------|-----------|---------|------|
| CNN（ResNet / VGG / DenseNet） | **BatchNorm** | Conv → BN → ReLU | GN（小 batch） |
| CNN + 小 batch（检测/分割） | **GroupNorm** | G = 32，Conv → GN → ReLU | IN（G = C） |
| Transformer Encoder（BERT） | **LayerNorm** | Pre-LN + Post-LN | RMS Norm |
| Transformer Decoder（GPT/Llama） | **LayerNorm (RMS Norm)** | Pre-LN，RMS Norm 加速 | 完整 LN |
| RNN / LSTM | **LayerNorm** | 每时间步/层后加 LN | — |
| Vision Transformer (ViT) | **LayerNorm** | 遵循 Transformer 规范 | — |
| GAN / 生成模型 | **InstanceNorm / GN** | 风格迁移用 IN，通用用 GN | BN 在大 batch 可用 |
| 强化学习（Policy Network） | **LayerNorm** | batch size 小 + 序列特征 | GN |
| 图神经网络 (GNN) | **LayerNorm / GraphNorm** | LN 通用，GraphNorm 专为图 | BN 在大 batch 可用 |

### 按 batch size 选择

| Batch Size | 推荐 | 原因 |
|-----------|------|------|
| >= 32 | BatchNorm | 统计量稳定，有正则化效果 |
| 16 - 32 | BatchNorm / GN | BN 仍有效，GN 安全选择 |
| 8 - 16 | BatchNorm / GN | BN 开始不稳定，GN 无差异 |
| 2 - 8 | GroupNorm | 检测/分割典型值 |
| 1 | GroupNorm / LayerNorm / InstanceNorm | BN 完全不可用 |

### 按任务选择

| 任务 | 推荐 | 说明 |
|------|------|------|
| 图像分类 | BN（大 batch）/ GN（小 batch） | ImageNet 标配 BN |
| 目标检测 | GN（batch size 受限） | Mask R-CNN 已验证 GN 优于 BN |
| 语义分割 | GN | 单样本高分辨率，batch 有限 |
| 文本分类 | LN | Transformer 标配 |
| 机器翻译 | LN | 变长序列 + 自回归生成 |
| 文本生成 | LN (RMS Norm) | GPT/Llama 系列 |
| 风格迁移 | IN | 实例级特征归一化 |
| 视频分类 | GN | 每帧消耗显存大 |
| 多模态（CLIP） | LN | Transformer backbone |

## 关键参数速查

| 方法 | 可学习参数 | 额外状态 | 推理行为 |
|------|-----------|---------|---------|
| BatchNorm | C × 2（gamma, beta） | C × 2（running mean/var） | 依赖 running stats |
| LayerNorm | C × 2（gamma, beta） | 无 | 训练推理一致 |
| GroupNorm | C × 2（gamma, beta） | 无 | 训练推理一致 |
| InstanceNorm | C × 2（gamma, beta） | C × 2（可选 running stats） | 训练推理一致 |
| RMS Norm | C × 1（仅 gamma，无 beta） | 无 | 训练推理一致 |

## 公式对比

假设输入形状为 `[N, C, H, W]`（4D 张量）：

| 方法 | 均值/方差计算维度 | 特点 |
|------|-----------------|------|
| BN | `(N, H, W)` | 每个通道独立，跨样本 |
| LN | `(C, H, W)` | 每个样本独立，跨通道 |
| GN | `(C/G, H, W)` | 每组内跨通道，跨空间 |
| IN | `(H, W)` | 每个通道独立，单样本 |

## 实现层面的注意事项

- **BN 的 running stats**：动量默认 0.1（PyTorch）或 0.99（TensorFlow），大 batch 时可降低动量增加近期统计权重。
- **Pre-LN vs Post-LN 的选择**：Pre-LN 训练更稳定，适合深层 Transformer（>12 层）；Post-LN 配合 warmup 在小模型中仍可使用。
- **GN 组数与通道数关系**：确保 `C % G == 0`，否则框架会警告或截断。
- **混合精度训练**：BN 在 FP16 下易溢出，建议使用 `torch.cuda.amp` 自动管理或改用 GN/LN。
- **ONNX 导出**：BN 的 running stats 导出最成熟；GN/LN 的 ONNX 支持因框架版本而异。

## 参见

- [[../normalization/normalization]] — 归一化层总览与核心技巧
- [[batch-normalization]] — BN 原理与使用详解
- [[layer-normalization]] — LN 与 Transformer 详解
- [[group-normalization]] — GN 在小 batch 场景详解
- [[../regularization/regularization]] — BN 的正则化效果
- [[../batch-size/batch-size]] — batch size 选型指南
- [[../cnn-cv/cnn-cv]] — CNN 架构中的归一化选择
- [[../nlp/nlp]] — NLP 任务中的归一化选择
- [[../llm-rl/llm-rl]] — 大模型训练中的 LN 配置
- [[../../contradictory]]
