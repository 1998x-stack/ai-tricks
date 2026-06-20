---
layout: default
title: 深度学习调参技巧 Wiki
description: 从知乎 30+ 位一线从业者回答中提取、分类、深化的结构化知识库
---

# 深度学习调参技巧 Wiki

> 覆盖从经典 CNN 到大模型 RLHF 的全栈调参方法论 — 72 个页面，350+ 条技巧，14 组矛盾分析

## 快速入口

### 基础调参

| 类别 | 描述 |
|------|------|
| [优化器与学习率](./wiki/optimizer-lr/optimizer-lr.md) | Adam, SGD+Momentum, 学习率调度, Weight Decay, LR Range Test |
| [批大小与梯度累积](./wiki/batch-size/batch-size.md) | 梯度累积, 泛化影响, 批大小-学习率缩放法则 |
| [初始化方法](./wiki/initialization/initialization.md) | Xavier, He, 初始化理论 |
| [归一化层](./wiki/normalization/normalization.md) | BatchNorm, LayerNorm, GroupNorm, 归一化速查表 |
| [激活函数](./wiki/activation/activation.md) | ReLU, 激活函数选择, LeakyReLU/PReLU |
| [正则化与防过拟合](./wiki/regularization/regularization.md) | Dropout, 过拟合/欠拟合诊断, 早停, Label Smoothing |

### 领域专项

| 类别 | 描述 |
|------|------|
| [CNN与视觉任务](./wiki/cnn-cv/cnn-cv.md) | 感受野, 跳跃连接, 深度可分离卷积, FPN, 3×3 法则 |
| [NLP与序列任务](./wiki/nlp/nlp.md) | BERT微调, Embedding对抗, 子词分词, 文本增强 |
| [推荐系统](./wiki/recsys/recsys.md) | 热启动, 负采样, 特征分箱, 多任务学习 |

### 大模型

| 类别 | 描述 |
|------|------|
| [大模型SFT](./wiki/llm-sft/llm-sft.md) | LoRA, QLoRA, SFT数据构造, Chat Template, 方案选型 |
| [大模型RL对齐](./wiki/llm-rl/llm-rl.md) | PPO-RLHF, DPO, Reward Hacking, KL散度, GRPO |

### 工程与诊断

| 类别 | 描述 |
|------|------|
| [基础设施与工程](./wiki/infra/infra.md) | 混合精度, 分布式训练, 梯度检查点, W&B Sweeps |
| [数据与评估](./wiki/data-eval/data-eval.md) | 交叉验证, Focal Loss, 数据增强, 类别不平衡, 评估指标 |
| [调试与故障排查](./wiki/debugging/debugging.md) | NaN诊断, 小数据过拟合测试, 梯度裁剪, 训练曲线诊断 |

### 专题

| 页面 | 描述 |
|------|------|
| [矛盾观点汇总](./contradictory.md) | 14 组跨作者观点冲突及分析 |

---

## 关于本项目

本知识库使用 [Obsidian](https://obsidian.md) 风格的双向链接（`[[wikilinks]]`）构建，推荐在 Obsidian 中作为 Vault 打开以获得最佳浏览体验。

- **72 个 Markdown 文件** — 14 个主页面 + 58 个知识点深度页面
- **1200+ 交叉引用** — 连接相关知识，构建知识网络
- **全中文内容** — 保留原始技术术语和行业表达
- **来源可追溯** — 每条技巧标注原始作者

[GitHub 仓库](https://github.com/1998x-stack/ai-tricks) · [矛盾观点](./contradictory.md) · [原始素材](./raw/tricks.md)
