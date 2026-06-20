# 深度学习调参技巧 Wiki

> 从知乎「深度学习调参有哪些技巧？」问题下 30+ 位一线从业者的回答中提取、分类、深化的结构化知识库 —— 覆盖从经典 CNN 到大模型 RLHF 的全栈调参方法论。

[![Pages](https://img.shields.io/badge/GitHub_Pages-在线浏览-blue)](https://1998x-stack.github.io/ai-tricks/)
[![Obsidian](https://img.shields.io/badge/Obsidian-兼容-7C3AED)](https://obsidian.md)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Files](https://img.shields.io/badge/Wiki页面-72个-orange)](./wiki/)

---

## 是什么

这是一个**深度学习调参的结构化知识库**，原始素材来自知乎问题「[深度学习调参有哪些技巧？](https://www.zhihu.com/question/25097993)」（14,000+ 关注者，164 个回答）中多名一线从业者的实战经验。我们从原始回答中：

1. **提取** 500+ 条可操作的调参技巧
2. **分类** 为 14 个主题领域
3. **深化** 为 58 个知识点的详细解释
4. **发现** 14 组相互矛盾的观点并逐一分析

结果是一个适合日常查阅、系统学习和团队共享的**知识网络**。

## 快速导航

| 类别 | 核心知识点 |
|------|-----------|
| [优化器与学习率](./wiki/optimizer-lr/optimizer-lr.md) | Adam, SGD+Momentum, 学习率调度, Weight Decay, LR Range Test |
| [批大小与梯度累积](./wiki/batch-size/batch-size.md) | 梯度累积, 泛化影响, 缩放法则 |
| [初始化方法](./wiki/initialization/initialization.md) | Xavier, He, 初始化理论 |
| [归一化层](./wiki/normalization/normalization.md) | BatchNorm, LayerNorm, GroupNorm, 速查表 |
| [激活函数](./wiki/activation/activation.md) | ReLU, 激活函数选择, LeakyReLU/PReLU |
| [正则化与防过拟合](./wiki/regularization/regularization.md) | Dropout, 过拟合/欠拟合, 早停, Label Smoothing |
| [CNN与视觉任务](./wiki/cnn-cv/cnn-cv.md) | 感受野, 跳跃连接, 深度可分离卷积, FPN, 3×3法则 |
| [NLP与序列任务](./wiki/nlp/nlp.md) | BERT微调, Embedding对抗, 子词分词, 文本增强 |
| [推荐系统](./wiki/recsys/recsys.md) | 热启动, 负采样, 特征分箱, 多任务学习 |
| [大模型SFT](./wiki/llm-sft/llm-sft.md) | LoRA, QLoRA, SFT数据构造, Chat Template, 方案选型 |
| [大模型RL对齐](./wiki/llm-rl/llm-rl.md) | PPO-RLHF, DPO, Reward Hacking, KL散度, GRPO |
| [基础设施与工程](./wiki/infra/infra.md) | 混合精度, 分布式训练, 梯度检查点, W&B Sweeps |
| [数据与评估](./wiki/data-eval/data-eval.md) | 交叉验证, Focal Loss, 数据增强, 类别不平衡, 评估指标 |
| [调试与故障排查](./wiki/debugging/debugging.md) | NaN诊断, 小数据过拟合测试, 梯度裁剪, 训练曲线诊断 |

另见：[矛盾观点汇总](./contradictory.md) — 14 组跨作者观点冲突及其分析

## 特点

- **实战导向** — 每条技巧来自一线从业者的真实经验，非教科书复述
- **深度解释** — 58 个知识页面覆盖「是什么、为什么重要、如何使用、常见误区」
- **可交叉引用** — 1200+ Obsidian 风格 [[双向链接]] 连接相关知识
- **矛盾追踪** — `contradictory.md` 记录了 14 组观点冲突并附分析
- **中文本地化** — 保留原始中文表达的技术术语和行业黑话
- **持续可扩展** — 目录结构支持添加新的类别和知识点

## 如何使用

### 在 Obsidian 中打开（推荐）
```bash
# 克隆仓库
git clone https://github.com/1998x-stack/ai-tricks.git

# 在 Obsidian 中作为 Vault 打开
# File → Open Vault → 选择 ai-tricks 目录
```

### 在 GitHub 上浏览
直接通过 GitHub 的 Markdown 渲染浏览所有页面，链接可直接点击跳转。

### 在任意 Markdown 编辑器中查看
所有文件均为标准 Markdown，兼容 VS Code、Typora、Notion 等工具（Obsidian 特有语法 `[[wikilinks]]` 在部分工具中可能显示为纯文本）。

## 项目结构

```
ai-tricks/
├── README.md                     # 项目主页
├── contradictory.md              # 14 组矛盾观点及分析
├── wiki/                         # 14 个主题 × 知识页面
│   ├── optimizer-lr/             # 优化器与学习率 (6 文件)
│   ├── batch-size/               # 批大小与梯度累积 (4 文件)
│   ├── initialization/           # 初始化方法 (4 文件)
│   ├── normalization/            # 归一化层 (5 文件)
│   ├── activation/               # 激活函数 (4 文件)
│   ├── regularization/           # 正则化与防过拟合 (5 文件)
│   ├── cnn-cv/                   # CNN与视觉任务 (6 文件)
│   ├── nlp/                      # NLP与序列任务 (5 文件)
│   ├── recsys/                   # 推荐系统 (5 文件)
│   ├── llm-sft/                  # 大模型SFT (6 文件)
│   ├── llm-rl/                   # 大模型RL对齐 (6 文件)
│   ├── infra/                    # 基础设施与工程 (5 文件)
│   ├── data-eval/                # 数据与评估 (6 文件)
│   └── debugging/                # 调试与故障排查 (5 文件)
└── raw/
    └── tricks.md                 # 原始知乎抓取数据
```

## 统计

| 指标 | 数值 |
|------|------|
| 主题类别 | 14 |
| 知识点页面 | 58 |
| 总文件数 | 72 |
| 总行数 | ~9,000 |
| 双向链接 | 1,200+ |
| 技巧条目 | 350+ |
| 引述作者 | 30+ |
| 矛盾观点 | 14 组 |

## 贡献

欢迎提交 PR 补充新的调参技巧、修正错误或扩展知识点。请保持：
- 中文内容
- 五段式结构（是什么 / 为什么重要 / 如何使用 / 常见误区 / 参见）
- Obsidian 双向链接语法
- 标注来源作者

## 声明

本知识库内容来源于知乎公开回答，原始版权归各位原作者所有。本项目仅做提取、分类和系统化整理，供学习参考使用。如有版权问题，请联系处理。

---

<p align="center">
  <sub>用 <a href="https://claude.ai/code">Claude Code</a> 构建 · 2026</sub>
</p>
