# BERT 微调指南

## 是什么

BERT（Bidirectional Encoder Representations from Transformers）及其变体（RoBERTa、DeBERTa、ALBERT 等）是 NLP 领域最广泛使用的预训练语言模型。BERT 微调指在特定下游任务上对预训练权重进行小学习率更新，使其适配目标任务的过程。微调的核心在于：用很小的学习率（通常 `1e-5` 级别）和特定的训练策略（warmup、衰减）来保持预训练学到的语言知识不被破坏，同时融入任务特有信号。

## 为什么重要

预训练语言模型参数量巨大（BERT-base 1.1 亿，BERT-large 3.4 亿），从零训练不仅成本高昂而且数据需求巨大。微调让从业者可以用少量标注数据获得强大性能。但微调非常敏感：学习率过高会导致灾难性遗忘（catastrophic forgetting），过低则无法有效适配下游任务。这也是为什么微调策略本身就是一个被广泛研究的课题——微调方式不对，预训练的收益可能完全丧失。

与视觉领域的微调不同（常见 `1e-4` 到 `1e-3`），BERT 类模型需要更低的学习率。这一差异源于预训练任务的特性：MLM（Masked Language Model）训练出的表示已经很强大，只需要"轻微引导"就能适配下游任务。此外，Transformer 的自注意力机制对参数扰动比 CNN 更敏感，过大的更新步长会迅速破坏已经学好的注意力模式。

## 如何使用

### 学习率

BERT 微调的典型学习率范围是 `1e-5` ~ `5e-5`。具体选择与任务类型相关：

- 文本分类（情感分析、主题分类）：`2e-5` 是常见起点
- 文本匹配（STS、NLI）：`1e-5` ~ `3e-5`，注意不要过大
- 序列标注（NER、POS）：`2e-5` ~ `5e-5`
- 阅读理解：`3e-5` ~ `5e-5`
- 生成任务（如 BART、T5 微调）：`3e-5` ~ `5e-5`

建议用对数刻度搜索：尝试 `[1e-5, 2e-5, 3e-5, 5e-5]`，对比验证集损失变化。

### Warmup

微调必须使用 warmup。推荐 warmup 步数占总步数的 `3%~10%`：

- 简单任务（句子分类）：`3%~5%`
- 复杂任务（阅读理解、长文本分类）：`5%~10%`
- 数据量较小：warmup 比例取更高值

Warmup 的作用是在训练初期用很小的学习率"预热"优化器状态（尤其是 Adam 的一阶/二阶动量估计），避免前期大步长导致预训练权重被急剧扰动。

### CLS Token 的使用

BERT 的 `[CLS]` 向量被设计为句子级表示，但在实际使用中需要注意：

1. **CLS 有效的情况**：在 BERT 预训练阶段，CLS 被用于 Next Sentence Prediction（NSP）任务，因此当你的下游任务与 NSP 类似（如句子对分类、蕴含判断）时，CLS 表现较好。
2. **CLS 不可靠的情况**：在文本匹配（语义相似度）任务中，CLS 的向量表示可能不如 pooling 后的平均表示。这是因为匹配任务需要细粒度的 token 级交互，而 CLS 经过多层压缩后丢失了局部信息。
3. **更好的替代方案**：使用所有 token 的 mean pooling 或加权求和（self-attention 聚合）。参考 [[../../contradictory]] 中对 pooling vs attention 的进一步讨论。

### Pooling 策略

| 策略 | 做法 | 适用场景 |
|------|------|----------|
| CLS | 取 `[CLS]` 位置的输出 | 分类、NLI |
| Mean Pooling | 所有 token 输出取平均 | 文本匹配、语义表示 |
| Max Pooling | 所有 token 输出取最大值 | 关键信息提取 |
| Weighted Pooling | 可学习的注意力加权求和 | 多任务、复杂场景 |
| First-Last Avg | 取前几层和后几层平均再拼接 | 更鲁棒的句子表示 |

实践中，mean pooling 是 CLS 之外的默认选择——简单、稳定、在文本匹配中常常优于 CLS。更复杂的选择是使用额外的 attention 层聚合所有 token，参考 [[nlp]] 中"序列聚合：Attention 优于简单 Pooling"一节。

## 常见误区

1. **学习率从 `1e-4` 开始**：这是最常见的错误。BERT 微调的合理范围很少超过 `5e-5`。从 `1e-4` 开始几乎必然导致损失爆炸或不收敛。如果看到训练损失在第一步就跳到 10+，基本是学习率过大。
2. **不做 warmup**：跳过 warmup 在短训练中可能看不出问题，但在长训练或大数据集上会导致前期损失震荡、收敛变慢。
3. **全程相信 CLS 向量**：CLS 不是万能的句子表示。在匹配和检索场景中，务必与 pooling 方法对比验证。
4. **冻结 BERT 只训练分类头**：对于非对齐任务（即预训练和下游差异较大的场景），只训练分类头可能效果很差。建议全参数微调，或使用 Layer-wise Learning Rate Decay（LLRD）。
5. **忽略序列长度**：超过 512 token 的输入会被 BERT 截断。如果有大量长文本，考虑使用 Longformer 等变体或分块策略（chunking + pooling）。
6. **单卡和分布式学习率不一致**：分布式训练时 batch size 成倍增大，应相应降低学习率（线性缩放法则：lr_new = lr_base × sqrt(batch_new / batch_base)）。

## 参见

- [[nlp]] — NLP 与序列任务核心技巧，包括 warmup 设置与学习率建议
- [[subword-tokenization]] — 子词分词对 BERT 词表设计的影响
- [[../optimizer-lr/optimizer-lr]] — 优化器与学习率调度，AdamW 权重衰减设置
- [[../regularization/regularization]] — 正则化方法：weight decay、dropout 在微调中的使用
- [[../../contradictory]] — Pooling vs Attention 的深入讨论
- [[embedding-perturbation]] — Embedding 对抗扰动（FGM / PGD），可与微调结合使用
