# NLP与序列任务

## 概述
NLP与序列任务在调参上有许多独特之处：序列结构需要特定的归一化方式（LN而非BN）、聚合方式（Attention优于Pooling），而预训练语言模型（BERT、GPT等）则带来了微调学习率、对抗训练、子词分词等一系列专项技巧。

## 详细知识

- [[bert-finetuning]] — BERT微调完整指南：学习率、CLS使用、pooling策略
- [[embedding-perturbation]] — FGM/PGD对抗训练：Embedding层扰动提升鲁棒性
- [[subword-tokenization]] — 子词分词：BPE/WordPiece/SentencePiece原理与选择
- [[text-data-augmentation]] — NLP数据增强：同义词替换、回译、随机删除等

## 核心技巧

### BERT微调学习率
- **来源**: 爱睡觉的KKY
- **适用场景**: BERT/RoBERTa/DeBERTa等预训练语言模型微调
- **原文**: *"lr ，最重要的参数，一般nlp bert类模型在1e-5级别附近，warmup，衰减"*
- **要点**: BERT类模型微调学习率通常在 1e-5 ~ 5e-5 范围，从零训练Transformer时常见起点为 1e-4 ~ 3e-4。务必配合warmup（约总步数 3%~10%）和cosine/linear衰减。
- **参见**: [[../optimizer-lr/optimizer-lr]], [[../optimizer-lr/learning-rate-scheduling]]

### 不要迷信CLS向量
- **来源**: Tang AI
- **适用场景**: 文本匹配、句子对分类等使用BERT做表征的任务
- **原文**: *"bert不要太过于相信cls的embedding能力，还是要看看它有没有做过相关任务，特别对于文本匹配场景"*
- **要点**: CLS token 的向量表示并非万能，尤其在文本匹配场景中需要验证其是否经过相关任务训练；可考虑使用 pooling 或 attention 聚合所有 token 的表示替代 CLS。
- **参见**: [[../recsys/multi-task-recsys]]

### Embedding对抗扰动（FGM / PGD）
- **来源**: Tang AI
- **适用场景**: NLP 分类、序列标注、文本匹配等任务的泛化提升
- **原文**: *"对于nlp任务，采用embedding扰动的方式有奇效，比如Fast Gradient Method（FGM）和Projected Gradient Descent（PGD）"*
- **要点**: 在 embedding 层添加对抗扰动（FGM 单步、PGD 多步），可视为一种数据增强，能显著提升模型鲁棒性和泛化能力，实现成本低。
- **参见**: [[text-data-augmentation]], [[../regularization/regularization]]

### Bi-Attention / Co-Attention 用于机器阅读理解
- **来源**: Tang AI
- **适用场景**: 机器阅读理解（MRC）、文本匹配、推理任务
- **原文**: *"对于机器阅读任务，在bert层后加bi-attention或者coattention有奇效"*
- **要点**: 在 BERT 编码层之后添加双向注意力（bi-attention）或协同注意力（co-attention）机制，能显式建模两个序列间的交互，对 MRC 任务提升显著。
- **参见**: [[../recsys/multi-task-recsys]]

### Longformer 优于 BERT 变体处理长文本
- **来源**: Tang AI
- **适用场景**: 长文档分类、长文本序列标注、长文本匹配
- **原文**: *"对于长文本来说longformer的效果高于各种bert变体"*
- **要点**: Longformer 使用稀疏注意力机制，支持更长序列（4096+），在处理长文本任务时效果优于标准 BERT 及其变体（RoBERTa、ALBERT 等）。BERT 的 512 长度限制对长文档是明显瓶颈。
- **参见**: [[../recsys/multi-task-recsys]], [[transformer-variants]]

### 序列特征使用 LayerNorm，非序列使用 BatchNorm
- **来源**: 爱睡觉的KKY / Tang AI
- **适用场景**: 序列模型的特征归一化
- **原文**: *"序列输入上LN，非序列上BN"* / *"序列特征才用layernorm，正常用batchnorm"*
- **要点**: 序列数据（文本、时序）因为长度可变且依赖 batch 统计量不稳定，应使用 LayerNorm；而图像等非序列数据使用 BatchNorm。错误使用 BN 会导致收敛困难。
- **参见**: [[../normalization/normalization]]

### 序列聚合：Attention 优于简单 Pooling
- **来源**: 爱睡觉的KKY
- **适用场景**: 序列特征聚合为定长向量（分类、匹配）
- **原文**: *"reduce function 一般attention 优于简单pooling"*
- **要点**: 将序列 token 表示聚合成一个向量时，self-attention 或 cross-attention 加权求和的效果优于 mean pooling 或 max pooling；在多任务场景下需要为不同任务构建独立的 QKV。
- **参见**: [[../recsys/multi-task-recsys]], [[../recsys/multi-task-recsys]]

### RNN 序列激活函数优先选用 tanh
- **来源**: hyshhh
- **适用场景**: RNN、LSTM、GRU 等循环神经网络
- **原文**: *"构建序列神经网络（RNN）时要优先选用tanh激活函数"*
- **要点**: RNN 内部隐状态更新时 tanh 优于 ReLU，因为 tanh 的输出范围 (-1, 1) 能有效缓解梯度爆炸且提供稳定的梯度流；ReLU 在 RNN 中容易导致激活值失控。
- **参见**: [[activation-functions]]

### Word Embedding 初始化
- **来源**: Jarvix / 爱睡觉的KKY
- **适用场景**: 词向量层初始化
- **原文**: *"embedding 一般选择截断 normalize"* / 使用 uniform 初始化替换 xavier 后训练速度和效果显著提升
- **要点**: Word Embedding 层推荐使用截断正态分布（truncated normal）或均匀分布（uniform）初始化，不要无脑用 xavier。正确的初始化方法对 embedding 质量影响巨大。
- **参见**: [[../initialization/initialization]]

### 文本数据增强
- **来源**: Tang AI
- **适用场景**: NLP 分类、序列标注、文本生成等任务的数据扩充
- **原文**: *"在nlp领域可以用大量数据增强方式-比如，同义词替换，随机抛弃词语，句子改写，句子转译等"*
- **要点**: NLP 数据增强的常用方法包括：同义词替换（SR）、随机词语丢弃（RD）、回译（back-translation）、句子改写。这些方法在数据量不足时能显著提升模型泛化能力。
- **参见**: [[text-data-augmentation]]

### Transformer Warmup
- **来源**: LindgeWAI / BugBuster喵
- **适用场景**: Transformer 系列模型（BERT、GPT、LLaMA 等）的训练
- **原文**: *"lr_scheduler的warmup步数设置：一般遵循模型复杂，步数多点；数据集大，步数少点 (参考BERT，默认设成总训练轮次的1/10，也可1/20)"*
- **要点**: Transformer 训练基本必须开启 warmup。推荐 warmup 步数为总步数的 3%~10%（复杂模型取高值，大数据集取低值）。warmup + cosine/linear decay 是最常见的搭配。warmup 的作用是防止初期大学习率导致的不稳定。
- **参见**: [[../optimizer-lr/optimizer-lr]], [[transformer-training]]

### 子词分词（BPE / SentencePiece）
- **来源**: LindgeWAI
- **适用场景**: NLP 模型词表构建与文本预处理
- **原文**: *"子词分割方法（如字节对编码 BPE、Unigram 语言模型等）可以将词汇切分成更小的单元，从而减少词表规模，同时保留词汇的语义信息"*
- **要点**: 使用 BPE 或 SentencePiece 进行子词分词可有效控制词表规模（通常 32K~64K），覆盖未登录词，并适配任意语言。SentencePiece 将原始文本视为 Unicode 序列，无需预分词。
- **参见**: [[tokenization]]

### 序列分桶策略（Bucketing）
- **来源**: LindgeWAI
- **适用场景**: 序列模型训练效率优化
- **原文**: *"分桶策略可以根据输入序列的长度将数据分成不同的桶，从而在训练过程中更高效地利用计算资源。这种方法可以减少 padding 的数量，提高训练效率。"*
- **要点**: 将相同或相近长度的序列分到同一 batch，减少无效 padding 的计算量。可用 `bucket_by_sequence_length` 实现，显著提升吞吐量。
- **参见**: [[training-efficiency]]

### LSTM 参数与 Word2Vec 初始化
- **来源**: 摘星狐狸
- **适用场景**: LSTM、BiLSTM 等序列模型
- **原文**: *"调整LSTM参数与Word2Vec初始化"*
- **要点**: LSTM 的 hidden size 和层数需要根据任务复杂度调整；使用预训练的 Word2Vec 或 GloVe 初始化 embedding 层可以加速收敛。在预训练语言模型普及前这是一个关键技巧，现在仍适用于资源受限或特定领域场景。
- **参见**: [[../initialization/initialization]], [[word-embedding]]

### NLP 场景优先使用 Adam 优化器
- **来源**: 爱睡觉的KKY
- **适用场景**: NLP 模型训练
- **原文**: *"优化器，nlp，抽象层次较高或目标函数非常不平滑的问题adam优先"*
- **要点**: NLP 任务的损失函数通常较为不平滑，Adam 的自适应学习率机制相比 SGD 能更快收敛且更稳定。推荐 AdamW（带解耦权重衰减）作为默认选择。
- **参见**: [[../optimizer-lr/optimizer-lr]]

### 梯度裁剪对 RNN / Transformer 的重要性
- **来源**: BugBuster喵 / 摘星狐狸
- **适用场景**: RNN、LSTM、Transformer 等深度序列模型训练
- **原文**: *"梯度裁剪建议默认加，尤其是 RNN、Transformer、大模型微调，clip norm 1.0 是常见起点"* / *"梯度归一、梯度裁剪一定很重要"*
- **要点**: 序列模型（尤其是 RNN 和 Transformer）在长序列训练时容易出现梯度爆炸，梯度裁剪（gradient norm clipping）是标准且有效的解决方案，建议 clip norm 设为 1.0 作为起点。
- **参见**: [[../optimizer-lr/optimizer-lr]], [[training-stability]]
