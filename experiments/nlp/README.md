# nlp 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [nlp-001-01](nlp-001-01/README.md) BERT微调学习率 | `optimizer.lr` | `0.001` → `2e-05` |
| [nlp-002-01](nlp-002-01/README.md) 不要迷信CLS向量 | `model.pooling` | `cls` → `masked_mean` |
| [nlp-002-02](nlp-002-02/README.md) 不要迷信CLS向量 | `model.pooling` | `cls` → `masked_max` |
| [nlp-003-01](nlp-003-01/README.md) Embedding对抗扰动（FGM / PGD） | `training.adversarial` | `none` → `fgm` |
| [nlp-003-02](nlp-003-02/README.md) Embedding对抗扰动（FGM / PGD） | `training.adversarial` | `fgm` → `pgd` |
| [nlp-004-01](nlp-004-01/README.md) Bi-Attention / Co-Attention 用于机器阅读理解 | `model.cross_attention` | `False` → `True` |
| [nlp-004-02](nlp-004-02/README.md) Bi-Attention / Co-Attention 用于机器阅读理解 | `model.cross_attention_kind` | `none` → `co_attention` |
| [nlp-005-01](nlp-005-01/README.md) Longformer 优于 BERT 变体处理长文本 | `model.attention_window` | `128` → `512` |
| [nlp-006-01](nlp-006-01/README.md) 序列特征使用 LayerNorm，非序列使用 BatchNorm | `model.norm` | `batch` → `layer` |
| [nlp-007-01](nlp-007-01/README.md) 序列聚合：Attention 优于简单 Pooling | `model.pooling` | `masked_mean` → `attention` |
| [nlp-008-01](nlp-008-01/README.md) RNN 序列激活函数优先选用 tanh | `model.rnn_activation` | `relu` → `tanh` |
| [nlp-009-01](nlp-009-01/README.md) Word Embedding 初始化 | `model.embedding_init` | `xavier_uniform` → `truncated_normal` |
| [nlp-010-01](nlp-010-01/README.md) 文本数据增强 | `data.augmentation` | `none` → `synonym_replace` |
| [nlp-010-02](nlp-010-02/README.md) 文本数据增强 | `data.augmentation` | `none` → `word_dropout` |
| [nlp-010-03](nlp-010-03/README.md) 文本数据增强 | `data.augmentation` | `none` → `back_translation` |
| [nlp-011-01](nlp-011-01/README.md) Transformer Warmup | `optimizer.warmup_ratio` | `0.0` → `0.1` |
| [nlp-012-01](nlp-012-01/README.md) 子词分词（BPE / SentencePiece） | `data.tokenizer_kind` | `word` → `bpe` |
| [nlp-012-02](nlp-012-02/README.md) 子词分词（BPE / SentencePiece） | `data.tokenizer_kind` | `bpe` → `unigram` |
| [nlp-013-01](nlp-013-01/README.md) 序列分桶策略（Bucketing） | `data.batch_order` | `random` → `length_bucket` |
| [nlp-014-01](nlp-014-01/README.md) LSTM 参数与 Word2Vec 初始化 | `model.embedding_source` | `random` → `word2vec` |
| [nlp-014-02](nlp-014-02/README.md) LSTM 参数与 Word2Vec 初始化 | `model.recurrent_init` | `normal` → `orthogonal_svd` |
| [nlp-015-01](nlp-015-01/README.md) NLP 场景优先使用 Adam 优化器 | `optimizer.kind` | `sgd` → `adam` |
| [nlp-016-01](nlp-016-01/README.md) 梯度裁剪对 RNN / Transformer 的重要性 | `training.clip_norm` | `0.0` → `1.0` |
