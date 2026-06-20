# 子词分词（Subword Tokenization）

## 是什么

子词分词是 NLP 中将文本切分为比词小、比字符大的单元的方法。它的核心矛盾是：词级分词（word-level）会导致词表过大且无法处理未登录词，字符级分词（character-level）又会丢失语义信息并产生过长的序列。子词分词在两者之间取得平衡——它将常见词保留为完整单元，将罕见词拆分为已知的子词片段。

主流的三种子词分词算法：

- **BPE（Byte Pair Encoding）**：最初是数据压缩算法，被 Sennrich 等人引入 NLP。从字符级开始，反复合并出现频率最高的相邻 token 对，直到达到设定的词表大小。GPT、GPT-2、BART 等使用 BPE。
- **WordPiece**：与 BPE 类似但合并标准不同。WordPiece 选择合并能使语言模型似然增加最多的 token 对，而非最频繁的 token 对。BERT、DistilBERT 使用 WordPiece。
- **SentencePiece**：与 BPE 和 WordPiece 不同，SentencePiece 将输入视为 Unicode 字符序列，不需要预分词（pre-tokenization），因此更加语言无关。它内部可基于 BPE 或 Unigram Language Model。LLaMA、Gemma 等现代模型使用 SentencePiece。

| 特性 | BPE | WordPiece | SentencePiece |
|------|-----|-----------|---------------|
| 合并标准 | 频率最高 | 似然提升最大 | BPE 或 Unigram |
| 需要预分词 | 是 | 是 | 否 |
| 语言无关 | 部分 | 部分 | 完全 |
| 代表模型 | GPT, RoBERTa | BERT | T5, LLaMA |
| 原始文本处理 | 需要空格分词 | 需要空格分词 | Unicode 序列 |

## 为什么重要

子词分词是现代 NLP 模型的基石，它直接影响四个关键方面：

1. **词表大小**：典型子词词表为 32K ~ 64K，远小于词级分词的百万级，也大于字符级的几百级。这个规模在 GPU 显存和表示能力之间取得了良好的平衡——词表太大导致 Embedding 层参数过多，词表太小则序列过长。
2. **未登录词处理**：任何词都可以被拆分为已知子词片段，理论上没有 OOV（Out-of-Vocabulary）问题。例如 "embiggen" 可能被拆为 "em" + "big" + "gen"，即使模型从未见过这个词。
3. **语言适应性**：子词分词对不同语言的影响差异巨大。英语等字母语言受益于子词的模糊匹配能力；中文等表意语言则面临特殊挑战——单字已经包含丰富语义，字词边界模糊。
4. **训练速度**：词表大小直接影响 Embedding 层的参数量和 softmax 的计算量。子词分词显著减少了参数规模，使训练更快。

此外，分词方案的差异也是模型之间不可忽视的兼容性壁垒——不同模型的分词器不能互换使用，这也是 [[../../contradictory]] 中讨论的"看似相同实则不同"的典型例子。

## 如何使用

### 词表大小选择

- **大型模型**（>1B 参数）：通常用 50K ~ 64K。更大的词表意味着更少的 token 数/序列长度，但增加了 Embedding 和 softmax 参数。
- **中型模型**（100M ~ 1B）：32K 是常见选择。BERT-base（110M）使用 30K，RoBERTa 使用 50K。
- **小模型或特定领域**：8K ~ 16K。领域专用语料（如生物医学）的词表可以更小、更专注。
- **多语言模型**：需要更大的词表（100K ~ 250K 甚至更多），因为需要覆盖多种语言的字符和子词模式。XLM-R 使用 250K。

### 训练自己的分词器

使用 HuggingFace `tokenizers` 库训练：

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["corpus.txt"],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)
```

关键参数说明：
- **vocab_size**：词表大小，如前所述
- **min_frequency**：一个子词对至少出现多少次才被合并。默认 2，通常不需要调整
- **special_tokens**：特殊 token 的顺序会影响它们的 ID 分配，且一旦训练完成就不可随意更改

对于 SentencePiece，则要注意两个额外参数：
- `character_coverage`：模型覆盖的字符比例，多语言模型设为 0.9995，单语言可设为 0.999
- `split_by_whitespace`：是否基于空格切分。中文、日文等无空格语言应设为 false

### 对训练的影响

1. **序列长度**：子词分词的序列长度介于词级和字符级之间。英语平均每个词 1.2~1.5 个子词。
2. **信息密度**：不同 token 的信息量不同。常见的 token（如 "the"、"a"）几乎不携带信息，而罕见 token 的信息密度更高。现代模型用 [[../optimizer-lr/optimizer-lr]] 中的自适应方法来处理这种不均衡。
3. **分词的噪声**：分词本身是有损的。"unbelievable" 被拆为 "un" + "believe" + "able" 时丢失了单词整体的含义。但损失通常可以忽略，因为模型在下一层可以重建交互。

## 常见误区

1. **不同模型的分词器可互换**：不能。BERT 的 WordPiece 词表与 GPT 的 BPE 词表完全不同。用错分词器会产出无意义的 token ID 序列。务必找到模型对应的 tokenizer 文件。
2. **词表越大越好**：并非如此。词表增大会带来 Embedding 层的参数量线性增长（vocab_size × hidden_dim），且 softmax 的计算量也成正比增加。256K 词表的 Embedding 层就有 `256K × 768 ≈ 197M` 参数——几乎和 BERT-base 的其他部分一样多。
3. **忽略特殊 token**：`[CLS]`, `[SEP]`, `<s>`, `</s>`, `<pad>` 等特殊 token 的 ID 是分词器内部约定的。微调时如果自行修改了分词器配置却没有更新模型头，可能导致 embedding 错位。
4. **SentencePiece 不需要预分词＝直接喂原始文本**：虽然 SentencePiece 不依赖空格分词，但 Unicode 归一化（NFKC/NFD）对结果有显著影响。不同归一化方式可能产生不同的分词结果。
5. **在微调时重训分词器**：预训练模型的分词器是模型的一部分。微调时修改词表（增删 token）会破坏预训练权重——新增 token 的 Embedding 需要从头学，导致严重的冷启动问题。

## 参见

- [[bert-finetuning]] — BERT 微调中的词表与分词器使用
- [[text-data-augmentation]] — 分词结果的增强方法
- [[nlp]] — NLP 核心技巧概览
- [[../optimizer-lr/optimizer-lr]] — 分词的序列长度影响 batch 大小进而影响学习率选择
- [[../../contradictory]] — 分词对模型兼容性的影响讨论
