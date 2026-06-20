# 正则化与防过拟合

## 概述

正则化是对抗过拟合的核心手段，其首要原则是**先过拟合再正则化**——先确保模型容量足够拟合训练集，再通过正则化手段缩小模型与测试集之间的泛化差距。正则化的优先级是：数据 > Dropout > Weight Decay > 模型裁剪，诊断依据始终是训练集与验证集 loss 的差值。

---

## 详细知识

- [[dropout]] — Dropout深入解析：训练vs推理、比例选择、与BN的配合
- [[overfitting-underfitting]] — 过拟合与欠拟合诊断框架：训练/验证loss差值解读
- [[early-stopping]] — 早停策略：耐心值选择、ASHA/Hyperband、何时不要过早停止
- [[label-smoothing]] — 标签平滑：公式、作用机制、与知识蒸馏的交互

## 核心技巧

### 先过拟合再正则化（Overfit First）
- **来源**: 爱睡觉的KKY / 清明 / BugBuster喵 / TNotes / 匿名
- **适用场景**: 所有深度学习任务，尤其是从零训练新模型
- **原文**: "先overfit 再trade off，首先保证你的model capacity能够过拟合，再尝试减小模型，各种正则化方法" ——爱睡觉的KKY
- **要点**: 先在小数据上过拟合（甚至100~32条样本），这是验证代码正确性和模型容量的最高效手段。不能过拟合 → 检查代码bug、学习率、标签质量。
- **参见**: [[overfitting-underfitting]], [[../optimizer-lr/learning-rate-scheduling]], [[../data-eval/data-eval]]

### Dropout
- **来源**: 爱睡觉的KKY / 匿名 / DOTA / BugBuster喵 / 罗浩 / 随机漫步的傻瓜 / 四点半的理一
- **适用场景**: 全连接层、CNN、Transformer微调，预训练模型内部
- **原文**: "Dropout, Dropout, Dropout(不仅仅可以防止过拟合, 其实这相当于做人力成本最低的Ensemble)" ——匿名
- **要点**: 默认比例从0.2~0.5，若不知设多少直接选0.5。测试时必须关闭。预训练模型中Dropout ratio不一定要大，0到0.1常见，有时reset到0有奇效。一过拟合不要猛加Dropout，先看数据量、标签质量、weight decay、早停。
- **参见**: [[dropout]], [[../data-eval/eval-metrics]]

### Weight Decay（L2 正则化）
- **来源**: 阿里云云栖号 / 爱睡觉的KKY / 四点半的理一 / BugBuster喵 / Sam聊算法 / 匿名 / 苏辄
- **适用场景**: 所有模型，Transformer与ResNet对weight decay敏感度差异大
- **原文**: "可以先从0.01开始，如果模型不稳定，往0.001甚至0.0001调" ——四点半的理一
- **要点**: 常用范围1e-4~0.1。ResNet类相对不敏感，Transformer类大batch时weight decay设大会导致不收敛。小数据集和浅层结构需更大weight decay（1e-3~1e-2），大数据和深层结构设小（1e-4~1e-6）。AdamW保留weight decay效果，提升泛化。建议先0.01试，不稳定则调小。
- **参见**: [[overfitting-underfitting]], [[../optimizer-lr/adam]], [[../optimizer-lr/learning-rate-scheduling]]

### L1 正则化
- **来源**: 苏辄 / 匿名
- **适用场景**: 需要稀疏权重的场景（特征选择、模型压缩）
- **原文**: "L1 正则化：对稀疏性敏感，权重趋于零" ——苏辄
- **要点**: 适用于需要稀疏性的场景，通过惩罚权重的绝对值使权重趋于零，起到特征选择作用。L2则平滑权重变化，惩罚较大权重值。
- **参见**: [[weight-decay]]

### Early Stopping（早停）
- **来源**: 爱睡觉的KKY / 炫光 / 匿名 / 苏辄 / BugBuster喵
- **适用场景**: 任何训练流程，防止验证集loss回升后的过拟合
- **原文**: "不要过早的early stopping，有时候收敛平台在后段，你会错过，参考1. ，先过拟合train set" ——爱睡觉的KKY
- **要点**: 监控验证集loss，在连续若干epoch不下降时停止。但不要过早早停——有可能收敛发生在训练后段。ASHA/Hyperband这类方法让跑得差的组合尽早停。早停短训适合筛掉明显垃圾组合，不能完全替代完整训练。随机种子策略：跑一批随机种子后早停，挑loss下降快的种子深度训练。
- **参见**: [[early-stopping]], [[../optimizer-lr/learning-rate-scheduling]]

### 数据增强（Data Augmentation）作为正则化
- **来源**: 爱睡觉的KKY / 匿名 / 忍猫 / 四点半的理一 / BugBuster喵
- **适用场景**: CV（裁剪/翻转/旋转/平移/色彩变换）、NLP（同义词替换/随机丢弃/回译）、推荐（采样策略）
- **原文**: "数据越多越好！总是使用数据增强，如水平翻转，旋转，缩放裁剪等。这可以帮助大幅度提高精确度。" ——匿名（亚马逊工程师）
- **要点**: 数据增强是最有效的正则化手段，比任何参数调整都重要。Imagenet级别的大数据集没有不过拟合的，必须上各种数据增强硬调。小数据微调时不要过度增强破坏信号。Mixup/CutMix是更高级的增强形式。
- **参见**: [[data-augmentation]]

### 标签平滑（Label Smoothing）
- **来源**: DOTA
- **适用场景**: 分类任务，防止过拟合于hard label
- **原文**: "Label smoothing将hard label转变成soft label，使网络优化更加平滑。它通常用于减少训练DNN的过拟合问题并进一步提高分类性能。"
- **要点**: 将hard target与均匀分布加权平均：`targets = (1 - label_smooth) * targets + label_smooth / num_classes`。使模型对预测不那么"自信"，提升泛化能力。
- **参见**: [[label-smoothing]], [[../data-eval/eval-metrics]]

### SpatialDropout
- **来源**: 匿名（亚马逊工程师经验）
- **适用场景**: 多路卷积特征融合后的正则化，底层特征图
- **原文**: "如果你有两个或更多的卷积层对相同的输入进行操作，那么在特征连接后使用SpatialDropout。由于这些卷积层是在相同的输入上操作的，输出特征很可能是相关的。SpatialDropout删除了那些相关的特征，并防止网络中的过拟合。"
- **要点**: 主要用于较低的卷积层，整通道丢弃而非单神经元丢弃，打破特征图之间的相关性。当多路卷积共享输入时效果最好。
- **参见**: [[dropout]]

### Max-Norm 约束（权重约束）
- **来源**: 匿名（亚马逊工程师经验）
- **适用场景**: 防止权重过大、梯度爆炸，作为L2的补充
- **原文**: "与L2正则化相反，在你的损失函数中惩罚高权重，这个约束直接正则化你的权重。"
- **要点**: 在Keras中通过 `kernel_constraint=max_norm(2.)` 设置。限制网络权值的上界，直接约束而非惩罚。有助于防止梯度爆炸。权重始终有界。
- **参见**: [[weight-decay]], [[gradient-clipping]]

### Flooding（训练loss阈值）
- **来源**: DOTA
- **适用场景**: 训练loss降到很低后继续提升泛化
- **原文**: "当training loss大于一个阈值时，进行正常的梯度下降；当training loss低于阈值时，会反过来进行梯度上升"
- **要点**: 在loss函数上加flooding操作：`flood = (loss - b).abs() + b`，使训练loss保持在一个阈值附近，让模型做"random walk"，期望进入平坦的损失区域，实现test loss的double decent。
- **参见**: [[../data-eval/eval-metrics]], [[../batch-size/batch-size-generalization]]

### Mixup 与 CutMix
- **来源**: BugBuster喵
- **适用场景**: 图像分类、表示学习，数据增强的升级形式
- **原文**: "数据调参包括数据清洗、采样比例、增强策略、类别平衡、hard negative mining、标签平滑、mixup、cutmix"
- **要点**: Mixup将两张图按比例混合，标签也按比例混合；CutMix裁剪一张图的一部分贴到另一张图上。二者都强制模型学习更平滑的决策边界，是强大的正则化手段。属于数据调参的高级技巧。
- **参见**: [[data-augmentation]], [[label-smoothing]]

### Ensemble 集成方法
- **来源**: BBuf / 匿名 / DOTA / 清明
- **适用场景**: 比赛刷分、追求泛化性能上限
- **原文**: "可以使用不同的初始化方式训练出模型，然后做ensemble。可以使用不同超参数(如学习率，batch_size，优化器)训练出不同模型，然后做ensemble。" ——BBuf
- **要点**: 常用方式：(1) 不同初始化训练多个模型做voting/soft-voting；(2) 不同超参数（lr/batch_size/优化器）训练不同模型；(3) 不同网络架构（VGG/ResNet/Xception）提取特征加权组合；(4) Cyclic LR自动收敛到多个局部最小值得到多个模型做集成。Dropout本身是低成本的隐式Ensemble。Inception/Shortcut/Dense Connection也相当于集成。
- **参见**: [[dropout]], [[dropout]]

### BN 的正则化效果与 Dropout 的交互
- **来源**: BBuf / 匿名 / 罗浩 / 苏辄 / DOTA / 四点半的理一
- **适用场景**: 构建CNN/深度网络，BN与Dropout的取舍
- **原文**: "如果有BN了全连接层就没必要加Dropout了。" ——BBuf
- **要点**: BN兼具加速训练和轻微正则化效果。过拟合时优先加Dropout再加BN。过拟合越严重，Dropout+BN加的位置就越多，有奇效（甚至对Embedding层加）。一般建议：CV任务尽量加BN，RNN/序列任务用LN。小batch时考虑GN替代BN。
- **参见**: [[../normalization/normalization]]

### 泛化诊断框架（过拟合 vs 欠拟合）
- **来源**: 爱睡觉的KKY / 四点半的理一 / BugBuster喵 / 匿名 / 清明
- **适用场景**: 任何训练实验的第一步——诊断
- **原文**: "训练集loss低但验证集loss高，那是典型的过拟合，优先加数据，其次加正则化，最后才考虑降低模型容量。" ——四点半的理一
- **要点**: 用训练/验证loss差值为诊断依据：差值大=过拟合→加数据/正则化；两边都高=欠拟合→增大模型/学习率。不要同时调超过两个超参数。每次实验前先写假设，跑完对比。
- **参见**: [[overfitting-underfitting]], [[../data-eval/data-eval]], [[../debugging/small-data-overfit-test]]

### 梯度裁剪（Gradient Clipping）与泛化
- **来源**: 匿名 / BugBuster喵 / Sam聊算法
- **适用场景**: RNN、Transformer、大模型微调，梯度爆炸
- **原文**: "梯度裁剪建议默认加，尤其是RNN、Transformer、大模型微调，clip norm 1.0是常见起点。" ——BugBuster喵
- **要点**: 梯度裁剪可以防止梯度爆炸导致的NaN，间接帮助训练稳定从而改善泛化。但剪裁过大可能损伤性能——正常收敛的模型加clip反而可能下降。DeepSeek实验建议clip norm阈值取1.0。
- **参见**: [[gradient-clipping]]

### SGD 比 Adam 泛化更好的取舍
- **来源**: 四点半的理一 / 匿名 / BugBuster喵
- **适用场景**: 训练后期追求泛化性能上限
- **原文**: "SGD+momentum可以实现找到全局最小值，但它依赖于鲁棒初始化。我建议你使用SGD+动量，因为它能达到更好的最佳效果。" ——匿名（亚马逊工程师）
- **要点**: SGD+Momentum泛化能力通常比Adam强1~2个点。大厂标准做法：前几个epoch用Adam快速冷启动，之后切换到SGD进行微调（两阶段训练）。AdamW在Adam基础上保留weight decay效果，提升泛化。
- **参见**: [[../optimizer-lr/adam]]

---

## 常见正则化超参数速查

| 方法 | 推荐值范围 | 备注 |
|------|-----------|------|
| Dropout | 0.2~0.5（通用），0~0.1（大模型微调） | 过高可能欠拟合 |
| Weight Decay | 1e-6~0.1 | 小数据设大，大数据设小 |
| L2 正则 | 首选1.0，超过10很少见 | 经验值 |
| Label Smoothing | 0.1 | 常用默认值 |
| Max-Norm | 2.0~4.0 | Keras kernel_constraint |
| Flooding 阈值 b | 取决于任务 | 从训练loss水平设定 |
| Gradient Clip Norm | 1.0 | DeepSeek推荐值 |
| LoRA Dropout | 0~0.1 | 小数据可设0 |
| LoRA Rank | 8~32 | 数据少时大rank更易过拟合 |

## 过拟合反直觉要点

- 测试集loss降低不意味着没有过拟合——如果负样本占比大，可能负样本学得爆好而正样本凉了（查看F1/AUC）
- 验证集loss降不代表泛化好——要检查验证集切分是否泄漏了未来信息或相似模板
- 在推荐系统中，一个epoch可能就过拟合了
- 特征要一个一个加——某些特征可能导致模型过拟合
- 最后两层之间加微步幅有时也能作为正则手段

## 参见

- [[overfitting-underfitting]] — 过拟合与欠拟合的诊断框架与应对策略
- [[early-stopping]] — 早停策略：耐心值选择、ASHA/Hyperband
- [[../normalization/normalization]] — BN/LN/GN 的正则化效果与选择
- [[../data-eval/data-eval]] — 如何诊断过拟合与欠拟合
- [[../optimizer-lr/learning-rate-scheduling]] — 学习率与正则化的配合
- [[../optimizer-lr/adam]] — SGD vs Adam vs AdamW 的泛化差异
- [[data-augmentation]] — 数据增强详细策略
- [[dropout]] — Dropout 变体与实现
- [[label-smoothing]] — 标签平滑实现细节
- [[dropout]] — 集成方法进阶
- [[../data-eval/eval-metrics]] — 损失函数选择与正则
- [[gradient-clipping]] — 梯度裁剪的配置
