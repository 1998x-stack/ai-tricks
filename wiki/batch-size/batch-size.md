# 批大小与梯度累积

## 概述

批大小（batch size）是每次参数更新时使用的样本数量，它直接影响训练速度、梯度稳定性与模型泛化能力。梯度累积（gradient accumulation）则是在显存受限时等效扩大批大小的关键技术，通过多个小批次累积梯度后再统一更新参数。

## 详细知识

- [[gradient-accumulation]] — 梯度累积原理、实现与等效批次计算
- [[batch-size-generalization]] — 批大小如何影响泛化：平坦vs尖锐极小值
- [[batch-size-lr-scaling]] — 批大小与学习率的缩放法则（线性缩放 + DeepSeek Scaling Law）

## 核心技巧

### batch size 在表示学习/对比学习领域越大越好
- **来源**: 爱睡觉的KKY
- **适用场景**: 对比学习、表示学习（如 SimCLR、MoCo、CLIP）
- **原文**: > batch size 在表示学习，对比学习领域一般越大越好，显存不够上累计梯度，否则模型可能不收敛… 其他领域看情况；
- **要点**: 对比学习依赖大量负样本区分正负对，小批次容易导致模型坍塌，必须用大批次或梯度累积等效扩大批次。
- **参见**: [[batch-size-generalization]], [[gradient-accumulation]], [[../optimizer-lr/optimizer-lr]]

### batch size 并非越大越好——小批次有时有奇效
- **来源**: 炫光
- **适用场景**: 通用深度学习，特别是性能意外下降时
- **原文**: > batch有的时候并非越大越好，虽然batch越大训练越快，但有的时候batch太大网络性能会降低，有的时候减小batch有奇效
- **要点**: 大批次使梯度估计更稳定但可能导致泛化下降，必要时减小 batch size 作为调试手段。

### PPO / RLHF 阶段 batch size 宁大勿小
- **来源**: Tang AI
- **适用场景**: RLHF、PPO 强化学习训练
- **原文**: > RLHF阶段的batch_size宁大勿小。更大的批次可以提供更稳定的梯度估计，尤其对于PPO。如果显存不足，应优先使用Gradient Accumulation来等效扩大批次大小。
- **要点**: PPO 的策略更新对梯度噪声敏感，大批次提供稳定梯度估计，显存不足时用梯度累积替代减小批次。
- **参见**: [[gradient-accumulation]], [[../optimizer-lr/optimizer-lr]]

### GRPO 训练 batch size 越大效果越稳定
- **来源**: Tang AI
- **适用场景**: GRPO 偏好优化训练
- **原文**: > GRPO训练，batch size越大，其实效果越稳定，尤其是模型能力没有那么好的情况下。
- **要点**: 基座模型能力较弱时，大批次带来的稳定梯度对 GRPO 训练尤为关键。

### 推荐系统 batch size 最佳范围 32-512
- **来源**: Tang AI
- **适用场景**: 推荐系统 CTR 预估、召回排序模型
- **原文**: > batch size对于推荐来说32-64-128-512测试效果再高一般也不会正向了，再低训练太慢了。
- **要点**: 推荐系统 batch size 存在有效上限（约 512），超出后收益递减；低于 32 则训练过慢。
- **参见**: [[batch-size-generalization]], [[../optimizer-lr/optimizer-lr]]

### batch size 与学习率的线性缩放关系
- **来源**: 四点半的理一
- **适用场景**: CNN、ResNet 类模型调参
- **原文**: > batch size和学习率存在线性缩放关系——batch size翻倍，学习率通常也可以适当放大。具体倍数关系业界没有定论，但有一个被广泛验证的经验值：batch size从64扩到256时，学习率可以从0.1对应扩到0.4左右。这个比例不是死的，但给你一个锚点。
- **要点**: batch size 翻倍时学习率可按比例放大，64→256 对应 lr 0.1→0.4 是常用锚点，但非固定公式。
- **参见**: [[batch-size-lr-scaling]], [[../optimizer-lr/optimizer-lr]]

### 大 batch 训练快但泛化不一定更好，需记录 effective batch size
- **来源**: BugBuster喵
- **适用场景**: A100/H100 多卡训练、梯度累积场景
- **原文**: > batch size 也重要，但它不是越大越好。大 batch 训练快，梯度稳定，但泛化不一定更好。今天大家用 A100、H100、L40S、国产卡集群，gradient accumulation 很常见，看到的 batch size 往往是 effective batch size。调参时要记录清楚：单卡 batch、卡数、累积步数、总 batch 到底是多少。很多复现实验复现不出来，就是这里没说清。
- **要点**: 使用多卡和梯度累积时必须记录「单卡 batch × 卡数 × 累积步数 = effective batch size」的完整公式，否则实验无法复现。
- **参见**: [[gradient-accumulation]], [[batch-size-generalization]], [[multi-gpu]]

### 批量大小受硬件内存限制，多 GPU 总批量为单卡批量乘以 GPU 数
- **来源**: 阿里云云栖号
- **适用场景**: 多 GPU 分布式训练
- **原文**: > 与学习率不同，其值不影响计算训练时间。批量大小受硬件内存的限制，而学习率则不然。建议使用适合硬件内存的较大批量大小，并使用更大的学习速率。如果服务器有多个GPU，则总批量大小是单个GPU上的批量大小乘以GPU的数量。
- **要点**: 多卡训练时总 batch = per_device_batch × num_gpus，应优先用满硬件内存的较大单卡 batch。

### 小批量：梯度噪声正则化 vs 大批量：训练加速
- **来源**: 白小鱼
- **适用场景**: 通用深度学习，复杂度与泛化权衡
- **原文**: > 选取合适的batch值也很重要，小batch有助于通过噪声注入进行规范化，但如果学习任务设置的很复杂，这可能是有害的。此外，运行许多小batch需要更多的时间。相反，大batch可以加快训练速度，甚至有更好的泛化性能。
- **要点**: 小批次注入梯度噪声起到正则化作用（但复杂任务中有害），大批次加速训练且有时泛化更好，需任务具体判断。

### 小批次显存小但梯度不稳定，大批次显存大但易陷入局部最优
- **来源**: 苏辄
- **适用场景**: 初学者理解 batch size 基本原理
- **原文**: > 每次输入神经网络的样本数量。小批次减少显存用量、但是权重更新频繁，梯度不稳定。大批次显存用量大，训练稳定、但是容易陷入局部最优。
- **要点**: batch size 的本质是显存-稳定性-泛化的三向权衡：小→更新频繁噪声大，大→稳定但可能收敛到尖锐极小值。

### DeepSeek LLM 批大小与学习率的 Scaling Law
- **来源**: Sam聊算法
- **适用场景**: 大语言模型预训练超参选择
- **原文**: > 给定计算成本 C，大多数超参的最优值是基本固定的，只需调整学习率和 batch size，而学习率的最优值随 C 增加而下降，batch size 的最优值随 C 增加而上升，满足幂律：η_opt = 0.3118·C^(-0.1250)，B_opt = 0.2920·C^(0.3271)
- **要点**: DeepSeek 实验表明，预训练中仅 batch size 和学习率需随计算量调整，其他超参固定；计算量增大时应开大 batch size、调小学习率。
- **参见**: [[batch-size-lr-scaling]], [[../optimizer-lr/optimizer-lr]], [[learning-rate-scheduler]]

### 梯度累积代码实现
- **来源**: （代码段）
- **适用场景**: PyTorch 下显存不足时等效扩大 batch size
- **原文**: > 梯度累积可以在不增加内存消耗的情况下模拟大批次训练。通过在多个小批次上累积梯度，然后在达到指定累积次数后进行一次参数更新，可以有效提高训练效率。
- **要点**: 核心实现为 loss = loss / accumulation_steps，每积累 accumulation_steps 次才执行 optimizer.step()，注意 loss 需除以累积步数以保持梯度量级一致。
- **参见**: [[gradient-accumulation]], [[multi-gpu]]

### mini batch size 数十到数百皆可，SGD 默认设定
- **来源**: 匿名用户
- **适用场景**: 经典 CNN / 全连接网络训练
- **原文**: > sgd，mini batch size从几十到几百皆可。步长0.1，可手动收缩，weight decay取0.005，momentum取0.9。
- **要点**: 早期深度学习实践中，mini batch size 在数十到数百范围内均可工作，配合 SGD + momentum 是可靠基线。

### 开源默认 batch 不一定适合你的数据
- **来源**: BugBuster喵
- **适用场景**: 基于开源项目微调或二次开发
- **原文**: > 开源 repo 的默认配置往往是为了跑通 demo，不一定适合你的数据。尤其是 batch、lr、max length、eval step、save step、模板、数据过滤。复制配置可以开始，但不能停止思考。
- **要点**: 开源项目的默认 batch size 以跑通 demo 为目标，迁移到自己的数据时务必重新调整，不宜盲从。

### 调参应先锁定学习率，再调 batch size
- **来源**: 四点半的理一
- **适用场景**: 超参调试顺序
- **原文**: > 新手最容易犯的错，是同时调学习率、batch size、正则化系数，一口气改五六个参数……锁定学习率之后，你再调 batch size。
- **要点**: 不要同时调超过两个超参数。先确定学习率再调 batch size 的串行策略效率远高于并行乱调。
- **参见**: [[batch-size-generalization]], [[../optimizer-lr/optimizer-lr]]

### batch size 在不同任务中影响不同
- **来源**: BBuf
- **适用场景**: 跨任务类型（CV / NLP）的调参认知
- **原文**: > batch_size：在不同类型的任务中，batch_size的影响也不同，大家可以看看这篇batch_size对模型性能影响的文章
- **要点**: 同一 batch size 在不同任务（分类、检测、分割、序列标注）中的最优值差异很大，应参考对应任务的经验报告。

### 分桶策略（Bucketing）减少 padding 提高训练效率
- **来源**: （代码段）
- **适用场景**: 序列模型训练，文本/语音等变长输入
- **原文**: > 分桶策略可以根据输入序列的长度将数据分成不同的桶，从而在训练过程中更高效地利用计算资源。这种方法可以减少 padding 的数量，提高训练效率。
- **要点**: 对变长序列按长度分桶，同桶内使用相同 batch size，减少无效 padding 计算，间接提升有效 batch 的信息密度。

### 推荐系统热启动时 batch size 需注意
- **来源**: Tang AI
- **适用场景**: 推荐模型增量训练 / 热启动
- **原文**: > 对于推荐系统的老汤模型，建议热启动……必须 embedding+DNN 全部热启动，然后仅对倒数第一层作扰动
- **要点**: 热启动场景中 batch size 建议与原训练一致，变更 batch size 会改变 BN 统计量和梯度分布，导致热启动失效。

### PPO 学习率通常比 SFT 小一个数量级
- **来源**: Tang AI
- **适用场景**: RLHF / PPO 训练
- **原文**: > PPO的学习率通常需要比SFT小一个数量级。例如，如果SFT阶段的学习率是2e-5，PPO阶段的初始学习率建议设置为1e-6到3e-6之间
- **要点**: PPO 使用小学习率+大批次的组合，学习率降低一个数量级的同时 batch size 放大，两者配合才能稳定训练。
- **参见**: [[batch-size-lr-scaling]], [[../optimizer-lr/optimizer-lr]]
