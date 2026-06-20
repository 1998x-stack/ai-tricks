# 优化器与学习率

## 概述

优化器与学习率是深度学习调参中最重要的超参数组合，直接影响模型是否能够收敛以及最终泛化性能。本章汇总了来自知乎"深度学习调参有哪些技巧？"问题下多名答主关于优化器选择、学习率设定及两者配合的实战经验。

## 详细知识

- [[adam]] — Adam优化器原理、超参数与使用场景
- [[sgd-momentum]] — SGD+Momentum深入解析，泛化性优势
- [[learning-rate-scheduling]] — 六种学习率调度策略完整指南
- [[weight-decay]] — 权重衰减机制、与AdamW的关系、按模型类型调参
- [[lr-range-test]] — 学习率范围测试：如何科学找到最佳LR

## 核心技巧

### 优化器优先选Adam
- **来源**: 炫光
- **适用场景**: 通用深度学习任务，尤其是起步阶段
- **原文**: 不知道用啥优化器，无脑Adam，对绝大多数问题都有不错的效果
- **要点**: Adam是泛用性最强的默认优化器，适合绝大多数问题
- **参见**: [[adam]], [[adam]]

### 选Adam还是SGD看任务性质
- **来源**: 爱睡觉的KKY
- **适用场景**: NLP vs CV任务选择优化器
- **原文**: 优化器，nlp，抽象层次较高或目标函数非常不平滑的问题adam优先，其他可以尝试下sgd（一般需要的迭代次数高于sgd）
- **要点**: NLP/抽象层次高/目标函数不平滑用Adam，CV等任务可尝试SGD
- **参见**: [[adam]], [[sgd-momentum]], [[adam]]

### SGD+Momentum最终效果常优于Adam
- **来源**: BBuf、四点半的理一、陈城南、摘星狐狸、黑马程序员Python
- **适用场景**: 追求极致泛化性能、训练后期精调
- **原文**: 我基本都是带动量的SGD。如果优化不动可以试试Adam。
- **要点**: SGD+Momentum虽然训练更慢更难，但泛化能力通常比Adam高一两个点
- **参见**: [[sgd-momentum]], [[adam]]

### 两阶段训练法：Adam过渡到SGD
- **来源**: 四点半的理一
- **适用场景**: 大模型/大数据量/长训练周期，追求最优效果
- **原文**: 很多大厂的标准做法是：前几个epoch用Adam，后面的epoch切换到SGD。这叫"两阶段训练"，用好了效果提升非常明显。
- **要点**: 早期用Adam快速收敛，后期切SGD+Momentum提升泛化
- **参见**: [[adam]], [[learning-rate-scheduling]], [[adam]], [[learning-rate-scheduling]]

### AdamW替代Adam，权重衰减更合理
- **来源**: 陈城南、hyshhh、苏辄、四点半的理一
- **适用场景**: Transformer/LoRA微调，大规模训练
- **原文**: 初始构建时选 Adam 优化器（也可以AdamW）
- **要点**: AdamW修正了Adam的权重衰减实现，尤其适合Transformer类模型和LoRA微调
- **参见**: [[adam]], [[weight-decay]], [[../llm-sft/lora]]

### 数据稀疏用Adagrad，稠密用Adam
- **来源**: Tang AI
- **适用场景**: 推荐系统，稀疏/稠密特征分布差异大的场景
- **原文**: 对于稀疏特征多的模型采用adagrad，稠密特征多的模型用adam
- **要点**: 稀疏特征多 -> Adagrad，稠密特征多 -> Adam
- **参见**: [[adam]], [[adam]], [[../recsys/recsys]]

### NLP通用LR范围
- **来源**: 爱睡觉的KKY
- **适用场景**: NLP Bert类模型
- **原文**: lr，最重要的参数，一般nlp bert类模型在1e-5级别附近，warmup，衰减
- **要点**: NLP Bert类模型初始LR在1e-5附近，配合warmup和衰减
- **参见**: [[learning-rate-scheduling]], [[lr-range-test]], [[learning-rate-scheduling]]

### CV通用LR范围
- **来源**: 爱睡觉的KKY
- **适用场景**: CV分类/检测模型
- **原文**: cv类模型在1e-3级别附近，衰减
- **要点**: CV类模型初始LR在1e-3附近，后续衰减
- **参见**: [[learning-rate-scheduling]], [[lr-range-test]], [[../cnn-cv/cnn-cv]]

### 初始LR二分搜索法
- **来源**: 炫光
- **适用场景**: 任何新任务的初始LR选择
- **原文**: 学习率的设置，一般可以先从1e-3、3e-4、1e-4开始，同时跑三炉丹，然后看效果，用类似二分的方法迭代搜索
- **要点**: 同时跑三个数量级附近的LR，迭代缩小范围
- **参见**: [[lr-range-test]], [[lr-range-test]]

### LR范围的土办法：预热扫描
- **来源**: 四点半的理一
- **适用场景**: 快速确定合理LR范围
- **原文**: 从1e-5开始，以10倍为单位往上跳，比如1e-4、1e-3、1e-2，观察loss曲线。正常情况下，loss应该平稳下降；如果从某个点开始loss开始发散甚至爆炸，那个点之前的第一档就是你的候选学习率。
- **要点**: 十倍递增扫描LR，选发散前一档
- **参见**: [[lr-range-test]], [[lr-range-test]]

### LR Range Test
- **来源**: hyshhh
- **适用场景**: 精确查找最佳初始学习率
- **原文**: 先用 学习率搜索 (LR Range Test) 找到最佳初始学习率
- **要点**: 使用ExponentialLR从极小LR指数增长，观察loss曲线找到最佳LR
- **参见**: [[lr-range-test]], [[lr-range-test]]

### Cyclical LR的范围测试法
- **来源**: 阿里云云栖号
- **适用场景**: Cyclical LR的上下界确定
- **原文**: LR范围测试：运行模型几个epoch，同时让学习率在高低学习率值之间线性增加。对于浅层的3层架构，最大设置为0.01，而对于resnet这样的网络，学习率最大可以设置为3.0。
- **要点**: 线性增加LR跑几个epoch确定上下界，最大LR的十分之一设为最小LR
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]]

### Cosine退火+warmup万能模板
- **来源**: 陈城南、摘星狐狸、BugBuster喵
- **适用场景**: 绝大多数深度学习的默认LR调度方案
- **原文**: 万能模板 Adam + Cos退火 构建基础：初始构建时选 Adam 优化器，学习率初始时可以设置3.5e-4。学习率变化选择 warm-up + cos退火。
- **要点**: Adam + Warmup + Cosine Decay是最通用的训练配置
- **参见**: [[adam]], [[learning-rate-scheduling]], [[learning-rate-scheduling]]

### Warmup步数设置原则
- **来源**: LindgeWAI、BugBuster喵
- **适用场景**: Transformer/LoRA/大模型训练
- **原文**: lr_scheduler的warmup步数设置：一般遵循模型复杂，步数多点；数据集大，步数少点（参考BERT，默认设成总训练轮次的1/10，也可1/20）
- **要点**: 模型越复杂warmup越长，数据越大warmup越短，BERT默认用总步数1/10
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]]

### 阶梯下降（Step Decay）精调法
- **来源**: 陈城南
- **适用场景**: 训练后期进一步压低loss
- **原文**: 固定lr值，然后训，发现损失不咋下降了，设置一个在这个时间点下降0.1倍（一般都是10倍下降，经验值）
- **要点**: loss平台期将LR x0.1，继续下降到下一个平台期再降
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]]

### Multistep LR衰减 + 余弦
- **来源**: BBuf
- **适用场景**: CV任务，max_iter确定的场景
- **原文**: 学习率衰减策略我一般使用multistep方式，step_size的设置要看视你的的max_iter而定
- **要点**: 经验初始LR为0.01，使用multistep方式按步数衰减
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]]

### Cyclical LR周期性重启
- **来源**: DOTA
- **适用场景**: Ensemble集成、跳出局部最优
- **原文**: 每隔一段时间重启学习率，这样在单位时间内能收敛到多个局部最小值，可以得到很多个模型做集成。
- **要点**: 周期性LR重启可收集多个局部最优解用于模型集成
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]], [[../regularization/dropout]]

### LR按验证集瓶颈手动折半
- **来源**: 摘星狐狸
- **适用场景**: 无调度器的手动训练、验证集遇到瓶颈
- **原文**: 到后期验证集遇到瓶颈的时候就除以2（或5）再继续，直到LR变得非常非常小，训练也停止了。
- **要点**: 验证集不提升时LR减半或减五分之一继续训练
- **参见**: [[lr-range-test]]

### 指数衰减LR
- **来源**: 炫光
- **适用场景**: 长期训练，稳定的LR衰减策略
- **原文**: 学习率衰减很有用，设置一个指数衰减的scheduler，衰减率0.99，是一个不错的选择
- **要点**: 指数衰减率0.99使LR缓慢下降，训练集大时可增大衰减率
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]]

### 衰减系数参考值
- **来源**: 京东白条-王惠东
- **适用场景**: 通用LR衰减系数选择
- **原文**: 学习率一般要随着训练进行衰减。衰减系数设0.1，0.3，0.5均可，衰减时机，可以是验证集准确率不再上升时
- **要点**: 衰减系数可选0.1/0.3/0.5，在验证集平台期触发
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]]

### 初始LR的阶梯搜索法
- **来源**: 黑马程序员Python
- **适用场景**: 通用LR初值选择
- **原文**: 有三个学习率起点（即1e-1，1e-3和1e-6）。如果对预训练模型微调，考虑小于1e-3（比如1e-4）的低学习率。从头训练则考虑大于等于1e-3。
- **要点**: 三个LR起点：0.1（从头训）、0.001（通用）、1e-6（微调预训练模型用1e-4）
- **参见**: [[lr-range-test]], [[lr-range-test]]

### ReduceLROnPlateau + EarlyStopping
- **来源**: 黑马程序员Python
- **适用场景**: 验证集驱动自适应LR衰减
- **原文**: 如果验证损失在某些epoch停止改善，减小学习率；如果验证损失停止改善在某些epoch，停止训练过程。
- **要点**: 验证集不下降时自动降低LR，再不下降则早停
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]], [[early-stopping]]

### LR过高的症状与对策
- **来源**: Captain Jack、唐AI、京东白条-王惠东、TNotes
- **适用场景**: loss爆炸、NaN、训练不稳定
- **原文**: 太大：loss爆炸或者nan；太小：半天loss没反映
- **要点**: LR过高的典型表现：loss震荡/发散/NaN/输出重复无意义内容
- **参见**: [[lr-range-test]]

### LR过低的症状
- **来源**: Captain Jack
- **适用场景**: 训练缓慢但loss不降
- **原文**: 太小：半天loss没反映（但是，LR需要降低的情况也是这样，这里可视化网络中间结果，不是weights，有效果，两者可视化结果是不一样的，太小的话中间结果有点水波纹或者噪点的样子）
- **要点**: LR太小时loss几乎不动，中间层特征有水波纹状噪点
- **参见**: [[lr-range-test]]

### 需要降低LR的标志
- **来源**: Captain Jack
- **适用场景**: 判断LR是否需要衰减
- **原文**: 需要进一步降低了：loss在当前LR下一路降了下来，但是半天不再降了。
- **要点**: loss在平台期不再下降时需降低LR
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]], [[lr-range-test]]

### LR先小后大再衰减策略
- **来源**: Captain Jack
- **适用场景**: loss设计不合理时，初始容易爆的场景
- **原文**: 先上一个小LR保证不爆，等loss降下来了，再慢慢升LR，之后当然还会慢慢再降LR
- **要点**: 先小LR保证稳定 -> 再升LR加速 -> 最后再降LR收敛
- **参见**: [[learning-rate-scheduling]], [[lr-range-test]], [[learning-rate-scheduling]]

### Batch Size与LR的线性缩放关系
- **来源**: 四点半的理一
- **适用场景**: 调整batch size时同步调整LR
- **原文**: batch size翻倍，学习率通常也可以适当放大。batch size从64扩到256时，学习率可以从0.1对应扩到0.4左右。
- **要点**: batch size放大n倍，LR也可放大n倍左右
- **参见**: [[../batch-size/batch-size]] | [[lr-range-test]]

### 显存不足时梯度累积等效增大batch
- **来源**: LindgeWAI、爱睡觉的KKY
- **适用场景**: 表示学习/对比学习/大batch受限
- **原文**: batch size 在表示学习，对比学习领域一般越大越好，显存不够上累计梯度
- **要点**: 梯度累积等效扩大batch size，注意同步缩放LR
- **参见**: [[gradient-accumulation]]

### Momentum通用设置范围
- **来源**: 匿名用户、随机漫步的傻瓜、阿里云云栖号、hyshhh
- **适用场景**: SGD+Momentum的默认动量值
- **原文**: momentum取0.9
- **要点**: SGD+Momentum的动量值通常取0.9，可尝试0.8-0.99范围
- **参见**: [[sgd-momentum]], [[sgd-momentum]]

### 循环动量（Cyclical Momentum）
- **来源**: 阿里云云栖号
- **适用场景**: Cyclical LR配合动量联动
- **原文**: 使用循环学习率时，初始化一个小的学习率，使其开始收敛，并减少周期动量。当学习率增加时，使用递减的循环动量加快收敛。
- **要点**: LR增加时动量减小，LR减小时动量增加，两者联动效果更好
- **参见**: [[sgd-momentum]], [[learning-rate-scheduling]], [[sgd-momentum]]

### Weight Decay默认值及调参
- **来源**: 四点半的理一、BugBuster喵、阿里云云栖号
- **适用场景**: 正则化防过拟合，尤其Transformer
- **原文**: ResNet类模型对weight decay相对不敏感，但Transformer类模型，尤其是大batch训练时，weight decay设大了会让模型完全无法收敛。我的经验是先从0.01开始，如果不稳定往0.001甚至0.0001调。
- **要点**: ResNet对weight decay不敏感；Transformer很敏感，从0.01开始试
- **参见**: [[weight-decay]]

### 梯度裁剪防止NaN
- **来源**: Tang AI、BugBuster喵、陈城南
- **适用场景**: RNN/Transformer/大模型微调，防止梯度爆炸
- **原文**: 当出现训练不稳定（如损失值突然飙升），可启用梯度裁剪，裁剪值一般为0.2。
- **要点**: 梯度裁剪默认建议加上，clip norm 1.0为常见起点，但裁剪可能影响性能
- **参见**: [[gradient-clipping]]

### PPO训练LR比SFT小一个数量级
- **来源**: Tang AI
- **适用场景**: RLHF/PPO强化学习阶段
- **原文**: PPO的学习率通常需要比SFT小一个数量级。例如SFT是2e-5，PPO初始建议1e-6到3e-6，过高易导致模式崩溃。
- **要点**: PPO_LR = SFT_LR x 0.1，极端保守从1e-6开始
- **参见**: [[learning-rate-scheduling]], [[../llm-rl/llm-rl]], [[lr-range-test]]

### DPO中LR与Beta的配合
- **来源**: Tang AI
- **适用场景**: DPO偏好优化
- **原文**: DPO训练损失下降很快但生成效果差，可能是因为beta值过低或学习率过高。调高beta增加对SFT模型的约束，降低学习率使用1e-7到5e-6微调。
- **要点**: DPO的LR要极保守（1e-7~5e-6），beta控制模型偏离SFT的程度
- **参见**: [[dpo]] | [[../llm-rl/llm-rl]]

### 大模型SFT初始LR与weight decay
- **来源**: Tang AI
- **适用场景**: 大模型SFT微调
- **原文**: 调参的时候，可以对learning rate，weight decay从大往小调，通常lr从1E-4开始，weight decay从0.25开始。
- **要点**: SFT微调lr从1e-4下探，weight_decay从0.25下探
- **参见**: [[weight-decay]], [[../llm-sft/llm-sft]], [[lr-range-test]]

### 大模型预训练的Scaling Law
- **来源**: Sam聊算法
- **适用场景**: LLM预训练超参随计算量的变化
- **原文**: 给定计算成本C，大多数超参的最优值基本固定，只需调整学习率和batch size。η_opt = 0.3118 * C^(-0.1250), B_opt = 0.2920 * C^(0.3271)
- **要点**: 计算量越大，LR越小（幂律-0.125），batch越大（幂律0.327）
- **参见**: [[../batch-size/batch-size-lr-scaling]] | [[../llm-sft/llm-sft]]

### DeepSeek固定超参最优值
- **来源**: Sam聊算法
- **适用场景**: LLM预训练超参设定
- **原文**: β1=0.9, β2=0.95, weight_decay=0.1；初始化标准差0.06；梯度裁剪阈值1.0；warmup steps 2000；multistep scheduler在80%和90% tokens处各降一次。
- **要点**: 大部分超参固定，warmup 2000步，multistep两阶段衰减
- **参见**: [[weight-decay]], [[../llm-sft/llm-sft]]

### LoRA微调LR范围
- **来源**: BugBuster喵
- **适用场景**: LoRA/QLoRA微调预训练大模型
- **原文**: LoRA调参里最常调的是学习率、rank、alpha、dropout。学习率1e-4到3e-4是常见范围，小数据强基座可更低。rank 8/16/32开始。
- **要点**: LoRA lr在1e-4~3e-4常见，小数据更保守；rank不是越大越好
- **参见**: [[lora]] | [[lr-range-test]]

### 不要同时调超过两个超参数
- **来源**: 四点半的理一
- **适用场景**: 调参方法论
- **原文**: 不要同时调超过两个超参数。这是新手和大神最本质的区别。
- **要点**: 每次只动1-2个参数，否则无法归因效果变化
- **参见**: [[lr-range-test]]

### 先锁定LR再调其他参数
- **来源**: 四点半的理一
- **适用场景**: 调参顺序
- **原文**: 新手最容易犯的错，是同时调学习率、batch size、正则化系数，一口气改五六个参数。我的做法是先把学习率锁定在一个合理范围，再动其他参数。
- **要点**: 调参第一步：锁定LR范围，再动其他参数
- **参见**: [[lr-range-test]]

### 优先调LR再谈其他
- **来源**: 无忧无虑
- **适用场景**: 模型效果不满意时的第一反应
- **原文**: 1、优先调 learning rate！优先调 learning rate！优先调 learning rate！学习速率会很大程度上影响模型的表现。
- **要点**: LR是第一优先级超参数
- **参见**: [[lr-range-test]]

### LR是调参的第一优先级，其次是数据
- **来源**: BugBuster喵
- **适用场景**: 调参优先级排序，大规模业务
- **原文**: 绝大多数任务里，最值得先调的永远是学习率和数据相关参数。学习率不对，其他全白搭。
- **要点**: LR > 数据 > 模型结构 > 其他超参数
- **参见**: [[lr-range-test]]

### 最简可用配置
- **来源**: 四点半的理一
- **适用场景**: 快速获取baseline的默认配置
- **原文**: Learning rate从1e-3开始，用AdamW优化器，batch size 64到256之间，weight decay默认0.01，epoch数不低于30。
- **要点**: LR=1e-3, AdamW, bs=64~256, wd=0.01, epoch>=30
- **参见**: [[lr-range-test]]

### 初始LR：0.1搭配SGD+Momentum
- **来源**: 陈城南、BBuf、匿名用户
- **适用场景**: CV任务，SGD训练
- **原文**: SGD一般初始学习率为0.1（具体还得调），然后选择Cos或者Step下降
- **要点**: SGD初始LR一般为0.1，配合momentum 0.9
- **参见**: [[sgd-momentum]], [[lr-range-test]]

### LR太大连ReLU神经元都能弄死
- **来源**: Captain Jack
- **适用场景**: ReLU神经元"死亡"问题
- **原文**: LR在可以工作的最大值下往小收一收，免得ReLU把神经元弄死了。
- **要点**: LR太大会使ReLU神经元永久死亡（dying ReLU），LR应偏保守
- **参见**: [[lr-range-test]] | [[../activation/activation-selection]]

### LR过小 vs LR进入平台期 的可视化区别
- **来源**: Captain Jack
- **适用场景**: 判断LR问题是太小还是需要进一步衰减
- **原文**: 太小的话中间结果有点水波纹或者噪点的样子，因为filter学习太慢的原因；需要进一步降低的情况也是如此，但可视化中间结果两者不一样。
- **要点**: LR太小时中间特征有水波纹；LR需要降低时loss不再下降但特征正常
- **参见**: [[lr-range-test]] | [[../data-eval/eval-metrics]]

### SFT后loss下降但效果不好的诊断
- **来源**: Tang AI
- **适用场景**: 大模型SFT训练中期评估
- **原文**: 对于大模型训练，到了中后期loss小效果不一定好，甚至会出现loss先下降后上升的情况。正确判断方法是用checkpoint进行accuracy或其他业务指标判断。
- **要点**: SFT中后期不能只看loss，需用checkpoint评估业务指标
- **参见**: [[../llm-sft/llm-sft]] | [[eval-metrics]]

### 训练初期不要加激活函数和BN
- **来源**: Tang AI
- **适用场景**: 新模型起步阶段debug
- **原文**: 做新模型的时候，最开始不要加激活函数，不要加batchnorm，不要加dropout，先就纯模型。然后再一步一步的实验。
- **要点**: 新模型先跑纯结构确认能收敛，再逐步加激活函数/BN/dropout
- **参见**: [[lr-range-test]]

### 确认随机种子固定
- **来源**: 爱睡觉的KKY
- **适用场景**: 对比实验可靠性
- **原文**: 随机数种子设定好，否则很多对比实验结论不一定准确
- **要点**: 固定seed确保实验可重复，否则LR调的结论可能是噪声
- **参见**: [[../data-eval/cross-validation]]

### 不要过早Early Stopping
- **来源**: 爱睡觉的KKY
- **适用场景**: 训练过程监控
- **原文**: 不要过早的early stopping，有时候收敛平台在后段，你会错过
- **要点**: 训练初期平台期不代表结束，LR衰减后可能继续收敛
- **参见**: [[early-stopping]]

### 先过拟合再收敛
- **来源**: 爱睡觉的KKY、清明、BugBuster喵
- **适用场景**: 确认模型容量足够的debug方法
- **原文**: 先overfit再trade off，首先保证你的模型capacity能够过拟合，再尝试减小模型，各种正则化方法
- **要点**: 调参第一步：确保模型能在小数据上过拟合
- **参见**: [[../debugging/small-data-overfit-test]]

### 参数初始化与LR配合
- **来源**: BBuf、Jarvix、Captain Jack、京东白条-王惠东
- **适用场景**: 模型初始化方法选择
- **原文**: 对于weight的初始化一般都是使用xavier初始化
- **要点**: Xavier（tanh/sigmoid）或He初始化（ReLU），好的初始化使LR更有效
- **参见**: [[../initialization/initialization]]

### 对照训练/验证loss曲线调LR
- **来源**: 四点半的理一、BugBuster喵
- **适用场景**: LR问题的系统性诊断
- **原文**: loss降得太慢，学习率可能小；loss震荡或发散，学习率可能大；训练分上去验证不动，可能过拟合或验证集不匹配；训练验证都低，可能模型容量问题。
- **要点**: 通过loss曲线形态判断LR问题：慢->小, 震荡->大, 平台->需衰减
- **参见**: [[lr-range-test]] | [[../debugging/training-curve-diagnosis]]

### 最优配置：SGD + 0.1lr + 0.9momentum + weight decay 0.005
- **来源**: 匿名用户
- **适用场景**: 经典CNN快速获得baseline
- **原文**: sgd，步长0.1，weight decay取0.005，momentum取0.9。dropout加relu。做完这些就应该跑出基本靠谱的结果。
- **要点**: 经典CNN基线：SGD, lr=0.1, momentum=0.9, wd=0.005, dropout+relu
- **参见**: [[sgd-momentum]], [[lr-range-test]]

### 正确初始化比调参更重要
- **来源**: Jarvix
- **适用场景**: 模型不收敛时的排错方向
- **原文**: 一次惨痛的教训是用初始化cnn的参数，最后acc只能到70%多，仅仅改成xavier，acc可以到98%。
- **要点**: 初始化不对会导致无论怎么调LR都无效，先确认初始化正确
- **参见**: [[../initialization/initialization]]

### 小数据微调：dropout和weight decay不宜过大
- **来源**: BugBuster喵
- **适用场景**: 小数据微调预训练模型
- **原文**: 小数据微调时weight decay太大可能损伤预训练能力，太小又容易过拟合。dropout在大预训练模型微调中不一定要大，0到0.1常见。
- **要点**: 小数据微调时weight decay保守（0.01开始），dropout偏小（0~0.1）
- **参见**: [[weight-decay]] | [[dropout]] | [[../llm-sft/llm-sft]]

### LR对Transformer的重要性
- **来源**: BugBuster喵
- **适用场景**: Transformer系列模型
- **原文**: Transformer 类模型很多时候 1e-4 到 3e-4 是常见起点，具体看模型大小和batch。warmup我现在几乎默认会开，尤其是Transformer。
- **要点**: Transformer lr在1e-4~3e-4，warmup默认必开
- **参见**: [[../nlp/bert-finetuning]] | [[learning-rate-scheduling]]

### 余弦退火的实现代码
- **来源**: LindgeWAI
- **适用场景**: 自定义WarmupCosineScheduler
- **原文**: WarmupCosineScheduler代码实现，warmup期间线性增长，之后cosine衰减
- **要点**: WarmupCosineScheduler可用Python自定义实现，warmup线性/之后余弦
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]]

### Cyclic LR是比固定衰减更好的策略
- **来源**: 阿里云云栖号
- **适用场景**: 替代step/指数衰减的通用方案
- **原文**: 实验表明训练期间使用不同的学习率总体上是有益的，因此建议在一个取值范围内周期性地改变学习率，而不是将其设定为固定值。
- **要点**: Cyclic LR在上下界间周期性变化，优于固定衰减策略
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]]

### 权重衰减与数据集规模的关系
- **来源**: 阿里云云栖号
- **适用场景**: 不同规模数据的正则化强度选择
- **原文**: 较小的数据集和模型结构设置较大的权重衰减值，而较大的数据集和更深的模型结构设置较小的值。
- **要点**: 小数据/浅层结构 -> 大weight decay；大数据/深层结构 -> 小weight decay
- **参见**: [[weight-decay]]

### LR太大时的泛化正则化效应
- **来源**: 阿里云云栖号
- **适用场景**: LR与正则化的关系
- **原文**: 如果使用恒定的学习率而不是使用学习率范围进行搜索，则最佳权重衰减会有所不同。由于较大的学习率提供正则化，因此较小的权重衰减值是最佳的。
- **要点**: 大的LR本身就有正则化效果，此时weight decay可设小
- **参见**: [[weight-decay]]

### 恒等映射与LR无关但影响优化
- **来源**: anchor
- **适用场景**: 深度网络结构设计
- **原文**: shortcut connection里，找不到concat，用add凑合吧，反之亦然。
- **要点**: shortcut/residual connection降低优化难度，使LR调节更鲁棒
- **参见**: [[../cnn-cv/skip-connections]]

### BN层对LR的加速效应
- **来源**: BBuf、anchor、Captain Jack
- **适用场景**: 任何深层网络的训练
- **原文**: Batch Normalization可以很大程度的加快收敛速度。建议搭建自己网络的时候尽量加上BN。
- **要点**: BN加快收敛，允许使用更大的LR
- **参见**: [[batch-normalization]]

### 先训大模型再裁剪优于直接训小模型
- **来源**: anchor
- **适用场景**: 模型容量与性能的权衡
- **原文**: 先训一个大模型然后裁剪，也许比直接训一个小模型性能好。
- **要点**: 先训练大容量模型后压缩，比直接训练小模型效果更好，LR策略也需相应调整
- **参见**: [[../regularization/overfitting-underfitting]]

### LR要随迭代次数2倍递减
- **来源**: Tang AI
- **适用场景**: 推荐系统等经典场景
- **原文**: 学习率最好是从高到底2倍速度递减一般从0.01开始。
- **要点**: LR从0.01开始，按2倍速度递减
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]]

### 推荐系统常用batch size与LR
- **来源**: Tang AI
- **适用场景**: 推荐系统
- **原文**: batch size对于推荐来说32-64-128-512测试效果再高一般也不会正向了，再低训练太慢了。
- **要点**: 推荐系统batch size 32~512，过大会负向
- **参见**: [[../recsys/recsys]]

### Warm-up + Cosine Decay的精调选项
- **来源**: 陈城南
- **适用场景**: 对已有baseline做精细调参
- **原文**: lr也可以调调，适当放大缩小看看，cos退火的起始点或者warm-up的持续时长，也可以试试改改
- **要点**: 可精调：cos起始点、warmup持续长度、LR初值
- **参见**: [[learning-rate-scheduling]], [[learning-rate-scheduling]]

### 仅调LR和batch size：大模型Scaling Law
- **来源**: Sam聊算法
- **适用场景**: LLM超大算力预训练
- **原文**: 给定计算成本C，大多数超参的最优值是基本固定的，只需调整学习率和batch size
- **要点**: 超大规模训练时只有LR和batch需要随算力调整，其余超参固定
- **参见**: [[../batch-size/batch-size-lr-scaling]] | [[../llm-sft/llm-sft]]

### 微调预训练模型时需低LR
- **来源**: BBuf（提了finetune方式）
- **适用场景**: 使用带backbone的预训练模型
- **原文**: 使用了带backbone的网络，如训练VGG16-SSD建议选择finetune的方式，从头训练不仅费时费力，甚至难以收敛。
- **要点**: 微调预训练模型需用更低的LR（通常1e-4及以下）
- **参见**: [[../llm-sft/llm-sft]] | [[lr-range-test]]

### 训练与推理的LR保持一致原则
- **来源**: BugBuster喵
- **适用场景**: 大模型微调chat template一致性
- **原文**: 大模型微调还有一个常见坑：chat template不一致。训练用一种模板，推理用另一种模板，效果会非常难看。
- **要点**: 虽然说的是template，但原理一致：训练推流程一致的LR才能复用
- **参见**: [[../llm-sft/llm-sft]]

### 复杂任务先小LR保不爆
- **来源**: Captain Jack
- **适用场景**: Loss设计复杂的任务、初始训练不稳定
- **原文**: 如果上面的Loss设计那块你没法合理，初始情况下容易爆，先上一个小LR保证不爆，等loss降下来了，再慢慢升LR
- **要点**: 不确定时先保守小LR跑稳定，再逐步提高到合适值
- **参见**: [[lr-range-test]]

### PPO RLHF完整LR策略
- **来源**: Tang AI
- **适用场景**: RLHF/PPO全流程训练
- **原文**: PPO学习率通常比SFT小一个数量级；RLHF阶段batch_size宁大勿小；梯度裁剪0.2；reward归一化；KL惩罚从0.001开始。
- **要点**: PPO调参：LR更小、batch更大、grad clip、reward归一化
- **参见**: [[../llm-rl/llm-rl]] | [[../llm-rl/ppo-rlhf]]

### 模型训练初期输出重复/无意义 -> LR太高
- **来源**: Tang AI
- **适用场景**: 大模型训练模式崩溃
- **原文**: 模型训练初期就输出大量重复或者无意义的内容，学习率过高。过大的学习率可能导致模型参数更新过于剧烈，跳出了有效的参数空间，导致mode collapse。
- **要点**: 训练初期输出重复/无意义内容 -> 立即降低LR
- **参见**: [[lr-range-test]] | [[../llm-rl/reward-hacking]]

### 梯度检查与裁剪的实践
- **来源**: 摘星狐狸
- **适用场景**: 手动梯度检查、防止梯度爆炸
- **原文**: 如果你没用过Theano或者Torch，梯度实现就只能靠自己人肉算了。梯度实现还真的挺容易出错的。梯度裁剪的一般步骤：计算梯度，计算范数，裁剪，再应用。
- **要点**: 梯度裁剪通用做法：计算梯度范数 -> 按阈值裁剪 -> 应用更新
- **参见**: [[gradient-clipping]]

### Adam收敛快但可能泛化差
- **来源**: 黑马程序员Python、罗浩.ZJU
- **适用场景**: 需要权衡收敛速度与泛化性能
- **原文**: 如果你关心快速收敛，使用自适应优化器如Adam，但它可能会陷入局部极小，提供了糟糕的泛化。SGD+momentum可以实现找到全局最小值，但需要更长时间收敛。
- **要点**: Adam收敛快但泛化可能差，SGD+Momentum慢但效果好
- **参见**: [[adam]], [[adam]]

### Adam默认LR推荐3e-4
- **来源**: 量子位
- **适用场景**: 通用Deep Learning默认设置
- **原文**: Adam优化器学习速率3e-4
- **要点**: 使用Adam时默认LR=3e-4是一个广泛使用的经验值
- **参见**: [[lr-range-test]] | [[adam]]

### 初始LR推荐3.5e-4
- **来源**: 陈城南
- **适用场景**: CV任务Adam+Cosine基线
- **原文**: 学习率初始时可以设置3.5e-4，可以适当进行调整，基本就在5e-4 ~ 5e-5之间
- **要点**: CV通用初始LR 3.5e-4（Adam），范围5e-4~5e-5
- **参见**: [[lr-range-test]] | [[../cnn-cv/cnn-cv]]

### 学习率热启动（warm restart）
- **来源**: Tang AI
- **适用场景**: 推荐系统老汤模型热启动
- **原文**: 对于推荐系统的老汤模型，建议热启动，但是热启动大概率也是追平
- **要点**: 老模型热启动需embedding+DNN全量加载，仅对倒数第一层做扰动
- **参见**: [[learning-rate-scheduling]] | [[../recsys/recsys]]

### 超参优化方法对比
- **来源**: 京东白条-王惠东、摘星狐狸、阿里云云栖号、李文博
- **适用场景**: 自动调参工具选择
- **原文**: Grid Search在所有候选参数中循环遍历，最费时但保证最优；Random Search效率更高，常与粗到细结合；Bayesian Optimization利用历史实验结果，效率最高。
- **要点**: 网格搜索(n<=4维) -> 随机搜索(高维) -> 贝叶斯优化(最先进)
- **参见**: [[../infra/wandb-sweeps]]

### 自动调参工具推荐
- **来源**: 润润、BugBuster喵、苏辄
- **适用场景**: 系统化超参搜索
- **原文**: wandb可以大幅提升深度学习效率，支持网格搜索和随机搜索。Optuna对中小团队友好，Ray Tune适合集群，W&B记录可视化方便。
- **要点**: 自动调参工具：wandb(入门友好), Optuna(中小团队), Ray Tune(集群)
- **参见**: [[../infra/wandb-sweeps]] | [[../infra/wandb-sweeps]]

### 网格搜索精调的实用技巧
- **来源**: 陈城南
- **适用场景**: 有限预算下的精细调参
- **原文**: 调参时一定要先文字记录，然后跑模型填空，一定要保证自己知道自己这个实验的目的是什么
- **要点**: 网格搜索时每个实验必须记录目的和假设，避免盲目试参
- **参见**: [[../infra/wandb-sweeps]] | [[../infra/wandb-sweeps]]

### 由粗到细随机搜索推荐
- **来源**: 量子位、京东白条-王惠东
- **适用场景**: 超参数范围不确定时
- **原文**: 由粗到细的随机搜索可以缩小超高性能参数的范围
- **要点**: 先用随机搜索大范围采样确认有效区间，再在小区间内精细搜索
- **参见**: [[../infra/wandb-sweeps]]

### 无脑用SGD+Momentum
- **来源**: Captain Jack、匿名用户、随机漫步的傻瓜
- **适用场景**: CV任务、不愿意纠结优化器选择
- **原文**: sgd adam这些选择上，看你个人选择。一般对网络不是决定性的。反正我无脑用sgd + momentum。
- **要点**: SGD+Momentum在CV领域是最稳妥的选择
- **参见**: [[sgd-momentum]], [[../cnn-cv/cnn-cv]]

### 预热扫描的实用操作
- **来源**: 四点半的理一
- **适用场景**: 快速定位合理LR
- **原文**: 从1e-5开始，以10倍为单位往上跳，观察loss曲线。平稳下降到某一点开始发散，那发散前一档就是候选LR。
- **要点**: 1e-5 -> 1e-4 -> 1e-3 -> 1e-2逐档扫描找发散临界点
- **参见**: [[lr-range-test]], [[lr-range-test]]

### LR和Momentum的组合搜索
- **来源**: 阿里云云栖号
- **适用场景**: Cyclical LR + Cyclical Momentum
- **原文**: 最佳学习率取决于动量，而动量又取决于学习率。在不引起训练不稳定的情况下尽可能设置大的动量值。
- **要点**: LR和Momentum相互依赖，需联合搜索；使用Cyclical策略时两者应反向周期变化
- **参见**: [[sgd-momentum]], [[learning-rate-scheduling]], [[sgd-momentum]]

### SFT + RL 两阶段LR策略
- **来源**: Tang AI
- **适用场景**: RLHF全流程
- **原文**: SFT是为了让模型有能力去回答问题，而RL仅仅是将这个能力推到上限。SFT后用RL精调，KL散度从0.001开始。
- **要点**: SFT阶段LR较大让模型学到能力，RL阶段LR缩小一个数量级做精调
- **参见**: [[../llm-sft/llm-sft]] | [[../llm-rl/llm-rl]]

### 模型未充分训练时LR不够的问题
- **来源**: 唐AI
- **适用场景**: 大模型训练loss指标异常的诊断
- **原文**: 到了中后期，loss小效果不一定好。从小步数的checkpoint进行accuracy或其他业务指标判断。
- **要点**: 大模型训练后期LR衰减太猛可能导致loss虽低但业务指标差，需checkpoint评估
- **参见**: [[lr-range-test]] | [[eval-metrics]]

### 梯度累积做等效大批次的LR策略
- **来源**: LindgeWAI
- **适用场景**: 显存不足时扩大有效batch size
- **原文**: 梯度累积：增大batch，规避GPU内存的限制
- **要点**: 梯度累积使有效batch变大，相应需调整LR（batch翻倍LR可翻倍）
- **参见**: [[gradient-accumulation]] | [[../batch-size/batch-size]]

### 逐特征加特征排查LR效果
- **来源**: Tang AI
- **适用场景**: 推荐系统特征工程
- **原文**: 做推荐的同学，一般特征不要直接全怼进去，最好是一个一个的加特征进行效果测试，因为有些特征可能导致模型过拟合
- **要点**: 推荐系统特征逐个加入测试，避免引入干扰特征使LR失效
- **参见**: [[../recsys/recsys]] | [[../recsys/feature-binning]]
