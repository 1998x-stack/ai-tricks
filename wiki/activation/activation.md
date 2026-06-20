# 激活函数

## 概述

激活函数是神经网络中引入非线性的核心组件，选型直接影响收敛速度与最终性能。ReLU 是隐层默认首选，输出层则需根据任务类型选择 softmax/sigmoid/线性，且**最后一层永远不要用 ReLU**。

## 详细知识

- [[relu]] — ReLU深入解析：公式、优势、dying ReLU问题与解法
- [[activation-selection]] — 激活函数完整选型指南：隐层vs输出层、按任务类型
- [[leaky-relu-prelu]] — LeakyReLU、PReLU等ReLU变体：何时从ReLU升级

## 核心技巧

### ReLU 是隐层的默认激活函数
- **来源**: 多位作者（爱睡觉的KKY、苏辄、罗浩.ZJU 等）
- **适用场景**: 所有深度神经网络的中间隐层，尤其 CV 领域
- **原文**: "无脑用ReLU(CV领域)" / "relu 是一个很万能的激活函数，可以很好的防止梯度弥散问题" / "请好好使用ReLU激活函数" / "用极简的方式实现非线性激活，缓解梯度消失"
- **要点**: ReLU 计算简单、收敛快、有效缓解梯度消失，CV 领域可直接作为默认选择。
- **参见**: [[relu]], [[../initialization/initialization]]

### 最后一层切勿使用 ReLU
- **来源**: 爱睡觉的KKY、罗浩.ZJU
- **适用场景**: 所有网络的输出层
- **原文**: "最后一层千万不要relu，因为relu可能把很多数值都干为0" / "最后一层的激活函数千万慎用relu"
- **要点**: ReLU 会将负值截断为 0，输出层使用会导致信息丢失，破坏预测结果。连续值用 identity，分类用 softmax，回归直接 wx+b。
- **参见**: [[relu]], [[../data-eval/eval-metrics]]

### 输出层激活函数按任务类型选择
- **来源**: 王惠东（京东）、苏辄
- **适用场景**: 输出层设计
- **原文**: "多分类任务选用softmax输出，二分类任务选用sigmoid输出，回归任务选用线性输出" / "如果是连续的用identify，分类的用softmax，拟合回归的话直接wx +b就行"
- **要点**: 多分类 → softmax；二分类 → sigmoid；回归 → 线性/identity/无激活函数。
- **参见**: [[../data-eval/eval-metrics]]

### 中间隐层优先用 ReLU，RNN 用 tanh
- **来源**: 王惠东（京东）
- **适用场景**: 隐层激活函数选型
- **原文**: "对于中间隐层，则优先选择relu激活函数（relu激活函数可以有效的解决sigmoid和tanh出现的梯度弥散问题，多次实验表明它会比其他激活函数以更快的速度收敛）。另外，构建序列神经网络（RNN）时要优先选用tanh激活函数。"
- **要点**: ReLU 收敛速度优于 sigmoid/tanh；RNN/序列网络优先用 tanh。
- **参见**: [[relu]], [[activation-selection]], [[../normalization/normalization]]

### Sigmoid 和 tanh 的梯度消失陷阱
- **来源**: 苏辄
- **适用场景**: 避免梯度消失
- **原文**: "Sigmoid：适合概率输出，但可能导致梯度消失"
- **要点**: Sigmoid/tanh 在饱和区梯度趋近于零，深层网络中易导致梯度消失。sigmoid 仅推荐用于二分类输出层或需要概率输出的场景，tanh 推荐用于 RNN。
- **参见**: [[activation-selection]], [[../normalization/normalization]]

### LeakyReLU / PReLU 应对 ReLU 神经元死亡
- **来源**: 爱睡觉的KKY、多位 CNN 从业者
- **适用场景**: 当 ReLU 出现神经元死亡（dying ReLU）问题时
- **原文**: "对于激活函数，relu也在大多数情况下表现不错，当然也可以试试leakyrelu和prelu，prelu貌似效果还不错" / "如果还想再提升精度可以将ReLU改成PReLU试试" / "为了解决 ReLU 激活函数中的梯度消失问题，也可以更换成Leaky ReLU激活函数等其他变体"
- **要点**: ReLU 的负半轴恒为 0 可能导致神经元永久失活。LeakyReLU 引入小斜率、PReLU 引入可学习斜率，是 dying ReLU 的标准解法。
- **参见**: [[leaky-relu-prelu]], [[../regularization/regularization]]

### Softmax 变体（AM-Softmax 等）增大类间间距
- **来源**: 知乎社区
- **适用场景**: 当分类精度因类别相似不够高时
- **原文**: "如果你的分类精度不够是因为有两类或者多类太相近造成的，考虑使用其他softmax，比如amsoftmax" / "各种魔改的softmax能更好的增大类间差距"
- **要点**: 标准 softmax 仅优化类别分离，AM-Softmax、ArcFace、CosFace 等变体通过加 margin 增大类间距离，适合细粒度分类和人脸识别。
- **参见**: [[../data-eval/eval-metrics]]

### 召回场景：softmax 优于 sigmoid（listwise 学习）
- **来源**: 爱睡觉的KKY
- **适用场景**: 推荐系统召回
- **原文**: "推荐召回用softmax效果比sigmoid更好，意思就是召回更适合对比学习那种listwise学习方式"
- **要点**: 推荐召回本质是 listwise 排序问题，softmax 的多类别对比学习方式优于 sigmoid 的点式二分类。
- **参见**: [[../data-eval/eval-metrics]]

### ReLU 学习率不能设太大，否则神经元易死
- **来源**: 知乎社区
- **适用场景**: 学习率调节 + ReLU 搭配
- **原文**: "LR在可以工作的最大值下往小收一收, 免得ReLU把神经元弄死了"
- **要点**: 学习率过大会使大量 ReLU 神经元陷入永不激活的状态。在可工作的最大 LR 下适当调小，在收敛速度和神经元存活率之间平衡。
- **参见**: [[relu]], [[../optimizer-lr/adam]]

### Conv → MaxPool → ReLU 顺序更省计算
- **来源**: 答主「十九」
- **适用场景**: CNN 网络结构设计
- **原文**: "在ReLU之前使用最大池化来节省一些计算。由于ReLU阈值的值为0：f(x)=max(0,x)和最大池化只有max激活：f(x)=max(x1,x2,...,xi)，使用Conv > MaxPool > ReLU 而不是Conv > ReLU > MaxPool" / "MaxPool > ReLU = max(0, max(0.5, -0.5)) = 0.5 和 ReLU > MaxPool = max(max(0,0.5), max(0,-0.5)) = 0.5 ... 使用MaxPool > ReLU可以节省一个max操作"
- **要点**: 对于 ReLU（非负阈值），MaxPool 和 ReLU 复合后在数学上等价于 Conv → MaxPool → ReLU 也可得到相同结果，且少做一次 ReLU 的逐元素操作，节省计算量。
- **参见**: [[../cnn-cv/3x3-conv-rule]]

### 欠拟合时可尝试加入 Mish / ReLU 等激活层
- **来源**: 爱睡觉的KKY
- **适用场景**: 模型欠拟合时
- **原文**: "考虑增加relu，mish等作为某些层的激活函数"
- **要点**: 模型欠拟合时，除增大宽度深度外，可在特定层加入 ReLU/Mish 等激活函数来增加非线性表达能力。
- **参见**: [[activation-selection]], [[../debugging/small-data-overfit-test]]

### ReLU6：限制上界，适合移动设备
- **来源**: 苏辄
- **适用场景**: 移动端 / 低算力设备
- **原文**: "ReLu6：限制激活值的上界，超过6的统一算为6，减轻运算压力，适合移动设备"
- **要点**: ReLU6 将输出截断至 [0, 6] 区间，减少激活值的动态范围，降低量化难度和计算压力，MobileNet 等轻量网络常用。
- **参见**: [[../infra/mixed-precision]]

### ReLU 激活函数推荐搭配 He 初始化
- **来源**: 王惠东（京东）、多位答主
- **适用场景**: 权重初始化策略
- **原文**: "relu激活函数初始化推荐使用He normal，tanh初始化推荐使用Glorot normal" / "Xavier初始化适用于常见的激活函数（如tanh和sigmoid），而He初始化适用于ReLU激活函数" / "在使用ReLU激活函数时，通常推荐使用这种高斯初始化，通过缩放方差为2/n，以更好地适应ReLU的性质"
- **要点**: ReLU → He normal（方差 2/n）；tanh/sigmoid → Xavier/Glorot normal。激活函数与初始化方法必须匹配，错误搭配会导致梯度爆炸或消失。
- **参见**: [[relu]], [[../initialization/initialization]]

### 调参顺序：隐层数 → 神经元数 → 激活函数
- **来源**: 知乎社区
- **适用场景**: 网络调参流程
- **原文**: "先疯狂加隐藏层，一直加到过拟合，然后开始减隐藏层，然后开始调神经元个数，然后开始调激活函数，然后开始调隐藏层的类型..."
- **要点**: 激活函数的调优优先级低于网络深度和宽度。应先用 ReLU 做 baseline，再尝试 PReLU 等变体来提升精度。
- **参见**: [[activation-selection]], [[../debugging/small-data-overfit-test]]

### 做新模型的第一步：不加激活函数
- **来源**: 爱睡觉的KKY
- **适用场景**: 新模型调试起点
- **原文**: "做新模型的时候，最开始不要加激活函数，不要加batchnorm，不要加dropout，先就纯模型。然后再一步一步的实验"
- **要点**: 先从纯线性模型起步，确认 baseline 正确后再逐步加入激活函数和正则化，避免一次性引入过多因素导致问题无法定位。
- **参见**: [[../normalization/normalization]], [[../regularization/regularization]]

### 业务场景中不宜过早更换激活函数
- **来源**: 知乎社区
- **适用场景**: 工程实践
- **原文**: "很多人一看效果不行就加层、改 attention、改激活函数、换 loss。大多数业务场景里，这些不是第一优先级。先用成熟结构，先把数据、学习率、训练流程做好。"
- **要点**: 业务调参的优先级：数据质量 → 学习率 → 训练流程 → 网络结构。改激活函数属于结构创新，应在基础工作做好之后考虑。
- **参见**: [[../data-eval/data-eval]]
