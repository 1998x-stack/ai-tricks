# activation 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [activation-001-01](activation-001-01/README.md) ReLU 是隐层的默认激活函数 | `model.activation` | `sigmoid` → `relu` |
| [activation-002-01](activation-002-01/README.md) 最后一层切勿使用 ReLU | `model.output_activation` | `relu` → `identity` |
| [activation-003-01](activation-003-01/README.md) 输出层激活函数按任务类型选择 | `model.output_contract` | `softmax_before_cross_entropy` → `logits_cross_entropy` |
| [activation-003-02](activation-003-02/README.md) 输出层激活函数按任务类型选择 | `model.output_contract` | `relu_regression` → `linear_regression` |
| [activation-003-03](activation-003-03/README.md) 输出层激活函数按任务类型选择 | `model.output_contract` | `sigmoid_before_bce_logits` → `logits_bce` |
| [activation-004-01](activation-004-01/README.md) 中间隐层优先用 ReLU，RNN 用 tanh | `model.rnn_activation` | `relu` → `tanh` |
| [activation-004-02](activation-004-02/README.md) 中间隐层优先用 ReLU，RNN 用 tanh | `model.activation` | `sigmoid` → `relu` |
| [activation-005-01](activation-005-01/README.md) Sigmoid 和 tanh 的梯度消失陷阱 | `model.activation` | `sigmoid` → `tanh` |
| [activation-006-01](activation-006-01/README.md) LeakyReLU / PReLU 应对 ReLU 神经元死亡 | `model.activation` | `relu` → `leaky_relu` |
| [activation-006-02](activation-006-02/README.md) LeakyReLU / PReLU 应对 ReLU 神经元死亡 | `model.activation` | `relu` → `prelu` |
| [activation-007-01](activation-007-01/README.md) Softmax 变体（AM-Softmax 等）增大类间间距 | `loss.kind` | `cross_entropy` → `am_softmax` |
| [activation-007-02](activation-007-02/README.md) Softmax 变体（AM-Softmax 等）增大类间间距 | `loss.kind` | `cross_entropy` → `arcface` |
| [activation-008-01](activation-008-01/README.md) 召回场景：softmax 优于 sigmoid（listwise 学习） | `loss.retrieval` | `pointwise_bce` → `listwise_softmax` |
| [activation-009-01](activation-009-01/README.md) ReLU 学习率不能设太大，否则神经元易死 | `optimizer.lr` | `0.1` → `0.01` |
| [activation-010-01](activation-010-01/README.md) Conv → MaxPool → ReLU 顺序更省计算 | `model.pool_relu_order` | `relu_pool` → `pool_relu` |
| [activation-011-01](activation-011-01/README.md) 欠拟合时可尝试加入 Mish / ReLU 等激活层 | `model.activation` | `identity` → `mish` |
| [activation-011-02](activation-011-02/README.md) 欠拟合时可尝试加入 Mish / ReLU 等激活层 | `model.activation` | `identity` → `relu` |
| [activation-012-01](activation-012-01/README.md) ReLU6：限制上界，适合移动设备 | `model.activation` | `relu` → `relu6` |
| [activation-013-01](activation-013-01/README.md) ReLU 激活函数推荐搭配 He 初始化 | `model.weight_init` | `xavier_normal` → `kaiming_normal` |
| [activation-013-02](activation-013-02/README.md) ReLU 激活函数推荐搭配 He 初始化 | `model.weight_init` | `kaiming_normal` → `xavier_normal` |
| [activation-014-01](activation-014-01/README.md) 调参顺序：隐层数 → 神经元数 → 激活函数 | `search.policy` | `activation_first` → `depth_width_activation` |
| [activation-015-01](activation-015-01/README.md) 做新模型的第一步：不加激活函数 | `model.activation` | `relu` → `identity` |
| [activation-016-01](activation-016-01/README.md) 业务场景中不宜过早更换激活函数 | `search.policy` | `activation_first` → `data_lr_first` |
