# initialization 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [initialization-001-01](initialization-001-01/README.md) Xavier/Glorot 初始化（通用首选） | `model.weight_init` | `normal_fan_in` → `xavier_normal` |
| [initialization-002-01](initialization-002-01/README.md) He 初始化（ReLU 专用） | `model.weight_init` | `xavier_normal` → `kaiming_normal` |
| [initialization-003-01](initialization-003-01/README.md) 截断正态初始化（Truncated Normal）用于 Embedding | `model.embedding_init` | `normal` → `truncated_normal` |
| [initialization-004-01](initialization-004-01/README.md) Uniform vs Normal 初始化的选择 | `model.embedding_init` | `normal` → `uniform_equal_variance` |
| [initialization-005-01](initialization-005-01/README.md) SVD 初始化用于 RNN | `model.recurrent_init` | `normal` → `orthogonal_svd` |
| [initialization-006-01](initialization-006-01/README.md) 零初始化的神话与陷阱 | `model.weight_init` | `zeros` → `xavier_normal` |
| [initialization-006-02](initialization-006-02/README.md) 零初始化的神话与陷阱 | `model.bias_init` | `zeros` → `normal` |
| [initialization-007-01](initialization-007-01/README.md) TensorFlow 与 PyTorch 初始化差异 | `training.framework` | `pytorch` → `tensorflow` |
| [initialization-008-01](initialization-008-01/README.md) 权重用随机初始化，Bias 恒为零 | `model.bias_init` | `normal` → `zeros` |
| [initialization-009-01](initialization-009-01/README.md) 不同初始化做 Ensemble | `inference.ensemble_init` | `['xavier_normal']` → `['xavier_normal', 'kaiming_normal', 'orthogonal']` |
| [initialization-009-02](initialization-009-02/README.md) 不同初始化做 Ensemble | `inference.ensemble_vote` | `hard` → `soft` |
| [initialization-010-01](initialization-010-01/README.md) 预训练模型加载后必须验证 | `validation.load_check` | `unchecked` → `keys_shapes_hash` |
| [initialization-011-01](initialization-011-01/README.md) 长训练中换初始化方法跳出局部最优 | `model.weight_init` | `default` → `xavier_uniform` |
| [initialization-012-01](initialization-012-01/README.md) DeepSeek 大模型初始化标准差 | `model.init_std` | `0.02` → `0.06` |
| [initialization-012-02](initialization-012-02/README.md) DeepSeek 大模型初始化标准差 | `model.init_std` | `0.02` → `0.04` |
| [initialization-013-01](initialization-013-01/README.md) 正交初始化用于深层网络 | `model.weight_init` | `normal_fan_in` → `orthogonal` |
| [initialization-014-01](initialization-014-01/README.md) 初始化对 CNN 的巨大影响（案例） | `model.weight_init` | `default` → `xavier_uniform` |
| [initialization-015-01](initialization-015-01/README.md) 先过拟合再谈初始化 | `search.policy` | `init_search_first` → `tiny_overfit_first` |
