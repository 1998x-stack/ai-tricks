# normalization 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [normalization-001-01](normalization-001-01/README.md) 搭建新模型先不加归一化层，逐步实验 | `model.norm` | `batch` → `none` |
| [normalization-002-01](normalization-002-01/README.md) 过拟合时先 Dropout 后 BN，严重时对 Embedding 层加 | `model.dropout` | `0.0` → `0.5` |
| [normalization-002-02](normalization-002-02/README.md) 过拟合时先 Dropout 后 BN，严重时对 Embedding 层加 | `model.norm` | `none` → `batch` |
| [normalization-002-03](normalization-002-03/README.md) 过拟合时先 Dropout 后 BN，严重时对 Embedding 层加 | `model.embedding_dropout` | `0.0` → `0.1` |
| [normalization-003-01](normalization-003-01/README.md) BN 能大幅加速收敛，有 BN 可省略 Dropout | `model.dropout` | `0.5` → `0.0` |
| [normalization-004-01](normalization-004-01/README.md) 序列特征用 LayerNorm，其他用 BatchNorm | `model.norm` | `batch` → `layer` |
| [normalization-005-01](normalization-005-01/README.md) Batch Size 决定归一化方案：大 Batch 用 BN，小 Batch 用 IN 或 GN | `model.norm` | `batch` → `group` |
| [normalization-005-02](normalization-005-02/README.md) Batch Size 决定归一化方案：大 Batch 用 BN，小 Batch 用 IN 或 GN | `model.norm` | `batch` → `instance` |
| [normalization-006-01](normalization-006-01/README.md) 多机多卡训练用同步 BN | `model.norm` | `batch` → `sync_batch` |
| [normalization-007-01](normalization-007-01/README.md) 各种 Norm 是低成本优化方向 | `model.norm` | `none` → `layer` |
| [normalization-007-02](normalization-007-02/README.md) 各种 Norm 是低成本优化方向 | `model.norm` | `none` → `weight` |
| [normalization-008-01](normalization-008-01/README.md) 不仅输入要归一化，输出也要 | `data.target_normalization` | `False` → `True` |
| [normalization-009-01](normalization-009-01/README.md) BN 是必选组件 | `model.norm` | `none` → `batch` |
| [normalization-010-01](normalization-010-01/README.md) BN 是大杀器，能极大加快训练 | `model.norm` | `none` → `batch` |
| [normalization-011-01](normalization-011-01/README.md) Group Normalization 可替代 BN | `model.norm` | `batch` → `group` |
| [normalization-012-01](normalization-012-01/README.md) Group Normalization 的 TensorFlow 实现 | `model.norm` | `batch` → `group` |
| [normalization-013-01](normalization-013-01/README.md) BN 的定义与原理 | `model.norm` | `none` → `batch` |
| [normalization-014-01](normalization-014-01/README.md) LRN 基本可弃用 | `model.norm` | `local_response` → `none` |
| [normalization-015-01](normalization-015-01/README.md) 适当考虑 BN | `model.norm` | `none` → `batch` |
