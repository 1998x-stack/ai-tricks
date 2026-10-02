# regularization 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [regularization-001-01](regularization-001-01/README.md) 先过拟合再正则化（Overfit First） | `search.policy` | `regularization_first` → `tiny_overfit_first` |
| [regularization-002-01](regularization-002-01/README.md) Dropout | `model.dropout` | `0.0` → `0.5` |
| [regularization-002-02](regularization-002-02/README.md) Dropout | `model.dropout` | `0.1` → `0.0` |
| [regularization-002-03](regularization-002-03/README.md) Dropout | `inference.dropout_enabled` | `True` → `False` |
| [regularization-003-01](regularization-003-01/README.md) Weight Decay（L2 正则化） | `optimizer.weight_decay` | `0.0` → `0.01` |
| [regularization-003-02](regularization-003-02/README.md) Weight Decay（L2 正则化） | `optimizer.weight_decay` | `0.01` → `0.001` |
| [regularization-003-03](regularization-003-03/README.md) Weight Decay（L2 正则化） | `optimizer.weight_decay` | `0.001` → `0.0001` |
| [regularization-004-01](regularization-004-01/README.md) L1 正则化 | `loss.l1` | `0.0` → `0.0001` |
| [regularization-005-01](regularization-005-01/README.md) Early Stopping（早停） | `training.patience` | `0` → `5` |
| [regularization-005-02](regularization-005-02/README.md) Early Stopping（早停） | `training.patience` | `2` → `10` |
| [regularization-005-03](regularization-005-03/README.md) Early Stopping（早停） | `search.pruning` | `none` → `asha` |
| [regularization-006-01](regularization-006-01/README.md) 数据增强（Data Augmentation）作为正则化 | `data.augmentation` | `none` → `label_preserving` |
| [regularization-006-02](regularization-006-02/README.md) 数据增强（Data Augmentation）作为正则化 | `data.augmentation` | `none` → `horizontal_flip` |
| [regularization-006-03](regularization-006-03/README.md) 数据增强（Data Augmentation）作为正则化 | `data.augmentation` | `none` → `random_crop` |
| [regularization-007-01](regularization-007-01/README.md) 标签平滑（Label Smoothing） | `loss.label_smoothing` | `0.0` → `0.1` |
| [regularization-008-01](regularization-008-01/README.md) SpatialDropout | `model.spatial_dropout` | `0.0` → `0.2` |
| [regularization-009-01](regularization-009-01/README.md) Max-Norm 约束（权重约束） | `training.max_weight_norm` | `0.0` → `2.0` |
| [regularization-010-01](regularization-010-01/README.md) Flooding（训练loss阈值） | `loss.flooding` | `0.0` → `0.1` |
| [regularization-011-01](regularization-011-01/README.md) Mixup 与 CutMix | `data.augmentation` | `none` → `mixup` |
| [regularization-011-02](regularization-011-02/README.md) Mixup 与 CutMix | `data.augmentation` | `none` → `cutmix` |
| [regularization-012-01](regularization-012-01/README.md) Ensemble 集成方法 | `inference.ensemble_members` | `1` → `3` |
| [regularization-012-02](regularization-012-02/README.md) Ensemble 集成方法 | `inference.ensemble_diversity` | `initialization` → `hyperparameters` |
| [regularization-012-03](regularization-012-03/README.md) Ensemble 集成方法 | `inference.ensemble_diversity` | `initialization` → `architecture` |
| [regularization-013-01](regularization-013-01/README.md) BN 的正则化效果与 Dropout 的交互 | `model.dropout` | `0.0` → `0.5` |
| [regularization-014-01](regularization-014-01/README.md) 泛化诊断框架（过拟合 vs 欠拟合） | `search.policy` | `random_changes` → `train_val_gap_diagnosis` |
| [regularization-015-01](regularization-015-01/README.md) 梯度裁剪（Gradient Clipping）与泛化 | `training.clip_norm` | `0.0` → `1.0` |
| [regularization-016-01](regularization-016-01/README.md) SGD 比 Adam 泛化更好的取舍 | `optimizer.kind` | `adam` → `sgd` |
