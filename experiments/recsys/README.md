# recsys 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [recsys-001-01](recsys-001-01/README.md) 热启动必须 Embedding + DNN 全部加载 | `checkpoint.warm_start` | `embedding_only` → `embedding_and_dnn` |
| [recsys-001-02](recsys-001-02/README.md) 热启动必须 Embedding + DNN 全部加载 | `checkpoint.perturb_scope` | `all_layers` → `last_layer` |
| [recsys-002-01](recsys-002-01/README.md) 连续特征分箱（优于神经网络直接处理） | `data.numeric_transform` | `raw` → `train_quantile_bins` |
| [recsys-002-02](recsys-002-02/README.md) 连续特征分箱（优于神经网络直接处理） | `data.outlier_bin` | `False` → `True` |
| [recsys-002-03](recsys-002-03/README.md) 连续特征分箱（优于神经网络直接处理） | `data.bin_selection` | `fixed_quantiles` → `train_iv_selected` |
| [recsys-003-01](recsys-003-01/README.md) 特征逐个加入测试 | `data.feature_policy` | `all_at_once` → `forward_selection` |
| [recsys-004-01](recsys-004-01/README.md) 召回负采样策略 | `data.negative_source` | `exposed_unclicked` → `global_random` |
| [recsys-005-01](recsys-005-01/README.md) 粗排用精排 Top-K 做负样本 | `data.negative_source` | `global_random` → `ranker_topk` |
| [recsys-006-01](recsys-006-01/README.md) 多任务学习：Gradient Surgery 优先，不推荐 Pareto | `training.multitask_gradient` | `sum` → `pcgrad` |
| [recsys-006-02](recsys-006-02/README.md) 多任务学习：Gradient Surgery 优先，不推荐 Pareto | `training.multitask_gradient` | `pareto` → `pcgrad` |
| [recsys-007-01](recsys-007-01/README.md) 多任务各 Loss 应在同一量级 | `loss.task_weighting` | `unscaled` → `train_scale_equalized` |
| [recsys-008-01](recsys-008-01/README.md) 推荐序列：Pooling 与 Attention 效果差异不大 | `model.pooling` | `masked_mean` → `attention` |
| [recsys-009-01](recsys-009-01/README.md) SENet 与 LHUC：可解释特征重要性 | `model.feature_gate` | `none` → `senet` |
| [recsys-009-02](recsys-009-02/README.md) SENet 与 LHUC：可解释特征重要性 | `model.feature_gate` | `none` → `lhuc` |
| [recsys-010-01](recsys-010-01/README.md) 不推荐用强化学习优化推荐系统 | `training.objective` | `supervised` → `policy_gradient` |
| [recsys-011-01](recsys-011-01/README.md) 召回用 Softmax 优于 Sigmoid | `loss.retrieval` | `pointwise_bce` → `listwise_softmax` |
| [recsys-012-01](recsys-012-01/README.md) Batch Size 范围：32 - 128 - 512 | `training.batch_size` | `32` → `128` |
| [recsys-012-02](recsys-012-02/README.md) Batch Size 范围：32 - 128 - 512 | `training.batch_size` | `128` → `512` |
| [recsys-013-01](recsys-013-01/README.md) 稀疏特征用 Adagrad，稠密特征用 Adam | `optimizer.sparse_kind` | `adam` → `adagrad` |
| [recsys-014-01](recsys-014-01/README.md) 数据稀疏时用自适应优化器 | `optimizer.kind` | `sgd` → `adagrad` |
| [recsys-015-01](recsys-015-01/README.md) Embedding L2 Normalization 反而不利于召回 | `model.embedding_l2_normalize` | `True` → `False` |
| [recsys-016-01](recsys-016-01/README.md) 数据量巨大时一个 Epoch 足矣 | `training.epochs` | `3` → `1` |
| [recsys-017-01](recsys-017-01/README.md) 模型构建顺序：先不加激活函数、BN、Dropout | `model.activation` | `relu` → `identity` |
| [recsys-018-01](recsys-018-01/README.md) 序列特征用 LayerNorm，非序列用 BatchNorm | `model.norm` | `batch` → `layer` |
| [recsys-019-01](recsys-019-01/README.md) 负样本过多时监控 F1/AUC 而非 Loss | `checkpoint.selection_metric` | `val_loss` → `val_auc` |
| [recsys-020-01](recsys-020-01/README.md) 两阶段训练：Adam 预热 + SGD 收尾 | `optimizer.kind` | `adam` → `adam_then_sgd` |
| [recsys-021-01](recsys-021-01/README.md) 确认数据无泄漏后再调参 | `data.split_policy` | `random` → `time` |
| [recsys-022-01](recsys-022-01/README.md) 负采样需避免污染正样本 | `data.exclude_known_positive` | `False` → `True` |
| [recsys-023-01](recsys-023-01/README.md) Attention 优于 Pooling（通用原则） | `model.pooling` | `masked_mean` → `attention` |
