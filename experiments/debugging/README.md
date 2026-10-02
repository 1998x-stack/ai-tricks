# debugging 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [debugging-001-01](debugging-001-01/README.md) 小数据过拟合测试 | `validation.preflight` | `none` → `tiny_overfit` |
| [debugging-002-01](debugging-002-01/README.md) 随机标签测试 | `data.train_labels` | `original` → `permuted` |
| [debugging-003-01](debugging-003-01/README.md) 五种最常见的实现Bug | `validation.preflight` | `none` → `shape_dtype_loss_mode` |
| [debugging-003-02](debugging-003-02/README.md) 五种最常见的实现Bug | `validation.shape_check` | `False` → `True` |
| [debugging-003-03](debugging-003-03/README.md) 五种最常见的实现Bug | `validation.dtype_check` | `False` → `True` |
| [debugging-003-04](debugging-003-04/README.md) 五种最常见的实现Bug | `training.force_eval_mode` | `True` → `False` |
| [debugging-003-05](debugging-003-05/README.md) 五种最常见的实现Bug | `loss.kind` | `cross_entropy_double_softmax` → `cross_entropy` |
| [debugging-004-01](debugging-004-01/README.md) NaN诊断：突然出现的NaN | `validation.nan_guard` | `none` → `input_and_loss` |
| [debugging-004-02](debugging-004-02/README.md) NaN诊断：突然出现的NaN | `loss.numeric_implementation` | `naive_log_softmax` → `stable_logsumexp` |
| [debugging-005-01](debugging-005-01/README.md) NaN诊断：逐渐发散到NaN | `optimizer.lr` | `1.0` → `0.01` |
| [debugging-006-01](debugging-006-01/README.md) torch.autograd.detect_anomaly() | `validation.autograd_anomaly` | `False` → `True` |
| [debugging-007-01](debugging-007-01/README.md) 梯度裁剪（临时修复） | `training.clip_norm` | `0.0` → `1.0` |
| [debugging-008-01](debugging-008-01/README.md) 过拟合诊断 | `model.dropout` | `0.0` → `0.5` |
| [debugging-009-01](debugging-009-01/README.md) 欠拟合诊断 | `model.width` | `32` → `64` |
| [debugging-009-02](debugging-009-02/README.md) 欠拟合诊断 | `model.depth` | `2` → `4` |
| [debugging-010-01](debugging-010-01/README.md) 训练不收敛：检查数据流水线 | `validation.data_alignment` | `unchecked` → `sample_id_checked` |
| [debugging-011-01](debugging-011-01/README.md) 训练不收敛：检查优化器参数 | `validation.optimizer_coverage` | `unchecked` → `all_trainable_checked` |
| [debugging-012-01](debugging-012-01/README.md) 损失函数排查 | `loss.numeric_implementation` | `naive_log_softmax` → `stable_logsumexp` |
| [debugging-013-01](debugging-013-01/README.md) 先调学习率 | `optimizer.lr` | `1e-06` → `0.001` |
| [debugging-014-01](debugging-014-01/README.md) 学习率预热扫描（LR Range Test） | `search.lr_policy` | `default` → `range_test` |
| [debugging-015-01](debugging-015-01/README.md) 可视化调试：Layer Activations | `tracking.activation_histograms` | `False` → `True` |
| [debugging-016-01](debugging-016-01/README.md) 可视化调试：Layer Weights | `tracking.weight_histograms` | `False` → `True` |
| [debugging-017-01](debugging-017-01/README.md) 可视化调试：Softmax输出分布 | `tracking.confidence_histograms` | `False` → `True` |
| [debugging-018-01](debugging-018-01/README.md) 对比训练集和验证集的Loss | `tracking.diagnostics` | `train_only` → `train_and_val` |
| [debugging-019-01](debugging-019-01/README.md) 不要怀疑NN的能力 | `model.width` | `32` → `128` |
| [debugging-020-01](debugging-020-01/README.md) 不要盲目信任论文默认参数 | `search.lr_policy` | `default` → `validation_search` |
| [debugging-021-01](debugging-021-01/README.md) 一次只改一个变量 | `search.policy` | `multiple_changes` → `one_factor` |
| [debugging-022-01](debugging-022-01/README.md) 记录一切 | `tracking.level` | `metrics_only` → `full_provenance` |
| [debugging-023-01](debugging-023-01/README.md) 先诊断再下手（消融思维） | `search.policy` | `random_changes` → `diagnose_then_ablate` |
| [debugging-024-01](debugging-024-01/README.md) 连续调参无提升时去看数据和错误样例 | `data.label_filter` | `none` → `error_audited` |
| [debugging-025-01](debugging-025-01/README.md) 确认模型加载了预训练权重 | `validation.load_check` | `unchecked` → `keys_shapes_hash` |
| [debugging-026-01](debugging-026-01/README.md) Debug流程：从简单到复杂 | `validation.preflight` | `none` → `tiny_overfit` |
| [debugging-027-01](debugging-027-01/README.md) 调试时要悲观主义 | `validation.fault_injection` | `none` → `known_fault_suite` |
| [debugging-028-01](debugging-028-01/README.md) 观察Loss胜于观察准确率 | `checkpoint.selection_metric` | `val_accuracy` → `val_loss` |
| [debugging-029-01](debugging-029-01/README.md) 确认训练/推理预处理一致性 | `data.preprocess_contract` | `mismatched` → `identical` |
| [debugging-030-01](debugging-030-01/README.md) 梯度检查与裁剪 | `training.clip_norm` | `0.0` → `1.0` |
| [debugging-030-02](debugging-030-02/README.md) 梯度检查与裁剪 | `validation.gradcheck` | `False` → `True` |
| [debugging-031-01](debugging-031-01/README.md) 早停策略 | `training.patience` | `0` → `5` |
| [debugging-032-01](debugging-032-01/README.md) 可视化Embedding（t-SNE） | `visualization.embedding` | `pca` → `pca_then_tsne` |
| [debugging-033-01](debugging-033-01/README.md) 数据泄露排查 | `data.split_policy` | `random` → `group` |
| [debugging-034-01](debugging-034-01/README.md) 验证集被反复调穿 | `search.holdout_policy` | `reuse_test` → `locked_test` |
| [debugging-035-01](debugging-035-01/README.md) 梯度累积与显存优化 | `training.accumulation_steps` | `1` → `4` |
