# data-eval 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [data-eval-001-01](data-eval-001-01/README.md) 先在小数据上过拟合，确认代码无误，再上全量 | `search.policy` | `full_data_first` → `tiny_overfit_first` |
| [data-eval-002-01](data-eval-002-01/README.md) 数据质量优先于模型调参 | `data.label_filter` | `none` → `audited_clean` |
| [data-eval-003-01](data-eval-003-01/README.md) 训练集分布必须覆盖测试集分布 | `data.domain_sampling` | `source_only` → `target_coverage` |
| [data-eval-004-01](data-eval-004-01/README.md) 交叉验证方式必须结合任务设计，时序数据防未来泄漏 | `data.split_policy` | `random` → `time` |
| [data-eval-005-01](data-eval-005-01/README.md) 训练/验证/测试集切分方式决定实验可信度 | `data.split_policy` | `random` → `group` |
| [data-eval-005-02](data-eval-005-02/README.md) 训练/验证/测试集切分方式决定实验可信度 | `data.split_policy` | `random` → `time` |
| [data-eval-006-01](data-eval-006-01/README.md) 验证集被反复调穿——保留最终测试集 | `search.holdout_policy` | `reuse_test` → `locked_test` |
| [data-eval-007-01](data-eval-007-01/README.md) 观察loss胜于观察准确率 | `checkpoint.selection_metric` | `val_accuracy` → `val_loss` |
| [data-eval-008-01](data-eval-008-01/README.md) 不要只看loss，要看任务相关指标 | `checkpoint.selection_metric` | `val_loss` → `val_f1` |
| [data-eval-009-01](data-eval-009-01/README.md) 离线AUC提升不一定等于线上效果提升 | `data.split_policy` | `random` → `time` |
| [data-eval-010-01](data-eval-010-01/README.md) 对比训练集和验证集的loss判断过拟合 | `tracking.diagnostics` | `train_only` → `train_and_val` |
| [data-eval-011-01](data-eval-011-01/README.md) 多任务学习中的loss平衡 | `loss.task_weighting` | `unscaled` → `train_scale_equalized` |
| [data-eval-012-01](data-eval-012-01/README.md) 回归任务的输出也需要normalization | `data.target_normalization` | `False` → `True` |
| [data-eval-013-01](data-eval-013-01/README.md) 数据预处理：zero-center与shuffle是基本功 | `data.standardize` | `False` → `True` |
| [data-eval-013-02](data-eval-013-02/README.md) 数据预处理：zero-center与shuffle是基本功 | `data.shuffle` | `False` → `True` |
| [data-eval-014-01](data-eval-014-01/README.md) 数据增强：CV用裁剪翻转旋转，NLP用同义词替换和回译 | `data.augmentation` | `none` → `horizontal_flip` |
| [data-eval-014-02](data-eval-014-02/README.md) 数据增强：CV用裁剪翻转旋转，NLP用同义词替换和回译 | `data.augmentation` | `none` → `rotation` |
| [data-eval-014-03](data-eval-014-03/README.md) 数据增强：CV用裁剪翻转旋转，NLP用同义词替换和回译 | `data.augmentation` | `none` → `synonym_replace` |
| [data-eval-014-04](data-eval-014-04/README.md) 数据增强：CV用裁剪翻转旋转，NLP用同义词替换和回译 | `data.augmentation` | `none` → `back_translation` |
| [data-eval-015-01](data-eval-015-01/README.md) 类别不平衡：Focal Loss与类别加权 | `loss.kind` | `cross_entropy` → `focal` |
| [data-eval-015-02](data-eval-015-02/README.md) 类别不平衡：Focal Loss与类别加权 | `loss.class_weights` | `none` → `inverse_frequency` |
| [data-eval-016-01](data-eval-016-01/README.md) Hard Negative Mining提升召回模型效果 | `data.negative_source` | `global_random` → `hard_mined` |
| [data-eval-017-01](data-eval-017-01/README.md) 数据也是超参数——采样比、增强强度、混合比例 | `data.class_sampling` | `natural` → `balanced` |
| [data-eval-017-02](data-eval-017-02/README.md) 数据也是超参数——采样比、增强强度、混合比例 | `data.augmentation_strength` | `0.1` → `0.3` |
| [data-eval-017-03](data-eval-017-03/README.md) 数据也是超参数——采样比、增强强度、混合比例 | `data.multiturn_fraction` | `0.1` → `0.5` |
| [data-eval-018-01](data-eval-018-01/README.md) 标签噪声检测与数据清洗 | `data.label_filter` | `none` → `audited_clean` |
| [data-eval-019-01](data-eval-019-01/README.md) Embedding可视化：先PCA降维再用t-SNE | `visualization.embedding` | `tsne_raw` → `pca_then_tsne` |
| [data-eval-020-01](data-eval-020-01/README.md) 数据版本管理（DVC） | `tracking.dataset_version` | `path_only` → `content_hash` |
| [data-eval-021-01](data-eval-021-01/README.md) 大模型评测：人工Eval Set + OpenCompass + RAGAS | `evaluation.protocol` | `loss_only` → `frozen_task_suite` |
| [data-eval-022-01](data-eval-022-01/README.md) 数据风格统一性决定模型是否学偏 | `data.answer_style` | `mixed` → `canonical` |
| [data-eval-023-01](data-eval-023-01/README.md) LLM模型越强，数据质量越重要 | `data.label_filter` | `none` → `audited_clean` |
| [data-eval-024-01](data-eval-024-01/README.md) 数据分布偏移导致领域适配问题 | `data.domain_sampling` | `source_only` → `target_coverage` |
| [data-eval-025-01](data-eval-025-01/README.md) 错误分析比盲调参数更有价值 | `search.policy` | `random_changes` → `error_slice_first` |
| [data-eval-026-01](data-eval-026-01/README.md) 随机种子必须固定，否则对比实验不可信 | `training.seed_policy` | `independent_unlogged` → `paired_logged` |
| [data-eval-027-01](data-eval-027-01/README.md) 标签平滑（Label Smoothing） | `loss.label_smoothing` | `0.0` → `0.1` |
| [data-eval-028-01](data-eval-028-01/README.md) 超参数间接依赖测试集导致的偏差 | `search.holdout_policy` | `reuse_test` → `locked_test` |
| [data-eval-029-01](data-eval-029-01/README.md) 确认分类网络学习充分——观察Softmax输出分布 | `tracking.diagnostics` | `accuracy_only` → `confidence_and_calibration` |
| [data-eval-030-01](data-eval-030-01/README.md) 训练集和验证集存在泄漏时的处理 | `data.deduplicate` | `none` → `cross_split` |
