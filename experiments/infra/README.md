# infra 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [infra-001-01](infra-001-01/README.md) 混合精度训练（FP16 / BF16） | `training.precision` | `fp32` → `bf16` |
| [infra-001-02](infra-001-02/README.md) 混合精度训练（FP16 / BF16） | `training.precision` | `fp32` → `fp16` |
| [infra-001-03](infra-001-03/README.md) 混合精度训练（FP16 / BF16） | `training.loss_scaling` | `none` → `dynamic` |
| [infra-002-01](infra-002-01/README.md) 分布式训练 | `training.world_size` | `1` → `2` |
| [infra-002-02](infra-002-02/README.md) 分布式训练 | `training.sharding` | `ddp` → `zero2` |
| [infra-002-03](infra-002-03/README.md) 分布式训练 | `training.sharding` | `zero2` → `zero3` |
| [infra-003-01](infra-003-01/README.md) 梯度检查点（Gradient Checkpointing） | `training.gradient_checkpointing` | `False` → `True` |
| [infra-004-01](infra-004-01/README.md) W&B Sweeps 超参搜索 | `search.sampler` | `grid` → `random` |
| [infra-004-02](infra-004-02/README.md) W&B Sweeps 超参搜索 | `search.sampler` | `random` → `bayesian` |
| [infra-005-01](infra-005-01/README.md) 实验追踪与配置管理 | `tracking.level` | `metrics_only` → `full_provenance` |
| [infra-006-01](infra-006-01/README.md) Code Pipeline 模块化架构 | `engineering.pipeline` | `monolithic` → `modular` |
| [infra-007-01](infra-007-01/README.md) Runner 类训练框架 | `engineering.runner` | `inline` → `runner` |
| [infra-008-01](infra-008-01/README.md) 分桶策略（Bucketing） | `data.batch_order` | `random` → `length_bucket` |
| [infra-009-01](infra-009-01/README.md) GPU 内存优化：num_workers 与 pin_memory | `data.num_workers` | `0` → `4` |
| [infra-009-02](infra-009-02/README.md) GPU 内存优化：num_workers 与 pin_memory | `data.pin_memory` | `False` → `True` |
| [infra-010-01](infra-010-01/README.md) torch.tensor() vs torch.as_tensor() vs torch.from_numpy() | `data.tensor_conversion` | `tensor_copy` → `from_numpy_shared` |
| [infra-010-02](infra-010-02/README.md) torch.tensor() vs torch.as_tensor() vs torch.from_numpy() | `data.tensor_conversion` | `tensor_copy` → `as_tensor_shared` |
| [infra-011-01](infra-011-01/README.md) 随机种子设定 | `training.seed_policy` | `independent_unlogged` → `paired_logged` |
| [infra-012-01](infra-012-01/README.md) 梯度累积（Gradient Accumulation） | `training.accumulation_steps` | `1` → `4` |
| [infra-013-01](infra-013-01/README.md) 模型 Checkpoint 保存与加载 | `checkpoint.restore` | `weights_only` → `full_training_state` |
| [infra-013-02](infra-013-02/README.md) 模型 Checkpoint 保存与加载 | `checkpoint.selection_metric` | `val_last` → `val_loss` |
| [infra-014-01](infra-014-01/README.md) 超参数搜索工具选型 | `search.sampler` | `grid` → `bayesian` |
| [infra-015-01](infra-015-01/README.md) 计算最优训练：学习率与 Batch Size 的 Scaling Law | `search.scaling_policy` | `fixed` → `deepseek_compute_law` |
| [infra-016-01](infra-016-01/README.md) 小样本"过拟合"验证法 | `search.policy` | `full_data_first` → `tiny_overfit_first` |
| [infra-017-01](infra-017-01/README.md) Warmup + Cosine 退火学习率调度 | `optimizer.schedule` | `constant` → `warmup_cosine` |
