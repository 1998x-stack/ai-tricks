# batch-size 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [batch-size-001-01](batch-size-001-01/README.md) batch size 在表示学习/对比学习领域越大越好 | `training.contrastive_batch` | `32` → `256` |
| [batch-size-001-02](batch-size-001-02/README.md) batch size 在表示学习/对比学习领域越大越好 | `training.contrastive_negative_pool` | `microbatch` → `global_batch` |
| [batch-size-002-01](batch-size-002-01/README.md) batch size 并非越大越好——小批次有时有奇效 | `training.batch_size` | `256` → `32` |
| [batch-size-003-01](batch-size-003-01/README.md) PPO / RLHF 阶段 batch size 宁大勿小 | `training.rollout_batch` | `32` → `256` |
| [batch-size-004-01](batch-size-004-01/README.md) GRPO 训练 batch size 越大效果越稳定 | `training.prompt_batch` | `8` → `64` |
| [batch-size-005-01](batch-size-005-01/README.md) 推荐系统 batch size 最佳范围 32-512 | `training.batch_size` | `32` → `512` |
| [batch-size-006-01](batch-size-006-01/README.md) batch size 与学习率的线性缩放关系 | `optimizer.lr` | `0.1` → `0.4` |
| [batch-size-006-02](batch-size-006-02/README.md) batch size 与学习率的线性缩放关系 | `optimizer.lr` | `0.1` → `0.2` |
| [batch-size-007-01](batch-size-007-01/README.md) 大 batch 训练快但泛化不一定更好，需记录 effective batch size | `tracking.batch_metadata` | `nominal_only` → `micro_world_accum_effective` |
| [batch-size-008-01](batch-size-008-01/README.md) 批量大小受硬件内存限制，多 GPU 总批量为单卡批量乘以 GPU 数 | `training.world_size` | `1` → `2` |
| [batch-size-009-01](batch-size-009-01/README.md) 小批量：梯度噪声正则化 vs 大批量：训练加速 | `training.batch_size` | `32` → `256` |
| [batch-size-010-01](batch-size-010-01/README.md) 小批次显存小但梯度不稳定，大批次显存大但易陷入局部最优 | `training.batch_size` | `16` → `128` |
| [batch-size-011-01](batch-size-011-01/README.md) DeepSeek LLM 批大小与学习率的 Scaling Law | `search.scaling_policy` | `fixed` → `deepseek_compute_law` |
| [batch-size-012-01](batch-size-012-01/README.md) 梯度累积代码实现 | `training.accumulation_steps` | `1` → `4` |
| [batch-size-012-02](batch-size-012-02/README.md) 梯度累积代码实现 | `training.accumulation_normalization` | `unscaled` → `sample_weighted` |
| [batch-size-012-03](batch-size-012-03/README.md) 梯度累积代码实现 | `training.tail_accumulation` | `drop` → `flush_sample_weighted` |
| [batch-size-013-01](batch-size-013-01/README.md) mini batch size 数十到数百皆可，SGD 默认设定 | `training.batch_size` | `32` → `128` |
| [batch-size-014-01](batch-size-014-01/README.md) 开源默认 batch 不一定适合你的数据 | `training.batch_size` | `256` → `64` |
| [batch-size-015-01](batch-size-015-01/README.md) 调参应先锁定学习率，再调 batch size | `search.policy` | `joint_random` → `lr_then_batch` |
| [batch-size-016-01](batch-size-016-01/README.md) batch size 在不同任务中影响不同 | `training.batch_size` | `32` → `128` |
| [batch-size-017-01](batch-size-017-01/README.md) 分桶策略（Bucketing）减少 padding 提高训练效率 | `data.batch_order` | `random` → `length_bucket` |
| [batch-size-018-01](batch-size-018-01/README.md) 推荐系统热启动时 batch size 需注意 | `training.batch_size` | `32` → `128` |
| [batch-size-019-01](batch-size-019-01/README.md) PPO 学习率通常比 SFT 小一个数量级 | `optimizer.lr` | `2e-05` → `2e-06` |
