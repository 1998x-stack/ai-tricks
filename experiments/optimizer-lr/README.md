# optimizer-lr 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [optimizer-lr-001-01](optimizer-lr-001-01/README.md) 优化器优先选Adam | `optimizer.kind` | `sgd` → `adam` |
| [optimizer-lr-002-01](optimizer-lr-002-01/README.md) 选Adam还是SGD看任务性质 | `optimizer.kind` | `sgd` → `adam` |
| [optimizer-lr-003-01](optimizer-lr-003-01/README.md) SGD+Momentum最终效果常优于Adam | `optimizer.kind` | `adam` → `sgd` |
| [optimizer-lr-004-01](optimizer-lr-004-01/README.md) 两阶段训练法：Adam过渡到SGD | `optimizer.kind` | `adam` → `adam_then_sgd` |
| [optimizer-lr-005-01](optimizer-lr-005-01/README.md) AdamW替代Adam，权重衰减更合理 | `optimizer.kind` | `adam` → `adamw` |
| [optimizer-lr-006-01](optimizer-lr-006-01/README.md) 数据稀疏用Adagrad，稠密用Adam | `optimizer.sparse_kind` | `adam` → `adagrad` |
| [optimizer-lr-006-02](optimizer-lr-006-02/README.md) 数据稀疏用Adagrad，稠密用Adam | `optimizer.dense_kind` | `sgd` → `adam` |
| [optimizer-lr-007-01](optimizer-lr-007-01/README.md) NLP通用LR范围 | `optimizer.lr` | `0.001` → `2e-05` |
| [optimizer-lr-008-01](optimizer-lr-008-01/README.md) CV通用LR范围 | `optimizer.lr` | `1e-05` → `0.001` |
| [optimizer-lr-009-01](optimizer-lr-009-01/README.md) 初始LR二分搜索法 | `search.lr_policy` | `grid` → `log_bisection` |
| [optimizer-lr-010-01](optimizer-lr-010-01/README.md) LR范围的土办法：预热扫描 | `search.lr_policy` | `default` → `range_test` |
| [optimizer-lr-011-01](optimizer-lr-011-01/README.md) LR Range Test | `search.lr_policy` | `default` → `range_test` |
| [optimizer-lr-012-01](optimizer-lr-012-01/README.md) Cyclical LR的范围测试法 | `search.lr_policy` | `default` → `range_test` |
| [optimizer-lr-013-01](optimizer-lr-013-01/README.md) Cosine退火+warmup万能模板 | `optimizer.schedule` | `constant` → `warmup_cosine` |
| [optimizer-lr-014-01](optimizer-lr-014-01/README.md) Warmup步数设置原则 | `optimizer.warmup_ratio` | `0.03` → `0.1` |
| [optimizer-lr-015-01](optimizer-lr-015-01/README.md) 阶梯下降（Step Decay）精调法 | `optimizer.schedule` | `constant` → `step` |
| [optimizer-lr-016-01](optimizer-lr-016-01/README.md) Multistep LR衰减 + 余弦 | `optimizer.schedule` | `multistep` → `cosine` |
| [optimizer-lr-017-01](optimizer-lr-017-01/README.md) Cyclical LR周期性重启 | `optimizer.schedule` | `constant` → `cyclic` |
| [optimizer-lr-018-01](optimizer-lr-018-01/README.md) LR按验证集瓶颈手动折半 | `optimizer.schedule` | `constant` → `plateau_half` |
| [optimizer-lr-019-01](optimizer-lr-019-01/README.md) 指数衰减LR | `optimizer.schedule` | `constant` → `exponential` |
| [optimizer-lr-020-01](optimizer-lr-020-01/README.md) 衰减系数参考值 | `optimizer.gamma` | `0.9` → `0.99` |
| [optimizer-lr-021-01](optimizer-lr-021-01/README.md) 初始LR的阶梯搜索法 | `search.lr_policy` | `linear_grid` → `log_grid` |
| [optimizer-lr-022-01](optimizer-lr-022-01/README.md) ReduceLROnPlateau + EarlyStopping | `optimizer.schedule` | `constant` → `plateau_half` |
| [optimizer-lr-022-02](optimizer-lr-022-02/README.md) ReduceLROnPlateau + EarlyStopping | `training.patience` | `0` → `5` |
| [optimizer-lr-023-01](optimizer-lr-023-01/README.md) LR过高的症状与对策 | `optimizer.lr` | `1.0` → `0.01` |
| [optimizer-lr-024-01](optimizer-lr-024-01/README.md) LR过低的症状 | `optimizer.lr` | `1e-06` → `0.001` |
| [optimizer-lr-025-01](optimizer-lr-025-01/README.md) 需要降低LR的标志 | `optimizer.schedule` | `constant` → `plateau_half` |
| [optimizer-lr-026-01](optimizer-lr-026-01/README.md) LR先小后大再衰减策略 | `optimizer.schedule` | `constant` → `warmup_cosine` |
| [optimizer-lr-027-01](optimizer-lr-027-01/README.md) Batch Size与LR的线性缩放关系 | `optimizer.lr` | `0.1` → `0.4` |
| [optimizer-lr-028-01](optimizer-lr-028-01/README.md) 显存不足时梯度累积等效增大batch | `training.accumulation_steps` | `1` → `4` |
| [optimizer-lr-029-01](optimizer-lr-029-01/README.md) Momentum通用设置范围 | `optimizer.momentum` | `0.0` → `0.9` |
| [optimizer-lr-030-01](optimizer-lr-030-01/README.md) 循环动量（Cyclical Momentum） | `optimizer.momentum_schedule` | `constant` → `cyclic` |
| [optimizer-lr-031-01](optimizer-lr-031-01/README.md) Weight Decay默认值及调参 | `optimizer.weight_decay` | `0.0` → `0.01` |
| [optimizer-lr-032-01](optimizer-lr-032-01/README.md) 梯度裁剪防止NaN | `training.clip_norm` | `0.0` → `1.0` |
| [optimizer-lr-033-01](optimizer-lr-033-01/README.md) PPO训练LR比SFT小一个数量级 | `optimizer.lr` | `2e-05` → `2e-06` |
| [optimizer-lr-034-01](optimizer-lr-034-01/README.md) DPO中LR与Beta的配合 | `rl.dpo_beta` | `0.01` → `0.1` |
| [optimizer-lr-034-02](optimizer-lr-034-02/README.md) DPO中LR与Beta的配合 | `optimizer.lr` | `0.0001` → `1e-05` |
| [optimizer-lr-035-01](optimizer-lr-035-01/README.md) 大模型SFT初始LR与weight decay | `optimizer.lr` | `0.0001` → `5e-05` |
| [optimizer-lr-035-02](optimizer-lr-035-02/README.md) 大模型SFT初始LR与weight decay | `optimizer.weight_decay` | `0.25` → `0.1` |
| [optimizer-lr-036-01](optimizer-lr-036-01/README.md) 大模型预训练的Scaling Law | `search.scaling_policy` | `fixed` → `deepseek_compute_law` |
| [optimizer-lr-037-01](optimizer-lr-037-01/README.md) DeepSeek固定超参最优值 | `optimizer.beta2` | `0.999` → `0.95` |
| [optimizer-lr-037-02](optimizer-lr-037-02/README.md) DeepSeek固定超参最优值 | `optimizer.beta1` | `0.85` → `0.9` |
| [optimizer-lr-037-03](optimizer-lr-037-03/README.md) DeepSeek固定超参最优值 | `optimizer.weight_decay` | `0.01` → `0.1` |
| [optimizer-lr-037-04](optimizer-lr-037-04/README.md) DeepSeek固定超参最优值 | `training.clip_norm` | `0.0` → `1.0` |
| [optimizer-lr-038-01](optimizer-lr-038-01/README.md) LoRA微调LR范围 | `optimizer.lr` | `0.0001` → `0.0003` |
| [optimizer-lr-039-01](optimizer-lr-039-01/README.md) 不要同时调超过两个超参数 | `search.policy` | `multiple_changes` → `one_factor` |
| [optimizer-lr-040-01](optimizer-lr-040-01/README.md) 先锁定LR再调其他参数 | `search.policy` | `joint_random` → `lr_then_batch` |
| [optimizer-lr-041-01](optimizer-lr-041-01/README.md) 优先调LR再谈其他 | `search.policy` | `architecture_first` → `lr_first` |
| [optimizer-lr-042-01](optimizer-lr-042-01/README.md) LR是调参的第一优先级，其次是数据 | `search.policy` | `architecture_first` → `data_lr_first` |
| [optimizer-lr-043-01](optimizer-lr-043-01/README.md) 最简可用配置 | `training.baseline_recipe` | `project_default` → `sgd_0.1_m0.9_wd0.005` |
| [optimizer-lr-044-01](optimizer-lr-044-01/README.md) 初始LR：0.1搭配SGD+Momentum | `optimizer.lr` | `0.001` → `0.1` |
| [optimizer-lr-045-01](optimizer-lr-045-01/README.md) LR太大连ReLU神经元都能弄死 | `optimizer.lr` | `0.1` → `0.01` |
| [optimizer-lr-046-01](optimizer-lr-046-01/README.md) LR过小 vs LR进入平台期 的可视化区别 | `tracking.diagnostics` | `loss_only` → `loss_lr_gradient` |
| [optimizer-lr-047-01](optimizer-lr-047-01/README.md) SFT后loss下降但效果不好的诊断 | `checkpoint.selection_metric` | `val_loss` → `val_exact_match` |
| [optimizer-lr-048-01](optimizer-lr-048-01/README.md) 训练初期不要加激活函数和BN | `model.activation` | `relu` → `identity` |
| [optimizer-lr-049-01](optimizer-lr-049-01/README.md) 确认随机种子固定 | `training.seed_policy` | `independent_unlogged` → `paired_logged` |
| [optimizer-lr-050-01](optimizer-lr-050-01/README.md) 不要过早Early Stopping | `training.patience` | `2` → `10` |
| [optimizer-lr-051-01](optimizer-lr-051-01/README.md) 先过拟合再收敛 | `search.policy` | `regularization_first` → `tiny_overfit_first` |
| [optimizer-lr-052-01](optimizer-lr-052-01/README.md) 参数初始化与LR配合 | `model.weight_init` | `default` → `kaiming_normal` |
| [optimizer-lr-053-01](optimizer-lr-053-01/README.md) 对照训练/验证loss曲线调LR | `optimizer.schedule` | `constant` → `plateau_half` |
| [optimizer-lr-054-01](optimizer-lr-054-01/README.md) 最优配置：SGD + 0.1lr + 0.9momentum + weight decay 0.005 | `training.baseline_recipe` | `project_default` → `sgd_0.1_m0.9_wd0.005` |
| [optimizer-lr-055-01](optimizer-lr-055-01/README.md) 正确初始化比调参更重要 | `model.weight_init` | `default` → `xavier_uniform` |
| [optimizer-lr-056-01](optimizer-lr-056-01/README.md) 小数据微调：dropout和weight decay不宜过大 | `model.dropout` | `0.5` → `0.1` |
| [optimizer-lr-056-02](optimizer-lr-056-02/README.md) 小数据微调：dropout和weight decay不宜过大 | `optimizer.weight_decay` | `0.1` → `0.001` |
| [optimizer-lr-057-01](optimizer-lr-057-01/README.md) LR对Transformer的重要性 | `optimizer.lr` | `0.001` → `0.0001` |
| [optimizer-lr-058-01](optimizer-lr-058-01/README.md) 余弦退火的实现代码 | `optimizer.schedule` | `constant` → `cosine` |
| [optimizer-lr-059-01](optimizer-lr-059-01/README.md) Cyclic LR是比固定衰减更好的策略 | `optimizer.schedule` | `step` → `cyclic` |
| [optimizer-lr-060-01](optimizer-lr-060-01/README.md) 权重衰减与数据集规模的关系 | `optimizer.weight_decay` | `0.0001` → `0.01` |
| [optimizer-lr-061-01](optimizer-lr-061-01/README.md) LR太大时的泛化正则化效应 | `optimizer.lr` | `0.0001` → `0.001` |
| [optimizer-lr-062-01](optimizer-lr-062-01/README.md) 恒等映射与LR无关但影响优化 | `model.residual` | `False` → `True` |
| [optimizer-lr-063-01](optimizer-lr-063-01/README.md) BN层对LR的加速效应 | `model.norm` | `none` → `batch` |
| [optimizer-lr-064-01](optimizer-lr-064-01/README.md) 先训大模型再裁剪优于直接训小模型 | `training.compression_policy` | `small_from_scratch` → `large_then_prune_finetune` |
| [optimizer-lr-065-01](optimizer-lr-065-01/README.md) LR要随迭代次数2倍递减 | `optimizer.gamma` | `0.9` → `0.5` |
| [optimizer-lr-066-01](optimizer-lr-066-01/README.md) 推荐系统常用batch size与LR | `training.batch_size` | `32` → `128` |
| [optimizer-lr-067-01](optimizer-lr-067-01/README.md) Warm-up + Cosine Decay的精调选项 | `optimizer.warmup_ratio` | `0.03` → `0.1` |
| [optimizer-lr-067-02](optimizer-lr-067-02/README.md) Warm-up + Cosine Decay的精调选项 | `optimizer.min_lr` | `0.0` → `1e-07` |
| [optimizer-lr-068-01](optimizer-lr-068-01/README.md) 仅调LR和batch size：大模型Scaling Law | `search.scaling_policy` | `fixed` → `deepseek_compute_law` |
| [optimizer-lr-069-01](optimizer-lr-069-01/README.md) 微调预训练模型时需低LR | `optimizer.lr` | `0.001` → `2e-05` |
| [optimizer-lr-070-01](optimizer-lr-070-01/README.md) 训练与推理的LR保持一致原则 | `data.preprocess_contract` | `mismatched` → `identical` |
| [optimizer-lr-071-01](optimizer-lr-071-01/README.md) 复杂任务先小LR保不爆 | `optimizer.lr` | `0.001` → `0.0001` |
| [optimizer-lr-072-01](optimizer-lr-072-01/README.md) PPO RLHF完整LR策略 | `optimizer.schedule` | `constant` → `warmup_cosine` |
| [optimizer-lr-073-01](optimizer-lr-073-01/README.md) 模型训练初期输出重复/无意义 -> LR太高 | `optimizer.lr` | `0.0001` → `1e-06` |
| [optimizer-lr-074-01](optimizer-lr-074-01/README.md) 梯度检查与裁剪的实践 | `training.clip_norm` | `0.0` → `1.0` |
| [optimizer-lr-075-01](optimizer-lr-075-01/README.md) Adam收敛快但可能泛化差 | `optimizer.kind` | `adam` → `sgd` |
| [optimizer-lr-076-01](optimizer-lr-076-01/README.md) Adam默认LR推荐3e-4 | `optimizer.lr` | `0.001` → `0.0003` |
| [optimizer-lr-077-01](optimizer-lr-077-01/README.md) 初始LR推荐3.5e-4 | `optimizer.lr` | `0.001` → `0.00035` |
| [optimizer-lr-078-01](optimizer-lr-078-01/README.md) 学习率热启动（warm restart） | `optimizer.schedule` | `cosine` → `cosine_restarts` |
| [optimizer-lr-079-01](optimizer-lr-079-01/README.md) 超参优化方法对比 | `search.sampler` | `grid` → `random` |
| [optimizer-lr-080-01](optimizer-lr-080-01/README.md) 自动调参工具推荐 | `search.sampler` | `grid` → `bayesian` |
| [optimizer-lr-081-01](optimizer-lr-081-01/README.md) 网格搜索精调的实用技巧 | `search.policy` | `wide_grid` → `coarse_then_fine_grid` |
| [optimizer-lr-082-01](optimizer-lr-082-01/README.md) 由粗到细随机搜索推荐 | `search.policy` | `wide_random` → `coarse_then_fine_random` |
| [optimizer-lr-083-01](optimizer-lr-083-01/README.md) 无脑用SGD+Momentum | `optimizer.kind` | `adam` → `sgd` |
| [optimizer-lr-084-01](optimizer-lr-084-01/README.md) 预热扫描的实用操作 | `search.lr_policy` | `default` → `range_test` |
| [optimizer-lr-085-01](optimizer-lr-085-01/README.md) LR和Momentum的组合搜索 | `search.policy` | `lr_only` → `lr_momentum_joint` |
| [optimizer-lr-085-02](optimizer-lr-085-02/README.md) LR和Momentum的组合搜索 | `optimizer.momentum` | `0.85` → `0.95` |
| [optimizer-lr-086-01](optimizer-lr-086-01/README.md) SFT + RL 两阶段LR策略 | `optimizer.rl_lr_ratio` | `1.0` → `0.1` |
| [optimizer-lr-087-01](optimizer-lr-087-01/README.md) 模型未充分训练时LR不够的问题 | `training.epochs` | `5` → `20` |
| [optimizer-lr-088-01](optimizer-lr-088-01/README.md) 梯度累积做等效大批次的LR策略 | `training.accumulation_steps` | `1` → `4` |
| [optimizer-lr-089-01](optimizer-lr-089-01/README.md) 逐特征加特征排查LR效果 | `data.feature_policy` | `all_at_once` → `forward_selection` |
