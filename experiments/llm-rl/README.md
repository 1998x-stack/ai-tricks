# llm-rl 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [llm-rl-001-01](llm-rl-001-01/README.md) SFT是RL的前提，RL不能弥补SFT的不足 | `checkpoint.policy_start` | `base` → `sft` |
| [llm-rl-002-01](llm-rl-002-01/README.md) 关注reward而非loss | `checkpoint.selection_metric` | `val_loss` → `val_reward` |
| [llm-rl-003-01](llm-rl-003-01/README.md) 确定规则reward有效，主观打分reward效果有限 | `reward.kind` | `subjective_judge` → `verifiable_rule` |
| [llm-rl-004-01](llm-rl-004-01/README.md) 纯RL可在特定场景跳过SFT | `checkpoint.policy_start` | `sft` → `base` |
| [llm-rl-005-01](llm-rl-005-01/README.md) 多步checkpoint检验 | `checkpoint.eval_every_steps` | `500` → `50` |
| [llm-rl-006-01](llm-rl-006-01/README.md) SFT基线能力门槛 | `checkpoint.base_success_stratum` | `below_20pct` → `above_20pct` |
| [llm-rl-007-01](llm-rl-007-01/README.md) 奖励设计：规则完整性 | `reward.validation` | `format_only` → `format_and_correctness` |
| [llm-rl-008-01](llm-rl-008-01/README.md) Reward Hacking的解决方案 | `reward.hack_guard` | `none` → `heldout_adversarial_tests` |
| [llm-rl-008-02](llm-rl-008-02/README.md) Reward Hacking的解决方案 | `reward.judge_policy` | `single` → `ensemble` |
| [llm-rl-008-03](llm-rl-008-03/README.md) Reward Hacking的解决方案 | `reward.length_cap` | `0` → `1024` |
| [llm-rl-009-01](llm-rl-009-01/README.md) KL散度：从小的值开始 | `rl.kl_beta` | `0.1` → `0.01` |
| [llm-rl-010-01](llm-rl-010-01/README.md) KL爆炸：增加惩罚权重 | `rl.kl_beta` | `0.01` → `0.1` |
| [llm-rl-011-01](llm-rl-011-01/README.md) KL过低导致reward不涨 | `rl.kl_beta` | `1.0` → `0.01` |
| [llm-rl-012-01](llm-rl-012-01/README.md) 基模不做SFT时可不开KL | `rl.kl_beta` | `0.01` → `0.0` |
| [llm-rl-013-01](llm-rl-013-01/README.md) PPO学习率：SFT的十分之一 | `optimizer.lr` | `2e-05` → `2e-06` |
| [llm-rl-014-01](llm-rl-014-01/README.md) 混入SFT数据防止遗忘 | `rl.sft_mix` | `0.0` → `0.1` |
| [llm-rl-015-01](llm-rl-015-01/README.md) Reward Model输出归一化 | `reward.normalize` | `False` → `True` |
| [llm-rl-016-01](llm-rl-016-01/README.md) Batch size宁大勿小 | `training.rollout_batch` | `32` → `256` |
| [llm-rl-017-01](llm-rl-017-01/README.md) 梯度裁剪值0.2 | `training.clip_norm` | `1.0` → `0.2` |
| [llm-rl-018-01](llm-rl-018-01/README.md) 余弦LR调度 | `optimizer.schedule` | `constant` → `cosine` |
| [llm-rl-019-01](llm-rl-019-01/README.md) 高LR导致Mode Collapse | `optimizer.lr` | `0.0001` → `1e-06` |
| [llm-rl-020-01](llm-rl-020-01/README.md) 长度偏差修正 | `reward.length_normalize` | `False` → `True` |
| [llm-rl-021-01](llm-rl-021-01/README.md) 奖励拆分多维度 | `reward.components` | `['correctness']` → `['correctness', 'format', 'safety']` |
| [llm-rl-021-02](llm-rl-021-02/README.md) 奖励拆分多维度 | `reward.component_weighting` | `unscaled` → `normalized` |
| [llm-rl-022-01](llm-rl-022-01/README.md) Critic Value Loss波动剧烈 | `rl.value_clip` | `0.0` → `0.2` |
| [llm-rl-022-02](llm-rl-022-02/README.md) Critic Value Loss波动剧烈 | `rl.value_lr` | `1e-05` → `1e-06` |
| [llm-rl-023-01](llm-rl-023-01/README.md) 熵系数过低与模式坍塌 | `rl.entropy_coef` | `0.0` → `0.01` |
| [llm-rl-024-01](llm-rl-024-01/README.md) Base-RL + 拒绝采样 + 冷启动微调 | `checkpoint.policy_start` | `base` → `rejection_sample_sft` |
| [llm-rl-025-01](llm-rl-025-01/README.md) reward不涨：检查基模能力上限 | `checkpoint.policy_start` | `weak_base` → `strong_base` |
| [llm-rl-026-01](llm-rl-026-01/README.md) Beta参数过高：chosen/rejected概率差增长慢 | `rl.dpo_beta` | `0.5` → `0.1` |
| [llm-rl-027-01](llm-rl-027-01/README.md) Beta过低或LR过高：损失下降快但生成效果差 | `rl.dpo_beta` | `0.01` → `0.1` |
| [llm-rl-027-02](llm-rl-027-02/README.md) Beta过低或LR过高：损失下降快但生成效果差 | `optimizer.lr` | `0.0001` → `1e-05` |
| [llm-rl-028-01](llm-rl-028-01/README.md) Chosen-Rejected对质量 | `data.preference_filter` | `unfiltered` → `audited_pairs` |
| [llm-rl-029-01](llm-rl-029-01/README.md) DPO不是SFT的补救 | `checkpoint.policy_start` | `weak_sft` → `strong_sft` |
| [llm-rl-030-01](llm-rl-030-01/README.md) Batch size越大越稳定 | `training.prompt_batch` | `8` → `64` |
| [llm-rl-031-01](llm-rl-031-01/README.md) 拒绝采样 + 冷启动微调提升Base-RL | `checkpoint.policy_start` | `base` → `rejection_sample_sft` |
