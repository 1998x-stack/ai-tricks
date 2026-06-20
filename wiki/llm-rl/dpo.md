# Direct Preference Optimization (DPO)

## 是什么

DPO（Direct Preference Optimization）是一种无需显式奖励模型的偏好优化方法。与 PPO-RLHF 不同，DPO 直接在偏好数据上优化策略，绕过了训练奖励模型和 PPO 策略优化的两阶段流程。

### 核心机制

DPO 基于 Bradley-Terry 偏好模型，推导出策略 π_θ 的对数概率与隐式奖励函数 r_θ 之间的关系：

`r_θ(x, y) = β * log(π_θ(y|x) / π_ref(y|x)) + β * log(Z(x))`

其中 β 是控制 KL 约束强度的超参数，π_ref 是参考策略（通常是 SFT 模型），Z(x) 是配分函数。

DPO 的损失函数直接最大化 chosen 响应被偏好（相对 rejected 响应）的对数几率：

`L_DPO = -E[log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x)))]`

其中 y_w 是 chosen 响应，y_l 是 rejected 响应，σ 是 sigmoid 函数。

**直觉**：DPO 通过增大 chosen 响应的概率（相对于 π_ref）并降低 rejected 响应的概率，隐式地学习了一个奖励函数。整个过程不需要单独训练 RM，也不需要 PPO 的 Actor-Critic 架构。

### 与 PPO 的核心差异

| 维度 | PPO-RLHF | DPO |
|------|----------|-----|
| 奖励模型 | 需要训练显式 RM | 隐式奖励，无额外模型 |
| Critic 网络 | 需要 | 不需要 |
| 训练阶段 | RM 训练 + PPO 微调 | 单阶段直接优化 |
| 采样 | On-policy | Off-policy（偏好数据静态） |
| 计算开销 | 4 模型同时推理 | 策略 + 参考 2 模型 |
| 稳定性 | 依赖超参数调优 | 相对稳健 |

## 为什么重要

DPO 最大的价值在于**简化流程**。PPO-RLHF 需要训练 RM、维护 Critic、做 on-policy 采样——工程复杂度高，超参数敏感。DPO 只需已标注的偏好数据集和一次优化，大幅降低了 RLHF 的准入门槛。

### 当 DPO 优于 PPO

1. **RM 质量受限时**：如果偏好数据质量高但难以训练出好的 RM（如数据量小或标注噪声大），DPO 直接优化偏好信号，避免了 RM 作为中间瓶颈的信息损失。
2. **计算资源有限时**：DPO 只需维护两个模型（策略 + 参考），无需 Critic 和 RM，GPU 显存和训练时间显著降低。
3. **离线偏好数据充分时**：DPO 是 off-policy 方法，可直接利用已有的静态偏好数据集，不需要 on-policy 采样。
4. **偏好信号二元明确时**：当 human labelers 对 chosen/rejected 的判断高度一致时，DPO 能有效学习偏好。

### 当 PPO 仍不可替代

- RL 探索至关重要时（如数学推理、编程竞赛），on-policy 采样让模型在搜索空间中不断尝试新路径
- RM 质量很高且需要细粒度连续打分时（非二元偏好）
- 需要多轮迭代 RL（RM 可更新，策略和 RM 共同进化）

## 如何使用

### Beta 参数深度解析

Beta 是 DPO 最重要的超参数，理解它对训练成功至关重要：

- **Beta 的角色**：控制隐式奖励的温度（temperature）。Beta 越大，隐式奖励对策略偏离的惩罚越强，更新越保守；Beta 越小，模型更大胆地拟合偏好数据。
- **Beta 与 KL 约束**：Beta 在 DPO 中扮演类似 PPO 中 KL 惩罚系数的角色。Beta→∞ 时策略完全冻结（不做任何偏好优化）；Beta→0 时策略无约束地拟合偏好数据，极易过拟合。
- **Beta 过高的症状**：Chosen 和 Rejected 的概率差增长缓慢，训练信号弱。需要降低 Beta。
- **Beta 过低的症状**：Loss 快速下降，但生成质量差甚至不如 SFT。Beta 过低导致模型偏离参考策略过远，丢失通用能力。
- **推荐范围**：0.1 - 0.5。从 0.1 开始尝试，监控 chosen/rejected 的 log-prob 差值变化。

### Beta 与 LR 的协同

低 Beta 时配合高 LR 最为危险，两者都会让模型过快地偏离 SFT。出现 Loss 快速下降 + 生成质量差时：

1. 调高 Beta（增加对参考模型的约束）
2. 降低 LR（典型值 1e-7 到 5e-6）
3. 或两者同时调整

### Chosen / Rejected 对的质量

DPO 对偏好数据的质量极其敏感：

- **差异过小**：Chosen 和 Rejected 回答质量接近，DPO 的偏好信号弱，模型学不到有效信息。
- **差异过大**：Model 容易学到表面特征（如"长回答更好"），而非真正的质量差异。
- **最佳实践**：Chosen 和 Rejected 应在同一难度水平上，确保质量差异来自"回答正确/更有帮助"而非"回答风格不同"。
- **数量比例**：单条 prompt 通常配 1 个 chosen + 1 个 rejected，但 N 选 1 偏好（多个回复排序）也可使用，需要注意信号强度的差异。

### 数据准备要点

DPO 对数据的依赖比 PPO 更强，因为没有 on-policy 采样来校正分布：

- **采样分布**：偏好数据最好来自 SFT 模型自身的采样，再由人类标注。使用 GPT-4 等外部模型生成响应时，需注意分布不匹配——DPO 依赖参考模型 π_ref 做约束，如果偏好数据完全来自不同分布，约束会失效。
- **多样性与去重**：Prompt 多样性比单 prompt 的标注数量更重要。同一 prompt 重复出现会放大过拟合风险。
- **标注质量审核**：抽样检查标注一致性。低一致性标注（chosen 和 rejected 难以区分，或 chosen 实际更差）直接向训练注入噪声。
- **Chosen 的绝对质量**：Chosen 示例本身需要达到足够的质量水平。如果 chosen 本身质量低，DPO 只是引导模型"从差到不那么差"，而非"从差到好"。

### 迭代式 DPO 实践

DPO 虽然是 off-policy 方法，但可以通过迭代来弥补：

1. **SFT 阶段**：确保 SFT 模型在目标任务上有基本能力（20%+ baseline）。参见 [[../llm-sft/llm-sft]]。
2. **偏好数据采集**：SFT 模型采样 → 人类标注 → 质量审核 → 去重
3. **初次 DPO**：Beta=0.1，LR=1e-6，训练 1-3 个 epoch
4. **评估**：检查 chosen/rejected 的 log-prob 差值、生成质量、与 SFT 模型的 KL 散度
5. **迭代**：用当前策略重新生成响应，重新标注，再次 DPO。迭代能部分恢复 on-policy 的信号质量

### 训练监控要点

DPO 训练中需要同时跟踪以下指标来判断训练状态：

- **Chosen/Rejected 对数概率差**：正常应随训练稳步增长但速度放缓。增长过快提示 Beta 太低或 LR 太高。
- **与参考模型的 KL 散度**：监控策略偏移程度。KL 增长过快说明在过拟合偏好数据。
- **验证集 DPO Loss**：如果 Loss 持续下降但验证集上生成质量停滞或下降，应停止训练。
- **生成响应长度变化**：与 [[reward-hacking]] 一样，长度异常变化可能是过拟合信号。

## 常见误区

1. **DPO 可以替代 SFT**：错误。DPO（和所有偏好优化方法）建立在 SFT 之上。SFT 质量差时 DPO 无法补救。如果 SFT 模型在目标任务上能力不足，DPO 只会放大其缺陷。参见 [[../llm-sft/llm-sft]]。
2. **DPO Loss 低 = 效果好**：错误。DPO Loss 快速下降到接近 0 往往是 Beta 过低或 LR 过高的警告信号，说明模型已经完全记住了偏好数据，可能失去了泛化能力。
3. **Beta 是次要参数**：错误。Beta 是最重要的超参数，直接控制隐式奖励的尺度和约束强度，需要仔细调整。
4. **偏好数据越多越好**：DPO 对数据质量而非数量更敏感。少量高质量、差异合适的偏好对优于大量噪声数据。
5. **DPO 无需监控**：DPO 虽然比 PPO 简单，但仍需监控 chosen/rejected 的 log-prob 差值、生成质量变化、KL 散度（策略与参考的偏移）。

## 参见

- [[ppo-rlhf]] — PPO-RLHF，DPO 的主要对比方案
- [[kl-divergence-rlhf]] — KL 散度在偏好优化中的作用
- [[reward-hacking]] — Reward Hacking，DPO 隐式奖励同样可能被 hack
- [[grpo]] — GRPO，另一种简化 RL 的方法
- [[../llm-sft/llm-sft]] — SFT 是 DPO 的前提
- [[../optimizer-lr/optimizer-lr]] — 学习率与 Beta 的协同调参
- [[../../contradictory]] — 与直觉相反的 DPO 调参经验
