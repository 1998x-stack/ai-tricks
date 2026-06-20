# Group Relative Policy Optimization (GRPO)

## 是什么

GRPO（Group Relative Policy Optimization）是由 DeepSeek 提出的一种无 Critic 模型的强化学习算法，专为大语言模型的对齐训练设计。与传统 PPO-RLHF 不同，GRPO 不需要训练价值模型（Critic）来估计 Advantage，而是通过**对同一 prompt 生成多个响应，利用组内相对表现**来计算 Advantage。

### 核心机制

GRPO 的训练循环如下：

1. 对每个 prompt x，当前策略 π_θ 生成一组响应 {y_1, y_2, ..., y_G}
2. 奖励模型（Rule-based 或 Model-based）为每个响应打出 r_i
3. 在组内计算 Advantage：`A_i = (r_i - mean(r)) / std(r)`——即每个响应相对于组均值的偏离程度
4. 用这些 Advantage 更新策略 π_θ

GRPO 的损失函数与 PPO 类似，包含 clipped surrogate objective 和 KL 散度惩罚，但没有 Critic Loss：

`L_GRPO(θ) = -1/G * Σ_i [min(ρ_i * A_i, clip(ρ_i, 1-ε, 1+ε) * A_i) - β * KL(π_θ || π_ref)]`

其中 ρ_i = π_θ(y_i|x) / π_ref(y_i|x) 是重要性采样比率。

### 与 PPO 的架构对比

| 维度 | PPO-RLHF | GRPO |
|------|----------|------|
| Advantage 来源 | Critic 网络估计 V(s) | 组内响应得分对比 |
| 模型数量 | Actor + Critic + Reference + RM | Actor + Reference + RM |
| 显存需求 | 高（需维护 Critic） | 较低（无需 Critic） |
| 每轮采样 | 每个 prompt 1 个 response | 每个 prompt G 个 response |
| 适用场景 | 通用 RLHF | 规则 reward 明确的场景 |

### 数学直觉

GRPO 的核心洞察是：**给定同一 prompt，好的回答和差的回答之间的差异本身就包含了偏好信息**。通过对同一 prompt 采样 G 个回答并在组内做归一化，GRPO 实现了两个目的：

1. 自动消除 prompt 难度差异的影响（简单 prompt 的所有回答 reward 都高，但组内归一化后只有相对高的才获得正 Advantage）
2. 无需专门训练 Critic 来估计 baseline——组内平均起到了天然 baseline 的作用

## 为什么重要

### GRPO 的核心优势

1. **简化训练流程**：去掉 Critic 网络意味着少维护一个模型，少一套损失函数，训练框架更简洁。
2. **显存效率**：对于 70B 量级的模型，少一个网络意味着显著的显存节省。这使得 GRPO 在相同的硬件条件下可以训练更大的模型或使用更大的 batch。
3. **无需连续价值估计**：Critic 网络在语言模型场景中面临"如何定义状态价值"的困难——语言生成没有自然的 discounted return 概念。GRPO 绕过了这个难题。
4. **天然适配规则 reward**：DeepSeek-R1 的实践表明，GRPO 在数学推理、编程等可以用确定性规则做 reward 的场景效果突出。

### 适用场景与局限性

**最适用**：
- 有明确的 rule-based reward（数学题、编程竞赛、逻辑推理）
- 资源受限（GPU 显存不足，无法跑完整 PPO）
- prompt 间难度差异大（组内归一化自动处理）

**不太适用**：
- 需要 Critic 的泛化能力来指导探索的场景
- RM 打分噪声大、组内方差太小时（信号会弱）
- 每 prompt 采样 G 个响应的成本不可接受时

## 如何使用

### Group Size G 的选择

组大小 G 直接影响 Advantage 估计的质量：

- **G 越小**：组均值不可靠，Advantage 噪声大，训练不稳定
- **G 越大**：Advantage 估计越精确，但采样成本线性增加
- **经验值**：G=8 是最低可用值，G=16-64 为推荐区间
- **模型能力弱时特别需要大 G**：基模输出质量不稳定，需要更多的样本来获得可靠的分组对比

### Batch Size 的影响

GRPO 同样遵循"batch size 越大越稳定"的规律。特别地：

- on-policy 采样量 = batch_size × G（每个 prompt 采 G 个）
- 更大的 batch 提供更稳定的梯度估计
- 小 batch + 小 G 的组合极易导致训练发散

这与 PPO 是一致的——稳定的梯度估计是所有策略优化方法的基础。参见 [[ppo-rlhf]]。

### 拒绝采样 + 冷启动微调

冷启动微调是 GRPO / Base-RL 训练流程中经实践证明有效的策略：

**步骤**：
1. 用 GRPO 训练一定步数的基模（Base-RL）
2. 在当前策略下对 prompt 集做拒绝采样：每个 prompt 生成 N 个响应（N 可以很大，如 64-128）
3. 筛选出 reward 最高的一批响应作为"伪 SFT 数据"
4. 用这些数据对**原始基模（或当前模型）**做简单 SFT（冷启动微调）
5. 继续 GRPO 训练

**为什么有效**：
- 基模在初期探索阶段质量不稳定，冷启动用高 reward 样本做一轮 SFT 相当于"把好成果固化下来""
- 这为后续 RL 提供了一个比原始 SFT 更强的起点
- 是突破 reward 瓶颈的有效手段——当 reward 长时间不涨时，拒绝采样 + 冷启动往往能打开新局面

**条件**：基模本身应具备一定的生成能力。如果采样 N 个响应都得不到高 reward 样本，说明基模能力上限到了，冷启动也无济于事。

### Reward 设计注意事项

GRPO 中 Reward 设计尤为重要，因为组内对比放大了 reward 的差异信息：

- Rule-based reward 规则要全面但不过多
- 正负例数量要平衡——如果负例远多于正例且容易获得，组内 mean 会偏负，导致正 Advantage 膨胀
- 长度偏差同样影响 GRPO——如果 RM 偏好长文本，长回答在组内获得正 Advantage 的概率更大

参见 [[reward-hacking]]。

## 常见误区

1. **GRPO 完全替代 PPO**：GRPO 和 PPO 在不同场景各有优势。需要 Critic 提供价值引导或 RM 打分连续时 PPO 仍更合适。
2. **G 越大越好**：G 增大收益递减。G=128 相比 G=64 的改进通常远小于 G=16 到 G=32。要考虑计算成本效益比。
3. **去掉 Critic 就不需要 RL 稳定性技巧**：GRPO 仍然需要 KL 惩罚、梯度裁剪、reward 归一化等技巧来保证稳定训练。
4. **拒绝采样 + 冷启动是可选优化**：对于 Base-RL 场景，这个流程往往是突破瓶颈的关键，不是可选的"锦上添花"。
5. **GRPO 不需要 SFT**：GRPO 同样依赖 SFT 模型提供基准能力。基模不做 SFT 直接 GRPO 在大部分场景效果有限。

## 参见

- [[ppo-rlhf]] — PPO-RLHF，GRPO 的主要对比方案
- [[dpo]] — DPO，另一种离线偏好优化方法
- [[kl-divergence-rlhf]] — GRPO 中的 KL 约束角色
- [[reward-hacking]] — Reward Hacking，GRPO 中组内对比可能放大局部偏差
- [[../llm-sft/llm-sft]] — SFT 作为 GRPO 的基础
- [[../../contradictory]] — 关于 GRPO 的反直觉发现
