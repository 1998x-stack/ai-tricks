# PPO for RLHF

## 是什么

PPO（Proximal Policy Optimization）是 RLHF（基于人类反馈的强化学习）中最主流的策略优化算法。它将 PPO 的信任区域约束机制引入语言模型微调，通过奖励模型（Reward Model, RM）的反馈信号引导模型生成更符合人类偏好的内容。

PPO-RLHF 的标准架构包含四个模型：

- **策略模型（Actor）**：即正在训练的语言模型，负责生成文本。参数通过 PPO 更新。
- **参考模型（Reference Model）**：SFT 模型的冻结副本。不参与梯度更新，仅用于计算 KL 散度，约束 Actor 不偏离过远。
- **价值模型（Critic）**：估计状态价值函数 V(s)，辅助计算 Advantage。通常与 Actor 共享部分参数或是同规模模型。
- **奖励模型（Reward Model）**：为 Actor 生成的文本打分的模型，通常在人类偏好数据上训练，参数冻结。

训练循环：Actor 基于 prompt 生成多个 response → RM 为每个 response 打分 → 结合 KL 惩罚项计算最终奖励 → Critic 估计 Advantage → PPO Loss 更新 Actor 和 Critic 参数。KL 惩罚项 `-β * KL(π_θ || π_ref)` 确保策略不背离参考模型过远。

## 为什么重要

### 为什么 PPO 而非其他 RL 算法

RLHF 场景下 PPO 成为首选，并非偶然：

1. **信任区域约束**：PPO 的 clipped surrogate objective 限制每次更新的步长，防止策略一步崩溃。这对语言模型至关重要——一步过大更新可能摧毁语言能力和知识。
2. **双重保护机制**：PPO 的 clip 限幅 + KL 惩罚项构成双重防线。clip 控制单步更新幅度，KL 控制整体分布偏移。相比 TRPO（计算二阶导数，LLM 场景不可行），PPO 仅需一阶梯度，计算可行。
3. **On-policy 采样**：PPO 是 on-policy 算法，每次更新使用当前策略的最新采样。在 RM 固定的 RLHF 中，on-policy 特性保证奖励信号反映当前策略的真实质量。
4. **工程成熟度**：InstructGPT / ChatGPT / Llama-Chat 等代表性工作均使用 PPO，社区积累了丰富的调参经验和开源实现（DeepSpeed Chat、TRL、OpenRLHF）。

### PPO 在 RLHF 中的独特挑战

与标准 RL（游戏、机器人控制）不同，RLHF 的"环境"不是物理世界而是语言空间：

- 奖励来自学出来的 RM，而非环境真实回报，本身有噪声和偏差
- 动作空间（词表）极其巨大
- 单个 episode 长度（sequence length）可达数千 token
- 模型不能"试错"太多——偏离 SFT 过远会丢失语言能力

PPO 的保守更新策略恰好匹配这些约束。

## 如何使用

### 学习率：SFT 的十分之一

PPO 学习率通常比 SFT 小一个数量级：

- SFT 学习率：~2e-5
- PPO 初始学习率：1e-6 到 3e-6

**原理**：SFT 阶段模型从零学习任务形式，需要较大步长探索参数空间。PPO 阶段模型已有基本生成能力，只需在偏好方向微调。过大的 LR 会让参数跳离有效区域，导致 Mode Collapse（输出重复或无意义内容）。参见 [[../optimizer-lr/optimizer-lr]]。

### Batch size 宁大勿小

RLHF 的 batch size 越大越好。原因：

- PPO 的 Advantage 估计依赖 Critic 网络，小 batch 下 Critic 难以收敛，导致 Advantage 方差大
- 梯度估计方差随 batch 增大而减小，训练更稳定
- 模型能力越弱，越需要大 batch 来平滑信号

显存不足时使用 Gradient Accumulation 等效扩大 batch size。不建议用很小的 batch（如 4-8）硬跑。

### 梯度裁剪值 0.2

梯度裁剪是训练不稳定的第一道防线：

- 推荐裁剪值：0.2
- 触发场景：Loss 突然飙升、reward 大幅震荡
- 裁剪值过低（<0.1）会阻碍正常梯度更新；过高（>1.0）失去保护意义

### 训练稳定性全景

RLHF 训练不稳定是多因素问题，需要多角度协同：

| 方面 | 推荐配置 | 解决的问题 |
|------|----------|-----------|
| KL 惩罚系数 | 从 0.001 开始爬坡 | 策略偏离过远 |
| 学习率 | SFT 的 1/10 | 更新步长过大导致 Mode Collapse |
| 梯度裁剪 | 0.2 | 梯度爆炸 |
| Reward 归一化 | 训练前对 RM 输出标准化 | 梯度消失/爆炸 |
| Advantage 归一化 | 对 Advantage 做 Z-score | Critic 训练不稳定 |
| 熵系数 | 0.01-0.05 | 探索不足、生成同质化 |
| LR Schedule | Cosine + Warmup | LR 突变 |
| 混入 SFT 数据 | Prompt 池中 5%-10% | 灾难性遗忘 |

**奖励模型输出的归一化**：PPO 训练前务必对 RM 输出做归一化（通常 Z-score）。不同 RM 的打分范围不同（如有的 0-5，有的 -10 到 +10），未经归一化会导致梯度幅度与 reward scale 耦合，难以设置统一的学习率。

**Critic Value Loss 波动**：如果 Critic 的 Loss 剧烈震荡，通常因为 reward 方差太大。处理方法：对 reward 做归一化、对 advantage 做归一化、增大 batch size。

**余弦 LR 调度 + Warmup**：推荐 cosine LR schedule 配合 warmup 步骤。Warmup 阶段让模型逐步适应 RL 更新，避免开局就大更新导致崩溃。

## 常见误区

1. **关注 Loss 而非 Reward**：RL 训练的核心监控指标是 reward 曲线，loss 仅作参考。Loss 下降而 reward 不动，往往是策略在坍塌而不是在学习。
2. **SFT 不达标就上 PPO**：SFT 模型在目标数据上如果连 20% 的正确率都达不到，PPO 无法挽救。PPO 只能将已有的能力推向上限，不能创造不存在的能力。参见 [[../llm-sft/llm-sft]]。
3. **KL 惩罚越大越安全**：KL 系数过大导致模型被过度束缚，reward 不增长或者增长缓慢。需要在小值（0.001）基础上逐步调整。
4. **只看 Reward 不人工评估**：Reward 上涨但输出质量下降是 Reward Hacking 的典型信号。必须定期取不同 checkpoints 做人类评估。
5. **小 batch + 高 LR 开局**：这是最常见的训练崩溃组合。小 batch 导致梯度噪声大，高 LR 让噪声被放大，几步内模型就崩了。

## 参见

- [[kl-divergence-rlhf]] — KL 散度的作用与调参策略
- [[reward-hacking]] — Reward Hacking 的全面解析
- [[dpo]] — DPO，无 RM 的偏好优化替代方案
- [[grpo]] — GRPO，去 Critic 的分组策略优化
- [[../llm-sft/llm-sft]] — SFT 作为 RL 的前提条件
- [[../optimizer-lr/optimizer-lr]] — 学习率调度与 warmup 策略
- [[../../contradictory]] — 与直觉相反的 RLHF 经验
