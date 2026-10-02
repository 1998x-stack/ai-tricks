# 实验前需要纠偏的 Wiki 断言

保留 Wiki 原始归因，不把作者经验强行改成通用定律。下列纠偏也写入对应实验协议。

| 原位置 | 实验解释 |
|---|---|
| [批大小：对比学习](../wiki/batch-size/batch-size.md) | 普通梯度累积不会自动增加单次 InfoNCE 的负样本池；需明确跨 microbatch 收集表示/重新计算。含 BN 的累积也不严格等价于大 batch。 |
| [He 初始化](../wiki/initialization/initialization.md) | 方差是 `2/fan_in`，标准差是 `sqrt(2/fan_in)`。 |
| [零初始化](../wiki/initialization/initialization.md) | 全零隐藏层保持对称性；单层线性模型和深层网络需分开讨论。 |
| [随机标签测试](../wiki/debugging/debugging.md) | 大容量网络可以记忆随机训练标签；独立验证表现才是泛化判据。 |
| [损失输入](../wiki/debugging/debugging.md) | PyTorch CrossEntropyLoss 接收 logits；不要先 softmax。BCEWithLogitsLoss 同理。 |
| [SFT 门槛与数据量](../wiki/llm-sft/llm-sft.md) | “20% 门槛”“每领域 400 条”“SFT 不需要 CoT”均需任务条件，不能提前当作结论。 |
| [训练/推理学习率](../wiki/optimizer-lr/optimizer-lr.md) | 普通推理不执行参数更新，没有推理学习率；对照的是预处理一致性。 |

A/B 中 B 只是候选操作，不保证更优。原文中的绝对词、经验范围、商业案例数值都不作为测试必须通过的断言。
