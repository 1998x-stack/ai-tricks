# llm-sft-012-04 · QLoRA 的代价与稳定性

[Wiki 原位置：第 97 行](../../../wiki/llm-sft/llm-sft.md) · 作者：BugBuster喵

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `training.gradient_checkpointing` |
| A | `false` |
| B | `true` |
| 任务配置 | `sft` |
| 主指标 | `peak_cuda_memory_bytes`（min） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/llm-sft/llm-sft-012-04/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/llm-sft/llm-sft-012-04/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: BugBuster喵
- **适用场景**: 显存受限时的 QLoRA 微调
- **原文**: "QLoRA 很省显存，但别以为它没有代价。量化配置、计算精度、梯度 checkpoint、sequence packing 都会影响训练稳定性。bf16 能用就用。长上下文训练要特别注意显存和位置编码适配，不是把 max length 改到 32k 就完事。"
- **要点**: QLoRA 虽然大幅降低了显存需求，但引入了额外的调参维度：量化配置（4-bit NF4 vs FP4）、计算精度、是否开启梯度 checkpoint、是否使用 sequence packing，这些都会影响训练稳定性和最终效果。优先使用 bf16（硬件支持前提下），长上下文训练时要特别注意位置编码的适配——不是简单改 `max_length` 就能解决。
- **参见**: [../infra/infra](../../../wiki/infra/infra.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
