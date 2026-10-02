# llm-rl-003-01 · 确定规则reward有效，主观打分reward效果有限

[Wiki 原位置：第 31 行](../../../wiki/llm-rl/llm-rl.md) · 作者：Tang AI

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `reward.kind` |
| A | `"subjective_judge"` |
| B | `"verifiable_rule"` |
| 任务配置 | `reasoning_rl` |
| 主指标 | `test_exact_match`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/llm-rl/llm-rl-003-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/llm-rl/llm-rl-003-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Tang AI
- **适用场景**: reward设计
- **原文**: "对于有明显规则可以作为critic的问题，比如数学或者逻辑问题，RL是有效果的，但是如果没有明显规则，需要用大模型或者other模型打分的问题，可能RL就没有那么大的效果了。"
- **要点**: 数学/逻辑等有明确规则的场景RL效果好；主观评价场景（如"C罗比梅西伟大吗"）效果有限。
- **参见**: [reward-hacking](../../../wiki/llm-rl/reward-hacking.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
