# llm-rl-025-01 · reward不涨：检查基模能力上限

[Wiki 原位置：第 189 行](../../../wiki/llm-rl/llm-rl.md) · 作者：Tang AI

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `checkpoint.policy_start` |
| A | `"weak_base"` |
| B | `"strong_base"` |
| 任务配置 | `reasoning_rl` |
| 主指标 | `test_exact_match`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `policy` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/llm-rl/llm-rl-025-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/llm-rl/llm-rl-025-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Tang AI
- **适用场景**: reward停滞诊断
- **原文**: "reward始终不上涨，在排除了一切可能的原因后，建议用训练前的模型针对一些case rollout出多个回复（n可以大一点），看下这些回复的奖励是不是都特别低，如果都特别低，那说明基模的能力上限就如此，想要通过探索来提升表现是行不通的，建议换模型或者对SFT模型进行优化。"
- **要点**: Reward不上涨 → rollout多个回复检查奖励分布 → 如果都低则说明基模能力已达上限，需换模型或优化SFT。
- **参见**: [[../llm-sft/llm-sft]], [[model-capacity]]

---

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。本实验比较整体策略，策略内可能改变多项配置；不作单参数因果解释。
