# llm-rl-007-01 · 奖励设计：规则完整性

[Wiki 原位置：第 63 行](../../../wiki/llm-rl/llm-rl.md) · 作者：Tang AI

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `reward.validation` |
| A | `"format_only"` |
| B | `"format_and_correctness"` |
| 任务配置 | `reasoning_rl` |
| 主指标 | `test_exact_match`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/llm-rl/llm-rl-007-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/llm-rl/llm-rl-007-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Tang AI
- **适用场景**: reward function设计
- **原文**: "reward的规则一定要考虑完整，且不能太多reward rule，否则会让模型彻底偏向某个规则，或者直接reward hacking。比如，对于仅有1个正例或者1个负例的数据集，如果完成正例的回答是很难的，而回答负例是很简单的，且正例数据比负例少，模型就倾向于全部数据都打负例，这样模型依然可以得到高分。"
- **要点**: 规则要完整但不宜过多；注意正负样本比例平衡，防止模型走捷径（全打负例获高分）。
- **参见**: [[reward-hacking]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
