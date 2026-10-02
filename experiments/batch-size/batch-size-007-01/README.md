# batch-size-007-01 · 大 batch 训练快但泛化不一定更好，需记录 effective batch size

[Wiki 原位置：第 55 行](../../../wiki/batch-size/batch-size.md) · 作者：BugBuster喵

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `tracking.batch_metadata` |
| A | `"nominal_only"` |
| B | `"micro_world_accum_effective"` |
| 任务配置 | `vision` |
| 主指标 | `diagnostic_fault_recall`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/batch-size/batch-size-007-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/batch-size/batch-size-007-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: BugBuster喵
- **适用场景**: A100/H100 多卡训练、梯度累积场景
- **原文**: > batch size 也重要，但它不是越大越好。大 batch 训练快，梯度稳定，但泛化不一定更好。今天大家用 A100、H100、L40S、国产卡集群，gradient accumulation 很常见，看到的 batch size 往往是 effective batch size。调参时要记录清楚：单卡 batch、卡数、累积步数、总 batch 到底是多少。很多复现实验复现不出来，就是这里没说清。
- **要点**: 使用多卡和梯度累积时必须记录「单卡 batch × 卡数 × 累积步数 = effective batch size」的完整公式，否则实验无法复现。
- **参见**: [gradient-accumulation](../../../wiki/batch-size/gradient-accumulation.md), [batch-size-generalization](../../../wiki/batch-size/batch-size-generalization.md), [../infra/distributed-training](../../../wiki/infra/distributed-training.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
