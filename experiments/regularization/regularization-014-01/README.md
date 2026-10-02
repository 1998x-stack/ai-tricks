# regularization-014-01 · 泛化诊断框架（过拟合 vs 欠拟合）

[Wiki 原位置：第 109 行](../../../wiki/regularization/regularization.md) · 作者：爱睡觉的KKY / 四点半的理一 / BugBuster喵 / 匿名 / 清明

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `search.policy` |
| A | `"random_changes"` |
| B | `"train_val_gap_diagnosis"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `policy` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/regularization/regularization-014-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/regularization/regularization-014-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: 爱睡觉的KKY / 四点半的理一 / BugBuster喵 / 匿名 / 清明
- **适用场景**: 任何训练实验的第一步——诊断
- **原文**: "训练集loss低但验证集loss高，那是典型的过拟合，优先加数据，其次加正则化，最后才考虑降低模型容量。" ——四点半的理一
- **要点**: 用训练/验证loss差值为诊断依据：差值大=过拟合→加数据/正则化；两边都高=欠拟合→增大模型/学习率。不要同时调超过两个超参数。每次实验前先写假设，跑完对比。
- **参见**: [[overfitting-underfitting]], [[../data-eval/data-eval]], [[../debugging/small-data-overfit-test]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。本实验比较整体策略，策略内可能改变多项配置；不作单参数因果解释。
