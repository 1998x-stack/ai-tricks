# data-eval-008-01 · 不要只看loss，要看任务相关指标

[Wiki 原位置：第 66 行](../../../wiki/data-eval/data-eval.md) · 作者：Tang AI / anchor / BugBuster喵

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `checkpoint.selection_metric` |
| A | `"val_loss"` |
| B | `"val_f1"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/data-eval/data-eval-008-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/data-eval/data-eval-008-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Tang AI / anchor / BugBuster喵
- **适用场景**: 类别不平衡、排序、生成等任务
- **原文**: "对于负样本太多的数据集，测试集的loss降并不代表没有过拟合，试试看看f1或者auc有没有降低" / "不要只看精度，考虑其他metrics，并始终以上线后的可视化效果为最终评判标准。" / "训练loss下降很舒服，但业务指标可能不动。分类看校准，排序看NDCG、转化，检测看不同尺寸目标，生成看事实性和可用性。"
- **要点**: Loss下降可能只是因为负样本学得好。始终使用与业务目标对齐的评估指标（F1、AUC、NDCG、召回率、精确率等），不能只依赖loss。
- **参见**: [eval-metrics](../../../wiki/data-eval/eval-metrics.md), [class-imbalance](../../../wiki/data-eval/class-imbalance.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
