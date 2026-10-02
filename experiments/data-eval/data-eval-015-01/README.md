# data-eval-015-01 · 类别不平衡：Focal Loss与类别加权

[Wiki 原位置：第 115 行](../../../wiki/data-eval/data-eval.md) · 作者：Tang AI / anchor / DOTA / 黑马程序员

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `loss.kind` |
| A | `"cross_entropy"` |
| B | `"focal"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/data-eval/data-eval-015-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/data-eval/data-eval-015-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Tang AI / anchor / DOTA / 黑马程序员
- **适用场景**: 类别极度不平衡的分类任务（异常检测、长尾识别）
- **原文**: "Focal loss对于极大不平衡的数据集确实有奇效，其中gamma因子可以成10倍数衰减" / "如果你的分类精度不够是样本不均衡造成的，考虑使用focal loss。" / "如果你有高度不平衡的数据问题，在训练期间应用类别加权操作。"
- **要点**: Focal Loss通过调制系数降低易分类样本的loss权重，使训练更关注难样本。gamma通常从0.5开始尝试，以10倍步长搜索。也可用sklearn的class_weight计算类别权重。过采样/欠采样也是常用手段。
- **参见**: [class-imbalance](../../../wiki/data-eval/class-imbalance.md), [focal-loss](../../../wiki/data-eval/focal-loss.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
