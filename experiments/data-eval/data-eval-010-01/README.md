# data-eval-010-01 · 对比训练集和验证集的loss判断过拟合

[Wiki 原位置：第 80 行](../../../wiki/data-eval/data-eval.md) · 作者：Captain Jack / 无忧无虑

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `tracking.diagnostics` |
| A | `"train_only"` |
| B | `"train_and_val"` |
| 任务配置 | `vision` |
| 主指标 | `diagnostic_fault_recall`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/data-eval/data-eval-010-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/data-eval/data-eval-010-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Captain Jack / 无忧无虑
- **适用场景**: 训练过程中的诊断
- **原文**: "判断过拟合，训练是否足够，是否需要early stop的依据" / "模型在训练集上的误差很小……但是验证集的 loss 相对较大。那就是过拟合了。在训练集和验证集上的loss都比较大……那可能就是欠拟合了。"
- **要点**: 训练loss低+验证loss高=过拟合（加正则化、加数据）；两边loss都高=欠拟合（加模型容量、调学习率）。这是调参诊断的第一步。
- **参见**: [../regularization/overfitting-underfitting](../../../wiki/regularization/overfitting-underfitting.md), `[[underfitting]]`（原 Wiki 未解析链接）

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
