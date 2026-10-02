# debugging-026-01 · Debug流程：从简单到复杂

[Wiki 原位置：第 191 行](../../../wiki/debugging/debugging.md) · 作者：Josh Robin (伯克利) / 量子位

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `validation.preflight` |
| A | `"none"` |
| B | `"tiny_overfit"` |
| 任务配置 | `vision` |
| 主指标 | `diagnostic_fault_recall`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/debugging/debugging-026-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/debugging/debugging-026-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Josh Robin (伯克利) / 量子位
- **适用场景**: 系统的深度学习调试流程
- **原文**: "从不最简单模型入手；成功搭建模型，重现结果；分解偏差各项，逐步拟合数据"
- **要点**: 1) 从最简单的架构开始；2) 在小数据上跑通；3) 逐步增加复杂度；4) 由粗到细搜索超参数；5) 欠拟合增大模型，过拟合加数据。
- **参见**: [[../optimizer-lr/optimizer-lr]], [[../data-eval/data-eval]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
