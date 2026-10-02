# llm-sft-009-01 · Loss 不可靠，必须看 Checkpoint 指标

[Wiki 原位置：第 72 行](../../../wiki/llm-sft/llm-sft.md) · 作者：Tang AI

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `checkpoint.selection_metric` |
| A | `"val_loss"` |
| B | `"val_exact_match"` |
| 任务配置 | `sft` |
| 主指标 | `test_rougeL`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/llm-sft/llm-sft-009-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/llm-sft/llm-sft-009-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Tang AI
- **适用场景**: SFT 训练监控
- **原文**: "事实上loss，对于大模型训练仅能做一个基本的判断了，基本上，到了中后期，loss小，效果不一定好，甚至还会出现loss先下降后上升的情况，即使是在训练集。所以，正确的判断方法还是，从小步数的checkpoint，进行accuray或者其他业务指标的判断，正常情况，只要模型在正常训练，该指标通常也是下降逐渐到收敛的一个状态。因此，不能太依赖loss指标，仅仅做初步参考。"
- **要点**: 大模型 SFT 训练中后期，loss 与模型实际效果之间关联度很低。可能出现 loss 下降但效果变差，或者 loss 先降后升的情况。正确做法是定期保存 checkpoint，在每个 checkpoint 上跑业务指标（准确率、F1、人工评测等）来评估效果。Loss 仅作为训练是否正常的初步参考。
- **参见**: [[../data-eval/data-eval]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
