# normalization-011-01 · Group Normalization 可替代 BN

[Wiki 原位置：第 86 行](../../../wiki/normalization/normalization.md) · 作者：吴育昕 / 何恺明 (FAIR)

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `model.norm` |
| A | `"batch"` |
| B | `"group"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/normalization/normalization-011-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/normalization/normalization-011-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: 吴育昕 / 何恺明 (FAIR)
- **适用场景**: 小 batch 训练 / 检测分割
- **原文**: > Face book AI research（FAIR）吴育昕-恺明联合推出重磅新作Group Normalization（GN），提出使用Group Normalization 替代深度学习里程碑式的工作Batch normalization。一句话概括，Group Normbalization（GN）是一种新的深度学习归一化方式，可以替代BN。
- **要点**: GN 将通道分为若干组，每组内计算均值和方差，不依赖 batch 维度，在小 batch（batch_size=1~8）场景效果优于 BN。
- **参见**: [group-normalization](../../../wiki/normalization/group-normalization.md), [../cnn-cv/cnn-cv](../../../wiki/cnn-cv/cnn-cv.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
