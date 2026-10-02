# regularization-013-01 · BN 的正则化效果与 Dropout 的交互

[Wiki 原位置：第 102 行](../../../wiki/regularization/regularization.md) · 作者：BBuf / 匿名 / 罗浩 / 苏辄 / DOTA / 四点半的理一

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `model.dropout` |
| A | `0.0` |
| B | `0.5` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{"model.norm": "batch"}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/regularization/regularization-013-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/regularization/regularization-013-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: BBuf / 匿名 / 罗浩 / 苏辄 / DOTA / 四点半的理一
- **适用场景**: 构建CNN/深度网络，BN与Dropout的取舍
- **原文**: "如果有BN了全连接层就没必要加Dropout了。" ——BBuf
- **要点**: BN兼具加速训练和轻微正则化效果。过拟合时优先加Dropout再加BN。过拟合越严重，Dropout+BN加的位置就越多，有奇效（甚至对Embedding层加）。一般建议：CV任务尽量加BN，RNN/序列任务用LN。小batch时考虑GN替代BN。
- **参见**: [[../normalization/normalization]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
