# nlp-013-01 · 序列分桶策略（Bucketing）

[Wiki 原位置：第 99 行](../../../wiki/nlp/nlp.md) · 作者：LindgeWAI

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `data.batch_order` |
| A | `"random"` |
| B | `"length_bucket"` |
| 任务配置 | `nlp` |
| 主指标 | `examples_per_second`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/nlp/nlp-013-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/nlp/nlp-013-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: LindgeWAI
- **适用场景**: 序列模型训练效率优化
- **原文**: *"分桶策略可以根据输入序列的长度将数据分成不同的桶，从而在训练过程中更高效地利用计算资源。这种方法可以减少 padding 的数量，提高训练效率。"*
- **要点**: 将相同或相近长度的序列分到同一 batch，减少无效 padding 的计算量。可用 `bucket_by_sequence_length` 实现，显著提升吞吐量。
- **参见**: `[[training-efficiency]]`（原 Wiki 未解析链接）

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
