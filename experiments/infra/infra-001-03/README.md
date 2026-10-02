# infra-001-03 · 混合精度训练（FP16 / BF16）

[Wiki 原位置：第 16 行](../../../wiki/infra/infra.md) · 作者：LindgeWAI、知乎社区（匿名）

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `training.loss_scaling` |
| A | `"none"` |
| B | `"dynamic"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/infra/infra-001-03/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/infra/infra-001-03/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: LindgeWAI、知乎社区（匿名）
- **适用场景**: 大模型微调与大规模训练，显存受限时
- **原文**: "混合精度训练可以在保持模型性能的同时减少内存消耗和计算时间。通过在训练过程中使用半精度浮点数（FP16）和全精度浮点数（FP32）的混合，可以有效提高训练效率。" / "bf16 优先于 fp16，前提是硬件支持。fp16 下 loss scale、溢出问题有时会让你误判参数效果。"
- **要点**: BF16 优先于 FP16（前提是硬件支持）；FP16 需注意 loss scaling 和溢出问题；使用 `apex.amp` 或 PyTorch 原生 AMP 实现；常用 opt_level='O1'。
- **参见**: [../batch-size/batch-size](../../../wiki/batch-size/batch-size.md), [../optimizer-lr/optimizer-lr](../../../wiki/optimizer-lr/optimizer-lr.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
