# nlp-011-01 · Transformer Warmup

[Wiki 原位置：第 85 行](../../../wiki/nlp/nlp.md) · 作者：LindgeWAI / BugBuster喵

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `optimizer.warmup_ratio` |
| A | `0.0` |
| B | `0.1` |
| 任务配置 | `nlp` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{"optimizer.schedule": "warmup_cosine"}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/nlp/nlp-011-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/nlp/nlp-011-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: LindgeWAI / BugBuster喵
- **适用场景**: Transformer 系列模型（BERT、GPT、LLaMA 等）的训练
- **原文**: *"lr_scheduler的warmup步数设置：一般遵循模型复杂，步数多点；数据集大，步数少点 (参考BERT，默认设成总训练轮次的1/10，也可1/20)"*
- **要点**: Transformer 训练基本必须开启 warmup。推荐 warmup 步数为总步数的 3%~10%（复杂模型取高值，大数据集取低值）。warmup + cosine/linear decay 是最常见的搭配。warmup 的作用是防止初期大学习率导致的不稳定。
- **参见**: [../optimizer-lr/optimizer-lr](../../../wiki/optimizer-lr/optimizer-lr.md), `[[transformer-training]]`（原 Wiki 未解析链接）

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
