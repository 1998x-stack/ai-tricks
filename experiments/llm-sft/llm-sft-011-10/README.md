# llm-sft-011-10 · LoRA 调参指南

[Wiki 原位置：第 85 行](../../../wiki/llm-sft/llm-sft.md) · 作者：BugBuster喵

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `training.batch_size` |
| A | `4` |
| B | `16` |
| 任务配置 | `sft` |
| 主指标 | `test_rougeL`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/llm-sft/llm-sft-011-10/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/llm-sft/llm-sft-011-10/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: BugBuster喵
- **适用场景**: LoRA 微调的超参数选择
- **原文**: "LoRA 调参里，最常调的是学习率、rank、alpha、dropout、target modules、训练轮数、最大长度、batch。rank 常见从 8、16、32 开始，很多业务任务 rank 8 或 16 已经够用。rank 不是越大越好，数据少时大 rank 更容易过拟合。alpha 常见 16、32、64。LoRA dropout 0 到 0.1 之间试。学习率 1e-4 到 3e-4 是常见范围，但小数据、强基座模型可以更低。训练轮数不要太多，很多 SFT 两三个 epoch 已经开始记数据了。"
- **要点**: LoRA 调参要点：
  - **Rank**: 8/16/32 起步，多数业务任务 8 或 16 足够。数据量少时大 rank 反而容易过拟合。
  - **Alpha**: 16/32/64 常见。
  - **Dropout**: 0~0.1 范围内尝试。
  - **LR**: 1e-4 到 3e-4 是常见范围，小数据或强基座模型可以用更低。
  - **训练轮数**: 不要太多，2-3 个 epoch 后模型可能已经开始"记数据"而非学习范式。
- **参见**: [[../optimizer-lr/optimizer-lr]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
