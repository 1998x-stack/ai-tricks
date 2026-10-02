# llm-sft-016-02 · Max Length 设置

[Wiki 原位置：第 125 行](../../../wiki/llm-sft/llm-sft.md) · 作者：BugBuster喵

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `model.rope_scaling` |
| A | `"none"` |
| B | `"position_interpolation"` |
| 任务配置 | `sft` |
| 主指标 | `test_rougeL`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/llm-sft/llm-sft-016-02/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/llm-sft/llm-sft-016-02/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: BugBuster喵
- **适用场景**: SFT 训练数据截断策略
- **原文**: "长上下文训练要特别注意显存和位置编码适配，不是把 max length 改到 32k 就完事。"
- **要点**: `max_length` 不是可以无脑设大的参数。增加 max length 会：
  - 大幅增加显存消耗（attention 计算量随长度平方增长）
  - 需要位置编码适配（RoPE 等的外推能力有限）
  - 可能导致训练不稳定
  建议根据实际数据分布设定，统计训练数据中 95% 分位的长度作为 max length 起点，再进行微调。
- **参见**: [../infra/infra](../../../wiki/infra/infra.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
