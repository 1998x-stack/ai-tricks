# llm-sft-017-03 · SFT 常见陷阱

[Wiki 原位置：第 136 行](../../../wiki/llm-sft/llm-sft.md) · 作者：BugBuster喵

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `data.preprocess_contract` |
| A | `"mismatched"` |
| B | `"identical"` |
| 任务配置 | `sft` |
| 主指标 | `test_rougeL`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/llm-sft/llm-sft-017-03/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/llm-sft/llm-sft-017-03/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: BugBuster喵
- **适用场景**: SFT 训练流程自查
- **原文**: "大模型 SFT 里，assistant 部分和 user 部分都参与 loss，模型学得乱七八糟。"
- **要点**: SFT 训练中几个最常见的"调参之前先修 bug"场景：
  1. **Loss 计算包含 user 部分**: SFT 时 loss 应只计算 assistant 回复部分的 token，user 问句和 system prompt 应该被 mask 掉。如果 user 部分也被计入 loss，模型会学到"预测用户的问法"，严重干扰训练效果。
  2. **Tokenizer 不一致**: 训练时使用的 tokenizer 与推理时不匹配，导致生成结果异常。
  3. **训练和推理预处理不一致**: 数据清洗、截断策略、特殊 token 处理等必须完全相同。
  4. **Chat template 不一致**（见上条独立技巧）。
- **参见**: [[../data-eval/data-eval]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
