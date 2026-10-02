# llm-sft-015-03 · Packing 策略

[Wiki 原位置：第 118 行](../../../wiki/llm-sft/llm-sft.md) · 作者：BugBuster喵

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `data.packing_positions` |
| A | `"continuous"` |
| B | `"reset_per_example"` |
| 任务配置 | `sft` |
| 主指标 | `test_rougeL`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/llm-sft/llm-sft-015-03/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/llm-sft/llm-sft-015-03/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: BugBuster喵
- **适用场景**: SFT 训练效率优化
- **原文**: "大模型微调里，数据调参包括...是否 packing、最大长度设置..."（上下文多处提及 sequence packing）
- **要点**: Sequence Packing 将多个短样本拼接为一个训练序列，可显著提升训练吞吐和 GPU 利用率。但 packing 需要正确处理 attention mask 和 position ID，确保不同样本之间不会互相干扰。packing 策略的选择（按长度相似度打包 vs 随机打包）也会影响训练稳定性。
- **参见**: [../infra/infra](../../../wiki/infra/infra.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
