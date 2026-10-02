# infra-015-01 · 计算最优训练：学习率与 Batch Size 的 Scaling Law

[Wiki 原位置：第 128 行](../../../wiki/infra/infra.md) · 作者：Sam聊算法

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `search.scaling_policy` |
| A | `"fixed"` |
| B | `"deepseek_compute_law"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `policy` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/infra/infra-015-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/infra/infra-015-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Sam聊算法
- **适用场景**: 大模型预训练超参缩放
- **原文**: "给定计算成本 C，大多数超参的最优值是基本固定的，只需调整学习率和 batch size，而学习率 eta 的最优值随 C 增加而下降，batch size 的最优值随 C 增加而上升，满足幂律：eta_opt = 0.3118 * C^(-0.1250), B_opt = 0.2920 * C^(0.3271)"
- **要点**: 来自 DeepSeek LLM 技术报告；计算预算增大时应扩大 batch size 并降低学习率；其他超参（beta、weight_decay、梯度裁剪）取固定最优值即可；公式提供理论锚点，实际仍需网格搜索微调。
- **参见**: [../batch-size/batch-size](../../../wiki/batch-size/batch-size.md), [../optimizer-lr/optimizer-lr](../../../wiki/optimizer-lr/optimizer-lr.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。本实验比较整体策略，策略内可能改变多项配置；不作单参数因果解释。
