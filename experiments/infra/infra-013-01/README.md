# infra-013-01 · 模型 Checkpoint 保存与加载

[Wiki 原位置：第 112 行](../../../wiki/infra/infra.md) · 作者：陈城南、LindgeWAI

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `checkpoint.restore` |
| A | `"weights_only"` |
| B | `"full_training_state"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/infra/infra-013-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/infra/infra-013-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: 陈城南、LindgeWAI
- **适用场景**: 训练中断恢复、模型版本管理
- **原文**: "最终模型我们可以使用中间的 checkpoint" / "一定记得取不同训练步骤的 checkpoint 针对某些问题进行检验" / "最优 checkpoint、评估结果、错误样例"
- **要点**: Runner 类中 `save_model` 保存模型参数，`load_model` 加载；在验证指标达到最优时保存 checkpoint；大模型训练时保留多个中间 checkpoint 便于回溯；记录 checkpoint 对应的代码版本、数据版本和配置。
- **参见**: [infra](../../../wiki/infra/infra.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
