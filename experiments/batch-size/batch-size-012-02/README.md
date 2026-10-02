# batch-size-012-02 · 梯度累积代码实现

[Wiki 原位置：第 87 行](../../../wiki/batch-size/batch-size.md) · 作者：（代码段）

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `training.accumulation_normalization` |
| A | `"unscaled"` |
| B | `"sample_weighted"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/batch-size/batch-size-012-02/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/batch-size/batch-size-012-02/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: （代码段）
- **适用场景**: PyTorch 下显存不足时等效扩大 batch size
- **原文**: > 梯度累积可以在不增加内存消耗的情况下模拟大批次训练。通过在多个小批次上累积梯度，然后在达到指定累积次数后进行一次参数更新，可以有效提高训练效率。
- **要点**: 核心实现为 loss = loss / accumulation_steps，每积累 accumulation_steps 次才执行 optimizer.step()，注意 loss 需除以累积步数以保持梯度量级一致。
- **参见**: [[gradient-accumulation]], [[../infra/distributed-training]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
