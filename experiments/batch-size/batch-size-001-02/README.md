# batch-size-001-02 · batch size 在表示学习/对比学习领域越大越好

[Wiki 原位置：第 15 行](../../../wiki/batch-size/batch-size.md) · 作者：爱睡觉的KKY

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `training.contrastive_negative_pool` |
| A | `"microbatch"` |
| B | `"global_batch"` |
| 任务配置 | `retrieval` |
| 主指标 | `test_recall_at_10`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/batch-size/batch-size-001-02/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/batch-size/batch-size-001-02/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: 爱睡觉的KKY
- **适用场景**: 对比学习、表示学习（如 SimCLR、MoCo、CLIP）
- **原文**: > batch size 在表示学习，对比学习领域一般越大越好，显存不够上累计梯度，否则模型可能不收敛… 其他领域看情况；
- **要点**: 对使用 in-batch negatives 的对比学习，负样本池大小会改变损失。普通梯度累积只累加各 microbatch 的梯度，不自动增加 softmax 分母中的负样本；如需大负样本池，须额外使用跨批表示缓存/重算或全局表示收集。参考 [Gao 等的梯度缓存方法](https://arxiv.org/abs/2101.06983)，不能把其效果归于普通累积。
- **参见**: [batch-size-generalization](../../../wiki/batch-size/batch-size-generalization.md), [gradient-accumulation](../../../wiki/batch-size/gradient-accumulation.md), [../optimizer-lr/optimizer-lr](../../../wiki/optimizer-lr/optimizer-lr.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。梯度累积不会自动扩大 InfoNCE 单次 softmax 的负样本池；单独对照 microbatch/global_batch。
