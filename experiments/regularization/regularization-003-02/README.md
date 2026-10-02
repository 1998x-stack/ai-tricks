# regularization-003-02 · Weight Decay（L2 正则化）

[Wiki 原位置：第 32 行](../../../wiki/regularization/regularization.md) · 作者：阿里云云栖号 / 爱睡觉的KKY / 四点半的理一 / BugBuster喵 / Sam聊算法 / 匿名 / 苏辄

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `optimizer.weight_decay` |
| A | `0.01` |
| B | `0.001` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/regularization/regularization-003-02/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/regularization/regularization-003-02/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: 阿里云云栖号 / 爱睡觉的KKY / 四点半的理一 / BugBuster喵 / Sam聊算法 / 匿名 / 苏辄
- **适用场景**: 所有模型，Transformer与ResNet对weight decay敏感度差异大
- **原文**: "可以先从0.01开始，如果模型不稳定，往0.001甚至0.0001调" ——四点半的理一
- **要点**: 常用范围1e-4~0.1。ResNet类相对不敏感，Transformer类大batch时weight decay设大会导致不收敛。小数据集和浅层结构需更大weight decay（1e-3~1e-2），大数据和深层结构设小（1e-4~1e-6）。AdamW保留weight decay效果，提升泛化。建议先0.01试，不稳定则调小。
- **参见**: [overfitting-underfitting](../../../wiki/regularization/overfitting-underfitting.md), [../optimizer-lr/adam](../../../wiki/optimizer-lr/adam.md), [../optimizer-lr/learning-rate-scheduling](../../../wiki/optimizer-lr/learning-rate-scheduling.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
