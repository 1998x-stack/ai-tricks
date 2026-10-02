# regularization-012-01 · Ensemble 集成方法

[Wiki 原位置：第 95 行](../../../wiki/regularization/regularization.md) · 作者：BBuf / 匿名 / DOTA / 清明

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `inference.ensemble_members` |
| A | `1` |
| B | `3` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/regularization/regularization-012-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/regularization/regularization-012-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: BBuf / 匿名 / DOTA / 清明
- **适用场景**: 比赛刷分、追求泛化性能上限
- **原文**: "可以使用不同的初始化方式训练出模型，然后做ensemble。可以使用不同超参数(如学习率，batch_size，优化器)训练出不同模型，然后做ensemble。" ——BBuf
- **要点**: 常用方式：(1) 不同初始化训练多个模型做voting/soft-voting；(2) 不同超参数（lr/batch_size/优化器）训练不同模型；(3) 不同网络架构（VGG/ResNet/Xception）提取特征加权组合；(4) Cyclic LR自动收敛到多个局部最小值得到多个模型做集成。Dropout本身是低成本的隐式Ensemble。Inception/Shortcut/Dense Connection也相当于集成。
- **参见**: [dropout](../../../wiki/regularization/dropout.md), [dropout](../../../wiki/regularization/dropout.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
