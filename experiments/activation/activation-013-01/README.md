# activation-013-01 · ReLU 激活函数推荐搭配 He 初始化

[Wiki 原位置：第 99 行](../../../wiki/activation/activation.md) · 作者：王惠东（京东）、多位答主

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `model.weight_init` |
| A | `"xavier_normal"` |
| B | `"kaiming_normal"` |
| 任务配置 | `recommendation` |
| 主指标 | `test_ndcg_at_10`（max） |
| 固定条件 | `{"model.activation": "relu"}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/activation/activation-013-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/activation/activation-013-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: 王惠东（京东）、多位答主
- **适用场景**: 权重初始化策略
- **原文**: "relu激活函数初始化推荐使用He normal，tanh初始化推荐使用Glorot normal" / "Xavier初始化适用于常见的激活函数（如tanh和sigmoid），而He初始化适用于ReLU激活函数" / "在使用ReLU激活函数时，通常推荐使用这种高斯初始化，通过缩放方差为2/n，以更好地适应ReLU的性质"
- **要点**: ReLU → He normal（方差 2/n）；tanh/sigmoid → Xavier/Glorot normal。激活函数与初始化方法必须匹配，错误搭配会导致梯度爆炸或消失。
- **参见**: [relu](../../../wiki/activation/relu.md), [../initialization/initialization](../../../wiki/initialization/initialization.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
