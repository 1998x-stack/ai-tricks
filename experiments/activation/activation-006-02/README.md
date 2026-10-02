# activation-006-02 · LeakyReLU / PReLU 应对 ReLU 神经元死亡

[Wiki 原位置：第 50 行](../../../wiki/activation/activation.md) · 作者：爱睡觉的KKY、多位 CNN 从业者

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `model.activation` |
| A | `"relu"` |
| B | `"prelu"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/activation/activation-006-02/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/activation/activation-006-02/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: 爱睡觉的KKY、多位 CNN 从业者
- **适用场景**: 当 ReLU 出现神经元死亡（dying ReLU）问题时
- **原文**: "对于激活函数，relu也在大多数情况下表现不错，当然也可以试试leakyrelu和prelu，prelu貌似效果还不错" / "如果还想再提升精度可以将ReLU改成PReLU试试" / "为了解决 ReLU 激活函数中的梯度消失问题，也可以更换成Leaky ReLU激活函数等其他变体"
- **要点**: ReLU 的负半轴恒为 0 可能导致神经元永久失活。LeakyReLU 引入小斜率、PReLU 引入可学习斜率，是 dying ReLU 的标准解法。
- **参见**: [leaky-relu-prelu](../../../wiki/activation/leaky-relu-prelu.md), [../regularization/regularization](../../../wiki/regularization/regularization.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
