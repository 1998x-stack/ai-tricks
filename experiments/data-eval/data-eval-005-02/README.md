# data-eval-005-02 · 训练/验证/测试集切分方式决定实验可信度

[Wiki 原位置：第 45 行](../../../wiki/data-eval/data-eval.md) · 作者：无忧无虑 / BugBuster喵

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `data.split_policy` |
| A | `"random"` |
| B | `"time"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `policy` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/data-eval/data-eval-005-02/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/data-eval/data-eval-005-02/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: 无忧无虑 / BugBuster喵
- **适用场景**: 所有监督学习任务
- **原文**: "调参请在验证集上！调参请在验证集上！调参请在验证集上！" / "保留一个真正不碰的最终测试集，最好按时间或业务单元切。重要版本才评一次。"
- **要点**: 验证集用于调参，测试集只能用于最终评估。反复在验证集上调参会使其成为训练的一部分（验证集过拟合）。最终测试集必须保持纯净。
- **参见**: [cross-validation](../../../wiki/data-eval/cross-validation.md), [../regularization/overfitting-underfitting](../../../wiki/regularization/overfitting-underfitting.md)

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。本实验比较整体策略，策略内可能改变多项配置；不作单参数因果解释。
