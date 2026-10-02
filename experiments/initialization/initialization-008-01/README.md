# initialization-008-01 · 权重用随机初始化，Bias 恒为零

[Wiki 原位置：第 62 行](../../../wiki/initialization/initialization.md) · 作者：炫光 / 匿名用户

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `model.bias_init` |
| A | `"normal"` |
| B | `"zeros"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/initialization/initialization-008-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/initialization/initialization-008-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: 炫光 / 匿名用户
- **适用场景**: 所有网络层的偏置项初始化
- **原文**: "对于weight的初始化我一般都是使用xavier初始化。当然也可以可以尝试何凯明大神的He初始化。对于bias的初始化全置于0" / "weight用高斯分布初始化，bias全初始化为0"
- **要点**: Bias 始终初始化为 0，不需要随机初始化。权重使用与激活函数匹配的初始化策略（Xavier/He/Kaiming）。

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
