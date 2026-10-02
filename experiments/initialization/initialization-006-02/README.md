# initialization-006-02 · 零初始化的神话与陷阱

[Wiki 原位置：第 49 行](../../../wiki/initialization/initialization.md) · 作者：京东白条

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `model.bias_init` |
| A | `"zeros"` |
| B | `"normal"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/initialization/initialization-006-02/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/initialization/initialization-006-02/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: 京东白条
- **适用场景**: 理解为什么不能全零初始化（bias 除外）
- **原文**: "将权重矩阵W初始化方法改成了全为0的初始化...在训练完第一个epoch后预测精度就达到了85%以上，最终20个epoch后精度达到92%"
- **要点**: 权重的全零初始化在单层 Softmax 回归中可以工作（凸优化），但在多层网络中无法打破隐藏单元的对称性，可能使它们始终学到相同特征。权重用随机初始化，**bias 全初始化为 0**。
- **参见**: `[[optimizer]]`（原 Wiki 未解析链接）

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。零初始化保持隐藏单元对称性，不能解释为打破对称性；线性模型是单独情形。
