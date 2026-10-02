# recsys-017-01 · 模型构建顺序：先不加激活函数、BN、Dropout

[Wiki 原位置：第 128 行](../../../wiki/recsys/recsys.md) · 作者：Tang AI

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `model.activation` |
| A | `"relu"` |
| B | `"identity"` |
| 任务配置 | `recommendation` |
| 主指标 | `test_ndcg_at_10`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/recsys/recsys-017-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/recsys/recsys-017-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Tang AI
- **适用场景**: 推荐模型初次搭建
- **原文**: > 做新模型的时候，最开始不要加激活函数，不要加batchnorm，不要加dropout，先就纯模型。然后再一步一步的实验，不要过于信赖经典的模型结构(除非它是预训练的)，比如加了dropout一定会有效果，或者加了batchnorm一定会有提升所以先加上，首先你要确定你的模型处于什么阶段，到底是欠拟合还是过拟合，然后再确定解决手段。
- **要点**: 新模型应先裸跑确认 baseline，逐步依据过拟合 / 欠拟合状态添加 Dropout、BN、激活函数，而非一开始全加上。
- **参见**: `[[model-debug]]`（原 Wiki 未解析链接）

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
