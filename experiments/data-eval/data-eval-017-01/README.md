# data-eval-017-01 · 数据也是超参数——采样比、增强强度、混合比例

[Wiki 原位置：第 129 行](../../../wiki/data-eval/data-eval.md) · 作者：BugBuster喵

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `data.class_sampling` |
| A | `"natural"` |
| B | `"balanced"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/data-eval/data-eval-017-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/data-eval/data-eval-017-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: BugBuster喵
- **适用场景**: 分类、检测、推荐、大模型SFT
- **原文**: "我越来越倾向于把数据也当成超参数。比如分类任务里，正负样本比例就是超参数。推荐任务里，负采样策略就是超参数。检测任务里，resize 尺寸和增强强度就是超参数。大模型 SFT 里，单轮和多轮数据比例就是超参数。"
- **要点**: 数据配置（采样比例、增强策略、混合比例、截断策略、模板格式）对效果的影响往往大于模型结构参数。应有意识地对数据参数进行调优。
- **参见**: [[hyperparameter-tuning]], [[data-augmentation]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
