# infra-005-01 · 实验追踪与配置管理

[Wiki 原位置：第 48 行](../../../wiki/infra/infra.md) · 作者：知乎社区（匿名）

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `tracking.level` |
| A | `"metrics_only"` |
| B | `"full_provenance"` |
| 任务配置 | `vision` |
| 主指标 | `diagnostic_fault_recall`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/infra/infra-005-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/infra/infra-005-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: 知乎社区（匿名）
- **适用场景**: 实验管理、结果复现
- **原文**: "调参记录必须严格。每次实验至少记录代码版本、数据版本、配置、随机种子、硬件、训练时间、最优 checkpoint、评估结果、错误样例。别相信自己的记忆。" / "Hydra 或 OmegaConf 用来管理配置也很重要，不然实验一多，你自己都不知道哪组参数跑出了结果。"
- **要点**: 每次实验必须记录代码版本+数据版本+配置+种子+硬件+checkpoint+评估结果；使用 W&B / MLflow 追踪实验；使用 Hydra / OmegaConf 管理配置；模型/数据/参数三者分离（数据路径存 txt，参数配置存 yaml）。
- **参见**: [[../data-eval/data-eval]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
