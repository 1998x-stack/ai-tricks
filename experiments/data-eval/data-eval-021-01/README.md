# data-eval-021-01 · 大模型评测：人工Eval Set + OpenCompass + RAGAS

[Wiki 原位置：第 157 行](../../../wiki/data-eval/data-eval.md) · 作者：BugBuster喵

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `evaluation.protocol` |
| A | `"loss_only"` |
| B | `"frozen_task_suite"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/data-eval/data-eval-021-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/data-eval/data-eval-021-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: BugBuster喵
- **适用场景**: LLM微调、RAG系统、大模型应用
- **原文**: "要做人工标注的 eval set，要覆盖典型问题、边界问题、长尾问题，还要记录失败类型。现在行业里稍微成熟一点的大模型团队，基本都会先搭评测，再谈微调。没有评测的大模型微调，就是在算力上投币。" / "大模型评测方面可以看 OpenCompass、HELM，中文场景再补一下 C-Eval。做 RAG 可以看看 RAGAS。"
- **要点**: 大模型时代不能只看loss或BLEU/ROUGE。需要构建人工标注的评估集，覆盖典型/边界/长尾问题。OpenCompass用于综合评测，RAGAS用于RAG系统评估。
- **参见**: [[llm-evaluation]], [[ragas]], [[opencompass]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
