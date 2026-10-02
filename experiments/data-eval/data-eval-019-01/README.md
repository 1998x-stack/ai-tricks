# data-eval-019-01 · Embedding可视化：先PCA降维再用t-SNE

[Wiki 原位置：第 143 行](../../../wiki/data-eval/data-eval.md) · 作者：Tang AI

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `visualization.embedding` |
| A | `"tsne_raw"` |
| B | `"pca_then_tsne"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/data-eval/data-eval-019-01/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/data-eval/data-eval-019-01/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Tang AI
- **适用场景**: 理解模型学到的表征、检查embedding质量
- **原文**: "对于embedding可视化的问题最好是先用pca降维，然后采用t-sne进行可视化，但是，对于类别很多的情况，个人建议还是肉眼抽样观察，t-sne只能对于类目不多的情况的embedding进行可视化。"
- **要点**: PCA用于初始降维（去噪声），t-SNE进一步降维到2D/3D可视化。t-SNE在类别多时效果不佳，此时应人工抽样观察bad case。
- **参见**: `[[visualization]]`（原 Wiki 未解析链接）, `[[embedding]]`（原 Wiki 未解析链接）

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
