# data-eval-014-04 · 数据增强：CV用裁剪翻转旋转，NLP用同义词替换和回译

[Wiki 原位置：第 108 行](../../../wiki/data-eval/data-eval.md) · 作者：Tang AI / 无忧无虑 / 黑马程序员

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `data.augmentation` |
| A | `"none"` |
| B | `"back_translation"` |
| 任务配置 | `vision` |
| 主指标 | `test_accuracy`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/data-eval/data-eval-014-04/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/data-eval/data-eval-014-04/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Tang AI / 无忧无虑 / 黑马程序员
- **适用场景**: 数据量不足或过拟合时的通用手段
- **原文**: "数据增强要结合任务本身来设计" / "图像裁剪，对称变换，旋转平移，都可以让模型在验证集上的表现更好。" / "数据越多越好！总是使用数据增强，如水平翻转，旋转，缩放裁剪等。"
- **要点**: CV领域：裁剪、翻转、旋转、色彩抖动、亮度变化、cutmix、mixup。NLP领域：同义词替换、随机丢弃词语、句子改写、回译（back-translation）。增强强度本身也是一个需要调节的超参数。
- **参见**: [[data-augmentation]], [[../regularization/overfitting-underfitting]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
