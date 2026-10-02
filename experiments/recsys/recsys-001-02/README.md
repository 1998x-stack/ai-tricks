# recsys-001-02 · 热启动必须 Embedding + DNN 全部加载

[Wiki 原位置：第 16 行](../../../wiki/recsys/recsys.md) · 作者：Tang AI

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `checkpoint.perturb_scope` |
| A | `"all_layers"` |
| B | `"last_layer"` |
| 任务配置 | `recommendation` |
| 主指标 | `test_ndcg_at_10`（max） |
| 固定条件 | `{}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `single_factor` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/recsys/recsys-001-02/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/recsys/recsys-001-02/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

- **来源**: Tang AI
- **适用场景**: 推荐系统老汤模型迭代
- **原文**: > 对于推荐系统的老汤模型，建议热启动，但是热启动大概率也是追平，小概率找到了之前没有找到的关键特征，可以超越，其他操作笔者的实验都是负向，除了热启动，而且就算热启动，如果只启动embedding层也是不行的，必须embedding+DNN全部热启动，然后仅对倒数第一层作扰动，或者类似wide&deep，新加一个结构对最后一层个性化扰动，为啥最后一层，因为，你扰动太多又负向了。
- **要点**: 只热启 Embedding 层不够，必须 Embedding + DNN 全部热启，且仅对倒数第一层做扰动；对更多层做扰动几乎必定负向。
- **参见**: [[warm-start]]

## 解释边界

真实数据完整训练的待执行协议，未产生训练结论。所列数值是预注册对照，不代表作者经验在所有模型上成立。最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。
