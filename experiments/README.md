# AI 技巧 → 真实数据 A/B 实验

本目录是**进行中的实现**，尚未完成“所有技巧的全部真实训练”。没有在本机下载 Hugging Face 数据或模型，也没有把测试夹具的日志冒充训练结果。

当前状态：

- 14 类、412 个带作者归因的 Wiki 主条目，拆出 **517 个具体 A/B 协议**。
- 每个协议有 **134 行 Python 编排入口**、相邻 JSON 配置、中文说明、Wiki 路径和行号。入口由模板生成，技巧实现通过公共驱动复用；不能把 517 个入口等同于 517 个已完成的专用训练实现。
- **180 个对照通过视觉/SFT 基础驱动配置兼容检查，具体场景仍待复核，均未经过远程 GPU 完整训练验证**；**337 个对照仍需专用驱动**。详见 [逐项驱动状态](catalog/DRIVER_STATUS.md)。
- 58 个知识详情页已完整保留在目录数据中，尚未逐操作审计；原文有 351 段未与引文直接匹配的候选，包含重复、代码和抓取噪声。不能据此声称原始素材已无遗漏。
- 真实训练完成数：**0**。尚缺远程主机、GPU 资源和时间预算；当前不会自动租用 GPU、启动付费任务或在本机运行训练。

## 分类入口

| 类别 | 索引 |
|---|---|
| 优化器与学习率 | [89 个主条目](optimizer-lr/README.md) |
| 批大小与梯度累积 | [19 个主条目](batch-size/README.md) |
| 初始化 | [15 个主条目](initialization/README.md) |
| 归一化 | [15 个主条目](normalization/README.md) |
| 激活函数 | [16 个主条目](activation/README.md) |
| 正则化 | [16 个主条目](regularization/README.md) |
| CNN / CV | [68 个主条目](cnn-cv/README.md) |
| NLP | [16 个主条目](nlp/README.md) |
| 推荐系统 | [23 个主条目](recsys/README.md) |
| SFT | [22 个主条目](llm-sft/README.md) |
| RL 对齐 | [31 个主条目](llm-rl/README.md) |
| 工程基础设施 | [17 个主条目](infra/README.md) |
| 数据与评估 | [30 个主条目](data-eval/README.md) |
| 调试 | [35 个主条目](debugging/README.md) |

## 本机可运行：只看计划、校验代码

```bash
python3 experiments/activation/activation-001-01/experiment.py --plan
python3 -m experiments.tools.audit
python3 -m unittest discover -s tests -v
```

计划命令只依赖 Python 标准库，不读取 HF 数据，也不加载模型。统计和绘图测试需要 NumPy、Matplotlib；测试夹具只在临时目录中生成并清除，不能作为实验结论。

## 远程完整训练

1. 在远程 Linux GPU 主机检出此仓库，安装 `experiments/requirements.txt`。
2. 复制 [remote.example.json](configs/remote.example.json) 为远程环境配置。填入 `allowed_hostnames`、明确的时间上限、完整训练步数上限，并设置 `remote_training_confirmed=true`。示例故意保留空主机名单及零预算，不代表已经获得资源。
3. 确认所选实验的任务、模型、数据字段、切分和驱动相符。不能把默认视觉 profile 用于检测、推荐或 PPO；未实现的驱动必须先补齐。
4. 在该远程主机执行，结果目录必须是全新目录：

```bash
python3 experiments/activation/activation-001-01/experiment.py \
  --run --environment remote.json --output /workspace/runs/relu-ab
```

远程数据采用 Hugging Face `streaming=True`。远程模型权重、HF 元数据缓存及 checkpoint 可以落盘；**本机禁下载不等于远程零缓存**。固定完整 revision，官方 train/test 再用稳定样本序号做互斥验证切分。主机检查先于 `datasets`/`transformers` 调用，直接调用驱动也会检查。

训练不是“smoke test”：视觉默认完整 CIFAR-10 训练分区，SFT 默认完整 `train_sft`，没有 `.take(32)` 或合成数据兜底。训练轮数与预算需按远程资源确认。预算不足会失败并保留日志，不生成成功报告。

## 实验约束与输出

每个实验默认三个配对种子 `17, 42, 2026`，交替 A/B 顺序。固定数据/模型版本、切分和评测代码；只改变协议中的干预。策略类实验单独标为 `policy`，不能把整体策略效果解释成某一参数的效果。

视觉训练按样本数安排验证，防止 batch 大小变化同时改变验证频率。完整 epoch 预算优先，`max_steps` 是安全上限。早停 patience 的单位为验证次数。SFT 按监督 token 数归一化梯度，固定从左截断以保留最终回答。模型变化、数据过滤和不同训练轮数的成本必须单独报告。

每个 seed/arm 输出：

- `request.json`：实际配置、种子、模型与数据版本、计划哈希。
- `train.log`、`history.csv`：真实 stdout/stderr 与训练/验证历史。
- `predictions.jsonl`：最终测试样本 ID 与预测；A/B 必须使用同一组测试 ID。
- `checkpoint_manifest.json`：仅按验证集选择的 checkpoint 及文件哈希。
- `result.json`、`verified_result.json`：原始和通过契约校验的结果。
- 失败时写 `failure.json`，不丢弃失败种子、不补造成功指标。

全组完成后生成 `paired_metrics.csv`、`summary.json`、`comparison.png`、`comparison.svg`、`REPORT.md`。bootstrap 单位是 seed 对，保留负结果；三种子区间只作描述，不作多重比较下的显著性证明。

视觉主指标是准确率；SFT 默认报告 ROUGE-L、exact match 和验证 NLL。开放对话的字面重叠**不能证明正确性或人类偏好提升**，最终业务质量仍需任务专用评测。吞吐目前包括训练期间验证开销，显存记录训练阶段峰值。

## 可追溯性与继续开发

- [主条目完整内容与作者](catalog/tricks.json)
- [517 个实验协议](catalog/experiments.json)
- [覆盖审计](catalog/coverage.json)、[代码审计](catalog/code_audit.json)
- [保留的知识详情页](catalog/detail_pages.json)、[原文待核对候选](catalog/raw_unmatched_candidates.json)
- [HF 数据与模型版本](DATASETS.md)、[需要纠偏的原文断言](ERRATA.md)
- [训练驱动接口](DRIVER_CONTRACT.md)

协议以 [tools/protocols.py](tools/protocols.py) 中逐项定义为准，不能靠关键词自动替换实验行为。修改后运行：

```bash
python3 -m experiments.tools.build_catalog
python3 -m experiments.tools.make_remote_config
python3 -m experiments.tools.audit
```

Wiki 内容发生变化时来源哈希会使旧协议校验失败，必须重建并复核。生成器目前仍以主页面的条目顺序编号；移动/插入条目须同时审查协议 ID 对应关系，禁止未经核对直接重编号历史结果。

## 校订与稳定编号

[本轮审查记录](REVIEW.md)列出已修复问题、回归证据和仍未完成的范围。`tools/source_ids.json` 固定来源文件与标题的 ID，移动条目不会重新分配旧实验；新增或改名需显式更新映射，并核对协议语义。生成入口统一记录 running / failed / complete 状态。
