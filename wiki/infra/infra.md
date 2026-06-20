# 基础设施与工程

## 概述

工程化的训练流程是深度学习项目从"跑通"到"可靠"的分水岭。本章涵盖代码模块化、配置管理、分布式训练、混合精度、实验追踪、超参搜索等基础设施层面的工程技巧，帮助构建可复现、可扩展、可维护的训练体系。

## 详细知识

- [[mixed-precision]]: 使用FP16/BF16混合精度训练降低显存消耗和计算时间，BF16优先于FP16（硬件支持时）。
- [[distributed-training]]: 通过数据并行（DDP）或模型并行将训练分布到多卡多机，DeepSpeed和Megatron-LM是大模型主流方案。
- [[gradient-checkpointing]]: 以时间换空间，前向传播时不保存中间激活，反向传播时重新计算，显著降低长序列和大模型训练的显存占用。
- [[wandb-sweeps]]: W&B Sweeps支持grid/random/bayesian超参自动搜索，可并行调参并多人协作共享实验结果。

## 核心技巧

### 混合精度训练（FP16 / BF16）

- **来源**: LindgeWAI、知乎社区（匿名）
- **适用场景**: 大模型微调与大规模训练，显存受限时
- **原文**: "混合精度训练可以在保持模型性能的同时减少内存消耗和计算时间。通过在训练过程中使用半精度浮点数（FP16）和全精度浮点数（FP32）的混合，可以有效提高训练效率。" / "bf16 优先于 fp16，前提是硬件支持。fp16 下 loss scale、溢出问题有时会让你误判参数效果。"
- **要点**: BF16 优先于 FP16（前提是硬件支持）；FP16 需注意 loss scaling 和溢出问题；使用 `apex.amp` 或 PyTorch 原生 AMP 实现；常用 opt_level='O1'。
- **参见**: [[../batch-size/batch-size]], [[../optimizer-lr/optimizer-lr]]

### 分布式训练

- **来源**: LindgeWAI
- **适用场景**: 多卡多机大规模训练加速
- **原文**: "分布式训练可以将训练任务分布到多个计算设备上，从而加速训练过程。通过使用数据并行或模型并行的方式，可以有效提高训练效率。"
- **要点**: 使用 `torch.distributed.init_process_group(backend, rank, world_size)` 初始化；数据并行（DataParallel/DDP）适合单机多卡，模型并行（Model Parallel）适合超大模型；大模型场景下 DeepSpeed、Megatron-LM 是主流方案。
- **参见**: [[../batch-size/batch-size]]

### 梯度检查点（Gradient Checkpointing）

- **来源**: 爱睡觉的KKY
- **适用场景**: 显存不足时的训练优化
- **原文**: "显存不够用的时候，gradient checkpointing可以起到降低显存的效果"
- **要点**: 以时间换空间，在前向传播时不保存中间激活值，反向传播时重新计算，可显著降低显存占用，适合长序列和大模型微调场景。
- **参见**: [[../batch-size/batch-size]]

### W&B Sweeps 超参搜索

- **来源**: Tang AI、知乎社区（匿名）
- **适用场景**: 自动化超参数搜索与实验记录
- **原文**: "wandb 一款可以大幅提升深度学习效率的神器。可以但不限于用来记录训练测试等深度学习项目开发过程。" / "Weights & Biases 的官方教程里有不少实用的实验管理案例；Optuna 文档里的 pruning 和 sampler 部分值得看"
- **要点**: W&B Sweeps 支持 grid/random/bayesian 搜索，自动并行调参；调用 `wandb.sweep(sweep_config)` + `wandb.agent(sweep_id, function=main)` 即可启动；支持分布式搜索，可多人协作共享实验视图。
- **参见**: [[../batch-size/batch-size]]

### 实验追踪与配置管理

- **来源**: 知乎社区（匿名）
- **适用场景**: 实验管理、结果复现
- **原文**: "调参记录必须严格。每次实验至少记录代码版本、数据版本、配置、随机种子、硬件、训练时间、最优 checkpoint、评估结果、错误样例。别相信自己的记忆。" / "Hydra 或 OmegaConf 用来管理配置也很重要，不然实验一多，你自己都不知道哪组参数跑出了结果。"
- **要点**: 每次实验必须记录代码版本+数据版本+配置+种子+硬件+checkpoint+评估结果；使用 W&B / MLflow 追踪实验；使用 Hydra / OmegaConf 管理配置；模型/数据/参数三者分离（数据路径存 txt，参数配置存 yaml）。
- **参见**: [[../data-eval/data-eval]]

### Code Pipeline 模块化架构

- **来源**: 陈城南
- **适用场景**: 工程化训练代码组织
- **原文**: "一般cv模型的代码都是由这几个模块构成，Data(dataset/dataloader), Model本身，optimizer优化器，loss functions，再加一个 trainer（其实就是组织整个框架的训练顺序）"
- **要点**: 标准 pipeline 分为 Data → Model → Optimizer → Loss → Trainer 五大模块；代码应模块化以方便维护和复用；不必从零写起，基于开源 baseline 修改，但要确保修改部分 bug-free。
- **参见**: [[../data-eval/data-eval]]

### Runner 类训练框架

- **来源**: LindgeWAI
- **适用场景**: 标准化训练流程封装
- **原文**: "可以将机器学习模型的基本要素封装成一个 Runner 类。除上述提到的要素外，再加上模型保存、模型加载等功能。"
- **要点**: Runner 类通常包含 `__init__`（模型/优化器/loss/指标）、`train`（训练循环）、`evaluate`（验证）、`predict`（预测）、`save_model`（保存）、`load_model`（加载）六个核心方法。训练过程分为四阶段：初始化 → 训练 → 评价 → 预测。
- **参见**: [[../data-eval/eval-metrics]]

### 分桶策略（Bucketing）

- **来源**: LindgeWAI
- **适用场景**: 序列任务训练加速，减少 padding 浪费
- **原文**: "分桶策略可以根据输入序列的长度将数据分成不同的桶，从而在训练过程中更高效地利用计算资源。这种方法可以减少 padding 的数量，提高训练效率。"
- **要点**: 按序列长度将数据分到不同桶（如 [10, 20, 30, 40, 50]），每个桶内 padding 到该桶最大长度；减少无效 padding 计算，大幅提升 Transformer/RNN 训练效率；TensorFlow 通过 `bucket_by_sequence_length` 实现，PyTorch 需自定义 collate_fn。
- **参见**: [[../batch-size/batch-size]]

### GPU 内存优化：num_workers 与 pin_memory

- **来源**: LindgeWAI
- **适用场景**: DataLoader 性能调优
- **原文**: "当使用 torch.utils.data.DataLoader 时，设置 num_workers > 0，而不是默认值0，同时设置 pin_memory=True，而不是默认值False"
- **要点**: `num_workers > 0` 通过多进程数据加载避免 GPU 空闲等待；`pin_memory=True` 锁页内存加速 CPU→GPU 传输；两者结合可显著提升数据吞吐。
- **参见**: [[../batch-size/batch-size]]

### torch.tensor() vs torch.as_tensor() vs torch.from_numpy()

- **来源**: LindgeWAI
- **适用场景**: 避免不必要的 tensor 内存拷贝
- **原文**: "torch.tensor() 总是会复制数据，如果你要转换一个numpy数组，使用 torch.as_tensor() 或 torch.from_numpy() 来避免复制数据"
- **要点**: `torch.tensor()` 总会复制数据；`torch.as_tensor()` 和 `torch.from_numpy()` 尽可能共享内存（零拷贝）；涉及 numpy→tensor 转换时优先使用后两者以节省内存和提升速度。
- **参见**: [[../data-eval/data-eval]]

### 随机种子设定

- **来源**: LindgeWAI、爱睡觉的KKY、知乎社区（匿名）
- **适用场景**: 实验可复现性
- **原文**: "万能随机种子：42、3407" / "随机数种子设定好，否则很多对比实验结论不一定准确" / "随机种子不固定。深度学习有随机性，尤其小数据任务。没有 seed，不记录环境，结果就很难复现。"
- **要点**: 固定 seed（如 42、3407）确保结果可复现；小数据任务至少换两个种子跑验证，仅提升 0.1 但方差 0.3 的提升不可靠；固定 seed 不保证完全确定，但避免实验变记忆游戏。
- **参见**: [[../data-eval/data-eval]]

### 梯度累积（Gradient Accumulation）

- **来源**: LindgeWAI、知乎社区（匿名）
- **适用场景**: 显存不足时模拟大批次训练
- **原文**: "梯度累积可以在不增加内存消耗的情况下模拟大批次训练。通过在多个小批次上累积梯度，然后在达到指定累积次数后进行一次参数更新，可以有效提高训练效率。"
- **要点**: `loss = loss / accumulation_steps` → `loss.backward()` → 每 accum_steps 步后 `optimizer.step()` + `zero_grad()`；等效扩大 batch size；调参时需记录清楚"单卡 batch × 卡数 × 累积步数 = 总有效 batch"。
- **参见**: [[../batch-size/batch-size]]

### 模型 Checkpoint 保存与加载

- **来源**: 陈城南、LindgeWAI
- **适用场景**: 训练中断恢复、模型版本管理
- **原文**: "最终模型我们可以使用中间的 checkpoint" / "一定记得取不同训练步骤的 checkpoint 针对某些问题进行检验" / "最优 checkpoint、评估结果、错误样例"
- **要点**: Runner 类中 `save_model` 保存模型参数，`load_model` 加载；在验证指标达到最优时保存 checkpoint；大模型训练时保留多个中间 checkpoint 便于回溯；记录 checkpoint 对应的代码版本、数据版本和配置。
- **参见**: [[infra]]

### 超参数搜索工具选型

- **来源**: 知乎社区（匿名）
- **适用场景**: 自动化超参搜索
- **原文**: "如果团队有条件，上 Optuna、Ray Tune、Weights & Biases Sweeps、MLflow、Ax 都可以。Optuna 对中小团队很友好，接入成本不高。Ray Tune 适合集群资源多的情况。W&B 的记录和可视化做得顺手。MLflow 在企业内部环境更容易被接受。"
- **要点**: 中小团队首选 Optuna（轻量友好）；集群资源多选 Ray Tune；协作实验选 W&B Sweeps；企业内部选 MLflow；NNI（微软）也值得尝试。随机搜索优于网格搜索——真正敏感参数通常只有少数几个。
- **参见**: [[../batch-size/batch-size]], [[../optimizer-lr/optimizer-lr]]

### 计算最优训练：学习率与 Batch Size 的 Scaling Law

- **来源**: Sam聊算法
- **适用场景**: 大模型预训练超参缩放
- **原文**: "给定计算成本 C，大多数超参的最优值是基本固定的，只需调整学习率和 batch size，而学习率 eta 的最优值随 C 增加而下降，batch size 的最优值随 C 增加而上升，满足幂律：eta_opt = 0.3118 * C^(-0.1250), B_opt = 0.2920 * C^(0.3271)"
- **要点**: 来自 DeepSeek LLM 技术报告；计算预算增大时应扩大 batch size 并降低学习率；其他超参（beta、weight_decay、梯度裁剪）取固定最优值即可；公式提供理论锚点，实际仍需网格搜索微调。
- **参见**: [[../batch-size/batch-size]], [[../optimizer-lr/optimizer-lr]]

### 小样本"过拟合"验证法

- **来源**: Captain Jack、陈城南、黑马程序员Python、知乎社区
- **适用场景**: 验证代码正确性与模型容量
- **原文**: "刚开始，先上小规模数据，模型往大了放，只要不爆显存，能用256个filter你就别用128个。直接奔着过拟合去。" / "先小样本过拟合，确认代码和数据没问题" / "先overfit 再trade off，首先保证你的模型 capacity 能够过拟合"
- **要点**: 用小数据+大模型验证训练流程是否正确：若无法过拟合则检查代码bug（数据加载、loss 计算、参数更新）而非调参。这是资深工程师与新人最大的区别——先确保 pipeline 正确，再谈优化。
- **参见**: [[../debugging/debugging]]

### Warmup + Cosine 退火学习率调度

- **来源**: LindgeWAI、陈城南、知乎社区
- **适用场景**: 几乎所有 Transformer 和大模型训练
- **原文**: "warmup 我现在几乎默认会开，尤其是 Transformer。比例一般 3% 到 10% 之间试。scheduler 常用 cosine 或 linear decay" / "万能模板 Adam + Cos退火构建基础"
- **要点**: warmup 步数一般设总轮次 1/10（参考 BERT）；cosine 退火从初始 lr 平滑降到极小值（如 1e-7）；warmup + cosine 组合是 Transformer 和大模型训练的事实标准。
- **参见**: [[../optimizer-lr/optimizer-lr]]
