# W&B Sweeps 超参搜索

## 是什么

W&B Sweeps 是 Weights & Biases 平台提供的自动化超参数搜索工具。用户定义搜索空间和策略后，Sweeps 自动启动多个 agent 在并行 worker 中尝试不同的超参数组合，记录每次实验的完整指标（loss、accuracy、学习率曲线等），并在 W&B Dashboard 上可视化对比。

核心流程：定义 sweep 配置（YAML 或 Python dict） → 创建 sweep → 启动 agent(s) → 自动搜索并记录结果。

```python
sweep_config = {
    "method": "random",                         # 搜索策略
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [16, 32, 64]},
        "dropout": {"distribution": "uniform", "min": 0.1, "max": 0.5},
    }
}

sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, function=train_main, count=20)
```

## 为什么重要

- **自动化调参**：手动调参容易陷入局部最优、遗漏参数组合，且难以系统性地探索高维空间。Sweeps 将这一过程自动化，且能在多 GPU/多机器上并行搜索。
- **可复现与可追溯**：每次搜索的参数组合、代码版本、运行环境、完整指标都记录在 W&B，随时可以回溯"哪个配置跑出最好结果"。
- **多人协作**：团队成员可以共享同一个 sweep，各自启动 agent 贡献搜索结果，所有结果自动汇总到同一个 Dashboard。

## 如何使用

### 搜索策略选择

| 方法 | 原理 | 推荐场景 |
|------|------|----------|
| **Grid Search** | 穷举参数网格中所有组合 | 参数少（< 3 个）、每个参数取值少 |
| **Random Search** | 在搜索空间随机采样 | 通用首选，高维空间比 Grid 高效 |
| **Bayesian Search** | 用历史结果构建概率模型，引导搜索向最优区域收敛 | 搜索预算有限、需要尽快找到好参数 |

**实用建议**：默认使用 Random Search。真正敏感的参数通常只有 2-3 个，随机搜索在高维空间中比网格搜索采样效率高得多。Bayesian 搜索在搜索次数较少（< 50 次）时表现好，但每次决策有额外计算开销。

### Sweep 配置实战

```yaml
# sweep.yaml
program: train.py
method: bayesian
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform
    min: 0.00001
    max: 0.01
  optimizer:
    values: ["adam", "sgd"]
  num_layers:
    values: [2, 4, 6]
  dropout:
    distribution: uniform
    min: 0.0
    max: 0.5
```

启动方式：
```bash
wandb sweep sweep.yaml
wandb agent <sweep_id>
```

### Agent 管理

- **单机单 agent**：`wandb agent sweep_id` —— 依次执行搜索任务。
- **单机多 agent**：开多个终端窗口分别运行 `wandb agent sweep_id` —— 并行搜索加速。
- **多机多 agent**：团队成员或集群节点各自运行同一个 sweep_id 的 `wandb.agent()` —— 自动协调任务分配。

### 最佳实践

1. **先粗后细**：第一次搜索用 Random 方法，范围设宽（学习率 1e-5 到 1e-1），跑少量次数（10-20）找到大致好的区域。然后缩小范围，用 Bayesian 精搜。
2. **固定不敏感参数**：momentum、weight_decay 等参数可以固定为经验值，不纳入搜索空间，减少维度。
3. **early termination**：结合早停（early stopping）或 W&B 的 `wandb.config` 记录中间指标，避免为明显不好的配置浪费计算资源。
4. **记录关键元数据**：在训练脚本中通过 `wandb.config.update()` 记录 seed、数据版本、模型结构等，方便复现最佳配置。

## 常见误区

1. **Grid Search 最保险**：网格搜索在 3 个以上参数时组合数爆炸（如 5 × 5 × 5 × 5 = 625 组）。实际项目中 80% 的情况下 Random Search 能更快找到好参数。

2. **Bayesian 搜索永远最好**：贝叶斯优化的计算开销随搜索次数线性增长（高斯过程拟合 O(n³)），在小预算（< 20 次）时随机搜索经常表现更好。且贝叶斯方法对噪声敏感——如果你的验证指标波动大，它可能误判参数区域的好坏。

3. **Sweep 和本地训练互不影响**：多个 agent 共享同一个 sweep 时，需要确保实验之间不会冲突（如写同一个日志文件、占用同一个端口）。建议每个 agent 使用独立的输出目录。

4. **最佳参数可直接用于生产**：Sweep 找到的最佳参数是在验证集上选取的，可能轻微过拟合验证集。最终方案应在独立测试集上验证，并确认不同随机种子下的稳定性。

## 参见

- [[../batch-size/gradient-accumulation]] —— batch size 是 sweeps 中重要的搜索参数之一
- [[../optimizer-lr/optimizer-lr]] —— 学习率是最常搜索的超参数，与优化器选择紧密耦合
- [[../../contradictory]] —— 知识矛盾：社区对 grid vs random vs bayesian 的优先级存在不同主张
