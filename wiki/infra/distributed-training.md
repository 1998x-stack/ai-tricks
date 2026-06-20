# 分布式训练

## 是什么

分布式训练（Distributed Training）是将深度学习训练任务分布到多个计算设备（GPU）或节点上并行执行的技术。根据并行策略的不同，主要分为三类：

- **数据并行（Data Parallelism）**：每个设备持有完整的模型副本，但只处理一部分数据，前向与反向计算后通过 all-reduce 通信同步梯度。最常用，实现简单。
- **模型并行（Model Parallelism）**：将模型的不同层（或同一层的参数分片）放在不同设备上，数据依次流过各设备。适合单卡放不下的超大模型。
- **流水线并行（Pipeline Parallelism）**：模型按层切成多个 stage，每个 stage 放在不同设备，微批次交替流过各 stage 以提高设备利用率。是模型并行的进阶版。

PyTorch 通过 `torch.distributed` 包实现分布式通信，核心函数为 `init_process_group`。

## 为什么重要

- **突破单卡显存上限**：大模型（> 7B 参数）单卡无法加载，必须通过模型并行或流水线并行跨卡部署。
- **线性或近线性加速**：数据并行下，理想情况 8 卡可获得接近 8 倍训练吞吐量，通信开销随模型大小和数据量缩放。
- **缩短实验周期**：将数周的训练压缩到数天甚至数小时，是快速迭代的前提。

## 如何使用

### 基础初始化（DDP）

```python
import torch.distributed as dist
import torch.multiprocessing as mp

def train(rank, world_size):
    dist.init_process_group(
        backend="nccl",          # NVIDIA GPU 用 nccl
        init_method="tcp://...", # 或 env://
        rank=rank,
        world_size=world_size
    )
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    # 标准训练循环...
```

启动命令：
```bash
torchrun --nproc_per_node=4 train.py
# 或
python -m torch.distributed.run --nproc_per_node=4 train.py
```

### DataParallel vs DDP

| 特性 | DataParallel | DDP (DistributedDataParallel) |
|------|-------------|-------------------------------|
| 架构 | 单进程多线程 | 多进程（每卡一个进程） |
| 通信 | Python GIL 受限 | C10D 后端，无 GIL 问题 |
| 速度 | 慢（主卡瓶颈） | 快（近线性加速） |
| 推荐 | 仅原型验证 | 生产环境首选 |

**黄金规则**：永远不要在生产环境使用 `DataParallel`。即使是单机多卡，也应使用 `DDP`。

### DeepSpeed / Megatron 适用场景

- **DeepSpeed**：ZeRO 优化（内存分片）将优化器状态、梯度、参数分布到各卡上，使数据并行也能训练超出单卡显存的模型。ZeRO Stage 2 是常见配置，Stage 3 进一步分片参数。适合 1B-100B 参数级别的模型。
- **Megatron-LM**：专为大语言模型设计的模型并行 + 流水线并行 + 数据并行组合方案。适合 10B+ 参数的 Transformer 模型。
- **FSDP（Fully Sharded Data Parallel）**：PyTorch 原生实现类 ZeRO Stage 3 的方案，接口与 DDP 类似，社区采用率逐步上升。

### 关键建议

- **从 DDP 开始**：大多数场景（单机多卡、中等模型）DDP 足够了。不要过早引入 DeepSpeed/Megatron。
- **当单卡连一个 micro batch 都放不下时**：意味着需要模型并行、流水线并行或 ZeRO Stage 3。
- **通信瓶颈**：多节点训练时，网络带宽可能成为瓶颈。NVLink（单机内）优于 PCIe，InfiniBand（跨机）优于以太网。

## 常见误区

1. **DDP 自动处理 batch size 缩放**：DDP 不会自动调整学习率。总 batch size 增大时（`单卡 batch × 卡数`），学习率通常需要按比例增大（linear scaling rule）或者配合 warmup 平滑过渡。

2. **DataParallel 也能用**：DataParallel 将模型复制到各卡、收集梯度的过程受 GIL 限制，且主卡显存和计算负载不成比例地高于其他卡。DDP 在实现上几乎处处优于 DP。

3. **分布式训练只解决速度问题**：对于大模型，分布式训练首先解决的是"能不能装下"的问题（通过 ZeRO、模型并行等手段），其次才是"能不能加速"。

4. **nccl 后端是万能选择**：NCCL 确实是 GPU 分布式训练的最佳后端，但在 CPU-only 环境或调试时需要切换为 `gloo`。多机训练时 NCCL 需要正确的网络配置（IP 可达、防火墙放行）。

## 参见

- [[../batch-size/gradient-accumulation]] —— 分布式训练中梯度累积与 all-reduce 通信的交互
- [[../optimizer-lr/optimizer-lr]] —— 分布式训练的 learning rate scaling rule
- [[../../contradictory]] —— 知识矛盾：关于 linear scaling rule 的适用范围存在不同观点
