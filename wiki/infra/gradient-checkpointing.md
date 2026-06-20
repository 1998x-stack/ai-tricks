# 梯度检查点

## 是什么

梯度检查点（Gradient Checkpointing，又称 Activation Checkpointing）是一种以计算换显存的技术。标准训练中，前向传播会保存所有中间激活值（activations）供反向传播计算梯度使用。梯度检查点的做法是：前向传播时只保存少数关键节点（checkpoint）的激活值，丢弃其余中间结果；反向传播时重新计算这些被丢弃的激活值。

核心公式：**显存节省 O(√n) 到 O(n)，额外计算开销约 1 次额外前向传播**。其中 n 是网络层数。具体节省量取决于 checkpoint 的粒度。

在 PyTorch 中通过 `torch.utils.checkpoint.checkpoint` 实现：

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.block1, x)   # block1 的中间激活不保存
    x = checkpoint(self.block2, x)   # block2 的中间激活不保存
    x = self.block3(x)               # 这个保留
    return x
```

HuggingFace Transformers 提供 `gradient_checkpointing_enable()` 一键开启。

## 为什么重要

- **适配显存不足的场景**：当模型过大导致 CUDA Out of Memory (OOM) 时，梯度检查点是第一道防线——在调整模型结构、换更大 GPU 或使用分布式训练之前，先试试梯度检查点。
- **扩大 batch size**：减少激活值显存后可以设置更大的 batch size，从而提升训练稳定性和吞吐量。
- **训练长序列 Transformer**：序列长度增加时，自注意力层的中间激活呈平方增长，梯度检查点对长序列训练几乎是必需品。

## 如何使用

### PyTorch 基础用法

```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# 单层检查点
def forward(self, x):
    return checkpoint(self.heavy_layer, x, use_reentrant=False)

# 顺序模块检查点
def forward(self, x):
    return checkpoint_sequential(self.blocks, segments=4, x, use_reentrant=False)
```

### HuggingFace Transformers

```python
model = AutoModelForCausalLM.from_pretrained("model-name")
model.gradient_checkpointing_enable()
# 大部分 HF 模型一行开启，会自动处理需要 checkpoint 的层
```

### 选择哪些层 checkpoint

- **优先 checkpoint 激活值大的层**：Transformer 中注意力层和 FFN 层的中间激活占用了绝大部分显存，是 checkpoint 的主要目标。
- **跳过小层**：如 embedding 层、layer norm、简单的激活函数（ReLU、GELU）——它们的中间激活占比小，checkpoint 的收益有限，反而增加不必要的重计算开销。
- **不要 checkpoint 自定义复杂操作**：如果某个操作的 forward 包含随机性（dropout、drop path），checkpoint 可能导致反向传播的结果与期望不一致（除非保证 seed 可重现）。

### 与混合精度的交互

梯度检查点与 `torch.cuda.amp` 可以正常配合使用，但需注意：

```python
with torch.cuda.amp.autocast():
    x = checkpoint(self.block, x, use_reentrant=False)
    # checkpoint 内部的自动类型转换由 autocast 上下文的传播处理
```

`use_reentrant=False` 模式与 AMP 兼容性更好，建议在 PyTorch 2.x 中默认使用。

## 常见误区

1. **梯度检查点 = 免费的显存**：不是免费的。它节省显存，代价是增加约 20-30% 的训练时间（一次额外的前向传播）。在显存充裕时不建议开启。

2. **所有层都 checkpoint 效果最好**：只 checkpoint 激活值占比较大的层即可（如 Transformer 的注意力 + FFN）。checkpoint 过多反而增加不必要的调度和重计算开销。通常 checkpoint 4-8 层中选 1 层作为 check point 是平衡点。

3. **checkpoint 影响梯度质量**：不影响。数学上 checkpoint 的计算结果与不 checkpoint 完全一致，只是重新计算了中间激活，不改变梯度的数值。

4. **与 DataParallel/DDP 冲突**：梯度检查点与 DDP 兼容，但与 `torch.nn.DataParallel` 配合时可能出现问题。如果使用 DP，建议先迁移到 DDP。

5. **开启后训练变慢一定是 checkpoint 的锅**：训练变慢可能来自 (a) checkpoint 粒度太细，调度开销过大；(b) batch size 变大后数据加载成为新瓶颈；(c) 混合精度的 autocast 范围扩大导致落回 FP32 的操作增加。

## 参见

- [[../batch-size/gradient-accumulation]] —— 梯度检查点配合梯度累积进一步降低有效显存占用
- [[../optimizer-lr/optimizer-lr]] —— checkpoint 开启后 batch size 增大，可能需调整学习率
- [[../../contradictory]] —— 知识矛盾：社区对 checkpoint 粒度的最佳实践存在不同推荐
