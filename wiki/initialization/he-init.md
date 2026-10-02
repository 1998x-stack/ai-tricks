# He/Kaiming 初始化

## 是什么

He 初始化按 fan 和激活函数增益设置初始权重尺度。原素材中的王惠东等作者建议 ReLU 搭配 He 初始化；具体参数定义以 [PyTorch 初始化接口](https://docs.pytorch.org/docs/stable/nn.init.html)为准。

对 ReLU、`mode="fan_in"`：

- 权重方差：`2 / fan_in`。
- 权重标准差：`sqrt(2 / fan_in)`，不要把方差当作标准差。
- 正态形式：`Normal(mean=0, variance=2/fan_in)`；若 API 接收 `std`，传平方根。
- 均匀形式：`Uniform(-sqrt(6/fan_in), sqrt(6/fan_in))`。

对称零均值输入经过 ReLU 后，**二阶矩**减半，但输出均值变为正，不能把二阶矩直接称为方差。以零均值高斯输入 Z 为例：

`E[ReLU(Z)^2] = Var(Z)/2`，而 `Var(ReLU(Z)) = Var(Z)*(1/2 - 1/(2*pi))`。

初始化推导是在独立性等近似下控制信号尺度，不是保证任意网络收敛的定理。

## 为什么重要

初始尺度过大可能导致饱和或爆炸，过小可能使深层信号衰减。He 初始化给 ReLU 网络提供有依据的起点，但不能替代残差设计、归一化、合适的学习率或数据检查。

`fan_in` 倾向于保持前向尺度；`fan_out` 倾向于保持反向梯度尺度。卷积 fan 还包含卷积核空间大小，不仅是输入通道数。

## 如何使用

```python
import torch.nn as nn

# ReLU：明确 fan 和 nonlinearity。
nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")

# LeakyReLU：必须把实际负斜率传给 a。
slope = 0.01
nn.init.kaiming_normal_(
    layer.weight, a=slope, mode="fan_in", nonlinearity="leaky_relu"
)
if layer.bias is not None:
    nn.init.zeros_(layer.bias)
```

LeakyReLU 的增益为 `sqrt(2/(1+a*a))`；`a` 默认值不是自动从模型激活层读取的。PReLU 可按初始负斜率初始化，训练中斜率变化不意味着每步重新初始化权重。

`nn.Linear`/`nn.Conv2d` 的默认初始化不能简单理解为“已经按 ReLU 增益初始化”。应查看模块版本与实际参数，避免仅看到 Kaiming 函数名就认定其配置与上述调用相同。

A/B 固定输入、模型拓扑、激活函数、种子与优化器，只改初始化；记录逐层激活二阶矩、梯度范数、训练曲线及最终验证表现。对预训练模型的微调不要重新初始化整个 backbone。

## 常见误区

- **标准差写成 2/fan_in**：应取平方根。
- **把 ReLU 输出方差称为精确减半**：精确减半的是对称输入的二阶矩。
- **不传 LeakyReLU 的实际 a**：初始化函数不会自动查询激活层。
- **He + BN 保证收敛**：初始化只是训练条件之一；BN 也并非所有网络必需。
- **把一个 CNN 结果外推到所有模型**：需重新检验激活、残差路径和归一化布局。

## 参见

- [[initialization]] — 作者来源与技巧主条目
- [[xavier-init]] — Xavier 初始化
- [[why-initialization-matters]] — 初始尺度
- [[../activation/leaky-relu-prelu]] — 负斜率
- [初始化实验目录](../../experiments/initialization/README.md)
