"""CIFAR 分辨率的 ResNet-18/34 实际模型；不访问网络、不加载示例权重。"""
import torch
from torch import nn


from experiments.drivers.capabilities import MODEL_AXES


def activation(name):
    factories = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh,
                 'leaky_relu': lambda: nn.LeakyReLU(.01), 'prelu': nn.PReLU,
                 'mish': nn.Mish, 'relu6': nn.ReLU6, 'identity': nn.Identity}
    if name not in factories:
        raise ValueError(f'不支持的激活函数: {name}')
    return factories[name]()


class ChannelLayerNorm(nn.Module):
    """在每个空间位置沿通道归一化，不依赖 batch 统计量。"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def normalization(name, channels):
    factories = {'none': nn.Identity,
                 'batch': lambda: nn.BatchNorm2d(channels),
                 'layer': lambda: ChannelLayerNorm(channels),
                 'group': lambda: nn.GroupNorm(min(8, channels), channels),
                 'instance': lambda: nn.InstanceNorm2d(channels, affine=True),
                 'local_response': lambda: nn.LocalResponseNorm(5)}
    if name not in factories:
        raise ValueError(f'不支持的归一化: {name}')
    return factories[name]()


class BasicBlock(nn.Module):
    def __init__(self, inputs, outputs, stride, config):
        super().__init__()
        kernel = config.get('model.kernel_size', 3)
        if kernel not in (3, 5):
            raise ValueError('此驱动支持 3x3 / 5x5 卷积')
        norm = config.get('model.norm', 'batch')
        act = config.get('model.activation', 'relu')
        self.residual = config.get('model.residual', True)
        self.conv1 = nn.Conv2d(inputs, outputs, kernel, stride, kernel // 2, bias=False)
        self.norm1 = normalization(norm, outputs)
        self.act1 = activation(act)
        self.conv2 = nn.Conv2d(outputs, outputs, kernel, 1, kernel // 2, bias=False)
        self.norm2 = normalization(norm, outputs)
        self.act2 = activation(act)
        self.spatial_dropout = nn.Dropout2d(config.get('model.spatial_dropout', 0.0))
        self.projection = (nn.Identity() if stride == 1 and inputs == outputs else
                           nn.Sequential(nn.Conv2d(inputs, outputs, 1, stride, bias=False),
                                         normalization(norm, outputs)))

    def forward(self, x):
        hidden = self.act1(self.norm1(self.conv1(x)))
        hidden = self.norm2(self.conv2(self.spatial_dropout(hidden)))
        if self.residual:
            hidden = hidden + self.projection(x)
        return self.act2(hidden)


class ResNet(nn.Module):
    def __init__(self, config, classes=10):
        super().__init__()
        width = config.get('model.width', 64)
        # depth 在实验中定义为每 stage 的 basic block 数，2 = ResNet-18。
        blocks_per_stage = config.get('model.depth', 2)
        if not 1 <= blocks_per_stage <= 6 or width < 8 or width % 8:
            raise ValueError('非法网络宽度/每阶段深度')
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, bias=False),
                                  normalization(config.get('model.norm', 'batch'), width),
                                  activation(config.get('model.activation', 'relu')))
        stages = []
        channels = width
        for stage in range(4):
            target = width * 2 ** stage
            for block in range(blocks_per_stage):
                stride = 2 if stage > 0 and block == 0 else 1
                stages.append(BasicBlock(channels, target, stride, config))
                channels = target
        self.blocks = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(config.get('model.dropout', 0.0))
        self.head = nn.Linear(channels, classes)
        self.output_activation = activation(config.get('model.output_activation', 'identity'))
        self.initialize(config)

    def initialize(self, config):
        kind = config.get('model.weight_init', 'kaiming_normal')
        for layer in self.modules():
            if not isinstance(layer, (nn.Conv2d, nn.Linear)):
                continue
            weight = layer.weight
            if kind in ('default', 'kaiming_normal'):
                nn.init.kaiming_normal_(weight, nonlinearity='relu')
            elif kind == 'xavier_normal':
                nn.init.xavier_normal_(weight)
            elif kind == 'xavier_uniform':
                nn.init.xavier_uniform_(weight)
            elif kind == 'normal_fan_in':
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                nn.init.normal_(weight, std=fan_in ** -.5)
            elif kind == 'orthogonal':
                nn.init.orthogonal_(weight)
            elif kind == 'zeros':
                nn.init.zeros_(weight)
            else:
                raise ValueError(f'不支持初始化 {kind}')
            if 'model.init_std' in config:
                nn.init.normal_(weight, std=config['model.init_std'])
            if layer.bias is not None:
                if config.get('model.bias_init', 'zeros') == 'normal':
                    nn.init.normal_(layer.bias, std=.02)
                else:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        features = self.blocks(self.stem(x))
        logits = self.head(self.dropout(self.pool(features).flatten(1)))
        return self.output_activation(logits)
