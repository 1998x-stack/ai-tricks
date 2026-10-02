"""不依赖 torch 的训练配置、预算与验证时点校验。"""
import math


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} 必须为 >= {minimum} 的整数')
    return value


def finite_number(value, name, minimum=0., maximum=None, upper_open=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} 必须是有限数值')
    if value < minimum or (maximum is not None and (value >= maximum if upper_open else value > maximum)):
        raise ValueError(f'{name} 超出支持范围')
    return value


def validate_numbers(config):
    for name, value in config.items():
        if name in {'training.batch_size','training.accumulation_steps','training.epochs',
                    'adapter.rank','adapter.alpha','model.width','model.depth'}:
            positive_int(value,name)
        elif name == 'data.max_length':
            positive_int(value,name,minimum=2)
        elif name == 'training.patience':
            positive_int(value,name,minimum=0)
        elif name in {'model.dropout','model.spatial_dropout','adapter.dropout',
                      'optimizer.beta1','optimizer.beta2','optimizer.momentum',
                      'optimizer.warmup_ratio'}:
            finite_number(value,name,maximum=1.,upper_open=True)
        elif name == 'loss.label_smoothing':
            finite_number(value,name,maximum=1.)
        elif name in {'optimizer.lr','optimizer.weight_decay','optimizer.min_lr','optimizer.gamma',
                      'training.clip_norm','training.max_weight_norm','model.init_std',
                      'loss.l1','loss.flooding','data.brightness_jitter'}:
            finite_number(value,name)
        elif name in {'model.residual','data.standardize','data.shuffle',
                      'training.gradient_checkpointing','loss.assistant_only'}:
            if type(value) is not bool:
                raise ValueError(f'{name} 必须为布尔值')
    if 'optimizer.lr' in config and config['optimizer.lr'] <= 0:
        raise ValueError('optimizer.lr 必须 > 0')


def validate_active_axis(config, axis):
    """拒绝已经接受但实际上没有参与计算的单因素干预。"""
    schedule = config.get('optimizer.schedule','constant')
    if axis == 'optimizer.gamma' and schedule not in ('step','multistep','exponential'):
        raise ValueError('gamma 干预需要 step/multistep/exponential 调度')
    if axis == 'optimizer.warmup_ratio' and schedule != 'warmup_cosine':
        raise ValueError('warmup_ratio 干预需要 warmup_cosine 调度')
    if axis == 'optimizer.min_lr' and schedule == 'constant':
        raise ValueError('constant 调度下 min_lr 不参与计算')
    if axis in ('optimizer.beta1','optimizer.beta2') and config.get('optimizer.kind','sgd') not in ('adam','adamw','adam_then_sgd'):
        raise ValueError('beta 干预需要 Adam 系优化器')
    if axis == 'optimizer.momentum' and config.get('optimizer.kind','sgd') not in ('sgd','adam_then_sgd'):
        raise ValueError('momentum 干预需要 SGD')
    if axis == 'model.weight_init' and 'model.init_std' in config:
        raise ValueError('init_std 会覆盖 weight_init，无法解释初始化方法对照')


def validate_budget(profile, config):
    budget = profile['budget']
    positive_int(config.get('training.epochs',budget['epochs']),'epochs')
    positive_int(budget['max_steps'],'max_steps')
    positive_int(budget['eval_every_examples'],'eval_every_examples')


class EvaluationClock:
    """以累计样本数决定验证时点；尾部只验证一次，不受 microbatch 数直接影响。"""
    def __init__(self, interval):
        self.interval = positive_int(interval,'evaluation interval')
        self.next = interval
        self.last = 0

    def due(self, seen, final=False):
        positive_int(seen,'examples seen',minimum=0)
        if seen < self.last:
            raise ValueError('样本计数不能倒退')
        if seen == self.last or (not final and seen < self.next):
            return False
        self.last = seen
        while self.next <= seen:
            self.next += self.interval
        return True


def require_complete_epoch(actual, expected):
    if actual != expected:
        raise RuntimeError(f'数据流样本数与注册值不符: 实际 {actual}，期望 {expected}；拒绝标记完整训练')
