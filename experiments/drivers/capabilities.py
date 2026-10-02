"""无第三方依赖的驱动能力声明。"""
MODEL_AXES = {
    'model.activation', 'model.norm', 'model.dropout', 'model.spatial_dropout',
    'model.residual', 'model.width', 'model.depth', 'model.kernel_size',
    'model.weight_init', 'model.bias_init', 'model.init_std', 'model.output_activation',
}

MODEL_VALUES = {
    'model.activation': {'sigmoid','tanh','relu','leaky_relu','prelu','mish','identity','relu6'},
    'model.norm': {'none','batch','layer','group','instance','local_response'},
    'model.weight_init': {'default','kaiming_normal','xavier_uniform','xavier_normal','normal_fan_in','zeros','orthogonal'},
    'model.bias_init': {'zeros','normal'},
    'model.output_activation': {'identity','relu'},
    'optimizer.kind': {'sgd','adam','adamw','adam_then_sgd','adagrad'},
    'training.precision': {'fp32','fp16','bf16'},
}
