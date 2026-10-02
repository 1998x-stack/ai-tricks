"""无第三方依赖的驱动能力声明。"""
MODEL_AXES = {
    'model.activation', 'model.norm', 'model.dropout', 'model.spatial_dropout',
    'model.residual', 'model.width', 'model.depth', 'model.kernel_size',
    'model.weight_init', 'model.bias_init', 'model.init_std', 'model.output_activation',
}
