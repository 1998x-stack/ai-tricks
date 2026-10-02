# cnn-cv 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [cnn-cv-001-01](cnn-cv-001-01/README.md) 默认使用3x3卷积 | `model.kernel_size` | `5` → `3` |
| [cnn-cv-002-01](cnn-cv-002-01/README.md) 1xN卷积核处理长条形目标 | `model.kernel_shape` | `[3, 3]` → `[1, 7]` |
| [cnn-cv-003-01](cnn-cv-003-01/README.md) ACNet结构（白给的精度提升） | `model.conv_block` | `conv3x3` → `acnet_fused` |
| [cnn-cv-004-01](cnn-cv-004-01/README.md) 深度可分离卷积（Depthwise Separable） | `model.conv_block` | `standard` → `depthwise_separable` |
| [cnn-cv-005-01](cnn-cv-005-01/README.md) 小卷积核堆叠优于大卷积核 | `model.conv_block` | `one_5x5` → `two_3x3` |
| [cnn-cv-006-01](cnn-cv-006-01/README.md) 传统手工卷积核 | `model.first_filter` | `learned` → `sobel_initialized` |
| [cnn-cv-007-01](cnn-cv-007-01/README.md) MaxPool放在ReLU之前 | `model.pool_relu_order` | `relu_pool` → `pool_relu` |
| [cnn-cv-008-01](cnn-cv-008-01/README.md) 经典特征提取模型 | `model.feature_extraction` | `online` → `cached_frozen` |
| [cnn-cv-009-01](cnn-cv-009-01/README.md) 超参搜索用浅层网络，确定后用深层 | `search.proxy_depth` | `full` → `shallow` |
| [cnn-cv-010-01](cnn-cv-010-01/README.md) 膨胀残差网络适合稠密预测 | `model.last_dilation` | `1` → `2` |
| [cnn-cv-011-01](cnn-cv-011-01/README.md) 不轻易改网络结构 | `search.policy` | `architecture_first` → `data_lr_first` |
| [cnn-cv-012-01](cnn-cv-012-01/README.md) Backbone构建层次化Neck | `model.neck` | `last_level` → `fpn` |
| [cnn-cv-013-01](cnn-cv-013-01/README.md) 检测不能盲目去掉FPN | `model.neck` | `last_level` → `fpn` |
| [cnn-cv-014-01](cnn-cv-014-01/README.md) BiFPN有用 | `model.neck` | `fpn` → `bifpn` |
| [cnn-cv-015-01](cnn-cv-015-01/README.md) Anchor层灵活设计 | `model.anchors` | `default` → `train_kmeans` |
| [cnn-cv-016-01](cnn-cv-016-01/README.md) 清楚感受野大小 | `model.receptive_field_policy` | `default` → `object_scale_matched` |
| [cnn-cv-017-01](cnn-cv-017-01/README.md) 最后一层感受野覆盖最大有意义物体 | `model.last_dilation` | `1` → `2` |
| [cnn-cv-018-01](cnn-cv-018-01/README.md) 大感受野不总是更好 | `model.last_dilation` | `4` → `2` |
| [cnn-cv-019-01](cnn-cv-019-01/README.md) 相对感受野决定下采样时机 | `model.downsample_policy` | `uniform` → `relative_receptive_field` |
| [cnn-cv-020-01](cnn-cv-020-01/README.md) 前几层下采样频率高，中间层降低 | `model.downsample_policy` | `uniform` → `early_dense_late_sparse` |
| [cnn-cv-021-01](cnn-cv-021-01/README.md) 下采样用大卷积核保分辨率 | `model.downsample_kernel` | `3` → `5` |
| [cnn-cv-022-01](cnn-cv-022-01/README.md) 空洞卷积替代最后下采样 | `model.last_stage` | `stride2` → `dilation2_stride1` |
| [cnn-cv-023-01](cnn-cv-023-01/README.md) Low-level任务少下采样 | `model.output_stride` | `16` → `8` |
| [cnn-cv-024-01](cnn-cv-024-01/README.md) 微步幅+通道翻倍 | `model.downsample_channels` | `constant` → `double` |
| [cnn-cv-025-01](cnn-cv-025-01/README.md) Upsample+1x1 Conv替代Transposed Conv | `model.upsample` | `transpose_conv` → `bilinear_1x1` |
| [cnn-cv-026-01](cnn-cv-026-01/README.md) Shortcut连接几乎无痛 | `model.residual` | `False` → `True` |
| [cnn-cv-027-01](cnn-cv-027-01/README.md) 残差和密集连接做模块替换 | `model.conv_block` | `plain` → `dense` |
| [cnn-cv-027-02](cnn-cv-027-02/README.md) 残差和密集连接做模块替换 | `model.conv_block` | `plain` → `residual` |
| [cnn-cv-028-01](cnn-cv-028-01/README.md) 分割中Skip连接保边界 | `model.encoder_skip` | `False` → `True` |
| [cnn-cv-029-01](cnn-cv-029-01/README.md) 疯狂给框→AP高但FP多 | `inference.proposal_limit` | `100` → `1000` |
| [cnn-cv-030-01](cnn-cv-030-01/README.md) Gaussian YOLOv3减少FP | `model.box_head` | `deterministic` → `gaussian_uncertainty` |
| [cnn-cv-031-01](cnn-cv-031-01/README.md) Anchor数量与位置灵活设计 | `model.anchors` | `default` → `train_kmeans` |
| [cnn-cv-032-01](cnn-cv-032-01/README.md) 多尺度输入提升效果 | `data.image_sizes` | `[224]` → `[160, 224, 288]` |
| [cnn-cv-033-01](cnn-cv-033-01/README.md) 多尺度池化模块捕获上下文 | `model.context_pool` | `global` → `spatial_pyramid` |
| [cnn-cv-034-01](cnn-cv-034-01/README.md) Filter数量为2的幂 | `model.channels` | `60` → `64` |
| [cnn-cv-034-02](cnn-cv-034-02/README.md) Filter数量为2的幂 | `model.channels` | `64` → `96` |
| [cnn-cv-035-01](cnn-cv-035-01/README.md) 第一层Filter不要太少 | `model.first_channels` | `8` → `32` |
| [cnn-cv-036-01](cnn-cv-036-01/README.md) 深度可分离卷积后通道数乘2 | `model.depthwise_width_multiplier` | `1` → `2` |
| [cnn-cv-037-01](cnn-cv-037-01/README.md) 分类网络训练充分性判别 | `tracking.diagnostics` | `accuracy_only` → `confidence_and_calibration` |
| [cnn-cv-038-01](cnn-cv-038-01/README.md) Loss设计：分类Softmax，回归L2 | `loss.kind` | `mse_on_class_ids` → `cross_entropy` |
| [cnn-cv-038-02](cnn-cv-038-02/README.md) Loss设计：分类Softmax，回归L2 | `model.output_contract` | `relu_regression` → `linear_regression` |
| [cnn-cv-039-01](cnn-cv-039-01/README.md) 分类精度不够→类间太近用AmSoftmax | `loss.kind` | `cross_entropy` → `am_softmax` |
| [cnn-cv-040-01](cnn-cv-040-01/README.md) 分类精度不够→样本不均衡用Focal Loss | `loss.kind` | `cross_entropy` → `focal` |
| [cnn-cv-041-01](cnn-cv-041-01/README.md) FCN做Landmark，不用FC回归 | `model.landmark_head` | `coordinate_fc` → `heatmap_fcn` |
| [cnn-cv-042-01](cnn-cv-042-01/README.md) 亮度变换增强 | `data.brightness_jitter` | `0.0` → `0.2` |
| [cnn-cv-043-01](cnn-cv-043-01/README.md) 数据增强要结合任务设计 | `data.augmentation` | `generic` → `label_preserving` |
| [cnn-cv-044-01](cnn-cv-044-01/README.md) Batch Size越大越好 | `training.batch_size` | `32` → `256` |
| [cnn-cv-045-01](cnn-cv-045-01/README.md) 一阶段分类不好→换两阶段 | `model.detector` | `one_stage` → `two_stage` |
| [cnn-cv-046-01](cnn-cv-046-01/README.md) 检测预训练提升前景背景区分 | `checkpoint.pretrain_task` | `classification` → `detection` |
| [cnn-cv-047-01](cnn-cv-047-01/README.md) 建议Finetune而非从头训练 | `checkpoint.initialization` | `random` → `pretrained` |
| [cnn-cv-048-01](cnn-cv-048-01/README.md) 检查Pretrained权重是否加载成功 | `validation.load_check` | `unchecked` → `keys_shapes_hash` |
| [cnn-cv-049-01](cnn-cv-049-01/README.md) 超参搜索阶段用浅层预训练模型 | `search.proxy_depth` | `full` → `shallow` |
| [cnn-cv-050-01](cnn-cv-050-01/README.md) Finetune时学习率要小 | `optimizer.lr` | `0.001` → `2e-05` |
| [cnn-cv-051-01](cnn-cv-051-01/README.md) Finetune后加Dropout | `model.dropout` | `0.0` → `0.2` |
| [cnn-cv-052-01](cnn-cv-052-01/README.md) 深度决定表达力，宽度决定信息量 | `model.depth` | `2` → `4` |
| [cnn-cv-052-02](cnn-cv-052-02/README.md) 深度决定表达力，宽度决定信息量 | `model.width` | `32` → `64` |
| [cnn-cv-053-01](cnn-cv-053-01/README.md) 理想体型：高高瘦瘦 | `model.shape_policy` | `wide_shallow` → `narrow_deep` |
| [cnn-cv-054-01](cnn-cv-054-01/README.md) 加深过程中慢慢变胖 | `model.channel_schedule` | `constant` → `progressive_double` |
| [cnn-cv-055-01](cnn-cv-055-01/README.md) 宽 vs 深的工程选择 | `model.shape_policy` | `deep_narrow` → `shallow_wide` |
| [cnn-cv-056-01](cnn-cv-056-01/README.md) 先过拟合再Trade-off | `search.policy` | `regularization_first` → `tiny_overfit_first` |
| [cnn-cv-057-01](cnn-cv-057-01/README.md) 观察Loss而非Accuracy | `checkpoint.selection_metric` | `val_accuracy` → `val_loss` |
| [cnn-cv-058-01](cnn-cv-058-01/README.md) CV类学习率：1e-3级别 | `optimizer.lr` | `1e-05` → `0.001` |
| [cnn-cv-059-01](cnn-cv-059-01/README.md) 权重初始化 | `model.weight_init` | `default` → `xavier_uniform` |
| [cnn-cv-060-01](cnn-cv-060-01/README.md) Batch Normalization必须用 | `model.norm` | `none` → `batch` |
| [cnn-cv-061-01](cnn-cv-061-01/README.md) 多任务Loss平衡 | `loss.task_weighting` | `unscaled` → `train_scale_equalized` |
| [cnn-cv-062-01](cnn-cv-062-01/README.md) Ensemble策略 | `inference.ensemble_members` | `1` → `3` |
| [cnn-cv-063-01](cnn-cv-063-01/README.md) 先训大模型再裁剪 | `training.compression_policy` | `small_from_scratch` → `large_then_prune_finetune` |
| [cnn-cv-064-01](cnn-cv-064-01/README.md) 不从零加激活函数/BN/Dropout | `model.activation` | `relu` → `identity` |
| [cnn-cv-065-01](cnn-cv-065-01/README.md) SpatialDropout防多分支相关特征过拟合 | `model.spatial_dropout` | `0.0` → `0.2` |
| [cnn-cv-066-01](cnn-cv-066-01/README.md) 均值减法可能对灰度图像有害 | `data.center_grayscale` | `True` → `False` |
| [cnn-cv-067-01](cnn-cv-067-01/README.md) 多尝试新Loss | `loss.kind` | `cross_entropy` → `focal` |
| [cnn-cv-068-01](cnn-cv-068-01/README.md) 不要只看精度 | `evaluation.protocol` | `accuracy_only` → `accuracy_latency_memory` |
