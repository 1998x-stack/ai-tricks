"""逐条人工指定的干预；不根据标题关键字猜测实验行为。"""
import json

# 格式：配置路径 | A | B。JSON 值保证数字/布尔不会变成字符串。
# 一个父条目可拆成多个单因素实验；复合流程以 policy 为实验单位。
TABLES = {}
TABLES['activation'] = '''
model.activation|"sigmoid"|"relu"
model.output_activation|"relu"|"identity"
model.output_contract|"softmax_before_cross_entropy"|"logits_cross_entropy"
model.rnn_activation|"relu"|"tanh"
model.activation|"sigmoid"|"tanh"
model.activation|"relu"|"leaky_relu"
loss.kind|"cross_entropy"|"am_softmax"
loss.retrieval|"pointwise_bce"|"listwise_softmax"
optimizer.lr|0.1|0.01
model.pool_relu_order|"relu_pool"|"pool_relu"
model.activation|"identity"|"mish"
model.activation|"relu"|"relu6"
model.weight_init|"xavier_normal"|"kaiming_normal"
search.policy|"activation_first"|"depth_width_activation"
model.activation|"relu"|"identity"
search.policy|"activation_first"|"data_lr_first"
'''
TABLES['batch-size'] = '''
training.contrastive_batch|32|256
training.batch_size|256|32
training.rollout_batch|32|256
training.prompt_batch|8|64
training.batch_size|32|512
optimizer.lr|0.1|0.4
tracking.batch_metadata|"nominal_only"|"micro_world_accum_effective"
training.world_size|1|2
training.batch_size|32|256
training.batch_size|16|128
search.scaling_policy|"fixed"|"deepseek_compute_law"
training.accumulation_steps|1|4
training.batch_size|32|128
training.batch_size|256|64
search.policy|"joint_random"|"lr_then_batch"
training.batch_size|32|128
data.batch_order|"random"|"length_bucket"
training.batch_size|32|128
optimizer.lr|0.00002|0.000002
'''
TABLES['initialization'] = '''
model.weight_init|"normal_fan_in"|"xavier_normal"
model.weight_init|"xavier_normal"|"kaiming_normal"
model.embedding_init|"normal"|"truncated_normal"
model.embedding_init|"normal"|"uniform_equal_variance"
model.recurrent_init|"normal"|"orthogonal_svd"
model.weight_init|"zeros"|"xavier_normal"
training.framework|"pytorch"|"tensorflow"
model.bias_init|"normal"|"zeros"
inference.ensemble_init|["xavier_normal"]|["xavier_normal","kaiming_normal","orthogonal"]
validation.load_check|"unchecked"|"keys_shapes_hash"
model.weight_init|"default"|"xavier_uniform"
model.init_std|0.02|0.06
model.weight_init|"normal_fan_in"|"orthogonal"
model.weight_init|"default"|"xavier_uniform"
search.policy|"init_search_first"|"tiny_overfit_first"
'''
TABLES['normalization'] = '''
model.norm|"batch"|"none"
model.dropout|0.0|0.5
model.dropout|0.5|0.0
model.norm|"batch"|"layer"
model.norm|"batch"|"group"
model.norm|"batch"|"sync_batch"
model.norm|"none"|"layer"
data.target_normalization|false|true
model.norm|"none"|"batch"
model.norm|"none"|"batch"
model.norm|"batch"|"group"
model.norm|"batch"|"group"
model.norm|"none"|"batch"
model.norm|"local_response"|"none"
model.norm|"none"|"batch"
'''
TABLES['regularization'] = '''
search.policy|"regularization_first"|"tiny_overfit_first"
model.dropout|0.0|0.5
optimizer.weight_decay|0.0|0.01
loss.l1|0.0|0.0001
training.patience|0|5
data.augmentation|"none"|"label_preserving"
loss.label_smoothing|0.0|0.1
model.spatial_dropout|0.0|0.2
training.max_weight_norm|0.0|2.0
loss.flooding|0.0|0.1
data.augmentation|"none"|"mixup"
inference.ensemble_members|1|3
model.dropout|0.0|0.5
search.policy|"random_changes"|"train_val_gap_diagnosis"
training.clip_norm|0.0|1.0
optimizer.kind|"adam"|"sgd"
'''
TABLES['infra'] = '''
training.precision|"fp32"|"bf16"
training.world_size|1|2
training.gradient_checkpointing|false|true
search.sampler|"grid"|"random"
tracking.level|"metrics_only"|"full_provenance"
engineering.pipeline|"monolithic"|"modular"
engineering.runner|"inline"|"runner"
data.batch_order|"random"|"length_bucket"
data.num_workers|0|4
data.tensor_conversion|"tensor_copy"|"from_numpy_shared"
training.seed_policy|"independent_unlogged"|"paired_logged"
training.accumulation_steps|1|4
checkpoint.restore|"weights_only"|"full_training_state"
search.sampler|"grid"|"bayesian"
search.scaling_policy|"fixed"|"deepseek_compute_law"
search.policy|"full_data_first"|"tiny_overfit_first"
optimizer.schedule|"constant"|"warmup_cosine"
'''
TABLES['nlp'] = '''
optimizer.lr|0.001|0.00002
model.pooling|"cls"|"masked_mean"
training.adversarial|"none"|"fgm"
model.cross_attention|false|true
model.attention_window|128|512
model.norm|"batch"|"layer"
model.pooling|"masked_mean"|"attention"
model.rnn_activation|"relu"|"tanh"
model.embedding_init|"xavier_uniform"|"truncated_normal"
data.augmentation|"none"|"synonym_replace"
optimizer.warmup_ratio|0.0|0.1
data.tokenizer_kind|"word"|"bpe"
data.batch_order|"random"|"length_bucket"
model.embedding_source|"random"|"word2vec"
optimizer.kind|"sgd"|"adam"
training.clip_norm|0.0|1.0
'''
TABLES['recsys'] = '''
checkpoint.warm_start|"embedding_only"|"embedding_and_dnn"
data.numeric_transform|"raw"|"train_quantile_bins"
data.feature_policy|"all_at_once"|"forward_selection"
data.negative_source|"exposed_unclicked"|"global_random"
data.negative_source|"global_random"|"ranker_topk"
training.multitask_gradient|"sum"|"pcgrad"
loss.task_weighting|"unscaled"|"train_scale_equalized"
model.pooling|"masked_mean"|"attention"
model.feature_gate|"none"|"senet"
training.objective|"supervised"|"policy_gradient"
loss.retrieval|"pointwise_bce"|"listwise_softmax"
training.batch_size|32|128
optimizer.sparse_kind|"adam"|"adagrad"
optimizer.kind|"sgd"|"adagrad"
model.embedding_l2_normalize|true|false
training.epochs|3|1
model.activation|"relu"|"identity"
model.norm|"batch"|"layer"
checkpoint.selection_metric|"val_loss"|"val_auc"
optimizer.kind|"adam"|"adam_then_sgd"
data.split_policy|"random"|"time"
data.exclude_known_positive|false|true
model.pooling|"masked_mean"|"attention"
'''
TABLES['data-eval'] = '''
search.policy|"full_data_first"|"tiny_overfit_first"
data.label_filter|"none"|"audited_clean"
data.domain_sampling|"source_only"|"target_coverage"
data.split_policy|"random"|"time"
data.split_policy|"random"|"group"
search.holdout_policy|"reuse_test"|"locked_test"
checkpoint.selection_metric|"val_accuracy"|"val_loss"
checkpoint.selection_metric|"val_loss"|"val_f1"
data.split_policy|"random"|"time"
tracking.diagnostics|"train_only"|"train_and_val"
loss.task_weighting|"unscaled"|"train_scale_equalized"
data.target_normalization|false|true
data.standardize|false|true
data.augmentation|"none"|"horizontal_flip"
loss.kind|"cross_entropy"|"focal"
data.negative_source|"global_random"|"hard_mined"
data.class_sampling|"natural"|"balanced"
data.label_filter|"none"|"audited_clean"
visualization.embedding|"tsne_raw"|"pca_then_tsne"
tracking.dataset_version|"path_only"|"content_hash"
evaluation.protocol|"loss_only"|"frozen_task_suite"
data.answer_style|"mixed"|"canonical"
data.label_filter|"none"|"audited_clean"
data.domain_sampling|"source_only"|"target_coverage"
search.policy|"random_changes"|"error_slice_first"
training.seed_policy|"independent_unlogged"|"paired_logged"
loss.label_smoothing|0.0|0.1
search.holdout_policy|"reuse_test"|"locked_test"
tracking.diagnostics|"accuracy_only"|"confidence_and_calibration"
data.deduplicate|"none"|"cross_split"
'''
TABLES['debugging'] = '''
validation.preflight|"none"|"tiny_overfit"
data.train_labels|"original"|"permuted"
validation.preflight|"none"|"shape_dtype_loss_mode"
validation.nan_guard|"none"|"input_and_loss"
optimizer.lr|1.0|0.01
validation.autograd_anomaly|false|true
training.clip_norm|0.0|1.0
model.dropout|0.0|0.5
model.width|32|64
validation.data_alignment|"unchecked"|"sample_id_checked"
validation.optimizer_coverage|"unchecked"|"all_trainable_checked"
loss.numeric_implementation|"naive_log_softmax"|"stable_logsumexp"
optimizer.lr|0.000001|0.001
search.lr_policy|"default"|"range_test"
tracking.activation_histograms|false|true
tracking.weight_histograms|false|true
tracking.confidence_histograms|false|true
tracking.diagnostics|"train_only"|"train_and_val"
model.width|32|128
search.lr_policy|"default"|"validation_search"
search.policy|"multiple_changes"|"one_factor"
tracking.level|"metrics_only"|"full_provenance"
search.policy|"random_changes"|"diagnose_then_ablate"
data.label_filter|"none"|"error_audited"
validation.load_check|"unchecked"|"keys_shapes_hash"
validation.preflight|"none"|"tiny_overfit"
validation.fault_injection|"none"|"known_fault_suite"
checkpoint.selection_metric|"val_accuracy"|"val_loss"
data.preprocess_contract|"mismatched"|"identical"
training.clip_norm|0.0|1.0
training.patience|0|5
visualization.embedding|"pca"|"pca_then_tsne"
data.split_policy|"random"|"group"
search.holdout_policy|"reuse_test"|"locked_test"
training.accumulation_steps|1|4
'''
TABLES['llm-sft'] = '''
inference.prompt_policy|"minimal"|"validated_instruction"
data.label_filter|"single_teacher"|"multi_judge_agreement"
optimizer.lr|0.0001|0.00005
data.domain_sampling|"source_only"|"target_coverage"
data.answer_style|"mixed"|"canonical"
data.sft_format|"cot_fewshot"|"direct_answer"
data.samples_per_domain|100|400
data.curriculum|"random"|"easy_to_hard"
checkpoint.selection_metric|"val_loss"|"val_exact_match"
model.scale|"small"|"large"
adapter.rank|8|16
model.quantization|"none"|"nf4"
data.chat_template_contract|"mismatched"|"identical"
data.multiturn_fraction|0.1|0.5
data.packing|false|true
data.max_length|512|2048
loss.assistant_only|false|true
inference.knowledge_source|"parameters"|"retrieval"
training.alignment_policy|"sft_only"|"sft_then_rl"
training.epochs|6|3
optimizer.lr|0.001|0.00002
data.curriculum|"random"|"easy_to_hard"
'''
TABLES['llm-rl'] = '''
checkpoint.policy_start|"base"|"sft"
checkpoint.selection_metric|"val_loss"|"val_reward"
reward.kind|"subjective_judge"|"verifiable_rule"
checkpoint.policy_start|"sft"|"base"
checkpoint.eval_every_steps|500|50
checkpoint.base_success_stratum|"below_20pct"|"above_20pct"
reward.validation|"format_only"|"format_and_correctness"
reward.hack_guard|"none"|"heldout_adversarial_tests"
rl.kl_beta|0.1|0.01
rl.kl_beta|0.01|0.1
rl.kl_beta|1.0|0.01
rl.kl_beta|0.01|0.0
optimizer.lr|0.00002|0.000002
rl.sft_mix|0.0|0.1
reward.normalize|false|true
training.rollout_batch|32|256
training.clip_norm|1.0|0.2
optimizer.schedule|"constant"|"cosine"
optimizer.lr|0.0001|0.000001
reward.length_normalize|false|true
reward.components|["correctness"]|["correctness","format","safety"]
rl.value_clip|0.0|0.2
rl.entropy_coef|0.0|0.01
checkpoint.policy_start|"base"|"rejection_sample_sft"
checkpoint.policy_start|"weak_base"|"strong_base"
rl.dpo_beta|0.5|0.1
rl.dpo_beta|0.01|0.1
data.preference_filter|"unfiltered"|"audited_pairs"
checkpoint.policy_start|"weak_sft"|"strong_sft"
training.prompt_batch|8|64
checkpoint.policy_start|"base"|"rejection_sample_sft"
'''
TABLES['cnn-cv'] = '''
model.kernel_size|5|3
model.kernel_shape|[3,3]|[1,7]
model.conv_block|"conv3x3"|"acnet_fused"
model.conv_block|"standard"|"depthwise_separable"
model.conv_block|"one_5x5"|"two_3x3"
model.first_filter|"learned"|"sobel_initialized"
model.pool_relu_order|"relu_pool"|"pool_relu"
model.feature_extraction|"online"|"cached_frozen"
search.proxy_depth|"full"|"shallow"
model.last_dilation|1|2
search.policy|"architecture_first"|"data_lr_first"
model.neck|"last_level"|"fpn"
model.neck|"last_level"|"fpn"
model.neck|"fpn"|"bifpn"
model.anchors|"default"|"train_kmeans"
model.receptive_field_policy|"default"|"object_scale_matched"
model.last_dilation|1|2
model.last_dilation|4|2
model.downsample_policy|"uniform"|"relative_receptive_field"
model.downsample_policy|"uniform"|"early_dense_late_sparse"
model.downsample_kernel|3|5
model.last_stage|"stride2"|"dilation2_stride1"
model.output_stride|16|8
model.downsample_channels|"constant"|"double"
model.upsample|"transpose_conv"|"bilinear_1x1"
model.residual|false|true
model.conv_block|"plain"|"dense"
model.encoder_skip|false|true
inference.proposal_limit|100|1000
model.box_head|"deterministic"|"gaussian_uncertainty"
model.anchors|"default"|"train_kmeans"
data.image_sizes|[224]|[160,224,288]
model.context_pool|"global"|"spatial_pyramid"
model.channels|60|64
model.first_channels|8|32
model.depthwise_width_multiplier|1|2
tracking.diagnostics|"accuracy_only"|"confidence_and_calibration"
loss.kind|"mse_on_class_ids"|"cross_entropy"
loss.kind|"cross_entropy"|"am_softmax"
loss.kind|"cross_entropy"|"focal"
model.landmark_head|"coordinate_fc"|"heatmap_fcn"
data.brightness_jitter|0.0|0.2
data.augmentation|"generic"|"label_preserving"
training.batch_size|32|256
model.detector|"one_stage"|"two_stage"
checkpoint.pretrain_task|"classification"|"detection"
checkpoint.initialization|"random"|"pretrained"
validation.load_check|"unchecked"|"keys_shapes_hash"
search.proxy_depth|"full"|"shallow"
optimizer.lr|0.001|0.00002
model.dropout|0.0|0.2
model.depth|2|4
model.shape_policy|"wide_shallow"|"narrow_deep"
model.channel_schedule|"constant"|"progressive_double"
model.shape_policy|"deep_narrow"|"shallow_wide"
search.policy|"regularization_first"|"tiny_overfit_first"
checkpoint.selection_metric|"val_accuracy"|"val_loss"
optimizer.lr|0.00001|0.001
model.weight_init|"default"|"xavier_uniform"
model.norm|"none"|"batch"
loss.task_weighting|"unscaled"|"train_scale_equalized"
inference.ensemble_members|1|3
training.compression_policy|"small_from_scratch"|"large_then_prune_finetune"
model.activation|"relu"|"identity"
model.spatial_dropout|0.0|0.2
data.center_grayscale|true|false
loss.kind|"cross_entropy"|"focal"
evaluation.protocol|"accuracy_only"|"accuracy_latency_memory"
'''
TABLES['optimizer-lr'] = '''
optimizer.kind|"sgd"|"adam"
optimizer.kind|"sgd"|"adam"
optimizer.kind|"adam"|"sgd"
optimizer.kind|"adam"|"adam_then_sgd"
optimizer.kind|"adam"|"adamw"
optimizer.sparse_kind|"adam"|"adagrad"
optimizer.lr|0.001|0.00002
optimizer.lr|0.00001|0.001
search.lr_policy|"grid"|"log_bisection"
search.lr_policy|"default"|"range_test"
search.lr_policy|"default"|"range_test"
search.lr_policy|"default"|"range_test"
optimizer.schedule|"constant"|"warmup_cosine"
optimizer.warmup_ratio|0.03|0.1
optimizer.schedule|"constant"|"step"
optimizer.schedule|"multistep"|"cosine"
optimizer.schedule|"constant"|"cyclic"
optimizer.schedule|"constant"|"plateau_half"
optimizer.schedule|"constant"|"exponential"
optimizer.gamma|0.9|0.99
search.lr_policy|"linear_grid"|"log_grid"
optimizer.schedule|"constant"|"plateau_half"
optimizer.lr|1.0|0.01
optimizer.lr|0.000001|0.001
optimizer.schedule|"constant"|"plateau_half"
optimizer.schedule|"constant"|"warmup_cosine"
optimizer.lr|0.1|0.4
training.accumulation_steps|1|4
optimizer.momentum|0.0|0.9
optimizer.momentum_schedule|"constant"|"cyclic"
optimizer.weight_decay|0.0|0.01
training.clip_norm|0.0|1.0
optimizer.lr|0.00002|0.000002
rl.dpo_beta|0.01|0.1
optimizer.lr|0.0001|0.00005
search.scaling_policy|"fixed"|"deepseek_compute_law"
optimizer.beta2|0.999|0.95
optimizer.lr|0.0001|0.0003
search.policy|"multiple_changes"|"one_factor"
search.policy|"joint_random"|"lr_then_batch"
search.policy|"architecture_first"|"lr_first"
search.policy|"architecture_first"|"data_lr_first"
training.baseline_recipe|"project_default"|"sgd_0.1_m0.9_wd0.005"
optimizer.lr|0.001|0.1
optimizer.lr|0.1|0.01
tracking.diagnostics|"loss_only"|"loss_lr_gradient"
checkpoint.selection_metric|"val_loss"|"val_exact_match"
model.activation|"relu"|"identity"
training.seed_policy|"independent_unlogged"|"paired_logged"
training.patience|2|10
search.policy|"regularization_first"|"tiny_overfit_first"
model.weight_init|"default"|"kaiming_normal"
optimizer.schedule|"constant"|"plateau_half"
training.baseline_recipe|"project_default"|"sgd_0.1_m0.9_wd0.005"
model.weight_init|"default"|"xavier_uniform"
model.dropout|0.5|0.1
optimizer.lr|0.001|0.0001
optimizer.schedule|"constant"|"cosine"
optimizer.schedule|"step"|"cyclic"
optimizer.weight_decay|0.0001|0.01
optimizer.lr|0.0001|0.001
model.residual|false|true
model.norm|"none"|"batch"
training.compression_policy|"small_from_scratch"|"large_then_prune_finetune"
optimizer.gamma|0.9|0.5
training.batch_size|32|128
optimizer.warmup_ratio|0.03|0.1
search.scaling_policy|"fixed"|"deepseek_compute_law"
optimizer.lr|0.001|0.00002
data.preprocess_contract|"mismatched"|"identical"
optimizer.lr|0.001|0.0001
optimizer.schedule|"constant"|"warmup_cosine"
optimizer.lr|0.0001|0.000001
training.clip_norm|0.0|1.0
optimizer.kind|"adam"|"sgd"
optimizer.lr|0.001|0.0003
optimizer.lr|0.001|0.00035
optimizer.schedule|"cosine"|"cosine_restarts"
search.sampler|"grid"|"random"
search.sampler|"grid"|"bayesian"
search.policy|"wide_grid"|"coarse_then_fine_grid"
search.policy|"wide_random"|"coarse_then_fine_random"
optimizer.kind|"adam"|"sgd"
search.lr_policy|"default"|"range_test"
search.policy|"lr_only"|"lr_momentum_joint"
optimizer.rl_lr_ratio|1.0|0.1
training.epochs|5|20
training.accumulation_steps|1|4
data.feature_policy|"all_at_once"|"forward_selection"
'''


def definitions():
    result = {}
    for category, table in TABLES.items():
        for i, row in enumerate(table.strip().splitlines(), 1):
            axis, a, b = row.split('|')
            result[f'{category}-{i:03}'] = (axis, json.loads(a), json.loads(b))
    return result

# 复合条目逐操作拆分；每行会生成一个独立实验，而非一次同时改多个参数。
EXTRA = {
 'activation-003': [('model.output_contract','relu_regression','linear_regression'), ('model.output_contract','sigmoid_before_bce_logits','logits_bce')],
 'activation-004': [('model.activation','sigmoid','relu')],
 'activation-006': [('model.activation','relu','prelu')],
 'activation-007': [('loss.kind','cross_entropy','arcface')],
 'activation-011': [('model.activation','identity','relu')],
 'activation-013': [('model.weight_init','kaiming_normal','xavier_normal')],
 'batch-size-001': [('training.contrastive_negative_pool','microbatch','global_batch')],
 'batch-size-006': [('optimizer.lr',0.1,0.2)],
 'batch-size-012': [('training.accumulation_normalization','unscaled','sample_weighted'), ('training.tail_accumulation','drop','flush_sample_weighted')],
 'initialization-006': [('model.bias_init','zeros','normal')],
 'initialization-009': [('inference.ensemble_vote','hard','soft')],
 'initialization-012': [('model.init_std',0.02,0.04)],
 'normalization-002': [('model.norm','none','batch'), ('model.embedding_dropout',0.0,0.1)],
 'normalization-005': [('model.norm','batch','instance')],
 'normalization-007': [('model.norm','none','weight')],
 'regularization-002': [('model.dropout',0.1,0.0), ('inference.dropout_enabled',True,False)],
 'regularization-003': [('optimizer.weight_decay',0.01,0.001), ('optimizer.weight_decay',0.001,0.0001)],
 'regularization-005': [('training.patience',2,10), ('search.pruning','none','asha')],
 'regularization-006': [('data.augmentation','none','horizontal_flip'), ('data.augmentation','none','random_crop')],
 'regularization-011': [('data.augmentation','none','cutmix')],
 'regularization-012': [('inference.ensemble_diversity','initialization','hyperparameters'), ('inference.ensemble_diversity','initialization','architecture')],
 'infra-001': [('training.precision','fp32','fp16'), ('training.loss_scaling','none','dynamic')],
 'infra-002': [('training.sharding','ddp','zero2'), ('training.sharding','zero2','zero3')],
 'infra-004': [('search.sampler','random','bayesian')],
 'infra-009': [('data.pin_memory',False,True)],
 'infra-010': [('data.tensor_conversion','tensor_copy','as_tensor_shared')],
 'infra-013': [('checkpoint.selection_metric','val_last','val_loss')],
 'nlp-002': [('model.pooling','cls','masked_max')],
 'nlp-003': [('training.adversarial','fgm','pgd')],
 'nlp-004': [('model.cross_attention_kind','none','co_attention')],
 'nlp-010': [('data.augmentation','none','word_dropout'), ('data.augmentation','none','back_translation')],
 'nlp-012': [('data.tokenizer_kind','bpe','unigram')],
 'nlp-014': [('model.recurrent_init','normal','orthogonal_svd')],
 'recsys-001': [('checkpoint.perturb_scope','all_layers','last_layer')],
 'recsys-002': [('data.outlier_bin',False,True), ('data.bin_selection','fixed_quantiles','train_iv_selected')],
 'recsys-006': [('training.multitask_gradient','pareto','pcgrad')],
 'recsys-009': [('model.feature_gate','none','lhuc')],
 'recsys-012': [('training.batch_size',128,512)],
 'data-eval-005': [('data.split_policy','random','time')],
 'data-eval-013': [('data.shuffle',False,True)],
 'data-eval-014': [('data.augmentation','none','rotation'), ('data.augmentation','none','synonym_replace'), ('data.augmentation','none','back_translation')],
 'data-eval-015': [('loss.class_weights','none','inverse_frequency')],
 'data-eval-017': [('data.augmentation_strength',0.1,0.3), ('data.multiturn_fraction',0.1,0.5)],
 'debugging-003': [('validation.shape_check',False,True), ('validation.dtype_check',False,True), ('training.force_eval_mode',True,False), ('loss.kind','cross_entropy_double_softmax','cross_entropy')],
 'debugging-004': [('loss.numeric_implementation','naive_log_softmax','stable_logsumexp')],
 'debugging-009': [('model.depth',2,4)],
 'debugging-030': [('validation.gradcheck',False,True)],
 'llm-sft-003': [('optimizer.weight_decay',0.25,0.1)],
 'llm-sft-011': [('adapter.rank',16,32), ('adapter.alpha',16,32), ('adapter.alpha',32,64), ('adapter.dropout',0.0,0.1), ('optimizer.lr',0.0001,0.0003), ('adapter.target_modules',['q_proj','v_proj'],['q_proj','k_proj','v_proj','o_proj']), ('training.epochs',2,3), ('data.max_length',512,2048), ('training.batch_size',4,16)],
 'llm-sft-012': [('model.quantization','fp4','nf4'), ('training.precision','fp16','bf16'), ('training.gradient_checkpointing',False,True)],
 'llm-sft-015': [('data.packing_attention','cross_example','block_diagonal'), ('data.packing_positions','continuous','reset_per_example')],
 'llm-sft-016': [('model.rope_scaling','none','position_interpolation')],
 'llm-sft-017': [('data.tokenizer_contract','mismatched','identical'), ('data.preprocess_contract','mismatched','identical')],
 'llm-sft-022': [('data.deduplicate','none','train_only'), ('data.refusal_fraction',0.0,0.1), ('data.domain_fraction',1.0,0.8)],
 'llm-rl-008': [('reward.judge_policy','single','ensemble'), ('reward.length_cap',0,1024)],
 'llm-rl-021': [('reward.component_weighting','unscaled','normalized')],
 'llm-rl-022': [('rl.value_lr',0.00001,0.000001)],
 'llm-rl-027': [('optimizer.lr',0.0001,0.00001)],
 'cnn-cv-027': [('model.conv_block','plain','residual')],
 'cnn-cv-034': [('model.channels',64,96)],
 'cnn-cv-038': [('model.output_contract','relu_regression','linear_regression')],
 'cnn-cv-052': [('model.width',32,64)],
 'optimizer-lr-006': [('optimizer.dense_kind','sgd','adam')],
 'optimizer-lr-022': [('training.patience',0,5)],
 'optimizer-lr-034': [('optimizer.lr',0.0001,0.00001)],
 'optimizer-lr-035': [('optimizer.weight_decay',0.25,0.1)],
 'optimizer-lr-037': [('optimizer.beta1',0.85,0.9), ('optimizer.weight_decay',0.01,0.1), ('training.clip_norm',0.0,1.0)],
 'optimizer-lr-056': [('optimizer.weight_decay',0.1,0.001)],
 'optimizer-lr-067': [('optimizer.min_lr',0.0,0.0000001)],
 'optimizer-lr-085': [('optimizer.momentum',0.85,0.95)],
}

# 对确有错误/过强断言的原 Wiki 保留原文，并在实验中显式纠偏。
CAVEATS = {
 'batch-size-001': '梯度累积不会自动扩大 InfoNCE 单次 softmax 的负样本池；单独对照 microbatch/global_batch。',
 'initialization-002': 'He normal 的方差为 2/fan_in，标准差为 sqrt(2/fan_in)，不是 2/fan_in。',
 'initialization-006': '零初始化保持隐藏单元对称性，不能解释为打破对称性；线性模型是单独情形。',
 'debugging-002': '高容量网络可以记忆随机训练标签；判据是独立验证集接近机会水平，不能把训练拟合直接判为泄漏。',
 'debugging-003': 'PyTorch CrossEntropyLoss 接收原始 logits；BCEWithLogitsLoss 亦接收 logits，不应提前 softmax/sigmoid。',
 'llm-sft-001': '20% 是作者经验值而非普适能力门槛；不可据此宣称 SFT 理论上不可能成功。',
 'llm-sft-007': '400 条只是预注册对照样本量，不是普适充分样本定理。',
 'llm-sft-006': 'CoT 是否有用依赖任务；保持目标、token 预算并报告长度成本，不预设直接答案一定更优。',
 'llm-rl-006': '这是不同基础能力模型之间的分层比较，会受到规模与预训练差异混杂，不作单因素因果结论。',
 'cnn-cv-005': '同宽同通道下比较参数与感受野；非线性、边界效应和延时不同，不等价于相同函数。',
 'optimizer-lr-070': '推理无 optimizer.step()，没有推理学习率；此条按 Wiki 正文实际含义测试训练/推理预处理一致性。',
}
