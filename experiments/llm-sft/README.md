# llm-sft 实验索引

| 实验 | 具体干预 | A → B |
|---|---|---|
| [llm-sft-001-01](llm-sft-001-01/README.md) Prompt Engineering First（先调提示词，再做SFT） | `inference.prompt_policy` | `minimal` → `validated_instruction` |
| [llm-sft-002-01](llm-sft-002-01/README.md) 数据构造：强模型生成 + 多模型校验 | `data.label_filter` | `single_teacher` → `multi_judge_agreement` |
| [llm-sft-003-01](llm-sft-003-01/README.md) 学习率与权重衰减范围 | `optimizer.lr` | `0.0001` → `5e-05` |
| [llm-sft-003-02](llm-sft-003-02/README.md) 学习率与权重衰减范围 | `optimizer.weight_decay` | `0.25` → `0.1` |
| [llm-sft-004-01](llm-sft-004-01/README.md) 数据多样性：训练分布须覆盖测试分布 | `data.domain_sampling` | `source_only` → `target_coverage` |
| [llm-sft-005-01](llm-sft-005-01/README.md) 输出风格统一性 | `data.answer_style` | `mixed` → `canonical` |
| [llm-sft-006-01](llm-sft-006-01/README.md) SFT 不需要思维链和 Few-shot | `data.sft_format` | `cot_fewshot` → `direct_answer` |
| [llm-sft-007-01](llm-sft-007-01/README.md) 每领域约400条高质量数据即可 | `data.samples_per_domain` | `100` → `400` |
| [llm-sft-008-01](llm-sft-008-01/README.md) 课程学习：由易到难 | `data.curriculum` | `random` → `easy_to_hard` |
| [llm-sft-009-01](llm-sft-009-01/README.md) Loss 不可靠，必须看 Checkpoint 指标 | `checkpoint.selection_metric` | `val_loss` → `val_exact_match` |
| [llm-sft-010-01](llm-sft-010-01/README.md) 模型越大效果越好 | `model.scale` | `small` → `large` |
| [llm-sft-011-01](llm-sft-011-01/README.md) LoRA 调参指南 | `adapter.rank` | `8` → `16` |
| [llm-sft-011-02](llm-sft-011-02/README.md) LoRA 调参指南 | `adapter.rank` | `16` → `32` |
| [llm-sft-011-03](llm-sft-011-03/README.md) LoRA 调参指南 | `adapter.alpha` | `16` → `32` |
| [llm-sft-011-04](llm-sft-011-04/README.md) LoRA 调参指南 | `adapter.alpha` | `32` → `64` |
| [llm-sft-011-05](llm-sft-011-05/README.md) LoRA 调参指南 | `adapter.dropout` | `0.0` → `0.1` |
| [llm-sft-011-06](llm-sft-011-06/README.md) LoRA 调参指南 | `optimizer.lr` | `0.0001` → `0.0003` |
| [llm-sft-011-07](llm-sft-011-07/README.md) LoRA 调参指南 | `adapter.target_modules` | `['q_proj', 'v_proj']` → `['q_proj', 'k_proj', 'v_proj', 'o_proj']` |
| [llm-sft-011-08](llm-sft-011-08/README.md) LoRA 调参指南 | `training.epochs` | `2` → `3` |
| [llm-sft-011-09](llm-sft-011-09/README.md) LoRA 调参指南 | `data.max_length` | `512` → `2048` |
| [llm-sft-011-10](llm-sft-011-10/README.md) LoRA 调参指南 | `training.batch_size` | `4` → `16` |
| [llm-sft-012-01](llm-sft-012-01/README.md) QLoRA 的代价与稳定性 | `model.quantization` | `none` → `nf4` |
| [llm-sft-012-02](llm-sft-012-02/README.md) QLoRA 的代价与稳定性 | `model.quantization` | `fp4` → `nf4` |
| [llm-sft-012-03](llm-sft-012-03/README.md) QLoRA 的代价与稳定性 | `training.precision` | `fp16` → `bf16` |
| [llm-sft-012-04](llm-sft-012-04/README.md) QLoRA 的代价与稳定性 | `training.gradient_checkpointing` | `False` → `True` |
| [llm-sft-013-01](llm-sft-013-01/README.md) Chat Template 一致性 | `data.chat_template_contract` | `mismatched` → `identical` |
| [llm-sft-014-01](llm-sft-014-01/README.md) SFT 数据混合比例 | `data.multiturn_fraction` | `0.1` → `0.5` |
| [llm-sft-015-01](llm-sft-015-01/README.md) Packing 策略 | `data.packing` | `False` → `True` |
| [llm-sft-015-02](llm-sft-015-02/README.md) Packing 策略 | `data.packing_attention` | `cross_example` → `block_diagonal` |
| [llm-sft-015-03](llm-sft-015-03/README.md) Packing 策略 | `data.packing_positions` | `continuous` → `reset_per_example` |
| [llm-sft-016-01](llm-sft-016-01/README.md) Max Length 设置 | `data.max_length` | `512` → `2048` |
| [llm-sft-016-02](llm-sft-016-02/README.md) Max Length 设置 | `model.rope_scaling` | `none` → `position_interpolation` |
| [llm-sft-017-01](llm-sft-017-01/README.md) SFT 常见陷阱 | `loss.assistant_only` | `False` → `True` |
| [llm-sft-017-02](llm-sft-017-02/README.md) SFT 常见陷阱 | `data.tokenizer_contract` | `mismatched` → `identical` |
| [llm-sft-017-03](llm-sft-017-03/README.md) SFT 常见陷阱 | `data.preprocess_contract` | `mismatched` → `identical` |
| [llm-sft-018-01](llm-sft-018-01/README.md) 什么时候不应该用 SFT（改用 RAG） | `inference.knowledge_source` | `parameters` → `retrieval` |
| [llm-sft-019-01](llm-sft-019-01/README.md) SFT vs RL 的关系 | `training.alignment_policy` | `sft_only` → `sft_then_rl` |
| [llm-sft-020-01](llm-sft-020-01/README.md) SFT 训练轮数不宜过多 | `training.epochs` | `6` → `3` |
| [llm-sft-021-01](llm-sft-021-01/README.md) 全量微调的学习率范围 | `optimizer.lr` | `0.001` → `2e-05` |
| [llm-sft-022-01](llm-sft-022-01/README.md) SFT 数据的"难易程度"标注 | `data.curriculum` | `random` → `easy_to_hard` |
| [llm-sft-022-02](llm-sft-022-02/README.md) SFT 数据的"难易程度"标注 | `data.deduplicate` | `none` → `train_only` |
| [llm-sft-022-03](llm-sft-022-03/README.md) SFT 数据的"难易程度"标注 | `data.refusal_fraction` | `0.0` → `0.1` |
| [llm-sft-022-04](llm-sft-022-04/README.md) SFT 数据的"难易程度"标注 | `data.domain_fraction` | `1.0` → `0.8` |
