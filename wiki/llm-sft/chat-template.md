# Chat Template 对话模板

## 是什么

Chat Template（对话模板）是将结构化对话消息序列转换为模型可处理的 token 序列的格式规范。它定义了如何将 system prompt、user 消息、assistant 消息、工具调用等不同类型的消息按特定顺序排列，并插入特殊 token 来标记角色边界。

一个典型的 chat template 定义了以下映射：

```
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is LoRA?"},
    {"role": "assistant", "content": "LoRA is..."},
]
```

不同的模型使用不同的模板格式：

- **LLaMA 系列**（ChatML 变体）：`<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`
- **Mistral / LLaMA 原生**：`<s>[INST] ... [/INST] ... </s>` 或 `[INST] ... [/INST]`
- **DeepSeek**：`<|system|>...<|end|>\n<|user|>...<|end|>\n<|assistant|>...<|end|>`
- **Qwen / Yi**：`<|im_start|>system\n...<|im_end|>\n`（类似 LLaMA 的 ChatML 变体，但 token id 不同）
- **原始格式**：无特殊 token，直接拼接字符串。

Chat template 通常通过 Hugging Face 的 `tokenizer.apply_chat_template()` 方法应用，该方法在模型配置中预定义了模板字符串。

## 为什么重要

Chat template 一致性是大模型微调中最隐蔽、最容易忽视但又破坏性最大的问题之一：

- **训练和推理必须使用完全相同的 template**：如果训练时用一种模板格式，推理时用另一种，模型的输出会出现严重的格式混乱——多余的特殊 token、角色标签显示在回答中、回答被截断、甚至完全无法工作。
- **Loss 无法反映这个问题**：这是最危险的地方。Chat template 不一致时，训练 loss 仍然会正常下降，看起来一切正常，但模型的生成结果一塌糊涂。因为模型学会了在每个位置预测下一个 token，而"下一个 token"受 template 格式直接影响。
- **不同模型对 template 的敏感度不同**：有些模型（如经过大量 ChatML 格式训练的模型）对 template 极其敏感，模板稍有不同就会导致生成退化；有些模型（如原始 LLaMA）相对鲁棒，但仍需保持一致。
- **Tokenizer 内置了 template 信息**：Hugging Face 的 tokenizer 在 `tokenizer_config.json` 中保存了 `chat_template` 字段。很多人在加载模型时忽略了 check 这个字段是否正确。

## 如何使用

### 训练前的 template 确认步骤

在开始任何 SFT 训练前，执行以下检查：

1. **加载 tokenizer，确认 chat_template 字段**：检查 `tokenizer.chat_template` 是否非空。如果为空，大部分框架会回退到默认格式，但这个默认格式可能不是你想要的。

2. **确认 template 与模型设计匹配**：不是所有 template 都适用于所有模型。某些模型的 tokenizer 词表中没有对应的特殊 token（如 `<|im_start|>`），或对应的 token id 不同。如果在 `tokenizer_config.json` 中找不到定义，尝试从模型官方仓库获取。

3. **写一个端到端的验证函数**：用同一个 prompt 分别经过训练时的 template 和推理时的 template，输出结果应当完全一致：
   ```python
   # 训练时的 template
   train_template = tokenizer.apply_chat_template(messages, tokenize=False)
   # 推理时的 template
   infer_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   # 注意：推理时的 template 通常多了 add_generation_prompt，去掉了 assistant 消息和结束 token
   ```

4. **将 template 写入训练配置文件**：不要依赖默认值。在训练脚本中显式指定 template，并通过 CI/CD 流水线中的单元测试验证训练和推理的一致性。

### 常见 template 的选择

选择 template 时的决策依据：

- **如果微调目标是某个特定模型家族**：使用该模型官方定义的 template。例如微调 LLaMA-3 应使用 LlamaTokenizer 默认的 ChatML 格式。
- **如果微调目标是通用对话能力**：使用对话能力最强的 template 格式。ChatML (`<|im_start|>`, `<|im_end|>`) 是目前最广泛采用的格式，兼容性好。
- **自定义格式**：如果你有自己的特殊需求（如需区分不同角色、多工具调用），可以设计自定义 template，但必须保证训练和推理一致，且 tokenizer 词表中包含所有特殊 token。

### 训练时的 Attention Mask 设置

在训练中，chat template 必须与 attention mask 和 loss mask 配合：

- System prompt 和 user 消息应包含在上下文内（attention mask = 1），但不计算 loss（loss mask = 0）。
- Assistant 回复部分计算 loss（loss mask = 1）。
- 特殊 token（如 `<|im_end|>`）通常不计算 loss，但这不是严格必需的——取决于具体实现。
- Loss mask 错误是 SFT 排在前列的 bug：如果 user 部分也被计入 loss，模型会学习"预测用户的输入"，严重干扰训练效果。

### 验证步骤

每次修改 template 或切换模型后，运行以下验证：

1. 对一组测试消息序列，用训练时的 template 方式格式化。
2. 用推理时的 template 方式（含 `add_generation_prompt=True`）格式化。
3. 确认 prefix 一致——推理模板生成的 token 序列的前缀应与训练模板的对应部分匹配。
4. 检查特殊 token 的 id——确保 `tokenizer.eos_token_id`, `tokenizer.bos_token_id`, `tokenizer.pad_token_id` 等在训练和推理中一致。

## 常见误区

### "同一种 template 在所有模型中通用"

不同的模型可能实现了同一个 template 格式但使用了不同的 token id。例如，A 模型和 B 模型都支持 ChatML 格式，但 `<|im_start|>` 在 A 的 tokenizer 中可能是 token id 32000，在 B 中可能是 32010。使用错一个 tokenizer 就能让模型输出乱码。

### "Tokenizer 的默认值就是正确的"

Tokenizer 的 `chat_template` 字段在不同版本的 transformers 库中可能不同。模型下载时的 `tokenizer_config.json` 可能被覆盖，或者 Hugging Face 社区的某些 tokenizer 上传时根本没设置 chat_template。始终显式检查。

### "add_generation_prompt 无关紧要"

`add_generation_prompt=True` 会在推理 template 末尾添加准备让模型开始生成的标识（如 `\n<|im_start|>assistant\n`）。如果训练时使用了这个标识而推理时没有，或者反过来，模型的输出开头可能会缺失或多余一些 token。

### "训练和推理可以用不同的 frameworks"

很多人训练时用 Hugging Face Trainer，推理时用 vLLM 或 TGI，然后发现模型输出不对。不同推理框架对 chat template 的自动处理可能不同——vLLM 默认使用 tokenizer 的 chat_template，但如果你手动禁用了它，或者 TGI 的 `add_special_tokens` 参数不同，结果就会不一致。

### "Fine-tune 后换一个 template 重新训练就能解决"

在微调后更换 template 意味着需要重新训练。特别是一些模型在微调过程中已经"记住了"训练时的 template 格式，换一个 template 会让模型感到混乱。最好第一次微调前就确定好 template。

### "Chat template 和 tokenizer 是独立的"

Chat template 与 tokenizer 深度绑定。特殊 token 必须存在于 tokenizer 的词表中，否则 tokenizer 无法正确处理。如果自定义 template 使用了新的特殊 token，需要先通过 `tokenizer.add_special_tokens()` 将其加入词表，并调整模型 embedding 层。

## 参见

- [[lora]] — LoRA 微调中 template 一致性的重要性
- [[../infra/infra]] — 训练基础设施中的 tokenizer 和 template 配置
- [[../llm-rl/llm-rl]] — RL 阶段同样需要 chat template 一致性
- [[../optimizer-lr/optimizer-lr]] — 不同训练阶段的 template 与学习率配合
- [[../../contradictory]] — 关于不同 template 格式优劣的不同观点
- [[sft-data-construction]] — SFT 数据格式与 template 的配合
