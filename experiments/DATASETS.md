# Hugging Face 数据与模型登记

仅核对元数据，没有下载数据分片或权重。列出的版本固定在仓库 JSON 中。当前直接接入的只有 CIFAR-10 和 UltraChat，其余是候选，需专用任务适配。

| 数据集 | 固定版本 | 状态 |
|---|---|---|
| [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10) | `0b2714987fa478483af9968de7c934580d0bb9a2` | 视觉驱动待远程验证 |
| [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) | `e6281661ce1c48d982bc483cf8a173c1bbeb5d31` | 候选；专用适配待实现 |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | `8049631c405ae6576f93f445c6b8166f76f5505a` | SFT 驱动待远程验证 |
| [trl-lib/ultrafeedback_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) | `47124cb5778f5d50de1c7676a412828f3ea7c555` | 候选；专用适配待实现 |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `740312add88f781978c0658806c59bc2815b9866` | 候选；专用适配待实现 |
| [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e` | 候选；专用适配待实现 |
| [detection-datasets/coco](https://huggingface.co/datasets/detection-datasets/coco) | `cf0b22332314a937e9dc8a1957b21725430bb41d` | 候选；专用适配待实现 |
| [zhoubolei/scene_parse_150](https://huggingface.co/datasets/zhoubolei/scene_parse_150) | `ac1c0c0e23875e74cd77aca0fd725fd6a35c3667` | 候选；专用适配待实现 |
| [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) | `7b6d24c440a36b6815f21b70d25016731768db1f` | 候选；专用适配待实现 |
| [sentence-transformers/msmarco](https://huggingface.co/datasets/sentence-transformers/msmarco) | `2a16a97b87802876f28636ca673e872248336951` | 候选；专用适配待实现 |

## 模型

SFT 候选：[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，revision `7ae557604adf67be50417f59c2c2f167def9a775`。这是待资源确认的模型配置，不代表大模型规模实验已完成。

视觉使用仓库内的 CIFAR 分辨率 ResNet，模型代码以 SHA-256 固定，不下载预训练权重。初始化/激活实验可从头训练；预训练加载/微调类技巧须另行接入预训练 profile，不应冒充完成。

## 数据边界

Amazon 评论数据不包含完整曝光未点击日志，不足以支持全部推荐系统负采样断言。COCO/场景分割/SQuAD/MS MARCO 的任务字段和评测尚未接入，不能只填仓库 ID 就认为实验可用。开放对话 ROUGE-L 不能取代人工偏好或正确性评价。

HF 官方说明：[流式读取](https://huggingface.co/docs/datasets/stream)、[版本化加载](https://huggingface.co/docs/datasets/loading)。远程仍可能缓存元数据；模型权重和 checkpoint 只允许保存在远程。
