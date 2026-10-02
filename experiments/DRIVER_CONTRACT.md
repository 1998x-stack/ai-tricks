# 真实训练驱动接口

每个实验入口负责配对运行、预算、来源记录、结果校验和绘图；技巧本身由驱动实现。**没有实现的参数必须报错**，不能仅在 JSON 中回显后跑一个与该技巧无关的模型。

当前驱动：[视觉](drivers/vision.py)、[SFT](drivers/sft.py)。覆盖状态见 [逐项表](catalog/DRIVER_STATUS.md)。现有驱动尚未经过远程 GPU 验证，不能称为已跑通真实训练。

## 输入

驱动由参数数组启动（不经过 shell）：

```text
python -m your_driver --request /remote/run/request.json --output /remote/run
```

请求含 `experiment_id`、`plan_hash`、`seed`、`arm`、唯一 `axis`、完整扁平 `config`、主指标 `metric`、`direction`、数据和模型完整 revision、`profile`。

`profile` 指定 HF dataset/model、明确的 train/validation/test 切分、列名、固定 baseline、步数/epoch/生成长度预算，以及主机白名单。数据读取前调用 `require_remote(profile)`；不要在模块 import 时触网。驱动只能下载/缓存到指定远程目录。

## 结果

`result.json` 必须回传请求中的所有来源字段，并提供：

- `applied_config`：真正实施的完整配置，与请求精确一致。
- `data_kind = "huggingface_real"`。
- `train_examples_seen > 0`、`optimizer_steps > 0`，不能拿未经训练的评测冒充完整训练。
- `metrics`：有限数值；必须有协议声明的主指标。
- `test_ids_sha256`：按排序后样本 ID 用换行连接的 UTF-8 SHA-256。

这些字段是执行契约，不是凭空证明数据真实性的认证机制。审核驱动代码和保留训练日志仍然必要。

`history.csv` 至少包含 `step,train_loss,val_loss`。`predictions.jsonl` 每行必须有唯一 `id`，保存真实标签/参考及预测。验证器重算预测 ID 摘要并检查两个 arm 相等。

`checkpoint_manifest.json` 必須有 `selected_by_validation=true`、`path`、`sha256`、选择步数和验证指标；checkpoint 必须在本次输出目录下存在且哈希匹配。不得拿测试集选 checkpoint。

## 每类驱动的额外工作

- 检测：真实框标注、matching/NMS、mAP 和 FP，不能用分类准确率代替。
- 分割：像素标签、void mask、mIoU 和边界指标。
- 推荐：用户/物品/时间切分、去除正样本污染、曝光与随机负样本的可辨识来源、NDCG/Recall；评论数据不能直接冒充曝光日志。
- DPO：固定参考策略、chosen/rejected 条件 log-prob、偏好数据质量控制。
- PPO/GRPO：真实 rollout、规则/奖励模型、KL、策略熵及独立任务正确率。GRPO 不能代替含 critic 的 PPO 实验。
- 故障诊断：预注册故障清单，在同一真实训练流水线上注入故障，记录检出率、误报率和耗时。
- 搜索策略：相同 trial/token/GPU 时间预算，单独锁定最终测试集；不直接复用普通单次训练。
- 基础设施：真实 GPU/多卡执行与同步计时；内存公式或 CPU 模拟不代表 GPU 结果。

驱动完成后，把实际支持字段写入资源配置及 `tools/audit.py` 的能力检查。仅更新 `supported_axes` 列表而没有代码实现不算完成。
