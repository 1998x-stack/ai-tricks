# 矛盾观点汇总

不同作者在深度学习调参实践中存在多处相互矛盾的观点。以下按主题分类记录。

---

## 1. Adam vs SGD：默认优化器之争

| 观点 | 主张者 |
|------|--------|
| 「无脑用Adam，对绝大多数问题都有不错的效果」 | 炫光 |
| 「Adam是最好的选择」 | 京东白条 |
| 「Adam 是默认选择，适合Transformer结构」 | hyshhh |
| 「Adam+cos退火构建基础」 | 陈城南 |
| 「无脑用sgd + momentum」 | Captain Jack |
| 「SGD+momentum能达到更好的最优点，泛化能力比Adam强1-2个点」 | 四点半的理一 |
| 「SGD+Momentum被广泛应用于各种问题领域」 | 黑马程序员 |
| 「Adam收敛快但可能陷入局部极小，泛化差」 | 黑马程序员 |
| 「Adam收敛快但sgd+momentum性能好，我用的没感觉性能好」 | 罗浩.ZJU |
| 「SGD更难训，但一般最优模型也更好」 | 陈城南 |

**分析**: 核心分歧在于 **收敛速度 vs 最终泛化性能**。Adam快速收敛但可能泛化不如SGD。折中方案是「两阶段训练」：前期Adam快速冷启动，后期切SGD精调（四点半的理一）。

> 参见: [[optimizer-lr]]

---

## 2. 批大小：越大越好 vs 未必

| 观点 | 主张者 |
|------|--------|
| 「batch size在表示学习、对比学习领域一般越大越好」 | 爱睡觉的KKY |
| 「RLHF阶段的batch size宁大勿小」 | Tang AI |
| 「GRPO训练batch size越大效果越稳定」 | Tang AI |
| 「根据硬件条件使用尽可能大的批量大小」 | 阿里云云栖号 |
| 「batch有的时候并非越大越好，有的时候减小batch有奇效」 | 炫光 |
| 「小批量有助于通过噪声注入进行规范化」 | 白小鱼 |
| 「对推荐来说32-64-128-512测试效果再高一般也不会正向了」 | 匿名作者 |
| 「64到256之间」 | 四点半的理一 |

**分析**: 大batch稳定但泛化可能差，小batch噪声大但有助于逃离局部最优。具体取决于任务类型和数据规模。

> 参见: [[batch-size]]

---

## 3. Dropout与BN能否共存

| 观点 | 主张者 |
|------|--------|
| 「如果有BN了全连接层就没必要加Dropout了」 | BBuf |
| 「Dropout, Dropout, Dropout...不仅仅可以防止过拟合，相当于Ensemble」 | Captain Jack |
| 「过拟合越严重dropout+bn加的地方就越多」 | 匿名作者 |
| 「加Dropout，加BN，加Data Argument」 | 无忧无虑 |
| 「dropout现在反而没以前那么万能，尤其是大预训练模型微调，dropout不一定要大，0到0.1常见」 | BugBuster喵 |

**分析**: BN本身有正则化效果，理论上可部分替代Dropout。但实践中很多成功案例两者共用。大模型微调场景下Dropout重要性下降。

> 参见: [[regularization]], [[normalization]]

---

## 4. 早停策略：耐心等待 vs 及时止损

| 观点 | 主张者 |
|------|--------|
| 「不要过早的early stopping，有时候收敛平台在后段，你会错过」 | 爱睡觉的KKY |
| 「如果验证集loss在几个epoch停止改善，减小学习率；如果10个epoch停止改善，停止训练」 | 黑马程序员 |
| 「当随着训练的持续在测试集上不增反降时，使用提前终止」 | 京东白条 |
| 「有些情况前面一段时间看不出起色，然后开始稳定学习」 | Captain Jack |

**分析**: 矛盾在于"耐心"的尺度。建议在充分验证模型能过拟合小数据集后，使用验证loss plateau作为早停信号，但不要过早。

> 参见: [[regularization]], [[debugging]]

---

## 5. 大模型训练中Loss的可靠性

| 观点 | 主张者 |
|------|--------|
| 「观察loss胜于观察准确率...loss是不会有这么诡异的情况发生的」 | Captain Jack |
| 「loss要比accuracy有用的多」 | 随机漫步的傻瓜 |
| 「loss对于大模型训练仅能做一个基本的判断...loss小，效果不一定好...不能太依赖loss指标」 | Tang AI |
| 「训练loss下降很舒服，但业务指标可能不动」 | BugBuster喵 |
| 「RL训练的模型更应该关注reward是否快速增长到收敛，而不是看loss」 | Tang AI |

**分析**: 小模型时代loss是可靠指标，但大模型（尤其RLHF/DPO阶段）loss与下游效果脱节严重。需要辅助以checkpoint级别的业务指标评估。

> 参见: [[llm-sft]], [[llm-rl]], [[data-eval]]

---

## 6. Embedding L2归一化：推荐 vs 反对

| 观点 | 主张者 |
|------|--------|
| 领域专家普遍推荐对embedding做L2 norm | 业界常规 |
| 「做召回的同学，不要迷信专家们说的embedding做l2 norm，笔者就踩过这个坑，直接对embedding l2结果效果贼垃圾」 | Tang AI |

**分析**: L2归一化理论上稳定了embedding空间，但实际效果可能因任务和数据而异。Tang AI的负面经验值得注意——建议在自己的数据上做对照实验。

> 参见: [[recsys]]

---

## 7. SFT训练中是否需要思维链（CoT）+ Few-shot

| 观点 | 主张者 |
|------|--------|
| 「思维链+fewshot肯定可以提高原始模型能力...但是，SFT就不用加思维链和fewshot了，直接用样本堆死它」 | Tang AI |
| 业界普遍认为CoT数据能提升推理能力 | 业界常规 |

**分析**: Tang AI的论点是在SFT阶段，与其费力标注CoT+fewshot数据，不如堆大量高质量直接问答对。但这可能取决于任务——数学/逻辑推理任务中CoT数据的价值更明显。

> 参见: [[llm-sft]]

---

## 8. 学习率起始值：众说纷纭

| 建议起始LR | 主张者 | 场景 |
|-----------|--------|------|
| 0.1 或 0.01 | 京东白条、BBuf、匿名用户 | 通用/CV |
| 1e-3 | 炫光、四点半的理一 | 通用 |
| 3e-4 ~ 5e-4 | 陈城南、hyshhh | 通用/Adam |
| 1e-4 | Tang AI | LLM SFT |
| 1e-5 级别 | 爱睡觉的KKY | NLP/BERT |
| 1e-6 ~ 2e-5 | Tang AI | LLM PPO |
| 0.3118 * C^(-0.1250) | Sam聊算法 | Scaling Law |

**分析**: 这些看似矛盾的值实际上对应不同模型规模和任务类型。核心规律：模型越大/预训练程度越高 → LR越小。Scaling Law给出了计算量C与最优LR的定量关系。

> 参见: [[optimizer-lr]]

---

## 9. 激活函数选择：ReLU vs PReLU vs LeakyReLU

| 观点 | 主张者 |
|------|--------|
| 「无脑用ReLU(CV领域)」 | Captain Jack |
| 「relu在大多数情况下表现不错」 | 炫光 |
| 「我更倾向于直接使用ReLU」 | BBuf |
| 「prelu貌似效果还不错」 | 炫光 |
| 「可以将ReLU改成PReLU试试提升精度」 | BBuf |
| 「leaky-relu...优先选择relu激活函数」 | 京东白条 |

**分析**: 一致认为ReLU是安全默认选择。PReLU/LeakyReLU在特定场景可能有提升，但不是稳定收益。没人强烈反对ReLU。

> 参见: [[activation]]

---

## 10. Xavier初始化 vs 其他初始化：统一论 vs 任务特定论

| 观点 | 主张者 |
|------|--------|
| 「无脑用xavier」 | Captain Jack |
| 「仅仅改成xavier，acc可以到98%」 | Jarvix |
| 「改为uniform，训练速度飙升，结果也飙升」 | Jarvix（对embedding） |
| 「relu激活函数初始化推荐使用He normal」 | 京东白条 |
| 「Xavier适用于tanh和sigmoid，He适用于ReLU」 | 摘星狐狸 |

**分析**: "无脑Xavier"已过时。当前共识：**激活函数决定初始化方法**——ReLU用He，tanh/sigmoid用Xavier。Embedding初始化另有考量。

> 参见: [[initialization]], [[activation]]

---

## 11. BN（批归一化）是否必须使用

| 观点 | 主张者 |
|------|--------|
| 「batchnorm一定要用」 | anchor |
| 「batchnorm也是大杀器，可以大大加快训练速度和模型性能」 | 罗浩.ZJU |
| 「构建网络时最好要加上这个组件」 | 京东白条 |
| 「batch normalization我一直没用，虽然我知道这个很好，我不用仅仅是因为我懒」 | Captain Jack |
| 「做新模型的时候，最开始不要加...batchnorm...先就纯模型」 | 匿名作者 |
| 「序列输入上LN，非序列上BN」 | 爱睡觉的KKY |

**分析**: 大多数人强烈推荐BN。但匿名作者提出了有意义的渐进式策略：先不加BN跑通纯模型，再逐步加入。且Transformer类模型LN是主流。

> 参见: [[normalization]]

---

## 12. 推荐系统中序列特征聚合：Pooling vs Attention

| 观点 | 主张者 |
|------|--------|
| 「对于推荐序列来说pooling和attention的效果差别真的不大，基本没有diff」 | Tang AI |
| 业界普遍在序列建模中偏好Attention/Transformer | 业界常规 |

**分析**: Tang AI的实证结论可能受限于具体场景。Attention理论上表达能力更强，但代价是更多参数和计算。简单场景下pooling确实可能足够。

> 参见: [[recsys]], [[nlp]]

---

## 13. Softmax vs Sigmoid 用于召回

| 观点 | 主张者 |
|------|--------|
| 「推荐召回用softmax效果比sigmoid更好...召回更适合对比学习那种listwise学习方式」 | Tang AI |
| 业界Pointwise召回常用sigmoid | 业界常规 |

**分析**: Listwise（softmax）天然适合召回场景因为它在候选集上做相对比较。Pointwise（sigmoid）更适合CTR预估等绝对打分场景。

> 参见: [[recsys]]

---

## 14. 网格搜索 vs 随机搜索 vs 贝叶斯优化

| 观点 | 主张者 |
|------|--------|
| 「Random Search比Gird Search更有效」 | 京东白条 |
| 「网格搜索会把算力浪费在不重要的组合上」 | BugBuster喵 |
| 「随机搜索通常更有效」 | BugBuster喵 |
| 「贝叶斯搜索是目前公认的比较好的自动搜索方法」 | 李文博 |
| 「暴力调参最可取」 | Captain Jack |
| 「由粗到细的随机搜索、贝叶斯优化」 | 量子位 |

**分析**: 共识是Grid Search在高维空间效率最低。推荐策略：**粗粒度随机搜索→缩小范围→贝叶斯优化精调**。

> 参见: [[infra]]

---

## 总结

以上14组矛盾中，有些是**真矛盾**（同一条件下互斥建议），有些是**条件依赖**（不同场景下的最优解不同）。读者应根据自己的具体场景——任务类型、模型规模、数据量、计算预算——来判断适用哪一方观点，最可靠的方式是在自己的数据和模型上做对照实验。

核心原则：**不要因为两个大V说了相反的话就原地不动。选一个方向，跑一组对照实验，让数据告诉你答案。**
