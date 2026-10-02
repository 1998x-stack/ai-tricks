# 中文离线阅读文件

- [EPUB 电子书](AI-Tricks-中文实验手册.epub)：收录 72 个 Wiki 页面、517 个实验协议及 6 篇方法/来源说明。分级目录按阅读方法、14 个知识领域和 14 个实验附录编排；116 处公式使用原生 MathML。内部知识链接离线可用。
- [XMind 思维导图](AI-Tricks-中文思维导图.xmind)：15 张画布（总览＋14 个领域），包含 412 条来源技巧及全部 517 个 A/B 协议。展开技巧查看来源、场景、具体要点、干预变量、A/B 值、指标、固定条件、产物和解释边界；完整文本保存在节点备注。

EPUB 可导入 Apple Books、Calibre 等支持 EPUB 3 的阅读器；公式显示取决于阅读器的 MathML 支持。XMind 文件采用原生 JSON ZIP 格式，在支持该格式的新版 XMind 中打开。总览节点链接可跳转到相应分类画布；技巧默认折叠，避免一次展开数千个节点。

原作者归因随正文保留，原始回答版权归各位作者。外部 GitHub 来源/代码链接需要联网，指向 main 分支；未发布的变更需合入后才可在线查看。本书与导图均明确记录真实训练完成数为 0，协议不等于效果结论，也没有附加任何模型权重或训练数据。

## 重建

需要 Python 3.11+、`site/requirements.txt` 的依赖及 Pandoc（本次使用 3.10）。

```bash
python tools/build_exports.py                  # 两种格式
python tools/build_exports.py --format epub    # 仅电子书
python tools/build_exports.py --format xmind   # 仅导图
python -m unittest discover -s tests -p test_exports.py -v
```

生成器从现有 Wiki 与 catalog 读取，不下载训练数据。XMind ID 与压缩时间固定，便于重复生成；EPUB 由 Pandoc 打包并保留原生公式。

自动检查覆盖 ZIP 完整性、EPUB XML/清单及所有内部锚点、595 份来源文档、116 处公式、XMind 唯一节点 ID、跨画布跳转和 412/517 项覆盖。当前环境未安装 XMind 桌面应用，因此没有声称完成该应用的实际打开验收。

本次新增 3 项导出回归检查，完整测试集共 40 项全部通过；人工查看封面及解包后的章节排版。验收范围为结构、链接与浏览器渲染，未完成 EPUBCheck 或各款阅读器的逐一验收。
