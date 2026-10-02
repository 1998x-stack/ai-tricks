# 中文 GitHub Pages：建设、预览与验收

站点由仓库内容离线生成，采用中文技术手册的阅读布局：宋体标题、深绿色强调、米白背景、章节编号、正文目录、来源回链与状态说明。重点是具体知识与实验协议，不用虚构实验曲线装饰页面。

## 内容与功能

| 内容 / 能力 | 实现 |
|---|---|
| 首页 | 问题入口、领域地图、知识规模、真实完成状态 |
| 知识正文 | 14 章与 58 篇详情，作者归因、引用、目录和关联实验 |
| 实验目录 | 517 个独立协议详情；关键词、领域、驱动状态筛选；每页 30 项 |
| 全文搜索 | 构建期生成正文索引，中文/英文/参数名多词匹配；不上传查询 |
| 方法与来源 | 对照设计、数据隔离、预算、主指标、置信区间边界、校订和原始资料 |
| 数学公式 | 构建期转为原生 MathML，无需加载外部公式 CDN 或开启 JavaScript |
| 代码阅读 | 等宽字体、局部横向滚动、复制按钮；代码中的 wikilink 保持原样 |
| 手机与键盘 | 折叠菜单、跳转正文、可见焦点、搜索快捷键、Escape 关闭、减少动态效果 |
| 无 JavaScript | 知识和协议完整可读，517 项目录完整展示；搜索页提示浏览目录 |
| 静态资产 | CSS、JS、SVG 和检索索引全部同源，无第三方字体和分析追踪 |

搜索基于子串匹配，不做中文语义召回。不存在的 Wiki 引用显示为带解释的普通文字；构建测试检查所有真正生成的内部链接及锚点。网站没有远程训练结果，页面将协议示意和效果结论明确区分。

## 构建与预览

需要 Python 3.11+。这些命令只安装网页构建依赖，不下载模型或训练数据。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r site/requirements.txt
.venv/bin/python tools/build_site.py --output .site-preview/ai-tricks --base-url /ai-tricks
python3 -m http.server 8765 --directory .site-preview
```

浏览 `http://localhost:8765/ai-tricks/`。项目子路径与部署路径保持一致，能及时暴露资源路径错误。`--base-url ''` 可构建根路径站点。默认输出 `_site/`；构建目录、浏览器截图与训练输出均不纳入 Git。

构建入口为 `tools/build_site.py`；版式和交互分别位于 `assets/site.css`、`assets/site.js`。编辑 Wiki / 实验来源后，先重新生成实验目录并完成审计，再构建站点。新增页面需进入 `Site.pages` 映射，才会生成阅读页与搜索条目。

构建不会清理输出目录中的旧文件；发布从 Actions 干净 checkout 开始。本地调整路由后应使用新的输出目录，防止旧页面掩盖链接问题。

## 自动化检查

```bash
# 完整回归：统计、绘图、张量语义和全站静态检查。
# CPU torch 只用于手工构造的测试张量，不加载 HF 权重。
python -m pip install numpy matplotlib torch -r site/requirements.txt
python -m unittest discover -s tests -v
python -m experiments.tools.audit

# 可选浏览器依赖：仅下载 Chromium 浏览器，不是训练数据。
python -m pip install -r site/browser-requirements.txt
python -m playwright install chromium
python tools/build_site.py --output .site-preview/ai-tricks
python site/browser_checks.py --root .site-preview
```

静态检查覆盖：每个来源页面/实验的 HTML 和索引覆盖、唯一页面主标题、中文语言、项目子路径、全站内部文件及锚点、公式渲染、代码隔离和真实状态。浏览器检查覆盖：桌面截图、320/390px 页面溢出、关键词/类别/状态筛选、分页、URL 恢复、中文全文搜索、键盘焦点、Escape、无 JavaScript 阅读，以及浏览器与 HTTP 错误。截图保存到 `site-test-results/`，供人工视觉复查。

## GitHub Pages 工作流

`.github/workflows/pages.yml` 在 push / PR 时执行离线审计、单元测试、静态构建和 Chromium 验收；仅 `main` 分支通过后部署。部署作业使用 `github-pages` environment，权限限于 `pages: write` 和 `id-token: write`。PR 只验证，不部署。

仓库需将 Settings → Pages → Build and deployment → Source 设为 **GitHub Actions**。旧的 Cayman/Jekyll `_config.yml` 已移除；当前发布的是生成的 HTML，不能再选择分支 Jekyll 构建。

本次修改只在本地构建、测试和分开提交，没有推送或修改线上 Pages 设置。待变更合入 main、设置发布源且工作流成功后，预期访问地址为 `https://1998x-stack.github.io/ai-tricks/`；该地址不是本次线上验收报告。

官方流程依据：[GitHub 自定义 Pages 工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)、[部署 action](https://github.com/actions/deploy-pages)。公式转换使用 [latex2mathml](https://github.com/roniemartinez/latex2mathml)，版本固定在 requirements 中。
