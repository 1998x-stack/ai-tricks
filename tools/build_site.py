"""从真实 Wiki/协议构建完整中文静态站；不依赖 Jekyll 插件、不访问网络。"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import markdown
from latex2mathml.converter import convert as mathml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.tools.audit import implemented

CATEGORIES = {
 'optimizer-lr':('优化器与学习率','Adam · SGD · Warmup · 学习率调度'),
 'batch-size':('批大小与梯度累积','有效批量 · 尾批处理 · 梯度归一化'),
 'initialization':('参数初始化','Xavier · He · 正交初始化'),
 'normalization':('归一化方法','BatchNorm · LayerNorm · GroupNorm'),
 'activation':('激活函数','ReLU · PReLU · 输出层设计'),
 'regularization':('正则化与泛化','Dropout · 早停 · 标签平滑'),
 'cnn-cv':('CNN 与视觉任务','卷积 · 感受野 · 特征融合'),
 'nlp':('NLP 与序列建模','BERT · 词向量 · 文本增强'),
 'recsys':('推荐与排序系统','负采样 · 热启动 · 多任务学习'),
 'llm-sft':('大模型监督微调','LoRA · 数据构造 · Chat Template'),
 'llm-rl':('大模型强化学习','PPO · DPO · GRPO · 奖励设计'),
 'infra':('训练工程与基础设施','混合精度 · 分布式 · 实验追踪'),
 'data-eval':('数据与评估','数据切分 · 类别不平衡 · 评测指标'),
 'debugging':('诊断与故障排查','小样本检查 · NaN · 梯度与曲线'),
}
E = html.escape


def slug(value, separator='-'):
    return re.sub(r'[-\s]+',separator,re.sub(r'[^\w\s-]','',value.strip().lower())).strip(separator) or 'section'


def strip_text(value):
    return re.sub(r'\s+',' ',re.sub(r'[`#*\[\]<>]','',value)).strip()


class Site:
    def __init__(self,output,base):
        self.output=Path(output)
        self.base='/' + base.strip('/') if base.strip('/') else ''
        self.specs=json.loads((ROOT/'experiments/catalog/experiments.json').read_text())
        self.tricks=json.loads((ROOT/'experiments/catalog/tricks.json').read_text())
        self.coverage=json.loads((ROOT/'experiments/catalog/coverage.json').read_text())
        self.ready={s['id']:implemented(s) for s in self.specs}
        self.pages={}
        for path in sorted((ROOT/'wiki').glob('*/*.md')):
            self.pages[path.resolve()]=path.relative_to(ROOT).with_suffix('.html').as_posix()
        for spec in self.specs:
            category=spec['source']['wiki'].split('/')[1]
            path=ROOT/'experiments'/category/spec['id']/'README.md'
            self.pages[path.resolve()]=f'experiments/{spec["id"]}/index.html'
        for category in CATEGORIES:
            self.pages[(ROOT/f'experiments/{category}/README.md').resolve()]='experiments/index.html?category='+category
        for source,target in [('experiments/README.md','guide/index.html'),('experiments/DRIVER_CONTRACT.md','drivers/index.html'),
                              ('experiments/DATASETS.md','datasets/index.html'),('experiments/ERRATA.md','errata/index.html'),
                              ('contradictory.md','contradictions/index.html'),('site/content/methodology.md','methodology/index.html')]:
            self.pages[(ROOT/source).resolve()]=target
        self.titles={}
        self.search=[]
        self.written=[]

    def url(self,path=''):
        path=path.removesuffix('index.html') if '?' not in path else path.replace('index.html?','?')
        return self.base+'/'+path.lstrip('/')

    def wiki_url(self,category):
        return self.url(f'wiki/{category}/{category}.html')

    def source_url(self,path):
        return 'https://github.com/1998x-stack/ai-tricks/blob/main/'+quote(str(path.relative_to(ROOT)).replace(os.sep,'/'))

    def resolve(self,href,source):
        decoded=html.unescape(href)
        parsed=urlsplit(decoded)
        if parsed.scheme or decoded.startswith('//') or decoded.startswith('#'):
            return decoded
        destination=(source.parent/unquote(parsed.path)).resolve()
        if destination in self.pages:
            link=self.url(self.pages[destination])
            return link+('#'+slug(unquote(parsed.fragment)) if parsed.fragment else '')
        if destination.is_file() and destination.is_relative_to(ROOT):
            return self.source_url(destination)+('#'+parsed.fragment if parsed.fragment else '')
        return None

    def render(self,path):
        text=path.read_text()
        text=re.sub(r'\A---\n.*?\n---\n','',text,flags=re.S)
        source_text=strip_text(text)
        title=re.search(r'^# (.+)',text,re.M)
        title=title[1] if title else path.stem
        formulas = {}
        def math(segment):
            def replace(match):
                value = match.group(0)
                block = value.startswith('$$')
                expression = value[2:-2] if block else value[1:-1]
                token = f'MATHSITETOKEN{len(formulas)}END'
                rendered = mathml(expression.strip(), display='block' if block else 'inline')
                formulas[token] = f'<div class="math-block">{rendered}</div>' if block else rendered
                return '\n\n'+token+'\n\n' if block else token
            return re.sub(r'\$\$[\s\S]+?\$\$|(?<!\\)\$(?!\$)[^$\n]+?(?<!\\)\$', replace, segment)
        # 只转换普通文本中的 wikilinks/数学公式，跳过 fenced code/inline code。
        def links(segment):
            def replace(match):
                target,_,alias=match[1].partition('|')
                filename,sep,fragment=target.partition('#')
                link=self.resolve((filename+'.md' if filename else '')+(sep+fragment),path)
                destination=(path.parent/(filename+'.md')).resolve() if filename else path
                if destination.is_file() and destination not in self.titles:
                    heading=re.search(r'^# (.+)',destination.read_text(),re.M)
                    self.titles[destination]=heading[1] if heading else filename
                label=alias or fragment or self.titles.get(destination,filename)
                if link:
                    return f'<a href="{E(link,quote=True)}">{E(label)}</a>'
                return f'<span class="unresolved" title="来源中引用的条目，仓库暂无对应页面">{E(label)}</span>'
            return re.sub(r'\[\[([^\]]+)\]\]',replace,segment)
        chunks=re.split(r'(```[\s\S]*?```|`[^`\n]+`)',text)
        text=''.join(chunk if i%2 else math(links(chunk)) for i,chunk in enumerate(chunks))
        md=markdown.Markdown(extensions=['fenced_code','tables','toc','sane_lists'],
                             extension_configs={'toc':{'slugify':slug,'toc_depth':'2-3'}})
        body=md.convert(text)
        for token, rendered in formulas.items():
            body = body.replace('<p>'+token+'</p>', rendered).replace(token, rendered)
        # 已经转换成站点 URL 的 wikilink 不再重写。
        def href(match):
            target=html.unescape(match[1])
            if target.startswith(self.base+'/') and not target.startswith('//'):
                return match[0]
            converted=self.resolve(target,path)
            return f'href="{E(converted,quote=True)}"' if converted else match[0]
        body=re.sub(r'href="([^"]+)"',href,body)
        body=re.sub(r'^<h1[^>]*>.*?</h1>\s*','',body,count=1,flags=re.S)
        return title,body,md.toc.replace('<div class="toc">','<div class="toc-list">'),source_text

    def shell(self,title,body,route,active='',description='有来源、可检验的深度学习调参知识与 A/B 实验协议。'):
        nav=[('topics/','知识目录','topics'),('experiments/','实验目录','experiments'),('methodology/','实验方法','methodology'),('sources/','来源与校订','sources')]
        links=''.join(f'<a href="{self.url(url)}"'+(' aria-current="page"' if key==active else '')+f'>{label}</a>' for url,label,key in nav)
        search=f'''<dialog class="search-dialog" id="search-dialog" aria-labelledby="search-dialog-title"><div class="dialog-head"><h2 id="search-dialog-title">检索知识与实验</h2><button type="button" data-close-search aria-label="关闭搜索">关闭 · Esc</button></div><label for="dialog-search" class="search-hint">术语、方法或实验编号</label><input class="field" id="dialog-search" type="search" placeholder="搜索 72 个知识页面与 517 个实验…" autocomplete="off"><p class="search-hint" role="status" aria-live="polite">输入关键词开始检索，支持中文与参数名。</p><ul class="search-results"></ul></dialog>'''
        return f'''<!doctype html><html lang="zh-CN" data-base="{E(self.base)}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light"><title>{E(title)} · AI Tricks</title><meta name="description" content="{E(description,quote=True)}"><link rel="canonical" href="https://1998x-stack.github.io{self.url(route)}"><meta property="og:title" content="{E(title,quote=True)} · AI Tricks"><meta property="og:description" content="{E(description,quote=True)}"><meta property="og:type" content="article"><meta property="og:locale" content="zh_CN"><meta name="theme-color" content="#216653"><link rel="icon" href="{self.url('assets/favicon.svg')}" type="image/svg+xml"><link rel="stylesheet" href="{self.url('assets/site.css')}"><script src="{self.url('assets/site.js')}" defer></script></head><body><a class="skip" href="#main">跳转到正文</a><header class="topbar"><div class="container header-inner"><a class="brand" href="{self.url()}" aria-label="AI Tricks 首页"><span class="brandmark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><span><strong>AI TRICKS</strong><small>深度学习实验手册</small></span></a><nav class="nav" id="main-nav" aria-label="主导航">{links}</nav><div class="header-tools"><a class="search-trigger" href="{self.url('search/')}" data-open-search>搜索 <span class="key">⌘ K</span></a><a href="https://github.com/1998x-stack/ai-tricks">GitHub ↗</a><button class="menu-toggle" type="button" aria-expanded="false" aria-controls="main-nav">菜单</button></div></div></header>{body}<footer class="footer"><div class="container footer-inner"><div>AI TRICKS <span aria-hidden="true"> / </span> 从经验出发，以证据判断。<br>内容保留原作者归因 · 不预设技巧普适有效</div><div class="footer-links"><a href="{self.url('guide/')}">阅读与运行指南</a><a href="{self.url('datasets/')}">数据登记</a><a href="{self.url('sources/')}">来源与覆盖状态</a><a href="https://github.com/1998x-stack/ai-tricks">参与修订 ↗</a></div></div></footer>{search}</body></html>'''

    def write(self,route,content):
        path=self.output/route
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content,encoding='utf-8')
        self.written.append(route)

    def topics(self):
        rows=[]
        for number,(key,(title,description)) in enumerate(CATEGORIES.items(),1):
            rows.append(f'<a class="topic-row" href="{self.wiki_url(key)}"><span class="topic-number">{number:02}</span><div><h3>{title}</h3><p>{description}</p></div><span class="topic-count">{self.coverage["categories"][key]} 条</span></a>')
        return '<div class="topic-grid">'+''.join(rows)+'</div>'

    def home(self):
        count=len(self.specs); ready=sum(self.ready.values()); pending=count-ready
        body=f'''<main id="main" class="container"><section class="hero"><div><div class="eyebrow">THE DEEP LEARNING FIELD GUIDE</div><h1>让每一次调参，<br><em>都有据可查。</em></h1><p class="hero-description">从优化器到大模型对齐，把分散的实战经验整理成可追溯的知识，把每一个具体操作写成可检验的 A/B 对照。</p><div class="actions"><a class="button" href="{self.url('topics/')}">开始阅读 <span aria-hidden="true">↗</span></a><a class="text-link" href="{self.url('experiments/')}">浏览实验目录 →</a></div></div><aside class="protocol-panel" aria-label="A/B 实验协议示意"><div class="panel-top"><b>EXPERIMENT / 001</b><span>协议示意 · 非实验结果</span></div><h2>一次，只回答一个问题。</h2><p class="panel-caption">相同数据、初始化与预算，比较隐层激活函数。</p><div class="ab-row"><span class="ab-letter">A</span><code>activation = "sigmoid"</code><small>对照组</small></div><div class="ab-row"><span class="ab-letter">B</span><code>activation = "relu"</code><small>实验组</small></div><div class="panel-flow"><span>固定种子</span><span>→</span><span>配对训练</span><span>→</span><span>独立评估</span></div><div class="panel-bottom"><span>SEEDS / 17 · 42 · 2026</span><span>保留正结果，也保留负结果</span></div></aside></section><section class="metrics-strip" aria-label="知识库规模"><div class="stat"><strong>14</strong><span>主题领域</span></div><div class="stat"><strong>72</strong><span>知识页面</span></div><div class="stat"><strong>{self.coverage['main_tips']}</strong><span>来源技巧</span></div><div class="stat"><strong>{count}</strong><span>细化实验协议</span></div></section><section class="section"><div class="section-heading"><div><span class="section-index">01 / START WITH A QUESTION</span><h2>从你正在解决的问题开始</h2></div><a class="text-link" href="{self.url('guide/')}">阅读指南 →</a></div><div class="guide-grid"><a class="guide-card" href="{self.wiki_url('debugging')}"><span class="number">01 · 训练诊断</span><h3>Loss 不降，先检查哪里？</h3><p>从小样本检查、参数更新到数值稳定性，建立排查顺序。</p><span class="arrow">↗</span></a><a class="guide-card" href="{self.wiki_url('optimizer-lr')}"><span class="number">02 · 优化与泛化</span><h3>学习率，应该如何比较？</h3><p>区分初始值、调度与 batch 的影响，让对照真正有效。</p><span class="arrow">↗</span></a><a class="guide-card" href="{self.wiki_url('llm-sft')}"><span class="number">03 · 大模型微调</span><h3>LoRA 之外，数据做对了吗？</h3><p>查看样本构造、回复 mask、模板一致性与评测边界。</p><span class="arrow">↗</span></a></div></section><section class="section"><div class="section-heading"><div><span class="section-index">02 / KNOWLEDGE ATLAS</span><h2>一张完整的调参知识地图</h2></div><p>基础方法 → 领域实践 → 训练工程</p></div>{self.topics()}</section><section class="section"><div class="section-heading"><div><span class="section-index">03 / EVIDENCE, NOT ASSUMPTION</span><h2>把完成状态，也放在明处</h2></div></div><div class="evidence-box"><div><h2>协议不等于证据。</h2><p>当前所有实验均未完成真实远程训练。基础驱动的配置兼容性，只是执行前的一项检查；场景适配、专用实现和实际结果分别记录。</p><a class="text-link" href="{self.url('sources/')}">查看完整覆盖说明 →</a></div><div><div class="progress-label"><span>基础驱动配置兼容</span><strong>{ready} / {count}</strong></div><div class="progress-track" aria-label="{ready} 个配置兼容，{pending} 个仍需专用实现"><span style="width:{ready/count*100:.2f}%"></span></div><div class="progress-label"><span>专用训练驱动待实现</span><strong>{pending}</strong></div><p class="status-note">真实训练完成：0 · 尚无可报告的效果提升</p></div></div></section></main>'''
        self.write('index.html',self.shell('深度学习实验手册',body,''))
        body=f'<main id="main" class="container"><div class="page-head"><div class="eyebrow">KNOWLEDGE ATLAS</div><h1>知识目录</h1><p class="lede">14 个领域，72 个页面。主页面保留具体技巧与作者引文，知识详情解释机制、使用步骤和适用边界。</p></div>{self.topics()}</main>'
        self.write('topics/index.html',self.shell('知识目录',body,'topics/','topics'))

    def sidebar(self,active=''):
        links=''.join(f'<a class="{"active" if key==active else ""}" href="{self.wiki_url(key)}">{name}</a>' for key,(name,_) in CATEGORIES.items())
        return f'<aside class="sidebar" aria-label="知识目录"><p class="sidebar-title">KNOWLEDGE / 14 TOPICS</p>{links}<div class="divider"></div><a href="{self.url("experiments/")}">A/B 实验目录 →</a><a href="{self.url("methodology/")}">实验方法与解释边界 →</a></aside>'

    def document(self,path,route,spec=None):
        title,body,toc,text=self.render(path)
        category=spec['source']['wiki'].split('/')[1] if spec else path.parent.name
        category_name=CATEGORIES.get(category,('项目说明',''))[0]
        breadcrumb=f'<a href="{self.url()}">首页</a><span>/</span><a href="{self.url("topics/")}">{category_name}</a><span>/</span><span>{E(title)}</span>'
        related=''; notice=''; active='methodology' if route.startswith('methodology/') else ('sources' if route.startswith(('errata/','datasets/','drivers/','contradictions/')) else 'topics')
        if spec:
            active='experiments'
            status='基础驱动配置兼容，场景待复核' if self.ready[spec['id']] else '专用训练驱动待实现'
            notice=f'<div class="notice"><strong>执行状态：{status}。</strong>本页是预注册实验协议，未产生真实训练日志、图表或效果结论。</div>'
            source=ROOT/spec['source']['wiki']
            source_link=self.url(self.pages[source.resolve()])+'#'+slug(spec['source']['heading'])
            related=f'<section class="related"><h2>返回来源技巧</h2><a href="{E(source_link)}">{E(spec["source"]["heading"])} →</a><p>Wiki 作者归因：{E(spec["source"]["author"])}</p></section>'
        else:
            relatives=[s for s in self.specs if s['source']['wiki']==str(path.relative_to(ROOT))]
            if relatives:
                links=''.join(f'<li><a href="{self.url("experiments/"+s["id"]+"/")}"><code>{s["id"]}</code>{E(s["title"])} · {E(s["axis"])}</a></li>' for s in relatives)
                related=f'<section class="related"><h2>本章对应实验 · {len(relatives)}</h2><p>查看具体 A/B 取值、评价指标和当前执行状态。</p><ul>{links}</ul></section>'
        reading=max(1,len(text)//600)
        content=f'<main id="main" class="container doc-shell">{self.sidebar(category)}<div class="document"><div class="page-head"><div class="breadcrumb">{breadcrumb}</div><div class="eyebrow">{"EXPERIMENT PROTOCOL" if spec else "FIELD NOTES"}</div><h1>{E(title)}</h1><div class="doc-meta"><span>约 {reading} 分钟阅读</span><span>{category_name}</span><a href="{self.source_url(path)}">查看 Markdown 源文件 ↗</a></div>{notice}</div><div class="article-with-toc"><article class="prose">{body}{related}</article><aside class="toc" aria-label="本页目录"><p>本页内容</p>{toc}</aside></div></div></main>'
        self.write(route,self.shell(title,content,route,active,strip_text(text)[:140]))
        self.search.append({'title':title,'text':text,'excerpt':strip_text(text)[:145],
                            'url':self.url(route),'kind':'实验协议' if spec else '知识与说明'})

    def experiment_directory(self):
        options=''.join(f'<option value="{key}">{name}</option>' for key,(name,_) in CATEGORIES.items())
        rows=[]
        for s in self.specs:
            category=s['source']['wiki'].split('/')[1]
            ready=self.ready[s['id']]
            label='配置兼容' if ready else '待专用实现'
            chip=f'<span class="chip {"" if ready else "pending"}">{label}</span>'
            a=E(json.dumps(s['a'],ensure_ascii=False));b=E(json.dumps(s['b'],ensure_ascii=False))
            rows.append(f'<tr data-category="{category}" data-status="{"compatible" if ready else "pending"}"><td><a href="{self.url("experiments/"+s["id"]+"/")}">{E(s["title"])}</a><small>{s["id"]}</small><span class="mobile-only">{chip}</span></td><td><code>{E(s["axis"])}</code><small>A: {a}<br>B: {b}</small></td><td><code>{E(s["metric"])}</code><small>{"越低越好" if s["direction"]=="min" else "越高越好"}</small></td><td>{chip}</td></tr>')
        body=f'''<main id="main" class="container"><div class="page-head"><div class="eyebrow">EXPERIMENT REGISTRY</div><h1>每个技巧，都有具体的对照。</h1><p class="lede">{len(self.specs)} 个细化协议，逐项列明变量、A/B 取值与主指标。按领域和实现状态筛选，进入详情查看来源、运行命令与解释边界。</p><div class="notice">真实训练完成：0。“配置兼容”不代表场景已验证，更不代表 B 组效果更好。</div></div><div class="filter-bar"><label class="grow" for="experiment-query">检索技巧 / 参数 / ID<input class="field" id="experiment-query" type="search" placeholder="例如：LoRA、学习率、loss.label_smoothing"></label><label for="experiment-category">主题领域<select class="field" id="experiment-category"><option value="">全部领域</option>{options}</select></label><label for="experiment-status">代码状态<select class="field" id="experiment-status"><option value="">全部状态</option><option value="compatible">基础驱动配置兼容</option><option value="pending">待专用实现</option></select></label><button id="reset-filters" type="button">重置</button></div><p class="result-count" id="experiment-count" role="status">共 {len(self.specs)} 个实验协议。开启 JavaScript 可筛选与分页。</p><table class="experiment-table"><thead><tr><th scope="col">技巧与协议</th><th scope="col">具体干预 · A / B</th><th scope="col">主指标</th><th scope="col">代码状态</th></tr></thead><tbody id="experiment-rows">{''.join(rows)}</tbody></table><p class="empty" id="empty-experiments" hidden>没有匹配的实验。请清空关键词或重置筛选。</p><div class="pagination" hidden><button id="previous-page" type="button">← 上一页</button><span>每页 30 个实验</span><button id="next-page" type="button">下一页 →</button></div></main>'''
        self.write('experiments/index.html',self.shell('A/B 实验目录',body,'experiments/','experiments'))

    def sources(self):
        ready=sum(self.ready.values())
        cards=''.join(f'<a class="source-card" href="{self.url(route)}"><h3>{title} ↗</h3><p>{description}</p></a>' for route,title,description in [
            ('errata/','原文断言与技术校订','区分作者经验、编辑解释与可检验论断。'),('contradictions/','矛盾观点专题','不同任务、数据规模与实验条件下，经验可能相反。'),
            ('datasets/','Hugging Face 数据登记','只核对元数据，固定版本；不把候选数据误称为已接入。'),('drivers/','真实训练驱动契约','配置、日志、预测、checkpoint 和结果校验的要求。')])
        body=f'''<main id="main" class="container"><div class="page-head"><div class="eyebrow">PROVENANCE & COVERAGE</div><h1>经验有出处，结论有边界。</h1><p class="lede">这份手册整理自仓库保留的知乎原始素材。作者引文保留原意；编辑解释可以校订，技术断言应通过实验和一手资料核实。</p></div><div class="source-grid">{cards}</div><section class="section prose"><h2>当前覆盖账本</h2><table><thead><tr><th>对象</th><th>数量</th><th>真实状态</th></tr></thead><tbody><tr><td>带来源主条目</td><td>412</td><td>保留作者归因、原文、适用场景与要点</td></tr><tr><td>细化 A/B 协议</td><td>517</td><td>已有独立入口；不等于 517 个专用训练实现</td></tr><tr><td>基础驱动配置兼容</td><td>{ready}</td><td>配置离线检查通过，场景与 GPU 训练仍待验证</td></tr><tr><td>专用实现待完成</td><td>{len(self.specs)-ready}</td><td>禁止静默替换成无关分类实验</td></tr><tr><td>知识详情页</td><td>58</td><td>已保留，尚未全部逐操作审计</td></tr><tr><td>原文未直接匹配候选</td><td>{self.coverage['raw_unmatched_candidates']}</td><td>含重复、代码与抓取噪声，不等于遗漏技巧数</td></tr><tr><td>真实训练完成</td><td>0</td><td>没有效果结论，没有补造曲线</td></tr></tbody></table><h2>如何核对一条技巧</h2><ol><li>进入知识主页面，查看作者、原文、适用场景与编辑要点。</li><li>进入对应实验，核对干预是否只改变声明的因素，以及主指标是否符合任务。</li><li>检查代码状态，区分基础驱动兼容、专用场景实现与真实训练完成。</li><li>训练后核对数据/模型 revision、种子、测试 ID、checkpoint 哈希和完整日志。</li></ol><h2>原始资料与版权</h2><p>原始回答版权归各位原作者。本站进行分类、解释和可检验协议整理，供学习与研究使用。尚未直接核对的原文归因会保留“待核对”状态，不把规范化文本匹配当成事实认证。</p><p><a href="{self.source_url(ROOT/'raw/tricks.md')}">查看原始素材</a> · <a href="{self.source_url(ROOT/'experiments/catalog/tricks.json')}">机器可读技巧目录</a> · <a href="{self.source_url(ROOT/'experiments/catalog/coverage.json')}">覆盖审计数据</a></p></section></main>'''
        self.write('sources/index.html',self.shell('来源与覆盖状态',body,'sources/','sources'))

    def build(self):
        self.output.mkdir(parents=True,exist_ok=True)
        self.home();self.experiment_directory();self.sources()
        spec_by_path={str((ROOT/'experiments'/s['source']['wiki'].split('/')[1]/s['id']/'README.md').resolve()):s for s in self.specs}
        for path,route in self.pages.items():
            if '?' not in route and path.is_file():
                self.document(path,route,spec_by_path.get(str(path)))
        body=f'<main id="main" class="container search-page"><div class="page-head"><div class="eyebrow">SEARCH THE HANDBOOK</div><h1>检索知识与实验</h1><p class="lede">搜索完整知识正文、实验 ID、参数名和具体 A/B 操作。</p></div><label for="page-search">搜索关键词</label><input class="field" type="search" id="page-search" placeholder="例如：梯度累积、随机标签、LoRA"><p id="page-search-count" class="search-hint" role="status">输入关键词，或通过 <a href="{self.url("topics/")}">知识目录</a> 浏览。</p><noscript><p>全文检索需要 JavaScript；所有知识和实验页面仍可直接阅读。</p></noscript><ul id="page-results" class="search-results"></ul></main>'
        self.write('search/index.html',self.shell('全文检索',body,'search/'))
        self.write('404.html',self.shell('页面未找到',f'<main id="main" class="container"><div class="page-head"><div class="eyebrow">404 / NOT FOUND</div><h1>这个地址暂时没有内容。</h1><p class="lede">页面可能已调整。可以检索关键词，或返回知识目录继续阅读。</p><a class="button" href="{self.url("topics/")}">返回知识目录 →</a></div></main>','404.html'))
        shutil.copytree(ROOT/'site/assets',self.output/'assets',dirs_exist_ok=True)
        self.write('assets/search-index.json',json.dumps(self.search,ensure_ascii=False,separators=(',',':')))
        self.write('.nojekyll','')
        entries=''.join(f'<url><loc>https://1998x-stack.github.io{E(self.url(route))}</loc></url>' for route in self.written if route.endswith('.html') and route!='404.html')
        self.write('sitemap.xml','<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+entries+'</urlset>')
        self.write('build-manifest.json',json.dumps({'html_pages':sum(p.endswith('.html') for p in self.written),'search_records':len(self.search),'base_url':self.base,'real_training_runs':0},ensure_ascii=False,indent=2))
        print(f'Built {sum(p.endswith(".html") for p in self.written)} HTML pages; {len(self.search)} search records -> {self.output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'_site')
    parser.add_argument('--base-url',default='/ai-tricks')
    args=parser.parse_args()
    Site(args.output,args.base_url).build()
