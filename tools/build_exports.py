"""Export the complete Chinese handbook to EPUB 3 and native XMind JSON archives."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_site import CATEGORIES, Site, slug


def identity(value):
    return 'n' + hashlib.sha256(value.encode()).hexdigest()[:24]


def archive(path, files):
    """Stable IDs and ZIP timestamps keep generated mind maps reproducible."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            package.writestr(info, data.encode())


def topic(key, title, children=(), note='', href='', folded=False):
    result = {'id': identity(key), 'class': 'topic', 'title': title}
    if children:
        result['children'] = {'attached': list(children)}
    if note:
        result['notes'] = {'plain': {'content': note}}
    if href:
        result['href'] = href
    if folded:
        result['branch'] = 'folded'
    return result


def xmind(site, output):
    sheets = []
    overview = []
    experiment_count = 0
    for number, (category, (name, description)) in enumerate(CATEGORIES.items(), 1):
        tricks = [item for item in site.tricks if item['category'] == category]
        entries = []
        for item in tricks:
            key = item['id']
            details = []
            # Each source field remains a separate expandable node; full original body is in notes.
            for index, match in enumerate(re.finditer(r'^- \*\*([^*]+)\*\*[:：]\s*(.*)$', item['body'], re.M)):
                field, value = match.groups()
                details.append(topic(f'{key}/field/{index}', field+'：'+value, note=value))
            protocols = []
            for spec in (s for s in site.specs if s['parent_id'] == key):
                experiment_count += 1
                expkey = spec['id']
                status = '基础驱动配置兼容，场景待复核' if site.ready[expkey] else '专用训练驱动待实现'
                fields = [
                    ('干预变量', spec['axis']),
                    ('A · 对照组', json.dumps(spec['a'], ensure_ascii=False)),
                    ('B · 实验组', json.dumps(spec['b'], ensure_ascii=False)),
                    ('主指标', spec['metric']+'；'+('越低越好' if spec['direction']=='min' else '越高越好')),
                    ('固定条件', json.dumps(spec['controls'], ensure_ascii=False)),
                    ('配对控制', '、'.join(spec['paired_controls'])),
                    ('预算规则', spec['budget_rule']),
                    ('执行状态', status+'；真实训练未执行'),
                    ('日志与绘图', '、'.join(spec['artifacts'])),
                    ('解释边界', spec['limitations']),
                ]
                path = ROOT/'experiments'/category/expkey/'README.md'
                protocols.append(topic(expkey, expkey+' · '+spec['axis'],
                    [topic(expkey+'/'+label, label+'：'+value) for label, value in fields],
                    note=path.read_text(), href=site.source_url(path), folded=True))
            details.append(topic(key+'/protocols', f'A/B 实验 · {len(protocols)} 项', protocols))
            entries.append(topic(key, item['title'], details, note=item['body'],
                href=site.source_url(ROOT/item['source']['wiki'])+'#L'+str(item['source']['line']), folded=True))
        detail_pages = []
        for path in sorted((ROOT/'wiki'/category).glob('*.md')):
            if path.stem == category:
                continue
            title = re.search(r'^# (.+)', path.read_text(), re.M)[1]
            sections = re.split(r'^## (.+)\n', path.read_text(), flags=re.M)
            branches = [topic(str(path.relative_to(ROOT))+'/'+str(i), sections[i], note=sections[i+1])
                        for i in range(1, len(sections)-1, 2)]
            detail_pages.append(topic(str(path.relative_to(ROOT)), title, branches,
                                     note=path.read_text(), href=site.source_url(path), folded=True))
        branches = [topic(category+'/tricks', f'具体技巧 · {len(entries)} 条', entries),
                    topic(category+'/details', f'知识详情 · {len(detail_pages)} 篇', detail_pages, folded=True)]
        root = topic(category+'/root', name, branches, note=description)
        root['structureClass'] = 'org.xmind.ui.logic.right'
        sheets.append({'id': identity(category+'/sheet'), 'class': 'sheet',
                       'title': f'{number:02d} {name}', 'rootTopic': root})
        overview.append(topic(category+'/overview', f'{number:02d} {name} · {len(entries)} 条',
                              note=description, href='xmind:#'+root['id']))
    overview.append(topic('coverage', '阅读说明与完成状态', [
        topic('coverage/one', '412 条来源技巧 → 517 个 A/B 协议；不是 517 项训练结果'),
        topic('coverage/two', '真实训练完成：0；180 项配置兼容，337 项待专用实现'),
        topic('coverage/three', '展开技巧节点查看具体字段；备注保留完整文本'),
        topic('coverage/four', '14 张分类画布；总览节点链接跳转到对应分类'),
        topic('coverage/five', '来源链接指向 GitHub main，相关提交发布后可在线访问'),
    ]))
    root = topic('overview/root', 'AI TRICKS\n深度学习知识与实验地图', overview)
    root['structureClass'] = 'org.xmind.ui.map.unbalanced'
    sheets.insert(0, {'id': identity('overview/sheet'), 'class': 'sheet', 'title': '00 总览', 'rootTopic': root})
    files = {
        'content.json': json.dumps(sheets, ensure_ascii=False, separators=(',', ':')),
        'metadata.json': json.dumps({'creator': {'name': 'AI Tricks Exporter', 'version': '1.0'}, 'dataStructureVersion': '2'}),
        'manifest.json': json.dumps({'file-entries': {'content.json': {}, 'metadata.json': {}}}),
    }
    archive(output/'AI-Tricks-中文思维导图.xmind', files)
    return {'sheets': len(sheets), 'tricks': len(site.tricks), 'experiments': experiment_count}


def epub(site, output):
    groups = [('阅读指南与实验方法', [ROOT/p for p in (
        'experiments/README.md', 'site/content/methodology.md', 'experiments/DRIVER_CONTRACT.md',
        'experiments/DATASETS.md', 'experiments/ERRATA.md', 'contradictory.md')])]
    for category, (title, _) in CATEGORIES.items():
        paths = sorted((ROOT/'wiki'/category).glob('*.md'))
        main = ROOT/'wiki'/category/(category+'.md')
        groups.append((title, [main]+[p for p in paths if p != main]))
    for category, (title, _) in CATEGORIES.items():
        groups.append(('实验附录 · '+title, [ROOT/'experiments'/category/s['id']/'README.md'
                       for s in site.specs if s['source']['wiki'].split('/')[1] == category]))
    ids = {path.resolve(): identity(str(path.relative_to(ROOT))) for _, paths in groups for path in paths}
    routes = {site.url(site.pages[path]): key for path, key in ids.items()}
    for category in CATEGORIES:
        routes[site.url('experiments/index.html?category='+category)] = 'experiments-'+category
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"></head><body>',
        '<h1 id="reading-notes">阅读说明</h1><p>本书收录 72 个 Wiki 页面及 517 个 A/B 实验协议。'
        '保留来源与校订；协议不是已验证结论。真实训练完成数为 0。</p>'
        '<p>正文和内部跳转可离线阅读；查看 GitHub 原始代码需要联网。'
        '电子书未附模型、数据或训练产物。目录分为阅读方法、14 个知识领域与14 个实验附录。</p>']
    spec_by_path = {str((ROOT/'experiments'/s['source']['wiki'].split('/')[1]/s['id']/'README.md').resolve()): s for s in site.specs}
    for group, paths in groups:
        group_id = 'group-'+identity(group)
        if group.startswith('实验附录'):
            category = next(k for k, (v, _) in CATEGORIES.items() if group == '实验附录 · '+v)
            group_id = 'experiments-'+category
        parts.append(f'<h1 id="{group_id}">{html.escape(group)}</h1>')
        for path in paths:
            docid = ids[path.resolve()]
            title, body, _, _ = site.render(path)
            body = re.sub(r'id="([^"]+)"', lambda m: f'id="{docid}-{m[1]}"', body)
            def link(match):
                href = html.unescape(match[1])
                if href.startswith('#'):
                    target = '#'+docid+'-'+unquote(href[1:])
                else:
                    base, sep, fragment = href.partition('#')
                    if base in routes:
                        target = '#'+routes[base]+('-'+unquote(fragment) if sep else '')
                    elif href.startswith('/ai-tricks/'):
                        target = '#reading-notes'
                    else:
                        target = href
                return 'href="'+html.escape(target, quote=True)+'"'
            body = re.sub(r'href="([^"]+)"', link, body)
            body = re.sub(r'<(/?)h([1-6])(\b[^>]*)>', lambda m: f'<{m[1]}h{min(6,int(m[2])+1)}{m[3]}>', body)
            notice = ''
            if str(path.resolve()) in spec_by_path:
                spec = spec_by_path[str(path.resolve())]
                state = '基础驱动配置兼容，场景待复核' if site.ready[spec['id']] else '专用训练驱动待实现'
                notice = '<p class="notice">'+state+'。真实训练未执行，本页无效果结论。</p>'
            parts.append(f'<h2 id="{docid}">{html.escape(title)}</h2>{notice}{body}')
    parts.append('</body></html>')
    # Preserve original MathML: Pandoc's HTML -> TeX -> MathML roundtrip loses some operators.
    formulas = {}
    def protect_math(match):
        token = f'XMATHMLTOKEN{len(formulas)}END'
        formulas[token] = match[0]
        return token
    assembled = re.sub(r'<math\b[\s\S]*?</math>', protect_math, '\n'.join(parts))
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp)/'book.html'
        source.write_text(assembled)
        subprocess.run(['pandoc', str(source), '-f', 'html', '-t', 'epub3', '--toc', '--toc-depth=2',
            '--split-level=2', '--css='+str(ROOT/'exports/epub.css'),
            '--epub-cover-image='+str(ROOT/'exports/cover.svg'),
            '--metadata=title:AI Tricks｜深度学习知识与实验手册', '--metadata=lang:zh-CN',
            '--metadata=toc-title:目录',
            '--metadata=author:AI Tricks 知识库（原始回答归各位原作者）',
            '--metadata=rights:原始回答版权归各位原作者；编排与实验代码许可参见仓库。',
            '-o', str(output/'AI-Tricks-中文实验手册.epub')], check=True)
    path = output/'AI-Tricks-中文实验手册.epub'
    with zipfile.ZipFile(path) as package:
        entries = {name: package.read(name) for name in package.namelist()}
    math_pages = set()
    for name, payload in list(entries.items()):
        if name.endswith('.xhtml'):
            text = payload.decode()
            for token, formula in formulas.items():
                if token in text:
                    text = text.replace(token, formula)
                    math_pages.add(name)
            entries[name] = text.encode()
    package_name = next(name for name in entries if name.endswith('.opf'))
    package_xml = ET.fromstring(entries[package_name])
    for item in package_xml.findall('.//{http://www.idpf.org/2007/opf}item'):
        target = (Path(package_name).parent/item.attrib['href']).as_posix()
        if target in math_pages:
            properties = item.get('properties', '').split()
            if 'mathml' not in properties:
                item.set('properties', ' '.join(properties+['mathml']))
    entries[package_name] = ET.tostring(package_xml, encoding='utf-8', xml_declaration=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr('mimetype', entries.pop('mimetype'), compress_type=zipfile.ZIP_STORED)
        for name, payload in entries.items():
            package.writestr(name, payload)
    return {'source_documents': len(ids), 'sections': len(groups)+1, 'math_formulas': len(formulas)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['all', 'epub', 'xmind'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT/'exports')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    site = Site(args.output, '/ai-tricks')
    results = {}
    if args.format in ('all', 'xmind'):
        results['xmind'] = xmind(site, args.output)
    if args.format in ('all', 'epub'):
        results['epub'] = epub(site, args.output)
    print(json.dumps(results, ensure_ascii=False, indent=2))
