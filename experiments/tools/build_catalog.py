"""从当前 Wiki 可追溯地构建技巧目录与逐项 A/B 入口，不下载任何数据。"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from experiments.tools.protocols import CAVEATS, EXTRA, definitions
from experiments.utils.runtime import ROOT, write_json


def extract():
    items = []
    source_ids = json.loads((ROOT/'experiments/tools/source_ids.json').read_text())
    seen_ids = set()
    for path in sorted((ROOT / 'wiki').glob('*/*.md')):
        if path.stem != path.parent.name:
            continue
        text = path.read_text()
        matches = list(re.finditer(r'^### (.+)$', text, re.M))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[match.end():end].split('\n## ')[0].strip()
            if '**来源**' not in body:
                continue
            author = re.search(r'\*\*来源\*\*[:：]\s*(.*)', body)
            identity = source_ids.get(str(path.relative_to(ROOT)),{}).get(match[1])
            if identity is None or identity in seen_ids:
                raise ValueError(f'新增/重名标题需显式审核稳定 ID: {path}: {match[1]}')
            seen_ids.add(identity)
            items.append({'id': identity, 'category': path.stem,
                          'title': match[1], 'body': body,
                          'source': {'wiki': str(path.relative_to(ROOT)),
                                     'line': text[:match.start()].count('\n') + 1,
                                     'heading': match[1],
                                     'author': author[1] if author else '原 Wiki 未明确',
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}})
    return items



def render_wikilinks(body, source, output_directory):
    """从原 Wiki 相对目录解析引用；不存在的原红链接保留为文字，不制造新断链。"""
    def replace(match):
        target, _, alias = match[1].partition('|')
        filename, separator, anchor = target.partition('#')
        path = (source.parent / (filename + '.md')).resolve() if filename else source
        label = alias or target
        if not path.is_file():
            return f'`[[{match[1]}]]`（原 Wiki 未解析链接）'
        relative = os.path.relpath(path,output_directory)
        return f'[{label}]({relative}{separator}{anchor})'
    return re.sub(r'\[\[([^\]]+)\]\]',replace,body)

def raw_references(body, raw_lines):
    # 只认可原文连续片段匹配；不把语义相似当成准确引文。
    quote = re.search(r'\*\*原文\*\*[:：](.*)', body)
    if not quote:
        return []
    normalize = lambda s: re.sub(r'[\W_]+', '', s).lower()
    fragments = [normalize(s) for s in re.split(r'["“”/]', quote[1])]
    fragments = [s for s in fragments if len(s) >= 16]
    found = []
    for i, line in enumerate(raw_lines, 1):
        normalized = normalize(line)
        if any(f in normalized or (len(normalized) >= 24 and normalized in f) for f in fragments):
            found.append(i)
    return found


def choose_profile(item, axis):
    category, title = item['category'], item['title'].lower()
    if category == 'llm-rl' or axis.startswith(('rl.', 'reward.')) or 'ppo' in title or 'grpo' in title:
        return 'preference' if ('dpo' in title or 'chosen' in title or 'beta' in title) else 'reasoning_rl'
    if category == 'llm-sft' or axis.startswith('adapter.') or 'sft' in title or 'lora' in title:
        return 'sft'
    if category == 'recsys' or '推荐' in title:
        return 'recommendation'
    if '对比学习' in title or '召回' in title:
        return 'retrieval'
    if category == 'nlp' or 'rnn' in title or axis.startswith('model.recurrent'):
        return 'nlp'
    if category == 'cnn-cv' and any(t in title for t in ('检测','anchor','yolo','fp多','两阶段')):
        return 'detection'
    if category == 'cnn-cv' and any(t in title for t in ('分割','稠密','low-level','upsample','边界')):
        return 'segmentation'
    return 'vision'


def measurement(profile, axis):
    metrics = {'vision': 'test_accuracy', 'nlp': 'test_accuracy', 'sft': 'test_rougeL',
               'reasoning_rl': 'test_exact_match', 'preference': 'test_preference_accuracy',
               'recommendation': 'test_ndcg_at_10', 'retrieval': 'test_recall_at_10',
               'detection': 'test_map', 'segmentation': 'test_miou'}
    if axis.startswith(('validation.', 'tracking.', 'engineering.')):
        return 'diagnostic_fault_recall', 'max'
    if axis in ('training.gradient_checkpointing', 'training.sharding'):
        return 'peak_cuda_memory_bytes', 'min'
    if axis in ('data.batch_order', 'data.packing', 'data.num_workers', 'data.pin_memory',
                'data.tensor_conversion', 'model.pool_relu_order', 'training.world_size'):
        return 'examples_per_second', 'max'
    return metrics[profile], 'max'


def controls_for(item, axis):
    controls = {}
    if axis in ('optimizer.beta1','optimizer.beta2'):
        controls['optimizer.kind'] = 'adam'
    if axis == 'optimizer.gamma':
        controls['optimizer.schedule'] = 'exponential'
    if axis in ('optimizer.warmup_ratio','optimizer.min_lr'):
        controls['optimizer.schedule'] = 'warmup_cosine'
    if item['id'] in ('batch-size-006', 'optimizer-lr-027'):
        controls['training.batch_size'] = 256
    if item['id'] in ('normalization-003', 'regularization-013'):
        controls['model.norm'] = 'batch'
    if item['id'] in ('initialization-002', 'activation-013'):
        controls['model.activation'] = 'relu'
    if item['id'] in ('optimizer-lr-029', 'optimizer-lr-030', 'optimizer-lr-044'):
        controls['optimizer.kind'] = 'sgd'
    controls.pop(axis, None)
    return controls


def make_spec(item, axis, a, b, suffix):
    profile = choose_profile(item, axis)
    metric, direction = measurement(profile, axis)
    is_policy = any(key in axis for key in ('policy', 'recipe', 'framework', 'shape_policy'))
    limitation = ('真实数据完整训练的待执行协议，未产生训练结论。'
                  '所列数值是预注册对照，不代表作者经验在所有模型上成立。'
                  '最终数据/模型适配与驱动必须经核对，禁止以通用分类任务代替专门任务。')
    if is_policy:
        limitation += '本实验比较整体策略，策略内可能改变多项配置；不作单参数因果解释。'
    limitation += CAVEATS.get(item['id'], '')
    return {'id': item['id'] + suffix, 'parent_id': item['id'],
            'title': item['title'], 'source': item['source'],
            'axis': axis, 'a': a, 'b': b, 'profile': profile,
            'metric': metric, 'direction': direction, 'controls': controls_for(item, axis),
            'contrast_type': 'policy' if is_policy else 'single_factor',
            'status': 'protocol_ready_training_pending',
            'limitations': limitation,
            'secondary_metrics': ['train_loss','val_loss','wall_seconds','peak_cuda_memory_bytes'],
            'paired_controls': ['dataset_revision','model_revision','train_validation_test_ids',
                                'initialization_seed','data_order_seed','hardware','evaluation_code'],
            'budget_rule': '相同预注册样本/token 预算；早停/搜索/集成/压缩等策略另报告实际预算与总成本',
            'artifacts': ['request.json','train.log','history.csv','predictions.jsonl',
                          'checkpoint_manifest.json','result.json','comparison.png','comparison.svg']}


def generate():
    items = extract()
    mapping = definitions()
    if set(mapping) != {x['id'] for x in items}:
        raise ValueError('Wiki 主条目和人工协议不再一一对应；请先人工审核 protocols.py')
    raw_path = ROOT / 'raw/tricks.md'
    raw_lines = raw_path.read_text().splitlines()
    template = (ROOT / 'experiments/tools/entry.py.template').read_text()
    all_specs = []
    for item in items:
        item['source']['raw_exact_lines'] = raw_references(item['body'], raw_lines)
        item['source']['raw_attribution_status'] = (
            'normalized_fragment_matched' if item['source']['raw_exact_lines'] else 'wiki_attributed_raw_unverified')
        base = mapping[item['id']]
        variants = [base] + EXTRA.get(item['id'], [])
        item['experiments'] = []
        for i, (axis, a, b) in enumerate(variants):
            spec = make_spec(item, axis, a, b, f'-{i+1:02}')
            directory = ROOT / 'experiments' / item['category'] / spec['id']
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / 'experiment.json', spec)
            code = (template.replace('__TITLE__', item['title'].replace('"', '\\"'))
                    .replace('__WIKI__', item['source']['wiki'] + ':' + str(item['source']['line']))
                    .replace('__ID__', spec['id']))
            (directory / 'experiment.py').write_text(code)
            item['experiments'].append(spec['id'])
            all_specs.append(spec)
            wiki = '../../../' + item['source']['wiki']
            readme = f'''# {spec['id']} · {item['title']}

[Wiki 原位置：第 {item['source']['line']} 行]({wiki}) · 作者：{item['source']['author']}

## 具体对照

| 内容 | 预注册值 |
|---|---|
| 干预字段 | `{axis}` |
| A | `{json.dumps(a,ensure_ascii=False)}` |
| B | `{json.dumps(b,ensure_ascii=False)}` |
| 任务配置 | `{profile_name(spec)}` |
| 主指标 | `{spec['metric']}`（{spec['direction']}） |
| 固定条件 | `{json.dumps(spec['controls'],ensure_ascii=False)}`，另固定数据/模型版本、切分、种子、硬件 |
| 对照类型 | `{spec['contrast_type']}` |
| 状态 | 协议和编排入口已生成；完整训练尚未执行，专用驱动支持情况见总表 |

本条只检验上述明确的干预；父条目其他操作见同类别索引。

## 操作与产物

```bash
python experiments/{item['category']}/{spec['id']}/experiment.py --plan
# 下行仅可在已配置的远程机器执行
python experiments/{item['category']}/{spec['id']}/experiment.py --run --environment remote.json
```

默认 17、42、2026 三个配对种子，A/B 交替顺序。每组独立重建模型与优化器，
只按验证集选择 checkpoint，最后用共同且锁定的测试集评价。
训练产生每步 loss、验证指标、耗时、显存、预测结果、checkpoint 清单及原始日志；
完整六次运行后生成配对差值表、bootstrap 描述区间、PNG/SVG 曲线。
不要用测试集选参数、丢弃负结果、补造日志或把未实现的干预静默忽略。

## 原始细节（保留 Wiki 归因）

{render_wikilinks(item['body'], ROOT/item['source']['wiki'], directory)}

## 解释边界

{spec['limitations']}
'''
            (directory / 'README.md').write_text('\n'.join(line.rstrip() for line in readme.splitlines()) + '\n')
    write_json(ROOT / 'experiments/catalog/tricks.json', items)
    write_json(ROOT / 'experiments/catalog/experiments.json', all_specs)
    matched = {n for item in items for n in item['source']['raw_exact_lines']}
    # 这是候选待审计账本，不把启发式候选误称为已提取的全部技巧。
    unresolved = [{'line': n, 'text': line} for n, line in enumerate(raw_lines, 1)
                  if len(line.strip()) >= 55 and n not in matched]
    write_json(ROOT / 'experiments/catalog/raw_unmatched_candidates.json', unresolved)
    leaf_pages = []
    for path in sorted((ROOT / 'wiki').glob('*/*.md')):
        if path.stem != path.parent.name:
            leaf_pages.append({'path': str(path.relative_to(ROOT)),
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'text': path.read_text(), 'status': 'retained_not_atomically_audited'})
    write_json(ROOT / 'experiments/catalog/detail_pages.json', leaf_pages)
    coverage = {'main_tips': len(items), 'atomic_protocols': len(all_specs),
                'categories': dict(Counter(x['category'] for x in items)),
                'detail_pages_retained': len(leaf_pages),
                'raw_normalized_matched_lines': len(matched), 'raw_unmatched_candidates': len(unresolved),
                'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                'full_training_runs': 0, 'all_source_content_atomically_audited': False}
    write_json(ROOT / 'experiments/catalog/coverage.json', coverage)
    for category in coverage['categories']:
        rows = [s for s in all_specs if s['parent_id'].startswith(category + '-')]
        lines = [f'# {category} 实验索引', '', '| 实验 | 具体干预 | A → B |', '|---|---|---|']
        for spec in rows:
            lines.append(f"| [{spec['id']}]({spec['id']}/README.md) {spec['title']} | `{spec['axis']}` | `{spec['a']}` → `{spec['b']}` |")
        (ROOT / 'experiments' / category / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


def profile_name(spec):
    return spec['profile']


if __name__ == '__main__':
    generate()
