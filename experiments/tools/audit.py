"""离线审计：代码行数、语法、来源、单因素协议与实际驱动覆盖。"""
import ast
import json
from collections import Counter
from pathlib import Path

from experiments.drivers.sft import SUPPORTED_AXES as SFT_AXES
from experiments.drivers.vision import SUPPORTED_AXES as VISION_AXES, validate_config
from experiments.utils.runtime import ROOT, build_plan, read_json, write_json



def implemented(spec):
    # 这仅检查配置兼容性；题目与真实场景的语义符合性仍需独立复核。
    from experiments.utils.runtime import preflight_native
    profiles=read_json(ROOT/'experiments/configs/remote.example.json')['profiles']
    profile=profiles.get(spec['profile'])
    if profile is None or spec['axis'] not in profile['driver']['supported_axes']:
        return False
    if spec['profile']=='sft' and '全量' in spec['title']:
        return False
    try:
        preflight_native(profile,spec)
    except ValueError:
        return False
    return True


def audit():
    specs = read_json(ROOT/'experiments/catalog/experiments.json')
    items = read_json(ROOT/'experiments/catalog/tricks.json')
    ids = [s['id'] for s in specs]
    assert len(ids) == len(set(ids)), '重复实验 ID'
    counts, statuses = [], Counter()
    table = ['# 驱动覆盖与真实执行状态', '',
             '“配置兼容”只表示基础驱动接受具体 A/B 配置，不证明题目场景覆盖或训练完成。', '',
             '| 实验 | 干预 | 驱动状态 |', '|---|---|---|']
    for spec in specs:
        category = next(x['category'] for x in items if x['id']==spec['parent_id'])
        path = ROOT/'experiments'/category/spec['id']/'experiment.py'
        text = path.read_text()
        lines = len(text.splitlines())
        assert 100 <= lines <= 1000, f'{spec["id"]} 行数越界: {lines}'
        ast.parse(text,filename=str(path))
        adjacent = read_json(path.with_suffix('.json'))
        assert adjacent == spec, '目录与相邻协议不一致'
        assert spec['axis'] not in spec['controls'], '对照因子被固定'
        plan = build_plan(spec,[17,42,2026])
        assert len(plan['runs']) == 6
        counts.append(lines)
        status = '基础驱动配置兼容，场景待复核' if implemented(spec) else '专用训练驱动待实现'
        statuses[status] += 1
        table.append(f"| [{spec['id']}](../{category}/{spec['id']}/README.md) | `{spec['axis']}` | {status} |")
    write_json(ROOT/'experiments/catalog/code_audit.json',{
        'scripts':len(specs),'line_count_min':min(counts),'line_count_max':max(counts),
        'syntax_checked':True,'offline_plans_checked':True,'driver_status':dict(statuses),
        'real_training_runs':0,'scope':'offline_structure_and_configuration_only',
    })
    (ROOT/'experiments/catalog/DRIVER_STATUS.md').write_text('\n'.join(table)+'\n')
    print(json.dumps({'scripts':len(specs),'lines':[min(counts),max(counts)],
                      'driver_status':dict(statuses)},ensure_ascii=False))


if __name__ == '__main__':
    audit()
