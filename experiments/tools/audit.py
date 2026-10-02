"""离线审计：代码行数、语法、来源、单因素协议与实际驱动覆盖。"""
import ast
import json
from collections import Counter
from pathlib import Path

from experiments.drivers.sft import SUPPORTED_AXES as SFT_AXES
from experiments.drivers.vision import SUPPORTED_AXES as VISION_AXES, validate_config
from experiments.utils.runtime import ROOT, build_plan, read_json, write_json


# 不能仅凭字段名宣称已实现：同一字段下有此驱动不支持的枚举值。
MODEL_VALUES = {
    'model.activation': {'sigmoid','tanh','relu','leaky_relu','prelu','mish','identity','relu6'},
    'model.norm': {'none','batch','layer','group','instance','local_response'},
    'model.weight_init': {'default','kaiming_normal','xavier_uniform','xavier_normal','normal_fan_in','zeros','orthogonal'},
    'model.output_activation': {'identity','relu'},
    'optimizer.kind': {'sgd','adam','adamw','adam_then_sgd','adagrad'},
}


def implemented(spec):
    profile,axis = spec['profile'],spec['axis']
    if profile == 'vision' and axis in VISION_AXES and spec['metric'] != 'diagnostic_fault_recall':
        for arm in ('a','b'):
            config = dict(spec['controls'])
            config[axis] = spec[arm]
            try:
                validate_config(config)
            except ValueError:
                return False
            for key,value in config.items():
                if key in MODEL_VALUES and value not in MODEL_VALUES[key]:
                    return False
        return True
    if profile == 'sft' and axis in SFT_AXES:
        # 全量微调和模型比较需要另一 profile；默认 LoRA 不可冒充全量实验。
        if '全量' in spec['title']:
            return False
        return set(spec['controls']) <= SFT_AXES
    return False


def audit():
    specs = read_json(ROOT/'experiments/catalog/experiments.json')
    items = read_json(ROOT/'experiments/catalog/tricks.json')
    ids = [s['id'] for s in specs]
    assert len(ids) == len(set(ids)), '重复实验 ID'
    counts, statuses = [], Counter()
    table = ['# 驱动覆盖与真实执行状态', '',
             '“已实现”仅表示存在对应真实训练代码，尚未经远程 GPU 完整训练验证。', '',
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
        status = '实现待远程验证' if implemented(spec) else '专用训练驱动待实现'
        statuses[status] += 1
        table.append(f"| [{spec['id']}](../{category}/{spec['id']}/README.md) | `{spec['axis']}` | {status} |")
    write_json(ROOT/'experiments/catalog/code_audit.json',{
        'scripts':len(specs),'line_count_min':min(counts),'line_count_max':max(counts),
        'syntax_checked':True,'offline_plans_checked':True,'driver_status':dict(statuses),
        'real_training_runs':0,'no_local_dataset_or_model_downloads':True,
    })
    (ROOT/'experiments/catalog/DRIVER_STATUS.md').write_text('\n'.join(table)+'\n')
    print(json.dumps({'scripts':len(specs),'lines':[min(counts),max(counts)],
                      'driver_status':dict(statuses)},ensure_ascii=False))


if __name__ == '__main__':
    audit()
