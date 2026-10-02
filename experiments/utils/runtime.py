"""远程训练执行契约。计划/导入阶段不连接 Hugging Face。"""
from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import platform
import re
import socket
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def validate_spec(spec):
    for field in ('id', 'parent_id', 'title', 'source', 'axis', 'a', 'b', 'metric',
                  'direction', 'profile', 'controls', 'limitations'):
        if field not in spec:
            raise ValueError(f'实验缺少字段: {field}')
    if spec['a'] == spec['b']:
        raise ValueError('A/B 取值必须不同')
    if spec['direction'] not in ('min', 'max'):
        raise ValueError('direction 必须为 min/max')
    source = ROOT / spec['source']['wiki']
    if not source.is_file():
        raise ValueError(f'Wiki 不存在: {source}')
    if file_hash(source) != spec['source']['sha256']:
        raise ValueError('Wiki 已变化；请重新生成目录并审核实验协议')


def build_plan(spec, seeds, environment=None):
    validate_spec(spec)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('至少两个互不重复的配对随机种子；默认三种子')
    plan = {'schema_version': 1, 'spec': spec, 'seeds': seeds,
            'environment': environment, 'runs': []}
    # 交替运行顺序，避免 A 总是在较空闲/较冷的设备上运行。
    for index, seed in enumerate(seeds):
        for arm in (('A', 'B') if index % 2 == 0 else ('B', 'A')):
            plan['runs'].append({'seed': seed, 'arm': arm,
                                 'intervention': {spec['axis']: spec[arm.lower()]}})
    plan['plan_hash'] = digest(plan)
    return plan


def check_remote(environment, spec):
    execution = environment['execution']
    hostname = socket.gethostname()
    if hostname not in execution['allowed_hostnames']:
        raise RuntimeError(f'当前主机 {hostname!r} 不在远程训练白名单；拒绝下载/训练')
    if platform.system() != 'Linux':
        raise RuntimeError('训练入口仅接受已配置的远程 Linux 主机；本机只可 --plan')
    if not execution.get('remote_training_confirmed', False):
        raise ValueError('需在远程资源配置中明确 remote_training_confirmed')
    if execution['max_total_seconds'] <= 0 or execution['max_run_seconds'] <= 0:
        raise ValueError('必须设置明确的正数时间预算')
    profile = environment['profiles'][spec['profile']]
    for artifact in ('dataset', 'model'):
        item = profile[artifact]
        # 禁止把 main/latest 当成可复现实验版本。
        pattern = r'[0-9a-f]{64}' if item.get('source') == 'code' else r'[0-9a-f]{40}'
        if not re.fullmatch(pattern, item['revision']):
            raise ValueError(f'{artifact}.revision 必须为完整版本哈希')
        if item.get('source') == 'code' and file_hash(ROOT / item['path']) != item['revision']:
            raise ValueError('本地模型代码版本哈希不匹配')
    driver = profile['driver']
    if spec['axis'] not in driver['supported_axes']:
        raise ValueError(f'驱动未实现干预 {spec["axis"]}；不得忽略配置或替换为玩具实验')
    if driver.get('kind') != 'real_training':
        raise ValueError('此任务只接受 real_training 驱动')
    if not isinstance(driver['argv'], list) or not driver['argv']:
        raise ValueError('driver.argv 必须为非空参数数组，不接受 shell 字符串')
    validate_splits(profile['splits'])
    if profile['budget']['max_steps'] < 1:
        raise ValueError('max_steps 必须预先指定')
    if profile.get('streaming') is not True:
        raise ValueError('必须显式使用远程 Hugging Face streaming')
    if len({digest(s) for s in profile['splits'].values()}) != 3:
        raise ValueError('train/validation/test 必须明确隔离')
    profile = dict(profile)
    profile['allowed_hostnames'] = execution['allowed_hostnames']
    return profile


def make_request(plan, run, profile):
    spec = plan['spec']
    controls = dict(profile.get('baseline', {}))
    controls.update(spec['controls'])
    controls.update(run['intervention'])
    return {'schema_version': 1, 'experiment_id': spec['id'],
            'plan_hash': plan['plan_hash'], 'seed': run['seed'], 'arm': run['arm'],
            'axis': spec['axis'], 'config': controls, 'profile': profile,
            'metric': spec['metric'], 'direction': spec['direction'],
            'dataset_revision': profile['dataset']['revision'],
            'model_revision': profile['model']['revision']}


def validate_result(result, request, directory):
    for name in ('experiment_id', 'plan_hash', 'seed', 'arm', 'dataset_revision', 'model_revision'):
        if result.get(name) != request[name]:
            raise ValueError(f'结果来源字段不匹配: {name}')
    if result.get('applied_config') != request['config']:
        raise ValueError('驱动没有确认完整应用 A/B 配置；拒绝统计')
    if result.get('data_kind') != 'huggingface_real':
        raise ValueError('结果必须来自真实 Hugging Face 数据')
    if not result.get('train_examples_seen', 0) > 0 or not result.get('optimizer_steps', 0) > 0:
        raise ValueError('没有训练样本或参数更新，不能视为完整训练结果')
    value = result['metrics'][request['metric']]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('主指标缺失或不是有限数值')
    # 每个驱动必须提供共同测试样本 ID 的摘要，不接受两组测试集不一致。
    if not re.fullmatch(r'[0-9a-f]{64}', result.get('test_ids_sha256', '')):
        raise ValueError('必须记录 test_ids_sha256')
    for name in ('history.csv', 'predictions.jsonl', 'checkpoint_manifest.json'):
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'真实训练产物缺失: {name}')
    checkpoints = read_json(Path(directory) / 'checkpoint_manifest.json')
    if not checkpoints.get('selected_by_validation'):
        raise ValueError('checkpoint 必须仅按预注册验证指标选择')
    checkpoint_path = (Path(directory) / checkpoints['path']).resolve()
    if not checkpoint_path.is_relative_to(Path(directory).resolve()):
        raise ValueError('checkpoint 路径越出本次实验目录')
    if file_hash(checkpoint_path) != checkpoints['sha256']:
        raise ValueError('checkpoint 内容哈希不匹配')
    with (Path(directory) / 'history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    if not history or not {'step','train_loss','val_loss'} <= set(history[0]):
        raise ValueError('训练历史字段不完整')
    for row in history:
        if any(not math.isfinite(float(row[k])) for k in ('step','train_loss','val_loss')):
            raise ValueError('训练历史中出现非有限数值')
    identities = []
    with (Path(directory) / 'predictions.jsonl').open() as stream:
        for line in stream:
            identities.append(json.loads(line)['id'])
    if not identities or len(identities) != len(set(identities)):
        raise ValueError('测试预测为空或存在重复样本 ID')
    actual_digest = hashlib.sha256('\n'.join(sorted(identities)).encode()).hexdigest()
    if actual_digest != result['test_ids_sha256']:
        raise ValueError('测试集 ID 摘要与预测文件不一致')
    return result


def execute_run(plan, run, profile, output, timeout):
    directory = Path(output) / f'seed-{run["seed"]}' / run['arm']
    directory.mkdir(parents=True, exist_ok=False)
    request = make_request(plan, run, profile)
    write_json(directory / 'request.json', request)
    start = time.monotonic()
    command = [sys.executable if x == '{python}' else x for x in profile['driver']['argv']]
    command += ['--request', str(directory / 'request.json'), '--output', str(directory)]
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = str(run['seed'])
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['HF_HOME'] = str(Path(plan['environment']['execution']['hf_cache']).expanduser())
    try:
        with (directory / 'train.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        result = validate_result(read_json(directory / 'result.json'), request, directory)
        result['wall_seconds'] = time.monotonic() - start
        write_json(directory / 'verified_result.json', result)
        return result
    except Exception as error:
        write_json(directory / 'failure.json', {'type': type(error).__name__, 'error': str(error),
                                               'wall_seconds': time.monotonic() - start})
        raise


def check_pairs(results):
    by_seed = {}
    for row in results:
        arms = by_seed.setdefault(row['seed'], {})
        if row['arm'] in arms:
            raise ValueError('重复 seed/arm')
        arms[row['arm']] = row
    for seed, arms in by_seed.items():
        if set(arms) != {'A', 'B'}:
            raise ValueError(f'{seed} 缺少完整 A/B 配对')
        if arms['A']['test_ids_sha256'] != arms['B']['test_ids_sha256']:
            raise ValueError('A/B 最终测试样本不同，禁止配对比较')
    return by_seed


def validate_splits(splits):
    """对支持的稳定 modulo 切分做真实的互斥检查，而非仅比较名称。"""
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('需要 train/validation/test 三个切分')
    names = list(splits)
    for i, name in enumerate(names):
        left = splits[name]
        if not isinstance(left, dict) or not left.get('source'):
            raise ValueError('切分必须指定 Hugging Face source')
        for other in names[i+1:]:
            right = splits[other]
            if left['source'] != right['source']:
                continue
            if 'modulo' not in left or 'modulo' not in right:
                raise ValueError('同一源 split 必须显式划分不重叠的 modulo 子集')
            period = math.lcm(left['modulo'], right['modulo'])
            if period <= 0 or period > 100000:
                raise ValueError('非法 modulo 周期')
            def kept(selector, index):
                return ((index % selector['modulo'] == selector['remainder'])
                        == selector.get('keep_equal', True))
            if any(kept(left, j) and kept(right, j) for j in range(period)):
                raise ValueError(f'{name} 与 {other} 的样本 ID 相交')
