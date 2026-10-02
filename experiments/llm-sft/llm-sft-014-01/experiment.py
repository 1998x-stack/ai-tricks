#!/usr/bin/env python3
"""SFT 数据混合比例。

Wiki: wiki/llm-sft/llm-sft.md:111
此文件由 tools/build_catalog.py 生成；修改逐项协议后重新生成。
--plan 可在本机运行，--run 仅允许已配置的远程真实训练。
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.utils.runtime import (
    build_plan, check_remote, digest, execute_run, file_hash, read_json, write_json,
)
from experiments.utils.reporting import plot_results, summarize, write_report

EXPERIMENT_ID = 'llm-sft-014-01'
SPEC_PATH = Path(__file__).with_suffix('.json')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--plan', action='store_true', help='仅输出计划，不联网')
    action.add_argument('--run', action='store_true', help='在已配置远程机器完整训练')
    parser.add_argument('--environment', type=Path, help='远程资源 JSON 配置')
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 42, 2026])
    parser.add_argument('--output', type=Path, help='全新的结果目录；禁止覆盖旧实验')
    return parser.parse_args()


def load_protocol():
    spec = read_json(SPEC_PATH)
    if spec['id'] != EXPERIMENT_ID:
        raise ValueError('脚本与相邻协议 ID 不一致')
    for arm in ('a', 'b'):
        if spec[arm] is None:
            raise ValueError('A/B 必须具体定义，不能用未填写的占位参数')
    if spec['axis'] in spec['controls']:
        raise ValueError('干预变量不能同时被列为控制常量')
    return spec


def provenance():
    # 代码版本和 dirty diff 摘要都保存；不读取环境中的令牌或 SSH 凭据。
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
    )
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'git_commit': commit.stdout.strip() if commit.returncode == 0 else None,
        'entry_sha256': file_hash(__file__),
        'code_tree_sha256': digest({str(p.relative_to(ROOT)): file_hash(p)
                                   for p in sorted((ROOT / 'experiments').rglob('*.py'))}),
        'protocol_sha256': file_hash(SPEC_PATH),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }


def prepare(args, spec):
    environment = read_json(args.environment) if args.environment else None
    plan = build_plan(spec, args.seeds, environment)
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan, None
    if environment is None:
        raise ValueError('--run 需要 --environment；没有远程资源时请使用 --plan')
    # 必须先校验主机和驱动能力，之后才允许任何数据访问与训练。
    profile = check_remote(environment, spec)
    return plan, profile


def run_all(args, plan, profile):
    output = (args.output or ROOT / 'experiments' / 'runs' / EXPERIMENT_ID).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    write_json(output / 'provenance.json', provenance())
    execution = plan['environment']['execution']
    started = time.monotonic()
    results = []
    for run in plan['runs']:
        remaining = execution['max_total_seconds'] - (time.monotonic() - started)
        if remaining <= 0:
            write_json(output / 'status.json', {'status': 'budget_exhausted',
                                               'completed_runs': len(results)})
            raise TimeoutError('达到总时间预算；不将不完整配对当成结果')
        timeout = min(remaining, execution['max_run_seconds'])
        print(f"{EXPERIMENT_ID} seed={run['seed']} arm={run['arm']}", flush=True)
        result = execute_run(plan, run, profile, output, timeout)
        results.append(result)
        write_json(output / 'progress.json', {'completed_runs': len(results),
                                             'total_runs': len(plan['runs'])})
    # 所有种子与两组完成后才能汇总，避免挑选成功/有利的结果。
    summary = summarize(results, plan['spec'], output)
    plot_results(results, plan['spec'], output)
    write_report(plan['spec'], summary, output)
    write_json(output / 'status.json', {'status': 'complete',
                                       'completed_runs': len(results)})
    return output


def main():
    args = arguments()
    spec = load_protocol()
    plan, profile = prepare(args, spec)
    if args.plan:
        return 0
    output = run_all(args, plan, profile)
    print(f'日志、训练曲线与配对比较已写入 {output}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, TimeoutError, FileExistsError) as error:
        print(f'实验未完成: {error}', file=sys.stderr)
        raise SystemExit(2)
