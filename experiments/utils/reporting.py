"""只使用驱动真实产生的指标，输出配对统计与独立 PNG/SVG。"""
import csv
from pathlib import Path

from .runtime import check_pairs, write_json


def summarize(results, spec, output):
    import numpy as np

    pairs = check_pairs(results)
    metric = spec['metric']
    deltas = np.array([arms['B']['metrics'][metric] - arms['A']['metrics'][metric]
                       for _, arms in sorted(pairs.items())], dtype=float)
    if len(deltas) < 2:
        raise ValueError('至少两组配对种子')
    # 重采样单位是 seed 配对，不是 epoch；保留负结果，不筛选幸运种子。
    bootstrap = np.random.default_rng(2026).choice(deltas, (10000, len(deltas))).mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    summary = {'id': spec['id'], 'metric': metric, 'direction': spec['direction'],
               'paired_seeds': len(deltas), 'delta_definition': 'B - A',
               'mean_delta': float(deltas.mean()), 'std_delta': float(deltas.std(ddof=1)),
               'bootstrap_95_percentile_interval': [float(low), float(high)],
               'inference': '描述性统计；少量种子及多重比较下不得宣称显著或普适有效',
               'limitations': spec['limitations']}
    write_json(Path(output) / 'summary.json', summary)
    with (Path(output) / 'paired_metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'A', 'B', 'B_minus_A'])
        for seed, arms in sorted(pairs.items()):
            a, b = (arms[k]['metrics'][metric] for k in ('A', 'B'))
            writer.writerow([seed, a, b, b - a])
    return summary


def plot_results(results, spec, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pairs = check_pairs(results)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed, arms in sorted(pairs.items()):
        values = [arms[k]['metrics'][spec['metric']] for k in ('A', 'B')]
        axes[0].plot(['A', 'B'], values, marker='o', alpha=.7, label=f'seed {seed}')
        for arm in ('A', 'B'):
            path = Path(output) / f'seed-{seed}' / arm / 'history.csv'
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if not rows or not {'step', 'train_loss', 'val_loss'} <= set(rows[0]):
                raise ValueError('history.csv 必须含 step/train_loss/val_loss，且至少一行')
            axes[1].plot([int(r['step']) for r in rows], [float(r['val_loss']) for r in rows],
                         label=f'{arm} seed {seed}', alpha=.7)
    axes[0].set_ylabel(spec['metric'])
    axes[0].set_title('Paired final evaluation')
    axes[1].set_xlabel('Optimizer step')
    axes[1].set_ylabel('Validation loss')
    axes[1].set_title('Actual training history')
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    figure.suptitle(spec['id'])
    for suffix in ('png', 'svg'):
        figure.savefig(Path(output) / f'comparison.{suffix}', dpi=160)
    plt.close(figure)


def write_report(spec, summary, output):
    low, high = summary['bootstrap_95_percentile_interval']
    text = f'''# {spec['id']} 实验结果

- 主指标：`{spec['metric']}`，方向：`{spec['direction']}`。
- A：`{spec['axis']} = {spec['a']}`。
- B：`{spec['axis']} = {spec['b']}`。
- 配对种子数：{summary['paired_seeds']}。
- B − A 均值：{summary['mean_delta']:.6g}。
- 配对 bootstrap 95% 描述区间：[{low:.6g}, {high:.6g}]。
- {summary['inference']}。

![真实训练曲线与配对指标](comparison.png)

原始记录见 `paired_metrics.csv` 与各 `seed-*/A|B/` 下的 `request.json`、
`train.log`、`history.csv`、`predictions.jsonl`、`checkpoint_manifest.json`。

## 解释边界

{spec['limitations']}
'''
    (Path(output) / 'REPORT.md').write_text(text, encoding='utf-8')
