"""Per-run metrics of task-stream experiments, as one table.

    python -m domainbed.scripts.stream_analysis outputs/e1 results/e1.csv

Each row is one run under <root>/<stream dir>/<algorithm[__tag]>/env<e>_seed<s>.
For runs on the whole stream (env -1):

    final_acc    validation accuracy at the last step, averaged over the tasks
    forgetting   for each task but the last, its best validation accuracy at the
                 end of a phase (its own or a later one) minus its final one,
                 averaged (Elenter et al., 2023); same definition as the paper
    learned_acc  validation accuracy of each task at the end of its own phase,
                 averaged (how well the stream was learned before forgetting)
    acc_k, forget_k, dual_k, eps_k, buf_k, replay_loss_k   per task k, last step
    dual_mean_k  dual variable of task k averaged over the checkpoints after its phase

For leave-one-task-out runs (env >= 0): test_acc of the checkpoint with the
lowest validation loss inside the final phase (the paper's rule).
All accuracies are in percent.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEQUENTIAL = ('Finetune', 'ER', 'ER_FF', 'PDCL_RS', 'PDCL')


def run_row(recs):
    args, last = recs[-1]['args'], recs[-1]
    T = sum(1 for k in last if k.endswith('_out_acc') and k.startswith('env'))
    test = args['test_envs']
    train = [e for e in range(T) if e not in test]
    n = len(train)
    steps = np.array([r['step'] for r in recs])
    phase = (steps[-1] + 1) // n
    row = {'stream': args.get('stream', 'rounds'), 'seed': args['seed'],
           'env': test[0] if test else -1, 'steps': int(steps[-1] + 1), 'n_tasks': n}
    acc = [100 * np.array([r[f'env{e}_out_acc'] for r in recs]) for e in train]
    at = lambda k, step: acc[k][np.argmin(np.abs(steps - step))]
    if test:
        seq = args['algorithm'] in SEQUENTIAL
        ok = [r for r in recs if not seq or r['step'] > (n - 1) * phase]
        row['test_acc'] = 100 * min(ok, key=lambda r: r['agg_val_loss'])['agg_test_acc']
        row['test_acc_last'] = 100 * last['agg_test_acc']
    forget = [max(at(k, l * phase) for l in range(k + 1, n)) - acc[k][-1] for k in range(n - 1)]
    row['final_acc'] = float(np.mean([a[-1] for a in acc]))
    row['forgetting'] = float(np.mean(forget))
    row['learned_acc'] = float(np.mean([at(k, (k + 1) * phase) for k in range(n - 1)] + [acc[-1][-1]]))
    for k in range(n):
        row[f'acc_{k}'] = float(acc[k][-1])
    for k in range(n - 1):
        row[f'forget_{k}'] = float(forget[k])
        for key in ('dual', 'eps', 'buf', 'replay_loss'):
            if f'{key}_{k}' in last:
                row[f'{key}_{k}'] = last[f'{key}_{k}']
        if f'dual_{k}' in last:
            after = [r[f'dual_{k}'] for r in recs if r['step'] > (k + 1) * phase]
            row[f'dual_mean_{k}'] = float(np.mean(after)) if after else 0.0
    return row


def table(root):
    rows = []
    for done in sorted(Path(root).rglob('done')):
        d = done.parent
        recs = [json.loads(l) for l in open(d / 'results.jsonl')]
        rows.append({'group': d.parent.parent.name, 'method': d.parent.name} | run_row(recs))
    return pd.DataFrame(rows)


def paired(df, a, b, value='final_acc', by=('group',)):
    """Mean and standard error over seeds of method a minus method b."""
    wide = df[df['method'].isin([a, b])].pivot_table(
        index=list(by) + ['env', 'seed'], columns='method', values=value)
    d = (wide[a] - wide[b]).dropna().groupby(level=list(by) + ['seed']).mean()
    g = d.groupby(level=list(by))
    return pd.DataFrame({'diff': g.mean(), 'se': g.std(ddof=1) / np.sqrt(g.count()), 'n': g.count()})


if __name__ == '__main__':
    out = table(sys.argv[1])
    out.to_csv(sys.argv[2], index=False)
    summary = out.groupby(['group', 'method'])[['final_acc', 'forgetting', 'learned_acc']].agg(['mean', 'sem'])
    print(summary.round(2).to_string())
