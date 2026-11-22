"""R1, "Why does replay hurt on the new-antigen stream?": tables and figure from the raw runs.

    python -m domainbed.scripts.r1_analysis

Reads outputs/r1/<stream>/<method>/env-1_seed<s>/results.jsonl; writes
results/r1_runs.csv (one row per run), results/r1_tables.md and
paper/figures/fig_r1_memory.{pdf,png}.

Methods: fine-tuning; ER (replay weight 1); PDCL (PDCL_RS as published). Memory:
200, 1,000, 3,000 stored examples in total, or "full" (hparam full_memory: replay
batches drawn from the whole training split of each earlier task).
All differences are paired by seed; "a ± b" is mean ± standard error over seeds;
bold = larger than two standard errors.

Per-run columns beyond those of stream_analysis.run_row (k = earlier task):
    last_acc     validation accuracy on the last task at the last step
    earlier_acc  the same, averaged over the earlier tasks
    val_loss, mem_loss   loss on the validation set and on the stored examples of
                 the earlier tasks at the last step, averaged over k
    dual_mean    dual variable averaged over the checkpoints after the phase and over k
    stored, stored_unique   stored examples per earlier task at the last step, and
                 how many of them are distinct, averaged over k
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from domainbed.scripts import paper_assets as pa
from domainbed.scripts.paper_assets import CYAN, NIGHT, RED, plt
from domainbed.scripts.stream_analysis import run_row

ROOT = Path('outputs/r1')
STREAMS = ['aglen', 'target', 'rounds']
MEMORY = ['200', '1000', '3000', 'full']
LABEL = {'200': '200', '1000': '1,000', '3000': '3,000', 'full': 'full'}
FT, ER, PDCL = 'Fine-tuning', 'ER', 'PDCL'
METHOD = {'Finetune': FT, 'ER': ER, 'PDCL_RS': PDCL}
VALUES = [('final_acc', 'Final accuracy (%)'), ('forgetting', 'Forgetting (points)'),
          ('last_acc', 'Accuracy on the last task (%)'),
          ('earlier_acc', 'Mean accuracy on the earlier tasks (%)')]


def load():
    rows = []
    for done in sorted(ROOT.rglob('done')):
        d = done.parent
        recs = [json.loads(l) for l in open(d / 'results.jsonl')]
        hp, last = recs[-1]['hparams'], recs[-1]
        method = METHOD[last['args']['algorithm']]
        memory = '' if method == FT else 'full' if hp.get('full_memory') else str(hp['buffer_size'])
        row = {'group': d.parent.parent.name, 'method': method, 'memory': memory} | run_row(recs)
        n = row['n_tasks']
        row['last_acc'] = row[f'acc_{n - 1}']
        row['earlier_acc'] = float(np.mean([row[f'acc_{k}'] for k in range(n - 1)]))
        for key, src in (('val_loss', 'val_loss'), ('mem_loss', 'mem_loss'), ('stored', 'buf'),
                         ('stored_unique', 'buf_unique'), ('dual_mean', None)):
            v = [row[f'dual_mean_{k}'] if src is None else last[f'{src}_{k}'] for k in range(n - 1)]
            for k in range(n - 1):
                row[f'{key}_{k}'] = v[k]
            row[key] = float(np.mean(v))
        rows.append(row)
    return pd.DataFrame(rows)


def seeds(df, g, m, mem, value):
    d = df[(df['group'] == g) & (df['method'] == m)]
    if m != FT:
        d = d[d['memory'] == mem]
    return d.set_index('seed')[value].sort_index()


def ms(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def stat(df, g, m, mem, value='final_acc'):
    return ms(seeds(df, g, m, mem, value))


def diff(df, g, a, b, mem, value='final_acc'):
    """Paired difference a - b over the seeds, same stream and memory."""
    return ms((seeds(df, g, a, mem, value) - seeds(df, g, b, mem, value)).dropna())


def pm(m, s, nd=2):
    return f'{m:.{nd}f} ± {s:.{nd}f}'


def spm(m, s, nd=2):
    t = f'{m:+.{nd}f} ± {s:.{nd}f}'
    return f'**{t}**' if abs(m) > 2 * s else t


def md(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]) + '\n'


def tables(df):
    T = []
    add = lambda title, header, rows: T.append(f'### T{len(T)}. {title}\n\n' + md(header, rows))
    n = df.groupby(['group', 'method', 'memory']).size()
    T.append(f'{len(df)} runs; seeds per stream, method and memory: {n.min()} to {n.max()}. '
             'Mean ± standard error over seeds; differences paired by seed; '
             '**bold** = larger than two standard errors.\n')

    for value, title in VALUES:
        rows = []
        for g in STREAMS:
            ft = pm(*stat(df, g, FT, '', value))
            for mem in MEMORY:
                rows.append([g, LABEL[mem], ft, pm(*stat(df, g, ER, mem, value)), pm(*stat(df, g, PDCL, mem, value)),
                             spm(*diff(df, g, ER, FT, mem, value)), spm(*diff(df, g, PDCL, FT, mem, value)),
                             spm(*diff(df, g, PDCL, ER, mem, value))])
        add(title, ['Stream', 'Memory', 'Fine-tuning', 'ER', 'PDCL', 'ER − FT', 'PDCL − FT', 'PDCL − ER'], rows)

    rows = []
    for g in STREAMS:
        ft = pm(*stat(df, g, FT, '', 'val_loss'), 3)
        for mem in MEMORY:
            rows.append([g, LABEL[mem], ft] +
                        [pm(*stat(df, g, m, mem, v), 3) for m in (ER, PDCL) for v in ('val_loss', 'mem_loss')] +
                        [spm(*diff(df, g, ER, FT, mem, 'val_loss'), 3), spm(*diff(df, g, PDCL, FT, mem, 'val_loss'), 3)])
    add('Loss on the earlier tasks at the last step, mean over the three earlier tasks: validation set '
        'and stored examples (at full memory the stored examples are the training split)',
        ['Stream', 'Memory', 'FT val.', 'ER val.', 'ER stored', 'PDCL val.', 'PDCL stored',
         'ER − FT val.', 'PDCL − FT val.'], rows)

    rows = [[g, LABEL[mem], pm(*stat(df, g, PDCL, mem, 'dual_mean'), 3)] +
            [pm(*stat(df, g, PDCL, mem, f'dual_mean_{k}'), 3) for k in range(3)]
            for g in STREAMS for mem in MEMORY]
    add('PDCL: dual variable, averaged over the checkpoints after the task\'s phase (ER uses weight 1)',
        ['Stream', 'Memory', 'Mean over tasks', 'Task 1', 'Task 2', 'Task 3'], rows)

    rows = []
    for g in STREAMS:
        for mem in MEMORY:
            for m in (ER, PDCL):
                d = df[(df['group'] == g) & (df['method'] == m) & (df['memory'] == mem)]
                cell = lambda key: ' / '.join(
                    f"{d[f'{key}_{k}'].mean():.0f} ({d[f'{key}_{k}'].min():.0f}–{d[f'{key}_{k}'].max():.0f})"
                    for k in range(3))
                rows.append([g, LABEL[mem], m, cell('stored'), cell('stored_unique')])
    add('Stored examples per earlier task at the end of the stream: mean (min–max over seeds), tasks 1 / 2 / 3',
        ['Stream', 'Memory', 'Method', 'Stored', 'Distinct'], rows)
    return '\n'.join(T)


def figure(df):
    """Method minus fine-tuning, final accuracy, against memory size (paired by seed;
    bars are two standard errors)."""
    pa.style()
    fig, axes = plt.subplots(1, 3, figsize=(180 * pa.MM, 62 * pa.MM), sharey=True)
    x = np.arange(len(MEMORY))
    for ax, g, letter in zip(axes, STREAMS, 'abc'):
        ax.grid(color='gray', alpha=0.7, linewidth=0.4, zorder=0)
        ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontweight='bold', fontsize=9)
        ax.axhline(0, color=NIGHT, linewidth=0.8, zorder=2)
        for m, color, marker, label, off in ((ER, CYAN, '^', 'Replay (ER)', -0.06), (PDCL, RED, 'D', 'PDCL', 0.06)):
            v = np.array([diff(df, g, m, FT, mem) for mem in MEMORY])
            ax.errorbar(x + off, v[:, 0], 2 * v[:, 1], color=color, marker=marker, capsize=2, label=label, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(['200', '1,000', '3,000', 'Full'])
        ax.set_xlabel(f'Memory Size (Stream `{g}`)'.replace('`', ''))
    axes[0].set_ylabel('Final Accuracy Minus\nFine-Tuning (Points)')
    axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(pad=0.6, w_pad=1.5)
    pa.FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(pa.FIG / f'fig_r1_memory.{ext}', dpi=300)
    plt.close(fig)


def main():
    df = load()
    df.round(5).to_csv('results/r1_runs.csv', index=False)
    text = tables(df)
    Path('results/r1_tables.md').write_text(text)
    figure(df)
    print(text)


if __name__ == '__main__':
    main()
