"""H3, "Dual variables that start at 1": per-run table, tables and figure from the raw runs.

    python -m domainbed.scripts.h3_analysis

Reads outputs/h3/<stream>/<method>/env-1_seed<s>/results.jsonl; writes
results/h3_runs.csv (one row per run), results/h3_curves.csv (dual variables of
the PDCL runs at every checkpoint), results/h3_tables.md and
paper/figures/fig_h3_init.{pdf,png}. With --cached the two tables are read back.

Methods: fine-tuning; ER with replay weight 1; PDCL (PDCL_RS, default: dual
variables start at 0); PDCL init 1 (hparam dual_init = 1).
All differences are paired by seed; "a ± b" is mean ± standard error over seeds;
bold = larger than two standard errors.
F1's numbers (results/f1_runs.csv) are quoted for context; they are other runs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from domainbed.scripts import paper_assets as pa
from domainbed.scripts.f1_analysis import diff, grid, md, ms, pm, save, seeds, spm, stat
from domainbed.scripts.paper_assets import CYAN, GREY, NIGHT, ORANGE, RED
from domainbed.scripts.stream_analysis import run_row

ROOT = Path('outputs/h3')
STREAMS = ['skew0.7', 'skew0.8', 'skew0.9', 'skew0.95', 'aglen', 'rounds']
NAME = {g: g.replace('skew', 'skew:') for g in STREAMS}
BUFFERS = [200, 1000]
FT, ER, DEFAULT, INIT1 = 'Fine-tuning', 'ER w=1', 'PDCL', 'PDCL init 1'
METHODS = [FT, ER, DEFAULT, INIT1]
ORACLE = 'PDCL oracle'            # F1 only
DUAL_STEPS = [1000, 1250, 1500, 2000, 3000, 3999]


def load():
    rows, curves = [], []
    for done in sorted(ROOT.rglob('done')):
        d = done.parent
        recs = [json.loads(l) for l in open(d / 'results.jsonl')]
        hp, alg = recs[-1]['hparams'], recs[-1]['args']['algorithm']
        if alg == 'Finetune':
            method, buffer = FT, 0
        elif alg == 'ER':
            assert hp['replay_weight'] == 1
            method, buffer = ER, hp['buffer_size']
        else:
            method, buffer = (INIT1 if hp.get('dual_init', 0) == 1 else DEFAULT), hp['buffer_size']
        row = {'group': d.parent.parent.name, 'method': method, 'buffer': buffer} | run_row(recs)
        if alg == 'PDCL_RS':
            n = row['n_tasks']
            row['dual_mean'] = float(np.mean([row[f'dual_mean_{k}'] for k in range(n - 1)]))
            for r in recs:
                curves.append({'group': row['group'], 'method': method, 'buffer': buffer, 'seed': row['seed'],
                               'step': r['step']} | {f'dual_{k}': r.get(f'dual_{k}', np.nan) for k in range(n - 1)})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(curves).round(5)


def closed(df, g, b, variant=INIT1, er=ER):
    """Share of the gap between default PDCL and ER that the variant closes (ratio of means)."""
    gap, d = diff(df, g, er, DEFAULT, b), diff(df, g, variant, DEFAULT, b)
    return f'{d[0] / gap[0]:.0%}' if gap[0] > 2 * gap[1] else 'no gap'


def dual_at(curves, g, m, b, k, step):
    c = curves[(curves['group'] == g) & (curves['method'] == m) & (curves['buffer'] == b) & (curves['step'] == step)]
    return ms(c[f'dual_{k}'])


def tables(df, curves, f1):
    T = []
    add = lambda title, header, rows: T.append(f'### {title}\n\n' + md(header, rows))
    head = [NAME[g] for g in STREAMS]
    n = df.groupby(['group', 'method', 'buffer']).size()
    T.append(f'{len(df)} runs; seeds per stream, method and memory: {n.min()} to {n.max()}.\n')

    for b in BUFFERS:
        add(f'T{len(T)}. Final accuracy (%), memory {b}', ['Method'] + head,
            [[m] + [pm(*stat(df, g, m, b)) for g in STREAMS] for m in METHODS])

    rows = []
    for b in BUFFERS:
        for ref in (DEFAULT, ER, FT):
            rows.append([f'init 1 − {ref}, memory {b}'] + [spm(*diff(df, g, INIT1, ref, b)) for g in STREAMS])
        rows.append([f'ER w=1 − PDCL (the gap), memory {b}'] + [spm(*diff(df, g, ER, DEFAULT, b)) for g in STREAMS])
        rows.append([f'share of the gap closed, memory {b}'] + [closed(df, g, b) for g in STREAMS])
        rows.append([f'PDCL − Fine-tuning, memory {b}'] + [spm(*diff(df, g, DEFAULT, FT, b)) for g in STREAMS])
        rows.append([f'ER w=1 − Fine-tuning, memory {b}'] + [spm(*diff(df, g, ER, FT, b)) for g in STREAMS])
    add(f'T{len(T)}. Final accuracy differences (paired by seed). "no gap" = ER w=1 is not above default PDCL '
        'by two standard errors', ['Quantity'] + head, rows)

    rows = []
    for g in ('aglen', 'rounds'):
        for b in BUFFERS:
            d = diff(df, g, INIT1, FT, b)
            rows.append([f'{NAME[g]}, memory {b}', 'within 2 SE of fine-tuning', spm(*d),
                         'held' if abs(d[0]) <= 2 * d[1] else 'failed'])
    for g in ('skew0.9', 'skew0.95'):
        for b in BUFFERS:
            gap, d = diff(df, g, ER, DEFAULT, b), diff(df, g, INIT1, DEFAULT, b)
            if gap[0] > 2 * gap[1]:
                out = 'held' if d[0] / gap[0] < 0.5 else 'failed'
            else:
                out = 'not testable (no gap)'
            rows.append([f'{NAME[g]}, memory {b}', 'closes less than half of ER w=1 − PDCL',
                         f'{spm(*d)} of {spm(*gap)} ({closed(df, g, b)})', out])
    add(f'T{len(T)}. Pre-registered predictions', ['Cell', 'Prediction', 'Measured', 'Outcome'], rows)

    rows = []
    for m in (DEFAULT, INIT1):
        for b in BUFFERS:
            rows.append([f'{m}, memory {b}'] + [pm(*stat(df, g, m, b, 'dual_mean'), 3) for g in STREAMS])
    add(f'T{len(T)}. Dual variables (mean over the checkpoints after each task\'s phase and over earlier tasks)',
        ['Method'] + head, rows)

    rows = []
    for g in ('skew0.9', 'aglen'):
        for b in BUFFERS:
            for m in (DEFAULT, INIT1):
                rows.append([f'{NAME[g]}, memory {b}, {m}'] +
                            [pm(*dual_at(curves, g, m, b, 0, s), 3) for s in DUAL_STEPS])
    add(f'T{len(T)}. Dual variable of the first task over training (it becomes an earlier task after step 999; '
        'the checkpoint at step 1000 is one dual step later)', ['Run'] + [f'step {s}' for s in DUAL_STEPS], rows)

    rows = []
    for g in ('skew0.9', 'aglen'):
        for b in BUFFERS:
            rows.append([f'{NAME[g]}, memory {b}'] + [
                ' / '.join(f"{stat(df, g, m, b, f'dual_{k}')[0]:.3f}" for k in range(3)) for m in (DEFAULT, INIT1)])
    add(f'T{len(T)}. Dual variables of tasks 1 / 2 / 3 at the last step', ['Cell', DEFAULT, INIT1], rows)

    # Context: F1 (other runs, same streams, memory sizes and seeds).
    rows = []
    for b in BUFFERS:
        rows.append([f'F1: oracle − PDCL, memory {b}'] + [spm(*diff(f1, g, ORACLE, DEFAULT, b)) for g in STREAMS])
        rows.append([f'F1: ER w=1 − PDCL, memory {b}'] + [spm(*diff(f1, g, ER, DEFAULT, b)) for g in STREAMS])
        rows.append([f'F1: share of that gap closed by the oracle, memory {b}'] +
                    [closed(f1, g, b, ORACLE) for g in STREAMS])
        rows.append([f'F1: oracle − Fine-tuning, memory {b}'] + [spm(*diff(f1, g, ORACLE, FT, b)) for g in STREAMS])
        rows.append([f'H3: init 1 − PDCL, memory {b}'] + [spm(*diff(df, g, INIT1, DEFAULT, b)) for g in STREAMS])
        rows.append([f'H3: share closed by init 1, memory {b}'] + [closed(df, g, b) for g in STREAMS])
    add(f'T{len(T)}. Context only: F1\'s oracle (constraints measured on validation data), from results/f1_runs.csv. '
        'These are other runs; differences are paired within F1', ['Quantity'] + head, rows)

    rows = []
    for m in (FT, ER, DEFAULT):
        for b in (BUFFERS if m != FT else [0]):
            rows.append([m if m == FT else f'{m}, memory {b}'] + [
                spm(*ms(seeds(df, g, m, b) - seeds(f1, g, m, b))) for g in STREAMS])
    add(f'T{len(T)}. Check: the rerun methods in H3 minus the same configuration and seeds in F1 '
        '(repeats on the shared GPUs are not bit-identical)', ['Method'] + head, rows)
    return '\n'.join(T)


def figure(df, curves):
    fig, axes = grid(1, 4, h=72, x=-0.3)
    above = dict(loc='lower center', bbox_to_anchor=(0.5, 1.12), fontsize=5.5, handletextpad=0.3, borderpad=0.3,
                 labelspacing=0.25, columnspacing=0.8)
    color = {DEFAULT: RED, ER: CYAN, INIT1: NIGHT}
    label = {DEFAULT: 'PDCL', ER: 'Replay w=1', INIT1: 'PDCL Init 1'}
    # a: init 1 minus default PDCL and minus ER; b: init 1 and default PDCL minus fine-tuning.
    panels = [([(INIT1, DEFAULT, RED, 'Minus PDCL'), (INIT1, ER, CYAN, 'Minus Replay w=1')],
               'PDCL Init 1 Minus Reference\n(Points of Final Accuracy)'),
              ([(DEFAULT, FT, RED, 'PDCL'), (INIT1, FT, NIGHT, 'PDCL Init 1')],
               'Final Accuracy Minus\nFine-Tuning (Points)')]
    for ax, (pairs, ylabel) in zip(axes[0, :2], panels):
        ax.axhline(0, color=NIGHT, linewidth=0.8, zorder=2)
        series = [(p, b) for p in pairs for b in BUFFERS]
        for i, ((m, ref, col, name), b) in enumerate(series):
            v = np.array([diff(df, g, m, ref, b) for g in STREAMS])
            x = np.arange(len(STREAMS)) + (i - (len(series) - 1) / 2) * 0.17
            ax.errorbar(x, v[:, 0], yerr=2 * v[:, 1], color=col, marker='o' if b == 200 else 's',
                        markerfacecolor=col if b == 200 else 'white', linestyle='none', capsize=1.5,
                        markeredgecolor=NIGHT if b == 200 else col, markeredgewidth=0.6, ecolor=col,
                        elinewidth=0.9, zorder=3, label=f'{name}, Memory {b}')
        ax.set_xticks(range(len(STREAMS)))
        ax.set_xticklabels([NAME[g] for g in STREAMS], rotation=40, ha='right')
        ax.set_ylabel(ylabel)
        ax.legend(ncol=1, **above)

    # c, d: the dual variable of every earlier task from the step it is created.
    for ax, g in zip(axes[0, 2:], ('skew0.9', 'aglen')):
        c = curves[curves['group'] == g]
        for m in (DEFAULT, INIT1):
            for b, style in zip(BUFFERS, ('-', '--')):
                for k in range(3):
                    d = c[(c['method'] == m) & (c['buffer'] == b)].pivot(index='step', columns='seed', values=f'dual_{k}')
                    d = d[d.index >= (k + 1) * 1000]
                    mean, se = d.mean(1), d.std(1, ddof=1) / np.sqrt(d.shape[1])
                    ax.fill_between(d.index, mean - se, mean + se, color=color[m], alpha=0.25, linewidth=0, zorder=2)
                    ax.plot(d.index, mean, color=color[m], linestyle=style, linewidth=1.0, zorder=3,
                            label=f'{label[m]}, Memory {b}' if k == 0 else None)
        for edge in (1000, 2000, 3000):
            ax.axvline(edge, color=GREY, linewidth=0.6, linestyle=':', zorder=1)
        ax.set_xlabel('Training Step')
        ax.set_xticks([0, 1000, 2000, 3000, 4000])
        ax.set_xticklabels(['0', '1k', '2k', '3k', '4k'])
        ax.set_xlim(0, 4000)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(f'Dual Variables of Earlier\nTasks, {NAME[g]}')
        ax.legend(ncol=1, **above)
    save(fig, 'fig_h3_init', w_pad=1.2)


def main():
    if '--cached' in sys.argv:
        df, curves = pd.read_csv('results/h3_runs.csv'), pd.read_csv('results/h3_curves.csv')
    else:
        df, curves = load()
        df.to_csv('results/h3_runs.csv', index=False)
        curves.to_csv('results/h3_curves.csv', index=False)
    f1 = pd.read_csv('results/f1_runs.csv')
    Path('results/h3_tables.md').write_text(tables(df, curves, f1))
    pa.style()
    figure(df, curves)


if __name__ == '__main__':
    main()
