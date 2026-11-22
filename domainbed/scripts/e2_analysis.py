"""E2, "Is the adaptivity needed?": tables and figures from the per-run table.

    python -m domainbed.scripts.stream_analysis outputs/e2 results/e2_runs_stream.csv
    python -m domainbed.scripts.stream_analysis outputs/e2_loo results/e2_runs_loo.csv
    python -m domainbed.scripts.e2_analysis

Reads results/e2_runs_stream.csv (whole-stream runs) and results/e2_runs_loo.csv
(leave-one-task-out runs); writes results/e2_tables.md and
paper/figures/fig_e2_{weight,memory,heldout}.{pdf,png}.
All differences are paired by seed; "a ± b" is mean ± standard error over seeds.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from domainbed.scripts import paper_assets as pa
from domainbed.scripts.paper_assets import CYAN, NIGHT, ORANGE, RED, plt

STREAMS = ['rounds', 'skew0.6', 'skew0.8', 'skew0.95', 'aglen']
NAME = {'rounds': 'rounds', 'skew0.6': 'skew:0.6', 'skew0.8': 'skew:0.8', 'skew0.95': 'skew:0.95',
        'aglen': 'aglen', 'skew0.9': 'skew:0.9'}
WEIGHTS = [0.01, 0.03, 0.1, 0.3, 1, 3]
ER = {w: f'ER__w{w}' for w in WEIGHTS}
BUF = {'ER': {50: 'ER__buf50', 200: 'ER__w1', 1000: 'ER__buf1000'},
       'PDCL_RS': {50: 'PDCL_RS__buf50', 200: 'PDCL_RS', 1000: 'PDCL_RS__buf1000'}}
PDCL_VARIANTS = {'dual step 0.001': 'PDCL_RS__dlr0.001', 'dual step 0.1': 'PDCL_RS__dlr0.1',
                 'epsilon x1.0': 'PDCL_RS__es1.0', 'epsilon x1.5': 'PDCL_RS__es1.5',
                 'epsilon x2.0': 'PDCL_RS__es2.0'}
MARGIN = 0.5                      # pre-registered: "within 0.5 points of PDCL"


def seeds(df, group, method, value='final_acc'):
    """Value per seed (averaged over the held-out tasks for leave-one-out runs)."""
    d = df[(df['group'] == group) & (df['method'] == method)]
    return d.groupby('seed')[value].mean()


def ms(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def stat(df, group, method, value='final_acc'):
    return ms(seeds(df, group, method, value))


def diff(df, group, a, b, value='final_acc'):
    """Paired difference a - b over the seeds both have."""
    d = (seeds(df, group, a, value) - seeds(df, group, b, value)).dropna()
    return ms(d)


def pm(m, s, nd=2):
    return f'{m:.{nd}f} ± {s:.{nd}f}'


def spm(m, s, nd=2):
    """Signed difference; bold when it exceeds two standard errors."""
    t = f'{m:+.{nd}f} ± {s:.{nd}f}'
    return f'**{t}**' if abs(m) > 2 * s else t


def md(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]) + '\n'


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def tables(df, loo):
    T = []
    add = lambda title, header, rows: T.append(f'### {title}\n\n' + md(header, rows))
    head = [NAME[g] for g in STREAMS]

    methods = [('Joint ERM', 'ERM'), ('Fine-tuning', 'Finetune')] + \
              [(f'ER w={w}', ER[w]) for w in WEIGHTS] + [('PDCL', 'PDCL_RS')] + \
              [(f'PDCL, {k}', v) for k, v in PDCL_VARIANTS.items()] + \
              [(f'ER w=1, memory {b}', BUF['ER'][b]) for b in (50, 1000)] + \
              [(f'PDCL, memory {b}', BUF['PDCL_RS'][b]) for b in (50, 1000)]
    for value, title in (('final_acc', 'T1. Final accuracy (%)'), ('forgetting', 'T2. Forgetting (points)'),
                         ('learned_acc', 'T3. Accuracy of each task at the end of its own phase (%)')):
        add(title, ['Method'] + head,
            [[lab] + [pm(*stat(df, g, m, value)) for g in STREAMS] for lab, m in methods])

    add('T4. Fixed-weight replay minus PDCL, final accuracy (paired)', ['w'] + head,
        [[w] + [spm(*diff(df, g, ER[w], 'PDCL_RS')) for g in STREAMS] for w in WEIGHTS])
    add('T5. Fixed-weight replay minus fine-tuning, final accuracy (paired)', ['w'] + head,
        [[w] + [spm(*diff(df, g, ER[w], 'Finetune')) for g in STREAMS] for w in WEIGHTS] +
        [['PDCL'] + [spm(*diff(df, g, 'PDCL_RS', 'Finetune')) for g in STREAMS]] +
        [['Joint ERM'] + [spm(*diff(df, g, 'ERM', 'Finetune')) for g in STREAMS]])
    add('T6. Fixed-weight replay minus PDCL, forgetting (paired; negative = forgets less)', ['w'] + head,
        [[w] + [spm(*diff(df, g, ER[w], 'PDCL_RS', 'forgetting')) for g in STREAMS] for w in WEIGHTS] +
        [['Fine-tuning'] + [spm(*diff(df, g, 'Finetune', 'PDCL_RS', 'forgetting')) for g in STREAMS]])

    # the pre-registered question
    rows = []
    for w in WEIGHTS:
        d = {g: diff(df, g, ER[w], 'PDCL_RS') for g in STREAMS}
        worst = min(d, key=lambda g: d[g][0])
        far = max(d, key=lambda g: abs(d[g][0]))
        rows.append([w, NAME[worst], spm(*d[worst]), NAME[far], spm(*d[far]),
                     'yes' if d[worst][0] >= -MARGIN else 'no',
                     'yes' if abs(d[far][0]) <= MARGIN else 'no',
                     f'{np.mean([d[g][0] for g in STREAMS]):+.2f}'])
    add('T7. One fixed weight for every stream? ER(w) minus PDCL',
        ['w', 'worst stream', 'difference there', 'largest gap (either sign)', 'difference there',
         'never more than 0.5 below PDCL', 'within ±0.5 everywhere', 'mean over streams'], rows)

    rows = []
    for g in STREAMS:
        best = max(WEIGHTS, key=lambda w: stat(df, g, ER[w])[0])
        ft = max(WEIGHTS, key=lambda w: diff(df, g, ER[w], 'Finetune')[0])
        assert best == ft
        dual = np.mean([stat(df, g, 'PDCL_RS', f'dual_mean_{k}')[0] for k in range(3)])
        rows.append([NAME[g], best, pm(*stat(df, g, ER[best])), pm(*stat(df, g, 'PDCL_RS')),
                     spm(*diff(df, g, ER[best], 'PDCL_RS')), spm(*diff(df, g, ER[best], 'Finetune')),
                     f'{dual:.3f}'])
    add('T8. Oracle-tuned replay: the best fixed weight of each stream against PDCL',
        ['Stream', 'best w', 'ER(best w)', 'PDCL', 'ER(best w) − PDCL', 'ER(best w) − fine-tuning',
         'PDCL mean dual'], rows)

    rows = []
    for g in STREAMS:
        r = [NAME[g]]
        for key in ('dual_mean', 'dual'):
            r += [pm(*stat(df, g, 'PDCL_RS', f'{key}_{k}'), 3) for k in range(3)]
        r += [pm(*stat(df, g, 'PDCL_RS', f'eps_{k}'), 3) for k in range(3)]
        r.append(f"{(seeds(df, g, 'PDCL_RS', 'dual_0') + seeds(df, g, 'PDCL_RS', 'dual_1') + seeds(df, g, 'PDCL_RS', 'dual_2') == 0).mean():.0%}")
        rows.append(r)
    add('T9. PDCL dual variables (the replay weight PDCL uses), default settings',
        ['Stream'] + [f'mean after phase, task {k}' for k in range(3)] + [f'final, task {k}' for k in range(3)]
        + [f'epsilon, task {k}' for k in range(3)] + ['seeds with all final duals 0'], rows)

    rows = []
    for lab, m in PDCL_VARIANTS.items():
        rows.append([lab] + [spm(*diff(df, g, m, 'PDCL_RS')) for g in STREAMS])
    for lab, m in PDCL_VARIANTS.items():
        rows.append([lab + ', forgetting'] + [spm(*diff(df, g, m, 'PDCL_RS', 'forgetting')) for g in STREAMS])
    for lab, m in [('default', 'PDCL_RS')] + list(PDCL_VARIANTS.items()):
        rows.append([lab + ', mean dual'] + [
            f"{np.mean([stat(df, g, m, f'dual_mean_{k}')[0] for k in range(3)]):.3f}" for g in STREAMS])
    for lab, m in [('default', 'PDCL_RS')] + list(PDCL_VARIANTS.items()):
        rows.append([lab + ', minus fine-tuning'] + [spm(*diff(df, g, m, 'Finetune')) for g in STREAMS])
    add('T10. PDCL sensitivity: variant minus default PDCL (final accuracy unless noted)',
        ['Variant'] + head, rows)

    rows = []
    for b in (50, 200, 1000):
        rows.append([f'ER w=1, memory {b}, final acc.'] + [pm(*stat(df, g, BUF['ER'][b])) for g in STREAMS])
        rows.append([f'PDCL, memory {b}, final acc.'] + [pm(*stat(df, g, BUF['PDCL_RS'][b])) for g in STREAMS])
    for b in (50, 200, 1000):
        rows.append([f'ER − fine-tuning, memory {b}'] + [spm(*diff(df, g, BUF['ER'][b], 'Finetune')) for g in STREAMS])
    for b in (50, 200, 1000):
        rows.append([f'PDCL − fine-tuning, memory {b}'] + [spm(*diff(df, g, BUF['PDCL_RS'][b], 'Finetune')) for g in STREAMS])
    for b in (50, 200, 1000):
        rows.append([f'PDCL − ER, memory {b}'] + [spm(*diff(df, g, BUF['PDCL_RS'][b], BUF['ER'][b])) for g in STREAMS])
    for a in ('ER', 'PDCL_RS'):
        for b in (50, 1000):
            rows.append([f'{pa.LABEL[a]}: memory {b} − memory 200'] +
                        [spm(*diff(df, g, BUF[a][b], BUF[a][200])) for g in STREAMS])
    for b in (50, 200, 1000):
        rows.append([f'ER forgetting, memory {b}'] + [pm(*stat(df, g, BUF['ER'][b], 'forgetting')) for g in STREAMS])
    for b in (50, 200, 1000):
        rows.append([f'PDCL forgetting, memory {b}'] + [pm(*stat(df, g, BUF['PDCL_RS'][b], 'forgetting')) for g in STREAMS])
    for b in (50, 200, 1000):
        rows.append([f'PDCL mean dual, memory {b}'] + [
            f"{np.mean([stat(df, g, BUF['PDCL_RS'][b], f'dual_mean_{k}')[0] for k in range(3)]):.3f}"
            for g in STREAMS])
    add('T11. Memory size', ['Quantity'] + head, rows)

    groups = ['aglen', 'skew0.9']
    algs = ['ERM', 'Finetune', 'ER', 'PDCL_RS']
    rows = [[pa.LABEL[a]] + [pm(*stat(loo, g, a, 'test_acc')) for g in groups]
            + [pm(*stat(loo, g, a, 'test_acc_last')) for g in groups] for a in algs]
    add('T12. Held-out task accuracy (%), leave one task out, 3,000 steps',
        ['Method'] + [NAME[g] + ', val.-loss checkpoint' for g in groups] + [NAME[g] + ', last step' for g in groups],
        rows)
    pairs = [('ER', 'Finetune'), ('PDCL_RS', 'Finetune'), ('PDCL_RS', 'ER'), ('ER', 'ERM'), ('PDCL_RS', 'ERM'),
             ('Finetune', 'ERM')]
    rows = [[f'{pa.LABEL[a]} − {pa.LABEL[b]}'] + [spm(*diff(loo, g, a, b, 'test_acc')) for g in groups]
            + [spm(*diff(loo, g, a, b, 'test_acc_last')) for g in groups] for a, b in pairs]
    add('T13. Held-out task accuracy, paired differences',
        ['Difference'] + [NAME[g] + ', val.-loss checkpoint' for g in groups] + [NAME[g] + ', last step' for g in groups],
        rows)
    rows = []
    for g in groups:
        for e in range(4):
            sub = loo[loo['env'] == e]
            rows.append([NAME[g], e] + [pm(*stat(sub, g, a, 'test_acc')) for a in algs]
                        + [spm(*diff(sub, g, 'ER', 'Finetune', 'test_acc')),
                           spm(*diff(sub, g, 'PDCL_RS', 'Finetune', 'test_acc')),
                           spm(*diff(sub, g, 'PDCL_RS', 'ER', 'test_acc'))])
    add('T14. Held-out accuracy by held-out task', ['Stream', 'held-out task'] + [pa.LABEL[a] for a in algs]
        + ['ER − fine-tuning', 'PDCL − fine-tuning', 'PDCL − ER'], rows)
    rows = []
    for g in groups:
        for k in range(2):
            rows.append([NAME[g], k, pm(*stat(loo, g, 'PDCL_RS', f'dual_mean_{k}'), 3),
                         pm(*stat(loo, g, 'PDCL_RS', f'dual_{k}'), 3)])
    add('T15. PDCL dual variables in the leave-one-out runs', ['Stream', 'task', 'mean after phase', 'final'], rows)
    # exploratory: shortfall of each method from the best candidate of each stream
    cand = [('Fine-tuning', 'Finetune')] + [(f'ER w={w}', ER[w]) for w in WEIGHTS] + [('PDCL', 'PDCL_RS')]
    best = {g: max((m for _, m in cand), key=lambda m: stat(df, g, m)[0]) for g in STREAMS}
    rows = []
    for lab, m in cand:
        d = [diff(df, g, m, best[g]) if m != best[g] else (0.0, 0.0) for g in STREAMS]
        rows.append([lab] + [spm(*x) if x[1] else 'best' for x in d]
                    + [f'{min(x[0] for x in d):+.2f}', f'{np.mean([x[0] for x in d]):+.2f}'])
    add('T16 (exploratory). Shortfall from the best of fine-tuning, the six fixed weights and PDCL on each stream',
        ['Method'] + head + ['worst stream', 'mean over streams'], rows)

    rows = []
    for g in STREAMS:
        for lab, m in (('ER w=1', 'ER__w1'), ('PDCL', 'PDCL_RS')):
            rows.append([NAME[g], lab] + [f"{stat(df, g, m, f'replay_loss_{k}')[0]:.3f}" for k in range(3)]
                        + [f"{stat(df, g, m, f'eps_{k}')[0]:.3f}" for k in range(3)]
                        + [f"{stat(df, g, m, f'acc_{k}')[0]:.1f}" for k in range(4)])
    add('T17 (exploratory). Loss on the stored examples at the end of the stream, the constraint level, '
        'and final validation accuracy per task',
        ['Stream', 'Method'] + [f'memory loss, task {k}' for k in range(3)] + [f'epsilon, task {k}' for k in range(3)]
        + [f'val. acc., task {k}' for k in range(4)], rows)
    return '\n'.join(T)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def save(fig, name, **layout):
    fig.tight_layout(**({'pad': 0.6, 'w_pad': 1.0} | layout))
    for ext in ('pdf', 'png'):
        fig.savefig(pa.FIG / f'{name}.{ext}', dpi=300)
    plt.close(fig)


def grid(n, h=62, letters=True):
    fig, axes = plt.subplots(1, n, figsize=(180 * pa.MM, h * pa.MM), squeeze=False)
    for ax, letter in zip(axes[0], 'abcdef'):
        ax.grid(color='gray', alpha=0.7, linewidth=0.4, zorder=0)
        if letters:
            ax.text(-0.3, 1.08, letter, transform=ax.transAxes, fontweight='bold', fontsize=9)
    return fig, axes[0]


def band(ax, m, s, color, label, ls='-'):
    ax.axhspan(m - s, m + s, color=color, alpha=0.25, linewidth=0, zorder=1)
    ax.axhline(m, color=color, linestyle=ls, linewidth=1.1, zorder=2, label=label)


def fig_weight(df):
    """Final accuracy against the fixed replay weight, one panel per stream."""
    fig, axes = grid(len(STREAMS), h=58)
    for ax, g in zip(axes, STREAMS):
        band(ax, *stat(df, g, 'Finetune'), ORANGE, 'Fine-Tuning')
        band(ax, *stat(df, g, 'PDCL_RS'), RED, 'PDCL')
        ax.axhline(stat(df, g, 'ERM')[0], color=NIGHT, linestyle=':', linewidth=1, zorder=2, label='Joint ERM')
        v = np.array([stat(df, g, ER[w]) for w in WEIGHTS])
        ax.errorbar(WEIGHTS, v[:, 0], yerr=v[:, 1], color=CYAN, marker='o', capsize=2, zorder=3,
                    markeredgecolor=NIGHT, markeredgewidth=0.4, label='Replay, Fixed Weight')
        dual = np.mean([stat(df, g, 'PDCL_RS', f'dual_mean_{k}')[0] for k in range(3)])
        if dual > 0:
            ax.axvline(max(dual, 0.006), color=RED, linestyle='--', linewidth=0.9, zorder=2,
                       label='PDCL Mean Dual Variable')
        ax.set_xscale('log')
        ax.set_xlim(0.005, 5)
        ax.set_xticks([0.01, 0.1, 1])
        ax.set_xticklabels(['0.01', '0.1', '1'])
        ax.set_xlabel('Replay Weight w')
        ax.set_title(NAME[g], fontsize=8)
    axes[0].set_ylabel('Final Accuracy (%)')
    h, l = axes[-1].get_legend_handles_labels()
    if len(h) < 5:
        for ax in axes:
            hh, ll = ax.get_legend_handles_labels()
            if len(hh) > len(h):
                h, l = hh, ll
    fig.legend(h, l, loc='upper center', ncol=5, columnspacing=1.4, handlelength=1.8)
    save(fig, 'fig_e2_weight', rect=(0, 0, 1, 0.92))


def fig_memory(df):
    """Final accuracy against memory size for ER and PDCL, one panel per stream."""
    fig, axes = grid(len(STREAMS), h=58)
    sizes = [50, 200, 1000]
    for ax, g in zip(axes, STREAMS):
        band(ax, *stat(df, g, 'Finetune'), ORANGE, 'Fine-Tuning')
        ax.axhline(stat(df, g, 'ERM')[0], color=NIGHT, linestyle=':', linewidth=1, zorder=2, label='Joint ERM')
        for a, color, off in (('ER', CYAN, 0.94), ('PDCL_RS', RED, 1.06)):
            v = np.array([stat(df, g, BUF[a][b]) for b in sizes])
            ax.errorbar(np.array(sizes) * off, v[:, 0], yerr=v[:, 1], color=color, marker='o', capsize=2,
                        zorder=3, markeredgecolor=NIGHT, markeredgewidth=0.4,
                        label='Replay (w = 1)' if a == 'ER' else 'PDCL')
        ax.set_xscale('log')
        ax.set_xlim(30, 1700)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(b) for b in sizes])
        ax.minorticks_off()
        ax.set_xlabel('Memory Size')
        ax.set_title(NAME[g], fontsize=8)
    axes[0].set_ylabel('Final Accuracy (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, columnspacing=1.4, handlelength=1.8)
    save(fig, 'fig_e2_memory', rect=(0, 0, 1, 0.92))


def fig_heldout(loo):
    """Held-out task accuracy of the four methods on the two streams."""
    groups, algs = ['aglen', 'skew0.9'], ['ERM', 'Finetune', 'ER', 'PDCL_RS']
    fig, axes = plt.subplots(1, 2, figsize=(120 * pa.MM, 62 * pa.MM), squeeze=False)
    for ax, g, letter in zip(axes[0], groups, 'ab'):
        ax.grid(color='gray', alpha=0.7, linewidth=0.4, zorder=0, axis='y')
        ax.text(-0.25, 1.08, letter, transform=ax.transAxes, fontweight='bold', fontsize=9)
        rng = np.random.RandomState(0)
        lo = 100
        for i, a in enumerate(algs):
            v = seeds(loo, g, a, 'test_acc').values
            m, s = ms(v)
            ax.bar(i, m, yerr=s, color=pa.COLOR[a], edgecolor=NIGHT, linewidth=0.5, capsize=2.5,
                   error_kw={'linewidth': 0.9}, zorder=3, width=0.7)
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=6, color='white', edgecolor=NIGHT,
                       linewidth=0.4, zorder=4)
            lo = min(lo, v.min())
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels(['Joint\nERM', 'Fine-\nTuning', 'Replay\n(w = 1)', 'PDCL'])
        ax.set_ylim(np.floor(lo - 1), None)
        ax.set_title(NAME[g], fontsize=8)
        ax.set_ylabel('Held-Out Task Accuracy (%)')
    save(fig, 'fig_e2_heldout', w_pad=2.0)


def main():
    df = pd.read_csv('results/e2_runs_stream.csv')
    loo = pd.read_csv('results/e2_runs_loo.csv')
    Path('results/e2_tables.md').write_text(tables(df, loo))
    pa.style()
    fig_weight(df)
    fig_memory(df)
    fig_heldout(loo)


if __name__ == '__main__':
    main()
