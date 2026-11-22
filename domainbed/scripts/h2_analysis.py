"""H2, "Is the forgetting a shifted output bias?": per-run table, tables and figure from the raw runs.

    python -m domainbed.scripts.h2_analysis

Reads every finished run under outputs/h2 (results.jsonl); writes results/h2_runs.csv
(one row per run, columns of stream_analysis.py), results/h2_tables.md and
paper/figures/fig_h2_balance.{pdf,png}.

Methods: Finetune, Finetune__bal (fine-tuning with the class-balanced loss, hparam
balanced_loss), ER, PDCL_RS. "a ± b" is mean ± standard error over the ten seeds;
differences are paired by seed; bold = larger than two standard errors.

Share of ER's gain recovered by balancing, on a stream where ER − fine-tuning exceeds
two standard errors:

    share = mean_seed(balanced − fine-tuning) / mean_seed(ER − fine-tuning)

with a 95% percentile interval from a paired bootstrap over seeds (the same resampled
seeds in numerator and denominator, 20,000 draws, fixed generator seed).

Balanced accuracy (mean of the two class recalls) is not computed: the trainer logs
accuracy and loss per task and split, not per class, and no model weights are saved.
"""
from pathlib import Path

import numpy as np

from domainbed.scripts import paper_assets as pa
from domainbed.scripts import stream_analysis as sa
from domainbed.scripts.paper_assets import CYAN, GREY, NIGHT, ORANGE, RED, YELLOW, plt

ROOT = Path('outputs/h2')
STREAMS = ['skew0.7', 'skew0.8', 'skew0.9', 'skew0.95', 'drift0.4', 'drift0.6', 'drift0.8',
           'agrate', 'rounds', 'aglen']
NAME = {g: g.replace('skew', 'skew:').replace('drift', 'drift:') for g in STREAMS}
FT, BAL, ER, PDCL = 'Finetune', 'Finetune__bal', 'ER', 'PDCL_RS'
METHODS = [FT, BAL, ER, PDCL]
LABEL = {FT: 'Fine-Tuning', BAL: 'Fine-Tuning, Balanced Loss', ER: 'Replay (ER)', PDCL: 'PDCL'}
COLOR = {FT: ORANGE, BAL: YELLOW, ER: CYAN, PDCL: RED}
N_BOOT = 20000


def seeds(df, g, m, value):
    return df[(df['group'] == g) & (df['method'] == m)].set_index('seed')[value].sort_index()


def stat(df, g, m, value):
    v = seeds(df, g, m, value)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def diff(df, g, a, b, value):
    d = (seeds(df, g, a, value) - seeds(df, g, b, value)).dropna()
    assert len(d) == 10, (g, a, b, len(d))
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def pm(v, bold=False, sign=False):
    s = f'{v[0]:+.2f} ± {v[1]:.2f}' if sign else f'{v[0]:.2f} ± {v[1]:.2f}'
    return f'**{s}**' if bold and abs(v[0]) > 2 * v[1] else s


def share(df, g, value='final_acc'):
    """Ratio of mean gains and its paired-bootstrap 95% interval."""
    num = (seeds(df, g, BAL, value) - seeds(df, g, FT, value)).values
    den = (seeds(df, g, ER, value) - seeds(df, g, FT, value)).values
    rng = np.random.RandomState(0)
    i = rng.randint(0, len(num), (N_BOOT, len(num)))
    boot = num[i].mean(1) / den[i].mean(1)
    return num.mean() / den.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def md(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
                     + ['| ' + ' | '.join(r) + ' |' for r in rows]) + '\n'


def tables(df):
    out, real, shares = [], [], {}
    for value, title in (('final_acc', 'final accuracy (%)'), ('forgetting', 'forgetting (points)')):
        rows = [[f'`{NAME[g]}`'] + [pm(stat(df, g, m, value)) for m in METHODS] for g in STREAMS]
        out.append(f'## T{len(out) + 1}. {title.capitalize()}\n\n'
                   + md(['Stream'] + [LABEL[m] for m in METHODS], rows))
        rows = [[f'`{NAME[g]}`'] + [pm(diff(df, g, a, b, value), bold=True, sign=True)
                                    for a, b in ((BAL, FT), (ER, FT), (PDCL, FT), (BAL, ER))] for g in STREAMS]
        out.append(f'## T{len(out) + 1}. Paired differences, {title}\n\n'
                   + md(['Stream', 'Balanced − fine-tuning', 'ER − fine-tuning', 'PDCL − fine-tuning',
                         'Balanced − ER'], rows))
    rows = []
    for g in STREAMS:
        er = diff(df, g, ER, FT, 'final_acc')
        if er[0] > 2 * er[1]:
            real.append(g)
            shares[g] = share(df, g)
            s = shares[g]
            rows.append([f'`{NAME[g]}`', pm(er, sign=True), pm(diff(df, g, BAL, FT, 'final_acc'), sign=True),
                         f'{s[0]:.2f}', f'[{s[1]:.2f}, {s[2]:.2f}]'])
    out.append(f'## T{len(out) + 1}. Share of ER\'s gain in final accuracy recovered by the balanced loss\n\n'
               'Streams where ER − fine-tuning exceeds two standard errors. Interval: 95%, paired bootstrap over '
               'seeds.\n\n'
               + md(['Stream', 'ER − fine-tuning', 'Balanced − fine-tuning', 'Share recovered', '95% interval'], rows))

    # pre-registered predictions
    verdict = []
    for g in STREAMS[:7]:
        if g in shares:
            s = shares[g]
            verdict.append([f'`{NAME[g]}`: share ≥ 0.5', f'{s[0]:.2f} [{s[1]:.2f}, {s[2]:.2f}]',
                            'held' if s[0] >= 0.5 else '**failed**'])
        else:
            verdict.append([f'`{NAME[g]}`: share ≥ 0.5', 'ER gain not above two standard errors', 'not tested'])
    if 'agrate' in shares:
        s = shares['agrate']
        verdict.append(['`agrate`: share < 0.5', f'{s[0]:.2f} [{s[1]:.2f}, {s[2]:.2f}]',
                        'held' if s[0] < 0.5 else '**failed**'])
    else:
        verdict.append(['`agrate`: share < 0.5', 'ER gain not above two standard errors', 'not tested'])
    for g in ('rounds', 'aglen'):
        d = diff(df, g, BAL, FT, 'final_acc')
        verdict.append([f'`{NAME[g]}`: balanced within two standard errors of fine-tuning', pm(d, sign=True),
                        'held' if abs(d[0]) <= 2 * d[1] else '**failed**'])
    head = '## Pre-registered predictions\n\n' + md(['Prediction', 'Observed', 'Outcome'], verdict)
    return head + '\n' + '\n'.join(out), shares


def figure(df, shares):
    MM = pa.MM
    fig, axes = plt.subplots(3, 1, figsize=(180 * MM, 175 * MM))
    x = np.arange(len(STREAMS))
    offs = np.linspace(-0.27, 0.27, 4)
    for ax, value, ylabel, letter in ((axes[0], 'final_acc', 'Final Accuracy (%)', 'a'),
                                      (axes[1], 'forgetting', 'Forgetting (points)', 'b')):
        for m, off in zip(METHODS, offs):
            v = np.array([stat(df, g, m, value) for g in STREAMS])
            ax.errorbar(x + off, v[:, 0], yerr=v[:, 1], color=COLOR[m], marker='o', linestyle='none', capsize=2,
                        zorder=3, markeredgecolor=NIGHT, markeredgewidth=0.4, label=LABEL[m])
        ax.set_ylabel(ylabel)
        ax.text(-0.08, 1.05, letter, transform=ax.transAxes, fontweight='bold', fontsize=9)
    ax = axes[2]
    for (a, lab, col), off in zip(((BAL, 'Balanced Loss − Fine-Tuning', YELLOW), (ER, 'Replay − Fine-Tuning', CYAN),
                                   (PDCL, 'PDCL − Fine-Tuning', RED)), (-0.2, 0, 0.2)):
        v = np.array([diff(df, g, a, FT, 'final_acc') for g in STREAMS])
        ax.errorbar(x + off, v[:, 0], yerr=v[:, 1], color=col, marker='o', linestyle='none', capsize=2, zorder=3,
                    markeredgecolor=NIGHT, markeredgewidth=0.4, label=lab)
    ax.axhline(0, color=ORANGE, linewidth=1.1, zorder=2)
    lo, top = ax.get_ylim()
    top += 0.14 * (top - lo)
    ax.set_ylim(lo, top)
    for g, s in shares.items():
        ax.text(STREAMS.index(g), top - 0.02 * (top - lo), f'{s[0]:.2f}', ha='center', va='top', fontsize=7,
                color=GREY)
    ax.set_title('Numbers: Share of the Replay Gain Recovered by the Balanced Loss', fontsize=7, color=GREY,
                 loc='right')
    ax.set_ylabel('Difference in Final Accuracy\n(points, paired)')
    ax.text(-0.08, 1.05, 'c', transform=ax.transAxes, fontweight='bold', fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=3, columnspacing=1.4, handlelength=1.2)
    for ax in axes:
        ax.grid(color='gray', alpha=0.7, linewidth=0.4, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([NAME[g] for g in STREAMS])
        ax.set_xlim(-0.6, len(STREAMS) - 0.4)
    axes[2].set_xlabel('Task Stream')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, columnspacing=1.4, handlelength=1.2)
    fig.tight_layout(pad=0.6, h_pad=1.5, rect=(0, 0, 1, 0.96))
    pa.FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(pa.FIG / 'fig_h2_balance.pdf', dpi=600)
    fig.savefig(pa.FIG / 'fig_h2_balance.png', dpi=300)
    plt.close(fig)


def main():
    df = sa.table(ROOT)
    n = df.groupby(['group', 'method']).size()
    assert len(n) == len(STREAMS) * len(METHODS) and (n == 10).all(), n
    assert (df['steps'] == 4000).all()
    df.to_csv('results/h2_runs.csv', index=False)
    text, shares = tables(df)
    Path('results/h2_tables.md').write_text(
        '# H2 tables\n\nWritten by `python -m domainbed.scripts.h2_analysis` from the raw runs under `outputs/h2` '
        f'({len(df)} runs, ten seeds per setting, 4,000 steps). Mean ± standard error over seeds; differences are '
        'paired by seed; bold = larger than two standard errors. Balanced accuracy is not reported: per-class '
        'accuracy is not logged.\n\n' + text)
    pa.style()
    figure(df, shares)
    print(text)


if __name__ == '__main__':
    main()
