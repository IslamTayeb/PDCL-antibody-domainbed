"""F1, "Honest constraints": per-run table, tables and figures from the raw runs.

    python -m domainbed.scripts.f1_analysis

Reads outputs/f1/<stream>/<method>/env-1_seed<s>/results.jsonl; writes
results/f1_runs.csv (one row per run), results/f1_curves.csv (one row per run
and checkpoint), results/f1_tables.md and paper/figures/fig_f1_*.{pdf,png}.
With --cached the two tables are read back instead of the raw runs.

Methods: fine-tuning; ER with a fixed replay weight w; PDCL (PDCL_RS, default);
PDCL oracle (dual_source=val); PDCL split (dual_source=split).
All differences are paired by seed; "a ± b" is mean ± standard error over seeds;
bold = larger than two standard errors.

Per-run columns beyond those of stream_analysis.run_row (k = earlier task):
    mem_loss, val_loss, held_loss   loss on the replayed stored examples, on the
                 validation set and (split) on the stored half that is not replayed,
                 averaged over the checkpoints after task k's phase and over k
    gap          val_loss - mem_loss
    *_final      the same at the last step
    dual_mean    dual variable averaged over the checkpoints after the phase and over k
    viol_val     share of (task, checkpoint) pairs after the phase at which the
                 validation loss of the task is above 1.1 x its value when the
                 phase ended (the constraint, measured honestly, is violated)
    viol_mem     the same for the loss on the replayed stored examples
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from domainbed.scripts import paper_assets as pa
from domainbed.scripts.paper_assets import GREY, NIGHT, ORANGE, RED, YELLOW, plt
from domainbed.scripts.stream_analysis import run_row

ROOT = Path('outputs/f1')
STREAMS = ['skew0.7', 'skew0.8', 'skew0.9', 'skew0.95', 'aglen', 'rounds']
SKEWED = STREAMS[:4]
NAME = {g: g.replace('skew', 'skew:') for g in STREAMS}
BUFFERS = [200, 1000]
WEIGHTS = [0.03, 0.3, 1, 3]
ER = [f'ER w={w}' for w in WEIGHTS]
DEFAULT, ORACLE, SPLIT = 'PDCL', 'PDCL oracle', 'PDCL split'
VARIANTS = [ORACLE, SPLIT]
METHODS = ['Fine-tuning'] + ER + [DEFAULT, ORACLE, SPLIT]
SOURCE = {'memory': DEFAULT, 'val': ORACLE, 'split': SPLIT}
SCALE = 1.1                       # epsilon_scale of every run
COLOR = {'Fine-tuning': ORANGE, DEFAULT: RED, ORACLE: NIGHT, SPLIT: YELLOW,
         **dict(zip(ER, ['#C9DEDD', '#8FBBBA', '#5E9695', '#2F6665']))}
SHORT = {'Fine-tuning': 'FT', DEFAULT: 'P', ORACLE: 'O', SPLIT: 'S', **{f'ER w={w}': str(w).replace('0.', '.') for w in WEIGHTS}}


# --------------------------------------------------------------------------
# raw runs -> tables
# --------------------------------------------------------------------------
def load():
    rows, curves = [], []
    for done in sorted(ROOT.rglob('done')):
        d = done.parent
        recs = [json.loads(l) for l in open(d / 'results.jsonl')]
        hp, alg = recs[-1]['hparams'], recs[-1]['args']['algorithm']
        if alg == 'Finetune':
            method, buffer = 'Fine-tuning', 0
        elif alg == 'ER':
            method, buffer = f"ER w={hp['replay_weight']}", hp['buffer_size']
        else:
            method, buffer = SOURCE[hp.get('dual_source', 'memory')], hp['buffer_size']
        row = {'group': d.parent.parent.name, 'method': method, 'buffer': buffer} | run_row(recs)
        n, steps = row['n_tasks'], np.array([r['step'] for r in recs])
        phase = row['steps'] // n
        per_task = {key: [] for key in ('mem_loss', 'val_loss', 'held_loss', 'mem_loss_final',
                                        'val_loss_final', 'held_loss_final', 'dual_mean')}
        viol = {'mem': [], 'val': []}
        for k in range(n - 1):
            after = steps >= (k + 1) * phase
            for key in ('mem', 'val', 'held'):
                v = np.array([r.get(f'{key}_loss_{k}', np.nan) for r in recs])
                per_task[f'{key}_loss'].append(v[after].mean())
                per_task[f'{key}_loss_final'].append(v[-1])
                row[f'{key}_loss_{k}'] = v[-1]
                if key != 'held':
                    viol[key] += list(v[after][1:] > SCALE * v[after][0])
            per_task['dual_mean'].append(row[f'dual_mean_{k}'])
        for key, v in per_task.items():
            row[key] = float(np.mean(v))
        row['gap'], row['gap_final'] = row['val_loss'] - row['mem_loss'], row['val_loss_final'] - row['mem_loss_final']
        row['viol_val'] = float(np.mean(viol['val']))
        row['viol_mem'] = float(np.mean(viol['mem'])) if method != 'Fine-tuning' else np.nan
        rows.append(row)
        for r in recs:
            c = {'group': row['group'], 'method': method, 'buffer': buffer, 'seed': row['seed'], 'step': r['step'],
                 'val_acc': 100 * r['agg_val_acc']}
            for k in range(n - 1):
                for key in ('dual', 'eps', 'mem_loss', 'val_loss', 'held_loss'):
                    c[f'{key}_{k}'] = r.get(f'{key}_{k}', np.nan)
            curves.append(c)
    return pd.DataFrame(rows), pd.DataFrame(curves).round(5)


def seeds(df, g, m, b, value='final_acc'):
    """Value per seed; fine-tuning has no memory and serves both memory sizes."""
    d = df[(df['group'] == g) & (df['method'] == m)]
    if m != 'Fine-tuning':
        d = d[d['buffer'] == b]
    return d.set_index('seed')[value].sort_index()


def ms(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def stat(df, g, m, b, value='final_acc'):
    return ms(seeds(df, g, m, b, value))


def diff(df, g, a, b_, buf, value='final_acc'):
    """Paired difference a - b_ over the seeds, same stream and memory."""
    return ms((seeds(df, g, a, buf, value) - seeds(df, g, b_, buf, value)).dropna())


def best_weight(df, g, b):
    """The fixed-weight ER with the highest mean final accuracy (picked on the same seeds)."""
    return max(ER, key=lambda m: stat(df, g, m, b)[0])


def pooled(df, groups, fn):
    """Mean ± SE over seeds of a per-seed series averaged over streams."""
    return ms(pd.concat([fn(g) for g in groups], axis=1).mean(axis=1))


def pm(m, s, nd=2):
    return f'{m:.{nd}f} ± {s:.{nd}f}'


def spm(m, s, nd=2):
    t = f'{m:+.{nd}f} ± {s:.{nd}f}'
    return f'**{t}**' if abs(m) > 2 * s else t


def sig(m, s):
    return '>' if m > 2 * s else '<' if m < -2 * s else '='


def md(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]) + '\n'


def tables(df):
    T = []
    add = lambda title, header, rows: T.append(f'### {title}\n\n' + md(header, rows))
    head = [NAME[g] for g in STREAMS]
    cells = [(g, b) for b in BUFFERS for g in STREAMS]
    best = {(g, b): best_weight(df, g, b) for g, b in cells}

    n = df.groupby(['group', 'method', 'buffer']).size()
    T.append(f'{len(df)} runs; seeds per stream, method and memory: {n.min()} to {n.max()}.\n')

    for value, title in (('final_acc', 'Final accuracy (%)'), ('forgetting', 'Forgetting (points)')):
        for b in BUFFERS:
            add(f'T{len(T)}. {title}, memory {b}', ['Method'] + head,
                [[m] + [pm(*stat(df, g, m, b, value)) for g in STREAMS] for m in METHODS])

    rows = []
    for b in BUFFERS:
        rows.append([f'best w, memory {b}'] + [best[g, b][5:] for g in STREAMS])
        rows.append([f'ER(best w), memory {b}'] + [pm(*stat(df, g, best[g, b], b)) for g in STREAMS])
        rows.append([f'ER(best w) − fine-tuning, memory {b}'] +
                    [spm(*diff(df, g, best[g, b], 'Fine-tuning', b)) for g in STREAMS])
    add(f'T{len(T)}. Best fixed weight (best of four, picked on the same seeds: biased upward)',
        ['Quantity'] + head, rows)

    for ref_name, ref in (('default PDCL', lambda g, b: DEFAULT), ('the best fixed weight', lambda g, b: best[g, b]),
                          ('fine-tuning', lambda g, b: 'Fine-tuning')):
        rows = []
        for b in BUFFERS:
            for m in [DEFAULT] + VARIANTS + (ER if ref_name == 'fine-tuning' else []):
                if m == DEFAULT and ref_name == 'default PDCL':
                    continue
                rows.append([f'{m}, memory {b}'] + [spm(*diff(df, g, m, ref(g, b), b)) for g in STREAMS])
        add(f'T{len(T)}. Minus {ref_name}, final accuracy (paired)', ['Method'] + head, rows)

    # prediction (a)
    rows = []
    for b in BUFFERS:
        for g in SKEWED:
            short = diff(df, g, best[g, b], DEFAULT, b)
            r = [NAME[g], b, best[g, b][5:], spm(*short)]
            for m in VARIANTS:
                d = diff(df, g, m, DEFAULT, b)
                closed = f'{d[0] / short[0]:.0%}' if short[0] > 2 * short[1] else 'no gap'
                r += [spm(*d), closed]
            d = diff(df, g, ORACLE, SPLIT, b)
            rows.append(r + [spm(*d)])
        for label, groups in (('skew:0.8–0.95 pooled', SKEWED[1:]), ('all four skewed pooled', SKEWED)):
            short = pooled(df, groups, lambda g: seeds(df, g, best[g, b], b) - seeds(df, g, DEFAULT, b))
            r = [label, b, '', spm(*short)]
            for m in VARIANTS:
                d = pooled(df, groups, lambda g: seeds(df, g, m, b) - seeds(df, g, DEFAULT, b))
                r += [spm(*d), f'{d[0] / short[0]:.0%}' if short[0] > 2 * short[1] else 'no gap']
            d = pooled(df, groups, lambda g: seeds(df, g, ORACLE, b) - seeds(df, g, SPLIT, b))
            rows.append(r + [spm(*d)])
    add(f'T{len(T)}. Prediction (a): variant minus default PDCL on the skewed streams, and the share of the gap '
        'between default PDCL and the best fixed weight that it closes (pooled rows are exploratory)',
        ['Stream', 'memory', 'best w', 'ER(best w) − PDCL', 'oracle − PDCL', 'gap closed', 'split − PDCL',
         'gap closed', 'oracle − split'], rows)

    # prediction (b)
    rows = []
    for b in BUFFERS:
        for g in ('aglen', 'rounds'):
            r = [NAME[g], b]
            for m in [DEFAULT] + VARIANTS:
                d = diff(df, g, m, 'Fine-tuning', b)
                r += [spm(*d), 'yes' if d[0] < -2 * d[1] else 'no']
            rows.append(r + [spm(*diff(df, g, 'ER w=1', 'Fine-tuning', b))])
    add(f'T{len(T)}. Prediction (b), safety: minus fine-tuning on `aglen` and `rounds`',
        ['Stream', 'memory', 'PDCL − FT', 'below by > 2 SE', 'oracle − FT', 'below by > 2 SE', 'split − FT',
         'below by > 2 SE', 'ER w=1 − FT'], rows)

    # prediction (c)
    for value, title in (('gap', 'validation loss minus loss on the replayed stored examples, earlier tasks, '
                                 'averaged over the checkpoints after each task\'s phase'),
                         ('gap_final', 'the same at the last step')):
        rows = []
        for m in METHODS[1:]:
            for b in BUFFERS:
                rows.append([f'{m}, memory {b}'] + [pm(*stat(df, g, m, b, value), 3) for g in STREAMS]
                            + [pm(*pooled(df, STREAMS, lambda g: seeds(df, g, m, b, value)), 3)])
            rows.append([f'{m}, 200 − 1000'] +
                        [spm(*ms(seeds(df, g, m, 200, value) - seeds(df, g, m, 1000, value)), 3) for g in STREAMS]
                        + [spm(*pooled(df, STREAMS, lambda g: seeds(df, g, m, 200, value)
                                       - seeds(df, g, m, 1000, value)), 3)])
        add(f'T{len(T)}. Prediction (c), gap: {title}', ['Method'] + head + ['mean over streams'], rows)
    rows = []
    for m in METHODS[1:]:
        for b in BUFFERS:
            for value in ('mem_loss', 'val_loss'):
                rows.append([f'{m}, memory {b}, {value.replace("_", " ")}'] +
                            [pm(*stat(df, g, m, b, value), 3) for g in STREAMS])
    rows.append(['Fine-tuning, val loss'] + [pm(*stat(df, g, 'Fine-tuning', 0, 'val_loss'), 3) for g in STREAMS])
    for b in BUFFERS:
        rows.append([f'{SPLIT}, memory {b}, held loss'] + [pm(*stat(df, g, SPLIT, b, 'held_loss'), 3) for g in STREAMS])
    add(f'T{len(T)}. Loss on the replayed stored examples and validation loss of earlier tasks '
        '(averaged over the checkpoints after each task\'s phase)', ['Quantity'] + head, rows)
    rows = []
    shortfall = lambda g, b: seeds(df, g, best[g, b], b) - seeds(df, g, DEFAULT, b)
    for b in BUFFERS:
        rows.append([f'ER(best w) − PDCL, memory {b}'] + [spm(*ms(shortfall(g, b))) for g in STREAMS]
                    + [spm(*pooled(df, SKEWED, lambda g: shortfall(g, b))), spm(*pooled(df, STREAMS, lambda g: shortfall(g, b)))])
    rows.append(['shortfall at 200 − shortfall at 1000'] + [spm(*ms(shortfall(g, 200) - shortfall(g, 1000))) for g in STREAMS]
                + [spm(*pooled(df, groups, lambda g: shortfall(g, 200) - shortfall(g, 1000))) for groups in (SKEWED, STREAMS)])
    add(f'T{len(T)}. Prediction (c), shortfall of default PDCL against the best fixed weight by memory size',
        ['Quantity'] + head + ['mean, skewed streams', 'mean, all streams'], rows)

    # mechanism
    rows = []
    for m in [DEFAULT] + VARIANTS:
        for b in BUFFERS:
            rows.append([f'{m}, memory {b}'] + [pm(*stat(df, g, m, b, 'dual_mean'), 3) for g in STREAMS])
    add(f'T{len(T)}. Dual variables (mean over the checkpoints after each task\'s phase and over earlier tasks)',
        ['Method'] + head, rows)
    rows = []
    for m in [DEFAULT] + VARIANTS:
        for b in BUFFERS:
            rows.append([f'{m}, memory {b}'] + [
                ' / '.join(f"{stat(df, g, m, b, f'eps_{k}')[0]:.3f}" for k in range(3)) for g in STREAMS])
    add(f'T{len(T)}. Constraint levels epsilon of tasks 0 / 1 / 2', ['Method'] + head, rows)
    rows = []
    for m in METHODS:
        for b in (BUFFERS if m != 'Fine-tuning' else [0]):
            label = m if m == 'Fine-tuning' else f'{m}, memory {b}'
            rows.append([label + ', validation'] + [f"{stat(df, g, m, b, 'viol_val')[0]:.0%}" for g in STREAMS])
            if m != 'Fine-tuning':
                rows.append([label + ', stored'] + [f"{stat(df, g, m, b, 'viol_mem')[0]:.0%}" for g in STREAMS])
    add(f'T{len(T)} (exploratory). Share of checkpoints after a task\'s phase at which its loss is above 1.1 x '
        'the loss it had when the phase ended, measured on the validation set and on the replayed stored examples',
        ['Method'] + head, rows)
    rows = []
    for b in BUFFERS:
        for m in [DEFAULT] + VARIANTS:
            rows.append([f'{m}, memory {b}'] + [spm(*diff(df, g, m, 'Fine-tuning', b, 'learned_acc')) for g in STREAMS])
    add(f'T{len(T)} (exploratory). Accuracy of each task at the end of its own phase, minus fine-tuning (paired)',
        ['Method'] + head, rows)
    rows = []
    for b in BUFFERS:
        for m in [DEFAULT] + VARIANTS:
            rows.append([f'{m} − ER w=1, memory {b}'] + [spm(*diff(df, g, m, 'ER w=1', b)) for g in STREAMS])
    add(f'T{len(T)} (exploratory). PDCL and variants minus plain ER (w = 1), final accuracy (paired)',
        ['Method'] + head, rows)
    return '\n'.join(T)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def save(fig, name, **layout):
    pa.FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(**({'pad': 0.6, 'w_pad': 1.0} | layout))
    for ext in ('pdf', 'png'):
        fig.savefig(pa.FIG / f'{name}.{ext}', dpi=300)
    plt.close(fig)


def grid(rows, cols, h, letters=True, x=-0.3):
    fig, axes = plt.subplots(rows, cols, figsize=(180 * pa.MM, h * pa.MM), squeeze=False)
    for ax, letter in zip(axes.flat, 'abcdefghijkl'):
        ax.grid(color='gray', alpha=0.7, linewidth=0.4, zorder=0)
        if letters:
            ax.text(x, 1.08, letter, transform=ax.transAxes, fontweight='bold', fontsize=9)
    return fig, axes


def fig_accuracy(df):
    """(i) Final accuracy of every method, one panel per stream and memory size."""
    fig, axes = grid(2, len(STREAMS), h=105, letters=False)
    for row, b in zip(axes, BUFFERS):
        for ax, g in zip(row, STREAMS):
            for i, m in enumerate(METHODS):
                mean, se = stat(df, g, m, b)
                ax.errorbar(i, mean, yerr=se, color=COLOR[m], marker='o', capsize=2, zorder=3,
                            markeredgecolor=NIGHT, markeredgewidth=0.4, ecolor=NIGHT, elinewidth=0.8, label=m)
            ax.axhline(stat(df, g, 'Fine-tuning', b)[0], color=ORANGE, linewidth=0.8, linestyle='--', zorder=2)
            ax.set_xticks(range(len(METHODS)))
            ax.set_xticklabels([SHORT[m] for m in METHODS], fontsize=5.5)
            ax.tick_params(axis='y', labelsize=7)
            ax.set_xlim(-0.6, len(METHODS) - 0.4)
            ax.set_title(NAME[g], fontsize=8)
        row[0].set_ylabel(f'Final Accuracy (%)\nMemory {b}')
    h, l = axes[0, 0].get_legend_handles_labels()
    label = {'Fine-tuning': 'Fine-Tuning (FT)', DEFAULT: 'PDCL (P)', ORACLE: 'PDCL Oracle (O)', SPLIT: 'PDCL Split (S)'}
    fig.legend(h, [label.get(x, x.replace('ER', 'Replay')) for x in l], loc='upper center', ncol=4,
               columnspacing=1.0, handletextpad=0.2)
    save(fig, 'fig_f1_accuracy', rect=(0, 0, 1, 0.91), w_pad=0.6)


def fig_differences(df):
    """(ii) Variant minus default PDCL, minus the best fixed weight, minus fine-tuning
    (paired by seed; bars are two standard errors)."""
    fig, axes = grid(1, 3, h=66, x=-0.22)
    refs = [('Default PDCL', lambda g, b: DEFAULT), ('Best Fixed Weight', lambda g, b: best_weight(df, g, b)),
            ('Fine-Tuning', lambda g, b: 'Fine-tuning')]
    series = [(DEFAULT, 200), (DEFAULT, 1000), (ORACLE, 200), (ORACLE, 1000), (SPLIT, 200), (SPLIT, 1000)]
    name = {DEFAULT: 'PDCL', ORACLE: 'PDCL Oracle', SPLIT: 'PDCL Split'}
    for ax, (title, ref) in zip(axes[0], refs):
        ax.axhline(0, color=NIGHT, linewidth=0.8, zorder=2)
        shown = [s for s in series if not (title == 'Default PDCL' and s[0] == DEFAULT)]
        for i, (m, b) in enumerate(shown):
            v = np.array([diff(df, g, m, ref(g, b), b) for g in STREAMS])
            x = np.arange(len(STREAMS)) + (i - (len(shown) - 1) / 2) * 0.13
            ax.errorbar(x, v[:, 0], yerr=2 * v[:, 1], color=COLOR[m], marker='o' if b == 200 else 's',
                        markerfacecolor=COLOR[m] if b == 200 else 'white', linestyle='none', capsize=1.5,
                        markeredgecolor=NIGHT if b == 200 else COLOR[m], markeredgewidth=0.6, ecolor=COLOR[m],
                        elinewidth=0.9, zorder=3, label=f'{name[m]}, Memory {b}')
        ax.set_xticks(range(len(STREAMS)))
        ax.set_xticklabels([NAME[g] for g in STREAMS], rotation=30, ha='right')
        ax.set_ylabel(f'Final Accuracy Minus\n{title} (Points)')
    h, l = axes[0, 2].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, columnspacing=1.0, handletextpad=0.2)
    save(fig, 'fig_f1_differences', rect=(0, 0, 1, 0.88), w_pad=1.6)


def fig_duals(curves, g='skew0.9'):
    """(iii) Dual variables over training, default PDCL against the variants."""
    fig, axes = grid(2, 3, h=110, x=-0.22)
    c = curves[curves['group'] == g]
    name = {DEFAULT: 'PDCL', ORACLE: 'PDCL Oracle', SPLIT: 'PDCL Split'}
    for row, b in zip(axes, BUFFERS):
        for k, ax in enumerate(row):
            for m in [DEFAULT] + VARIANTS:
                d = c[(c['method'] == m) & (c['buffer'] == b)].pivot(index='step', columns='seed', values=f'dual_{k}')
                mean, se = d.mean(1), d.std(1, ddof=1) / np.sqrt(d.shape[1])
                ax.fill_between(d.index, mean - se, mean + se, color=COLOR[m], alpha=0.3, linewidth=0, zorder=2)
                ax.plot(d.index, mean, color=COLOR[m], zorder=3, label=name[m])
            for edge in (1000, 2000, 3000):
                ax.axvline(edge, color=GREY, linewidth=0.6, linestyle=':', zorder=1)
            ax.set_title(f'Task {k + 1}, Memory {b}', fontsize=8)
            ax.set_xlabel('Training Step')
            ax.set_xticks([0, 1000, 2000, 3000, 4000])
        row[0].set_ylabel('Dual Variable')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, columnspacing=1.4)
    save(fig, 'fig_f1_duals', rect=(0, 0, 1, 0.95), w_pad=1.6)


def fig_gap(df):
    """(iv) Loss on the replayed stored examples against validation loss of earlier tasks."""
    fig, axes = grid(1, 3, h=70, x=-0.22)
    marker = dict(zip(STREAMS, 'ovs^DP'))
    hi = 0
    for ax, b in zip(axes[0], BUFFERS):
        for m in METHODS[1:]:
            for g in STREAMS:
                x, y = stat(df, g, m, b, 'mem_loss')[0], stat(df, g, m, b, 'val_loss')[0]
                hi = max(hi, x, y)
                ax.scatter(x, y, s=16, color=COLOR[m], marker=marker[g], edgecolor=NIGHT, linewidth=0.4, zorder=3)
        ax.set_xlabel('Loss on Stored Examples (Replayed)')
        ax.set_ylabel('Validation Loss, Earlier Tasks')
        ax.set_title(f'Memory {b}', fontsize=8)
    for ax in axes[0, :2]:
        ax.plot([0, hi * 1.05], [0, hi * 1.05], color=GREY, linewidth=0.8, linestyle='--', zorder=2)
        ax.set_xlim(0, hi * 1.05)
        ax.set_ylim(0, hi * 1.05)
    ax = axes[0, 2]
    for j, b in enumerate(BUFFERS):
        v = np.array([pooled(df, STREAMS, lambda g: seeds(df, g, m, b, 'gap')) for m in METHODS[1:]])
        ax.bar(np.arange(len(v)) + (j - 0.5) * 0.38, v[:, 0], yerr=v[:, 1], width=0.38,
               color=[COLOR[m] for m in METHODS[1:]], edgecolor=NIGHT, linewidth=0.5, capsize=1.5,
               hatch=None if b == 200 else '////', error_kw={'linewidth': 0.8}, zorder=3)
    ax.set_xticks(range(len(METHODS) - 1))
    ax.set_xticklabels([SHORT[m] for m in METHODS[1:]])
    ax.set_ylabel('Validation Minus Stored Loss\n(Mean Over Streams)')
    ax.set_title('Plain: Memory 200, Hatched: 1000', fontsize=8)
    handles = [plt.Line2D([], [], marker='o', linestyle='none', color=COLOR[m], markeredgecolor=NIGHT,
                          markeredgewidth=0.4, label=f'{m.replace("ER", "Replay").title().replace("Pdcl", "PDCL").replace("W=", "w=")} ({SHORT[m]})') for m in METHODS[1:]]
    handles += [plt.Line2D([], [], marker=marker[g], linestyle='none', color='white', markeredgecolor=NIGHT,
                           markeredgewidth=0.6, label=NAME[g]) for g in STREAMS]
    fig.legend(handles=handles, loc='upper center', ncol=7, columnspacing=1.0, handletextpad=0.2)
    save(fig, 'fig_f1_gap', rect=(0, 0, 1, 0.88), w_pad=1.6)


def main():
    if '--cached' in sys.argv:
        df, curves = pd.read_csv('results/f1_runs.csv'), pd.read_csv('results/f1_curves.csv')
    else:
        df, curves = load()
        df.to_csv('results/f1_runs.csv', index=False)
        curves.to_csv('results/f1_curves.csv', index=False)
    Path('results/f1_tables.md').write_text(tables(df))
    pa.style()
    fig_accuracy(df)
    fig_differences(df)
    fig_duals(curves)
    fig_gap(df)


if __name__ == '__main__':
    main()
