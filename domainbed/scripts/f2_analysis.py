"""F2, "Is replay's harm memorization?": per-run table, tables and figures from the raw runs.

    python -m domainbed.scripts.f2_analysis

Reads every finished run under outputs/f2 (results.jsonl); writes
results/f2_runs.csv (one row per run), results/f2_tables.md and
paper/figures/fig_f2_{batch,revisits,loss,decay}.{pdf,png}.

Per run, besides the columns of stream_analysis.py:

    memory, replay_batch, weight_decay   the setting (replay_batch 32 = trainer default)
    val_loss_prev, train_loss_prev       loss on the validation / training split of the
                                         three earlier tasks at the last step, averaged
    mem_loss_prev, mem_acc_prev          loss / accuracy on all stored examples of the
                                         three earlier tasks at the last step, averaged
    gap_prev                             val_loss_prev - mem_loss_prev
    *_after                              the same, averaged over every (checkpoint, task)
                                         pair after the task's own phase
    replays_task0                        times one stored example of task 0 is replayed if
                                         it stays in memory to the end of the stream
    replays_per_1000                     replays of a stored example per 1,000 steps,
                                         averaged over the stored examples of all phases
    replayed_per_step                    stored examples in a step's batch, mean over the
                                         steps after the first phase

The replay counts are computed, not measured: in a phase with n_k stored examples of
task k, each step replays min(replay_batch, n_k) of them, drawn without replacement.
All differences are paired by seed; "a ± b" is mean ± standard error over seeds; bold
= larger than two standard errors.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from domainbed import streams
from domainbed.scripts import paper_assets as pa
from domainbed.scripts import stream_analysis as sa
from domainbed.scripts.paper_assets import CYAN, GREY, NIGHT, ORANGE, RED, plt

ROOT = Path('outputs/f2')
STREAMS = ['aglen', 'rounds', 'skew0.9']
NAME = {'aglen': 'aglen', 'rounds': 'rounds', 'skew0.9': 'skew:0.9'}
MEMORIES, BATCHES, DECAYS = [50, 200], [32, 8, 2], [0.01, 0.1]
PHASE, N_TASKS, BS = 1000, 4, 32
FT = {0: 'Finetune', 0.01: 'Finetune__wd0.01', 0.1: 'Finetune__wd0.1'}
er_rb = lambda m, rb: f'ER__m{m}_rb{rb}'
er_wd = lambda m, wd: er_rb(m, 32) if wd == 0 else f'ER__m{m}_wd{wd}'
# (label, method, fine-tuning control with the same weight decay)
ER_SETTINGS = [(f'ER, memory {m}, replay batch {rb}', er_rb(m, rb), FT[0]) for m in MEMORIES for rb in BATCHES] + \
              [(f'ER, memory {m}, replay batch 32, weight decay {wd}', er_wd(m, wd), FT[wd])
               for m in MEMORIES for wd in DECAYS]
FT_SETTINGS = [('Fine-tuning', FT[0])] + [(f'Fine-tuning, weight decay {wd}', FT[wd]) for wd in DECAYS]
SETTINGS = FT_SETTINGS + [(lab, m) for lab, m, _ in ER_SETTINGS]


# --------------------------------------------------------------------------
# raw runs -> one row per run
# --------------------------------------------------------------------------
def extras(recs):
    last, hp = recs[-1], recs[-1]['hparams']
    prev = range(N_TASKS - 1)
    replay = recs[-1]['args']['algorithm'] != 'Finetune'
    rb = min(hp.get('replay_batch', 0) or hp['batch_size'], hp['batch_size'])
    row = {'memory': hp['buffer_size'] if replay else 0, 'replay_batch': rb if replay else 0,
           'weight_decay': hp['weight_decay'],
           'val_loss': float(np.mean([last[f'env{k}_out_loss'] for k in range(N_TASKS)])),
           'val_loss_prev': float(np.mean([last[f'env{k}_out_loss'] for k in prev])),
           'train_loss_prev': float(np.mean([last[f'env{k}_in_loss'] for k in prev]))}
    after = [(r, k) for r in recs for k in prev if r['step'] >= (k + 1) * PHASE]
    row['val_loss_after'] = float(np.mean([r[f'env{k}_out_loss'] for r, k in after]))
    if replay:
        row['mem_loss_prev'] = float(np.mean([last[f'mem_loss_{k}'] for k in prev]))
        row['mem_acc_prev'] = 100 * float(np.mean([last[f'mem_acc_{k}'] for k in prev]))
        row['mem_loss_after'] = float(np.mean([r[f'mem_loss_{k}'] for r, k in after]))
        row['gap_prev'] = row['val_loss_prev'] - row['mem_loss_prev']
        row['gap_after'] = row['val_loss_after'] - row['mem_loss_after']
        by_step = {r['step']: r for r in recs}
        draws = stored = task0 = 0.0
        for p in range(1, N_TASKS):
            n = [by_step[p * PHASE][f'buf_{k}'] for k in range(p)]
            assert min(n) > 0
            draws += PHASE * sum(min(rb, x) for x in n)
            stored += sum(n)
            task0 += PHASE * min(rb, n[0]) / n[0]
            row[f'stored_0_phase{p}'] = n[0]
        row['replays_task0'] = task0
        row['replays_per_1000'] = draws / stored
        row['replayed_per_step'] = draws / (PHASE * (N_TASKS - 1))
    return row


def load():
    rows, curves = [], []
    for done in sorted(ROOT.rglob('done')):
        d = done.parent
        recs = [json.loads(l) for l in open(d / 'results.jsonl')]
        key = {'group': d.parent.parent.name, 'method': d.parent.name}
        rows.append(key | sa.run_row(recs) | extras(recs))
        for r in recs:
            curves.append(key | {'seed': r['args']['seed'], 'step': r['step'],
                                 'val0': r['env0_out_loss'], 'mem0': r.get('mem_loss_0', np.nan)})
    return pd.DataFrame(rows), pd.DataFrame(curves)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def seeds(df, group, method, value='final_acc'):
    d = df[(df['group'] == group) & (df['method'] == method)]
    assert len(d) == 10, (group, method, len(d))
    return d.set_index('seed')[value].sort_index()


def ms(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def stat(df, group, method, value='final_acc'):
    return ms(seeds(df, group, method, value))


def diff(df, group, a, b, value='final_acc'):
    return ms(seeds(df, group, a, value) - seeds(df, group, b, value))


def pm(m, s, nd=2):
    return f'{m:.{nd}f} ± {s:.{nd}f}'


def spm(m, s, nd=2):
    t = f'{m:+.{nd}f} ± {s:.{nd}f}'
    return f'**{t}**' if abs(m) > 2 * s else t


def md(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]) + '\n'


def ratio_interval(num, den, n_boot=10000):
    """Seed bootstrap (2.5%, 97.5%) of mean(num) / mean(den), seeds resampled jointly."""
    rng = np.random.RandomState(0)
    i = rng.randint(0, len(num), (n_boot, len(num)))
    r = num.values[i].mean(1) / den.values[i].mean(1)
    return np.percentile(r, [2.5, 97.5])


def visits_of_a_fresh_example(data_dir='data'):
    """Times a training example of the current task is drawn during its own phase."""
    out = {}
    for g in STREAMS:
        sizes = (torch.load(Path(data_dir) / 'abrosetta_tokens.pt')['sizes'] if g == 'rounds'
                 else streams.task_order(data_dir, NAME[g])[1])
        out[g] = [PHASE * BS / (n - int(n * 0.2)) for n in sizes]
    return out


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def tables(df):
    T = []
    add = lambda title, header, rows, note='': T.append(f'### {title}\n\n' + md(header, rows) + (f'\n{note}\n' if note else ''))
    head = [NAME[g] for g in STREAMS]

    for value, title in (('final_acc', 'T1. Final accuracy (%)'), ('forgetting', 'T2. Forgetting (points)'),
                         ('learned_acc', 'T3. Accuracy of each task at the end of its own phase (%)')):
        add(title, ['Setting'] + head, [[lab] + [pm(*stat(df, g, m, value)) for g in STREAMS] for lab, m in SETTINGS])

    for value, title in (('final_acc', 'T4. ER minus fine-tuning with the same weight decay, final accuracy (paired)'),
                         ('forgetting', 'T5. ER minus fine-tuning with the same weight decay, forgetting '
                                        '(paired; negative = ER forgets less)'),
                         ('learned_acc', 'T6. ER minus fine-tuning with the same weight decay, accuracy of each '
                                         'task at the end of its own phase (paired)')):
        add(title, ['Setting'] + head,
            [[lab] + [spm(*diff(df, g, m, c, value)) for g in STREAMS] for lab, m, c in ER_SETTINGS])

    # the pre-registered test
    rows, verdict = [], {}
    for g in STREAMS:
        for m in MEMORIES:
            d32 = seeds(df, g, er_rb(m, 32)) - seeds(df, g, FT[0])
            for rb in (8, 2):
                d = seeds(df, g, er_rb(m, rb)) - seeds(df, g, FT[0])
                frac = d.mean() / d32.mean()
                lo, hi = ratio_interval(d, d32)
                real = abs(d32.mean()) > 2 * ms(d32)[1]
                rows.append([NAME[g], m, rb, spm(*ms(d32)), spm(*ms(d)), spm(*ms(d - d32)),
                             f'{frac:.2f} ({lo:.2f} to {hi:.2f})' if real else 'n/a (no effect at 32)'])
                verdict[g, m, rb] = (ms(d32), ms(d), ms(d - d32), frac, real)
    add('T7. The pre-registered test: ER minus fine-tuning at replay batch 32 and at a smaller replay batch',
        ['Stream', 'Memory', 'Replay batch', 'ER(32) − fine-tuning', 'ER(batch) − fine-tuning',
         'ER(batch) − ER(32)', 'share of the ER(32) effect that remains'], rows,
        'Share = ratio of the two mean differences; in brackets a 95% seed-bootstrap interval (exploratory). '
        'Prediction: on `aglen` the share at replay batch 2 is at most 0.5 at both memory sizes (at least half of '
        "ER's loss removed); on `skew:0.9` it is at least 0.5 (at least half of ER's gain kept).")

    rows = []
    for g in STREAMS:
        for wd in DECAYS:
            rows.append([NAME[g], wd, spm(*diff(df, g, FT[wd], FT[0]))] +
                        [x for m in MEMORIES for x in (spm(*diff(df, g, er_wd(m, wd), er_wd(m, 0))),
                                                       spm(*diff(df, g, er_wd(m, 0), FT[0])),
                                                       spm(*diff(df, g, er_wd(m, wd), FT[wd])),
                                                       spm(*ms(seeds(df, g, er_wd(m, wd)) - seeds(df, g, FT[wd])
                                                               - seeds(df, g, er_wd(m, 0)) + seeds(df, g, FT[0]))))])
    add('T8. Weight decay (replay batch 32), final accuracy, paired',
        ['Stream', 'Weight decay', 'fine-tuning: decay − none'] +
        [f'{x}, memory {m}' for m in MEMORIES for x in ('ER: decay − none', 'ER − fine-tuning, no decay',
                                                         'ER − fine-tuning, both with decay',
                                                         'change of ER − fine-tuning with decay')], rows)

    for when, title in (('prev', 'T9. Loss on the stored examples against validation loss of the three earlier '
                                 'tasks, at the last step'),
                        ('after', 'T10. The same, averaged over all checkpoints after each task\'s own phase')):
        rows = []
        for g in STREAMS:
            for lab, m in SETTINGS:
                er = m.startswith('ER')
                rows.append([NAME[g], lab,
                             pm(*stat(df, g, m, f'mem_loss_{when}'), 3) if er else '–',
                             pm(*stat(df, g, m, f'val_loss_{when}'), 3),
                             pm(*stat(df, g, m, f'gap_{when}'), 3) if er else '–']
                            + ([pm(*stat(df, g, m, 'mem_acc_prev'), 1) if er else '–',
                                pm(*stat(df, g, m, 'train_loss_prev'), 3)] if when == 'prev' else []))
        add(title, ['Stream', 'Setting', 'memory loss', 'validation loss', 'gap (validation − memory)']
            + (['memory accuracy (%)', 'loss on all training examples of the earlier tasks'] if when == 'prev' else []),
            rows)

    rows = []
    for m in MEMORIES:
        for rb in BATCHES:
            r = [m, rb]
            for col, nd in (('stored_0_phase1', 0), ('stored_0_phase2', 0), ('stored_0_phase3', 0),
                            ('replayed_per_step', 1), ('replays_per_1000', 0), ('replays_task0', 0)):
                v = np.array([stat(df, g, er_rb(m, rb), col)[0] for g in STREAMS])
                assert np.allclose(v, v[0]), (m, rb, col, v)        # the same on every stream and seed
                assert all(stat(df, g, er_rb(m, rb), col)[1] < 1e-9 for g in STREAMS)
                r.append(f'{v[0]:.{nd}f}')
            rows.append(r)
    fresh = visits_of_a_fresh_example()
    add('T11. How often a stored example is replayed (computed from the memory sizes in the logs)',
        ['Memory', 'Replay batch'] + [f'stored examples of task 0, phase {p}' for p in (2, 3, 4)] +
        ['stored examples in a step\'s batch (mean)', 'replays of a stored example per 1,000 steps',
         'replays of a task-0 example kept to the end'], rows,
        'Weight decay does not change these counts (same as replay batch 32). For comparison, a training example '
        'of the current task is drawn ' + ', '.join(
            (f'{min(v):.0f} to {max(v):.0f}' if round(min(v)) != round(max(v)) else f'{min(v):.0f}')
            + f' times on `{NAME[g]}`' for g, v in fresh.items())
        + ' during its own 1,000-step phase.')

    rows = []
    for g in STREAMS:
        for lab, m in SETTINGS:
            rows.append([NAME[g], lab] + [pm(*stat(df, g, m, f'acc_{k}'), 1) for k in range(N_TASKS)])
    add('T12. Final validation accuracy per task (%)', ['Stream', 'Setting'] + [f'task {k}' for k in range(N_TASKS)], rows)

    # exploratory: dose-response of ER - fine-tuning on the computed number of replays
    rows = []
    for g in STREAMS:
        pts = [(stat(df, g, er_rb(m, rb), 'replays_per_1000')[0], diff(df, g, er_rb(m, rb), FT[0])[0],
                diff(df, g, er_rb(m, rb), FT[0], 'forgetting')[0], stat(df, g, er_rb(m, rb), 'gap_after')[0])
               for m in MEMORIES for rb in BATCHES]
        x = np.log([p[0] for p in pts])
        rows.append([NAME[g]] + [f'{np.corrcoef(x, [p[i] for p in pts])[0, 1]:+.2f}' for i in (1, 2, 3)])
    add('T13 (exploratory). Across the six settings (memory x replay batch): correlation of log replays per '
        'stored example with the mean effect', ['Stream', 'ER − fine-tuning, final accuracy',
                                                'ER − fine-tuning, forgetting', 'gap validation − memory loss'], rows,
        'Six points per stream, means over ten seeds; a description, not a test.')
    return '\n'.join(T), verdict


def verdict_text(v):
    """The pre-registered prediction and failure criterion, evaluated."""
    out = ['### Pre-registered outcome (computed)\n']
    held = []
    for m in MEMORIES:
        (m32, s32), (m2, s2), (dm, ds), frac, real = v['aglen', m, 2]
        ok = real and m32 < 0 and frac <= 0.5
        held.append(ok)
        out.append(f'- `aglen`, memory {m}: ER(32) − fine-tuning {m32:+.2f} ± {s32:.2f}, ER(2) − fine-tuning '
                   f'{m2:+.2f} ± {s2:.2f}, ER(2) − ER(32) {dm:+.2f} ± {ds:.2f}; share of the loss that remains '
                   + (f'{frac:.2f}' if real else 'undefined (no loss at 32 above two standard errors)')
                   + f' -> "removes at least half" **{"held" if ok else "failed"}**.')
    for m in MEMORIES:
        (m32, s32), (m2, s2), (dm, ds), frac, real = v['skew0.9', m, 2]
        ok = real and m32 > 0 and frac >= 0.5
        held.append(ok)
        out.append(f'- `skew:0.9`, memory {m}: ER(32) − fine-tuning {m32:+.2f} ± {s32:.2f}, ER(2) − fine-tuning '
                   f'{m2:+.2f} ± {s2:.2f}, ER(2) − ER(32) {dm:+.2f} ± {ds:.2f}; share of the gain kept '
                   + (f'{frac:.2f}' if real else 'undefined') + f' -> "keeps at least half" **{"held" if ok else "failed"}**.')
    shrink = []
    for m in MEMORIES:
        l = [v['aglen', m, rb][1][0] for rb in (8, 2)]
        l32 = v['aglen', m, 2][0][0]
        dm, ds = v['aglen', m, 2][2]
        shrink.append(dm > 2 * ds)
        out.append(f'- Failure criterion, `aglen`, memory {m}: ER − fine-tuning is {l32:+.2f}, {l[0]:+.2f}, {l[1]:+.2f} '
                   f'at replay batch 32, 8, 2; ER(2) − ER(32) = {dm:+.2f} ± {ds:.2f} -> the loss '
                   f'**{"shrinks" if dm > 2 * ds else "does not shrink"}** (two standard errors).')
    out.append(f'\nPrediction as a whole (all four parts): **{"held" if all(held) else "failed"}**. '
               f'Failure criterion (loss on `aglen` does not shrink): '
               f'**{"not met at either memory size" if all(shrink) else "met at both memory sizes" if not any(shrink) else "met at one memory size"}**.\n')
    return '\n'.join(out)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
MEM_COLOR = {50: RED, 200: CYAN}
RB_COLOR = {32: NIGHT, 8: RED, 2: CYAN}


def save(fig, name, **layout):
    fig.tight_layout(**({'pad': 0.6, 'w_pad': 1.5} | layout))
    for ext in ('pdf', 'png'):
        fig.savefig(pa.FIG / f'{name}.{ext}', dpi=300)
    plt.close(fig)


def grid(rows, cols, h=62):
    fig, axes = plt.subplots(rows, cols, figsize=(180 * pa.MM, h * pa.MM), squeeze=False)
    for ax, letter in zip(axes.flat, 'abcdef'):
        ax.grid(color='gray', alpha=0.7, linewidth=0.4, zorder=0)
        ax.text(-0.22, 1.08, letter, transform=ax.transAxes, fontweight='bold', fontsize=9)
    return fig, axes


def legend_top(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=ncol, columnspacing=1.4, handlelength=1.8)


def fig_batch(df):
    """ER minus fine-tuning against the replay batch: final accuracy (top), forgetting (bottom)."""
    fig, axes = grid(2, 3, h=110)
    for i, (value, ylabel) in enumerate((('final_acc', 'Replay − Fine-Tuning,\nFinal Accuracy (points)'),
                                         ('forgetting', 'Replay − Fine-Tuning,\nForgetting (points)'))):
        for ax, g in zip(axes[i], STREAMS):
            ax.axhline(0, color=ORANGE, linewidth=1.1, zorder=2, label='Fine-Tuning')
            for m, off in zip(MEMORIES, (0.93, 1.07)):
                v = np.array([diff(df, g, er_rb(m, rb), FT[0], value) for rb in BATCHES])
                ax.errorbar(np.array(BATCHES) * off, v[:, 0], yerr=v[:, 1], color=MEM_COLOR[m], marker='o',
                            capsize=2, zorder=3, markeredgecolor=NIGHT, markeredgewidth=0.4, label=f'Replay, Memory {m}')
            ax.set_xscale('log', base=2)
            ax.set_xticks(BATCHES)
            ax.set_xticklabels([str(b) for b in BATCHES])
            ax.set_xlim(1.5, 43)
            ax.minorticks_off()
            ax.set_xlabel('Replay Batch per Earlier Task')
            ax.set_title(NAME[g], fontsize=8)
        axes[i][0].set_ylabel(ylabel)
    legend_top(fig, axes[0][0], 3)
    save(fig, 'fig_f2_batch', rect=(0, 0, 1, 0.95), h_pad=1.5)


def fig_revisits(df):
    """Exploratory: the same differences against the computed number of replays per stored example."""
    fig, axes = grid(1, 3)
    for ax, g in zip(axes[0], STREAMS):
        ax.axhline(0, color=ORANGE, linewidth=1.1, zorder=2, label='Fine-Tuning')
        for m in MEMORIES:
            x = [stat(df, g, er_rb(m, rb), 'replays_per_1000')[0] for rb in BATCHES]
            v = np.array([diff(df, g, er_rb(m, rb), FT[0]) for rb in BATCHES])
            ax.errorbar(x, v[:, 0], yerr=v[:, 1], color=MEM_COLOR[m], marker='o', capsize=2, zorder=3,
                        markeredgecolor=NIGHT, markeredgewidth=0.4, label=f'Replay, Memory {m}')
            for xi, vi, rb in zip(x, v[:, 0], BATCHES):
                ax.annotate(str(rb), (xi, vi), textcoords='offset points', xytext=(4, 4), fontsize=6, color=GREY)
        ax.set_xscale('log')
        ax.set_xlabel('Replays of a Stored Example\nper 1,000 Steps')
        ax.set_title(NAME[g], fontsize=8)
    axes[0][0].set_ylabel('Replay − Fine-Tuning,\nFinal Accuracy (points)')
    legend_top(fig, axes[0][0], 3)
    save(fig, 'fig_f2_revisits', rect=(0, 0, 1, 0.92))


def fig_loss(df, curves):
    """Task 0: loss on its stored examples (dashed) and its validation loss (solid) along the stream."""
    fig, axes = grid(2, 3, h=110)
    c = curves[curves['step'] >= PHASE]
    for i, m in enumerate(MEMORIES):
        for ax, g in zip(axes[i], STREAMS):
            ft = c[(c['group'] == g) & (c['method'] == FT[0])].groupby('step')['val0'].mean()
            ax.plot(ft.index / 1000, ft.values, color=ORANGE, zorder=3, label='Fine-Tuning, Validation')
            for rb in BATCHES:
                d = c[(c['group'] == g) & (c['method'] == er_rb(m, rb))].groupby('step')[['val0', 'mem0']].mean()
                ax.plot(d.index / 1000, d['val0'], color=RB_COLOR[rb], zorder=3, label=f'Replay Batch {rb}, Validation')
                ax.plot(d.index / 1000, d['mem0'], color=RB_COLOR[rb], linestyle='--', zorder=3,
                        label=f'Replay Batch {rb}, Stored Examples')
            ax.set_xlabel('Training Steps (Thousands)')
            ax.set_title(f'{NAME[g]}, Memory {m}', fontsize=8)
        axes[i][0].set_ylabel('Loss on the First Task')
    legend_top(fig, axes[0][0], 4)
    save(fig, 'fig_f2_loss', rect=(0, 0, 1, 0.92), h_pad=1.5)


def fig_decay(df):
    """ER minus fine-tuning with the same weight decay."""
    fig, axes = grid(1, 3)
    wds = [0] + DECAYS
    for ax, g in zip(axes[0], STREAMS):
        ax.axhline(0, color=ORANGE, linewidth=1.1, zorder=2, label='Fine-Tuning, Same Weight Decay')
        for m, off in zip(MEMORIES, (-0.06, 0.06)):
            v = np.array([diff(df, g, er_wd(m, wd), FT[wd]) for wd in wds])
            ax.errorbar(np.arange(3) + off, v[:, 0], yerr=v[:, 1], color=MEM_COLOR[m], marker='o', capsize=2,
                        zorder=3, markeredgecolor=NIGHT, markeredgewidth=0.4, label=f'Replay, Memory {m}')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['0', '0.01', '0.1'])
        ax.set_xlim(-0.4, 2.4)
        ax.set_xlabel('Weight Decay')
        ax.set_title(NAME[g], fontsize=8)
    axes[0][0].set_ylabel('Replay − Fine-Tuning,\nFinal Accuracy (points)')
    legend_top(fig, axes[0][0], 3)
    save(fig, 'fig_f2_decay', rect=(0, 0, 1, 0.92))


def main():
    df, curves = load()
    n = df.groupby(['group', 'method']).size()
    assert len(n) == len(STREAMS) * len(SETTINGS) and (n == 10).all(), n
    df.to_csv('results/f2_runs.csv', index=False)
    text, verdict = tables(df)
    Path('results/f2_tables.md').write_text(
        '# F2 tables\n\nWritten by `python -m domainbed.scripts.f2_analysis` from the raw runs under `outputs/f2` '
        f'({len(df)} runs, ten seeds per setting). Mean ± standard error over seeds; differences are paired by '
        'seed; bold = larger than two standard errors.\n\n' + verdict_text(verdict) + '\n' + text)
    pa.style()
    fig_batch(df)
    fig_revisits(df)
    fig_loss(df, curves)
    fig_decay(df)
    print(verdict_text(verdict))


if __name__ == '__main__':
    main()
