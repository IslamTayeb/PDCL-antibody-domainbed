"""H1, long training: tables and figure from outputs/h1 and the first round's runs.

    python -m domainbed.scripts.h1_analysis [--partial]

Reads outputs/h1 (rounds, rounds:3210, aglen, target; joint ERM, fine-tuning,
PDCL, ER with replay weight 0.03, 0.3, 1; 5,000 steps per task, seeds 0-9)
through stream_analysis.table and writes results/h1_runs.csv, and the
validation accuracy at every checkpoint to results/h1_curves.csv. If outputs/h1
is absent, both are read back from results/.

The 1,000-steps-per-task numbers come from the first round, same seeds:
  results/e1_runs_main.csv    joint ERM, fine-tuning, PDCL, ER (weight 1), all four
                              streams (results/e1_runs.csv of branch
                              claude/pdcl-e1-retention-dial-6f00e4, copied unchanged)
  results/e2_runs_stream.csv  ER with weight 0.03 and 0.3, on rounds and aglen only
ER with weight 0.03 or 0.3 was never run at 1,000 steps per task on rounds:3210
or target, so those cells are empty.

Writes results/h1_tables.md and paper/figures/fig_h1_long.{pdf,png}.
Final accuracy = validation accuracy at the last step, averaged over the four
tasks. All differences are paired by seed, mean ± standard error.
--partial keeps only the streams whose 60 runs are all finished (for a look
while training); without it the script refuses an incomplete set.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from domainbed.scripts import paper_assets as pa
from domainbed.scripts.stream_analysis import paired, table

ROOT = Path("outputs/h1")
GROUPS = {"rounds": "rounds", "rounds3210": "rounds:3210", "aglen": "aglen", "target": "target"}
METHODS = ["ERM", "Finetune", "ER__w0.03", "ER__w0.3", "ER__w1", "PDCL_RS"]
MEMORY = METHODS[2:]
ER = METHODS[2:5]
LABEL = {"ERM": "Joint ERM", "Finetune": "Fine-tuning", "ER__w0.03": "ER, w = 0.03", "ER__w0.3": "ER, w = 0.3",
         "ER__w1": "ER, w = 1", "PDCL_RS": "PDCL"}
SHORT = {"ERM": "Joint", "Finetune": "FT", "ER__w0.03": "ER 0.03", "ER__w0.3": "ER 0.3", "ER__w1": "ER 1",
         "PDCL_RS": "PDCL"}
COLOR = {"ERM": pa.NIGHT, "Finetune": pa.ORANGE, "ER__w0.03": "#B9D6D5", "ER__w0.3": pa.CYAN, "ER__w1": "#4E8785",
         "PDCL_RS": pa.RED}
E1, E2 = "results/e1_runs_main.csv", "results/e2_runs_stream.csv"
PHASE = 5000


def pm(m, s, nd=2, sign=False):
    return f"{m:{'+' if sign else ''}.{nd}f} ± {s:.{nd}f}"


def bold(d, se, text):
    return f"**{text}**" if abs(d) > 2 * se else text


def dpm(d, se):
    return bold(d, se, pm(d, se, 2, True))


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def stat(df, group, method, col="final_acc"):
    v = df[(df.group == group) & (df.method == method)][col]
    return v.mean(), v.sem()


def curves(root):
    """Validation accuracy (mean of the four tasks, percent) of every finished run at every checkpoint."""
    rows = []
    for done in sorted(Path(root).rglob("done")):
        d = done.parent
        for line in open(d / "results.jsonl"):
            r = json.loads(line)
            rows.append({"group": d.parent.parent.name, "method": d.parent.name, "seed": r["args"]["seed"],
                         "step": r["step"], "val_acc": 100 * np.mean([r[f"env{e}_out_acc"] for e in range(4)])})
    return pd.DataFrame(rows)


def first_round():
    """Per-run table at 1,000 steps per task, with the file each row comes from."""
    e1 = pd.read_csv(E1)
    e1 = e1[e1.group.isin(GROUPS) & e1.method.isin(["ERM", "Finetune", "ER", "PDCL_RS"])].copy()
    e1["method"] = e1.method.replace({"ER": "ER__w1"})
    e1["source"] = E1
    e2 = pd.read_csv(E2)
    e2 = e2[e2.group.isin(GROUPS) & e2.method.isin(["ER__w0.03", "ER__w0.3"])].copy()
    e2["source"] = E2
    old = pd.concat([e1, e2], ignore_index=True)
    assert (old.steps == 4000).all() and (old.groupby(["group", "method"]).size() == 10).all()
    return old


def best_er(df, g):
    """The replay weight with the highest mean final accuracy on the stream (picked on the same seeds)."""
    return max(ER, key=lambda m: stat(df, g, m)[0])


def table_final(df, groups):
    rows = [[GROUPS[g], LABEL[m], pm(*stat(df, g, m)), pm(*stat(df, g, m, "forgetting")),
             pm(*stat(df, g, m, "learned_acc"))] for g in groups for m in METHODS]
    return md(["Stream", "Method", "Final accuracy", "Forgetting", "Learned accuracy"], rows)


def table_vs_ft(df, groups):
    cols = MEMORY + ["ERM"]
    d = {m: paired(df, m, "Finetune") for m in cols}
    rows = [[GROUPS[g], pm(*stat(df, g, "Finetune"))] + [dpm(*d[m].loc[g, ["diff", "se"]]) for m in cols]
            for g in groups]
    return md(["Stream", "FT, final accuracy"] + [f"{SHORT[m]} − FT" for m in cols], rows)


def table_vs_er(df, groups):
    d = {m: paired(df, "PDCL_RS", m) for m in ER}
    rows = []
    for g in groups:
        b = best_er(df, g)
        rows.append([GROUPS[g]] + [dpm(*d[m].loc[g, ["diff", "se"]]) for m in ER]
                    + [SHORT[b].replace("ER ", "w = "), dpm(*d[b].loc[g, ["diff", "se"]])])
    return md(["Stream"] + [f"PDCL − {SHORT[m]}" for m in ER] + ["Best weight", "PDCL − best weight"], rows)


def table_length(df, old, groups):
    """Each method at 5,000 minus the same method at 1,000 steps per task, same seeds."""
    rows = []
    for g in groups:
        for m in METHODS:
            o = old[(old.group == g) & (old.method == m)].set_index("seed").final_acc
            n = df[(df.group == g) & (df.method == m)].set_index("seed").final_acc
            if o.empty:
                rows.append([GROUPS[g], LABEL[m], "not run", pm(n.mean(), n.sem()), "", ""])
                continue
            d = (n - o).dropna()
            assert len(d) == 10
            src = old[(old.group == g) & (old.method == m)].source.iloc[0]
            rows.append([GROUPS[g], LABEL[m], pm(o.mean(), o.sem()), pm(n.mean(), n.sem()),
                         dpm(d.mean(), d.sem()), f"`{src}`"])
    return md(["Stream", "Method", "1,000 steps per task", "5,000 steps per task", "5,000 − 1,000",
               "Source of the 1,000 column"], rows)


def table_gain_length(df, old, groups):
    """Gain over fine-tuning at both lengths."""
    rows = []
    for g in groups:
        for m in MEMORY:
            new_d = paired(df, m, "Finetune").loc[g]
            has = not old[(old.group == g) & (old.method == m)].empty
            old_d = paired(old, m, "Finetune").loc[g] if has else None
            rows.append([GROUPS[g], LABEL[m], dpm(old_d["diff"], old_d["se"]) if has else "not run",
                         dpm(new_d["diff"], new_d["se"])])
    return md(["Stream", "Method", "Method − FT at 1,000", "Method − FT at 5,000"], rows)


def table_peak(cv, groups):
    """Is the last step the best one? Seed-mean curve: its highest checkpoint against its last."""
    mean = cv.groupby(["group", "method", "step"]).val_acc.mean()
    rows = []
    for g in groups:
        for m in METHODS:
            c = mean.loc[g, m]
            wide = cv[(cv.group == g) & (cv.method == m)].pivot(index="seed", columns="step", values="val_acc")
            d = wide[c.index[-1]] - wide[c.idxmax()]
            rows.append([GROUPS[g], LABEL[m], f"{c.max():.2f}", f"{c.idxmax() + c.idxmax() % 2:,}",
                         f"{c.iloc[-1]:.2f}", dpm(d.mean(), d.sem()) if c.idxmax() != c.index[-1] else "last is best"])
    return md(["Stream", "Method", "Highest checkpoint", "at step", "Last step", "Last − highest (paired)"], rows)


def figure(df, old, cv, groups):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    pa.style()
    fig, axes = plt.subplots(3, len(groups), figsize=(180 * pa.MM, 165 * pa.MM), squeeze=False)
    mean = cv.groupby(["group", "method", "step"]).val_acc.mean()
    x = np.arange(len(METHODS))
    for j, g in enumerate(groups):
        a, b, c = axes[:, j]
        for ax in (a, b, c):
            ax.grid(color="gray", alpha=0.7, linewidth=0.4, zorder=0)
        a.set_title(GROUPS[g], fontsize=8)
        # a: validation accuracy at every checkpoint
        for m in METHODS:
            v = mean.loc[g, m]
            a.plot((v.index + 1) / 1000, v.values, color=COLOR[m], zorder=3, linewidth=1.1)
        for k in (1, 2, 3):
            a.axvline(k * PHASE / 1000, color=pa.GREY, linestyle=":", linewidth=0.7, zorder=2)
        a.set_xlim(0, 20)
        a.set_ylim(50, 72)
        a.set_xticks([0, 5, 10, 15, 20])
        a.set_xlabel("Training Step (thousands)")
        # b: final accuracy at both lengths
        for i, m in enumerate(METHODS):
            mu, se = stat(df, g, m)
            b.errorbar(i + 0.15, mu, yerr=se, color=COLOR[m], marker="o", capsize=1.5, elinewidth=0.7, zorder=3)
            if not old[(old.group == g) & (old.method == m)].empty:
                mu, se = stat(old, g, m)
                b.errorbar(i - 0.15, mu, yerr=se, color=COLOR[m], marker="o", markerfacecolor="white", capsize=1.5,
                           elinewidth=0.7, zorder=3)
        b.set_ylim(60, 71.5)
        # c: gain over fine-tuning at 5,000 steps per task, paired
        c.axhline(0, color=pa.GREY, linewidth=0.6, zorder=2)
        for i, m in enumerate(METHODS):
            if m == "Finetune":
                continue
            d, se = paired(df, m, "Finetune").loc[g, ["diff", "se"]]
            c.bar(i, d, yerr=2 * se, color=COLOR[m], width=0.65, zorder=3,
                  error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": pa.NIGHT})
        c.set_ylim(-3, 8)
        for ax in (b, c):
            ax.set_xticks(x)
            ax.set_xticklabels([SHORT[m] for m in METHODS], rotation=45, ha="right", rotation_mode="anchor")
            ax.set_xlim(-0.6, len(METHODS) - 0.4)
    axes[0, 0].set_ylabel("Validation Accuracy (%),\nmean of the four tasks")
    axes[1, 0].set_ylabel("Final Accuracy (%)")
    axes[2, 0].set_ylabel("Final Accuracy Gain over\nFine-tuning (points)")
    for row, letter in zip(axes, "abc"):
        row[0].text(-0.42, 1.08, letter, transform=row[0].transAxes, fontweight="bold", fontsize=9)
    fig.legend(handles=[Line2D([], [], color=COLOR[m], label=LABEL[m]) for m in METHODS],
               loc="upper center", ncol=6, columnspacing=1.2, handlelength=1.8, bbox_to_anchor=(0.5, 1.0))
    fig.legend(handles=[
        Line2D([], [], color=pa.GREY, marker="o", markerfacecolor="white", linestyle="",
               label="1,000 steps per task (panel b)"),
        Line2D([], [], color=pa.GREY, marker="o", linestyle="", label="5,000 steps per task (panel b)")],
        loc="upper center", ncol=2, columnspacing=1.2, handletextpad=0.2, bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(pad=0.6, w_pad=1.2, h_pad=1.4, rect=(0, 0, 1, 0.94))
    pa.FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(pa.FIG / "fig_h1_long.png", dpi=300)
    fig.savefig(pa.FIG / "fig_h1_long.pdf", dpi=600)
    plt.close(fig)


def main():
    partial = "--partial" in sys.argv
    if ROOT.exists():
        df, cv = table(ROOT), curves(ROOT)
        if not partial:
            df.to_csv("results/h1_runs.csv", index=False)
            cv.groupby(["group", "method", "seed", "step"]).val_acc.mean().round(4).reset_index().to_csv(
                "results/h1_curves.csv", index=False)
    else:
        df, cv = pd.read_csv("results/h1_runs.csv"), pd.read_csv("results/h1_curves.csv")
    counts = df.groupby(["group", "method"]).size()
    groups = [g for g in GROUPS if all(counts.get((g, m), 0) == 10 for m in METHODS)]
    if not partial:
        assert len(df) == 240 and len(groups) == 4 and (df.steps == 20000).all(), counts
    df, cv = df[df.group.isin(groups)], cv[cv.group.isin(groups)]
    old = first_round()
    text = (f"# H1, long training ({len(df)} runs from outputs/h1: 5,000 steps per task, 10 seeds per cell)\n\n"
            "Final accuracy = validation accuracy at the last step, mean of the four tasks, in percent. "
            "Mean ± standard error over seeds; differences are paired by seed; bold: beyond two standard errors.\n\n"
            "## Memory methods and joint ERM against fine-tuning, 5,000 steps per task\n\n" + table_vs_ft(df, groups)
            + "\n## PDCL against replay with a fixed weight, 5,000 steps per task\n\n"
              "Best weight = the weight with the highest mean final accuracy on that stream, picked on the same "
              "seeds (favours replay).\n\n" + table_vs_er(df, groups)
            + "\n## Secondary: 5,000 against 1,000 steps per task, same method, same seeds\n\n"
              "\"not run\": that replay weight was not run at 1,000 steps per task on that stream.\n\n"
            + table_length(df, old, groups)
            + "\n## Gain over fine-tuning at both lengths\n\n" + table_gain_length(df, old, groups)
            + "\n## Over-training: highest checkpoint of the seed-mean curve against the last step\n\n"
              "Validation accuracy, mean of the four tasks, checkpoints every 500 steps. For the sequential "
              "methods early checkpoints precede some tasks, so only a peak in the last phase (after step "
              "15,000) says that training went on too long.\n\n" + table_peak(cv, groups)
            + "\n## Methods, 5,000 steps per task\n\n" + table_final(df, groups))
    if not partial:
        Path("results/h1_tables.md").write_text(text)
        figure(df, old, cv, groups)
    print(text)


if __name__ == "__main__":
    main()
