"""F3, tolerable drift: tables and figure from outputs/f3 and the first round's runs.

    python -m domainbed.scripts.f3_analysis

Reads outputs/f3 (skew:0.75, skew:0.85, drift:0.4, 0.6, 0.8) through
stream_analysis.table and writes results/f3_runs.csv; adds the first round's
per-run table results/e1_runs_main.csv (results/e1_runs.csv of branch
claude/pdcl-e1-retention-dial-6f00e4, copied unchanged) for skew:0.5, 0.6, 0.7,
0.8, 0.9, 0.95 and rounds. Writes results/f3_tables.md and
paper/figures/fig_f3_dial.{pdf,png}.

Binder-rate swing = highest minus lowest binder rate among the tasks of a
stream: 2Q - 1 for skew:Q, D for drift:D, 0.20 for the real rounds.
Threshold of a method = the smallest swing at which it is more than two
standard errors above fine-tuning in final accuracy (paired by seed) and stays
so at all larger swings.
"""
from pathlib import Path

import pandas as pd

from domainbed.scripts import paper_assets as pa
from domainbed.scripts.stream_analysis import paired, table

SKEW = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
DRIFT = [0.4, 0.6, 0.8]
ALT = {f"skew{q}": round(2 * q - 1, 2) for q in SKEW}          # group -> swing
GRAD = {f"drift{d}": d for d in DRIFT}
DIALS = {"Alternating": ALT, "Gradual": GRAD}
SWING = ALT | GRAD | {"rounds": 0.20}
ROUNDS_SWING = 0.20
METHODS = ["ERM", "Finetune", "ER", "PDCL_RS"]
SHORT = {"ERM": "Joint", "Finetune": "FT", "ER": "ER", "PDCL_RS": "PDCL"}
STYLE = {"Alternating": "-", "Gradual": "--"}


def pm(m, s, nd=2, sign=False):
    return f"{m:{'+' if sign else ''}.{nd}f} ± {s:.{nd}f}"


def bold(d, se, text):
    return f"**{text}**" if abs(d) > 2 * se else text


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def stat(df, group, method, col):
    v = df[(df.group == group) & (df.method == method)][col]
    return v.mean(), v.sem()


def name(g):
    return "Rounds (real)" if g == "rounds" else g.replace("skew", "skew:").replace("drift", "drift:")


def gains(df):
    """Per stream: ER, PDCL and joint ERM minus fine-tuning, final accuracy, paired by seed."""
    return {m: paired(df, m, "Finetune") for m in ("ER", "PDCL_RS", "ERM")}


def threshold(gain, dial):
    """Smallest swing from which the gain stays above two standard errors."""
    groups = sorted(dial, key=dial.get)
    above = [gain.loc[g, "diff"] > 2 * gain.loc[g, "se"] for g in groups]
    first = None
    for g, ok in zip(reversed(groups), reversed(above)):
        if not ok:
            break
        first = dial[g]
    return first


def table_main(df, g_):
    rows = []
    for label, dial in list(DIALS.items()) + [("Reference", {"rounds": ROUNDS_SWING})]:
        for g in sorted(dial, key=dial.get):
            cells = [label, name(g), f"{dial[g]:.2f}", pm(*stat(df, g, "ERM", "final_acc")),
                     pm(*stat(df, g, "Finetune", "final_acc")), pm(*stat(df, g, "Finetune", "forgetting"))]
            for m in ("ER", "PDCL_RS"):
                d, se = g_[m].loc[g, ["diff", "se"]]
                cells.append(bold(d, se, pm(d, se, 2, True)))
            d, se = paired(df, "PDCL_RS", "ER").loc[g, ["diff", "se"]]
            cells.append(bold(d, se, pm(d, se, 2, True)))
            rows.append(cells)
    return md(["Dial", "Stream", "Swing", "Joint, final accuracy", "FT, final accuracy", "FT, forgetting",
               "ER − FT", "PDCL − FT", "PDCL − ER"], rows)


def table_thresholds(g_):
    fmt = lambda t, dial: f"{t:.2f}" if t is not None else f"none up to {max(dial.values()):.2f}"
    rows = [[SHORT[m]] + [fmt(threshold(g_[m], dial), dial) for dial in DIALS.values()] for m in ("ER", "PDCL_RS")]
    return md(["Method", "Alternating (swings 0 to 0.90)", "Gradual (swings 0.40, 0.60, 0.80)"], rows)


def table_equal_swing(df):
    """Gradual minus alternating at equal swing, same seeds: fine-tuning's final
    accuracy, its forgetting, and its shortfall against joint training."""
    wide = df.pivot_table(index="seed", columns=["method", "group"], values=["final_acc", "forgetting"])
    rows = []
    for d in DRIFT:
        a, g = f"skew{round((1 + d) / 2, 2)}", f"drift{d}"
        acc, fg = wide["final_acc"], wide["forgetting"]
        cols = [acc["Finetune"][g] - acc["Finetune"][a], fg["Finetune"][g] - fg["Finetune"][a],
                (acc["Finetune"][g] - acc["ERM"][g]) - (acc["Finetune"][a] - acc["ERM"][a])]
        rows.append([f"{d:.2f}", name(a), name(g)] + [bold(c.mean(), c.sem(), pm(c.mean(), c.sem(), 2, True))
                                                      for c in cols])
    return md(["Swing", "Alternating", "Gradual", "FT final accuracy, gradual − alternating",
               "FT forgetting, gradual − alternating", "FT − Joint, gradual − alternating"], rows)


def table_methods(df):
    rows = [[name(g), f"{SWING[g]:.2f}", pa.LABEL[m]] + [pm(*stat(df, g, m, c)) for c in
                                                         ("final_acc", "forgetting", "learned_acc")]
            for g in SWING for m in METHODS]
    return md(["Stream", "Swing", "Method", "Final accuracy", "Forgetting", "Learned accuracy"], rows)


def figure(df, g_):
    pa.style()
    fig, (a, b) = pa.panels(2, h=80)
    a2 = a.twinx()
    a2.spines["right"].set_visible(True)
    for label, dial in DIALS.items():
        groups = sorted(dial, key=dial.get)
        x = [dial[g] for g in groups]
        for ax, col, color, marker in ((a, "forgetting", pa.ORANGE, "o"), (a2, "final_acc", pa.NIGHT, "s")):
            v = pd.DataFrame([stat(df, g, "Finetune", col) for g in groups], columns=["m", "s"])
            ax.errorbar(x, v.m, yerr=v.s, color=color, marker=marker, linestyle=STYLE[label], capsize=1.5,
                        elinewidth=0.7, zorder=3)
        for m, marker in (("ER", "^"), ("PDCL_RS", "D")):
            d, se = g_[m].loc[groups, "diff"].values, g_[m].loc[groups, "se"].values
            b.plot(x, d, color=pa.COLOR[m], marker=marker, linestyle=STYLE[label], zorder=3,
                   label=f"{pa.LABEL[m]}, {label}")
            b.fill_between(x, d - 2 * se, d + 2 * se, color=pa.COLOR[m], alpha=0.15, linewidth=0, zorder=2)
    b.axhline(0, color=pa.GREY, linewidth=0.6, zorder=2)
    for ax in (a, b):
        ax.axvline(ROUNDS_SWING, color=pa.GREY, linestyle=":", linewidth=1, zorder=2)
        ax.set_xlabel("Binder-rate Swing (highest − lowest task rate)")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    a.annotate("Real rounds", (ROUNDS_SWING, 1), xycoords=("data", "axes fraction"), textcoords="offset points",
               xytext=(3, -8), fontsize=6, color=pa.GREY)
    a.set_ylabel("Forgetting of Fine-tuning (points)", color=pa.ORANGE)
    a2.set_ylabel("Final Accuracy of Fine-tuning (%)", color=pa.NIGHT)
    b.set_ylabel("Final Accuracy Gain over Fine-tuning (points)")
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], color=pa.ORANGE, marker="o", label="Forgetting"),
                      Line2D([], [], color=pa.NIGHT, marker="s", label="Final Accuracy"),
                      Line2D([], [], color=pa.GREY, linestyle="-", label="Alternating"),
                      Line2D([], [], color=pa.GREY, linestyle="--", label="Gradual")],
             loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2, handlelength=2.4)
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2, handlelength=2.4)
    fig.tight_layout(pad=0.6, w_pad=2.5)
    fig.savefig(pa.FIG / "fig_f3_dial.png", dpi=300)
    pa.save(fig, "fig_f3_dial")


def main():
    new = table(Path("outputs/f3")) if Path("outputs/f3").exists() else pd.DataFrame()
    if len(new):
        new.to_csv("results/f3_runs.csv", index=False)
    else:
        new = pd.read_csv("results/f3_runs.csv")
    old = pd.read_csv("results/e1_runs_main.csv")
    df = pd.concat([new, old[old.group.isin(SWING)]], ignore_index=True)
    df = df[df.method.isin(METHODS) & df.group.isin(SWING)]
    counts = df.groupby(["group", "method"]).size()
    assert len(counts) == len(SWING) * len(METHODS) and (counts == 10).all(), counts[counts != 10]
    g_ = gains(df)
    text = (f"# F3, tolerable drift ({len(new)} new runs from outputs/f3, {len(df) - len(new)} from the first round; "
            f"10 seeds per cell, 1,000 steps per task)\n\n"
            "Differences are paired by seed, mean ± standard error; bold: beyond two standard errors.\n\n"
            "## Dial\n\n" + table_main(df, g_)
            + "\n## Thresholds (smallest swing from which the method stays more than two standard errors above "
              "fine-tuning)\n\n" + table_thresholds(g_)
            + "\n## Gradual against alternating at equal swing\n\n" + table_equal_swing(df)
            + "\n## Methods\n\n" + table_methods(df))
    Path("results/f3_tables.md").write_text(text)
    figure(df, g_)
    print(text)


if __name__ == "__main__":
    main()
