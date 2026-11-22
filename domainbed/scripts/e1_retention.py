"""E1, the retention dial: tables and figures from outputs/e1 and outputs/e1_long.

    python -m domainbed.scripts.e1_retention

Writes results/e1_runs.csv and results/e1_long_runs.csv (one row per run, from
stream_analysis.table), results/e1_tables.md (every table of findings_e1.md)
and paper/figures/fig_e1_*.{pdf,png}.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from domainbed.scripts import paper_assets as pa
from domainbed.scripts.stream_analysis import paired, table

LENGTHS = {"e1": "1,000 steps per task, seeds 0-9", "e1_long": "5,000 steps per task, seeds 0-4"}
TITLE = {"e1": "1,000 Steps per Task", "e1_long": "5,000 Steps per Task"}
SKEW = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
GROUPS = ["rounds", "rounds3210", "aglen", "target", "agrate"] + [f"skew{q}" for q in SKEW]
NAME = {"rounds": "Rounds", "rounds3210": "Rounds, Reversed", "aglen": "Antigen Length",
        "target": "Target", "agrate": "Antigen Binder Rate"} | {f"skew{q}": f"Skew {q}" for q in SKEW}
METHODS = ["ERM", "Finetune", "ER", "PDCL_RS"]
MARKER = {"ERM": "s", "Finetune": "o", "ER": "^", "PDCL_RS": "D"}
PAIRS = [("PDCL_RS", "Finetune"), ("ER", "Finetune"), ("PDCL_RS", "ER"), ("ERM", "Finetune")]
SHORT = {"ERM": "Joint", "Finetune": "FT", "ER": "ER", "PDCL_RS": "PDCL"}
TASKS = range(3)                                   # tasks with a memory and a dual variable


def pm(m, s, nd=1, sign=False):
    return f"{m:{'+' if sign else ''}.{nd}f} ± {s:.{nd}f}"


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def stat(df, group, method, col):
    v = df[(df.group == group) & (df.method == method)][col]
    return v.mean(), v.sem()


def flag(d, se):
    """A difference counts as real when it exceeds two standard errors."""
    return "**" if abs(d) > 2 * se else ""


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def table_methods(df):
    rows = []
    for g in GROUPS:
        for m in METHODS:
            rows.append([NAME[g], pa.LABEL[m]] + [pm(*stat(df, g, m, c)) for c in
                                                  ("final_acc", "forgetting", "learned_acc")])
    return md(["Stream", "Method", "Final accuracy", "Forgetting", "Learned accuracy"], rows)


def table_paired(df, value):
    p = {pair: paired(df, *pair, value=value) for pair in PAIRS}
    rows = []
    for g in GROUPS:
        cells = []
        for pair in PAIRS:
            d, se = p[pair].loc[g, ["diff", "se"]]
            cells.append(f"{flag(d, se)}{pm(d, se, 2, True)}{flag(d, se)}")
        rows.append([NAME[g]] + cells)
    return md(["Stream"] + [f"{SHORT[a]} − {SHORT[b]}" for a, b in PAIRS], rows)


def dial(df):
    """Per stream: fine-tuning's forgetting and the gains of ER and PDCL over it."""
    out = pd.DataFrame({"ft_forget": [stat(df, g, "Finetune", "forgetting")[0] for g in GROUPS],
                        "ft_forget_se": [stat(df, g, "Finetune", "forgetting")[1] for g in GROUPS]},
                       index=GROUPS)
    for m in ("ER", "PDCL_RS"):
        p = paired(df, m, "Finetune").loc[GROUPS]
        out[f"{m}_gain"], out[f"{m}_se"] = p["diff"], p["se"]
    return out


def trend(df, method, n_boot=10000, seed=0):
    """Gain over fine-tuning against fine-tuning's forgetting, across streams:
    Spearman rho and least-squares slope of the stream means, with a 95% interval
    from resampling seeds (the same seeds on every stream)."""
    wide = df.pivot_table(index="seed", columns=["method", "group"], values=["final_acc", "forgetting"])
    f = wide["forgetting"]["Finetune"][GROUPS].values                 # [seeds, streams]
    gain = (wide["final_acc"][method][GROUPS] - wide["final_acc"]["Finetune"][GROUPS]).values
    fit = lambda rows: (spearmanr(f[rows].mean(0), gain[rows].mean(0))[0],
                        np.polyfit(f[rows].mean(0), gain[rows].mean(0), 1)[0])
    rng = np.random.RandomState(seed)
    boot = np.array([fit(rng.randint(len(f), size=len(f))) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    rho, slope = fit(np.arange(len(f)))
    return {"rho": rho, "rho_lo": lo[0], "rho_hi": hi[0], "slope": slope, "slope_lo": lo[1], "slope_hi": hi[1]}


def table_dial(df):
    d = dial(df)
    rows = [[NAME[g], pm(d.ft_forget[g], d.ft_forget_se[g], 2),
             pm(d.ER_gain[g], d.ER_se[g], 2, True), pm(d.PDCL_RS_gain[g], d.PDCL_RS_se[g], 2, True)]
            for g in d.sort_values("ft_forget").index]
    text = md(["Stream (by fine-tuning's forgetting)", "Fine-tuning's forgetting", "ER − FT", "PDCL − FT"], rows)
    for m in ("PDCL_RS", "ER"):
        t = trend(df, m)
        text += (f"\n{SHORT[m]} gain against fine-tuning's forgetting, 11 streams: Spearman rho "
                 f"{t['rho']:.2f} [{t['rho_lo']:.2f}, {t['rho_hi']:.2f}], slope {t['slope']:.2f} "
                 f"[{t['slope_lo']:.2f}, {t['slope_hi']:.2f}] points of accuracy per point of forgetting "
                 f"(95% interval, seeds resampled).\n")
    return text


def table_floor(df):
    """Prediction (c): PDCL against the better of fine-tuning and ER on each stream."""
    rows = []
    for g in GROUPS:
        best = max(("Finetune", "ER"), key=lambda m: stat(df, g, m, "final_acc")[0])
        d, se = paired(df, "PDCL_RS", best).loc[g, ["diff", "se"]]
        rows.append([NAME[g], SHORT[best], pm(d, se, 2, True), "yes" if d >= -0.5 else "**no**"])
    return md(["Stream", "Better of FT and ER", "PDCL − better", "Within 0.5 points"], rows)


def mechanism(df):
    """Per stream, averaged over the three earlier tasks and the seeds."""
    rows = []
    for g in GROUPS:
        p = df[(df.group == g) & (df.method == "PDCL_RS")]
        e = df[(df.group == g) & (df.method == "ER")]
        per_seed = lambda d, key: d[[f"{key}_{k}" for k in TASKS]].mean(1)
        gap = (p[[f"replay_loss_{k}" for k in TASKS]].values - p[[f"eps_{k}" for k in TASKS]].values)
        rows.append({"group": g,
                     "dual_mean": per_seed(p, "dual_mean").mean(), "dual_mean_se": per_seed(p, "dual_mean").sem(),
                     "dual_last": per_seed(p, "dual").mean(), "dual_last_se": per_seed(p, "dual").sem(),
                     "dual_max": p[[f"dual_{k}" for k in TASKS]].values.max(),
                     "active": 100 * (p[[f"dual_{k}" for k in TASKS]].values > 0).mean(),
                     "violated": 100 * (gap > 0).mean(),
                     "eps": per_seed(p, "eps").mean(), "eps_se": per_seed(p, "eps").sem(),
                     "pdcl_loss": per_seed(p, "replay_loss").mean(), "pdcl_loss_se": per_seed(p, "replay_loss").sem(),
                     "er_loss": per_seed(e, "replay_loss").mean(), "er_loss_se": per_seed(e, "replay_loss").sem()})
    return pd.DataFrame(rows).set_index("group")


def table_mechanism(df):
    m = mechanism(df)
    rows = [[NAME[g], pm(r.dual_mean, r.dual_mean_se, 3), pm(r.dual_last, r.dual_last_se, 3), f"{r.dual_max:.2f}",
             f"{r.active:.0f}", pm(r.eps, r.eps_se, 3), pm(r.pdcl_loss, r.pdcl_loss_se, 3), f"{r.violated:.0f}",
             pm(r.er_loss, r.er_loss_se, 3)] for g, r in m.iterrows()]
    return md(["Stream", "Dual, mean after the task", "Dual, last step", "Largest dual, last step",
               "Duals above 0 at the end (%)", "Constraint level", "Replay loss, PDCL",
               "Replay loss above level (%)", "Replay loss, ER"], rows)


def table_tasks(df):
    """Per earlier task: its constraint, PDCL's dual variable and what each method forgets of it."""
    rows = []
    for g in GROUPS:
        sub = {m: df[(df.group == g) & (df.method == m)] for m in METHODS}
        p = sub["PDCL_RS"]
        for k in TASKS:
            cell = lambda d, col, nd=1: pm(d[f"{col}_{k}"].mean(), d[f"{col}_{k}"].sem(), nd)
            rows.append([NAME[g], str(k + 1), cell(p, "eps", 3), cell(p, "replay_loss", 3),
                         cell(sub["ER"], "replay_loss", 3), cell(p, "dual_mean", 3), cell(p, "dual", 3)]
                        + [cell(sub[m], "forget") for m in ("Finetune", "PDCL_RS", "ER")])
    return md(["Stream", "Task", "Constraint level", "Replay loss, PDCL", "Replay loss, ER",
               "Dual, mean after the task", "Dual, last step", "Forgetting, FT", "Forgetting, PDCL",
               "Forgetting, ER"], rows)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout(pad=0.6, w_pad=2.5)
    fig.savefig(pa.FIG / f"{name}.png", dpi=300)
    pa.save(fig, name)


def fig_gain(frames):
    """Headline: gain over fine-tuning against fine-tuning's forgetting."""
    fig, axes = pa.panels(len(frames), h=85)
    for ax, (key, df) in zip(axes, frames.items()):
        d = dial(df)
        for m in ("ER", "PDCL_RS"):
            ax.errorbar(d.ft_forget, d[f"{m}_gain"], yerr=d[f"{m}_se"], xerr=d.ft_forget_se, linestyle="none",
                        marker=MARKER[m], color=pa.COLOR[m], label=pa.LABEL[m], capsize=1.5, elinewidth=0.7,
                        zorder=3)
        ax.axhline(0, color=pa.GREY, linewidth=0.6, zorder=2)
        gains = d[["ER_gain", "PDCL_RS_gain"]]
        span = gains.values.max() - gains.values.min()
        # Streams that crowd together on the x axis get their labels in a column
        # below the points, joined to the lower point by a thin line.
        x = d.ft_forget
        hi = gains.max(1)
        crowd = [g for g in GROUPS if (((x - x[g]).abs() < 2.5) & ((hi - hi[g]).abs() < 0.2 * span)).sum() > 1]
        for g in GROUPS:
            if g not in crowd:
                ax.annotate(NAME[g].replace("Skew ", "Q = "), (x[g], max(d.loc[g, f"{m}_gain"] + d.loc[g, f"{m}_se"] for m in ("ER", "PDCL_RS"))),
                            textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6, color=pa.GREY)
        crowd.sort(key=lambda g: gains.loc[g].min(), reverse=True)
        top, step = -0.1 * span, 0.065 * span
        for i, g in enumerate(crowd):
            ax.annotate(NAME[g].replace("Skew ", "Q = "), (x[g], gains.loc[g].min()),
                        xytext=(x[crowd].max() + 0.07 * (x.max() - x.min()), top - i * step),
                        fontsize=6, color=pa.GREY, va="center",
                        arrowprops={"arrowstyle": "-", "color": pa.GREY, "linewidth": 0.4, "shrinkB": 3, "shrinkA": 1, "relpos": (0, 0.5)})
        ax.set_ylim(min(top - (len(crowd) - 0.3) * step, gains.values.min() - 0.08 * span),
                    gains.values.max() + 0.14 * span)
        ax.set_xlabel("Forgetting of Fine-tuning (points)")
        ax.set_ylabel("Final Accuracy Gain over Fine-tuning (points)")
        ax.margins(x=0.08)
        pa.top_legend(ax, title=TITLE[key])
    save(fig, "fig_e1_gain")


def fig_skew(frames):
    """Final accuracy and forgetting of the four methods against the skew Q."""
    for key, df in frames.items():
        fig, (a, b) = pa.panels(2)
        for ax, col, label in ((a, "final_acc", "Final Accuracy (%)"), (b, "forgetting", "Forgetting (points)")):
            for m in METHODS:
                v = np.array([stat(df, f"skew{q}", m, col) for q in SKEW])
                ax.errorbar(SKEW, v[:, 0], yerr=v[:, 1], marker=MARKER[m], color=pa.COLOR[m], label=pa.LABEL[m],
                            capsize=1.5, elinewidth=0.7, zorder=3)
            ax.set_xticks(SKEW)
            ax.set_xticklabels([f"{q:g}" for q in SKEW])
            ax.set_xlabel("Skew Q (share of binders in tasks 1 and 3)")
            ax.set_ylabel(label)
            pa.top_legend(ax)
        save(fig, "fig_e1_skew" + ("" if key == "e1" else "_long"))


def fig_dual(frames):
    """Mean dual variable against Q, and the replay loss against the constraint level."""
    for key, df in frames.items():
        m = mechanism(df).loc[[f"skew{q}" for q in SKEW]]
        fig, (a, b) = pa.panels(2)
        a.errorbar(SKEW, m.dual_mean, yerr=m.dual_mean_se, marker="D", color=pa.RED, capsize=1.5, elinewidth=0.7,
                   label="Mean after the Task", zorder=3)
        a.errorbar(SKEW, m.dual_last, yerr=m.dual_last_se, marker="o", color=pa.ORANGE, capsize=1.5,
                   elinewidth=0.7, label="Last Step", zorder=3)
        a.axhline(1, color=pa.CYAN, linestyle=":", linewidth=1, zorder=2)
        a.annotate("Replay (ER) weight", (SKEW[0], 1), textcoords="offset points", xytext=(0, -9), fontsize=6,
                   color=pa.GREY)
        a.set_ylabel("PDCL Dual Variable")
        b.errorbar(SKEW, m.er_loss, yerr=m.er_loss_se, marker="^", color=pa.CYAN, capsize=1.5, elinewidth=0.7,
                   label="Replay (ER)", zorder=3)
        b.errorbar(SKEW, m.pdcl_loss, yerr=m.pdcl_loss_se, marker="D", color=pa.RED, capsize=1.5, elinewidth=0.7,
                   label="PDCL", zorder=3)
        b.errorbar(SKEW, m.eps, yerr=m.eps_se, color=pa.RED, linestyle="--", capsize=1.5, elinewidth=0.7,
                   label="PDCL Constraint Level", zorder=3)
        b.set_ylabel("Loss on Stored Examples")
        for ax in (a, b):
            ax.set_xticks(SKEW)
            ax.set_xticklabels([f"{q:g}" for q in SKEW])
            ax.set_xlabel("Skew Q (share of binders in tasks 1 and 3)")
            pa.top_legend(ax)
        save(fig, "fig_e1_dual" + ("" if key == "e1" else "_long"))


# --------------------------------------------------------------------------
def main():
    frames = {}
    for key in LENGTHS:
        df = table(Path("outputs", key)) if Path("outputs", key).exists() else pd.DataFrame()
        if len(df):
            df.to_csv(f"results/{key}_runs.csv", index=False)
            frames[key] = df
        elif Path(f"results/{key}_runs.csv").exists():
            frames[key] = pd.read_csv(f"results/{key}_runs.csv")
    text = ""
    for key, df in frames.items():
        counts = df.groupby(["group", "method"]).size()
        text += f"# {LENGTHS[key]} ({len(df)} runs, {counts.min()}-{counts.max()} per cell)\n\n"
        text += "## Methods\n\n" + table_methods(df)
        for value in ("final_acc", "forgetting"):
            text += f"\n## Paired differences, {value} (bold: beyond two standard errors)\n\n" + table_paired(df, value)
        text += "\n## Dial\n\n" + table_dial(df)
        text += "\n## PDCL against the better of fine-tuning and ER\n\n" + table_floor(df)
        text += "\n## Mechanism\n\n" + table_mechanism(df)
        text += "\n## Per task\n\n" + table_tasks(df) + "\n"
    Path("results/e1_tables.md").write_text(text)
    pa.style()
    fig_gain(frames)
    fig_skew(frames)
    fig_dual(frames)
    print(text)


if __name__ == "__main__":
    main()
