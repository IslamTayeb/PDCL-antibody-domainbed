"""E3: memory allocation (alpha) and the dual variables as a readout.

    python -m domainbed.scripts.e3_analysis            # tables and figures from results/e3_runs.csv
    python -m domainbed.scripts.e3_analysis --collect  # first rebuild the table from outputs/e3

Writes results/e3_runs.csv (with --collect), results/e3_tables.md,
results/e3_readout_points.csv and paper/figures/fig_e3_{allocation,readout}.{pdf,png}.
The pre-registered design is in findings.md, section 0.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from domainbed.scripts import stream_analysis
from domainbed.scripts.paper_assets import CYAN, FIG, GREY, NIGHT, ORANGE, RED, YELLOW, panels, save, style

RUNS = Path("results/e3_runs.csv")
LASTPHASE = Path("results/e3_dual_lastphase.csv")
STREAMS = ["hetero", "skew:0.9:8", "agrate", "rounds"]
MEMS = [40, 200]
ALPHAS = ["a0", "a0.5", "a1"]
PD = {a: f"PDCL_RS__{a}" for a in ALPHAS}
E1_GLOB = "/home/imt11/PDCL-antibody-domainbed/.claude/worktrees/pdcl-e1-retention-dial*/results/e1*.csv"
OUT = []


def say(*lines):
    for line in lines:
        print(line)
        OUT.append(line)


def collect():
    df = stream_analysis.table("outputs/e3")
    df["memory"] = df["group"].str.extract(r"_m(\d+)$")[0].astype(float)
    df.to_csv(RUNS, index=False)
    last_phase_duals().to_csv(LASTPHASE, index=False)
    return df


def last_phase_duals():
    """Exploratory: PDCL's dual variable of each earlier task averaged over the checkpoints of the
    last phase only (default PDCL, memory 200, 1,000 steps per task), from the raw run records."""
    import glob
    import json
    dirs = [(d, "E3") for d in glob.glob("outputs/e3/*_m200/PDCL_RS__a0.5/env-1_seed*")]
    dirs += [(d, "E1") for d in glob.glob(E1_GLOB.split("/results/")[0] + "/outputs/e1/*/PDCL_RS/env-1_seed*")]
    rows = []
    for d, source in sorted(dirs):
        if not Path(d, "done").exists():
            continue
        recs = [json.loads(l) for l in open(Path(d, "results.jsonl"))]
        a = recs[-1]["args"]
        T = sum(1 for k in recs[-1] if k.startswith("env") and k.endswith("_out_acc"))
        if a["steps"] != 1000 * T or (source == "E1" and a["stream"] in STREAMS):
            continue
        for k in range(T - 1):
            rows.append({"stream": a["stream"], "seed": a["seed"], "task": k, "dual_lastphase": float(np.mean(
                [r[f"dual_{k}"] for r in recs if r["step"] > (T - 1) * 1000]))})
    return pd.DataFrame(rows)


def pm(v, nd=2):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1) / np.sqrt(len(v)):.{nd}f}"


def diff(a, b, nd=2):
    """Paired difference a - b over seeds (Series indexed by seed); * marks |diff| > 2 SE."""
    d = (a - b).dropna()
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return m, se, f"{m:+.{nd}f} ± {se:.{nd}f}" + (" *" if abs(m) > 2 * se and se > 0 else "")


def md(header, rows):
    say("", "| " + " | ".join(header) + " |", "|" + "---|" * len(header))
    for r in rows:
        say("| " + " | ".join(str(c) for c in r) + " |")


def sel(df, stream, method, memory=None):
    q = df[(df["stream"] == stream) & (df["method"] == method)]
    if memory is not None:
        q = q[q["memory"] == memory]
    return q.set_index("seed").sort_index()


def early_cols(n_tasks):
    return [f"acc_{k}" for k in range(n_tasks // 2)]


def shares(run, n_tasks):
    """Share of the memory held by each stored task at the last step."""
    b = run[[f"buf_{k}" for k in range(n_tasks - 1)]].astype(float)
    return b.div(b.sum(axis=1), axis=0)


# --------------------------------------------------------------------------
def allocation(df):
    say("## 1. Allocation", "",
        "Final accuracy and forgetting in percent, mean ± SE over ten seeds. Differences are paired by seed;",
        "`*` marks a difference larger than two standard errors.")
    rows, drows, srows, fig_data = [], [], [], {}
    for stream in STREAMS:
        T = int(df[df["stream"] == stream]["n_tasks"].iloc[0])
        ref = {m: sel(df, stream, m) for m in ("ERM", "Finetune")}
        rows.append([stream, "–", "Joint ERM", pm(ref["ERM"]["final_acc"]), pm(ref["ERM"]["forgetting"])])
        rows.append([stream, "–", "Fine-tuning", pm(ref["Finetune"]["final_acc"]), pm(ref["Finetune"]["forgetting"])])
        for mem in MEMS:
            r = {a: sel(df, stream, PD[a], mem) for a in ALPHAS}
            er = sel(df, stream, "ER", mem)
            rows.append([stream, mem, "ER", pm(er["final_acc"]), pm(er["forgetting"])])
            for a in ALPHAS:
                rows.append([stream, mem, f"PDCL alpha {a[1:]}", pm(r[a]["final_acc"]), pm(r[a]["forgetting"])])
            for a in ("a0.5", "a1"):
                e = lambda x: x[early_cols(T)].mean(axis=1)
                d_acc = diff(r[a]["final_acc"], r["a0"]["final_acc"])
                drows.append([stream, mem, f"{a[1:]} − 0", d_acc[2],
                              diff(r[a]["forgetting"], r["a0"]["forgetting"])[2],
                              diff(r[a]["acc_0"], r["a0"]["acc_0"])[2], diff(e(r[a]), e(r["a0"]))[2]])
                if a == "a0.5":
                    fig_data[stream, mem] = d_acc[:2]
            drows.append([stream, mem, "0.5 − ER", diff(r["a0.5"]["final_acc"], er["final_acc"])[2],
                          diff(r["a0.5"]["forgetting"], er["forgetting"])[2], "", ""])
            drows.append([stream, mem, "0.5 − FT", diff(r["a0.5"]["final_acc"], ref["Finetune"]["final_acc"])[2],
                          diff(r["a0.5"]["forgetting"], ref["Finetune"]["forgetting"])[2], "", ""])
            sh = {a: shares(r[a], T) for a in ALPHAS}
            for a in ALPHAS:
                moved = 100 * 0.5 * (sh[a] - sh["a0"]).abs().sum(axis=1)
                duals = r[a][[f"dual_{k}" for k in range(T - 1)]].mean()
                srows.append([stream, mem, a[1:],
                              " / ".join(f"{v:.1f}" for v in r[a][[f"buf_{k}" for k in range(T - 1)]].mean()),
                              " / ".join(f"{100 * v:.0f}" for v in sh[a].mean()),
                              f"{moved.mean():.1f} (max {moved.max():.1f})",
                              " / ".join(f"{v:.2f}" for v in duals)])
                fig_data[stream, mem, a] = sh[a].mean().values
    say("", "### 1a. Final accuracy and forgetting")
    md(["Stream", "Memory", "Method", "Final accuracy", "Forgetting"], rows)
    say("", "### 1b. Paired differences",
        "", "Early tasks = first half of the stream (tasks 0–1 of four, 0–3 of eight).")
    md(["Stream", "Memory", "Difference", "Final accuracy", "Forgetting", "Accuracy task 0", "Accuracy early tasks"],
       drows)
    say("", "### 1c. How the memory was divided at the last step",
        "", "Stored examples and share per earlier task (the last task is never stored). *Moved* is the share of the",
        "memory that sits in a different task than under alpha 0 in the same seed (total variation, percent).",
        "Dual variables are the values at the last step.")
    md(["Stream", "Memory", "alpha", "Stored per task", "Share per task (%)", "Moved vs alpha 0 (%)",
        "Final dual per task"], srows)
    return fig_data


def per_task(df):
    say("", "### 1d. Per-task accuracy difference, alpha 1 − alpha 0 (largest reallocation)")
    rows = []
    for stream in STREAMS:
        T = int(df[df["stream"] == stream]["n_tasks"].iloc[0])
        for mem in MEMS:
            a1, a0 = sel(df, stream, PD["a1"], mem), sel(df, stream, PD["a0"], mem)
            rows.append([stream, mem, "; ".join(f"{k}: {diff(a1[f'acc_{k}'], a0[f'acc_{k}'])[2]}" for k in range(T))])
    md(["Stream", "Memory", "Task: difference"], rows)


# --------------------------------------------------------------------------
def readout_points(df):
    """One row per stream, seed and earlier task: PDCL's dual (default alpha, memory 200)
    and fine-tuning's forgetting of the task in the same seed."""
    rows = []

    def add(pdcl, ft, stream, source):
        T = int(pdcl["n_tasks"].iloc[0])
        for s in sorted(set(pdcl.index) & set(ft.index)):
            for k in range(T - 1):
                rows.append({"stream": stream, "source": source, "seed": s, "task": k,
                             "dual_mean": pdcl.loc[s, f"dual_mean_{k}"], "dual_last": pdcl.loc[s, f"dual_{k}"],
                             "ft_forget": ft.loc[s, f"forget_{k}"], "pdcl_forget": pdcl.loc[s, f"forget_{k}"]})

    for stream in STREAMS:
        add(sel(df, stream, PD["a0.5"], 200), sel(df, stream, "Finetune"), stream, "E3")
    used = []
    import glob
    for path in sorted(glob.glob(E1_GLOB)):
        e1 = pd.read_csv(path)
        e1 = e1[(e1["env"] == -1) & (e1["steps"] == 1000 * e1["n_tasks"])]
        for stream in sorted(set(e1["stream"]) - set(STREAMS) - {r["stream"] for r in rows}):
            p, f = sel(e1, stream, "PDCL_RS"), sel(e1, stream, "Finetune")
            if len(p) and len(f) and p.index.is_unique and f.index.is_unique:
                add(p, f, stream, "E1")
                used.append((stream, path))
    return pd.DataFrame(rows), used


def rho(x, y):
    return spearmanr(x, y)[0] if len(set(x)) > 1 and len(set(y)) > 1 else np.nan


def boot(pts, stat, n=10000, seed=0):
    """Bootstrap over seeds: the same resampled seeds in every stream."""
    rng = np.random.RandomState(seed)
    seeds = sorted(pts["seed"].unique())
    by = {s: g for s, g in pts.groupby("seed")}
    vals = [stat(pd.concat([by[s] for s in rng.choice(seeds, len(seeds))])) for _ in range(n)]
    return np.nanpercentile(vals, [2.5, 97.5])


def task_means(p):
    return p.groupby(["stream", "task"], sort=False)[["dual_mean", "ft_forget"]].mean()


def readout(df):
    pts, used = readout_points(df)
    pts.to_csv("results/e3_readout_points.csv", index=False)
    say("", "## 2. Dual readout", "",
        "x = PDCL's dual variable of an earlier task (mean over the checkpoints after the task's phase; default",
        "alpha 0.5, memory 200), y = fine-tuning's forgetting of that task in the same seed (percent).")
    if used:
        say("", "Streams added from the E1 table (read-only): " + ", ".join(f"`{s}`" for s, _ in used)
            + f" from `{used[0][1]}`. `rounds` and `agrate` are taken from the E3 runs.")
    else:
        say("", "No E1 table was available; only the four E3 streams are used.")
    stats = {
        "pooled over seeds, tasks and streams": lambda p: rho(p["dual_mean"], p["ft_forget"]),
        "task means (one point per task and stream)": lambda p: rho(*task_means(p).values.T),
        "stream means (one point per stream)": lambda p: rho(*p.groupby("stream")[["dual_mean", "ft_forget"]].mean().values.T),
        "mean of the within-stream correlations": lambda p: np.nanmean(
            [rho(g["dual_mean"], g["ft_forget"]) for _, g in p.groupby("stream")]),
    }
    rows, res = [], {}
    for scope, sub in [("E3 streams", pts[pts["source"] == "E3"])] + ([("E3 + E1 streams", pts)] if used else []):
        for name, stat in stats.items():
            lo, hi = boot(sub, stat, 2000 if "within" in name else 10000)
            npts = (len(sub) if name.startswith("pooled") else len(task_means(sub)) if name.startswith("task")
                    else sub["stream"].nunique())
            rows.append([scope, name, npts, f"{stat(sub):+.2f}", f"[{lo:+.2f}, {hi:+.2f}]"])
            res[scope, name] = stat(sub)
    say("", "### 2a. Spearman rank correlations (95% interval from a bootstrap over the ten seeds)")
    md(["Streams", "Level", "Points", "Spearman", "Bootstrap 95%"], rows)

    rows = []
    for stream, g in pts.groupby("stream", sort=False):
        tm = g.groupby("task")[["dual_mean", "ft_forget"]].mean()
        rows.append([stream, g["source"].iloc[0], len(g), f"{rho(g['dual_mean'], g['ft_forget']):+.2f}",
                     f"{rho(tm['dual_mean'], tm['ft_forget']):+.2f}" if len(tm) > 2 else "–",
                     f"{g['dual_mean'].mean():.3f}", f"{g['ft_forget'].mean():.2f}",
                     f"{100 * (g['dual_mean'] > 0).mean():.0f}"])
    say("", "### 2b. Within each stream")
    md(["Stream", "Runs from", "Points", "Spearman (seeds × tasks)", "Spearman (task means)", "Mean dual",
        "Mean FT forgetting", "Duals > 0 (%)"], rows)

    rows = []
    for stream, g in pts.groupby("stream", sort=False):
        for k, t in g.groupby("task"):
            rows.append([stream, k, pm(t["dual_mean"], 3), pm(t["dual_last"], 3), pm(t["ft_forget"]),
                         pm(t["pdcl_forget"])])
    say("", "### 2c. Per task (mean ± SE over seeds)")
    md(["Stream", "Task", "Dual (mean after phase)", "Dual (last step)", "FT forgetting", "PDCL forgetting"], rows)

    h = pts[pts["stream"] == "hetero"].pivot(index="seed", columns="task")
    rows = []
    for col, label in (("dual_mean", "dual, mean after phase"), ("dual_last", "dual, last step"),
                       ("ft_forget", "FT forgetting")):
        rows.append([label, diff(h[col][0], h[col][1], 3)[2], diff(h[col][2], h[col][1], 3)[2],
                     diff(h[col][0], h[col][2], 3)[2],
                     f"{int((h[col][0] > h[col][1]).sum())}/10, {int((h[col][2] > h[col][1]).sum())}/10"])
    say("", "### 2d. `hetero`: do the duals single out tasks 0 and 2 (binder rate 0.72) over task 1 (0.52)?",
        "", "Paired differences over seeds.")
    md(["Quantity", "Task 0 − task 1", "Task 2 − task 1", "Task 0 − task 2", "Seeds with 0 > 1, 2 > 1"], rows)
    exploratory(pts)
    return pts, res


def exploratory(pts):
    if not LASTPHASE.exists():
        return
    p = pts.merge(pd.read_csv(LASTPHASE), on=["stream", "seed", "task"])
    say("", "### 2e. Exploratory (not pre-registered): dual averaged over the last phase only",
        "", "The pre-registered dual is averaged over every phase after the task's own, so it also counts phases in",
        "which the task was costly but whose damage was undone by the end of the stream. Fine-tuning's forgetting",
        "is measured at the end. Here the dual is averaged over the checkpoints of the last phase only.")
    rows = []
    stats = (("pooled", lambda q, c: rho(q[c], q["ft_forget"])),
             ("task means", lambda q, c: rho(*q.groupby(["stream", "task"])[[c, "ft_forget"]].mean().values.T)),
             ("mean within stream", lambda q, c: np.nanmean([rho(g[c], g["ft_forget"]) for _, g in q.groupby("stream")])))
    for scope, sub in (("E3 streams", p[p["source"] == "E3"]), ("E3 + E1 streams", p)):
        for name, stat in stats:
            lo, hi = boot(sub, lambda q: stat(q, "dual_lastphase"), 2000)
            rows.append([scope, name, len(sub), f"{stat(sub, 'dual_mean'):+.2f}", f"{stat(sub, 'dual_lastphase'):+.2f}",
                         f"[{lo:+.2f}, {hi:+.2f}]"])
    md(["Streams", "Level", "Seed × task points", "Spearman, pre-registered dual", "Spearman, last-phase dual",
        "Bootstrap 95% (last-phase)"], rows)
    rows = []
    for stream in ("hetero", "skew:0.9:8"):
        g = p[p["stream"] == stream]
        rows.append([stream, " / ".join(f"{v:.3f}" for v in g.groupby("task")["dual_lastphase"].mean()),
                     " / ".join(f"{v:.1f}" for v in g.groupby("task")["ft_forget"].mean()),
                     f"{rho(g['dual_lastphase'], g['ft_forget']):+.2f}"])
    md(["Stream", "Last-phase dual per task", "FT forgetting per task", "Spearman (seeds × tasks)"], rows)
    h = p[p["stream"] == "hetero"].pivot(index="seed", columns="task")["dual_lastphase"]
    say("", f"`hetero`, last-phase dual, paired: task 0 − task 1 {diff(h[0], h[1], 3)[2]}; "
        f"task 2 − task 1 {diff(h[2], h[1], 3)[2]}.")


# --------------------------------------------------------------------------
def savefig(fig, name):
    fig.tight_layout(pad=0.6, w_pad=2.5)
    fig.savefig(FIG / f"{name}.png", dpi=300)
    save(fig, name)


def fig_allocation(fd):
    fig, (a, b) = panels(2)
    x = np.arange(len(STREAMS))
    for i, (mem, color) in enumerate(zip(MEMS, (RED, CYAN))):
        m = [fd[s, mem][0] for s in STREAMS]
        se = [fd[s, mem][1] for s in STREAMS]
        a.bar(x + (i - 0.5) * 0.36, m, 0.34, yerr=se, color=color, label=f"Memory {mem}", zorder=3,
              error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": NIGHT})
    a.axhline(0, color=NIGHT, linewidth=0.6, zorder=4)
    a.set_xticks(x)
    a.set_xticklabels(STREAMS)
    a.set_ylabel("Final Accuracy, α = 0.5 − α = 0 (points)")
    a.legend(loc="best")
    # Shares of the stored tasks, stacked: one bar per alpha, grouped by stream and memory.
    pos, ticks = 0, []
    for s in STREAMS:
        for mem in MEMS:
            for j, al in enumerate(ALPHAS):
                sh = 100 * fd[s, mem, al]
                tones = plt.get_cmap("cividis")(np.linspace(0.1, 0.9, len(sh)))
                b.bar(pos + j * 0.9, sh, 0.8, bottom=np.concatenate([[0], np.cumsum(sh)[:-1]]), color=tones,
                      edgecolor="white", linewidth=0.3, zorder=3)
            ticks.append((pos + 0.9, str(mem)))
            pos += 3.3
        b.text(pos - 3.75, -13, s, ha="center", va="top", fontsize=7)
        pos += 0.6
    b.set_xticks([t for t, _ in ticks])
    b.set_xticklabels([l for _, l in ticks], fontsize=7)
    b.set_xlabel(" ", labelpad=6)
    b.set_ylabel("Memory Share per Stored Task (%)")
    b.set_ylim(0, 100)
    b.set_title("Bars: α = 0, 0.5, 1; Segments: Tasks, Earliest at Bottom", fontsize=7)
    savefig(fig, "fig_e3_allocation")


def fig_readout(pts, res):
    fig, (a, b) = panels(2)
    colors = [RED, CYAN, ORANGE, NIGHT, YELLOW, GREY, "#5E81AC", "#A3BE8C", "#B48EAD", "#88C0D0", "#D08770",
              "#434C5E", "#81A1C1"]
    markers = "oD^sv<>PXph*d"
    for ax, sub, scope in ((a, pts[pts["source"] == "E3"], "E3 streams"), (b, pts, "E3 + E1 streams")):
        if scope not in {k[0] for k in res}:
            ax.set_axis_off()
            continue
        for i, (stream, g) in enumerate(sub.groupby("stream", sort=False)):
            tm = g.groupby("task")[["dual_mean", "ft_forget"]].mean()
            ax.scatter(tm["ft_forget"], tm["dual_mean"], color=colors[i % len(colors)],
                       marker=markers[i % len(markers)], s=16, label=stream, zorder=3,
                       edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Fine-tuning's Forgetting of the Task (points)")
        ax.set_ylabel("PDCL Dual Variable of the Task")
        ax.set_title(f"{scope}: Spearman {res[scope, 'task means (one point per task and stream)']:+.2f} (task means), "
                     f"{res[scope, 'pooled over seeds, tasks and streams']:+.2f} (pooled)", fontsize=7)
        ax.legend(loc="best", fontsize=6, ncol=2 if sub["stream"].nunique() > 6 else 1, handletextpad=0.2)
    savefig(fig, "fig_e3_readout")


def main():
    df = collect() if "--collect" in sys.argv else pd.read_csv(RUNS)
    say("# E3 tables (generated by `python -m domainbed.scripts.e3_analysis`)", "")
    counts = df.groupby(["group", "method"]).size()
    say(f"{len(df)} runs; every group and method has {counts.min()}–{counts.max()} seeds.")
    fd = allocation(df)
    per_task(df)
    pts, res = readout(df)
    style()
    fig_allocation(fd)
    fig_readout(pts, res)
    Path("results/e3_tables.md").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
