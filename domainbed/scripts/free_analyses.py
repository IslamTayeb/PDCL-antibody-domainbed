"""Analyses that need no new runs: scope of the forgetting result and three exploratory read-outs.

    python -m domainbed.scripts.free_analyses

Reads the raw runs of E1 (main and long), E2, E3, F1, F2 and F3 from the sibling
worktrees (not in git) and writes results/free_analyses.md and
paper/figures/fig_g1_tradeoff.{pdf,png}, fig_g2_next_round.{pdf,png}.

    A   two-by-two: antigens shared or new x binder rate stable or shifting (E1 main)
    B   where the real rounds sit on the binder-rate dial (E1 main + F3)
    C   PDCL minus fine-tuning in every cell of every root
    G1  last-task accuracy against earlier-task accuracy, PDCL against the ER curve
    G2  accuracy on the next round before it is trained on (real rounds)
    G3  share of earlier tasks whose validation loss stays within 1.1x

All numbers are mean ± standard error over seeds; differences are paired by seed
and bold when they exceed two standard errors. Accuracies are in percent.
A checkpoint logged at step s is taken after the update with index s, so the
checkpoint "at the end of phase k" (step 1000 (k+1), the convention of
stream_analysis.run_row) has taken one update of the next task.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from domainbed.scripts import paper_assets as pa
from domainbed.scripts.stream_analysis import run_row

REPO = Path(__file__).resolve().parents[2]
TREES = REPO.parent
ROOTS = {
    "E1": TREES / "pdcl-e1-retention-dial-6f00e4/outputs/e1",
    "E1 long": TREES / "pdcl-e1-retention-dial-6f00e4/outputs/e1_long",
    "E2": TREES / "pdcl-e2-adaptivity-09393c/outputs/e2",
    "E3": TREES / "pdcl-e3-allocation-3e05ed/outputs/e3",
    "F1": TREES / "pdcl-f1-honest-constraints-2145b2/outputs/f1",
    "F2": TREES / "pdcl-f2-replay-memorization-3869d5/outputs/f2",
    "F3": TREES / "pdcl-f3-tolerable-drift-0389fa/outputs/f3",
}
OUT, FIG = REPO / "results/free_analyses.md", REPO / "paper/figures"
SWING = {f"skew{q}": round(2 * q - 1, 2) for q in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)}
SWING |= {f"drift{d}": d for d in (0.4, 0.6, 0.8)} | {"rounds": 0.20, "rounds3210": 0.20}
SLACK = 1.1
N_BOOT = 2000


# ---------------------------------------------------------------- reading runs

def extras(recs):
    """Per-run quantities that run_row does not give."""
    last = recs[-1]
    n = sum(1 for k in last if k.startswith("env") and k.endswith("_out_acc"))
    steps = np.array([r["step"] for r in recs])
    phase = (steps[-1] + 1) // n
    at = lambda step: recs[int(np.argmin(np.abs(steps - step)))]
    row = {"memory": (last.get("hparams") or {}).get("buffer_size"),
           "earlier_acc": 100 * float(np.mean([last[f"env{k}_out_acc"] for k in range(n - 1)])),
           "last_acc": 100 * last[f"env{n - 1}_out_acc"]}
    for k in range(n - 1):
        end = (k + 1) * phase
        row[f"next_{k}"] = 100 * at(end)[f"env{k + 1}_out_acc"]
        row[f"next_strict_{k}"] = 100 * recs[int(np.flatnonzero(steps < end)[-1])][f"env{k + 1}_out_acc"]
    key = next((p for p in ("val_loss_{}", "env{}_out_loss") if p.format(0) in last), None)
    if key:
        ok = [last[key.format(k)] <= SLACK * at((k + 1) * phase)[key.format(k)] for k in range(n - 1)]
        row["kept_share"] = 100 * float(np.mean(ok))
        row["loss_ratio"] = float(np.mean([last[key.format(k)] / at((k + 1) * phase)[key.format(k)]
                                           for k in range(n - 1)]))
    return row


@lru_cache(None)
def load(name):
    """One row per finished whole-stream run of a root; also the count of unfinished run folders."""
    rows, root = [], ROOTS[name]
    for res in sorted(root.glob("*/*/*/results.jsonl")):
        d = res.parent
        if not (d / "done").exists():
            continue
        recs = [json.loads(l) for l in open(res)]
        if recs[-1]["args"]["test_envs"]:
            continue
        rows.append({"root": name, "group": d.parent.parent.name, "method": d.parent.name}
                    | run_row(recs) | extras(recs))
    started = sum(1 for d in root.glob("*/*/*") if d.is_dir())
    return pd.DataFrame(rows), started


def df_of(name):
    return load(name)[0]


# ---------------------------------------------------------------- small helpers

def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def cell(df, group, method, col):
    return df[(df.group == group) & (df.method == method)].set_index("seed")[col].sort_index()


def diff(a, b):
    """Mean, standard error and count of a - b over the seeds both have."""
    d = (a - b).dropna()
    return (*ms(d), len(d))


def pm(m, s, nd=2, sign=False):
    return f"{m:{'+' if sign else ''}.{nd}f} ± {s:.{nd}f}"


def bold(d, se, text):
    return f"**{text}**" if abs(d) > 2 * se else text


def dpm(d, se, nd=2):
    return bold(d, se, pm(d, se, nd, sign=True))


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"


def stream_name(g):
    return {"rounds": "rounds (real)", "rounds3210": "rounds, reversed"}.get(
        g, g.replace("skew", "skew:").replace("drift", "drift:"))


def weight(method):
    m = re.search(r"__w([\d.]+)", method)
    return float(m.group(1)) if m else None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- A, B

def analysis_a():
    df = df_of("E1")
    cells = [("shared", "stable", "skew0.5"), ("shared", "stable", "rounds"), ("shared", "shifting", "skew0.9"),
             ("new", "stable", "aglen"), ("new", "stable", "target"), ("new", "shifting", "agrate")]
    rows = []
    for antigens, rate, g in cells:
        ft_f, ft_a, erm = (cell(df, g, "Finetune", "forgetting"), cell(df, g, "Finetune", "final_acc"),
                           cell(df, g, "ERM", "final_acc"))
        d, se, n = diff(erm, ft_a)
        rows.append([antigens, rate, stream_name(g), pm(*ms(ft_f)), pm(*ms(ft_a)), pm(*ms(erm)), dpm(d, se), n])
    return ("## A. Two-by-two: what makes fine-tuning forget (E1 main, 4,000 steps)\n\n"
            + md(["Antigens", "Binder rate", "Stream", "Fine-tuning forgetting", "Fine-tuning final acc.",
                  "Joint final acc.", "Joint − fine-tuning", "Seeds"], rows))


def analysis_b():
    rows = []
    for root in ("E1", "F3"):
        df = df_of(root)
        for g in sorted(set(df.group) & set(SWING)):
            ft = cell(df, g, "Finetune", "final_acc")
            er, pd_ = diff(cell(df, g, "ER", "final_acc"), ft), diff(cell(df, g, "PDCL_RS", "final_acc"), ft)
            kind = "real rounds" if g.startswith("rounds") else "gradual" if g.startswith("drift") else "alternating"
            rows.append((SWING[g], kind != "real rounds", kind, g,
                         [f"{SWING[g]:.2f}", stream_name(g), kind, root, pm(*ms(cell(df, g, "Finetune", "forgetting"))),
                          dpm(*er[:2]), dpm(*pd_[:2]), er[2]]))
    rows = [r[-1] for r in sorted(rows, key=lambda r: r[:4])]
    return ("## B. Where the real rounds sit on the binder-rate dial (E1 main + F3)\n\n"
            "Swing = highest minus lowest binder rate among the tasks: 2Q − 1 for skew:Q, D for drift:D, "
            "0.20 for the real rounds (binder rates 0.49, 0.59, 0.39, 0.53). Differences are in final accuracy.\n\n"
            + md(["Swing", "Stream", "Dial", "Root", "Fine-tuning forgetting", "ER − fine-tuning",
                  "PDCL − fine-tuning", "Seeds"], rows))


# ---------------------------------------------------------------- C

def analysis_c():
    cells = []
    for root in ROOTS:
        df = df_of(root)
        if df.empty:
            continue
        ft_all = df[df.method == "Finetune"]
        pdcl = df[df.method.str.startswith("PDCL_RS")]
        for (g, m, mem, steps, stream), sub in pdcl.groupby(["group", "method", "memory", "steps", "stream"]):
            ft = ft_all[(ft_all.stream == stream) & (ft_all.steps == steps)]
            if ft.empty:
                continue
            d, se, n = diff(sub.set_index("seed").final_acc, ft.groupby("seed").final_acc.mean())
            tag = m.split("__")[1] if "__" in m else ""
            variant = next((v for v in ("val", "split") if tag.startswith(v)), "default")
            cells.append({"root": root, "group": g, "method": m, "memory": int(mem), "steps": int(steps),
                          "variant": variant, "d": d, "se": se, "n": n})
    c = pd.DataFrame(cells)
    c["z"] = c.d / c.se
    label = lambda r: f"{r.root} {stream_name(r.group)} {r.method} (memory {r.memory}, {r.steps} steps): {pm(r.d, r.se, sign=True)}"
    out = ["## C. PDCL minus fine-tuning, every cell\n",
           "A cell is one stream × PDCL variant × memory size × length inside one root, compared with the "
           "fine-tuning runs of the same root, stream, length and seeds (final accuracy, paired). "
           "`default` = the published rule (the dual variables read the loss of the replayed memory), "
           "including the cells that only change memory size, allocation α, dual step size or ε scale; "
           "`val` and `split` are the F1 variants. Nothing is corrected for the number of cells.\n"]
    rows = []
    for v in ("default", "val", "split"):
        s = c[c.variant == v]
        below, above = s[s.z < -2], s[s.z > 2]
        worst = s.loc[s.d.idxmin()]
        rows.append([v, len(s), len(below), len(above), len(s) - len(below) - len(above), label(worst)])
    out.append(md(["PDCL", "Cells", "> 2 SE below fine-tuning", "> 2 SE above", "Within 2 SE",
                   "Lowest difference"], rows))
    for v in ("default", "val", "split"):
        below = c[(c.variant == v) & (c.z < -2)].sort_values("d")
        out.append(f"Cells more than two standard errors below fine-tuning, `{v}`: "
                   + ("none." if below.empty else "") + "\n")
        out += [f"- {label(r)}" for r in below.itertuples()] + [""]
    by = c.groupby(["variant", "root"]).apply(
        lambda s: pd.Series({"cells": len(s), "below": int((s.z < -2).sum()), "above": int((s.z > 2).sum()),
                             "min": s.d.min(), "median": s.d.median(), "max": s.d.max()}),
        include_groups=False).reset_index()
    out.append(md(["PDCL", "Root", "Cells", "Below", "Above", "Lowest", "Median", "Highest"],
                  [[r.variant, r.root, int(r.cells), int(r.below), int(r.above), f"{r['min']:+.2f}",
                    f"{r['median']:+.2f}", f"{r['max']:+.2f}"] for _, r in by.iterrows()]))
    out.append("All cells:\n")
    c = c.sort_values(["variant", "root", "group", "method"])
    out.append(md(["PDCL", "Root", "Stream", "Method", "Memory", "Steps", "PDCL − fine-tuning", "Seeds"],
                  [[r.variant, r.root, stream_name(r.group), f"`{r.method}`", r.memory, r.steps,
                    dpm(r.d, r.se), r.n] for _, r in c.iterrows()]))
    return "\n".join(out)


# ---------------------------------------------------------------- G1

def curve_gap(cx, cy, px, py):
    """PDCL's last-task accuracy minus the ER curve at PDCL's earlier-task accuracy.

    The curve joins the points in order of replay weight; where it passes PDCL's
    earlier-task accuracy more than once the highest crossing is used. NaN outside its range."""
    ys = [y0 + (y1 - y0) * (px - x0) / (x1 - x0) if x1 != x0 else max(y0, y1)
          for x0, x1, y0, y1 in zip(cx[:-1], cx[1:], cy[:-1], cy[1:]) if min(x0, x1) <= px <= max(x0, x1)]
    return py - max(ys) if ys else float("nan")


def g1_sources():
    """(label, table, {name: method}, ER methods in order of weight, table holding joint training)."""
    e2, f1, e1 = df_of("E2"), df_of("F1"), df_of("E1")
    ers = sorted((m for m in e2.method.unique() if weight(m) is not None), key=weight)
    yield "E2", e2, {"PDCL": "PDCL_RS"}, ers, e2
    ers = sorted((m for m in f1.method.unique() if weight(m) is not None and m.endswith("buf200")), key=weight)
    yield "F1", f1, {"PDCL": "PDCL_RS__memory_buf200", "PDCL val": "PDCL_RS__val_buf200",
                     "PDCL split": "PDCL_RS__split_buf200"}, ers, e1


def analysis_g1():
    rng = np.random.default_rng(0)
    points, verdicts = [], []
    for src, df, pdcls, ers, erm_df in g1_sources():
        for g in sorted(df.group.unique(), key=lambda g: (SWING.get(g, 9), g)):
            get = lambda m, col, d=df: cell(d, g, m, col)
            curve = ["Finetune"] + ers
            for m in curve + list(pdcls.values()):
                points.append({"src": src, "group": g, "method": m, "x": ms(get(m, "earlier_acc")),
                               "y": ms(get(m, "last_acc"))})
            points.append({"src": src, "group": g, "method": "ERM", "x": ms(get("ERM", "earlier_acc", erm_df)),
                           "y": ms(get("ERM", "last_acc", erm_df))})
            X = np.array([get(m, "earlier_acc").values for m in curve])
            Y = np.array([get(m, "last_acc").values for m in curve])
            for name, m in pdcls.items():
                px, py = get(m, "earlier_acc").values, get(m, "last_acc").values
                gap = curve_gap(X.mean(1), Y.mean(1), px.mean(), py.mean())
                idx = rng.integers(0, len(px), (N_BOOT, len(px)))
                boot = np.array([curve_gap(X[:, i].mean(1), Y[:, i].mean(1), px[i].mean(), py[i].mean())
                                 for i in idx])
                se = float(np.nanstd(boot, ddof=1)) if np.isfinite(boot).sum() > 1 else float("nan")
                dominated = [f"{weight(c):g}" for c, x, y in zip(curve[1:], X.mean(1)[1:], Y.mean(1)[1:])
                             if x >= px.mean() and y >= py.mean()]
                if np.isnan(gap):
                    side = "more" if px.mean() > X.mean(1).max() else "less"
                    j = int(X.mean(1).argmax() if side == "more" else X.mean(1).argmin())
                    end = "fine-tuning" if j == 0 else f"ER weight {weight(curve[j]):g}"
                    verdict = (f"outside the curve (PDCL keeps {side} earlier-task accuracy than any point of it; "
                               f"against its nearest end, {end}: earlier {dpm(*ms(px - X[j]))}, "
                               f"last {dpm(*ms(py - Y[j]))})")
                else:
                    verdict = "above" if gap > 2 * se else "below" if gap < -2 * se else "on"
                verdicts.append({"src": src, "group": g, "name": name, "px": ms(px), "py": ms(py), "gap": gap,
                                 "se": se, "inside": float(np.isfinite(boot).mean()), "verdict": verdict,
                                 "dominated": ", ".join(dominated) or "none"})
    out = ["## G1 (exploratory). Last task against earlier tasks at the end of training\n",
           "Earlier = mean final validation accuracy of tasks 0–2; last = final validation accuracy of task 3. "
           "The ER curve joins fine-tuning (replay weight 0) and the fixed replay weights in order of weight, "
           "all at memory 200. The gap is PDCL's last-task accuracy minus the curve, interpolated linearly at "
           f"PDCL's earlier-task accuracy; its standard error is a bootstrap over seeds ({N_BOOT} resamples, the "
           "same seeds drawn for every method), and `inside` is the share of resamples in which PDCL fell inside "
           "the curve's range. Above/below = more than two standard errors. `Dominated by` lists the replay "
           "weights whose mean is at least as high as PDCL's on both axes. F1 has no joint-training runs; its "
           "joint row is taken from E1 main (same stream and seeds).\n"]
    out.append(md(["Root", "Stream", "PDCL", "Earlier acc.", "Last acc.", "Last − ER curve", "Inside",
                   "Verdict", "Dominated by ER weight"],
                  [[v["src"], stream_name(v["group"]), v["name"], pm(*v["px"]), pm(*v["py"]),
                    "–" if np.isnan(v["gap"]) else dpm(v["gap"], v["se"]), f"{v['inside']:.2f}", v["verdict"],
                    v["dominated"]] for v in verdicts]))
    vd = pd.DataFrame(verdicts)
    counts = vd.assign(v=vd.verdict.str.split(" \\(").str[0]).groupby(["src", "name", "v"]).size().unstack(fill_value=0)
    out.append("Count of streams by verdict:\n\n" + md(["Root", "PDCL"] + list(counts.columns),
                                                        [list(i) + list(r) for i, r in zip(counts.index, counts.values)]))
    for src in ("E2", "F1"):
        p = [q for q in points if q["src"] == src]
        methods = list(dict.fromkeys(q["method"] for q in p))
        groups = list(dict.fromkeys(q["group"] for q in p))
        look = {(q["group"], q["method"]): q for q in p}
        out.append(f"{src}, memory 200: earlier-task accuracy / last-task accuracy\n")
        out.append(md(["Method"] + [stream_name(g) for g in groups],
                      [[f"`{m}`"] + [f"{pm(*look[g, m]['x'], nd=1)} / {pm(*look[g, m]['y'], nd=1)}" for g in groups]
                       for m in methods]))
    fig_g1(pd.DataFrame(points))
    return "\n".join(out)


def fig_g1(points):
    pa.style()
    fig, axes = plt.subplots(2, 6, figsize=(180 * pa.MM, 88 * pa.MM))
    marks = {"Finetune": ("Fine-tuning", "s", "black"), "PDCL_RS": ("PDCL", "o", "#d62728"),
             "PDCL_RS__memory_buf200": ("PDCL", "o", "#d62728"), "PDCL_RS__val_buf200": ("PDCL val", "^", "#ff7f0e"),
             "PDCL_RS__split_buf200": ("PDCL split", "v", "#9467bd"), "ERM": ("Joint", "*", "#2ca02c")}
    for row, src in zip(axes, ("E2", "F1")):
        p = points[points.src == src]
        groups = list(dict.fromkeys(p.group))
        for ax, g in zip(row, groups):
            q = p[p.group == g]
            er = q[q.method.map(lambda m: weight(m) is not None)]
            ft = q[q.method == "Finetune"]
            line = pd.concat([ft, er])
            ax.plot([x[0] for x in line.x], [y[0] for y in line.y], "-", color="#1f77b4", linewidth=1, zorder=2)
            ax.errorbar([x[0] for x in er.x], [y[0] for y in er.y], xerr=[x[1] for x in er.x],
                        yerr=[y[1] for y in er.y], fmt="o", color="#1f77b4", markersize=2.5, elinewidth=0.6,
                        label="ER, fixed weight", zorder=3)
            for r in (er.iloc[0], er.iloc[-1]):
                ax.annotate(f"{weight(r.method):g}", (r.x[0], r.y[0]), textcoords="offset points", xytext=(3, 3),
                            fontsize=6, color="#1f77b4")
            for m, (lab, mk, col) in marks.items():
                for r in q[q.method == m].itertuples():
                    ax.errorbar(r.x[0], r.y[0], xerr=r.x[1], yerr=r.y[1], fmt=mk, color=col, elinewidth=0.6,
                                markersize=6 if mk == "*" else 4, label=lab, zorder=4)
            ax.set_title(f"{src}: {stream_name(g)}", fontsize=8)
            ax.grid(color="gray", alpha=0.5, linewidth=0.4)
            ax.tick_params(labelsize=7)
        for ax in row[len(groups):]:
            ax.axis("off")
    fig.supxlabel("Accuracy on the Earlier Tasks at the End of Training (%)", fontsize=8)
    fig.supylabel("Accuracy on the Last Task (%)", fontsize=8, x=0.008)
    handles = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    axes[0, 5].legend(handles.values(), handles.keys(), loc="center", fontsize=7)
    fig.tight_layout(pad=0.5, w_pad=0.6, h_pad=1.0, rect=(0.02, 0.03, 1, 1))
    save(fig, "fig_g1_tradeoff")


# ---------------------------------------------------------------- G2

def next_table(df, methods, ref="Finetune", prefix="next"):
    rows = []
    for k in range(3):
        base = cell(df, "rounds", ref, f"{prefix}_{k}")
        rows.append([f"{k + 1} → round {k + 2}"] + [
            pm(*ms(cell(df, "rounds", m, f"{prefix}_{k}"))) if m == ref
            else pm(*ms(cell(df, "rounds", m, f"{prefix}_{k}"))) + " (" + dpm(*diff(cell(df, "rounds", m, f"{prefix}_{k}"), base)[:2]) + ")"
            for m in methods])
    mean23 = lambda m: (cell(df, "rounds", m, f"{prefix}_1") + cell(df, "rounds", m, f"{prefix}_2")) / 2
    rows.append(["mean of 2 and 3"] + [pm(*ms(mean23(m))) if m == ref
                                         else pm(*ms(mean23(m))) + " (" + dpm(*diff(mean23(m), mean23(ref))[:2]) + ")"
                                         for m in methods])
    return md(["After round"] + [f"`{m}`" for m in methods], rows)


def analysis_g2():
    e1, e2, f1 = df_of("E1"), df_of("E2"), df_of("F1")
    trio = ["Finetune", "ER", "PDCL_RS"]
    out = ["## G2 (exploratory). Accuracy on the next round before training on it (real rounds)\n",
           "Validation accuracy on round k+1 at the checkpoint that ends the phase of round k (steps 1000, 2000, "
           "3000). That checkpoint has taken one update of round k+1 (see the header of the script); the "
           "`strict` table uses the last checkpoint before it (steps 750, 1750, 2750) instead. Joint training "
           "(ERM) draws from all four rounds from the first step, so it has always seen round k+1 and is not a "
           "valid \"rounds seen so far\" baseline; it is left out. In the first phase no method has anything to "
           "replay, so the three methods are the same model there and differ only from the second phase on. "
           "Roots are not pooled; E1 main is the primary table. In brackets: difference from fine-tuning, "
           "paired by seed.\n", "E1 main (primary), memory 200:\n", next_table(e1, trio)]
    rows = []
    for k in list(range(3)) + ["mean of 2 and 3"]:
        col = lambda m: (cell(e1, "rounds", m, f"next_{k}") if k != "mean of 2 and 3" else
                         (cell(e1, "rounds", m, "next_1") + cell(e1, "rounds", m, "next_2")) / 2)
        rows.append([k if isinstance(k, str) else f"{k + 1} → round {k + 2}"]
                    + [dpm(*diff(col(a), col(b))[:2]) for a, b in (("ER", "Finetune"), ("PDCL_RS", "Finetune"),
                                                                    ("PDCL_RS", "ER"))])
    out += ["E1 main, paired differences:\n",
            md(["After round", "ER − fine-tuning", "PDCL − fine-tuning", "PDCL − ER"], rows),
            "E1 main, strict (last checkpoint before the phase ends):\n", next_table(e1, trio, prefix="next_strict")]
    e2_m = ["Finetune"] + sorted((m for m in e2.method.unique() if weight(m) is not None), key=weight) + ["PDCL_RS"]
    f1_m = ["Finetune"] + sorted((m for m in f1.method.unique() if weight(m) is not None and m.endswith("buf200")),
                                 key=weight) + [f"PDCL_RS__{v}_buf200" for v in ("memory", "val", "split")]
    out += ["E2 (secondary), memory 200:\n", next_table(e2, e2_m),
            "F1 (secondary), memory 200:\n", next_table(f1, f1_m)]
    fig_g2(e1, e2, trio, e2_m)
    return "\n".join(out)


def fig_g2(e1, e2, trio, e2_m):
    pa.style()
    fig, axes = pa.panels(3, h=62)
    look = {"Finetune": ("Fine-tuning", "s", "black"), "ER": ("ER", "o", "#1f77b4"), "PDCL_RS": ("PDCL", "o", "#d62728")}
    x = np.arange(3)
    for i, m in enumerate(trio):
        lab, mk, col = look[m]
        v = [ms(cell(e1, "rounds", m, f"next_{k}")) for k in range(3)]
        axes[0].errorbar(x + 0.15 * (i - 1), [a for a, _ in v], yerr=[s for _, s in v], fmt=mk + "-", color=col,
                         label=lab, linewidth=0.8)
        if m != "Finetune":
            d = [diff(cell(e1, "rounds", m, f"next_{k}"), cell(e1, "rounds", "Finetune", f"next_{k}")) for k in range(3)]
            axes[1].errorbar(x + 0.1 * (i - 1.5), [a[0] for a in d], yerr=[a[1] for a in d], fmt=mk + "-", color=col,
                             label=lab, linewidth=0.8)
    axes[0].set_ylabel("Accuracy on the Next Round (%)")
    axes[1].set_ylabel("Difference from Fine-Tuning (points)")
    axes[1].axhline(0, color="black", linewidth=0.6)
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(["1 → 2", "2 → 3", "3 → 4"])
        ax.set_xlabel("Rounds Seen → Round Tested (E1)")
        pa.top_legend(ax, ncol=3)
    mean23 = lambda m: (cell(e2, "rounds", m, "next_1") + cell(e2, "rounds", m, "next_2")) / 2
    ers = [m for m in e2_m if weight(m) is not None]
    d = [diff(mean23(m), mean23("Finetune")) for m in ers]
    axes[2].errorbar([weight(m) for m in ers], [a[0] for a in d], yerr=[a[1] for a in d], fmt="o-", color="#1f77b4",
                     label="ER, fixed weight", linewidth=0.8)
    p = diff(mean23("PDCL_RS"), mean23("Finetune"))
    axes[2].axhspan(p[0] - p[1], p[0] + p[1], color="#d62728", alpha=0.2, linewidth=0)
    axes[2].axhline(p[0], color="#d62728", linewidth=1, label="PDCL")
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Replay Weight (E2; Mean of 2 → 3 and 3 → 4)")
    axes[2].set_ylabel("Difference from Fine-Tuning (points)")
    pa.top_legend(axes[2], ncol=2)
    fig.tight_layout(pad=0.6, w_pad=2.0)
    save(fig, "fig_g2_next_round")


# ---------------------------------------------------------------- G3

def kept_table(df, methods, groups):
    rows = []
    for m in methods:
        row = [f"`{m}`"]
        for g in groups:
            v, base = cell(df, g, m, "kept_share"), cell(df, g, "Finetune", "kept_share")
            d, se, _ = diff(v, base)
            row.append(pm(*ms(v), nd=0) if m == "Finetune" else bold(d, se, pm(*ms(v), nd=0)))
        allv = df[df.method == m].groupby("seed").kept_share.mean()
        base = df[df.method == "Finetune"].groupby("seed").kept_share.mean()
        d, se, _ = diff(allv, base)
        row.append(pm(*ms(allv), nd=0) if m == "Finetune" else pm(*ms(allv), nd=0) + f" ({dpm(d, se, 0)})")
        rows.append(row)
    return md(["Method"] + [stream_name(g) for g in groups] + ["All streams (− fine-tuning)"], rows)


def analysis_g3():
    have = {r: "kept_share" in df_of(r).columns and df_of(r).kept_share.notna().any() for r in ROOTS}
    out = ["## G3 (exploratory). Constraints on validation data\n",
           f"Per run: the share (%) of earlier tasks (0–2) whose final validation loss is at most {SLACK} × its "
           "validation loss at the end of that task's own phase; mean ± standard error over seeds. Bold = differs "
           "from fine-tuning on the same stream by more than two standard errors (paired); the last column averages "
           "each seed over the streams first. "
           "Per-task validation loss is logged only in " + " and ".join(r for r in ROOTS if have[r])
           + (" (`val_loss_k` in F1, `env{k}_out_loss` in F2); " if have["F1"] else "; ")
           + ", ".join(r for r in ROOTS if not have[r]) + " log validation accuracy and the memory's loss only, so "
           "E1 main and the E2 fixed weights cannot be computed. F1 holds the default PDCL (`memory`), the fixed "
           "replay weights and both variants at memory 200 and 1000; F2 has replay and fine-tuning only.\n"]
    f1 = df_of("F1")
    groups = sorted(f1.group.unique(), key=lambda g: (SWING.get(g, 9), g))
    for mem in (200, 1000):
        ms_ = ["Finetune"] + sorted((m for m in f1.method.unique() if weight(m) is not None and m.endswith(f"buf{mem}")),
                                    key=weight) + [f"PDCL_RS__{v}_buf{mem}" for v in ("memory", "val", "split")]
        out += [f"F1, memory {mem}:\n", kept_table(f1, ms_, groups)]
    pairs = []
    for mem in (200, 1000):
        for a, b in ((f"PDCL_RS__val_buf{mem}", f"PDCL_RS__memory_buf{mem}"), (f"PDCL_RS__memory_buf{mem}", f"ER__w1_buf{mem}"),
                     (f"PDCL_RS__val_buf{mem}", f"ER__w1_buf{mem}"), (f"PDCL_RS__val_buf{mem}", f"ER__w3_buf{mem}")):
            share = lambda m: f1[f1.method == m].groupby("seed").kept_share.mean()
            pairs.append([f"`{a}` − `{b}`", dpm(*diff(share(a), share(b))[:2], 0)])
    out += ["F1, all streams, paired differences in the share (points):\n", md(["Pair", "Difference"], pairs)]
    f2 = df_of("F2")
    if have["F2"]:
        out += ["F2 (replay batch and weight decay; no PDCL):\n",
                kept_table(f2, ["Finetune"] + sorted(m for m in f2.method.unique() if m != "Finetune"),
                           sorted(f2.group.unique(), key=lambda g: (SWING.get(g, 9), g)))]
    return "\n".join(out)


# ---------------------------------------------------------------- main

def coverage():
    rows = []
    for r in ROOTS:
        df, started = load(r)
        rows.append([r, f"`{ROOTS[r].relative_to(TREES)}`", len(df), started, df.group.nunique(),
                     ", ".join(str(s) for s in sorted(df.steps.unique())), df.seed.nunique()])
    return ("Runs used (whole-stream runs with a `done` file):\n\n"
            + md(["Root", "Path", "Finished runs", "Run folders", "Streams", "Steps", "Seeds"], rows))


def main():
    parts = ["# Analyses that need no new runs\n",
             "Written by `python -m domainbed.scripts.free_analyses`. Mean ± standard error over seeds; differences "
             "paired by seed, bold above two standard errors; accuracies in percent. Final accuracy = mean over "
             "tasks of the last checkpoint's validation accuracy; forgetting as in `stream_analysis.run_row`. "
             "A–C answer scope questions; G1–G3 are exploratory and were not pre-registered.\n",
             coverage(), analysis_a(), analysis_b(), analysis_c(), analysis_g1(), analysis_g2(), analysis_g3()]
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
