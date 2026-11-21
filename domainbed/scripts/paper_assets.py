"""Numbers, tables and figures for the paper, computed from outputs/b1*.

Usage: python -m domainbed.scripts.paper_assets
Writes paper/generated/{numbers.tex,tab_*.tex} and paper/figures/fig_*.pdf.
"""
import json
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOTS = {750: "outputs/b1_s750", 1500: "outputs/b1_s1500", 3000: "outputs/b1_short",
         6000: "outputs/b1_s6000", 15000: "outputs/b1", 30000: "outputs/b1_s30000"}
STREAM = {4000: "outputs/b1_stream_short", 20000: "outputs/b1_stream"}
SEQ = ("Finetune", "ER", "ER_FF", "PDCL_RS", "PDCL")
JOINT = ("ERM", "IRM", "VREx", "GroupDRO", "CORAL", "SD")
DG = JOINT[1:]
LABEL = {"ERM": "Joint ERM", "IRM": "IRM", "VREx": "V-REx", "GroupDRO": "GroupDRO", "CORAL": "CORAL",
         "SD": "SD", "Finetune": "Fine-tuning", "ER": "Replay (ER)", "ER_FF": "ER + FF",
         "PDCL_RS": "PDCL", "PDCL": "PDCL + FF"}
LOW = {"ERM": "ERM__lr1.5e-05_wd0", "Finetune": "Finetune__plr1.5e-05", "ER": "ER__plr1.5e-05",
       "PDCL_RS": "PDCL_RS__plr1.5e-05_es1.1"}
CYAN, RED, ORANGE, YELLOW, NIGHT, GREY = "#8FBBBA", "#BE616A", "#CF8770", "#EBCB8B", "#2e3440", "#4c566a"
COLOR = {"ERM": NIGHT, "Finetune": ORANGE, "ER": CYAN, "PDCL_RS": RED}
GEN, FIG = Path("paper/generated"), Path("paper/figures")
N = {}


@lru_cache(None)
def load(root, alg):
    runs = {}
    for d in sorted(Path(root, alg).glob("env*_seed*")):
        if (d / "done").exists():
            e, s = d.name.split("_")
            runs[int(e[3:]), int(s[4:])] = [json.loads(l) for l in open(d / "results.jsonl")]
    return runs


def eligible(recs, alg):
    # a sequential model must have trained on every round: only checkpoints inside the last phase
    if alg.split("__")[0] in SEQ:
        n = recs[-1]["step"] + 1
        d = 3 if recs[0]["args"]["test_envs"] else 4
        return [r for r in recs if r["step"] > (d - 1) * (n // d)]
    return recs


def pick(recs, alg, rule):
    el = eligible(recs, alg)
    if rule == "val":
        return max(el, key=lambda r: r["agg_val_acc"])
    if rule == "vloss":
        return min(el, key=lambda r: r["agg_val_loss"])
    if rule == "last":
        return recs[-1]
    return max(el, key=lambda r: r["agg_test_acc"])


def seed_means(root, alg, rule, key="agg_test_acc", scale=100.0):
    runs = load(root, alg)
    envs = {e for e, _ in runs}
    out = {}
    for s in sorted({s for _, s in runs}):
        if all((e, s) in runs for e in envs):
            out[s] = np.mean([scale * pick(runs[e, s], alg, rule)[key] for e in envs])
    return out


def mse(d):
    v = np.array(list(d.values()) if isinstance(d, dict) else d, dtype=float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)


def paired(a, b):
    k = sorted(set(a) & set(b))
    return mse([a[s] - b[s] for s in k])


def acc(b, alg, rule):
    return mse(seed_means(ROOTS[b], alg, rule))


def diff(b, a1, a2, rule):
    return paired(seed_means(ROOTS[b], a1, rule), seed_means(ROOTS[b], a2, rule))


def pm(m, s, nd=1):
    return f"${m:.{nd}f}\\pm{s:.{nd}f}$"


def spm(m, s, nd=1):
    return f"${m:+.{nd}f}\\pm{s:.{nd}f}$"


def trajectory(root, alg, key, scale=1.0):
    runs = load(root, alg)
    if key == "train_acc":
        # accuracy on the training splits of the training rounds
        runs = {(e, s): [{key: np.mean([r[f"env{i}_in_acc"] for i in range(4) if i != e])} | {"step": r["step"]}
                         for r in recs] for (e, s), recs in runs.items()}
    steps = np.array([r["step"] + 1 for r in next(iter(runs.values()))])
    seeds = sorted({s for _, s in runs})
    per_seed = np.array([np.mean([[scale * r[key] for r in runs[e, s]] for e, t in runs if t == s], axis=0)
                         for s in seeds])
    return steps, per_seed.mean(0), per_seed.std(0, ddof=1) / np.sqrt(len(seeds))


def end_stats(b, alg):
    """Quantities at the last step: accuracy on the final round, memory loss, constraint level, duals."""
    out = {k: [] for k in ("train", "val", "eps", "mem", "dual")}
    for (e, s), recs in load(ROOTS[b], alg).items():
        r = recs[-1]
        last = [x for x in range(4) if x != e][-1]
        out["train"].append(100 * r[f"env{last}_in_acc"])
        out["val"].append(100 * r[f"env{last}_out_acc"])
        for name, prefix in (("eps", "eps_"), ("mem", "replay_loss_"), ("dual", "dual_")):
            out[name] += [v for k, v in r.items() if k.startswith(prefix)]
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}


def run_metrics(rows, test_env):
    """Retention on the validation splits of the training rounds, in training order."""
    train_envs = [e for e in range(4) if e != test_env]
    n = len(train_envs)
    steps = np.array([r["step"] for r in rows])
    phase = (steps[-1] + 1) // n
    acc = [np.array([r[f"env{e}_out_acc"] for r in rows]) for e in train_envs]
    at = lambda k, step: acc[k][np.argmin(np.abs(steps - step))]
    # forgetting of round k: best accuracy at the end of a phase, from its own on,
    # minus final accuracy (Elenter et al., 2023)
    forgetting = [max(at(k, l * phase) for l in range(k + 1, n)) - acc[k][-1] for k in range(n - 1)]
    return {"forgetting": float(np.mean(forgetting)), "final_val": float(np.mean([a[-1] for a in acc]))}


def stream_metrics(root, alg):
    out = {}
    for (e, s), recs in load(root, alg).items():
        out[s] = {k: 100 * v for k, v in run_metrics(recs, e).items()}
    return out


def numbers():
    b0 = 15000
    steps, test, _ = trajectory(ROOTS[b0], "ERM", "agg_test_acc", 100)
    _, val, _ = trajectory(ROOTS[b0], "ERM", "agg_val_acc", 100)
    _, vl, _ = trajectory(ROOTS[b0], "ERM", "agg_val_loss")
    i = int(test.argmax())
    late = steps >= 1500
    N["erm_train_end"] = f"{trajectory(ROOTS[b0], 'ERM', 'train_acc', 100)[1][-1]:.0f}"
    N.update(erm_peak_test=f"{test[i]:.1f}", erm_peak_step=f"{int(round(steps[i], -2)):,}".replace(",", "{,}"),
             erm_test_end=f"{test[-1]:.1f}", erm_drop=f"{test[i] - test[-1]:.1f}",
             erm_val_lo=f"{val[late].min():.1f}", erm_val_hi=f"{val[late].max():.1f}",
             erm_vloss_min=f"{vl.min():.2f}", erm_vloss_end=f"{vl[-1]:.2f}",
             erm_vloss_min_step=f"{int(round(steps[int(vl.argmin())], -2)):,}".replace(",", "{,}"))
    for b in (15000, 30000):
        runs = load(ROOTS[b], "ERM")
        for rule in ("val", "vloss", "oracle"):
            st = np.mean([pick(r, "ERM", rule)["step"] + 1 for r in runs.values()])
            N[f"erm_step_{rule}_{b}"] = f"{int(round(st, -2)):,}".replace(",", "{,}")
    for b in ROOTS:
        algs = JOINT + SEQ if b != 30000 else ("ERM",) + SEQ
        for a in algs:
            for rule in ("val", "vloss", "last", "oracle"):
                m, s, _ = acc(b, a, rule)
                N[f"acc_{a}_{rule}_{b}"] = f"{m:.1f}"
        for rule in ("val", "vloss"):
            pairs = [("PDCL_RS", "Finetune"), ("PDCL_RS", "ER"), ("ER", "Finetune"), ("PDCL", "PDCL_RS"),
                     ("ER_FF", "ER"), ("PDCL_RS", "ERM"), ("Finetune", "ERM")]
            if b != 30000:
                pairs += [(a, "ERM") for a in DG]
            for a1, a2 in pairs:
                N[f"d_{a1}_{a2}_{rule}_{b}"] = spm(*diff(b, a1, a2, rule)[:2], nd=2)
        e = seed_means(ROOTS[b], "ERM", "vloss"), seed_means(ROOTS[b], "ERM", "val")
        N[f"d_rule_{b}"] = spm(*paired(*e)[:2], nd=2)
    for b in (3000, 15000):
        for a in ("Finetune", "ER", "PDCL_RS"):
            st = end_stats(b, a)
            N[f"train_{a}_{b}"], N[f"val_{a}_{b}"] = f"{st['train']:.0f}", f"{st['val']:.0f}"
            N[f"mem_{a}_{b}"], N[f"eps_{a}_{b}"] = f"{st['mem']:.3f}", f"{st['eps']:.3f}"
            N[f"dual_{a}_{b}"] = f"{st['dual']:.3f}"
    # lower learning rate at the long budget
    low = {a: seed_means(ROOTS[15000], t, "vloss") for a, t in LOW.items()}
    for a, t in LOW.items():
        m, s, n = mse(low[a])
        N[f"low_{a}"], N[f"low_n"] = f"{m:.1f}", str(n)
        N[f"low_gain_{a}"] = spm(*paired(low[a], seed_means(ROOTS[15000], a, "vloss"))[:2], nd=2)
    N["low_PDCL_RS_Finetune"] = spm(*paired(low["PDCL_RS"], low["Finetune"])[:2], nd=2)
    N["low_ER_Finetune"] = spm(*paired(low["ER"], low["Finetune"])[:2], nd=2)
    N["low_Finetune_ERM"] = spm(*paired(low["Finetune"], low["ERM"])[:2], nd=2)
    # other variants at the long budget (seeds 0-4), difference to the default configuration
    for tag in ("PDCL_RS__buf1000", "PDCL_RS__buf3000", "ER__buf1000", "ER__buf3000", "PDCL_RS__eps0.05",
                "PDCL_RS__dlr0.001", "PDCL_RS__dlr0.1", "PDCL_RS__plr5e-05_es2"):
        base = tag.split("__")[0]
        d = paired(seed_means(ROOTS[15000], tag, "vloss"), seed_means(ROOTS[15000], base, "vloss"))
        N["var_" + tag.replace("__", "_").replace(".", "p").replace("-", "m")] = spm(*d[:2], nd=2)
    N["low_gain_val_ERM"] = spm(*paired(seed_means(ROOTS[15000], LOW["ERM"], "val"), seed_means(ROOTS[15000], "ERM", "val"))[:2], nd=2)
    for a, tags in (("Finetune", ("plr1.5e-05", "plr0.00015")), ("ER", ("plr1.5e-05", "plr0.00015")),
                    ("PDCL_RS", ("plr1.5e-05_es1.1", "plr0.00015_es1.1"))):
        for name, tag in zip(("lo", "hi"), tags):
            d = paired(seed_means(ROOTS[3000], f"{a}__{tag}", "vloss"), seed_means(ROOTS[3000], a, "vloss"))
            N[f"short_{name}_{a}"] = spm(*d[:2], nd=2)
    good = (750, 1500, 3000, 6000)
    rng = lambda v: f"{min(v):.1f}--{max(v):.1f}"
    dm = lambda b, a1, a2, rule: diff(b, a1, a2, rule)[0]
    N["pdcl_ft_absmax"] = f"{max(abs(dm(b, 'PDCL_RS', 'Finetune', 'vloss')) for b in ROOTS):.1f}"
    N["er_ft_range"] = rng([-dm(b, "ER", "Finetune", "vloss") for b in good])
    N["pdcl_er_range"] = rng([dm(b, "PDCL_RS", "ER", "vloss") for b in good])
    N["ft_erm_range"] = rng([-dm(b, "Finetune", "ERM", "vloss") for b in good])
    N["pdcl_erm_range"] = rng([-dm(b, "PDCL_RS", "ERM", "vloss") for b in good])
    N["ft_erm_long_range"] = rng([-dm(b, "Finetune", "ERM", "vloss") for b in (15000, 30000)])
    three = ("VREx", "GroupDRO", "CORAL")
    N["dg_long_val_range"] = rng([dm(15000, a, "ERM", "val") for a in three])
    N["dg_short_val_absmax"] = f"{max(abs(dm(b, a, 'ERM', 'val')) for b in good for a in three + ('SD',)):.1f}"
    N["dg_long_vloss_absmax"] = f"{max(abs(dm(15000, a, 'ERM', 'vloss')) for a in three + ('SD',)):.1f}"
    N["erm_vloss_range"] = rng([acc(b, "ERM", "vloss")[0] for b in ROOTS if b >= 1500])
    N["erm_val_range_long"] = rng([acc(b, "ERM", "val")[0] for b in (15000, 30000)])
    N["ffpdcl_range"] = f"${min(dm(b, 'PDCL', 'PDCL_RS', 'vloss') for b in ROOTS):+.1f}$ to ${max(dm(b, 'PDCL', 'PDCL_RS', 'vloss') for b in ROOTS):+.1f}$"
    N["ffer_range"] = f"${min(dm(b, 'ER_FF', 'ER', 'vloss') for b in ROOTS):+.1f}$ to ${max(dm(b, 'ER_FF', 'ER', 'vloss') for b in ROOTS):+.1f}$"
    N["stream_seq_final_range"] = rng([mse([v["final_val"] for v in stream_metrics(r, a).values()])[0] for r in STREAM.values() for a in SEQ])
    N["stream_seq_forget_range"] = rng([mse([v["forgetting"] for v in stream_metrics(r, a).values()])[0] for r in STREAM.values() for a in SEQ])
    N["mem_ratio_3000"] = f"{end_stats(3000, 'PDCL_RS')['mem'] / end_stats(3000, 'ER')['mem']:.0f}"
    for b, root in STREAM.items():
        sm = {a: stream_metrics(root, a) for a in ("ERM",) + SEQ}
        for a in sm:
            for k in ("forgetting", "final_val"):
                m, s, _ = mse([v[k] for v in sm[a].values()])
                N[f"stream_{k}_{a}_{b}"] = f"{m:.1f}"
        for a in ("Finetune", "PDCL_RS"):
            d = paired({s: v["forgetting"] for s, v in sm[a].items()}, {s: v["forgetting"] for s, v in sm["ER"].items()})
            N[f"stream_forget_{a}_ER_{b}"] = spm(*d[:2])
    N["n_runs"] = f"{sum(len(list(Path(r).glob('*/env*_seed*/done'))) for r in list(ROOTS.values()) + list(STREAM.values())):,}".replace(",", "{,}")
    GEN.mkdir(parents=True, exist_ok=True)
    for old in GEN.glob("*"):
        old.unlink()
    (GEN / "numbers.tex").write_text("".join(
        f"\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n" for k, v in N.items()))


def tabular(cols, header, rows):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"


def tables():
    rows = []
    for a in JOINT:
        r = [LABEL[a]]
        for b in (3000, 15000):
            for rule in ("val", "vloss"):
                r.append(pm(*acc(b, a, rule)[:2]))
        rows.append(r)
    head = (r" & \multicolumn{2}{c}{3{,}000 steps} & \multicolumn{2}{c}{15{,}000 steps} \\" "\n"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
            r"Method & Val.\ accuracy & Val.\ loss & Val.\ accuracy & Val.\ loss")
    (GEN / "tab_joint.tex").write_text(tabular("lcccc", head, rows))

    rows = []
    for a in ("ERM",) + SEQ:
        rows.append([LABEL[a]] + [pm(*acc(b, a, "vloss")[:2]) for b in ROOTS])
        if a == "ERM":
            rows[-1][-1] += r" \\ \midrule %"
    rows[0][-1] = rows[0][-1].replace(r" \\ \midrule %", "")
    body_rows = [rows[0]] + rows[1:]
    head = "Method & " + " & ".join(f"{b:,}".replace(",", "{,}") for b in ROOTS)
    t = tabular("lcccccc", head, body_rows).replace(LABEL["Finetune"] + " &", "\\midrule\n" + LABEL["Finetune"] + " &", 1)
    (GEN / "tab_seq.tex").write_text(t)

    rows = []
    for a in ("ERM",) + SEQ:
        r = [LABEL[a]]
        for b, root in STREAM.items():
            sm = stream_metrics(root, a)
            for k in ("final_val", "forgetting"):
                r.append(pm(*mse([v[k] for v in sm.values()])[:2]))
        rows.append(r)
    head = (r" & \multicolumn{2}{c}{1{,}000 steps per round} & \multicolumn{2}{c}{5{,}000 steps per round} \\" "\n"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
            r"Method & Final accuracy & Forgetting & Final accuracy & Forgetting")
    (GEN / "tab_stream.tex").write_text(tabular("lcccc", head, rows))


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "axes.axisbelow": True, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 1.3, "lines.markersize": 4,
                         "legend.frameon": False})


MM = 1 / 25.4


def panels(n, h=70):
    fig, axes = plt.subplots(1, n, figsize=(180 * MM, h * MM), squeeze=False)
    for ax, letter in zip(axes[0], "abc"):
        ax.grid(color="gray", alpha=0.7, linewidth=0.4, zorder=0)
        ax.text(-0.2, 1.12, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)
    return fig, axes[0]


def top_legend(ax, ncol=2, **kw):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, columnspacing=1.2, handlelength=1.8, **kw)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6, w_pad=2.5)
    fig.savefig(FIG / f"{name}.pdf", dpi=600)
    plt.close(fig)


def budget_axis(ax, budgets):
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b / 1000:g}" for b in budgets])
    ax.minorticks_off()
    ax.set_xlabel("Total Training Steps (×10³)")


def fig_trajectory():
    root = ROOTS[15000]
    fig, (a, b) = panels(2)
    for key, color, label in (("train_acc", GREY, "Training"), ("agg_val_acc", CYAN, "Validation"),
                              ("agg_test_acc", RED, "Held-Out Round")):
        steps, m, s = trajectory(root, "ERM", key, 100)
        a.plot(steps / 1000, m, color=color, label=label, zorder=3)
        a.fill_between(steps / 1000, m - s, m + s, color=color, alpha=0.25, linewidth=0, zorder=2)
    a.set_ylabel("Accuracy (%)")
    steps, m, s = trajectory(root, "ERM", "agg_val_loss")
    b.plot(steps / 1000, m, color=NIGHT, zorder=3)
    b.fill_between(steps / 1000, m - s, m + s, color=NIGHT, alpha=0.2, linewidth=0, zorder=2)
    b.set_ylabel("Validation Loss")
    runs = load(root, "ERM")
    top_legend(a, ncol=3)
    for rule, style, label in (("val", "--", "Picked by Val. Accuracy"), ("vloss", ":", "Picked by Val. Loss")):
        st = np.mean([pick(r, "ERM", rule)["step"] + 1 for r in runs.values()]) / 1000
        a.axvline(st, color="black", linestyle=style, linewidth=1, zorder=2)
        b.axvline(st, color="black", linestyle=style, linewidth=1, zorder=2, label=label)
    for ax in (a, b):
        ax.set_xlabel("Training Step (×10³)")
    top_legend(b)
    save(fig, "fig_trajectory")


def fig_selection():
    fig, (a, b) = panels(2)
    budgets = list(ROOTS)
    for rule, color, label, marker in (("val", CYAN, "Val. Accuracy (Standard)", "o"),
                                       ("vloss", RED, "Val. Loss", "s"), ("last", GREY, "Last Checkpoint", "^")):
        ms_ = [acc(bud, "ERM", rule) for bud in budgets]
        a.errorbar(budgets, [m[0] for m in ms_], [m[1] for m in ms_], color=color, marker=marker, label=label,
                   capsize=2, zorder=3)
    budget_axis(a, budgets)
    a.set_ylabel("Held-Out Accuracy (%)")
    top_legend(a)
    x = np.arange(len(DG))
    for i, (rule, color, label) in enumerate((("val", CYAN, "Selected by Val. Accuracy"), ("vloss", RED, "Selected by Val. Loss"))):
        d = [diff(15000, m, "ERM", rule) for m in DG]
        b.bar(x + (i - 0.5) * 0.38, [v[0] for v in d], 0.38, yerr=[v[1] for v in d], color=color, edgecolor="black",
              linewidth=0.5, capsize=2, error_kw={"linewidth": 0.8}, label=label, zorder=3)
    b.axhline(0, color="black", linewidth=0.6, zorder=4)
    b.set_xticks(x)
    b.set_xticklabels([LABEL[m] for m in DG], fontsize=7)
    b.set_ylabel("Difference to ERM (Points)")
    b.set_ylim(-2.9, 1.9)
    top_legend(b)
    save(fig, "fig_selection")


def fig_budget():
    fig, (a, b) = panels(2)
    budgets = list(ROOTS)
    for alg, marker in (("ERM", "o"), ("Finetune", "s"), ("ER", "^"), ("PDCL_RS", "D")):
        ms_ = [acc(bud, alg, "vloss") for bud in budgets]
        a.errorbar(budgets, [m[0] for m in ms_], [m[1] for m in ms_], color=COLOR[alg], marker=marker,
                   label=LABEL[alg], capsize=2, zorder=3)
    budget_axis(a, budgets)
    a.set_ylabel("Held-Out Accuracy (%)")
    top_legend(a)
    for a1, a2, color, marker in (("PDCL_RS", "Finetune", RED, "D"), ("ER", "Finetune", CYAN, "^"), ("PDCL_RS", "ER", GREY, "o")):
        d = [diff(bud, a1, a2, "vloss") for bud in budgets]
        b.errorbar(budgets, [v[0] for v in d], [v[1] for v in d], color=color, marker=marker, capsize=2, zorder=3,
                   label=f"{LABEL[a1]} − {LABEL[a2]}".replace(" (ER)", ""))
    b.axhline(0, color="black", linewidth=0.6, zorder=2)
    budget_axis(b, budgets)
    b.set_ylabel("Paired Difference (Points)")
    top_legend(b)
    save(fig, "fig_budget")


def fig_mechanism():
    fig, (a, b) = panels(2)
    budgets = list(ROOTS)
    st = {alg: [end_stats(bud, alg) for bud in budgets] for alg in ("ER", "PDCL_RS")}
    a.plot(budgets, [s["mem"] for s in st["ER"]], color=CYAN, marker="^", label="Replay (ER)", zorder=3)
    a.plot(budgets, [s["mem"] for s in st["PDCL_RS"]], color=RED, marker="D", label="PDCL", zorder=3)
    a.plot(budgets, [s["eps"] for s in st["PDCL_RS"]], color=RED, linestyle="--", marker=None,
           label="PDCL Constraint Level", zorder=3)
    budget_axis(a, budgets)
    a.set_yscale("log")
    a.set_ylabel("Loss on Stored Examples")
    top_legend(a)
    s_ = [end_stats(bud, "PDCL_RS") for bud in budgets]
    b.plot(budgets, [s["train"] for s in s_], color=RED, marker="D", label="Training Split", zorder=3)
    b.plot(budgets, [s["val"] for s in s_], color=CYAN, marker="o", label="Validation Split", zorder=3)
    budget_axis(b, budgets)
    b.set_ylabel("Accuracy, Last Round (%)")
    top_legend(b)
    save(fig, "fig_mechanism")


def fig_tuning():
    fig, (a, b) = panels(2)
    algs = ("ERM", "Finetune", "ER", "PDCL_RS")
    x = np.arange(len(algs))
    for off, color, label, get in ((-0.12, GREY, "Learning Rate 5×10⁻⁵", lambda al: acc(15000, al, "vloss")),
                                   (0.12, RED, "Learning Rate 1.5×10⁻⁵", lambda al: mse(seed_means(ROOTS[15000], LOW[al], "vloss")))):
        v = [get(al) for al in algs]
        a.errorbar(x + off, [m[0] for m in v], [m[1] for m in v], color=color, marker="o", linestyle="none",
                   capsize=3, label=label, zorder=3)
    a.set_xticks(x)
    a.set_xticklabels([LABEL[al] for al in algs])
    a.set_xlim(-0.5, len(algs) - 0.5)
    a.set_ylabel("Held-Out Accuracy (%)")
    top_legend(a)
    algs = ("ERM",) + SEQ
    x = np.arange(len(algs))
    for i, (bud, color, label) in enumerate(((4000, CYAN, "1×10³ Steps Per Round"), (20000, RED, "5×10³ Steps Per Round"))):
        v = [mse([m["forgetting"] for m in stream_metrics(STREAM[bud], al).values()]) for al in algs]
        b.bar(x + (i - 0.5) * 0.38, [m[0] for m in v], 0.38, yerr=[m[1] for m in v], color=color, edgecolor="black",
              linewidth=0.5, capsize=2, error_kw={"linewidth": 0.8}, label=label, zorder=3)
    b.set_xticks(x)
    b.set_xticklabels(["Joint\nERM", "Fine-\ntuning", "ER", "ER\n+ FF", "PDCL", "PDCL\n+ FF"])
    b.set_ylabel("Forgetting (Points)")
    b.set_ylim(0, 5.5)
    top_legend(b)
    save(fig, "fig_tuning")


def main():
    numbers()
    tables()
    style()
    for old in FIG.glob("fig_*"):
        old.unlink()
    for f in (fig_trajectory, fig_selection, fig_budget, fig_mechanism, fig_tuning):
        f()
    print(json.dumps(N, indent=1))


if __name__ == "__main__":
    main()
