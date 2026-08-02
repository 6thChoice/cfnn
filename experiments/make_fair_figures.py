"""Fair Pareto figures (5-seed, fairness-fixed HPO) for all three experiments,
from the merged server results. R2 (median) vs actual params, log-x, per task."""
import json, glob, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statistics as st

ER = os.path.dirname(os.path.abspath(__file__))
COL = {"CFNN": "#d1495b", "MLP": "#2e4057", "KAN": "#66a182"}


def summ_from(pattern):
    raw = []
    for fp in glob.glob(pattern):
        raw.extend(json.load(open(fp)))
    g = defaultdict(list)
    for r in raw:
        g[(r["task"], r["family"], r["params"])].append(r["r2"])
    return {k: st.median(v) for k, v in g.items()}


PANELS = [
    ("lowbudget", f"{ER}/lowbudget_pareto_results/server_fair/pareto_raw_*.json",
     ["pole_eps0.001", "pole_eps0.01", "eis_two_tc"]),
    ("resonance", f"{ER}/resonance_pareto_results/server_fair/resonance_raw_*.json",
     ["res_Q_broad", "res_Q_med", "res_Q_sharp", "res_2peak"]),
    ("nmr (semi-real)", f"{ER}/nmr_pareto_results/server_fair/raw_*.json",
     ["ethanol", "caffeine", "strychnine"]),
]

for name, pat, tasks in PANELS:
    S = summ_from(pat)
    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.0), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for fam in ["CFNN", "MLP", "KAN"]:
            pts = sorted((p, r) for (t, f, p), r in S.items() if t == task and f == fam)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o-", color=COL[fam], label=fam, ms=4)
        threshold = 0.90 if name.startswith("nmr") else 0.99
        ax.axhline(threshold, ls="--", color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(task)
        ax.set_xlabel("params")
        ax.set_ylabel("R² (median; exact-param support)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Fair HPO Pareto — {name}")
    fig.tight_layout()
    out = f"{ER}/figures/fair_{name.split()[0]}_pareto.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
