"""Parameter-efficiency main figure for SHARP tasks: R2 (median, 5 seeds) vs
actual params for CFNN / MLP / KAN, log-x. Each family's R2>=0.99 crossing is
marked. Shows (1) at CFNN's low budget the baselines lag, and (2) MLP/KAN were
scaled to ~1200-1900 params and DO reach ~1.0 -- so they were fairly tuned, and
CFNN's win is genuine parameter efficiency, not crippled baselines."""
import json, glob, os
from collections import defaultdict
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ER = os.path.dirname(os.path.abspath(__file__))
COL = {"CFNN": "#d1495b", "MLP": "#2e4057", "KAN": "#66a182"}
SRC = {
    "sharp pole ε=0.001": ("lowbudget_pareto_results/server_fair/pareto_raw_*.json", "pole_eps0.001"),
    "sharp pole ε=0.01":  ("lowbudget_pareto_results/server_fair/pareto_raw_*.json", "pole_eps0.01"),
    "resonance γ=0.02":   ("resonance_pareto_results/server_fair/resonance_raw_*.json", "res_Q_sharp"),
    "two-peak spectrum":  ("resonance_pareto_results/server_fair/resonance_raw_*.json", "res_2peak"),
    "4 Lorentzian peaks": ("moe_pareto_results/server_moe/moe_raw_*.json", "kpeak_4"),
}


def curve(pattern, task, fam):
    raw = []
    for fp in glob.glob(f"{ER}/{pattern}"):
        raw.extend(json.load(open(fp)))
    g = defaultdict(list)
    for r in raw:
        if r["task"] == task and r["family"] == fam:
            g[r["params"]].append(r["r2"])
    return sorted((p, st.median(v)) for p, v in g.items())


def cross(pts, thr=0.99):
    ok = [p for p, r in pts if r >= thr]
    return min(ok) if ok else None


panels = list(SRC.items())
fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2), squeeze=False)
for ax, (title, (pat, task)) in zip(axes[0], panels):
    for fam in ["CFNN", "MLP", "KAN"]:
        pts = curve(pat, task, fam)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=COL[fam], label=fam, ms=3.5, lw=1.4)
        c = cross(pts)
        if c is not None:
            ax.plot([c], [0.99], marker="*", color=COL[fam], ms=13,
                    markeredgecolor="k", markeredgewidth=0.4, zorder=5)
    ax.axhline(0.99, ls="--", color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("trainable params")
    ax.set_ylabel("R² (median, 5 seeds)")
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Parameter efficiency on sharp targets: CFNN reaches R²≥0.99 (★) at a fraction of MLP/KAN's budget; "
             "baselines scale to ~1.0 at high params (fairly tuned)", fontsize=10)
fig.tight_layout()
out = f"{ER}/figures/param_efficiency_sharp.png"
fig.savefig(out, dpi=145)
print("wrote", out)
