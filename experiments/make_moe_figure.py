"""MoE probe figure: best-R2 vs peak count K (left) and min-params to R2>=0.99
vs K (right), for CFNN-MoE / CFNN / MLP / KAN on the synthetic K-peak sweep."""
import json, glob, os
from collections import defaultdict
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ER = os.path.dirname(os.path.abspath(__file__))
COL = {"CFNNMoE": "#e07a5f", "CFNN": "#d1495b", "MLP": "#2e4057", "KAN": "#66a182"}
KS = [1, 2, 4, 8, 16]

raw = []
for fp in glob.glob(f"{ER}/moe_pareto_results/server_moe/moe_raw_*.json"):
    raw.extend(json.load(open(fp)))
g = defaultdict(list)
for r in raw:
    g[(r["task"], r["family"], r["params"])].append(r["r2"])
summ = [(t, f, p, st.median(v)) for (t, f, p), v in g.items()]


def best(task, fam):
    rs = [r for t, f, p, r in summ if t == task and f == fam]
    return max(rs) if rs else float("nan")


def minp(task, fam, thr=0.99):
    pts = sorted((p, r) for t, f, p, r in summ if t == task and f == fam)
    ok = [p for p, r in pts if r >= thr]
    return min(ok) if ok else None


fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for fam in ["CFNNMoE", "CFNN", "MLP", "KAN"]:
    ys = [best(f"kpeak_{k}", fam) for k in KS]
    ax[0].plot(KS, ys, "o-", color=COL[fam], label=fam)
    mp = [minp(f"kpeak_{k}", fam) for k in KS]
    xs = [k for k, m in zip(KS, mp) if m is not None]
    yp = [m for m in mp if m is not None]
    ax[1].plot(xs, yp, "o-", color=COL[fam], label=fam)
ax[0].axhline(0.99, ls="--", color="gray", lw=0.8)
ax[0].set_xscale("log", base=2); ax[0].set_xticks(KS); ax[0].set_xticklabels(KS)
ax[0].set_xlabel("number of Lorentzian peaks K"); ax[0].set_ylabel("best R² (5 seeds)")
ax[0].set_title("Fidelity vs peak count"); ax[0].legend(fontsize=8)
ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
ax[1].set_xticks(KS); ax[1].set_xticklabels(KS)
ax[1].set_xlabel("number of Lorentzian peaks K"); ax[1].set_ylabel("min params to R²≥0.99")
ax[1].set_title("Economy at R²≥0.99 (missing = never reached)"); ax[1].legend(fontsize=8)
fig.suptitle("MoE probe: gating extends CFNN's peak-count reach, at the cost of few-peak economy; KAN leads at high K")
fig.tight_layout()
out = f"{ER}/figures/moe_peakcount.png"
fig.savefig(out, dpi=140)
print("wrote", out)
