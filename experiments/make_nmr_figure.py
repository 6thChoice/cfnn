# experiment_refine/make_nmr_figure.py
"""Plot NMR semi-real Pareto: R^2 (median, 5 seeds) vs actual params, one panel
per molecule (ethanol/caffeine/strychnine), three families overlaid."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiment_refine" / "nmr_pareto_results" / "summary.json"
FIG = ROOT / "experiment_refine" / "figures" / "nmr_pareto.png"
FIG.parent.mkdir(parents=True, exist_ok=True)

COL = {"CFNN": "#d1495b", "MLP": "#2e4057", "KAN": "#66a182"}
ORDER = ["ethanol", "caffeine", "strychnine"]


def main():
    summ = json.load(open(RES))
    tasks = [t for t in ORDER if any(s["task"] == t for s in summ)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for fam in ["CFNN", "MLP", "KAN"]:
            pts = sorted((s["params"], s["r2_median"]) for s in summ
                         if s["task"] == task and s["family"] == fam)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o-", color=COL[fam], label=fam, markersize=4)
        ax.axhline(0.99, ls="--", color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.02)
        ax.set_title(task); ax.set_xlabel("trainable params"); ax.set_ylabel("R² (median)")
        ax.legend(fontsize=8)
    fig.suptitle("Semi-real NMR line-shape: R² vs params (5 seeds)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
