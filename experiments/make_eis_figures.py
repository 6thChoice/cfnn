"""Generate EIS pilot figures from summary.json (one per sub-experiment)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiment_refine" / "eis_pilot_results"
FIG_DIR = ROOT / "experiment_refine" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _series(agg, xkey, ymean):
    by_model = {}
    for r in agg:
        by_model.setdefault(r["model"], []).append((r[xkey], r.get(ymean)))
    for m in by_model:
        by_model[m].sort(key=lambda t: t[0])
    return by_model


def line_plot(agg, xkey, ymean, title, fname, logx=True, logy=True):
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, pts in _series(agg, xkey, ymean).items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # Guard against log(0) or negative y values when logy is True
        if logy:
            # Filter out non-positive y values
            filtered = [(x, y) for x, y in zip(xs, ys) if y is not None and y > 0]
            if filtered:
                xs, ys = zip(*filtered)
                xs, ys = list(xs), list(ys)
            else:
                # If all values are non-positive, skip this model
                continue

        ax.plot(xs, ys, marker="o", label=model)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xkey)
    ax.set_ylabel(ymean)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def main():
    summary = json.load(open(OUT_DIR / "summary.json"))
    if "E1" in summary:
        line_plot(summary["E1"], "tau_ratio", "pole_err_median_mean",
                  "E1: pole recovery vs sharpness", "eis_E1_sharpness.png")
    if "E2" in summary:
        line_plot(summary["E2"], "n_points", "pole_err_median_mean",
                  "E2: data efficiency", "eis_E2_data_efficiency.png")
    if "E3" in summary:
        line_plot(summary["E3"], "params_count_mean", "pole_err_median_mean",
                  "E3: parameter efficiency (Pareto)", "eis_E3_param_efficiency.png")
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
