"""Summarize validation-selected common-budget NMR results."""
from __future__ import annotations

import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiment_refine/nmr_budgeted_results"
OUT = ROOT / "experiment_refine/paper/figures"
DERIVED = ROOT / "experiment_refine/paper/derived"
COLORS = {"CFNN": "#d1495b", "MLP": "#2e4057", "KAN": "#66a182"}
LABELS = {"CFNN": "CFNN-Hybrid", "MLP": "MLP", "KAN": "KAN"}


def median_iqr(values):
    a = np.asarray(values, dtype=float)
    return float(np.median(a)), float(np.percentile(a, 75) - np.percentile(a, 25))


def main():
    rows = []
    for path in sorted(SRC.glob("raw_*_*.json")):
        rows.extend(json.loads(path.read_text()))
    grouped = {}
    for r in rows:
        grouped.setdefault((r["task"], r["family"], r["budget"]), []).append(r)

    summary = []
    for key, vals in sorted(grouped.items()):
        task, family, budget = key
        params = median_iqr([r["params"] for r in vals])
        out = {"task": task, "family": family, "budget": budget,
               "n": len(vals), "params_median": params[0], "params_iqr": params[1]}
        for metric in ["test_r2", "f1", "ppm_error"]:
            out[f"{metric}_median"], out[f"{metric}_iqr"] = median_iqr([r[metric] for r in vals])
        summary.append(out)
    DERIVED.mkdir(exist_ok=True)
    (DERIVED / "nmr_budgeted_summary.json").write_text(json.dumps(summary, indent=2))

    tasks = ["ethanol", "caffeine", "strychnine"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex="col")
    for j, task in enumerate(tasks):
        for metric, ax in [("test_r2", axes[0, j]), ("f1", axes[1, j])]:
            for family in ["CFNN", "MLP", "KAN"]:
                pts = [r for r in summary if r["task"] == task and r["family"] == family]
                pts.sort(key=lambda r: r["budget"])
                if not pts:
                    continue
                x = [r["params_median"] for r in pts]
                y = [r[f"{metric}_median"] for r in pts]
                err = [r[f"{metric}_iqr"] / 2 for r in pts]
                ax.errorbar(x, y, yerr=err, marker="o", color=COLORS[family],
                            label=LABELS[family], capsize=3)
            ax.set_xscale("log")
            ax.grid(axis="y", color="#dddddd", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel("median [IQR]" if metric == "test_r2" else "center F1")
            if metric == "f1":
                ax.set_xlabel("selected parameters (median)")
            ax.set_title(task if metric == "test_r2" else "")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Validation-selected NMR reconstruction at common parameter budgets")
    fig.tight_layout()
    fig.savefig(OUT / "nmr_budgeted_recovery.pdf", bbox_inches="tight")
    fig.savefig(OUT / "nmr_budgeted_recovery.png", dpi=220, bbox_inches="tight")

    lines = ["# Validation-selected common-budget NMR results", "",
             "F1 and ppm error are selected using validation loss at fixed parameter budgets; medians and IQRs are over five seeds.",
             "", "| Molecule | Family | Budget | Params median [IQR] | R2 median [IQR] | F1 median [IQR] | ppm error median [IQR] |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['task']} | {LABELS[r['family']]} | {r['budget']} | {r['params_median']:.0f} [{r['params_iqr']:.0f}] | {r['test_r2_median']:.3f} [{r['test_r2_iqr']:.3f}] | {r['f1_median']:.2f} [{r['f1_iqr']:.2f}] | {r['ppm_error_median']:.4f} [{r['ppm_error_iqr']:.4f}] |")
    (DERIVED / "nmr_budgeted_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
