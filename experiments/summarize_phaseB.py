"""Aggregate Phase-B (unknown-order) raw results into summary + SUMMARY_B.md.

B1a: prediction r2 by (K, model). B1b: r2 + failure_rate by (K, noise, model)
and pooled by model. Metric = prediction R2 / curve_mse / failure_rate (r2<0)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_refine" / "eis_phaseB_results"


def agg(rows, gk, metrics):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in gk)].append(r)
    out = []
    for key, items in g.items():
        row = {k: v for k, v in zip(gk, key)}
        row["n"] = len(items)
        for m in metrics:
            if m == "failure_rate":
                row[m] = float(np.mean([1.0 if it["failed"] else 0.0 for it in items]))
            else:
                vals = [it[m] for it in items if it.get(m) is not None and np.isfinite(it[m])]
                row[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out.append(row)
    return out


def _table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in sorted(rows, key=lambda x: [str(x.get(c)) for c in cols]):
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    summary, md = {}, ["# EIS Phase B (Unknown-Order) Summary\n"]

    p = OUT / "B1a_order_mismatch_raw.json"
    if p.exists():
        rows = json.load(open(p))
        a = agg(rows, ["K", "model"], ["r2", "curve_mse"])
        summary["B1a"] = a
        md.append("\n## B1a — prediction R² vs true order K (noise=0.03)\n")
        md.append(_table(a, ["K", "model", "n", "r2_mean", "curve_mse_mean"]))

    p = OUT / "B1b_illposed_raw.json"
    if p.exists():
        rows = json.load(open(p))
        a = agg(rows, ["K", "noise", "model"], ["r2", "failure_rate"])
        summary["B1b_cells"] = a
        md.append("\n\n## B1b — ill-posed cells: R² + failure_rate by (K, noise, model)\n")
        md.append(_table(a, ["K", "noise", "model", "n", "r2_mean", "failure_rate"]))
        pooled = agg(rows, ["model"], ["r2", "failure_rate"])
        summary["B1b_pooled"] = pooled
        md.append("\n\n## B1b — pooled by model\n")
        md.append(_table(pooled, ["model", "n", "r2_mean", "failure_rate"]))

    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    (OUT / "SUMMARY_B.md").write_text("\n".join(md))
    print(f"wrote {OUT/'summary.json'} and SUMMARY_B.md")


if __name__ == "__main__":
    main()
