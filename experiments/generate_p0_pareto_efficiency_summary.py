#!/usr/bin/env python3
"""P0-3: numeric parameter-efficiency summaries from existing Pareto metrics.

The paper's revised logic needs a numeric table rather than only Pareto plots.
This script reads submit_codebase/pareto_runner/results/scientific_pareto/pareto_metrics.json
and writes concise summaries:
- per function/budget winner by R² and RMSE
- low-budget advantages
- parameter usage of the best CFNN vs best baseline
- threshold-style tables for R² >= thresholds when possible
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/zxc/CodeBase/cofrnet")
IN_PATH = ROOT / "submit_codebase" / "pareto_runner" / "results" / "scientific_pareto" / "pareto_metrics.json"
OUT_DIR = ROOT / "experiment_refine" / "p0_pareto_efficiency_summary"
CFNN_MODELS = {"Hybrid", "Standard", "Boost", "MoE", "CFNN", "CFNN-Hybrid", "CFNN-MoE", "CFNN-Boost"}
BASELINE_MODELS = {"MLP", "KAN"}
THRESHOLDS = [0.999, 0.9995, 0.9998]


def load_metrics():
    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    rows = []
    for func, entries in data.items():
        for key, rec in entries.items():
            row = dict(rec)
            row["function"] = func
            row["key"] = key
            row["budget"] = int(row.get("budget", key.split("_B")[-1]))
            row["model"] = str(row["model"])
            row["actual_params"] = int(row["actual_params"])
            row["r2_mean"] = float(row["r2_mean"])
            row["r2_std"] = float(row.get("r2_std", 0.0))
            row["rmse_mean"] = float(row["rmse_mean"])
            row["rmse_std"] = float(row.get("rmse_std", 0.0))
            rows.append(row)
    return rows


def best(rows, selector, key, reverse=False):
    cand = [r for r in rows if selector(r)]
    if not cand:
        return None
    return sorted(cand, key=key, reverse=reverse)[0]


def summarize(rows):
    funcs = sorted({r["function"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})
    per_cell = []
    for f in funcs:
        for b in budgets:
            cell = [r for r in rows if r["function"] == f and r["budget"] == b]
            if not cell:
                continue
            best_cfnn = best(cell, lambda r: r["model"] in CFNN_MODELS, key=lambda r: r["r2_mean"], reverse=True)
            best_base = best(cell, lambda r: r["model"] in BASELINE_MODELS, key=lambda r: r["r2_mean"], reverse=True)
            best_all = best(cell, lambda r: True, key=lambda r: r["r2_mean"], reverse=True)
            per_cell.append({
                "function": f,
                "budget": b,
                "best_all_model": best_all["model"],
                "best_all_r2": best_all["r2_mean"],
                "best_cfnn_model": best_cfnn["model"] if best_cfnn else None,
                "best_cfnn_r2": best_cfnn["r2_mean"] if best_cfnn else None,
                "best_cfnn_rmse": best_cfnn["rmse_mean"] if best_cfnn else None,
                "best_cfnn_params": best_cfnn["actual_params"] if best_cfnn else None,
                "best_baseline_model": best_base["model"] if best_base else None,
                "best_baseline_r2": best_base["r2_mean"] if best_base else None,
                "best_baseline_rmse": best_base["rmse_mean"] if best_base else None,
                "best_baseline_params": best_base["actual_params"] if best_base else None,
                "delta_r2_cfnn_minus_baseline": (best_cfnn["r2_mean"] - best_base["r2_mean"]) if best_cfnn and best_base else None,
                "param_ratio_cfnn_over_baseline": (best_cfnn["actual_params"] / best_base["actual_params"]) if best_cfnn and best_base else None,
                "rmse_ratio_cfnn_over_baseline": (best_cfnn["rmse_mean"] / best_base["rmse_mean"]) if best_cfnn and best_base else None,
            })

    threshold = defaultdict(dict)
    for f in funcs:
        frows = [r for r in rows if r["function"] == f]
        for t in THRESHOLDS:
            threshold[f][str(t)] = {}
            for group_name, models in [("best_cfnn", CFNN_MODELS), ("best_baseline", BASELINE_MODELS), ("MLP", {"MLP"}), ("KAN", {"KAN"}), ("Hybrid", {"Hybrid"}), ("Standard", {"Standard"})]:
                cand = [r for r in frows if r["model"] in models and r["r2_mean"] >= t]
                if cand:
                    chosen = sorted(cand, key=lambda r: (r["actual_params"], -r["r2_mean"]))[0]
                    threshold[f][str(t)][group_name] = {
                        "model": chosen["model"],
                        "params": chosen["actual_params"],
                        "budget": chosen["budget"],
                        "r2_mean": chosen["r2_mean"],
                        "rmse_mean": chosen["rmse_mean"],
                    }
                else:
                    threshold[f][str(t)][group_name] = None

    deltas = np.asarray([c["delta_r2_cfnn_minus_baseline"] for c in per_cell if c["delta_r2_cfnn_minus_baseline"] is not None])
    low_budget_cells = [c for c in per_cell if c["budget"] == min(budgets)]
    out = {
        "input": str(IN_PATH),
        "per_cell": per_cell,
        "thresholds": threshold,
        "aggregate": {
            "n_cells": len(per_cell),
            "mean_delta_r2_cfnn_minus_baseline": float(deltas.mean()) if deltas.size else None,
            "wins_delta_gt_0_005": int(np.sum(deltas > 0.005)) if deltas.size else 0,
            "ties_abs_delta_le_0_005": int(np.sum(np.abs(deltas) <= 0.005)) if deltas.size else 0,
            "losses_delta_lt_minus_0_005": int(np.sum(deltas < -0.005)) if deltas.size else 0,
            "low_budget_cells": low_budget_cells,
        },
    }
    return out


def write_md(summary):
    lines = ["# P0 Pareto efficiency numeric summary", ""]
    lines.append(f"Input: `{summary['input']}`")
    lines.append("")
    agg = summary["aggregate"]
    lines.append("## Overall R² comparison")
    lines.append("")
    lines.append(f"Across {agg['n_cells']} function × budget cells: wins={agg['wins_delta_gt_0_005']}, ties={agg['ties_abs_delta_le_0_005']}, losses={agg['losses_delta_lt_minus_0_005']} using |ΔR²|≤0.005 as tie.")
    lines.append(f"Mean ΔR²(best CFNN - best baseline): {agg['mean_delta_r2_cfnn_minus_baseline']:.6f}.")
    lines.append("")
    lines.append("This existing fair Pareto sweep mostly supports parity at high R², not a broad raw-accuracy win. Its useful role in the revised paper is parameter economy at comparable accuracy and identifying where functions are non-discriminating under fair standardization.")
    lines.append("")
    lines.append("## Per-cell best CFNN vs best baseline")
    lines.append("")
    lines.append("| Function | Budget | Best CFNN | R² | params | Best baseline | R² | params | ΔR² | CFNN/base params |")
    lines.append("|---|---:|---|---:|---:|---|---:|---:|---:|---:|")
    for c in summary["per_cell"]:
        lines.append(
            f"| {c['function']} | {c['budget']} | {c['best_cfnn_model']} | {c['best_cfnn_r2']:.6f} | {c['best_cfnn_params']} | "
            f"{c['best_baseline_model']} | {c['best_baseline_r2']:.6f} | {c['best_baseline_params']} | "
            f"{c['delta_r2_cfnn_minus_baseline']:.6f} | {c['param_ratio_cfnn_over_baseline']:.3f} |"
        )
    lines.append("")
    lines.append("## Threshold-style minimum parameters")
    lines.append("")
    for func, tdict in summary["thresholds"].items():
        lines.append(f"### {func}")
        lines.append("")
        lines.append("| R² threshold | Best CFNN min params | Best baseline min params | MLP | KAN | Hybrid | Standard |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for t, groups in tdict.items():
            def fmt(v):
                return "--" if v is None else f"{v['params']} ({v['model']}, R²={v['r2_mean']:.5f})"
            lines.append(f"| {t} | {fmt(groups['best_cfnn'])} | {fmt(groups['best_baseline'])} | {fmt(groups['MLP'])} | {fmt(groups['KAN'])} | {fmt(groups['Hybrid'])} | {fmt(groups['Standard'])} |")
        lines.append("")
    lines.append("## Paper-facing interpretation")
    lines.append("")
    lines.append("Do not use this fair standardized scientific-pareto result to claim broad CFNN raw-accuracy dominance: the fair sweep shows near-parity across cells. Use it to refine the narrative: CFNN-Hybrid can reach comparable high R² with fewer actual trainable parameters in some low-budget/singular cells, but several tasks become non-discriminating once baselines and target scaling are made fair. If the paper needs a stronger parameter-efficiency claim, it should rely on the separate special-function Pareto plots only after verifying their protocol matches the revised fairness standard.")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_metrics()
    summary = summarize(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_md(summary)
    print((OUT_DIR / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
