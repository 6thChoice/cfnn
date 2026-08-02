#!/usr/bin/env python3
"""NMR peak-recovery frontiers from validation-selected Pareto configurations.

The Pareto raw files already contain the best validation-selected config for
every task, family, seed, and capacity knob. This runner refits every stored
config and evaluates peak recovery, avoiding the legacy post-hoc choice of one
capacity from final dense-grid R2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import eis_models as M
import nmr_data as D
import nmr_peak_recovery as PR
from param_matched_utils import NumpySafeEncoder, count_trainable_params
from run_lowbudget_pareto import _std, _train, build, set_seed

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiment_refine" / "nmr_pareto_results" / "server_fair"
OUT = ROOT / "experiment_refine" / "nmr_peak_frontier_results"
SEEDS = [42, 123, 456, 789, 1024]
TASKS = {
    "ethanol": (1000, 2000),
    "caffeine": (1400, 2800),
    "strychnine": (2400, 4000),
}
FAMILIES = ["CFNN", "MLP", "KAN"]


def make_dataset(task: str, seed: int, smoke: bool):
    n_points, n_dense = TASKS[task]
    if smoke:
        n_points = min(n_points, 160)
        n_dense = min(n_dense, 400)
    return D.make_nmr_dataset(task, n_points, 0.03, seed, n_dense=n_dense)


def refit_stored_config(row, ds, smoke=False):
    """Refit one validation-selected raw Pareto row on all training points."""
    set_seed(row["seed"])
    model, params = build(row["family"], row["knob"], row["best_cfg"], ds["out_dim"])
    if model is None:
        raise RuntimeError(f"model unavailable for {row['family']}")
    if params != row["params"]:
        raise ValueError(f"parameter mismatch: stored={row['params']} rebuilt={params}")
    model = model.to(M.DEVICE)
    y_std, mean, scale = _std(ds["Y"])
    x_all = torch.tensor(ds["X"], device=M.DEVICE)
    y_all = torch.tensor(y_std, device=M.DEVICE)
    epochs = 80 if smoke else 800
    model = _train(model, x_all, y_all, x_all, y_all,
                   row["best_cfg"]["lr"], epochs, 80)
    return M.predict_numpy(model, ds["Xeval"]) * scale + mean


def dense_r2(y_true, y_pred):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    return float(1.0 - np.sum((yt - yp) ** 2) /
                 (np.sum((yt - yt.mean()) ** 2) + 1e-12))


def run_cell(task: str, family: str, seeds, smoke=False):
    source = SOURCE / f"raw_{task}_{family}.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    rows = [row for row in rows if row["seed"] in seeds]
    output = []
    for index, row in enumerate(rows, start=1):
        ds = make_dataset(task, row["seed"], smoke)
        pred = refit_stored_config(row, ds, smoke)
        x_ppm = ((ds["Xeval"].ravel() + 1.0) / 2.0 *
                 (ds["ppm_hi"] - ds["ppm_lo"]) + ds["ppm_lo"])
        metrics = PR.peak_recovery_metrics(ds["true_peaks_ppm"], x_ppm, pred.ravel())
        output.append({
            **row,
            "dense_r2_refit": dense_r2(ds["Yeval"], pred),
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "ppm_error_mean": metrics["ppm_error_mean"],
            "n_true": metrics["n_true"],
            "n_pred": metrics["n_pred"],
        })
        print(f"[{task}/{family}] {index}/{len(rows)} seed={row['seed']} "
              f"params={row['params']} F1={metrics['f1']:.3f}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds = [args.seed] if args.seed is not None else SEEDS
    rows = run_cell(args.task, args.family, seeds, args.smoke)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_seed{args.seed}" if args.seed is not None else ""
    path = OUT / f"peak_frontier_{args.task}_{args.family}{suffix}.json"
    path.write_text(json.dumps(rows, indent=2, cls=NumpySafeEncoder), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
