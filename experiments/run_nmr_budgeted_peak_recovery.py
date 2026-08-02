"""Peak recovery at pre-registered capacity levels.

The original NMR sweep already selected the hyper-parameter configuration for
each seed and capacity knob using validation loss. This runner reuses those
validation-selected configurations at three pre-registered capacity levels,
then evaluates clean-grid R2 and environment-center recovery once. It does not
select a capacity from the final evaluation R2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))
from run_lowbudget_pareto import SEEDS, _std, _train, build, set_seed  # noqa: E402
import eis_models as M  # noqa: E402
import nmr_data as D  # noqa: E402
import nmr_peak_recovery as PR  # noqa: E402
from param_matched_utils import NumpySafeEncoder  # noqa: E402

OUT = ROOT / "experiment_refine/nmr_budgeted_results"
TASKS = {
    "ethanol": lambda s: D.make_nmr_dataset("ethanol", 1000, 0.03, s, n_dense=2000),
    "caffeine": lambda s: D.make_nmr_dataset("caffeine", 1400, 0.03, s, n_dense=2800),
    "strychnine": lambda s: D.make_nmr_dataset("strychnine", 2400, 0.03, s, n_dense=4000),
}
# Approximate common budgets: raw parameter counts are 82/162/514 for CFNN,
# 76/298/1174 for MLP, and 117/295/1177 for KAN at these capacity levels.
FIXED_KNOBS = {"CFNN": [8, 16, 32], "MLP": [80, 300, 1200], "KAN": [4, 6, 24]}
FAMILIES = list(FIXED_KNOBS)
BUDGET_LABELS = [100, 300, 1200]


def r2(y_true, y_pred):
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return float(1.0 - np.sum((yt - yp) ** 2) /
                 (np.sum((yt - yt.mean()) ** 2) + 1e-12))


def load_validation_choices(task, family):
    candidates = [
        ROOT / f"experiment_refine/nmr_pareto_results/server_fair/raw_{task}_{family}.json",
        ROOT / f"experiment_refine/nmr_pareto_results/raw_{task}_{family}.json",
    ]
    path = next((p for p in candidates if p.exists()), candidates[0])
    rows = json.loads(path.read_text())
    return {(r["seed"], r["knob"]): r for r in rows}


def run_one(task, family, level, knob, seed, smoke=False):
    ds = TASKS[task](seed)
    choices = load_validation_choices(task, family)
    selected = choices[(seed, knob)]
    cfg = selected["best_cfg"]
    X, Y = ds["X"], ds["Y"]
    set_seed(seed)
    model, params = build(family, knob, cfg, ds["out_dim"])
    model = model.to(M.DEVICE)
    Yb, mean, scale = _std(Y)
    Xall = torch.tensor(X, device=M.DEVICE)
    Yall = torch.tensor(Yb, device=M.DEVICE)
    model = _train(model, Xall, Yall, Xall, Yall, cfg["lr"],
                   80 if smoke else 800, 80)
    pred = M.predict_numpy(model, ds["Xeval"]) * scale + mean
    xppm = (ds["Xeval"].ravel() + 1.0) / 2.0 * (ds["ppm_hi"] - ds["ppm_lo"]) + ds["ppm_lo"]
    peak = PR.peak_recovery_metrics(ds["true_peaks_ppm"], xppm, pred.ravel())
    return {"task": task, "family": family, "budget": level, "knob": knob,
            "seed": seed, "params": int(params), "best_cfg": cfg,
            "source_frontier_r2": selected["r2"],
            "test_r2": r2(ds["Yeval"], pred), "f1": float(peak["f1"]),
            "ppm_error": float(peak["ppm_error_mean"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    ap.add_argument("--family", choices=FAMILIES + ["all"], default="all")
    args = ap.parse_args()
    tasks = list(TASKS) if args.task == "all" else [args.task]
    families = FAMILIES if args.family == "all" else [args.family]
    seeds = SEEDS[:2] if args.smoke else SEEDS
    OUT.mkdir(exist_ok=True)
    rows = []
    for task in tasks:
        for family in families:
            for level, knob in zip(BUDGET_LABELS, FIXED_KNOBS[family]):
                for seed in seeds:
                    rows.append(run_one(task, family, level, knob, seed, args.smoke))
                print(f"[fixed-capacity-nmr] {task} {family} level={level}", flush=True)
    suffix = ("_smoke" if args.smoke else "") + f"_{args.task}_{args.family}"
    out = OUT / f"raw{suffix}.json"
    out.write_text(json.dumps(rows, indent=2, cls=NumpySafeEncoder))
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
