# experiment_refine/run_nmr_pareto.py
"""Semi-real NMR line-shape Pareto: real-molecule 1H spectra vs CFNN/MLP/KAN.
Reuses the exact fair-HPO machinery from run_lowbudget_pareto (config picked by
held-out validation, retrained on all points, R^2 on a dense clean grid).
Reports R^2 vs actual params + peak-recovery at each family's most economical
R^2>=0.99 config. Supersedes the un-HPO'd run_nmr_spectral.py."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))

from run_lowbudget_pareto import hpo_fit, knobs_and_grid, SEEDS, build, _train, _std, set_seed
import eis_models as M
import nmr_data as D
import nmr_peak_recovery as PR
from param_matched_utils import NumpySafeEncoder

OUT = ROOT / "experiment_refine" / "nmr_pareto_results"
OUT.mkdir(parents=True, exist_ok=True)

# realistic full-resolution sampling (sharp peaks over wide ppm window need it);
# the BUDGET axis is model params, not data points.
TASKS = {
    "ethanol":    lambda s, a: D.make_nmr_dataset("ethanol",    120 if a.smoke else 1000, 0.03, s, n_dense=2000),
    "caffeine":   lambda s, a: D.make_nmr_dataset("caffeine",   160 if a.smoke else 1400, 0.03, s, n_dense=2800),
    "strychnine": lambda s, a: D.make_nmr_dataset("strychnine", 240 if a.smoke else 2400, 0.03, s, n_dense=4000),
}

# extend the shared knob ranges upward: 22-peak strychnine needs more capacity.
# Capacity ceilings now live in the shared fair knobs_and_grid (2026-07-06
# fairness fix); no NMR-specific extension needed. Kept empty for provenance.
NMR_EXTRA = {"CFNN": [], "MLP": [], "KAN": []}


def nmr_knobs(family):
    base, _ = knobs_and_grid(family)
    return base + NMR_EXTRA[family]


def refit_predict(family, knob, ds, seed, args):
    """Pick best cfg by held-out val (same as hpo_fit), retrain on all points,
    return the dense reconstruction in ORIGINAL Y units (for peak picking)."""
    X, Y, out_dim = ds["X"], ds["Y"], ds["out_dim"]
    n = len(X)
    idx = np.arange(n)
    val = idx[::4]
    tr = np.setdiff1d(idx, val)
    import torch
    Ytr_b, m0, s0 = _std(Y[tr])
    Xtr = torch.tensor(X[tr], device=M.DEVICE); Ytr = torch.tensor(Ytr_b, device=M.DEVICE)
    Xva = torch.tensor(X[val], device=M.DEVICE)
    Yva = torch.tensor(((Y[val] - m0) / s0).astype("float32"), device=M.DEVICE)
    ep_sel = 80 if args.smoke else 500  # match hpo_fit's full early-stopped budget
    ep_fin = 80 if args.smoke else 800
    _, cfgs = knobs_and_grid(family)
    best = None
    for cfg in cfgs:
        set_seed(seed)
        model, _ = build(family, knob, cfg, out_dim)
        if model is None:
            return None
        model = model.to(M.DEVICE)
        model = _train(model, Xtr, Ytr, Xva, Yva, cfg["lr"], ep_sel, 50)
        pv = M.predict_numpy(model, X[val]) * s0 + m0
        vmse = float(np.mean((pv - Y[val]) ** 2))
        if best is None or vmse < best["vmse"]:
            best = {"cfg": cfg, "vmse": vmse}
    set_seed(seed)
    model, _ = build(family, knob, best["cfg"], out_dim)
    model = model.to(M.DEVICE)
    Yb, m1, s1 = _std(Y)
    Xall = torch.tensor(X, device=M.DEVICE); Yall = torch.tensor(Yb, device=M.DEVICE)
    model = _train(model, Xall, Yall, Xall, Yall, best["cfg"]["lr"], ep_fin, 80)
    return M.predict_numpy(model, ds["Xeval"]) * s1 + m1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", default="all", choices=list(TASKS) + ["all"])
    ap.add_argument("--family", default="all", choices=["CFNN", "MLP", "KAN", "all"])
    args = ap.parse_args()
    tasks = list(TASKS) if args.task == "all" else [args.task]
    fams = ["CFNN", "MLP", "KAN"] if args.family == "all" else [args.family]
    suf = ("" if args.task == "all" else f"_{args.task}") + ("" if args.family == "all" else f"_{args.family}")
    seeds = SEEDS[:2] if args.smoke else SEEDS

    raw = []
    for task in tasks:
        for family in fams:
            for knob in nmr_knobs(family):
                for seed in seeds:
                    ds = TASKS[task](seed, args)
                    r = hpo_fit(family, knob, ds, seed, args)
                    if r is not None:
                        r["task"] = task
                        raw.append(r)
            print(f"[nmr] {task} / {family} done")
    json.dump(raw, open(OUT / f"raw{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)

    g = defaultdict(list)
    for r in raw:
        g[(r["task"], r["family"], r["params"])].append(r["r2"])
    summ = [{"task": k[0], "family": k[1], "params": k[2],
             "r2_median": float(np.median(v)), "n": len(v)} for k, v in g.items()]
    json.dump(summ, open(OUT / f"summary{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)

    print("\n=== Min params to reach R2>=0.99 (per task/family) ===")
    for task in tasks:
        line = f"[{task}] "
        for fam in fams:
            pts = sorted((s["params"], s["r2_median"]) for s in summ
                         if s["task"] == task and s["family"] == fam)
            ok = [p for p, r in pts if r >= 0.99]
            best = max((r for _, r in pts), default=float("nan"))
            line += f" {fam}={min(ok) if ok else 'NEVER'}(best {best:.3f})"
        print(line)

    # ---- peak recovery at each family's most economical passing config ----
    peak_rows = []
    for task in tasks:
        for fam in fams:
            pts = sorted((s["params"], s["r2_median"]) for s in summ
                         if s["task"] == task and s["family"] == fam)
            if not pts:
                continue
            ok = [(p, r) for p, r in pts if r >= 0.99]
            target_params = (min(ok)[0] if ok else max(pts, key=lambda t: t[1])[0])
            # recover the knob that produced target_params for this family
            knob = next((kb for kb in nmr_knobs(fam)
                         if any(rr["family"] == fam and rr["knob"] == kb and rr["params"] == target_params
                                and rr["task"] == task for rr in raw)), None)
            if knob is None:
                continue
            f1s, errs = [], []
            for seed in seeds:
                ds = TASKS[task](seed, args)
                pred = refit_predict(fam, knob, ds, seed, args)
                if pred is None:
                    continue
                xppm = (ds["Xeval"].ravel() + 1.0) / 2.0 * (ds["ppm_hi"] - ds["ppm_lo"]) + ds["ppm_lo"]
                met = PR.peak_recovery_metrics(ds["true_peaks_ppm"], xppm, pred.ravel())
                f1s.append(met["f1"])
                if met["ppm_error_mean"] == met["ppm_error_mean"]:  # not nan
                    errs.append(met["ppm_error_mean"])
            peak_rows.append({"task": task, "family": fam, "params": target_params,
                              "passed_r2_099": bool(ok),
                              "f1_median": float(np.median(f1s)) if f1s else 0.0,
                              "ppm_error_median": float(np.median(errs)) if errs else float("nan"),
                              "n_seeds": len(f1s)})
    json.dump(peak_rows, open(OUT / f"peak_recovery{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)
    print("\n=== Peak recovery (F1 / ppm-error) at economical config ===")
    for r in peak_rows:
        print(f"  [{r['task']}] {r['family']:>5} params={r['params']:>5} "
              f"F1={r['f1_median']:.3f} ppmErr={r['ppm_error_median']:.4f} pass099={r['passed_r2_099']}")


if __name__ == "__main__":
    main()
