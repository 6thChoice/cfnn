"""Sharp-OBSERVABLE downstream test: Lorentzian resonance line-shapes.

Hypothesis (from the smooth-EIS finding): CFNN's low-budget advantage returns
when the sampled observable ITSELF is sharp (a resonance peak in |H(omega)|),
as opposed to EIS where the poles sit off the sampled jw axis and the curve is
smooth. Peaks are DENSELY sampled (>=8 points across FWHM) to avoid the
rf_resonator under-sampling degeneracy.

Tasks: single Lorentzian at 3 sharpness (Q) levels + a 2-peak "spectrum".
All three families (CFNN/MLP/KAN) HPO'd via the shared machinery in
run_lowbudget_pareto. Metric = R^2 on a dense clean grid. Reports R^2 vs params."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))

# reuse the exact same fair-HPO machinery used for the synthetic-pole Pareto
from run_lowbudget_pareto import hpo_fit, knobs_and_grid, SEEDS
from param_matched_utils import NumpySafeEncoder

OUT = ROOT / "experiment_refine" / "resonance_pareto_results"
OUT.mkdir(parents=True, exist_ok=True)


def lorentz(w, peaks):
    """peaks: list of (w0, gamma, amp). Physical resonance amplitude:
    peak height = amp/gamma^2 (sharper resonance -> taller, high dynamic range)."""
    y = np.zeros_like(w)
    for w0, gamma, amp in peaks:
        y = y + amp / ((w - w0) ** 2 + gamma * gamma)
    return y.astype(np.float32)


def make_resonance(peaks, n_points, noise, seed):
    """w in [-1,1], densely sampled; multiplicative noise; dense clean eval grid."""
    rng = np.random.default_rng(seed)
    w = np.linspace(-1.0, 1.0, n_points).astype(np.float32)
    y = lorentz(w, peaks)
    yn = (y * (1.0 + noise * rng.standard_normal(y.shape).astype(np.float32))).astype(np.float32)
    wd = np.linspace(-1.0, 1.0, 1000).astype(np.float32)
    yd = lorentz(wd, peaks)
    return {"X": w.reshape(-1, 1), "Y": yn.reshape(-1, 1),
            "Xeval": wd.reshape(-1, 1), "Yeval": yd.reshape(-1, 1),
            "out_dim": 1, "kind": "scalar"}


# gamma (=half-width) sweep. band width = 2, n=400 -> spacing 0.005.
# FWHM = 2*gamma, so points-across-FWHM = 2*gamma/0.005 = 400*gamma. All >= 8.
TASKS = {
    "res_Q_broad": lambda s, a: make_resonance([(0.15, 0.10, 1.0)], 120 if a.smoke else 400, 0.03, s),
    "res_Q_med":   lambda s, a: make_resonance([(0.15, 0.04, 1.0)], 120 if a.smoke else 400, 0.03, s),
    "res_Q_sharp": lambda s, a: make_resonance([(0.15, 0.02, 1.0)], 160 if a.smoke else 500, 0.03, s),
    "res_2peak":   lambda s, a: make_resonance([(-0.35, 0.03, 1.0), (0.40, 0.03, 0.7)],
                                               160 if a.smoke else 500, 0.03, s),
}


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
            knobs, _ = knobs_and_grid(family)
            for knob in knobs:
                for seed in seeds:
                    ds = TASKS[task](seed, args)
                    r = hpo_fit(family, knob, ds, seed, args)
                    if r is not None:
                        r["task"] = task
                        raw.append(r)
            print(f"[resonance] {task} / {family} done")
    json.dump(raw, open(OUT / f"resonance_raw{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)

    from collections import defaultdict
    g = defaultdict(list)
    for r in raw:
        g[(r["task"], r["family"], r["params"])].append(r["r2"])
    summ = [{"task": k[0], "family": k[1], "params": k[2],
             "r2_median": float(np.median(v)), "n": len(v)} for k, v in g.items()]
    json.dump(summ, open(OUT / f"resonance_summary{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)

    print("\n=== Min params to reach R2>=0.99 (per task/family) ===")
    for task in tasks:
        line = f"[{task}] "
        for fam in ["CFNN", "MLP", "KAN"]:
            pts = sorted((s["params"], s["r2_median"]) for s in summ if s["task"] == task and s["family"] == fam)
            ok = [p for p, r in pts if r >= 0.99]
            best = max((r for _, r in pts), default=float("nan"))
            line += f" {fam}={min(ok) if ok else 'NEVER'}(best {best:.3f})"
        print(line)


if __name__ == "__main__":
    main()
