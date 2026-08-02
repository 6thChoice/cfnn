"""Do the NESTED/ladder original-CoFrNet variants beat the additive Hybrid CFNN
on sharp / downstream tasks? Completes the variant matrix.

Families:
  CFNN          -- additive rational units (CFNNHybridEIS)   [what we've been using]
  CFNetStd      -- nested polynomial continued fraction (CFNet_Standard)
  CoFrNetLadder -- original-paper scalar-reciprocal ladder (CoFrNetDL)
  MLP, KAN      -- generic-network baselines
Tasks: synthetic K-peak Lorentzian sweep (K=1,2,4,8,16) + real caffeine/strychnine.
All families share the fair HPO machinery (run_lowbudget_pareto). Metric = R^2 on
a dense clean grid vs actual params. --task / --family for parallelism."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))

from run_lowbudget_pareto import hpo_fit, knobs_and_grid, SEEDS
from run_moe_pareto import make_kpeak
from param_matched_utils import NumpySafeEncoder
import nmr_data as D

OUT = ROOT / "experiment_refine" / "variants_pareto_results"
OUT.mkdir(parents=True, exist_ok=True)
FAMILIES = ["CFNN", "CFNetStd", "CoFrNetLadder", "MLP", "KAN"]

TASKS = {
    "kpeak_1":  lambda s, a: make_kpeak(1,  120 if a.smoke else 600, 0.03, s),
    "kpeak_2":  lambda s, a: make_kpeak(2,  120 if a.smoke else 600, 0.03, s),
    "kpeak_4":  lambda s, a: make_kpeak(4,  120 if a.smoke else 600, 0.03, s),
    "kpeak_8":  lambda s, a: make_kpeak(8,  160 if a.smoke else 700, 0.03, s),
    "kpeak_16": lambda s, a: make_kpeak(16, 200 if a.smoke else 800, 0.03, s),
    "caffeine":   lambda s, a: D.make_nmr_dataset("caffeine", 160 if a.smoke else 1400, 0.03, s, n_dense=2800),
    "strychnine": lambda s, a: D.make_nmr_dataset("strychnine", 240 if a.smoke else 2400, 0.03, s, n_dense=4000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", default="all", choices=list(TASKS) + ["all"])
    ap.add_argument("--family", default="all", choices=FAMILIES + ["all"])
    args = ap.parse_args()
    tasks = list(TASKS) if args.task == "all" else [args.task]
    fams = FAMILIES if args.family == "all" else [args.family]
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
            print(f"[variants] {task} / {family} done")
    json.dump(raw, open(OUT / f"variants_raw{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)

    g = defaultdict(list)
    for r in raw:
        g[(r["task"], r["family"], r["params"])].append(r["r2"])
    summ = [{"task": k[0], "family": k[1], "params": k[2],
             "r2_median": float(np.median(v)), "n": len(v)} for k, v in g.items()]
    json.dump(summ, open(OUT / f"variants_summary{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)

    print("\n=== min params @ R2>=0.99 | best-R2 ===")
    for task in tasks:
        line = f"[{task}] "
        for fam in fams:
            pts = sorted((s["params"], s["r2_median"]) for s in summ
                         if s["task"] == task and s["family"] == fam)
            ok = [p for p, r in pts if r >= 0.99]
            best = max((r for _, r in pts), default=float("nan"))
            line += f" {fam}={min(ok) if ok else 'NEVER'}(best {best:.3f})"
        print(line)


if __name__ == "__main__":
    main()
