"""Fill the variant matrix: the 3 previously-untested CFNN variants on the sharp
tasks (K-peak sweep + real NMR caffeine/strychnine).

  MLCFN   -- MultiLayerCFN: continued-fraction as element-wise activation (KAN-style).
             Standard forward -> uses the shared fair hpo_fit.
  Boost   -- EnsembleResCoFrNet: boosting ensemble (staged add_model + freeze).
  MoEens  -- MoE_Ensemble: RBF-gated mixture of nested CFNet_Core experts.
Boost/MoEens need staged training (incompatible with the generic _train), so we
port the repo's train_boost/train_moe verbatim into tensor form and wrap them in
the SAME fair protocol: HPO by held-out val, retrain on all, R^2 on a dense clean
grid, 5 seeds. Params reported = total numel (Boost freezes old stages -- frozen
weight still counts as model size). --task / --family for parallelism."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))

from run_lowbudget_pareto import hpo_fit, knobs_and_grid, SEEDS, _std, set_seed, r2
import cofrnet_variants as V
import eis_models as M
from run_moe_pareto import make_kpeak
from param_matched_utils import NumpySafeEncoder
import nmr_data as D

OUT = ROOT / "experiment_refine" / "staged_variants_results"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = M.DEVICE

TASKS = {
    "kpeak_1":  lambda s, a: make_kpeak(1,  120 if a.smoke else 600, 0.03, s),
    "kpeak_2":  lambda s, a: make_kpeak(2,  120 if a.smoke else 600, 0.03, s),
    "kpeak_4":  lambda s, a: make_kpeak(4,  120 if a.smoke else 600, 0.03, s),
    "kpeak_8":  lambda s, a: make_kpeak(8,  160 if a.smoke else 700, 0.03, s),
    "kpeak_16": lambda s, a: make_kpeak(16, 200 if a.smoke else 800, 0.03, s),
    "caffeine":   lambda s, a: D.make_nmr_dataset("caffeine", 160 if a.smoke else 1400, 0.03, s, n_dense=2800),
    "strychnine": lambda s, a: D.make_nmr_dataset("strychnine", 240 if a.smoke else 2400, 0.03, s, n_dense=4000),
}
FAMILIES = ["MLCFN", "Boost", "MoEens"]

STAGED_KNOBS = {"Boost": [1, 2, 4, 8, 16, 24], "MoEens": [1, 2, 4, 8, 16, 24]}
STAGED_GRID = {
    "Boost": [{"shallow_depth": sd, "poly_degree": pd, "lr": lr}
              for sd in (2, 4) for pd in (2, 3) for lr in (0.1, 0.5)],
    "MoEens": [{"shallow_depth": sd, "poly_degree": pd, "lr": lr}
               for sd in (2, 4) for pd in (2, 3) for lr in (1e-3, 1e-2)],
}


def _train_boost_t(model, Xtr, ytr, Xva, yva, num_stages, lr, epochs, patience, bs=64):
    crit = nn.MSELoss(); n = Xtr.shape[0]
    for _ in range(num_stages):
        model.add_model(); model.to(DEVICE); model.freeze_all_but_latest()
        opt = torch.optim.Adam([p for p in model.models[-1].parameters() if p.requires_grad], lr=lr)
        best, best_state, wait = float("inf"), None, 0
        for ep in range(epochs):
            model.train(); perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; opt.zero_grad()
                loss = crit(model(Xtr[idx]), ytr[idx])
                if not torch.isfinite(loss):
                    break
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                v = crit(model(Xva), yva).item()
            if math.isfinite(v) and v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: val.clone() for k, val in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
    return model


def _train_moe_t(model, Xtr, ytr, Xva, yva, num_experts, lr, epochs, patience, bs=64):
    crit = nn.MSELoss(); n = Xtr.shape[0]
    for _ in range(num_experts):
        if len(model.experts) == 0:
            center = Xtr.mean(dim=0).detach().cpu().numpy()
        else:
            model.eval()
            with torch.no_grad():
                res = (model(Xtr) - ytr).abs().squeeze(-1)
                center = Xtr[res.argmax()].detach().cpu().numpy()
        model.gating.add_expert_gate(center, initial_width_param=1.0, device=DEVICE)
        model.add_expert(); model.to(DEVICE)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        best, best_state, wait = float("inf"), None, 0
        for ep in range(epochs):
            model.train(); perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; opt.zero_grad()
                loss = crit(model(Xtr[idx]), ytr[idx])
                if not torch.isfinite(loss):
                    break
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                v = crit(model(Xva), yva).item()
            if math.isfinite(v) and v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: val.clone() for k, val in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
    return model


def _staged_build(family, cfg, out_dim):
    if family == "Boost":
        return V.EnsembleResCoFrNet(1, out_dim, shallow_depth=cfg["shallow_depth"],
                                    poly_degree=cfg["poly_degree"], learning_rate=0.5)
    return V.MoE_Ensemble({"input_dim": 1, "output_dim": out_dim,
                           "shallow_depth_per_cofrnet": cfg["shallow_depth"],
                           "polynomial_degree": cfg["poly_degree"]})


def _staged_train(family, model, Xtr, ytr, Xva, yva, knob, cfg, ep, pat):
    if family == "Boost":
        return _train_boost_t(model, Xtr, ytr, Xva, yva, knob, cfg["lr"], ep, pat)
    return _train_moe_t(model, Xtr, ytr, Xva, yva, knob, cfg["lr"], ep, pat)


def staged_hpo_fit(family, knob, ds, seed, args):
    X, Y, out_dim = ds["X"], ds["Y"], ds["out_dim"]
    n = len(X); idx = np.arange(n); val = idx[::4]; tr = np.setdiff1d(idx, val)
    Ytr_b, m0, s0 = _std(Y[tr])
    Xtr = torch.tensor(X[tr], device=DEVICE); ytr = torch.tensor(Ytr_b, device=DEVICE)
    Xva = torch.tensor(X[val], device=DEVICE)
    yva = torch.tensor(((Y[val] - m0) / s0).astype("float32"), device=DEVICE)
    ep = 30 if args.smoke else 150
    pat = 15 if args.smoke else 30
    best = None
    for cfg in STAGED_GRID[family]:
        set_seed(seed)
        m = _staged_build(family, cfg, out_dim).to(DEVICE)
        m = _staged_train(family, m, Xtr, ytr, Xva, yva, knob, cfg, ep, pat)
        pv = M.predict_numpy(m, X[val]) * s0 + m0
        vmse = float(np.mean((pv - Y[val]) ** 2))
        if best is None or vmse < best["vmse"]:
            best = {"cfg": cfg, "vmse": vmse}
    set_seed(seed)
    m = _staged_build(family, best["cfg"], out_dim).to(DEVICE)
    Yb, m1, s1 = _std(Y)
    Xall = torch.tensor(X, device=DEVICE); yall = torch.tensor(Yb, device=DEVICE)
    m = _staged_train(family, m, Xall, yall, Xall, yall, knob, best["cfg"], ep, pat)
    pred = M.predict_numpy(m, ds["Xeval"]) * s1 + m1
    pc = int(sum(p.numel() for p in m.parameters()))  # total size incl. frozen stages
    return {"family": family, "knob": knob, "params": pc, "r2": r2(ds["Yeval"], pred),
            "seed": seed, "best_cfg": best["cfg"]}


def knobs_for(family):
    if family == "MLCFN":
        return knobs_and_grid("MLCFN")[0]
    return STAGED_KNOBS[family]


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
            for knob in knobs_for(family):
                for seed in seeds:
                    ds = TASKS[task](seed, args)
                    if family == "MLCFN":
                        r = hpo_fit(family, knob, ds, seed, args)
                    else:
                        r = staged_hpo_fit(family, knob, ds, seed, args)
                    if r is not None:
                        r["task"] = task
                        raw.append(r)
            print(f"[staged] {task} / {family} done")
    json.dump(raw, open(OUT / f"staged_raw{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)

    g = defaultdict(list)
    for r in raw:
        g[(r["task"], r["family"], r["params"])].append(r["r2"])
    summ = [{"task": k[0], "family": k[1], "params": k[2],
             "r2_median": float(np.median(v)), "n": len(v)} for k, v in g.items()]
    json.dump(summ, open(OUT / f"staged_summary{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)
    print("\n=== min params @R2>=0.99 | best-R2 ===")
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
