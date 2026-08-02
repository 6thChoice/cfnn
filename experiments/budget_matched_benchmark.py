"""Strict realized-parameter benchmark.

Unlike the legacy runner, this module never converts a nominal budget through
family-specific divisors. Architectures are instantiated, counted, assigned
to one budget at most, and only then trained.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score

from budget_matching import Candidate, assign_unique_candidates, serialize_assignment
from fair_baselines import (build_cofrnet, build_gaussian_rbf, build_mlp,
                            build_pykan, build_rational_nn, build_siren)
from eis_models import CFNNHybridEIS
from run_independent_benchmark import targets

FAMILIES = ["CFNN-Hybrid", "MLP", "Gaussian RBF", "KAN",
            "CoFrNet-Standard", "Rational activation NN", "SIREN"]
BUDGETS = [32, 64, 128, 256, 512, 1024]
SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(family, d, out, arch, seed=1):
    if family == "CFNN-Hybrid":
        return CFNNHybridEIS(d, out, arch["units"], arch["degree"]), arch
    if family == "MLP":
        return build_mlp(d, out, arch["width"], arch["depth"], arch["activation"])
    if family == "Gaussian RBF":
        return build_gaussian_rbf(d, out, arch["basis"])
    if family == "KAN":
        return build_pykan(d, out, arch["width"], arch["grid"], arch["k"], device=str(DEVICE), seed=seed)
    if family == "CoFrNet-Standard":
        return build_cofrnet(d, out, arch["depth"], arch["degree"])
    if family == "Rational activation NN":
        return build_rational_nn(d, out, arch["width"], arch["depth"], arch["degree"])
    if family == "SIREN":
        return build_siren(d, out, arch["width"], arch["depth"], arch["omega"])
    raise ValueError(family)


def architecture_candidates(family, d, out):
    candidates = []
    if family == "CFNN-Hybrid":
        grid = ({"units": u, "degree": p} for u in range(1, 65) for p in (1, 2, 3, 4, 5, 8))
    elif family == "MLP":
        grid = ({"width": w, "depth": dep, "activation": act}
                for w in range(2, 145) for dep in (1, 2, 3)
                for act in ("tanh", "gelu"))
    elif family == "Gaussian RBF":
        grid = ({"basis": b} for b in range(2, 257))
    elif family == "KAN":
        grid = ({"width": w, "grid": g, "k": k}
                for w in range(2, 65) for g in (3, 5, 7, 10) for k in (3, 5))
    elif family == "CoFrNet-Standard":
        grid = ({"depth": dep, "degree": p}
                for dep in range(1, 65) for p in (1, 2, 3, 4, 5, 8))
    elif family == "Rational activation NN":
        grid = ({"width": w, "depth": dep, "degree": p}
                for w in range(2, 145) for dep in (1, 2, 3) for p in (1, 2, 3, 5))
    elif family == "SIREN":
        grid = ({"width": w, "depth": dep, "omega": om}
                for w in range(2, 145) for dep in (1, 2, 3) for om in (10., 30., 60.))
    else:
        raise ValueError(family)
    seen = set()
    for arch in grid:
        model, _ = build(family, d, out, arch)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        key = json.dumps(arch, sort_keys=True)
        if key not in seen:
            candidates.append(Candidate(key, count, arch))
            seen.add(key)
    return candidates


def fit_select(model, xtr, ytr, xv, yv, lr, weight_decay, max_epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.MSELoss(); best_val = float("inf"); best_epoch = 1; stale = 0
    for epoch in range(max_epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        value = loss(model(xtr), ytr); value.backward()
        if not torch.isfinite(value):
            return 1, float("inf"), "nonfinite_train_loss"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad(): val = loss(model(xv), yv).item()
        if not np.isfinite(val):
            return 1, float("inf"), "nonfinite_validation_loss"
        if val < best_val:
            best_val, best_epoch, stale = val, epoch + 1, 0
        else:
            stale += 1
        if stale >= 50:
            break
    return best_epoch, best_val, "ok"


def refit(model, x, y, epochs, lr, weight_decay, seed):
    torch.manual_seed(seed + 1000003)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.MSELoss()
    for _ in range(max(1, epochs)):
        opt.zero_grad(set_to_none=True); value = loss(model(x), y)
        value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", choices=["independent", "noniso", "core", "nmr", "energy", "all"], default="independent")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--lrs", default="0.003,0.01")
    ap.add_argument("--weight-decays", default="0.0")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if args.smoke:
        seeds, args.max_epochs = seeds[:1], 5
    lrs = [float(x) for x in args.lrs.split(",")]
    weight_decays = [float(x) for x in args.weight_decays.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]
    independent = [("gaussian_peaks", 1), ("fourier", 1), ("nonseparable", 3),
                   ("cusp", 1), ("kink", 1), ("compact_bump", 1),
                   ("skew_peak", 1), ("manifold_bump", 3)]
    core = [("pole_sharp", 1), ("pole_broad", 1), ("resonance", 1)]
    noniso = [("cusp", 1), ("compact_bump", 1), ("manifold_bump", 3)]
    energy = [("energy", 8)]
    nmr = [("nmr_ethanol", 1), ("nmr_caffeine", 1), ("nmr_strychnine", 1)]
    tasks = (independent if args.tasks == "independent" else noniso if args.tasks == "noniso"
             else core if args.tasks == "core" else nmr if args.tasks == "nmr" else energy if args.tasks == "energy"
             else independent + noniso + core + nmr + energy)
    records = []; out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text("[]")
    assignment_cache = {}
    for task, d in tasks:
        x, y = targets(task, d=d); n = len(x); i, j = int(.6*n), int(.8*n)
        xt, xv, xe = [torch.from_numpy(z).to(DEVICE) for z in (x[:i], x[i:j], x[j:])]
        ym, ys = y[:i].mean(0, keepdims=True), y[:i].std(0, keepdims=True) + 1e-6
        yn = (y - ym) / ys; yt, yv, ye = [torch.from_numpy(z).to(DEVICE) for z in (yn[:i], yn[i:j], yn[j:])]
        for family in FAMILIES:
            cache_key = (task, family, d, y.shape[1])
            candidates = architecture_candidates(family, d, y.shape[1])
            assignment = assign_unique_candidates(candidates, budgets, args.tolerance)
            assignment_cache[str(cache_key)] = serialize_assignment(assignment)
            for budget, candidate in assignment.items():
                if candidate is None:
                    continue
                arch = candidate.config
                for seed in seeds:
                    started = time.time(); best = None
                    if DEVICE.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(DEVICE)
                    for lr in lrs:
                        for wd in weight_decays:
                            model, _ = build(family, d, y.shape[1], arch, seed)
                            epoch, val, status = fit_select(model, xt, yt, xv, yv, lr, wd, args.max_epochs, seed)
                            if best is None or val < best["val"]:
                                best = {"lr": lr, "weight_decay": wd, "epoch": epoch, "val": val, "status": status}
                    final, _ = build(family, d, y.shape[1], arch, seed + 1000003)
                    final = refit(final, torch.cat([xt, xv]), torch.cat([yt, yv]), best["epoch"], best["lr"], best["weight_decay"], seed)
                    final.eval()
                    with torch.no_grad(): pred = final(xe).detach().cpu().numpy()
                    records.append({
                        "task": task, "family": family, "target_budget": budget,
                        "budget_tolerance": args.tolerance, "seed": seed,
                        "candidate_key": candidate.key, "config": arch,
                        "actual_parameters": candidate.actual_parameters,
                        "budget_relative_error": abs(candidate.actual_parameters-budget)/budget,
                        "selected_hyperparameters": best,
                        "r2": float(r2_score(ye.detach().cpu().numpy(), pred, multioutput="variance_weighted")),
                        "wall_seconds": time.time() - started,
                        "optimizer_steps": best["epoch"], "device": str(DEVICE),
                        "peak_memory_bytes": (torch.cuda.max_memory_allocated(DEVICE) if DEVICE.type == "cuda" else None),
                        "valid_match": True,
                    })
                    out.write_text(json.dumps({"records": records, "assignments": assignment_cache}, indent=2))
                    print(task, family, budget, seed, records[-1]["actual_parameters"], records[-1]["r2"], flush=True)


if __name__ == "__main__":
    main()
