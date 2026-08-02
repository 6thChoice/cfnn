"""Fair-HPO check: does a tuned MLP / KAN catch CFNN on EIS pole recovery?

Task: two_tc time-constant recovery, log-omega, 5 seeds. Every family is HPO'd
equally (config selected by held-out-frequency validation curve-MSE, then
retrained on all points, then poles recovered). Reports pole_err vs ACTUAL
params so the three families can be compared on one Pareto plane.
Metric: order-invariant pole_err_median (lower is better)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))

import eis_utils as U
import eis_models as M
from param_matched_utils import NumpySafeEncoder, count_trainable_params, find_mlp_config_for_budget

OUT = ROOT / "experiment_refine" / "eis_hpo_results"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 1024]
F_MIN, F_MAX = 1e-2, 1e5
TWO_TC = dict(Rs=10.0, R1=300.0, C1=1e-4, R2=800.0, C2=1e-6)

ACT = {"tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _std(Y):
    m = Y.mean(0, keepdims=True)
    s = Y.std(0, keepdims=True) + 1e-8
    return ((Y - m) / s).astype("float32"), m, s


class HpoMLP(nn.Module):
    def __init__(self, hidden, layers, act):
        super().__init__()
        A = ACT[act]
        L, d = [], 1
        for _ in range(layers):
            L += [nn.Linear(d, hidden), A()]
            d = hidden
        L.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def build(family, knob, cfg):
    """Return (model, actual_params). knob = MLP budget / KAN hidden / CFNN units."""
    if family == "MLP":
        c = find_mlp_config_for_budget(1, knob, 2, (cfg["depth"],), 512)
        m = HpoMLP(c["hidden_dim"], cfg["depth"], cfg["act"])
        return m, count_trainable_params(m)
    if family == "KAN":
        m = M.build_kan(1, 2, hidden=knob, grid=cfg["grid"])
        return m, (count_trainable_params(m) if m is not None else None)
    if family == "CFNN":
        m = M.CFNNHybridEIS(n_units=knob)
        return m, count_trainable_params(m)
    raise ValueError(family)


def grid(family):
    if family == "MLP":
        return [70, 200, 1000], [{"depth": dp, "act": a, "lr": lr}
                                 for dp in (2, 3) for a in ("tanh", "gelu")
                                 for lr in (1e-3, 5e-3, 1e-2)]
    if family == "KAN":
        return [6, 10, 16], [{"grid": g, "lr": lr} for g in (5,) for lr in (1e-3, 5e-3, 1e-2)]
    if family == "CFNN":
        return [2, 6, 12, 20], [{"lr": lr} for lr in (1e-3, 5e-3, 1e-2)]
    raise ValueError(family)


def _train(model, Xtr, Ytr, Xva, Yva, lr, epochs, patience):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_sd = deepcopy(model.state_dict())
    best, stale = float("inf"), 0
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = crit(model(Xtr), Ytr)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                v = crit(model(Xva), Yva).item()
            if math.isfinite(v) and v < best - 1e-6:
                best, stale = v, 0
                best_sd = deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
    model.load_state_dict(best_sd)
    return model


def hpo_fit(family, knob, ds, dense_f, seed, args):
    """HPO over the family's grid (val = held-out frequencies), retrain best on
    all points, recover poles. Returns a result row."""
    X, Y, f = ds["X"], ds["Y"], ds["f"]
    n = len(f)
    idx = np.arange(n)
    val = idx[::3]                      # ~33% held-out frequencies for selection
    tr = np.setdiff1d(idx, val)
    Ytr_b, m0, s0 = _std(Y[tr])
    Xtr = torch.tensor(X[tr], device=M.DEVICE)
    Ytr = torch.tensor(Ytr_b, device=M.DEVICE)
    Xva = torch.tensor(X[val], device=M.DEVICE)
    Yva = torch.tensor(((Y[val] - m0) / s0).astype("float32"), device=M.DEVICE)

    best = None
    _, cfgs = grid(family)
    ep_sel = 60 if args.smoke else 400
    for cfg in cfgs:
        set_seed(seed)
        model, pc = build(family, knob, cfg)
        if model is None:
            return None
        model = model.to(M.DEVICE)
        model = _train(model, Xtr, Ytr, Xva, Yva, cfg["lr"], ep_sel, patience=50)
        pv = M.predict_numpy(model, X[val]) * s0 + m0
        vmse = float(np.mean((pv - Y[val]) ** 2))
        if best is None or vmse < best["vmse"]:
            best = {"cfg": cfg, "vmse": vmse, "params": pc}

    # retrain best config on ALL points
    set_seed(seed)
    model, pc = build(family, knob, best["cfg"])
    model = model.to(M.DEVICE)
    Yb, m1, s1 = _std(Y)
    Xall = torch.tensor(X, device=M.DEVICE)
    Yall = torch.tensor(Yb, device=M.DEVICE)
    ep_final = 60 if args.smoke else 600
    model = _train(model, Xall, Yall, Xall, Yall, best["cfg"]["lr"], ep_final, patience=80)
    pred = M.predict_numpy(model, U.omega_features(dense_f, "log", F_MIN, F_MAX)) * s1 + m1
    Zp = pred[:, 0] + 1j * pred[:, 1]
    rec = U.recover_params_from_curve(dense_f, Zp, "two_tc")
    pe = U.pole_recovery_error("two_tc", rec["params"], ds["params"])
    return {"family": family, "knob": knob, "params": pc, "best_cfg": best["cfg"],
            "pole_err_median": pe["median"], "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dense_f = U.freq_grid(F_MIN, F_MAX, 400)
    seeds = SEEDS[:2] if args.smoke else SEEDS
    n_points = 40 if args.smoke else 80
    raw = []
    for family in ["CFNN", "MLP", "KAN"]:
        knobs, _ = grid(family)
        for knob in knobs:
            for seed in seeds:
                set_seed(seed)
                ds = U.make_dataset("two_tc", TWO_TC, F_MIN, F_MAX,
                                    n_points=n_points, sigma_rel=0.02, seed=seed, mode="log")
                r = hpo_fit(family, knob, ds, dense_f, seed, args)
                if r is not None:
                    raw.append(r)
        print(f"[hpo] {family} done")
    json.dump(raw, open(OUT / "hpo_raw.json", "w"), indent=2, cls=NumpySafeEncoder)

    # aggregate: median pole_err by (family, knob) with actual params
    from collections import defaultdict
    g = defaultdict(list)
    for r in raw:
        g[(r["family"], r["knob"], r["params"])].append(r["pole_err_median"])
    summary = [{"family": k[0], "knob": k[1], "params": k[2],
                "pole_err_median": float(np.median(v)), "n": len(v)}
               for k, v in g.items()]
    json.dump(summary, open(OUT / "hpo_summary.json", "w"), indent=2, cls=NumpySafeEncoder)
    print("\n=== HPO Pareto: pole_err_median (↓) vs actual params, per family ===")
    for row in sorted(summary, key=lambda x: (x["family"], x["params"])):
        print(f"  {row['family']:>5}  params={row['params']:>5}  pole_err={row['pole_err_median']:.3f}  (n={row['n']})")


if __name__ == "__main__":
    main()
