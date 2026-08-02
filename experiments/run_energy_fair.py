"""Fair re-audit of the Energy Efficiency downstream benchmark (UCI ENB2012,
real tabular data, 8 features -> heating load).

The prior repo result had CFNN-Hybrid (R2 0.9976) edging a FIXED, un-tuned MLP
(hidden=64,2 layers, R2 0.9970). This runner re-audits under the fair protocol we
established: every family (incl. a properly HPO'd MLP and a shape-fixed KAN) gets
the SAME held-out-validation HPO budget, 5 seeds, param counts reported. Question:
does CFNN-Hybrid's downstream win survive a tuned MLP + fixed KAN?

Deferred: Boost / MoE_Ensemble need custom staged trainers (add_model/add_expert),
handled separately. --family for parallelism."""
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
import eis_models as M
import cofrnet_variants as V
from param_matched_utils import count_trainable_params as C, NumpySafeEncoder

CSV = ROOT / "experiment_refine" / "energy_efficiency.csv"
OUT = ROOT / "experiment_refine" / "energy_fair_results"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 1024]
DEVICE = M.DEVICE
ACT = {"tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}
FAMILIES = ["MLP", "Hybrid", "CFNetStd", "CoFrNetLadder", "KAN"]


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_energy(seed):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(CSV)
    cols = [c for c in df.columns if c.upper().startswith("X")][:8]
    ycol = [c for c in df.columns if c.upper() == "Y1"][0]
    X = df[cols].values.astype("float32")
    y = df[[ycol]].values.astype("float32")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.10, random_state=seed)
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(ytr)
    return (xs.transform(Xtr).astype("float32"), ys.transform(ytr).astype("float32"),
            xs.transform(Xva).astype("float32"), ys.transform(yva).astype("float32"),
            xs.transform(Xte).astype("float32"), yte, ys)


class MLP(nn.Module):
    def __init__(self, ind, hid, lay, act):
        super().__init__()
        L, d = [], ind
        for _ in range(lay):
            L += [nn.Linear(d, hid), ACT[act]()]
            d = hid
        L.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def build(fam, knob, cfg, ind=8):
    if fam == "MLP":
        return MLP(ind, knob, cfg["depth"], cfg["act"])
    if fam == "Hybrid":
        return M.CFNNHybridEIS(ind, 1, n_units=knob, degree=cfg["degree"])
    if fam == "CFNetStd":
        return V.CFNet_Standard(ind, 1, depth=knob, poly_degree=cfg["poly_degree"])
    if fam == "CoFrNetLadder":
        return V.CoFrNetDL(ind, 1, num_full_ladders=knob,
                           max_diag_ladder_depth=cfg["max_depth"],
                           max_full_ladder_depth=cfg["max_depth"])
    if fam == "KAN":
        return M.build_kan(ind, 1, hidden=knob, grid=cfg["grid"])
    raise ValueError(fam)


GRID = {
    "MLP": ([8, 16, 32, 64, 128], [{"depth": dp, "act": a, "lr": lr}
            for dp in (2, 3) for a in ("tanh", "gelu") for lr in (3e-3, 1e-2)]),
    "Hybrid": ([2, 4, 8, 16, 20], [{"degree": dg, "lr": lr}
               for dg in (2, 3, 4, 5) for lr in (3e-3, 1e-2)]),
    "CFNetStd": ([2, 4, 8, 16], [{"poly_degree": pd, "lr": lr}
                 for pd in (2, 3, 4, 5) for lr in (3e-3, 1e-2)]),
    "CoFrNetLadder": ([4, 8, 16, 24], [{"max_depth": md, "lr": lr}
                      for md in (4, 8, 12) for lr in (3e-3, 1e-2)]),
    "KAN": ([4, 8, 16, 24], [{"grid": g, "lr": lr}
            for g in (5, 10, 20) for lr in (3e-3, 1e-2)]),
}


def r2(y_true, y_pred):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    ss = float(np.sum((yt - yp) ** 2))
    tot = float(np.sum((yt - yt.mean()) ** 2)) + 1e-12
    return 1.0 - ss / tot


def train(model, Xtr, ytr, Xva, yva, lr, epochs=2000, patience=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best_sd, best, stale = deepcopy(model.state_dict()), float("inf"), 0
    Xt = torch.tensor(Xtr, device=DEVICE); yt = torch.tensor(ytr, device=DEVICE)
    Xv = torch.tensor(Xva, device=DEVICE); yv = torch.tensor(yva, device=DEVICE)
    for ep in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        loss = crit(model(Xt), yt)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                v = crit(model(Xv), yv).item()
            if math.isfinite(v) and v < best - 1e-7:
                best, stale, best_sd = v, 0, deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
    model.load_state_dict(best_sd)
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, device=DEVICE)).cpu().numpy()


def run_family(fam, args):
    knobs, cfgs = GRID[fam]
    rows = []
    for seed in SEEDS[:2] if args.smoke else SEEDS:
        Xtr, ytr, Xva, yva, Xte, yte, ys = load_energy(seed)
        # HPO: pick (knob,cfg) by val R2 (standardized), retrain, eval test (raw scale)
        best = None
        for knob in knobs:
            for cfg in cfgs:
                set_seed(seed)
                m = build(fam, knob, cfg, ind=Xtr.shape[1])
                if m is None:
                    return []
                m = m.to(DEVICE)
                ep = 150 if args.smoke else 2000
                m = train(m, Xtr, ytr, Xva, yva, cfg["lr"], epochs=ep, patience=80)
                pv = predict(m, Xva)
                vr2 = r2(yva, pv)
                if best is None or vr2 > best["vr2"]:
                    best = {"knob": knob, "cfg": cfg, "vr2": vr2, "params": C(m)}
        set_seed(seed)
        m = build(fam, best["knob"], best["cfg"], ind=Xtr.shape[1]).to(DEVICE)
        m = train(m, Xtr, ytr, Xva, yva, best["cfg"]["lr"],
                  epochs=150 if args.smoke else 2000, patience=80)
        pred_std = predict(m, Xte)
        pred = ys.inverse_transform(pred_std)
        rows.append({"family": fam, "seed": seed, "params": best["params"],
                     "test_r2": r2(yte, pred),
                     "test_mse": float(np.mean((yte.ravel() - pred.ravel()) ** 2)),
                     "best_knob": best["knob"], "best_cfg": best["cfg"]})
        print(f"[energy] {fam} seed={seed} params={best['params']} testR2={rows[-1]['test_r2']:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--family", default="all", choices=FAMILIES + ["all"])
    args = ap.parse_args()
    fams = FAMILIES if args.family == "all" else [args.family]
    suf = "" if args.family == "all" else f"_{args.family}"
    raw = []
    for fam in fams:
        raw.extend(run_family(fam, args))
    json.dump(raw, open(OUT / f"energy_raw{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)
    print("\n=== Energy Efficiency (fair HPO, test R2, 5 seeds) ===")
    for fam in fams:
        rs = [r["test_r2"] for r in raw if r["family"] == fam]
        ps = [r["params"] for r in raw if r["family"] == fam]
        if rs:
            print(f"  {fam:>14}: R2 {np.mean(rs):.4f} +/- {np.std(rs):.4f}  (params~{int(np.median(ps))})")


if __name__ == "__main__":
    main()
