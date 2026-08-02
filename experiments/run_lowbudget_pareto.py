"""Low-budget Pareto: at tiny parameter counts, can CFNN's rational bias fit
sharp/pole structure where an equally-tiny MLP cannot?

Tasks:
  - synthetic sharp pole  y = 1/(x^2 + eps),  eps in {0.01, 0.001}  (1D->1D)
  - EIS two_tc (downstream anchor)                                   (1D->2D)
All three families (CFNN / MLP / KAN) are HPO'd equally (config picked by
held-out validation, retrained on all points). Metric = R^2 on a dense CLEAN
grid (fit quality of the true function). Reports R^2 vs ACTUAL params."""
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
import cofrnet_variants as V
from param_matched_utils import NumpySafeEncoder, count_trainable_params, find_mlp_config_for_budget

OUT = ROOT / "experiment_refine" / "lowbudget_pareto_results"
OUT.mkdir(parents=True, exist_ok=True)
# 5 seeds (full rigor) -- the fair 8-cfg sweep runs on a dedicated 22GB GPU
# server, so no need to trade seeds for speed.
SEEDS = [42, 123, 456, 789, 1024]
ACT = {"tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}
EIS_TWO_TC = dict(Rs=10.0, R1=300.0, C1=1e-4, R2=800.0, C2=1e-6)
F_MIN, F_MAX = 1e-2, 1e5


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _std(Y):
    m = Y.mean(0, keepdims=True)
    s = Y.std(0, keepdims=True) + 1e-8
    return ((Y - m) / s).astype("float32"), m, s


def r2(y_true, y_pred):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


# ── task datasets ────────────────────────────────────────────────────────
def make_pole(eps, n_points, noise, seed):
    """y = 1/(x^2+eps) on x in [-3,3]; x normalized to [-1,1]; dense clean eval."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-3.0, 3.0, n_points).astype(np.float32)
    y = (1.0 / (x * x + eps)).astype(np.float32)
    # multiplicative (relative) noise: does not drown the informative low-y tail
    yn = (y * (1.0 + noise * rng.standard_normal(y.shape).astype(np.float32))).astype(np.float32)
    Xd = np.linspace(-3.0, 3.0, 500).astype(np.float32)
    Yd = (1.0 / (Xd * Xd + eps)).astype(np.float32)
    return {"X": (x / 3.0).reshape(-1, 1), "Y": yn.reshape(-1, 1),
            "Xeval": (Xd / 3.0).reshape(-1, 1), "Yeval": Yd.reshape(-1, 1),
            "out_dim": 1, "kind": "scalar"}


def make_eis(n_points, noise, seed):
    ds = U.make_dataset("two_tc", EIS_TWO_TC, F_MIN, F_MAX, n_points, noise, seed, "log")
    dense_f = U.freq_grid(F_MIN, F_MAX, 400)
    Zt = U.voigt_impedance(dense_f, EIS_TWO_TC["Rs"], [EIS_TWO_TC["R1"], EIS_TWO_TC["R2"]],
                           [EIS_TWO_TC["C1"], EIS_TWO_TC["C2"]])
    Yeval = np.stack([Zt.real, Zt.imag], axis=1).astype(np.float32)
    return {"X": ds["X"], "Y": ds["Y"], "Xeval": U.omega_features(dense_f, "log", F_MIN, F_MAX),
            "Yeval": Yeval, "out_dim": 2, "kind": "complex"}


# ── models (activation-configurable MLP; CFNN/KAN from eis_models) ────────
class HpoMLP(nn.Module):
    def __init__(self, out_dim, hidden, layers, act):
        super().__init__()
        A = ACT[act]
        L, d = [], 1
        for _ in range(layers):
            L += [nn.Linear(d, hidden), A()]
            d = hidden
        L.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def build(family, knob, cfg, out_dim):
    if family == "MLP":
        c = find_mlp_config_for_budget(1, knob, out_dim, (cfg["depth"],), 512)
        m = HpoMLP(out_dim, c["hidden_dim"], cfg["depth"], cfg["act"])
        return m, count_trainable_params(m)
    if family == "KAN":
        m = M.build_kan(1, out_dim, hidden=knob, grid=cfg["grid"])
        return m, (count_trainable_params(m) if m is not None else None)
    if family == "CFNN":
        m = M.CFNNHybridEIS(input_dim=1, output_dim=out_dim, n_units=knob, degree=cfg["degree"])
        return m, count_trainable_params(m)
    if family == "CFNNMoE":
        m = M.CFNNMoE(input_dim=1, output_dim=out_dim, n_experts=knob,
                      degree=cfg["degree"], gate_hidden=cfg["gate_hidden"])
        return m, count_trainable_params(m)
    if family == "CFNetStd":  # nested polynomial continued fraction (original-style)
        m = V.CFNet_Standard(1, out_dim, depth=knob, poly_degree=cfg["poly_degree"])
        return m, count_trainable_params(m)
    if family == "CoFrNetLadder":  # original-paper scalar-reciprocal ladder
        m = V.CoFrNetDL(1, out_dim, num_full_ladders=knob,
                        max_diag_ladder_depth=cfg["max_depth"],
                        max_full_ladder_depth=cfg["max_depth"])
        return m, count_trainable_params(m)
    if family == "MLCFN":  # continued-fraction as element-wise activation (KAN-style)
        m = V.MultiLayerCFN(1, out_dim, hidden_dims=[knob],
                            cfn_depth=cfg["cfn_depth"],
                            cfn_num_basis_functions_per_term=cfg["nb"])
        return m, count_trainable_params(m)
    raise ValueError(family)


def knobs_and_grid(family):
    # Fair HPO (2026-07-06 fairness review): all three families get 8 configs;
    # KAN's grid_size -- its key capacity knob for sharp peaks -- is TUNED, not
    # frozen at 5; knob ranges reach comparable param ceilings (~1.2k-1.9k) so
    # "min params to R2>=0.99" is not a sampling artifact of one family topping
    # out early. Previously: MLP 8 cfgs / CFNN 4 / KAN 2 with grid frozen at 5,
    # which systematically favored MLP and crippled KAN on sharp observables.
    # Knob ladders trimmed to lean sets that still span ~13..~1300 params per
    # family (ceiling parity) while keeping the 8-config HPO budget symmetric.
    if family == "MLP":
        # fine low-end sampling (25, 80) restored so MLP's min-params threshold
        # isn't overstated by a gap between 13 and 53 params -- fairness cuts
        # both ways.
        return [12, 25, 45, 80, 150, 300, 600, 1200], [
            {"depth": dp, "act": a, "lr": lr}
            for dp in (2, 3) for a in ("tanh", "gelu") for lr in (3e-3, 1e-2)]
    if family == "KAN":
        # fine low-end (3,6,10) restored for the same reason as MLP: don't
        # overstate KAN's min-params by leaving a gap around its threshold.
        return [2, 3, 4, 6, 8, 10, 16, 24], [
            {"grid": g, "lr": lr} for g in (5, 10, 15, 20) for lr in (3e-3, 1e-2)]
    if family == "CFNN":
        # capped at 32 units (~700 params): rational units stack sequentially and
        # get pathologically slow past that, and CFNN's whole point is param
        # frugality at LOW budget -- its high-param ceiling is genuinely limited,
        # not a sampling gap. This does not affect the low-budget economy claim.
        return [1, 2, 4, 8, 16, 24, 32], [
            {"degree": dg, "lr": lr} for dg in (2, 3, 5, 8) for lr in (3e-3, 1e-2)]
    if family == "CFNNMoE":
        # knob = number of rational experts; 8 configs (degree x gate_hidden x lr)
        # to match the fair HPO budget. Gate params ARE counted, so the Pareto
        # honestly charges for the mixture machinery.
        return [1, 2, 4, 8, 16, 24, 32], [
            {"degree": dg, "gate_hidden": gh, "lr": lr}
            for dg in (3, 5) for gh in (4, 8) for lr in (3e-3, 1e-2)]
    if family == "CFNetStd":
        # knob = nested-CF depth; 8 configs (poly_degree x lr). Nested CF is very
        # param-frugal (deep -> unstable abs() denominator), so params stay low.
        return [1, 2, 4, 8, 16, 32], [
            {"poly_degree": pd, "lr": lr} for pd in (2, 3, 4, 5) for lr in (3e-3, 1e-2)]
    if family == "CoFrNetLadder":
        # knob = number of full ladders; 8 configs (max ladder depth x lr).
        return [1, 2, 4, 8, 16, 24, 32], [
            {"max_depth": md, "lr": lr} for md in (4, 8, 12, 16) for lr in (3e-3, 1e-2)]
    if family == "MLCFN":
        # knob = hidden width; 8 configs (cfn_depth x num_basis x lr).
        return [1, 2, 4, 8, 16], [
            {"cfn_depth": d, "nb": nb, "lr": lr}
            for d in (2, 3) for nb in (3, 5) for lr in (3e-3, 1e-2)]
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
            if math.isfinite(v) and v < best - 1e-7:
                best, stale = v, 0
                best_sd = deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
    model.load_state_dict(best_sd)
    return model


def hpo_fit(family, knob, ds, seed, args):
    X, Y = ds["X"], ds["Y"]
    out_dim = ds["out_dim"]
    n = len(X)
    idx = np.arange(n)
    val = idx[::4]
    tr = np.setdiff1d(idx, val)
    Ytr_b, m0, s0 = _std(Y[tr])
    Xtr = torch.tensor(X[tr], device=M.DEVICE)
    Ytr = torch.tensor(Ytr_b, device=M.DEVICE)
    Xva = torch.tensor(X[val], device=M.DEVICE)
    Yva = torch.tensor(((Y[val] - m0) / s0).astype("float32"), device=M.DEVICE)
    # Full epoch budget with early stopping: fast-converging models (CFNN) stop
    # early on their own; slower ones (MLP) get the epochs they need. A uniform
    # epoch CUT is NOT fair -- it handicaps slower-converging families more, so
    # we keep the generous ceiling and let patience do the trimming.
    ep_sel = 80 if args.smoke else 500
    ep_fin = 80 if args.smoke else 800
    best = None
    _, cfgs = knobs_and_grid(family)
    for cfg in cfgs:
        set_seed(seed)
        model, pc = build(family, knob, cfg, out_dim)
        if model is None:
            return None
        model = model.to(M.DEVICE)
        model = _train(model, Xtr, Ytr, Xva, Yva, cfg["lr"], ep_sel, 50)
        pv = M.predict_numpy(model, X[val]) * s0 + m0
        vmse = float(np.mean((pv - Y[val]) ** 2))
        if best is None or vmse < best["vmse"]:
            best = {"cfg": cfg, "vmse": vmse, "params": pc}
    # retrain best on all points
    set_seed(seed)
    model, pc = build(family, knob, best["cfg"], out_dim)
    model = model.to(M.DEVICE)
    Yb, m1, s1 = _std(Y)
    Xall = torch.tensor(X, device=M.DEVICE)
    Yall = torch.tensor(Yb, device=M.DEVICE)
    model = _train(model, Xall, Yall, Xall, Yall, best["cfg"]["lr"], ep_fin, 80)
    pred = M.predict_numpy(model, ds["Xeval"]) * s1 + m1
    score = r2(ds["Yeval"], pred)
    return {"family": family, "knob": knob, "params": pc, "best_cfg": best["cfg"],
            "r2": score, "seed": seed}


TASKS = {
    "pole_eps0.01": lambda seed, args: make_pole(0.01, 40 if args.smoke else 200, 0.03, seed),
    "pole_eps0.001": lambda seed, args: make_pole(0.001, 60 if args.smoke else 300, 0.03, seed),
    "eis_two_tc": lambda seed, args: make_eis(40 if args.smoke else 80, 0.02, seed),
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
            print(f"[pareto] {task} / {family} done")
    json.dump(raw, open(OUT / f"pareto_raw{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)

    from collections import defaultdict
    g = defaultdict(list)
    for r in raw:
        g[(r["task"], r["family"], r["params"])].append(r["r2"])
    summ = [{"task": k[0], "family": k[1], "params": k[2],
             "r2_median": float(np.median(v)), "n": len(v)} for k, v in g.items()]
    json.dump(summ, open(OUT / f"pareto_summary{suf}.json", "w"), indent=2, cls=NumpySafeEncoder)
    for task in tasks:
        print(f"\n=== {task}: R2_median (↑) vs actual params ===")
        for row in sorted([s for s in summ if s["task"] == task], key=lambda x: (x["family"], x["params"])):
            print(f"  {row['family']:>5} params={row['params']:>5} R2={row['r2_median']:.4f} (n={row['n']})")


if __name__ == "__main__":
    main()
