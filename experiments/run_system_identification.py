#!/usr/bin/env python3
"""Experiment 5: System Identification — Transfer Function Fitting.

Rational transfer functions are the canonical representation of linear systems:
    H(s) = (b₀ + b₁s + ... + bₙsⁿ) / (1 + a₁s + ... + aₘsᵐ)

CFNN's rational units naturally match this structure, while MLP needs many
parameters to approximate the rational frequency response.

Design:
1. Generate frequency response of rational transfer functions (Bode plots)
2. Compare CFNN vs MLP at matched parameter budgets
3. Test different system orders (1st order, 2nd order, higher order)

Transfer functions tested:
- 1st order low-pass: H(s) = 1/(1 + τs)
- 2nd order resonant: H(s) = ω₀²/(s² + 2ζω₀s + ω₀²)
- 3rd order: H(s) = (1 + τs) / ((1 + τ₁s)(1 + τ₂s)(1 + τ₃s))
"""
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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))
from param_matched_utils import NumpySafeEncoder, count_trainable_params, find_mlp_config_for_budget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "experiment_refine" / "system_identification_results"
SEEDS = [42, 123, 456, 789, 1024]

SYSTEMS = [
    {"name": "first_order_lp", "desc": "H(s) = 1/(1 + s)", "func": lambda s: 1 / (1 + 1j * s)},
    {"name": "second_order_crit", "desc": "H(s)=1/(s² + 2s + 1), ζ=1", "func": lambda s: 1 / ((1j * s)**2 + 2*1j*s + 1)},
    {"name": "second_order_under", "desc": "H(s)=4/(s² + 0.8s + 4), ζ=0.2", "func": lambda s: 4 / ((1j * s)**2 + 0.8*1j*s + 4)},
    {"name": "second_order_over", "desc": "H(s)=1/(s² + 4s + 1), ζ=2", "func": lambda s: 1 / ((1j * s)**2 + 4*1j*s + 1)},
    {"name": "third_order", "desc": "H(s)=1/((1+s)(1+0.5s)(1+0.2s))", "func": lambda s: 1 / ((1 + 1j*s)*(1 + 0.5j*s)*(1 + 0.2j*s))},
]


def mag_db(h: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.abs(h) + 1e-30)


def phase_deg(h: np.ndarray) -> np.ndarray:
    return np.angle(h, deg=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ─── Models ────────────────────────────────────────────────────────────

class BudgetMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class RationalUnit(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.P_proj = nn.Linear(input_dim, output_dim)
        self.P_coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * 0.05)
        self.Q_proj = nn.Linear(input_dim, output_dim)
        self.Q_coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * 0.05)
        self.degree = degree
    def forward(self, x):
        def _poly(proj, coeffs):
            z = torch.tanh(proj(x))
            powers = [torch.ones_like(z)]
            for d in range(1, self.degree + 1):
                powers.append(powers[-1] * z)
            return torch.sum(torch.stack(powers, dim=-1) * coeffs, dim=2)
        return _poly(self.P_proj, self.P_coeffs) / (_poly(self.Q_proj, self.Q_coeffs) ** 2 + 0.1)

class CFNNHybrid(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.linear_skip = nn.Linear(input_dim, output_dim)
        self.units = nn.ModuleList([RationalUnit(input_dim, output_dim, 4) for _ in range(8)])
    def forward(self, x):
        out = self.linear_skip(x)
        for u in self.units:
            out = out + u(x)
        return out


# ─── Training ──────────────────────────────────────────────────────────

def train_model(model, X_tr, y_tr, X_val, y_val, epochs, lr, wd=1e-5, patience=60):
    if y_tr.dim() == 1: y_tr = y_tr.unsqueeze(-1)
    if y_val.dim() == 1: y_val = y_val.unsqueeze(-1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_sd = deepcopy(model.state_dict())
    best_val, stale = float("inf"), 0
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(X_tr)
        if pred.dim() == 1: pred = pred.unsqueeze(-1)
        loss = crit(pred, y_tr)
        if not torch.isfinite(loss): break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vpred = model(X_val)
                if vpred.dim() == 1: vpred = vpred.unsqueeze(-1)
                v = crit(vpred, y_val).item()
            if math.isfinite(v) and v < best_val - 1e-6:
                best_val, stale = v, 0
                best_sd = deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience: break
    model.load_state_dict(best_sd)
    return model


def grid_train(builder, X_tr, y_tr, X_val, y_val, args):
    if y_tr.dim() == 1: y_tr = y_tr.unsqueeze(-1)
    if y_val.dim() == 1: y_val = y_val.unsqueeze(-1)
    crit_eval = nn.MSELoss()
    best_val, best_m, best_cfg = float("inf"), None, {}
    for lr in args.lrs:
        for wd in args.weight_decays:
            m = builder().to(DEVICE)
            m = train_model(m, X_tr, y_tr, X_val, y_val, args.epochs, lr, wd, args.patience)
            m.eval()
            with torch.no_grad():
                mpred = m(X_val)
                if mpred.dim() == 1: mpred = mpred.unsqueeze(-1)
                v = float(crit_eval(mpred, y_val).item())
            if v < best_val:
                best_val, best_m, best_cfg = v, m, {"lr": lr, "wd": wd}
    return best_m, best_cfg


# ─── Run one system ────────────────────────────────────────────────────

def run_one(system: dict, n_points: int, noise_db: float, seed: int, args) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # Log-spaced frequencies (typical Bode plot)
    freqs = np.logspace(np.log10(0.01), np.log10(100), n_points, dtype=np.float32)
    h = np.array([system["func"](f) for f in freqs], dtype=np.complex128)
    y = mag_db(h).reshape(-1, 1).astype(np.float32)  # magnitude in dB

    # Add noise (in dB)
    y += rng.normal(0, noise_db, y.shape).astype(np.float32)

    # Log frequency as input
    X = np.log10(freqs + 1e-10).reshape(-1, 1).astype(np.float32)

    # Split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=seed)

    # Standardize
    sx, sy = StandardScaler(), StandardScaler()
    X_tr_t = torch.tensor(sx.fit_transform(X_tr), dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(sx.transform(X_val), dtype=torch.float32, device=DEVICE)
    X_te_t = torch.tensor(sx.transform(X_te), dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(sy.fit_transform(y_tr), dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(sy.transform(y_val), dtype=torch.float32, device=DEVICE)
    y_te_o = y_te  # original scale

    results = {}

    # CFNN-Hybrid
    cfnn, cfnn_cfg = grid_train(lambda: CFNNHybrid(), X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    cfnn.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(cfnn(X_te_t).cpu().numpy().reshape(-1, 1))
    results["CFNN-Hybrid"] = {
        "params": count_trainable_params(cfnn), "cfg": cfnn_cfg,
        "test_RMSE": float(math.sqrt(mean_squared_error(y_te_o, pred))),
        "test_R2": float(r2_score(y_te_o, pred)),
    }

    # MLP-matched
    cfnn_p = count_trainable_params(CFNNHybrid())
    mlp_cfg = find_mlp_config_for_budget(1, cfnn_p, 1, (2, 3, 4), 256)
    mlp, mlp_cfg_chosen = grid_train(
        lambda: BudgetMLP(1, 1, mlp_cfg["hidden_dim"], mlp_cfg["num_layers"]),
        X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    mlp.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(mlp(X_te_t).cpu().numpy().reshape(-1, 1))
    results["MLP-matched"] = {
        "params": count_trainable_params(mlp), "cfg": mlp_cfg_chosen,
        "test_RMSE": float(math.sqrt(mean_squared_error(y_te_o, pred))),
        "test_R2": float(r2_score(y_te_o, pred)),
    }

    # MLP-1000
    mlp1k_cfg = find_mlp_config_for_budget(1, 1000, 1, (2, 3), 512)
    mlp1k, _ = grid_train(
        lambda: BudgetMLP(1, 1, mlp1k_cfg["hidden_dim"], mlp1k_cfg["num_layers"]),
        X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    mlp1k.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(mlp1k(X_te_t).cpu().numpy().reshape(-1, 1))
    results["MLP-1000"] = {
        "params": count_trainable_params(mlp1k),
        "test_RMSE": float(math.sqrt(mean_squared_error(y_te_o, pred))),
        "test_R2": float(r2_score(y_te_o, pred)),
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lrs", type=float, nargs="+", default=[5e-3, 1e-3])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0])
    args = parser.parse_args()

    if args.smoke:
        systems = SYSTEMS
        n_points_list = [50]
        noise_db_list = [3.0]
        seeds = [42]
        args.epochs = min(args.epochs, 80)
        args.patience = min(args.patience, 15)
    else:
        systems = SYSTEMS
        n_points_list = [30, 60, 120]
        noise_db_list = [0.0, 1.0, 3.0]
        seeds = SEEDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {}
    total = len(systems) * len(n_points_list) * len(noise_db_list) * len(seeds)
    done = 0

    for sys_ in systems:
        for np_ in n_points_list:
            for ndb in noise_db_list:
                for seed in seeds:
                    done += 1
                    print(f"[{done}/{total}] {sys_['name']} n={np_} noise={ndb}dB seed={seed}", flush=True)
                    key = f"{sys_['name']}@{np_}@{ndb}@{seed}"
                    raw[key] = run_one(sys_, np_, ndb, seed, args)
                    (OUT_DIR / "raw.json").write_text(json.dumps(raw, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    models_order = ["CFNN-Hybrid", "MLP-matched", "MLP-1000"]
    summary = {}
    for sys_ in systems:
        for np_ in n_points_list:
            for ndb in noise_db_list:
                for model in models_order:
                    r2s = []
                    for seed in seeds:
                        k = f"{sys_['name']}@{np_}@{ndb}@{seed}"
                        if k in raw and model in raw[k]:
                            r2s.append(raw[k][model]["test_R2"])
                    if r2s:
                        arr = np.array(r2s)
                        sk = f"{sys_['name']}@{np_}@{ndb}@{model}"
                        summary[sk] = {"system": sys_["name"], "n_points": np_, "noise_db": ndb,
                                       "model": model, "R2_mean": float(np.mean(arr)),
                                       "R2_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                                       "n_seeds": len(r2s)}

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    print("\n" + "=" * 120)
    print("System Identification: R² by system, #points")
    for sys_ in systems:
        print(f"\n--- {sys_['name']}: {sys_['desc']} ---")
        print(f"{'n':>6s} {'dB':>4s}", end="")
        for m in models_order: print(f"  {m:>20s}", end="")
        print()
        for np_ in n_points_list:
            for ndb in noise_db_list:
                print(f"{np_:>6d} {ndb:>4.0f}", end="")
                for m in models_order:
                    k = f"{sys_['name']}@{np_}@{ndb}@{m}"
                    if k in summary: print(f"  {summary[k]['R2_mean']:>8.4f}±{summary[k]['R2_std']:.4f}", end="")
                    else: print(f"  {'—':>15s}", end="")
                print()
    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
