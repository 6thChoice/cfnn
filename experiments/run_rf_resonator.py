#!/usr/bin/env python3
"""Experiment D: RF/Microwave Resonator Fitting.

Simulates S₂₁ parameters of coupled microwave resonators. Each resonance
is a Lorentzian dip/peak in the transmission response.

Design:
1. Generate S₂₁ for 1-4 coupled resonators with varying coupling/Q
2. Add measurement noise (VNA thermal noise)
3. Compare CFNN vs MLP at matched budget
4. Key metric: R², resonance frequency recovery
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))
from param_matched_utils import NumpySafeEncoder, count_trainable_params, find_mlp_config_for_budget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "experiment_refine" / "rf_resonator_results"
SEEDS = [42, 123, 456, 789, 1024]

# Resonator configurations
RF_CONFIGS = [
    {"name": "rf_single_highQ",  "n_res": 1,  "Q": "high",  "desc": "1 resonator, high Q (>5000)"},
    {"name": "rf_single_lowQ",   "n_res": 1,  "Q": "low",   "desc": "1 resonator, low Q (~200)"},
    {"name": "rf_double_coupled","n_res": 2,  "Q": "high",  "desc": "2 coupled resonators, high Q"},
    {"name": "rf_quad",          "n_res": 4,  "Q": "medium","desc": "4 resonators, medium Q (filter-like)"},
]

NOISE_DB = [0.0, 0.5, 2.0]  # VNA noise floor (dB)
N_POINTS = [60, 120, 240, 480]
FULL_N_POINTS = 2000


def lorentzian(x, A, mu, gamma):
    return A * gamma * gamma / ((x - mu) ** 2 + gamma * gamma)


def generate_s21(n_resonators: int, Q_type: str, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray, list]:
    """Generate S₂₁ magnitude (dB) for a multi-resonator system.

    Returns:
        freqs: frequency axis (GHz)
        s21_db: S₂₁ magnitude in dB
        resonators: list of (f0, Q, coupling) tuples
    """
    Q_map = {"high": (3000, 15000), "medium": (500, 3000), "low": (100, 500)}

    Q_range = Q_map.get(Q_type, (500, 5000))
    f_range = (1.0, 10.0)  # GHz

    resonators = []
    for i in range(n_resonators):
        f0 = rng.uniform(f_range[0], f_range[1])
        Q = rng.uniform(*Q_range)
        coupling = rng.uniform(-20, -3)  # dB depth
        resonators.append((f0, Q, coupling))

    resonators.sort(key=lambda r: r[0])

    freqs = np.linspace(f_range[0], f_range[1], FULL_N_POINTS, dtype=np.float32).reshape(-1, 1)

    # S₂₁ as product of Lorentzian dips (transmission through cascaded resonators)
    s21_linear = np.ones(len(freqs))
    for f0, Q, coupling_dB in resonators:
        gamma = f0 / (2 * Q)
        # Coupling depth as linear multiplier
        depth_linear = 10 ** (coupling_dB / 20)
        # Lorentzian dip: 1 - (1 - depth) * lorentzian(f, 1, f0, gamma)
        dip = 1 - (1 - depth_linear) * lorentzian(freqs.flatten(), 1, f0, gamma).flatten()
        s21_linear *= dip

    s21_db = (20 * np.log10(s21_linear + 1e-30)).astype(np.float32).reshape(-1, 1)
    return freqs, s21_db, resonators


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
    def __init__(self, input_dim=1, output_dim=1, degree=4, n_units=8):
        super().__init__()
        self.linear_skip = nn.Linear(input_dim, output_dim)
        self.units = nn.ModuleList([RationalUnit(input_dim, output_dim, degree) for _ in range(n_units)])
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


def run_one(cfg: dict, n_points: int, noise_db: float, seed: int, args) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    freqs, s21_db, resonators = generate_s21(cfg["n_res"], cfg["Q"], rng)
    X = freqs

    # Add noise
    y = s21_db + rng.normal(0, noise_db, s21_db.shape).astype(np.float32)

    idx = np.sort(rng.choice(len(X), n_points, replace=False))
    X_sub, y_sub = X[idx], y[idx]

    split = max(1, int(n_points * 0.8))
    X_tr_a, X_val_a = X_sub[:split], X_sub[split:]
    y_tr_a, y_val_a = y_sub[:split], y_sub[split:]

    X_te, y_te = X, s21_db

    sx, sy = StandardScaler(), StandardScaler()
    X_tr_t = torch.tensor(sx.fit_transform(X_tr_a), dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(sx.transform(X_val_a), dtype=torch.float32, device=DEVICE)
    X_te_t = torch.tensor(sx.transform(X_te), dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(sy.fit_transform(y_tr_a), dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(sy.transform(y_val_a), dtype=torch.float32, device=DEVICE)

    results = {}

    cfnn, cfnn_cfg = grid_train(lambda: CFNNHybrid(), X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    cfnn.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(cfnn(X_te_t).cpu().numpy().reshape(-1, 1))
    results["CFNN-Hybrid"] = {
        "params": count_trainable_params(cfnn), "cfg": cfnn_cfg,
        "test_RMSE": float(math.sqrt(mean_squared_error(y_te, pred))),
        "test_R2": float(r2_score(y_te, pred)),
    }

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
        "test_RMSE": float(math.sqrt(mean_squared_error(y_te, pred))),
        "test_R2": float(r2_score(y_te, pred)),
    }

    mlp1k_cfg = find_mlp_config_for_budget(1, 1000, 1, (2, 3), 512)
    mlp1k, _ = grid_train(
        lambda: BudgetMLP(1, 1, mlp1k_cfg["hidden_dim"], mlp1k_cfg["num_layers"]),
        X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    mlp1k.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(mlp1k(X_te_t).cpu().numpy().reshape(-1, 1))
    results["MLP-1000"] = {
        "params": count_trainable_params(mlp1k),
        "test_RMSE": float(math.sqrt(mean_squared_error(y_te, pred))),
        "test_R2": float(r2_score(y_te, pred)),
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
        configs = RF_CONFIGS[:1]
        n_points_list = [120]
        noise_list = [0.5]
        seeds = [42]
        args.epochs = min(args.epochs, 80)
        args.patience = min(args.patience, 15)
    else:
        configs = RF_CONFIGS
        n_points_list = N_POINTS
        noise_list = NOISE_DB
        seeds = SEEDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {}
    total = len(configs) * len(n_points_list) * len(noise_list) * len(seeds)
    done = 0

    for cfg in configs:
        for np_ in n_points_list:
            for ndb in noise_list:
                for seed in seeds:
                    done += 1
                    print(f"[{done}/{total}] {cfg['name']} n={np_} noise={ndb}dB seed={seed}", flush=True)
                    key = f"{cfg['name']}@{np_}@{ndb}@{seed}"
                    raw[key] = run_one(cfg, np_, ndb, seed, args)
                    (OUT_DIR / "raw.json").write_text(
                        json.dumps(raw, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    models_order = ["CFNN-Hybrid", "MLP-matched", "MLP-1000"]
    summary = {}
    for cfg in configs:
        for np_ in n_points_list:
            for ndb in noise_list:
                for model in models_order:
                    r2s = []
                    for seed in seeds:
                        k = f"{cfg['name']}@{np_}@{ndb}@{seed}"
                        if k in raw and model in raw[k]:
                            r2s.append(raw[k][model]["test_R2"])
                    if r2s:
                        arr = np.array(r2s)
                        sk = f"{cfg['name']}@{np_}@{ndb}@{model}"
                        summary[sk] = {
                            "config": cfg["name"], "n_points": np_, "noise_db": ndb,
                            "model": model,
                            "R2_mean": float(np.mean(arr)),
                            "R2_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                            "n_seeds": len(r2s),
                        }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    print("\n" + "=" * 120)
    print("RF Resonator Fitting")
    for cfg in configs:
        print(f"\n--- {cfg['name']}: {cfg['desc']} ---")
        print(f"{'n':>6s} {'dB':>4s}", end="")
        for m in models_order: print(f"  {m:>20s}", end="")
        print()
        for np_ in n_points_list:
            for ndb in noise_list:
                print(f"{np_:>6d} {ndb:>4.0f}", end="")
                for m in models_order:
                    k = f"{cfg['name']}@{np_}@{ndb}@{m}"
                    if k in summary:
                        print(f"  {summary[k]['R2_mean']:>8.4f}±{summary[k]['R2_std']:.4f}", end="")
                    else:
                        print(f"  {'—':>15s}", end="")
                print()
    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
