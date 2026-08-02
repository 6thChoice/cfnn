#!/usr/bin/env python3
"""Experiment 2: Michaelis-Menten enzyme kinetics fitting.

Real-world rational function: v = Vmax · S / (Km + S)
CFNN's rational unit naturally expresses this form as a degree-1 rational.

Design:
1. Generate synthetic MM curves with realistic noise (biological/measurement noise)
2. Vary key parameters: (Vmax, Km) combinations across a realistic range
3. Test few-shot learning: fit with limited data points (6, 8, 10, 15, 20 points)
4. Compare CFNN-Hybrid vs parameter-matched MLP vs KAN

Key questions:
- Can small CFNN extrapolate beyond the measured concentration range?
- Does the rational inductive bias give parameter efficiency for MM curves?
- How does performance degrade with fewer data points?
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
OUT_DIR = ROOT / "experiment_refine" / "michaelis_menten_results"
SEEDS = [42, 123, 456, 789, 1024]

# Realistic Michaelis-Menten parameter ranges (from biochemistry literature)
# Vmax in arbitrary units, Km in mM (millimolar)
MM_CONFIGS = [
    {"name": "fast_high",  "Vmax": 100, "Km": 0.5,   "desc": "Fast enzyme, high affinity"},
    {"name": "fast_low",   "Vmax": 100, "Km": 5.0,   "desc": "Fast enzyme, low affinity"},
    {"name": "slow_high",  "Vmax": 20,  "Km": 0.5,   "desc": "Slow enzyme, high affinity"},
    {"name": "slow_low",   "Vmax": 20,  "Km": 5.0,   "desc": "Slow enzyme, low affinity"},
    {"name": "typical",    "Vmax": 50,  "Km": 2.0,   "desc": "Typical textbook values"},
]

# Number of data points (concentration levels) for few-shot evaluation
N_POINTS = [6, 10, 16, 24]
FULL_N_POINTS = 100  # dense sampling for full curve comparison

# Concentration range (mM)
S_MIN, S_MAX = 0.05, 20.0

# Noise levels (biological + measurement)
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def michaelis_menten(S: np.ndarray, Vmax: float, Km: float) -> np.ndarray:
    """v = Vmax * S / (Km + S). S in mM."""
    return (Vmax * S / (Km + S)).reshape(-1, 1).astype(np.float32)


def generate_dataset(Vmax: float, Km: float, n_points: int, noise_std: float, seed: int):
    """Generate MM curve with n_points concentration levels and Gaussian noise."""
    rng = np.random.default_rng(seed)
    # Log-uniform sampling of concentrations (typical experimental design)
    S = np.exp(rng.uniform(np.log(S_MIN), np.log(S_MAX), size=n_points)).astype(np.float32).reshape(-1, 1)
    S_sorted = np.sort(S, axis=0)
    v_clean = michaelis_menten(S_sorted, Vmax, Km)
    v_noisy = v_clean + rng.normal(0, noise_std * np.mean(v_clean), size=v_clean.shape).astype(np.float32)
    return S_sorted, v_noisy, v_clean


# ─── Models ────────────────────────────────────────────────────────────

class BudgetMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class RationalUnit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, degree: int):
        super().__init__()
        self.P_proj = nn.Linear(input_dim, output_dim)
        self.P_coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * 0.05)
        self.Q_proj = nn.Linear(input_dim, output_dim)
        self.Q_coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * 0.05)
        self.degree = degree
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _poly(proj, coeffs):
            z = torch.tanh(proj(x))
            powers = [torch.ones_like(z)]
            for d in range(1, self.degree + 1):
                powers.append(powers[-1] * z)
            return torch.sum(torch.stack(powers, dim=-1) * coeffs, dim=2)
        return _poly(self.P_proj, self.P_coeffs) / (_poly(self.Q_proj, self.Q_coeffs) ** 2 + 0.1)


class CFNNHybrid(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        self.linear_skip = nn.Linear(input_dim, output_dim)
        self.units = nn.ModuleList([RationalUnit(input_dim, output_dim, 2) for _ in range(4)])
    def forward(self, x):
        out = self.linear_skip(x)
        for u in self.units:
            out = out + u(x)
        return out


# ─── KAN (optional) ────────────────────────────────────────────────────

KAN_PATH = ROOT / "git_codebase" / "noise_robustness_runner" / "code" / "interpretable_experiment" / "models"
if KAN_PATH.exists():
    sys.path.insert(0, str(KAN_PATH))
try:
    from kan_model import KAN as RepoKAN
except Exception:
    RepoKAN = None

def build_kan(hidden=8, layers=2, grid=6, order=3):
    if RepoKAN is not None:
        return RepoKAN(input_dim=1, output_dim=1, hidden_dim=hidden, num_layers=layers, grid_size=grid, spline_order=order)
    return None

KAN_MODEL = build_kan()

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
        if not torch.isfinite(loss):
            break
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
                if stale >= patience:
                    break
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


# ─── Run one config ────────────────────────────────────────────────────

def run_one(mm_cfg: dict, n_points: int, noise_std: float, seed: int, args) -> dict:
    set_seed(seed)

    S_full, v_clean = generate_dataset(mm_cfg["Vmax"], mm_cfg["Km"], FULL_N_POINTS, 0.0, seed)[:2]
    S_train, v_train, _ = generate_dataset(mm_cfg["Vmax"], mm_cfg["Km"], n_points, noise_std, seed)

    # Split train into train/val (80/20)
    split_idx = max(1, int(len(S_train) * 0.8))
    S_tr, S_val = S_train[:split_idx], S_train[split_idx:]
    y_tr, y_val = v_train[:split_idx], v_train[split_idx:]

    # Test: dense evaluation over full range (extrapolation)
    S_test = S_full
    y_test = v_clean

    # Standardize
    sx, sy = StandardScaler(), StandardScaler()
    X_tr_t = torch.tensor(sx.fit_transform(S_tr), dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(sx.transform(S_val), dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(sx.transform(S_test), dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(sy.fit_transform(y_tr), dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(sy.transform(y_val), dtype=torch.float32, device=DEVICE)

    results = {}

    # CFNN-Hybrid
    cfnn, cfnn_cfg = grid_train(lambda: CFNNHybrid(), X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    cfnn.eval()
    with torch.no_grad():
        pred_tr_cfnn = sy.inverse_transform(cfnn(X_tr_t).cpu().numpy().reshape(-1, 1))
        pred_cfnn = sy.inverse_transform(cfnn(X_test_t).cpu().numpy().reshape(-1, 1))
    results["CFNN-Hybrid"] = {
        "params": count_trainable_params(cfnn),
        "cfg": cfnn_cfg,
        "train_RMSE": float(math.sqrt(mean_squared_error(y_tr, pred_tr_cfnn))),
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred_cfnn))),
        "test_R2": float(r2_score(y_test, pred_cfnn)),
    }

    # MLP matched (use CFNN params as budget)
    CFNN_PARAMS = count_trainable_params(CFNNHybrid())
    mlp_cfg = find_mlp_config_for_budget(1, CFNN_PARAMS, 1, (1, 2, 3), 256)
    mlp, mlp_cfg_chosen = grid_train(
        lambda: BudgetMLP(1, 1, mlp_cfg["hidden_dim"], mlp_cfg["num_layers"]),
        X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    mlp.eval()
    with torch.no_grad():
        pred_tr_mlp = sy.inverse_transform(mlp(X_tr_t).cpu().numpy().reshape(-1, 1))
        pred_mlp = sy.inverse_transform(mlp(X_test_t).cpu().numpy().reshape(-1, 1))
    results["MLP-matched"] = {
        "params": count_trainable_params(mlp),
        "cfg": mlp_cfg_chosen,
        "train_RMSE": float(math.sqrt(mean_squared_error(y_tr, pred_tr_mlp))),
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred_mlp))),
        "test_R2": float(r2_score(y_test, pred_mlp)),
    }

    # MLP-500 (larger reference)
    mlp500_cfg = find_mlp_config_for_budget(1, 500, 1, (2, 3), 512)
    mlp500, _ = grid_train(
        lambda: BudgetMLP(1, 1, mlp500_cfg["hidden_dim"], mlp500_cfg["num_layers"]),
        X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    mlp500.eval()
    with torch.no_grad():
        pred_mlp500 = sy.inverse_transform(mlp500(X_test_t).cpu().numpy().reshape(-1, 1))
    results["MLP-500"] = {
        "params": count_trainable_params(mlp500),
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred_mlp500))),
        "test_R2": float(r2_score(y_test, pred_mlp500)),
    }

    # KAN
    if KAN_MODEL is not None:
        kan, kan_cfg = grid_train(lambda: build_kan(), X_tr_t, y_tr_t, X_val_t, y_val_t, args)
        kan.eval()
        with torch.no_grad():
            pred_kan = sy.inverse_transform(kan(X_test_t).cpu().numpy().reshape(-1, 1))
        results["KAN"] = {
            "params": count_trainable_params(kan),
            "cfg": kan_cfg,
            "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred_kan))),
            "test_R2": float(r2_score(y_test, pred_kan)),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 5e-3, 1e-3])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-5])
    args = parser.parse_args()

    if args.smoke:
        mm_configs = MM_CONFIGS[:1]  # just "fast_high"
        n_points_list = [10]
        noise_list = [0.05]
        seeds = [42]
        args.epochs = min(args.epochs, 80)
        args.patience = min(args.patience, 15)
    else:
        mm_configs = MM_CONFIGS
        n_points_list = N_POINTS
        noise_list = NOISE_LEVELS
        seeds = SEEDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {}
    total = len(mm_configs) * len(n_points_list) * len(noise_list) * len(seeds)
    done = 0

    for mm in mm_configs:
        for np_ in n_points_list:
            for noise in noise_list:
                for seed in seeds:
                    done += 1
                    print(f"[{done}/{total}] {mm['name']} n={np_} noise={noise} seed={seed}", flush=True)
                    key = f"{mm['name']}@{np_}@{noise}@{seed}"
                    raw[key] = run_one(mm, np_, noise, seed, args)
                    (OUT_DIR / "raw.json").write_text(
                        json.dumps(raw, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    # Summarize: group by (mm_config, n_points, noise) across seeds
    summary = {}
    for mm in mm_configs:
        for np_ in n_points_list:
            for noise in noise_list:
                for model_type in ["CFNN-Hybrid", "MLP-matched", "MLP-500", "KAN"]:
                    r2s, rmses = [], []
                    for seed in seeds:
                        key = f"{mm['name']}@{np_}@{noise}@{seed}"
                        if key in raw and model_type in raw[key] and "test_R2" in raw[key][model_type]:
                            r2s.append(raw[key][model_type]["test_R2"])
                            rmses.append(raw[key][model_type]["test_RMSE"])
                    if r2s:
                        arr = np.array(r2s)
                        rmse_arr = np.array(rmses)
                        sk = f"{mm['name']}@{np_}@{noise}@{model_type}"
                        summary[sk] = {
                            "mm_name": mm["name"], "n_points": np_, "noise": noise,
                            "model": model_type,
                            "R2_mean": float(np.mean(arr)),
                            "R2_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                            "RMSE_mean": float(np.mean(rmse_arr)),
                            "n_seeds": len(r2s),
                        }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    # Print table
    models_show = ["CFNN-Hybrid", "MLP-matched", "MLP-500", "KAN"] if KAN_MODEL else ["CFNN-Hybrid", "MLP-matched", "MLP-500"]
    print("\n" + "=" * 120)
    print("Michaelis-Menten: R² by model, #points, noise level")
    for mm in mm_configs:
        print(f"\n--- {mm['name']}: Vmax={mm['Vmax']}, Km={mm['Km']} ---")
        print(f"{'n_points':>8s} {'noise':>6s}", end="")
        for m in models_show:
            print(f"  {m:>20s}", end="")
        print()
        for np_ in n_points_list:
            for noise in noise_list:
                print(f"{np_:>8d} {noise:>6.2f}", end="")
                for m in models_show:
                    k = f"{mm['name']}@{np_}@{noise}@{m}"
                    if k in summary:
                        print(f"  {summary[k]['R2_mean']:>8.4f}±{summary[k]['R2_std']:.4f}", end="")
                    else:
                        print(f"  {'—':>15s}", end="")
                print()

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
