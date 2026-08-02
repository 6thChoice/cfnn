#!/usr/bin/env python3
"""Experiment 1: Pole Sharpness Sweep.

Function: y = 1/(z² + ε), z = 0.85·x1 - 0.55·x2 + 0.35·x3 - 0.10
Vary ε across a wide range to measure how each model's fit degrades as the pole sharpens.

Models compared (at matched parameter budgets):
- CFNN-Hybrid (124 params, fixed architecture)
- MLP-124 (parameter-matched to CFNN)
- MLP-500 / MLP-1500 (larger MLP budgets for reference)
- KAN (if repo available)
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
OUT_DIR = ROOT / "experiment_refine" / "pole_sharpness_sweep_results"
SEEDS = [42, 123, 456, 789, 1024]
EPSILON_VALUES = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]

# ─── Data ──────────────────────────────────────────────────────────────

def lorentzian_pole(X: np.ndarray, eps: float) -> np.ndarray:
    z = 0.85 * X[:, 0] - 0.55 * X[:, 1] + 0.35 * X[:, 2] - 0.10
    return (1.0 / (z * z + eps)).reshape(-1, 1).astype(np.float32)

# ─── Models ─────────────────────────────────────────────────────────────

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
    def __init__(self, input_dim: int = 3, output_dim: int = 1):
        super().__init__()
        self.linear_skip = nn.Linear(input_dim, output_dim)
        self.units = nn.ModuleList([RationalUnit(input_dim, output_dim, 5) for _ in range(6)])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

def build_kan(hidden=12, layers=2, grid=8, order=3):
    if RepoKAN is not None:
        return RepoKAN(input_dim=3, output_dim=1, hidden_dim=hidden, num_layers=layers, grid_size=grid, spline_order=order)
    return None

# ─── Training ──────────────────────────────────────────────────────────

def train_model(model, X_tr, y_tr, X_val, y_val, epochs, lr, wd=1e-5, patience=80):
    # Ensure targets and predictions are 2D [N, 1] not [N]
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
    best_val, best_m, best_cfg = float("inf"), None, {}
    # Ensure targets are 2D
    if y_tr.dim() == 1: y_tr = y_tr.unsqueeze(-1)
    if y_val.dim() == 1: y_val = y_val.unsqueeze(-1)
    crit_eval = nn.MSELoss()
    for act in args.activations:
        for lr in args.lrs:
            for wd in args.weight_decays:
                m = builder(act).to(DEVICE)
                m = train_model(m, X_tr, y_tr, X_val, y_val, args.epochs, lr, wd, args.patience)
                m.eval()
                with torch.no_grad():
                    mpred = m(X_val)
                    if mpred.dim() == 1: mpred = mpred.unsqueeze(-1)
                    v = float(crit_eval(mpred, y_val).item())
                if v < best_val:
                    best_val, best_m, best_cfg = v, m, {"act": act, "lr": lr, "wd": wd}
    return best_m, best_cfg

# ─── Per-(eps, seed) run ───────────────────────────────────────────────

CFNN_PARAMS = count_trainable_params(CFNNHybrid())
MLP_BUDGETS = {
    "MLP-124": find_mlp_config_for_budget(3, 124, 1, (1, 2, 3), 256),
    "MLP-500": find_mlp_config_for_budget(3, 500, 1, (2, 3), 512),
    "MLP-1500": find_mlp_config_for_budget(3, 1500, 1, (2, 3), 512),
}

KAN_MODEL = build_kan()
KAN_PARAMS = count_trainable_params(KAN_MODEL) if KAN_MODEL is not None else None

def run_one(eps: float, seed: int, args) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n = args.n_samples
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    y = lorentzian_pole(X, eps)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.35, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.85714, random_state=seed)

    sx, sy = StandardScaler(), StandardScaler()
    X_tr_t = torch.tensor(sx.fit_transform(X_tr), dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(sx.transform(X_val), dtype=torch.float32, device=DEVICE)
    X_te_t = torch.tensor(sx.transform(X_te), dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(sy.fit_transform(y_tr), dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(sy.transform(y_val), dtype=torch.float32, device=DEVICE)

    results = {}

    # CFNN-Hybrid
    cfnn, cfnn_cfg = grid_train(lambda _: CFNNHybrid(), X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    results["CFNN-Hybrid"] = {"params": count_trainable_params(cfnn), "cfg": cfnn_cfg, "model": cfnn}

    # MLP matched
    for label, cfg in MLP_BUDGETS.items():
        mlp, mlp_cfg = grid_train(
            lambda _, c=cfg: BudgetMLP(3, 1, c["hidden_dim"], c["num_layers"]),
            X_tr_t, y_tr_t, X_val_t, y_val_t, args)
        results[label] = {"params": count_trainable_params(mlp), "cfg": mlp_cfg, "model": mlp}

    # KAN
    if KAN_MODEL is not None:
        kan, kan_cfg = grid_train(
            lambda _: build_kan(), X_tr_t, y_tr_t, X_val_t, y_val_t, args)
        results["KAN"] = {"params": count_trainable_params(kan), "cfg": kan_cfg, "model": kan}

    # Evaluate
    out = {}
    for name, r in results.items():
        r["model"].eval()
        with torch.no_grad():
            pred = r["model"](X_te_t).cpu().numpy()
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            pred = sy.inverse_transform(pred)
        out[name] = {
            "trainable_params": r["params"],
            "best_cfg": r["cfg"],
            "MSE": float(mean_squared_error(y_te, pred)),
            "RMSE": float(math.sqrt(mean_squared_error(y_te, pred))),
            "R2": float(r2_score(y_te, pred)),
        }
    return out

# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--activations", nargs="+", default=["Tanh", "ReLU"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 3e-3])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0])
    args = parser.parse_args()

    epsilons = [0.01, 1.0] if args.smoke else EPSILON_VALUES
    seeds = [42] if args.smoke else SEEDS
    if args.smoke:
        args.epochs = min(args.epochs, 100)
        args.patience = min(args.patience, 15)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {}
    total = len(epsilons) * len(seeds)
    done = 0

    for eps in epsilons:
        for seed in seeds:
            done += 1
            print(f"[{done}/{total}] ε={eps:.4f} seed={seed}", flush=True)
            raw[f"{eps}@{seed}"] = run_one(eps, seed, args)
            (OUT_DIR / "raw.json").write_text(json.dumps(raw, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    # Build summary table
    models_ordered = ["CFNN-Hybrid", "MLP-124", "MLP-500", "MLP-1500"]
    if KAN_MODEL is not None:
        models_ordered.append("KAN")

    summary = {}
    for mod in models_ordered:
        for eps in epsilons:
            r2s = []
            for seed in seeds:
                key = f"{eps}@{seed}"
                if key in raw and mod in raw[key]:
                    r2s.append(raw[key][mod]["R2"])
            if r2s:
                arr = np.array(r2s)
                summary[f"{mod}@{eps}"] = {
                    "model": mod, "eps": eps,
                    "R2_mean": float(np.mean(arr)),
                    "R2_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "n_seeds": len(r2s),
                }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    # Pretty print
    print("\n" + "=" * 100)
    print("Pole Sharpness Sweep — R² at fixed parameter budget")
    print("=" * 100)
    header = f"{'ε':>8s}"
    for m in models_ordered:
        header += f"  {m:>18s}"
    print(header)
    print("-" * len(header))
    for eps in epsilons:
        row = f"{eps:8.4f}"
        for m in models_ordered:
            k = f"{m}@{eps}"
            if k in summary:
                row += f"  {summary[k]['R2_mean']:>8.4f}  "
            else:
                row += f"  {'—':>8s}  "
        print(row)
    print()

    # Write SUMMARY.md
    lines = ["# Pole Sharpness Sweep", "",
             f"Function: y = 1/(z² + ε), ε ∈ {{{', '.join(str(e) for e in epsilons)}}}", "",
             "| ε | " + " | ".join(models_ordered) + " |",
             "|---" + "|---" * len(models_ordered) + "|"]
    for eps in epsilons:
        row = f"| {eps:.4f}"
        for m in models_ordered:
            k = f"{m}@{eps}"
            if k in summary:
                row += f" | {summary[k]['R2_mean']:.4f} ± {summary[k]['R2_std']:.4f}"
            else:
                row += " | —"
        lines.append(row + " |")
    lines += ["", f"Device: {DEVICE} | seeds: {seeds} | n_samples={args.n_samples} | epochs={args.epochs}"]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Done. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
