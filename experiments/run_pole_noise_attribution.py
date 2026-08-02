#!/usr/bin/env python3
"""Experiment 4: Pole + Noise Attribution Bridge.

Combines the pole-structured target function from Exp 1 with the
high-dimensional noisy-feature design from the Noise experiment.

This directly tests whether CFNN's fitting advantage on rational functions
naturally leads to better attribution (NSR, Top-5, MIR), bridging the
gap between the two previously separate claims.

Design:
- Target: Lorentzian pole y = 1/(z² + ε) with sharpness sweep
- Features: 4 true (enter z), 4 noise, 2 redundant, 1 deceptive (11D total)
- Compare CFNN vs MLP at matched budget on both R² and attribution metrics
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
OUT_DIR = ROOT / "experiment_refine" / "pole_noise_attribution_results"
SEEDS = [42, 123, 456, 789, 1024]

EPSILONS = [0.001, 0.01, 0.05, 0.1, 0.5]  # sharpness sweep
NOISE_STDS = [0.3]  # observation noise on y

# Feature structure (same as Noise experiment)
N_TRUE = 4
N_NOISE = 4
N_REDUNDANT = 2
N_DECEPTIVE = 1
TRUE_COEFFS = np.array([1.5, 0.8, 0.5, 1.2])


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def lorentzian_pole_from_true(X_true: np.ndarray, eps: float) -> np.ndarray:
    """y = 1/(z² + ε) where z = w·X_true is a linear projection of true features."""
    z = X_true @ TRUE_COEFFS
    return (1.0 / (z * z + eps)).reshape(-1, 1).astype(np.float32)


def generate_data(eps: float, noise_std: float, n_samples: int, seed: int):
    """Generate data with pole target + high-d noise features (same structure as Noise exp)."""
    rng = np.random.default_rng(seed)
    n = n_samples

    # Generate true features
    X_true = rng.uniform(0.5, 2.0, (n, N_TRUE)).astype(np.float32)
    y = lorentzian_pole_from_true(X_true, eps)

    # Add observation noise
    y += rng.normal(0, noise_std * np.std(y), y.shape).astype(np.float32)

    # Generate noise features
    X_noise = rng.uniform(0.0, 1.0, (n, N_NOISE)).astype(np.float32)

    # Generate redundant features (noisy copies of true features)
    X_redundant = np.column_stack([
        X_true[:, i % N_TRUE] + rng.normal(0, 0.05, n) for i in range(N_REDUNDANT)
    ]).astype(np.float32)

    # Generate deceptive features (correlated with y in train, uncorrelated in test)
    X_deceptive = np.zeros((n, N_DECEPTIVE), dtype=np.float32)
    X = np.hstack([X_true, X_noise, X_redundant, X_deceptive])

    # Split
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.05 / 0.70, random_state=seed)

    # Deceptive feature: correlated with y on train, not on test
    dec_start = N_TRUE + N_NOISE + N_REDUNDANT
    for c in range(dec_start, dec_start + N_DECEPTIVE):
        # Train: correlate
        yr = y_train[:, 0]
        yr_std = (yr - yr.mean()) / (yr.std() + 1e-8)
        X_train[:, c] = (0.8 * yr_std + np.sqrt(1 - 0.64) * rng.normal(0, 1, len(yr))).astype(np.float32)
        # Test: random
        X_test[:, c] = rng.uniform(0, 1, len(y_test)).astype(np.float32)

    feature_types = (["true"] * N_TRUE + ["noise"] * N_NOISE +
                     ["redundant"] * N_REDUNDANT + ["deceptive"] * N_DECEPTIVE)
    true_idx = list(range(N_TRUE))
    noise_idx = list(range(N_TRUE, N_TRUE + N_NOISE))
    redundant_idx = list(range(N_TRUE + N_NOISE, N_TRUE + N_NOISE + N_REDUNDANT))
    deceptive_idx = list(range(dec_start, dec_start + N_DECEPTIVE))

    # Standardize
    sx, sy = StandardScaler(), StandardScaler()
    return {
        "X_train": sx.fit_transform(X_train).astype(np.float32),
        "y_train": sy.fit_transform(y_train).astype(np.float32),
        "X_val": sx.transform(X_val).astype(np.float32),
        "y_val": sy.transform(y_val).astype(np.float32),
        "X_test": sx.transform(X_test).astype(np.float32),
        "y_test": y_test.astype(np.float32),  # keep original scale for eval
        "y_scaler": sy,
        "feature_types": feature_types,
        "n_features": X.shape[1],
        "true_idx": true_idx,
        "noise_idx": noise_idx,
        "redundant_idx": redundant_idx,
        "deceptive_idx": deceptive_idx,
    }


# ─── Models ────────────────────────────────────────────────────────────

# Reuse CFNNHybrid and BudgetMLP definitions from run_pole_sharpness_sweep
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
    def __init__(self, input_dim=11, output_dim=1):
        super().__init__()
        self.linear_skip = nn.Linear(input_dim, output_dim)
        self.units = nn.ModuleList([RationalUnit(input_dim, output_dim, 4) for _ in range(8)])
    def forward(self, x):
        out = self.linear_skip(x)
        for u in self.units:
            out = out + u(x)
        return out


# ─── Attribution Metrics ───────────────────────────────────────────────

def shap_values(model, X, device):
    """Approximate SHAP values using gradient-based attribution (simple version)."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
    with torch.enable_grad():
        pred = model(X_t)
        grad = torch.autograd.grad(pred.sum(), X_t)[0]
    return grad.detach().cpu().numpy()


def attribution_metrics(shap, feature_types, true_idx, noise_idx):
    """Compute NSR, Top-5, MIR metrics (same as original Noise experiment)."""
    shap_abs = np.abs(shap).mean(axis=0)
    n_true, n_noise = len(true_idx), len(noise_idx)
    total_attribution = shap_abs.sum() + 1e-10

    # NSR: noise-to-signal ratio (mean |SHAP| on noise / mean |SHAP| on true)
    signal = shap_abs[true_idx].mean() if n_true > 0 else 1.0
    noise = shap_abs[noise_idx].mean() if n_noise > 0 else 0.0
    nsr = noise / (signal + 1e-10)

    # Top-5: fraction of total attribution captured by top 5 features
    sorted_shap = np.sort(shap_abs)[::-1]
    top5 = sorted_shap[:min(5, len(sorted_shap))].sum() / total_attribution

    # MIR: max-to-interquartile ratio
    if len(shap_abs) >= 4:
        q75, q25 = np.percentile(shap_abs, [75, 25])
        iqr = max(q75 - q25, 1e-10)
        mir = shap_abs.max() / iqr
    else:
        mir = float("nan")

    return {"NSR": float(nsr), "Top5": float(top5), "MIR": float(mir)}


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


# ─── Run one ───────────────────────────────────────────────────────────

def run_one(eps: float, noise_std: float, seed: int, args) -> dict:
    set_seed(seed)
    data = generate_data(eps, noise_std, args.n_samples, seed)
    nf = data["n_features"]

    X_tr_t = torch.tensor(data["X_train"], dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(data["X_val"], dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(data["X_test"], dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(data["y_train"], dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(data["y_val"], dtype=torch.float32, device=DEVICE)
    y_test = data["y_test"]
    sy = data["y_scaler"]

    results = {}

    for model_name, builder, mlp_budget in [
        ("CFNN-Hybrid", lambda: CFNNHybrid(input_dim=nf), None),
        ("MLP-matched", lambda: BudgetMLP(nf, 1,
            find_mlp_config_for_budget(nf, count_trainable_params(CFNNHybrid(input_dim=nf)), 1, (2, 3), 256)["hidden_dim"],
            find_mlp_config_for_budget(nf, count_trainable_params(CFNNHybrid(input_dim=nf)), 1, (2, 3), 256)["num_layers"]),
         count_trainable_params(CFNNHybrid(input_dim=nf))),
        ("MLP-1000", lambda: BudgetMLP(nf, 1,
            find_mlp_config_for_budget(nf, 1000, 1, (2, 3), 512)["hidden_dim"],
            find_mlp_config_for_budget(nf, 1000, 1, (2, 3), 512)["num_layers"]),
         1000),
    ]:
        model, cfg = grid_train(builder, X_tr_t, y_tr_t, X_val_t, y_val_t, args)
        model.eval()

        # Prediction metrics
        with torch.no_grad():
            pred_scaled = model(X_test_t).cpu().numpy().reshape(-1, 1)
        pred = sy.inverse_transform(pred_scaled)

        # Attribution metrics
        shap = shap_values(model, data["X_test"][:500], DEVICE)  # subsample for speed
        attr = attribution_metrics(shap, data["feature_types"],
                                    data["true_idx"], data["noise_idx"])

        results[model_name] = {
            "params": count_trainable_params(model),
            "cfg": cfg,
            "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
            "test_R2": float(r2_score(y_test, pred)),
            **attr,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lrs", type=float, nargs="+", default=[5e-3, 1e-3])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0])
    args = parser.parse_args()

    epsilons = [0.01, 0.5] if args.smoke else EPSILONS
    seeds = [42] if args.smoke else SEEDS
    if args.smoke:
        args.epochs = min(args.epochs, 80)
        args.patience = min(args.patience, 15)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {}
    total = len(epsilons) * len(seeds)
    done = 0

    for eps in epsilons:
        for seed in seeds:
            done += 1
            print(f"[{done}/{total}] ε={eps:.4f} seed={seed}", flush=True)
            raw[f"{eps}@{seed}"] = run_one(eps, 0.3, seed, args)
            (OUT_DIR / "raw.json").write_text(json.dumps(raw, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    # Summarize
    models_order = ["CFNN-Hybrid", "MLP-matched", "MLP-1000"]
    summary = {}
    for eps in epsilons:
        for model in models_order:
            r2s, nsrs, top5s = [], [], []
            for seed in seeds:
                k = f"{eps}@{seed}"
                if k in raw and model in raw[k]:
                    r2s.append(raw[k][model]["test_R2"])
                    nsrs.append(raw[k][model]["NSR"])
                    top5s.append(raw[k][model]["Top5"])
            if r2s:
                arr_r2 = np.array(r2s)
                arr_nsr = np.array(nsrs)
                arr_top5 = np.array(top5s)
                sk = f"{eps}@{model}"
                summary[sk] = {
                    "eps": eps, "model": model,
                    "R2_mean": float(np.mean(arr_r2)),
                    "R2_std": float(np.std(arr_r2, ddof=1)) if len(arr_r2) > 1 else 0.0,
                    "NSR_mean": float(np.mean(arr_nsr)),
                    "NSR_std": float(np.std(arr_nsr, ddof=1)) if len(arr_nsr) > 1 else 0.0,
                    "Top5_mean": float(np.mean(arr_top5)),
                    "n_seeds": len(r2s),
                }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    # Print table
    print("\n" + "=" * 100)
    print("Pole + Noise Attribution Bridge")
    print("=" * 100)
    print(f"{'ε':>8s}", end="")
    for m in models_order:
        print(f"  {m:>20s}R²    NSR   ", end="")
    print()
    for eps in epsilons:
        print(f"{eps:8.4f}", end="")
        for m in models_order:
            k = f"{eps}@{m}"
            if k in summary:
                print(f"  {summary[k]['R2_mean']:>8.4f}  {summary[k]['NSR_mean']:>7.3f}", end="")
            else:
                print(f"  {'—':>8s}  {'—':>7s}", end="")
        print()

    (OUT_DIR / "SUMMARY.md").write_text(
        "# Pole + Noise Attribution Bridge\n\n" +
        "| ε | " + " | ".join(f"{m} R² | {m} NSR" for m in models_order) + " |\n" +
        "|---" + "|" + "|".join("---:|:---" for _ in models_order) + "|\n" +
        "\n".join(f"| {eps:.4f} | " + " | ".join(
            f"{summary.get(f'{eps}@{m}', {}).get('R2_mean', 0):.4f} | {summary.get(f'{eps}@{m}', {}).get('NSR_mean', 0):.3f}"
            for m in models_order) + " |" for eps in epsilons) +
        f"\n\nDevice: {DEVICE} | seeds: {SEEDS}\n",
        encoding="utf-8")

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
