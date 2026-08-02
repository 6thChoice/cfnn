#!/usr/bin/env python3
"""Experiment B: Mass Spectrometry Peak Detection.

Simulates mass spectra with isotopic peak clusters (Lorentzian-like).
Tests CFNN's ability to resolve overlapping isotopic peaks.

Design:
1. Generate isotopic peak clusters for peptides/small molecules
2. Each peak is Lorentzian with mass-dependent width
3. Vary: charge state, resolution, overlap
4. Compare CFNN vs MLP on both reconstruction and peak detection
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
OUT_DIR = ROOT / "experiment_refine" / "mass_spec_results"
SEEDS = [42, 123, 456, 789, 1024]

# Mass spec configurations
MS_CONFIGS = [
    {"name": "ms_simple",   "n_clusters": 3,  "resolution": "high",   "desc": "3 peak clusters, high resolution"},
    {"name": "ms_medium",   "n_clusters": 5,  "resolution": "medium", "desc": "5 clusters, medium resolution"},
    {"name": "ms_complex",  "n_clusters": 8,  "resolution": "medium", "desc": "8 clusters, medium resolution"},
    {"name": "ms_crowded",  "n_clusters": 5,  "resolution": "low",    "desc": "5 clusters, low resolution (heavy overlap)"},
]

NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10]
N_POINTS = [60, 120, 240, 480]
FULL_N_POINTS = 2000
MZ_MIN, MZ_MAX = 400.0, 1600.0  # m/z range


def lorentzian(x, A, mu, gamma):
    return A * gamma * gamma / ((x - mu) ** 2 + gamma * gamma)


def _isotopic_distribution(mass: float, charge: int, rng: np.random.Generator) -> list[tuple]:
    """Generate approximate isotopic peak distribution for a molecule.

    Returns list of (rel_abundance, mz_shift) for isotopic peaks.
    Simplified: monoistopic + 1-4 13C peaks with decreasing abundance.
    """
    peaks = []
    # Monoistopic peak (M)
    peaks.append((1.0, 0.0))

    # 13C peak approximation: ~1.1% per carbon, assume ~mass/12 carbons
    n_carbons = max(1, int(mass / 12))
    for i in range(1, min(5, n_carbons // 2 + 1)):
        abundance = (n_carbons * 0.011) ** i / math.factorial(i)  # Poisson approx
        if abundance < 0.01:
            break
        mz_shift = i * 1.00335 / charge
        peaks.append((abundance, mz_shift))

    return peaks


def generate_ms_spectrum(
    n_clusters: int, resolution: str, rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple], np.ndarray]:
    """Generate a realistic mass spectrum with isotopologue clusters.

    Returns:
        x: m/z axis
        all_peaks: list of (A, mu, gamma) for every individual peak
        y_clean: clean spectrum
    """
    # Resolution parameters: FWHM = m/z / resolution
    res_map = {"high": 100000, "medium": 30000, "low": 5000}

    resolution_power = res_map[resolution]
    all_peaks = []

    # Generate clusters at random m/z positions
    cluster_mz = []
    for _ in range(n_clusters * 2):
        mz = rng.uniform(MZ_MIN + 50, MZ_MAX - 50)
        ok = True
        for existing in cluster_mz:
            if abs(mz - existing) < 30:  # minimum 30 Da between clusters
                ok = False
                break
        if ok:
            cluster_mz.append(mz)
        if len(cluster_mz) >= n_clusters:
            break

    for mz in cluster_mz:
        charge = rng.choice([1, 2, 3])
        # Base molecule mass
        mass = mz * charge - charge

        # Get isotopic distribution
        iso_peaks = _isotopic_distribution(mass, charge, rng)

        total_abundance = sum(a for a, _ in iso_peaks)
        cluster_intensity = rng.uniform(50.0, 500.0)

        for rel_ab, mz_shift in iso_peaks:
            peak_mz = mz + mz_shift
            peak_A = cluster_intensity * rel_ab / total_abundance
            # FWHM = m/z / R, gamma = FWHM/2
            fwhm = peak_mz / resolution_power
            gamma = fwhm / 2.0
            if peak_A > 0.5:  # filter tiny peaks
                all_peaks.append((peak_A, peak_mz, max(gamma, 0.005)))

    # Sort by m/z
    all_peaks.sort(key=lambda p: p[1])

    # Generate spectrum
    x = np.linspace(MZ_MIN, MZ_MAX, FULL_N_POINTS, dtype=np.float32).reshape(-1, 1)
    y_clean = np.zeros_like(x)
    for A, mu, gamma in all_peaks:
        y_clean += lorentzian(x, A, mu, gamma)

    return x, all_peaks, y_clean.astype(np.float32)


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


# ─── Run one ───────────────────────────────────────────────────────────

def run_one(ms_cfg: dict, n_points: int, noise_std: float, seed: int, args) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    x_full, all_peaks, y_clean = generate_ms_spectrum(ms_cfg["n_clusters"], ms_cfg["resolution"], rng)

    # Add noise
    y_noisy = y_clean + rng.normal(0, noise_std * np.mean(y_clean), y_clean.shape).astype(np.float32)

    # Sub-sample
    idx = np.sort(rng.choice(len(x_full), n_points, replace=False))
    x_train = x_full[idx]
    y_train = y_noisy[idx]

    split = max(1, int(n_points * 0.8))
    x_tr, x_val = x_train[:split], x_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    x_test, y_test = x_full, y_clean

    sx, sy = StandardScaler(), StandardScaler()
    X_tr_t = torch.tensor(sx.fit_transform(x_tr), dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(sx.transform(x_val), dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(sx.transform(x_test), dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(sy.fit_transform(y_tr), dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(sy.transform(y_val), dtype=torch.float32, device=DEVICE)

    results = {}

    # CFNN-Hybrid
    cfnn, cfnn_cfg = grid_train(lambda: CFNNHybrid(), X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    cfnn.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(cfnn(X_test_t).cpu().numpy().reshape(-1, 1))
    results["CFNN-Hybrid"] = {
        "params": count_trainable_params(cfnn), "cfg": cfnn_cfg,
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
        "test_R2": float(r2_score(y_test, pred)),
    }

    # MLP-matched
    cfnn_p = count_trainable_params(CFNNHybrid())
    mlp_cfg = find_mlp_config_for_budget(1, cfnn_p, 1, (2, 3, 4), 256)
    mlp, mlp_cfg_chosen = grid_train(
        lambda: BudgetMLP(1, 1, mlp_cfg["hidden_dim"], mlp_cfg["num_layers"]),
        X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    mlp.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(mlp(X_test_t).cpu().numpy().reshape(-1, 1))
    results["MLP-matched"] = {
        "params": count_trainable_params(mlp), "cfg": mlp_cfg_chosen,
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
        "test_R2": float(r2_score(y_test, pred)),
    }

    # MLP-1000
    mlp1k_cfg = find_mlp_config_for_budget(1, 1000, 1, (2, 3), 512)
    mlp1k, _ = grid_train(
        lambda: BudgetMLP(1, 1, mlp1k_cfg["hidden_dim"], mlp1k_cfg["num_layers"]),
        X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    mlp1k.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(mlp1k(X_test_t).cpu().numpy().reshape(-1, 1))
    results["MLP-1000"] = {
        "params": count_trainable_params(mlp1k),
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
        "test_R2": float(r2_score(y_test, pred)),
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
        configs = MS_CONFIGS[:1]
        n_points_list = [120]
        noise_list = [0.02]
        seeds = [42]
        args.epochs = min(args.epochs, 80)
        args.patience = min(args.patience, 15)
    else:
        configs = MS_CONFIGS
        n_points_list = N_POINTS
        noise_list = NOISE_LEVELS
        seeds = SEEDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {}
    total = len(configs) * len(n_points_list) * len(noise_list) * len(seeds)
    done = 0

    for cfg in configs:
        for np_ in n_points_list:
            for noise in noise_list:
                for seed in seeds:
                    done += 1
                    print(f"[{done}/{total}] {cfg['name']} n={np_} noise={noise} seed={seed}", flush=True)
                    key = f"{cfg['name']}@{np_}@{noise}@{seed}"
                    raw[key] = run_one(cfg, np_, noise, seed, args)
                    (OUT_DIR / "raw.json").write_text(
                        json.dumps(raw, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    models_order = ["CFNN-Hybrid", "MLP-matched", "MLP-1000"]
    summary = {}
    for cfg in configs:
        for np_ in n_points_list:
            for noise in noise_list:
                for model in models_order:
                    r2s = []
                    for seed in seeds:
                        k = f"{cfg['name']}@{np_}@{noise}@{seed}"
                        if k in raw and model in raw[k]:
                            r2s.append(raw[k][model]["test_R2"])
                    if r2s:
                        arr = np.array(r2s)
                        sk = f"{cfg['name']}@{np_}@{noise}@{model}"
                        summary[sk] = {
                            "config": cfg["name"], "n_points": np_, "noise": noise,
                            "model": model,
                            "R2_mean": float(np.mean(arr)),
                            "R2_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                            "n_seeds": len(r2s),
                        }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    print("\n" + "=" * 120)
    print("Mass Spec Peak Detection: R² by config")
    for cfg in configs:
        print(f"\n--- {cfg['name']}: {cfg['desc']} ---")
        print(f"{'n':>6s} {'noise':>5s}", end="")
        for m in models_order: print(f"  {m:>20s}", end="")
        print()
        for np_ in n_points_list:
            for noise in noise_list:
                print(f"{np_:>6d} {noise:>5.2f}", end="")
                for m in models_order:
                    k = f"{cfg['name']}@{np_}@{noise}@{m}"
                    if k in summary:
                        print(f"  {summary[k]['R2_mean']:>8.4f}±{summary[k]['R2_std']:.4f}", end="")
                    else:
                        print(f"  {'—':>15s}", end="")
                print()

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
