#!/usr/bin/env python3
"""[DEPRECATED 2026-07-06] Superseded by run_nmr_pareto.py (fair-HPO, real
molecule peak tables, KAN included). Kept for provenance; do NOT cite its
results — models here were not HPO'd. See NMR_半真实_结论.md.

Experiment A: NMR/MRI Spectral Deconvolution.

Simulates realistic NMR spectra as sums of Lorentzian peaks (chemical shifts).
Tests CFNN's ability to resolve overlapping peaks with few training points.

Design:
1. Generate NMR-like spectra: 2-10 Lorentzian peaks with realistic chemical shifts
2. Add noise (thermal + baseline drift)
3. Compare CFNN vs MLP at matched parameter budget
4. Metrics: R², peak position recovery error, peak area recovery
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
OUT_DIR = ROOT / "experiment_refine" / "nmr_spectral_results"
SEEDS = [42, 123, 456, 789, 1024]

# NMR spectral configurations: varying number of peaks and overlap
# Chemical shift range: 0-12 ppm (typical 1H NMR)
NMR_CONFIGS = [
    {"name": "nmr_sparse",  "n_peaks": 3,  "sep": "wide",    "desc": "3 resolved peaks, wide separation"},
    {"name": "nmr_medium",  "n_peaks": 5,  "sep": "medium",  "desc": "5 peaks, moderate overlap"},
    {"name": "nmr_dense",   "n_peaks": 8,  "sep": "dense",   "desc": "8 peaks, significant overlap"},
    {"name": "nmr_crowded", "n_peaks": 12, "sep": "crowded", "desc": "12 peaks, heavy overlap (protein-like)"},
]

# Noise levels (typical NMR SNR: 50:1 to 1000:1)
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10]

# Training points per spectrum
N_POINTS = [60, 120, 240, 480]

# Full resolution for evaluation
FULL_N_POINTS = 2000


def lorentzian(x: np.ndarray, A: float, mu: float, gamma: float) -> np.ndarray:
    """Lorentzian peak: A · γ² / ((x-μ)² + γ²)."""
    return A * gamma * gamma / ((x - mu) ** 2 + gamma * gamma)


def generate_nmr_spectrum(n_peaks: int, separation: str, rng: np.random.Generator
                          ) -> tuple[np.ndarray, list[tuple]]:
    """Generate a realistic NMR spectrum with random peak parameters.

    Returns:
        x: chemical shift axis (12 to 0 ppm, decreasing)
        peaks: list of (A, mu, gamma) tuples
    """
    # Chemical shift range: 0-12 ppm (typical 1H NMR), reversed direction
    # Typical peak widths (gamma): 0.01-0.05 ppm for small molecules
    # Peak amplitudes: 1-100 (relative)

    # Width depends on separation
    if separation == "wide":
        gamma_range = (0.03, 0.08)
        min_sep_factor = 3.0
    elif separation == "medium":
        gamma_range = (0.04, 0.10)
        min_sep_factor = 1.5
    elif separation == "dense":
        gamma_range = (0.05, 0.12)
        min_sep_factor = 0.8
    else:  # crowded
        gamma_range = (0.06, 0.15)
        min_sep_factor = 0.4

    # Generate peaks with minimum separation
    peaks = []
    mu_range = (1.0, 11.0)  # avoid edges
    max_attempts = 200

    for _ in range(max_attempts):
        if len(peaks) >= n_peaks:
            break
        mu = rng.uniform(*mu_range)
        gamma = rng.uniform(*gamma_range)
        A = rng.uniform(5.0, 100.0)

        # Check minimum separation from existing peaks
        ok = True
        for _, existing_mu, existing_gamma in peaks:
            avg_gamma = (gamma + existing_gamma) / 2
            if abs(mu - existing_mu) < min_sep_factor * avg_gamma:
                ok = False
                break
        if ok:
            peaks.append((A, mu, gamma))

    # Sort by chemical shift (descending, NMR convention)
    peaks.sort(key=lambda p: -p[1])

    # Full x axis: decreasing ppm (NMR convention)
    x = np.linspace(12.0, 0.0, FULL_N_POINTS, dtype=np.float32).reshape(-1, 1)

    return x, peaks


def generate_noisy_spectrum(x: np.ndarray, peaks: list[tuple],
                            noise_std: float, rng: np.random.Generator,
                            baseline_order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Generate clean + noisy spectrum with baseline drift."""
    y_clean = np.zeros_like(x)
    for A, mu, gamma in peaks:
        y_clean += lorentzian(x, A, mu, gamma)

    # Add baseline drift (polynomial)
    if baseline_order > 0:
        coeffs = rng.uniform(-0.5, 0.5, baseline_order + 1) * noise_std * np.mean(y_clean)
        baseline = np.polyval(coeffs, np.linspace(-1, 1, len(x))).reshape(-1, 1)
    else:
        baseline = 0

    y_noisy = y_clean + baseline + rng.normal(0, noise_std * np.mean(y_clean), y_clean.shape).astype(np.float32)

    return y_clean.astype(np.float32), y_noisy.astype(np.float32)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


# ─── Peak Recovery Metrics ─────────────────────────────────────────────

def compute_peak_metrics(y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray,
                         true_peaks: list[tuple]) -> dict:
    """Compute peak position and amplitude recovery metrics.

    Simple approach: find local maxima and match to ground truth peaks.
    """
    # Simple peak detection: find local maxima
    def find_peaks_1d(signal, min_height=0.05):
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if signal[i] > min_height * np.max(signal):
                    peaks.append((i, signal[i]))
        return peaks

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    true_peak_locs = [find_peaks_1d(y_true_flat)]
    pred_peak_locs = [find_peaks_1d(y_pred_flat)]

    # Match predicted peaks to nearest true peak
    matched = 0
    total_pos_error = 0.0
    matched_pred_indices = set()

    for (i, _) in true_peak_locs[0]:
        best_dist = float("inf")
        best_j = -1
        for j, (pi, _) in enumerate(pred_peak_locs[0]):
            if j in matched_pred_indices:
                continue
            dist = abs(pi - i)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        # Accept if within 20 points (0.12 ppm at 2000 pts / 12 ppm)
        if best_j >= 0 and best_dist < 20:
            matched += 1
            total_pos_error += best_dist * (12.0 / 2000)  # convert to ppm
            matched_pred_indices.add(best_j)

    n_true = len(true_peak_locs[0])
    precision = matched / max(len(pred_peak_locs[0]), 1)
    recall = matched / max(n_true, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    avg_pos_error = total_pos_error / max(matched, 1)

    return {
        "peak_precision": precision,
        "peak_recall": recall,
        "peak_f1": f1,
        "peak_pos_error_ppm": avg_pos_error,
    }


# ─── Run one config ────────────────────────────────────────────────────

def run_one(nmr_cfg: dict, n_points: int, noise_std: float, seed: int, args) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # Generate spectrum
    x_full, peaks = generate_nmr_spectrum(nmr_cfg["n_peaks"], nmr_cfg["sep"], rng)
    y_clean, y_noisy = generate_noisy_spectrum(x_full, peaks, noise_std, rng)

    # Sub-sample training points (random, sorted for stability)
    idx = np.sort(rng.choice(len(x_full), n_points, replace=False))
    x_train = x_full[idx]
    y_train = y_noisy[idx]

    # Split train/val
    split = max(1, int(n_points * 0.8))
    x_tr, x_val = x_train[:split], x_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    # Test: full clean spectrum (evaluate reconstruction)
    x_test, y_test = x_full, y_clean

    # Standardize
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
    pm = compute_peak_metrics(y_test, pred, x_test, peaks)
    results["CFNN-Hybrid"] = {
        "params": count_trainable_params(cfnn), "cfg": cfnn_cfg,
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
        "test_R2": float(r2_score(y_test, pred)),
        **pm,
    }

    # MLP matched
    cfnn_p = count_trainable_params(CFNNHybrid())
    mlp_cfg = find_mlp_config_for_budget(1, cfnn_p, 1, (2, 3, 4), 256)
    mlp, mlp_cfg_chosen = grid_train(
        lambda: BudgetMLP(1, 1, mlp_cfg["hidden_dim"], mlp_cfg["num_layers"]),
        X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    mlp.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(mlp(X_test_t).cpu().numpy().reshape(-1, 1))
    pm = compute_peak_metrics(y_test, pred, x_test, peaks)
    results["MLP-matched"] = {
        "params": count_trainable_params(mlp), "cfg": mlp_cfg_chosen,
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
        "test_R2": float(r2_score(y_test, pred)),
        **pm,
    }

    # MLP-1000
    mlp1k_cfg = find_mlp_config_for_budget(1, 1000, 1, (2, 3), 512)
    mlp1k, _ = grid_train(
        lambda: BudgetMLP(1, 1, mlp1k_cfg["hidden_dim"], mlp1k_cfg["num_layers"]),
        X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    mlp1k.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(mlp1k(X_test_t).cpu().numpy().reshape(-1, 1))
    pm = compute_peak_metrics(y_test, pred, x_test, peaks)
    results["MLP-1000"] = {
        "params": count_trainable_params(mlp1k),
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
        "test_R2": float(r2_score(y_test, pred)),
        **pm,
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
        configs = NMR_CONFIGS[:1]
        n_points_list = [120]
        noise_list = [0.02]
        seeds = [42]
        args.epochs = min(args.epochs, 80)
        args.patience = min(args.patience, 15)
    else:
        configs = NMR_CONFIGS
        n_points_list = N_POINTS
        noise_list = NOISE_LEVELS
        seeds = SEEDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {}
    total = len(configs) * len(n_points_list) * len(noise_list) * len(seeds)
    done = 0

    with open(str(OUT_DIR / "run_params.json"), "w") as f:
        json.dump({
            "device": str(DEVICE),
            "models": ["CFNN-Hybrid", "MLP-matched", "MLP-1000"],
            "seeds": seeds,
            "n_points": n_points_list,
            "noise_levels": noise_list,
            "configs": [c["name"] for c in configs],
            "epochs": args.epochs,
            "patience": args.patience,
            "lrs": args.lrs,
            "weight_decays": args.weight_decays,
            "full_n_points": FULL_N_POINTS,
        }, f, indent=2)

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

    # Summarize
    summary = {}
    models_order = ["CFNN-Hybrid", "MLP-matched", "MLP-1000"]
    for cfg in configs:
        for np_ in n_points_list:
            for noise in noise_list:
                for model in models_order:
                    r2s, f1s, pos_errs = [], [], []
                    for seed in seeds:
                        key = f"{cfg['name']}@{np_}@{noise}@{seed}"
                        if key in raw and model in raw[key]:
                            r2s.append(raw[key][model]["test_R2"])
                            f1s.append(raw[key][model]["peak_f1"])
                            pos_errs.append(raw[key][model]["peak_pos_error_ppm"])
                    if r2s:
                        arr = np.array(r2s)
                        sk = f"{cfg['name']}@{np_}@{noise}@{model}"
                        summary[sk] = {
                            "config": cfg["name"], "n_points": np_, "noise": noise,
                            "model": model,
                            "R2_mean": float(np.mean(arr)),
                            "R2_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                            "peak_F1_mean": float(np.mean(f1s)),
                            "peak_pos_error_ppm_mean": float(np.mean(pos_errs)),
                            "n_seeds": len(r2s),
                        }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    # Print table
    print("\n" + "=" * 120)
    print("NMR Spectral Deconvolution: R² by config")
    for cfg in configs:
        print(f"\n--- {cfg['name']}: {cfg['desc']} ({cfg['n_peaks']} peaks) ---")
        print(f"{'n':>6s} {'noise':>5s}", end="")
        for m in models_order: print(f"  {m:>22s}", end="")
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
