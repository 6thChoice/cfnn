#!/usr/bin/env python3
"""Resume spectral fitting: only run missing configs."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))
from param_matched_utils import NumpySafeEncoder, count_trainable_params, find_mlp_config_for_budget

# Copy the original imports and functions
import argparse
from copy import deepcopy
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "experiment_refine" / "spectral_fitting_results"
SEEDS = [42, 123, 456, 789, 1024]

SPECTRA = [
    {"name": "single_sharp",  "peaks": [(100, 0.0, 0.5)],                          "desc": "Single sharp peak"},
    {"name": "single_broad",  "peaks": [(100, 0.0, 3.0)],                          "desc": "Single broad peak"},
    {"name": "double_resolved", "peaks": [(80, -5.0, 1.0), (100, 5.0, 1.0)],       "desc": "Two resolved peaks"},
    {"name": "double_overlap", "peaks": [(80, -2.5, 2.0), (100, 2.5, 2.0)],        "desc": "Two overlapping peaks"},
    {"name": "triple",        "peaks": [(70, -6.0, 0.8), (100, 0.0, 1.2), (80, 6.0, 0.8)], "desc": "Three peaks, mixed widths"},
    {"name": "quad_resolved", "peaks": [(60, -9, 0.8), (80, -3, 0.8), (90, 3, 1.0), (70, 9, 1.0)], "desc": "Four resolved peaks"},
]

def lorentzian(x, A, mu, gamma):
    return A * gamma * gamma / ((x - mu) ** 2 + gamma * gamma)

def generate_spectrum(x, peaks):
    y = np.zeros_like(x)
    for A, mu, gamma in peaks:
        y += lorentzian(x, A, mu, gamma)
    return y.reshape(-1, 1).astype(np.float32)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

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

def run_one(spectrum, noise_std, n_points, seed, args):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    x_full = np.linspace(-15, 15, 2000, dtype=np.float32).reshape(-1, 1)
    y_clean = generate_spectrum(x_full, spectrum["peaks"])
    idx = np.sort(rng.choice(len(x_full), n_points, replace=False))
    x_train = x_full[idx]
    y_train = y_clean[idx] + rng.normal(0, noise_std * np.mean(y_clean), (n_points, 1)).astype(np.float32)
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
    cfnn, cfnn_cfg = grid_train(lambda: CFNNHybrid(), X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    cfnn.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(cfnn(X_test_t).cpu().numpy().reshape(-1, 1))
    results["CFNN-Hybrid"] = {"params": count_trainable_params(cfnn), "cfg": cfnn_cfg,
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
        "test_R2": float(r2_score(y_test, pred))}

    cfnn_p = count_trainable_params(CFNNHybrid())
    mlp_cfg = find_mlp_config_for_budget(1, cfnn_p, 1, (2, 3, 4), 256)
    mlp, mlp_cfg_chosen = grid_train(
        lambda: BudgetMLP(1, 1, mlp_cfg["hidden_dim"], mlp_cfg["num_layers"]),
        X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    mlp.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(mlp(X_test_t).cpu().numpy().reshape(-1, 1))
    results["MLP-matched"] = {"params": count_trainable_params(mlp), "cfg": mlp_cfg_chosen,
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
        "test_R2": float(r2_score(y_test, pred))}

    mlp1k_cfg = find_mlp_config_for_budget(1, 1000, 1, (2, 3), 512)
    mlp1k, _ = grid_train(
        lambda: BudgetMLP(1, 1, mlp1k_cfg["hidden_dim"], mlp1k_cfg["num_layers"]),
        X_tr_t, y_tr_t, X_val_t, y_val_t, args)
    mlp1k.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(mlp1k(X_test_t).cpu().numpy().reshape(-1, 1))
    results["MLP-1000"] = {"params": count_trainable_params(mlp1k),
        "test_RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
        "test_R2": float(r2_score(y_test, pred))}

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lrs", type=float, nargs="+", default=[5e-3, 1e-3])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0])
    args = parser.parse_args()

    # Load existing results
    import os
    raw_path = OUT_DIR / "raw.json"
    if raw_path.exists():
        with open(raw_path) as f:
            raw = json.load(f)
    else:
        raw = {}

    noise_levels = [0.0, 0.02, 0.05, 0.10]
    n_points_list = [30, 60, 120, 240]
    seeds = SEEDS

    total = len(SPECTRA) * len(noise_levels) * len(n_points_list) * len(seeds)

    # Build expected keys
    expected = set()
    for sp in SPECTRA:
        for noise in noise_levels:
            for np_ in n_points_list:
                for seed in seeds:
                    expected.add(f"{sp['name']}@{noise}@{np_}@{seed}")

    done = len(raw)
    missing = expected - set(raw.keys())
    print(f"Existing: {done}/{total}, Missing: {len(missing)}")

    for sp in SPECTRA:
        for noise in noise_levels:
            for np_ in n_points_list:
                for seed in seeds:
                    key = f"{sp['name']}@{noise}@{np_}@{seed}"
                    if key in raw:
                        continue
                    done += 1
                    print(f"[{done}/{total}] {sp['name']} noise={noise} n={np_} seed={seed}", flush=True)
                    raw[key] = run_one(sp, noise, np_, seed, args)
                    with open(raw_path, 'w', encoding='utf-8') as f:
                        json.dump(raw, f, indent=2, cls=NumpySafeEncoder)

    # Summarize
    models_order = ["CFNN-Hybrid", "MLP-matched", "MLP-1000"]
    summary = {}
    for sp in SPECTRA:
        for noise in noise_levels:
            for np_ in n_points_list:
                for model in models_order:
                    r2s = []
                    for seed in seeds:
                        k = f"{sp['name']}@{noise}@{np_}@{seed}"
                        if k in raw and model in raw[k]:
                            r2s.append(raw[k][model]["test_R2"])
                    if r2s:
                        arr = np.array(r2s)
                        sk = f"{sp['name']}@{noise}@{np_}@{model}"
                        summary[sk] = {
                            "spectrum": sp["name"], "noise": noise, "n_points": np_,
                            "model": model, "R2_mean": float(np.mean(arr)),
                            "R2_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                            "n_seeds": len(r2s),
                        }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, cls=NumpySafeEncoder), encoding="utf-8")
    print(f"\nDone. {len(raw)} total. Results in {OUT_DIR}")

if __name__ == "__main__":
    main()
