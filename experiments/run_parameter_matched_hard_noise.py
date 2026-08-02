#!/usr/bin/env python3
"""Parameter-matched MLP baselines for P1 hard-noise degradation experiment.

Current MLP: 3x96 = 19105 params (large-reference).
CFNN-Hybrid: 124 params.
KAN: ~1513 params.

Anchors: 124 (Hybrid), 500 (mid), 1500 (KAN budget), 19105 (current large MLP, reference).
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))
from param_matched_utils import NumpySafeEncoder, count_trainable_params, find_mlp_config_for_budget, budget_metadata

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "experiment_refine" / "parameter_matched_baselines" / "hard_noise"

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

ANCHORS = {
    "MLP_matched_124": 124,    # matched to CFNN-Hybrid
    "MLP_matched_500": 500,    # mid budget
    "MLP_matched_1500": 1500,  # matched to KAN budget
}
# Reference: current large MLP at 19105 params


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hard_target(X: np.ndarray, eps: float = 0.045) -> np.ndarray:
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    z = 0.85 * x1 - 0.55 * x2 + 0.35 * x3 - 0.10
    y = 1.0 / (z * z + eps) + 0.35 * x1 * x2
    return y.reshape(-1, 1).astype(np.float32)


class BudgetMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, activation: str):
        super().__init__()
        acts = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "GELU": nn.GELU}
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(acts[activation]())
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_one(noise_level: float, seed: int, args) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    n = args.n_samples
    noise_scale = args.noise_scale

    X_clean = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    y = hard_target(X_clean)

    if noise_level > 0:
        X_obs = X_clean + rng.normal(0.0, noise_level * noise_scale, size=X_clean.shape).astype(np.float32)
    else:
        X_obs = X_clean.copy()

    X_train, X_tmp, y_train, y_tmp = train_test_split(X_obs, y, test_size=0.35, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.85714, random_state=seed)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_t = torch.tensor(scaler_X.fit_transform(X_train), dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(scaler_X.transform(X_val), dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(scaler_X.transform(X_test), dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(scaler_y.fit_transform(y_train), dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(scaler_y.transform(y_val), dtype=torch.float32, device=DEVICE)

    results = {}
    for anchor_name, target_params in ANCHORS.items():
        cfg = find_mlp_config_for_budget(
            input_dim=3,
            target_params=target_params,
            output_dim=1,
            depths=tuple(args.depths),
            max_hidden=args.max_hidden,
        )

        best_val_loss = float("inf")
        best_state = None
        best_train_cfg = {}
        best_params = 0

        for activation in args.activations:
            for lr in args.lrs:
                for wd in args.weight_decays:
                    set_seed(seed + abs(hash(f"{anchor_name}{activation}{lr}{wd}")) % 10000)
                    model = BudgetMLP(3, 1, cfg["hidden_dim"], cfg["num_layers"], activation).to(DEVICE)
                    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                    criterion = nn.MSELoss()

                    m_best_state = deepcopy(model.state_dict())
                    m_best_val = float("inf")
                    stale = 0

                    for epoch in range(args.max_epochs):
                        model.train()
                        opt.zero_grad(set_to_none=True)
                        pred = model(X_train_t)
                        loss = criterion(pred, y_train_t)
                        if not torch.isfinite(loss):
                            break
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()

                        if epoch % 5 == 0:
                            model.eval()
                            with torch.no_grad():
                                val_loss = criterion(model(X_val_t), y_val_t).item()
                            if torch.isfinite(torch.tensor(val_loss)) and val_loss < m_best_val - 1e-6:
                                m_best_val = val_loss
                                m_best_state = deepcopy(model.state_dict())
                                stale = 0
                            else:
                                stale += 1
                                if stale >= args.patience:
                                    break

                    if m_best_val < best_val_loss:
                        best_val_loss = m_best_val
                        best_state = m_best_state
                        best_train_cfg = {"activation": activation, "lr": lr, "weight_decay": wd}
                        best_params = count_trainable_params(model)

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_test_t).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled)

        meta = budget_metadata("MLP-parameter-matched", best_params, target_params)
        results[anchor_name] = {
            **meta,
            "mlp_config": cfg,
            "best_train_cfg": best_train_cfg,
            "noise_level": noise_level,
            "test_metrics": {
                "MSE": float(mean_squared_error(y_test, pred)),
                "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
                "MAE": float(mean_absolute_error(y_test, pred)),
                "R2": float(r2_score(y_test, pred)),
            },
        }
    return results


def summarize(raw):
    """raw: dict of {f'{anchor}@{noise}@{seed}': results}"""
    grouped = {}
    for key, result in raw.items():
        parts = key.split("@")
        anchor, noise_str, seed_str = parts[0], parts[1], parts[2]
        noise = float(noise_str)
        gk = (anchor, noise)
        grouped.setdefault(gk, {"anchor": anchor, "noise_level": noise, "rows": []})["rows"].append(result)

    summary = {}
    for gk, group in grouped.items():
        vals = group["rows"]
        anchor, noise = gk
        metrics = {}
        for metric in ["MSE", "RMSE", "MAE", "R2"]:
            arr = np.array([v["test_metrics"][metric] for v in vals], dtype=float)
            metrics[f"{metric}_mean"] = float(np.mean(arr))
            metrics[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        summary[f"{anchor}@{noise}"] = {
            "anchor": anchor,
            "noise_level": noise,
            "target_params": int(vals[0]["target_params"]),
            "trainable_params_mean": float(np.mean([v["trainable_params"] for v in vals])),
            "n_seeds": len(vals),
            **metrics,
        }
    return summary


def write_outputs(raw, summary, args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raw.json").write_text(json.dumps(raw, indent=2, cls=NumpySafeEncoder), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, cls=NumpySafeEncoder), encoding="utf-8")

    lines = ["# Hard-Noise Degradation: Parameter-Matched MLP Baselines", ""]
    lines.append(f"Device: {DEVICE} | seeds: {DEFAULT_SEEDS} | anchors: {dict(ANCHORS)}")
    lines.append(f"Reference: MLP 3x96 = 19105 params (large-capacity), CFNN-Hybrid = 124 params, KAN ~1513 params.")
    lines.append("")
    lines.append("## R² per noise level")
    lines.append("")
    anchors = sorted(set(v["anchor"] for v in summary.values()))
    noise_levels = sorted(set(float(v["noise_level"]) for v in summary.values()))
    lines.append("| Noise | " + " | ".join(anchors) + " |")
    lines.append("|---" + "|---" * len(anchors) + "|")
    for nl in noise_levels:
        row = [f"{nl*100:.0f}%"]
        for a in anchors:
            key = f"{a}@{nl}"
            if key in summary:
                row.append(f"{summary[key]['R2_mean']:.4f} ± {summary[key]['R2_std']:.4f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## RMSE per noise level")
    lines.append("")
    lines.append("| Noise | " + " | ".join(anchors) + " |")
    lines.append("|---" + "|---" * len(anchors) + "|")
    for nl in noise_levels:
        row = [f"{nl*100:.0f}%"]
        for a in anchors:
            key = f"{a}@{nl}"
            if key in summary:
                row.append(f"{summary[key]['RMSE_mean']:.4f} ± {summary[key]['RMSE_std']:.4f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-samples", type=int, default=3500)
    parser.add_argument("--max-epochs", type=int, default=700)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--noise-scale", type=float, default=0.45)
    parser.add_argument("--max-hidden", type=int, default=256)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--activations", nargs="+", default=["Tanh", "ReLU", "GELU"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[3e-3, 1e-2, 1e-3])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-5])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        noise_levels = [0.0, 0.5]
        seeds = [42]
        args.max_epochs = min(args.max_epochs, 180)
        args.patience = min(args.patience, 20)
        args.activations = ["Tanh"]
        args.lrs = [1e-2]
        args.weight_decays = [0.0]
    else:
        noise_levels = DEFAULT_NOISE_LEVELS
        seeds = DEFAULT_SEEDS

    raw: dict = {}
    total = len(ANCHORS) * len(noise_levels) * len(seeds)
    done = 0
    for noise_level in noise_levels:
        for seed in seeds:
            all_anchor_results = run_one(noise_level, seed, args)
            for anchor_name in ANCHORS:
                done += 1
                print(f"[{done}/{total}] anchor={anchor_name} noise={noise_level:.1f} seed={seed}", flush=True)
                key = f"{anchor_name}@{noise_level}@{seed}"
                raw[key] = all_anchor_results[anchor_name]
                # incremental save
                write_outputs(raw, summarize(raw), args)

    summary = summarize(raw)
    write_outputs(raw, summary, args)
    print(f"Done. Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
