#!/usr/bin/env python3
"""Parameter-matched MLP baselines for Noise Robustness & Feature Attribution experiment.

Anchors: CFNN(71), Hybrid(692), MoE(215), Boost(1420).
Current MLP 2x64 (4993 params) kept as large-capacity reference.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiment_refine" / "CFNN-Experiments" / "Noise Robustness & Feature Attribution"
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(ROOT / "experiment_refine"))

from data_noise import load_noise_data
from param_matched_utils import NumpySafeEncoder, count_trainable_params, find_mlp_config_for_budget, budget_metadata

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "experiment_refine" / "parameter_matched_baselines" / "noise"

SEEDS = [42, 123, 456, 789, 2026]
FIXED_NOISE_FEATURES = 4

ANCHORS = {
    "CFNN_71": 71,
    "Hybrid_692": 692,
    "MoE_215": 215,
    "Boost_1420": 1420,
}


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


def train_with_early_stopping(model, train_loader, val_loader, *, lr, weight_decay, max_epochs, patience):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(DEVICE)
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0
    epochs_done = 0
    for epoch in range(max_epochs):
        epochs_done = epoch + 1
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                val_losses.append(criterion(model(vx), vy).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.to(DEVICE)
    return model, epochs_done, best_val


def compute_metrics(y_true, y_pred):
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = 1 - mse / max(np.var(y_true), 1e-30)
    return {"MSE": mse, "MAE": mae, "R2": r2}


def evaluate_config(task_cfg, train_cfg):
    X_train, y_train, X_val, y_val, X_test, y_test = (
        task_cfg["X_train"], task_cfg["y_train"],
        task_cfg["X_val"], task_cfg["y_val"],
        task_cfg["X_test"], task_cfg["y_test"],
    )
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=128)

    cfg = task_cfg["mlp_config"]
    model = BudgetMLP(task_cfg["input_dim"], 1, cfg["hidden_dim"], cfg["num_layers"], train_cfg["activation"])
    model, epochs_done, best_val = train_with_early_stopping(
        model, train_loader, val_loader,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        max_epochs=train_cfg["max_epochs"],
        patience=train_cfg["patience"],
    )
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(X_val).to(DEVICE)).cpu().numpy()
        test_pred = model(torch.from_numpy(X_test).to(DEVICE)).cpu().numpy()
    val_metrics = compute_metrics(y_val, val_pred)
    test_metrics = compute_metrics(y_test, test_pred)
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "epochs_done": epochs_done,
        "best_val_loss_std_space": best_val,
        "trainable_params": count_trainable_params(model),
    }


def run_one(anchor_name, target_params, seed, args):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = load_noise_data(seed=seed, n_samples=5000, n_noise_features=FIXED_NOISE_FEATURES)
    in_dim = data["n_features"]

    cfg = find_mlp_config_for_budget(
        input_dim=in_dim,
        target_params=target_params,
        output_dim=1,
        depths=tuple(args.depths),
        max_hidden=args.max_hidden,
    )
    task_cfg = {
        "input_dim": in_dim,
        "mlp_config": cfg,
        "X_train": data["X_train"],
        "y_train": data["y_train"],  # keep 2D
        "X_val": data["X_val"],
        "y_val": data["y_val"],      # keep 2D
        "X_test": data["X_test"],
        "y_test": data["y_test"],    # keep 2D
    }

    search = []
    for activation in args.activations:
        for lr in args.lrs:
            for weight_decay in args.weight_decays:
                train_cfg = {
                    "activation": activation,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                }
                result = evaluate_config(task_cfg, train_cfg)
                search.append({"train_cfg": train_cfg, **result})
    best = min(search, key=lambda r: r["val_metrics"]["MSE"])
    meta = budget_metadata("MLP-parameter-matched", best["trainable_params"], target_params)
    return {
        "anchor": anchor_name,
        "seed": seed,
        "mlp_config": cfg,
        **meta,
        "best_train_cfg": best["train_cfg"],
        "val_metrics": best["val_metrics"],
        "test_metrics": best["test_metrics"],
        "epochs_done": best["epochs_done"],
        "all_search_results": search,
    }


def aggregate(rows):
    grouped = {}
    for row in rows:
        key = row["anchor"]
        grouped.setdefault(key, {"anchor": key, "rows": []})["rows"].append(row)
    summary = {}
    for key, group in grouped.items():
        vals = group["rows"]
        metrics = {}
        for metric in ["MSE", "MAE", "R2"]:
            arr = np.array([v["test_metrics"][metric] for v in vals], dtype=float)
            metrics[f"{metric}_mean"] = float(np.mean(arr))
            metrics[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        params = np.array([v["trainable_params"] for v in vals], dtype=float)
        summary[key] = {
            "anchor": key,
            "target_params": int(vals[0]["target_params"]),
            "trainable_params_mean": float(np.mean(params)),
            "n_seeds": len(vals),
            **metrics,
        }
    return summary


def write_outputs(rows, args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    (OUT_DIR / "raw.json").write_text(json.dumps(rows, indent=2, cls=NumpySafeEncoder), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, cls=NumpySafeEncoder), encoding="utf-8")
    lines = ["# Noise Robustness Parameter-Matched MLP Baselines", ""]
    lines.append(f"Device: {DEVICE} | seeds: {SEEDS} | fixed_noise_features={FIXED_NOISE_FEATURES}")
    lines.append(f"Current MLP 2x64: 4993 params (large-capacity reference only)")
    lines.append("")
    lines.append("| Anchor | Target params | MLP params | MSE (↓) | MAE (↓) | R² (↑) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for item in summary.values():
        lines.append(
            f"| {item['anchor']} | {item['target_params']} | "
            f"{item['trainable_params_mean']:.0f} | "
            f"{item['MSE_mean']:.4f} ± {item['MSE_std']:.4f} | "
            f"{item['MAE_mean']:.4f} ± {item['MAE_std']:.4f} | "
            f"{item['R2_mean']:.4f} ± {item['R2_std']:.4f} |"
        )
    lines.append("")
    lines.append("Interpretation: MLP baselines are controlled to CFNN-family parameter budgets. "
                 "The current MLP 2x64 (4993 params) is a large-capacity reference, not parameter-matched.")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--max-hidden", type=int, default=256)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--activations", nargs="+", default=["Tanh", "ReLU", "GELU"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-5])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        seeds = [42]
        args.max_epochs = min(args.max_epochs, 80)
        args.patience = min(args.patience, 8)
        args.activations = ["Tanh"]
        args.lrs = [1e-2]
        args.weight_decays = [0.0]
    else:
        seeds = SEEDS

    rows = []
    for anchor_name, target_params in ANCHORS.items():
        for seed in seeds:
            print(f"[noise matched] anchor={anchor_name} target={target_params} seed={seed}", flush=True)
            rows.append(run_one(anchor_name, target_params, seed, args))
            write_outputs(rows, args)
    write_outputs(rows, args)
    print(f"Done. Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
