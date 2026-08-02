#!/usr/bin/env python3
"""Parameter-matched MLP baselines for Synthetic Table 1.

This runner uses CFNN-family parameter counts as budget anchors and trains MLP
baselines whose trainable parameter counts are as close as possible to those
anchors. Larger tuned MLP HPO results remain reference baselines, not the
primary fairness comparison.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SYN_ROOT = ROOT / "git_codebase" / "synthetic_function_fit_runner" / "code"
sys.path.insert(0, str(ROOT / "git_codebase"))
sys.path.insert(0, str(ROOT / "experiment_refine"))

from common.protocol import compute_regression_metrics, set_seed, standardize_data  # noqa: E402
from param_matched_utils import NumpySafeEncoder, budget_metadata, count_trainable_params, find_mlp_config_for_budget  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "experiment_refine" / "parameter_matched_baselines" / "synthetic"

TASKS = {
    "Bilinear_Ratio": {
        "module": SYN_ROOT / "nn_hard_1" / "main_mlp_fair_hpo.py",
        "generator": "generate_bilinear_data",
        "input_dim": 3,
        "anchors": {"CFNN_budget_200": 200},
    },
    "Runge": {
        "module": SYN_ROOT / "nn_hard_2" / "main_mlp_fair_hpo.py",
        "generator": "generate_runge_data",
        "input_dim": 1,
        "anchors": {"CFNN_budget_200": 200},
    },
    "Rational_Interaction": {
        "module": SYN_ROOT / "cfnn_prefer_1" / "main_mlp_fair_hpo.py",
        "generator": "generate_rational_interaction_data",
        "input_dim": 3,
        "anchors": {"P0_Hybrid_244": 244, "P0_MoE_200": 200},
    },
    "Nested_Rational": {
        "module": SYN_ROOT / "cfnn_prefer_2" / "main_mlp_fair_hpo.py",
        "generator": "generate_nested_rational_data",
        "input_dim": 3,
        "anchors": {"P0_Hybrid_196": 196, "P0_MoE_200": 200},
    },
}


class BudgetMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, activation: str):
        super().__init__()
        acts = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "GELU": nn.GELU}
        if activation not in acts:
            raise ValueError(f"Unsupported activation: {activation}")
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


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def train_with_early_stopping(model, train_loader, val_loader, *, lr: float, weight_decay: float, max_epochs: int, patience: int):
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


def evaluate_config(task, data_std, cfg, train_cfg):
    X_train, y_train, X_val, y_val, X_test, y_test, _x_scaler, y_scaler = data_std
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=128)
    model = BudgetMLP(task["input_dim"], 1, cfg["hidden_dim"], cfg["num_layers"], train_cfg["activation"])
    model, epochs_done, best_val = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        max_epochs=train_cfg["max_epochs"],
        patience=train_cfg["patience"],
    )
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(X_val).to(DEVICE)).cpu().numpy()
        test_pred = model(torch.from_numpy(X_test).to(DEVICE)).cpu().numpy()
    val_metrics = compute_regression_metrics(y_val, val_pred, y_scaler)
    test_metrics = compute_regression_metrics(y_test, test_pred, y_scaler)
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "epochs_done": epochs_done,
        "best_val_loss_std_space": best_val,
        "trainable_params": count_trainable_params(model),
    }


def run_one(task_name: str, task: dict, anchor_name: str, target_params: int, seed: int, args) -> dict:
    set_seed(seed)
    module = import_module(task["module"], f"synthetic_{task_name}_{seed}")
    generator = getattr(module, task["generator"])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = generator(n_samples=args.n_samples)
    data_std = standardize_data(X_train, y_train, X_val, y_val, X_test, y_test)

    cfg = find_mlp_config_for_budget(
        input_dim=task["input_dim"],
        target_params=target_params,
        output_dim=1,
        depths=tuple(args.depths),
        max_hidden=args.max_hidden,
    )
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
                result = evaluate_config(task, data_std, cfg, train_cfg)
                search.append({"train_cfg": train_cfg, **result})
    best = min(search, key=lambda r: r["val_metrics"]["RMSE"])
    meta = budget_metadata("MLP-parameter-matched", best["trainable_params"], target_params)
    return {
        "task": task_name,
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


def aggregate(rows: list[dict]) -> dict:
    grouped: dict[str, dict] = {}
    for row in rows:
        key = f"{row['task']}::{row['anchor']}"
        grouped.setdefault(key, {"task": row["task"], "anchor": row["anchor"], "rows": []})["rows"].append(row)
    summary = {}
    for key, group in grouped.items():
        vals = group["rows"]
        metrics = {}
        for metric in ["RMSE", "R2", "MSE", "MAE"]:
            arr = np.array([v["test_metrics"][metric] for v in vals], dtype=float)
            metrics[f"{metric}_mean"] = float(np.mean(arr))
            metrics[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        params = np.array([v["trainable_params"] for v in vals], dtype=float)
        ratios = np.array([v["param_ratio"] for v in vals], dtype=float)
        summary[key] = {
            "task": group["task"],
            "anchor": group["anchor"],
            "target_params": int(vals[0]["target_params"]),
            "trainable_params_mean": float(np.mean(params)),
            "param_ratio_mean": float(np.mean(ratios)),
            "comparison_type": vals[0]["comparison_type"],
            "n_seeds": len(vals),
            **metrics,
        }
    return summary


def write_outputs(rows: list[dict], args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    (OUT_DIR / "raw.json").write_text(json.dumps(rows, indent=2, cls=NumpySafeEncoder), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, cls=NumpySafeEncoder), encoding="utf-8")
    lines = ["# Synthetic Table 1 Parameter-Matched MLP Baselines", ""]
    lines.append("| Task | Anchor | Target params | MLP params | Param ratio | RMSE | R² | Type |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for item in summary.values():
        lines.append(
            f"| {item['task']} | {item['anchor']} | {item['target_params']} | "
            f"{item['trainable_params_mean']:.0f} | {item['param_ratio_mean']:.3f} | "
            f"{item['RMSE_mean']:.6f} ± {item['RMSE_std']:.6f} | "
            f"{item['R2_mean']:.6f} ± {item['R2_std']:.6f} | {item['comparison_type']} |"
        )
    lines.append("")
    lines.append("Interpretation: these MLP baselines are controlled to CFNN-family parameter budgets and should be used before any larger MLP reference when making fairness claims.")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--max-hidden", type=int, default=256)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--activations", nargs="+", default=["Tanh", "ReLU", "GELU"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-5])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.n_samples = min(args.n_samples, 800)
        args.max_epochs = min(args.max_epochs, 80)
        args.patience = min(args.patience, 8)
        args.activations = ["Tanh"]
        args.lrs = [1e-2]
        args.weight_decays = [0.0]
        seeds = [42]
    else:
        seeds = [42, 123, 456, 789, 1024]

    rows = []
    for task_name, task in TASKS.items():
        for anchor_name, target_params in task["anchors"].items():
            for seed in seeds:
                print(f"[synthetic matched] task={task_name} anchor={anchor_name} seed={seed}", flush=True)
                rows.append(run_one(task_name, task, anchor_name, target_params, seed, args))
                write_outputs(rows, args)
    write_outputs(rows, args)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
