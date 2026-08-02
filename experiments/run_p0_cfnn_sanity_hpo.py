#!/usr/bin/env python3
"""P0-1: small-budget CFNN-side sanity HPO for revised Table 1 logic.

Purpose
-------
The A2 rerun tuned only the MLP baseline and showed that clean low-dimensional
synthetic fitting no longer strongly separates MLP from CFNN. This script adds a
small, reviewer-facing CFNN-side sanity check so the paper can say the revised
logic is not based on tuning MLP alone.

Scope is intentionally limited:
- two sensitive Table-1 functions (Rational Interaction where MLP surpassed old
  CFNN, and Nested Rational where CFNN only slightly led tuned MLP)
- two representative stabilized CFNN variants (Hybrid and MoE)
- small HPO on seed 42, then evaluate selected config on 3 or 5 seeds.

Outputs
-------
experiment_refine/p0_cfnn_sanity_hpo_results/
  raw.json
  summary.json
  SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/zxc/CodeBase/cofrnet")
CODE_DIR = ROOT / "git_codebase" / "synthetic_function_fit_runner" / "code"
sys.path.insert(0, str(CODE_DIR))
from cfnet import HybridRationalNet, MoE_Ensemble  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "experiment_refine" / "p0_cfnn_sanity_hpo_results"
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

OLD_BEST_CFNN_RMSE = {
    "Rational_Interaction": 0.0308,
    "Nested_Rational": 0.0198,
}
TUNED_MLP_RMSE = {
    "Rational_Interaction": 0.021159,
    "Nested_Rational": 0.020980,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rational_interaction(X: np.ndarray) -> np.ndarray:
    y = (X[:, 0] + X[:, 1]) / (1.0 + X[:, 2] ** 2)
    return y.reshape(-1, 1).astype(np.float32)


def nested_rational(X: np.ndarray) -> np.ndarray:
    inner = 1.0 / X[:, 2]
    middle = X[:, 1] + inner
    y = X[:, 0] + 1.0 / middle
    return y.reshape(-1, 1).astype(np.float32)


FUNCTIONS: Dict[str, Dict] = {
    "Rational_Interaction": {
        "input_dim": 3,
        "domain": (-2.0, 2.0),
        "noise": 0.03,
        "fn": rational_interaction,
    },
    "Nested_Rational": {
        "input_dim": 3,
        "domain": (0.5, 2.5),
        "noise": 0.02,
        "fn": nested_rational,
    },
}


def make_data(function_name: str, seed: int, n_samples: int):
    cfg = FUNCTIONS[function_name]
    rng = np.random.default_rng(seed)
    low, high = cfg["domain"]
    X = rng.uniform(low, high, size=(n_samples, cfg["input_dim"])).astype(np.float32)
    y = cfg["fn"](X)
    y = y + cfg["noise"] * rng.normal(size=y.shape).astype(np.float32)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.35, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=30 / 35, random_state=seed)

    xs = StandardScaler().fit(X_train)
    ys = StandardScaler().fit(y_train)
    arrays = {
        "X_train": xs.transform(X_train).astype(np.float32),
        "X_val": xs.transform(X_val).astype(np.float32),
        "X_test": xs.transform(X_test).astype(np.float32),
        "y_train": ys.transform(y_train).astype(np.float32),
        "y_val": ys.transform(y_val).astype(np.float32),
        "y_test": ys.transform(y_test).astype(np.float32),
        "y_test_orig": y_test.astype(np.float32),
        "y_scaler": ys,
    }
    return arrays


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_name: str, input_dim: int, params: Dict) -> nn.Module:
    degree = int(params["degree"])
    size = int(params["size"])
    if model_name == "Hybrid":
        return HybridRationalNet(input_dim, 1, unit_degree=degree, num_units=size)
    if model_name == "MoE":
        hp = {"input_dim": input_dim, "output_dim": 1, "shallow_depth_per_cofrnet": 2, "polynomial_degree": degree}
        model = MoE_Ensemble(hp)
        # deterministic random centers after set_seed(); initialized in standardized space
        for _ in range(size):
            model.add_expert()
        model.gating.centers = nn.Parameter(torch.randn(size, input_dim) * 0.9)
        model.gating.widths = nn.Parameter(torch.ones(size, 1))
        return model
    raise ValueError(model_name)


def tensors(data: Dict):
    keys = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
    return [torch.tensor(data[k], dtype=torch.float32, device=DEVICE) for k in keys]


def train_eval(function_name: str, model_name: str, params: Dict, seed: int, *, n_samples: int, max_epochs: int, patience: int) -> Dict:
    set_seed(seed)
    data = make_data(function_name, seed, n_samples)
    X_train, X_val, X_test, y_train, y_val, y_test = tensors(data)
    model = build_model(model_name, FUNCTIONS[function_name]["input_dim"], params).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(params.get("weight_decay", 0.0)))
    loss_fn = nn.MSELoss()
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0
    epochs_done = 0
    start = time.time()

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(params.get("grad_clip", 5.0)))
        opt.step()
        epochs_done = epoch + 1

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(X_val), y_val).item())
            if math.isfinite(val_loss) and val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test).detach().cpu().numpy().reshape(-1, 1)
    y_pred = data["y_scaler"].inverse_transform(pred_scaled).reshape(-1)
    y_true = data["y_test_orig"].reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    out = {
        "function": function_name,
        "model": model_name,
        "seed": seed,
        "params": params,
        "n_trainable_params": count_params(model),
        "best_val_scaled_mse": best_val,
        "epochs_done": epochs_done,
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "seconds": float(time.time() - start),
    }
    return out


def param_grid(model_name: str, smoke: bool) -> List[Dict]:
    if smoke:
        base = [{"degree": 3, "size": 3, "lr": 1e-2, "weight_decay": 0.0, "grad_clip": 5.0}]
        return base
    if model_name == "Hybrid":
        # Compact paper-facing sweep: enough to test whether a nearby CFNN
        # recipe closes the clean-fitting gap, without attempting a full HPO.
        degrees = [3, 5]
        sizes = [8, 12]
    else:
        degrees = [3, 5]
        sizes = [5, 8]
    return [
        {"degree": d, "size": s, "lr": lr, "weight_decay": 0.0, "grad_clip": 5.0}
        for d in degrees for s in sizes for lr in [1e-2, 3e-3]
    ]


def aggregate(runs: List[Dict]) -> Dict:
    out = {"n": len(runs)}
    for metric in ["MSE", "RMSE", "MAE", "R2", "epochs_done", "n_trainable_params"]:
        vals = np.asarray([r[metric] for r in runs], dtype=float)
        out[f"{metric}_mean"] = float(vals.mean())
        out[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return out


def run(args) -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "device": str(DEVICE),
        "seeds": args.seeds,
        "hpo_seed": args.hpo_seed,
        "n_samples": args.n_samples,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "functions": {},
    }
    for function_name in args.functions:
        result["functions"][function_name] = {}
        for model_name in args.models:
            print(f"\n=== HPO {function_name} / {model_name} ===", flush=True)
            trials = []
            for i, params in enumerate(param_grid(model_name, args.smoke), start=1):
                metrics = train_eval(function_name, model_name, params, args.hpo_seed,
                                     n_samples=args.n_samples, max_epochs=args.max_epochs, patience=args.patience)
                trials.append(metrics)
                print(f"trial {i:02d}: cfg={params} val={metrics['best_val_scaled_mse']:.5g} test_rmse={metrics['RMSE']:.5g}", flush=True)
            trials.sort(key=lambda r: r["best_val_scaled_mse"])
            best_params = trials[0]["params"]
            print(f"Best {function_name}/{model_name}: {best_params}", flush=True)
            eval_runs = []
            for seed in args.seeds:
                r = train_eval(function_name, model_name, best_params, seed,
                               n_samples=args.n_samples, max_epochs=args.max_epochs, patience=args.patience)
                eval_runs.append(r)
                print(f"  seed {seed}: RMSE={r['RMSE']:.6f} R2={r['R2']:.6f}", flush=True)
            result["functions"][function_name][model_name] = {
                "best_params": best_params,
                "hpo_trials": trials,
                "eval_runs": eval_runs,
                "aggregate": aggregate(eval_runs),
                "old_best_cfnn_rmse": OLD_BEST_CFNN_RMSE.get(function_name),
                "tuned_mlp_rmse": TUNED_MLP_RMSE.get(function_name),
            }
            (OUT_DIR / "raw.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_outputs(result)
    return result


def write_outputs(result: Dict) -> None:
    summary = {}
    lines = ["# P0 CFNN-side small-budget sanity HPO", ""]
    lines.append(f"Device: `{result['device']}`; seeds: `{result['seeds']}`; n_samples={result['n_samples']}; max_epochs={result['max_epochs']}.")
    lines.append("")
    lines.append("This is a limited reviewer-facing sanity check, not a full CFNN-family HPO sweep.")
    lines.append("")
    for fname, models in result["functions"].items():
        lines.append(f"## {fname}")
        lines.append("")
        lines.append("| Model | params | RMSE mean±std | R² mean±std | best cfg | tuned MLP RMSE | old best CFNN RMSE |")
        lines.append("|---|---:|---:|---:|---|---:|---:|")
        summary[fname] = {}
        for m, payload in models.items():
            a = payload["aggregate"]
            summary[fname][m] = {"aggregate": a, "best_params": payload["best_params"]}
            lines.append(
                f"| {m} | {a['n_trainable_params_mean']:.0f} | "
                f"{a['RMSE_mean']:.6f} ± {a['RMSE_std']:.6f} | "
                f"{a['R2_mean']:.6f} ± {a['R2_std']:.6f} | "
                f"`{payload['best_params']}` | "
                f"{payload.get('tuned_mlp_rmse') or float('nan'):.6f} | "
                f"{payload.get('old_best_cfnn_rmse') or float('nan'):.6f} |"
            )
        lines.append("")
    lines.append("## Conservative use in paper")
    lines.append("")
    lines.append("Use this table to state that a small CFNN-side tuning check was run for the sensitive clean-fitting functions. If the tuned CFNN variants remain close to tuned MLP rather than decisively ahead, Table 1 should be framed as a fairness calibration, not as the main evidence for CFNN superiority.")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--functions", nargs="+", default=["Rational_Interaction", "Nested_Rational"], choices=list(FUNCTIONS))
    p.add_argument("--models", nargs="+", default=["Hybrid", "MoE"], choices=["Hybrid", "MoE"])
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS[:3])
    p.add_argument("--hpo-seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--max-epochs", type=int, default=700)
    p.add_argument("--patience", type=int, default=60)
    args = p.parse_args()
    if args.smoke:
        args.functions = args.functions[:1]
        args.models = args.models[:1]
        args.seeds = args.seeds[:1]
        args.n_samples = min(args.n_samples, 800)
        args.max_epochs = min(args.max_epochs, 40)
        args.patience = min(args.patience, 5)
    return args


if __name__ == "__main__":
    run(parse_args())
