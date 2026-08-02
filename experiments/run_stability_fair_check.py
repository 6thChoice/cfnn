#!/usr/bin/env python3
"""Production-model stability comparison for nested CFNN and CFNN-Hybrid.

Runs parameter-matched and component-matched diagnostics with the exact model
classes used by the Pareto and NMR experiments.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
import cofrnet_variants as V
import eis_models as M
from param_matched_utils import count_trainable_params

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "experiment_refine" / "stability_production_results"
SEEDS = [42, 123, 456, 789, 1024]
MODELS = ["CFNN", "CFNN-Hybrid"]
DESIGNS = {
    "parameter_matched": {"CFNN": 6, "CFNN-Hybrid": 3},
    "component_matched": {"CFNN": 6, "CFNN-Hybrid": 6},
}
FIXED_CFG = {"lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0}
HPO_GRID = [
    {"lr": lr, "weight_decay": wd, "grad_clip": clip}
    for lr in [1e-2, 1e-3]
    for wd in [0.0, 1e-4]
    for clip in [1.0]
]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_fn(x: np.ndarray) -> np.ndarray:
    # Runge peak + high-frequency component: stresses optimization and curvature.
    y = 1.0 / (1.0 + 25.0 * x[:, 0] ** 2) + 0.15 * np.sin(12.0 * x[:, 0])
    return y.reshape(-1, 1).astype(np.float32)


def make_data(seed: int, n: int = 1200):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 1)).astype(np.float32)
    y = target_fn(X)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.35, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.857142857, random_state=seed)
    xs = StandardScaler().fit(X_train)
    ys = StandardScaler().fit(y_train)
    tensors = []
    for arr in [xs.transform(X_train), xs.transform(X_val), xs.transform(X_test), ys.transform(y_train), ys.transform(y_val), ys.transform(y_test)]:
        tensors.append(torch.tensor(arr, dtype=torch.float32, device=DEVICE))
    return tensors


def build_model(name: str, design: str = "component_matched"):
    size = DESIGNS[design][name]
    if name == "CFNN":
        return V.CFNet_Standard(1, 1, depth=size, poly_degree=3)
    if name == "CFNN-Hybrid":
        return M.CFNNHybridEIS(1, 1, n_units=size, degree=3)
    raise ValueError(name)


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            if not torch.all(torch.isfinite(g)):
                return float("nan")
            total += float(torch.sum(g * g).item())
    return total ** 0.5


def train_once(model_name: str, cfg: Dict[str, float], seed: int,
               design: str = "component_matched", epochs: int = 240):
    set_seed(seed)
    X_train, X_val, X_test, y_train, y_val, y_test = make_data(seed)
    model = build_model(model_name, design).to(DEVICE)
    params = count_trainable_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    grad_norms: List[float] = []
    loss_history: List[float] = []
    failed = False
    fail_reason = ""

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        if not torch.isfinite(loss):
            failed = True
            fail_reason = "nonfinite_loss"
            break
        loss.backward()
        gn_before_clip = grad_norm(model)
        grad_norms.append(gn_before_clip)
        if not math.isfinite(gn_before_clip):
            failed = True
            fail_reason = "nonfinite_grad"
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        loss_history.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val), y_val).item())
        if math.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())

    if not failed:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_loss = float(loss_fn(model(X_test), y_test).item()) if not failed else float("inf")

    arr = np.asarray([g for g in grad_norms if math.isfinite(g)], dtype=float)
    grad_stats = {
        "grad_median": float(np.median(arr)) if arr.size else float("nan"),
        "grad_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)) if arr.size else float("nan"),
        "grad_std": float(arr.std()) if arr.size else float("nan"),
        "grad_p95": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "grad_max": float(arr.max()) if arr.size else float("nan"),
    }
    return {
        "model": model_name,
        "seed": seed,
        "cfg": cfg,
        "design": design,
        "params": params,
        "failed": failed,
        "fail_reason": fail_reason,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "final_train_loss": loss_history[-1] if loss_history else float("inf"),
        "epochs_completed": len(loss_history),
        **grad_stats,
    }


def aggregate(runs: List[Dict]) -> Dict:
    ok = [r for r in runs if not r["failed"] and math.isfinite(r["test_loss"])]
    out = {"n_runs": len(runs), "n_fail": len(runs) - len(ok),
           "fail_rate": (len(runs) - len(ok)) / len(runs),
           "params": int(np.median([r["params"] for r in runs]))}
    for key in ["test_loss", "best_val_loss", "grad_median", "grad_iqr", "grad_std", "grad_p95", "grad_max", "epochs_completed"]:
        vals = np.asarray([r[key] for r in ok if math.isfinite(r[key])], dtype=float)
        out[f"{key}_mean"] = float(vals.mean()) if vals.size else float("nan")
        out[f"{key}_std"] = float(vals.std()) if vals.size else float("nan")
    return out


def run_fixed(design: str) -> Dict[str, List[Dict]]:
    return {m: [train_once(m, FIXED_CFG, s, design) for s in SEEDS] for m in MODELS}


def run_tuned(design: str) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    tuned_runs: Dict[str, List[Dict]] = {}
    best_cfgs: Dict[str, Dict] = {}
    for model in MODELS:
        cfg_scores = []
        # Tune on first two seeds, then report all five seeds for selected config.
        for cfg in HPO_GRID:
            probe = [train_once(model, cfg, s, design, epochs=160) for s in SEEDS[:2]]
            score = aggregate(probe)["best_val_loss_mean"]
            cfg_scores.append((score, cfg))
        cfg_scores.sort(key=lambda x: x[0])
        best_cfgs[model] = cfg_scores[0][1]
        tuned_runs[model] = [train_once(model, best_cfgs[model], s, design) for s in SEEDS]
    return tuned_runs, best_cfgs


def write_summary(result: Dict) -> None:
    lines = ["# Production-model stability comparison", ""]
    lines.append("Task: 1D Runge peak + high-frequency component; degree=3; 5 seeds.")
    for design, data in result["designs"].items():
        lines.extend(["", f"## {design.replace('_', ' ').title()}", ""])
        for condition in ["fixed", "tuned"]:
            lines.append(f"### {condition.title()}")
            if condition == "tuned":
                lines.append(f"Best configs: `{data['best_cfgs']}`")
            lines.append("| Model | params | fail rate | test loss | grad std | grad p95 |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for model in MODELS:
                a = data[condition]["aggregate"][model]
                lines.append(
                    f"| {model} | {a['params']} | {a['fail_rate']:.0%} | "
                    f"{a['test_loss_mean']:.4f} ± {a['test_loss_std']:.4f} | "
                    f"{a['grad_std_mean']:.3f} ± {a['grad_std_std']:.3f} | "
                    f"{a['grad_p95_mean']:.3f} ± {a['grad_p95_std']:.3f} |"
                )
            lines.append("")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def plot_result(result: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = result["designs"]["parameter_matched"]["tuned"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    metrics = [("test_loss", "Held-out loss"), ("grad_std", "Within-run gradient std.")]
    colors = {"CFNN": "#7c8da6", "CFNN-Hybrid": "#2f8f83"}
    for ax, (metric, ylabel) in zip(axes, metrics):
        for x, model in enumerate(MODELS):
            vals = [row[metric] for row in data["runs"][model]]
            ax.scatter(np.full(len(vals), x), vals, color=colors[model], s=28, zorder=3)
            ax.hlines(np.mean(vals), x - 0.18, x + 0.18, color="black", lw=1.2)
        ax.set_xticks(range(len(MODELS)), ["Nested CFNN", "CFNN-Hybrid"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Production-model diagnostic (parameter matched)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stability_production_paired.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "stability_production_paired.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}")
    result = {"model_source": "experiment_refine production classes",
              "seeds": SEEDS, "fixed_cfg": FIXED_CFG, "hpo_grid": HPO_GRID,
              "designs": {}}
    for design in DESIGNS:
        print(f"Running {design} fixed recipe...")
        fixed_runs = run_fixed(design)
        print(f"Running {design} tuned recipe...")
        tuned_runs, best_cfgs = run_tuned(design)
        result["designs"][design] = {
            "fixed": {"runs": fixed_runs,
                      "aggregate": {m: aggregate(fixed_runs[m]) for m in MODELS}},
            "tuned": {"runs": tuned_runs,
                      "aggregate": {m: aggregate(tuned_runs[m]) for m in MODELS}},
            "best_cfgs": best_cfgs,
        }
    (OUT_DIR / "stability_production_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    write_summary(result)
    plot_result(result)
    print((OUT_DIR / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
