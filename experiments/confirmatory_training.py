"""Training and metric routines shared by confirmatory experiments."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import r2_score


@dataclass
class TrainingResult:
    model: torch.nn.Module
    status: str
    best_step: int
    optimizer_steps: int
    best_validation_nrmse: float
    curve: list[dict]
    wall_seconds: float
    peak_memory_bytes: int | None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nrmse_tensor(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = torch.std(target, dim=0, unbiased=False).clamp_min(1e-6)
    per_output = torch.sqrt(torch.mean((prediction - target) ** 2, dim=0)) / scale
    return torch.mean(per_output)


def train_with_validation(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
    *,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    min_steps: int,
    patience: int,
    checkpoints: Sequence[int] = (20, 80, 200, 500),
    gradient_clip: float = 5.0,
    x_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
) -> TrainingResult:
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_step = 0
    stale = 0
    curve = []
    checkpoint_set = set(int(step) for step in checkpoints)
    status = "ok"
    optimizer_steps = 0

    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        if not torch.isfinite(loss):
            status = "nonfinite_train_loss"
            break
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if not torch.isfinite(gradient_norm):
            status = "nonfinite_gradient"
            break
        optimizer.step()
        optimizer_steps = step
        model.eval()
        with torch.no_grad():
            validation_nrmse = float(_nrmse_tensor(model(x_validation), y_validation).item())
        if not math.isfinite(validation_nrmse):
            status = "nonfinite_validation_metric"
            break
        if step >= min_steps and validation_nrmse < best_validation - 1e-9:
            best_validation = validation_nrmse
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        elif step >= min_steps:
            stale += 1
        if step in checkpoint_set:
            _sync(device)
            with torch.no_grad():
                train_nrmse = float(_nrmse_tensor(model(x_train), y_train).item())
            point = {
                "step": step,
                "train_nrmse": train_nrmse,
                "validation_nrmse": validation_nrmse,
                "wall_seconds": time.perf_counter() - started,
            }
            if x_test is not None and y_test is not None:
                with torch.no_grad():
                    point["test_nrmse"] = float(_nrmse_tensor(model(x_test), y_test).item())
            curve.append(point)
        if step >= min_steps and stale >= patience:
            break

    if best_step == 0 and status == "ok":
        best_step = optimizer_steps
        best_validation = validation_nrmse
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    _sync(device)
    wall_seconds = time.perf_counter() - started
    if best_step not in {point["step"] for point in curve}:
        selected_point = {
            "step": best_step,
            "train_nrmse": None,
            "validation_nrmse": best_validation,
            "wall_seconds": wall_seconds,
            "label": "selected_checkpoint",
        }
        if x_test is not None and y_test is not None:
            with torch.no_grad():
                selected_point["test_nrmse"] = float(_nrmse_tensor(model(x_test), y_test).item())
        curve.append(selected_point)
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return TrainingResult(
        model=model,
        status=status,
        best_step=best_step,
        optimizer_steps=optimizer_steps,
        best_validation_nrmse=best_validation,
        curve=sorted(curve, key=lambda item: item["step"]),
        wall_seconds=wall_seconds,
        peak_memory_bytes=peak_memory,
    )


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape or not np.all(np.isfinite(prediction)):
        return {"nrmse": float("inf"), "r2": float("nan"), "max_abs_error": float("inf"),
                "status": "numerical_failure"}
    scale = np.maximum(np.std(truth, axis=0), 1e-12)
    nrmse = float(np.mean(np.sqrt(np.mean((prediction - truth) ** 2, axis=0)) / scale))
    r2 = float(r2_score(truth, prediction, multioutput="variance_weighted"))
    status = "ok" if r2 > 0.0 else "underfit"
    return {
        "nrmse": nrmse,
        "r2": r2,
        "max_abs_error": float(np.max(np.abs(prediction - truth))),
        "status": status,
    }
