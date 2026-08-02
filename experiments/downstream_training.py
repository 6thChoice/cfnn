"""Task-aware training with train-only preprocessing and validation selection."""
from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from confirmatory_protocol import DatasetBundle


@dataclass(frozen=True)
class PreparedBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_validation: torch.Tensor
    y_validation: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


@dataclass
class TaskTrainingResult:
    model: torch.nn.Module
    status: str
    best_step: int
    optimizer_steps: int
    best_validation_score: float
    best_global_validation_score: float
    gradient_clip_events: int
    gradient_clip_rate: float
    curve: list[dict]
    wall_seconds: float
    peak_memory_bytes: int | None
    prediction: np.ndarray
    validation_evaluations: int


ValidationScorer = Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]


def prepare_bundle(
    bundle: DatasetBundle,
    problem_type: str,
    device: torch.device,
) -> PreparedBundle:
    """Fit every transformation on training rows and transform all roles."""
    # Preserve small frequency increments on large absolute coordinates until
    # train-only centering and scaling have removed the large offset.
    x_train = np.asarray(bundle.x_train, dtype=np.float64)
    x_validation = np.asarray(bundle.x_validation, dtype=np.float64)
    x_test = np.asarray(bundle.x_test, dtype=np.float64)
    y_train = np.asarray(bundle.y_train, dtype=np.float32)
    y_validation = np.asarray(bundle.y_validation, dtype=np.float32)
    y_test = np.asarray(bundle.y_test, dtype=np.float32)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.maximum(x_std, 1e-6)
    if problem_type == "classification":
        y_mean = np.zeros((1, y_train.shape[1]), dtype=np.float32)
        y_std = np.ones((1, y_train.shape[1]), dtype=np.float32)
    elif problem_type in {"regression", "complex_regression"}:
        y_mean = y_train.mean(axis=0, keepdims=True)
        y_std = np.maximum(y_train.std(axis=0, keepdims=True), 1e-6)
    else:
        raise ValueError(f"unknown problem type: {problem_type}")

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)

    return PreparedBundle(
        x_train=tensor((x_train - x_mean) / x_std),
        y_train=tensor((y_train - y_mean) / y_std),
        x_validation=tensor((x_validation - x_mean) / x_std),
        y_validation=tensor((y_validation - y_mean) / y_std),
        x_test=tensor((x_test - x_mean) / x_std),
        y_test=tensor((y_test - y_mean) / y_std),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )


def _regression_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    scale = torch.std(target, dim=0, unbiased=False).clamp_min(1e-6)
    per_output = torch.sqrt(torch.mean((prediction - target) ** 2, dim=0)) / scale
    return float(torch.mean(per_output).item())


def _classification_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    truth = target.detach().cpu().numpy().reshape(-1)
    probability = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    if not np.all(np.isfinite(probability)):
        return float("inf")
    if len(np.unique(truth)) < 2:
        return float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target).item())
    return float(1.0 - roc_auc_score(truth, probability))


def fit_task_model(
    model: torch.nn.Module,
    prepared: PreparedBundle,
    problem_type: str,
    *,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    min_steps: int,
    patience: int,
    checkpoints: Sequence[int],
    gradient_clip: float,
    batch_size: int | None = None,
    batch_seed: int = 0,
    validation_interval: int = 1,
    validation_scorer: ValidationScorer | None = None,
    global_validation_scorer: ValidationScorer | None = None,
) -> TaskTrainingResult:
    """Train one model and select its checkpoint with validation data only."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    loss_function = (
        torch.nn.BCEWithLogitsLoss() if problem_type == "classification"
        else torch.nn.MSELoss()
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_global_score = float("inf")
    best_step = 0
    stale = 0
    status = "ok"
    curve = []
    optimizer_steps = 0
    checkpoint_set = {int(step) for step in checkpoints}
    validation_score = float("inf")
    global_validation_score = float("inf")
    train_count = len(prepared.x_train)
    effective_batch_size = (
        train_count if batch_size is None else min(int(batch_size), train_count)
    )
    if effective_batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(int(batch_seed))
    batch_order = None
    batch_cursor = train_count
    validation_interval = int(validation_interval)
    if validation_interval < 1:
        raise ValueError("validation_interval must be positive")
    validation_evaluations = 0
    gradient_clip_events = 0
    default_scorer = (
        _classification_score if problem_type == "classification" else _regression_score
    )
    primary_scorer = validation_scorer or default_scorer
    global_scorer = global_validation_scorer or default_scorer

    def score(scorer: ValidationScorer, prediction: torch.Tensor, target: torch.Tensor) -> float:
        value = scorer(prediction, target)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("validation scorers must return a scalar")
            value = value.detach().item()
        return float(value)

    for step in range(1, int(max_steps) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if effective_batch_size == train_count:
            x_batch = prepared.x_train
            y_batch = prepared.y_train
        else:
            if batch_cursor + effective_batch_size > train_count:
                batch_order = torch.randperm(
                    train_count, generator=batch_generator, device="cpu"
                ).to(device)
                batch_cursor = 0
            indices = batch_order[batch_cursor:batch_cursor + effective_batch_size]
            batch_cursor += effective_batch_size
            x_batch = prepared.x_train[indices]
            y_batch = prepared.y_train[indices]
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        if not torch.isfinite(loss):
            status = "nonfinite_train_loss"
            break
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        if float(gradient_norm) > float(gradient_clip):
            gradient_clip_events += 1
        if not torch.isfinite(gradient_norm):
            status = "nonfinite_gradient"
            break
        optimizer.step()
        optimizer_steps = step
        should_validate = step % validation_interval == 0 or step == int(max_steps)
        if should_validate:
            model.eval()
            with torch.no_grad():
                validation_output = model(prepared.x_validation)
                validation_score = score(
                    primary_scorer, validation_output, prepared.y_validation
                )
                global_validation_score = score(
                    global_scorer, validation_output, prepared.y_validation
                )
            validation_evaluations += 1
            if not math.isfinite(validation_score):
                status = "nonfinite_validation_metric"
                break
            if step >= int(min_steps) and validation_score < best_score - 1e-9:
                best_score = validation_score
                best_global_score = global_validation_score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            elif step >= int(min_steps):
                stale += validation_interval
            if step in checkpoint_set:
                curve.append({
                    "step": step,
                    "train_loss": float(loss.item()),
                    "validation_score": float(validation_score),
                    "global_validation_score": float(global_validation_score),
                    "wall_seconds": time.perf_counter() - started,
                })
            if step >= int(min_steps) and stale >= int(patience):
                break

    if best_step == 0 and status == "ok":
        best_step = optimizer_steps
        best_score = validation_score
        best_global_score = global_validation_score
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if len(prepared.x_test):
            raw_prediction = model(prepared.x_test).detach().cpu().numpy()
        else:
            raw_prediction = np.empty(
                (0, prepared.y_train.shape[1]), dtype=np.float32
            )
    if problem_type == "classification":
        prediction = 1.0 / (1.0 + np.exp(-raw_prediction))
    else:
        prediction = raw_prediction * prepared.y_std + prepared.y_mean
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return TaskTrainingResult(
        model=model,
        status=status,
        best_step=int(best_step),
        optimizer_steps=int(optimizer_steps),
        best_validation_score=float(best_score),
        best_global_validation_score=float(best_global_score),
        gradient_clip_events=int(gradient_clip_events),
        gradient_clip_rate=(
            float(gradient_clip_events / optimizer_steps) if optimizer_steps else 0.0
        ),
        curve=curve,
        wall_seconds=wall_seconds,
        peak_memory_bytes=peak_memory,
        prediction=np.asarray(prediction, dtype=np.float64),
        validation_evaluations=int(validation_evaluations),
    )
