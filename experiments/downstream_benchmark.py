"""Auditable experiment runner for the CFNN downstream validation study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)

torch.use_deterministic_algorithms(True, warn_only=True)
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from confirmatory_artifacts import build_run_manifest, freeze_budget_grid
from confirmatory_models import build_model, enumerate_candidates
from confirmatory_protocol import DatasetBundle, seeded_build
from confirmatory_training import compute_regression_metrics
from downstream_metrics import compute_complex_metrics
from downstream_protocol import (
    make_controlled_fano_bundle,
    make_conventional_bundle,
    make_group_transfer_bundle,
    make_measured_bundle,
)
from downstream_training import fit_task_model, prepare_bundle
from resonance_windows import WindowSpec


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        _json_safe(value), indent=2, sort_keys=True, allow_nan=False
    ))
    temporary.replace(path)


def _code_hashes(manifest: dict) -> dict[str, str]:
    return {
        Path(item["path"]).name: item["sha256"]
        for item in manifest.get("code_files", [])
    }


def ensure_protocol_manifest(path: Path, manifest: dict) -> None:
    """Create one protocol manifest or reject a mixed config/code run."""
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if (
            existing.get("config_sha256") != manifest.get("config_sha256")
            or _code_hashes(existing) != _code_hashes(manifest)
        ):
            raise ValueError("mixed protocol manifest: config or code hashes differ")
        return
    _write_json(path, manifest)


def _config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(
            _json_safe(value), sort_keys=True, allow_nan=False
        ) + "\n")
        handle.flush()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_inventory(
    config: dict,
    output_root: Path,
    *,
    tasks: list[str],
    families: list[str],
    budgets: list[int],
    maximum_candidates: int | None = None,
) -> dict:
    """Enumerate and freeze diverse realized-parameter candidates before training."""
    maximum = int(maximum_candidates or config["maximum_candidates_per_budget"])
    inventory = {
        "protocol_version": config["protocol_version"],
        "config_sha256": _config_hash(config),
        "tasks": {},
    }
    cache = {}
    for task in tasks:
        task_config = config["tasks"][task]
        input_dim = int(task_config["input_dim"])
        output_dim = int(task_config["output_dim"])
        inventory["tasks"][task] = {}
        for family in families:
            key = (family, input_dim, output_dim)
            if key not in cache:
                cache[key] = enumerate_candidates(family, input_dim, output_dim)
            inventory["tasks"][task][family] = freeze_budget_grid(
                cache[key],
                budgets,
                tolerance=float(config["budget_tolerance"]),
                minimum_per_budget=int(config["minimum_candidates_per_budget"]),
                maximum_per_budget=maximum,
            )
    _write_json(Path(output_root) / "candidate_manifest.json", inventory)
    return inventory


def load_confirmatory_tasks(config: dict, selection_path: Path) -> list[str]:
    """Return the frozen six-task set, refusing access before selection exists."""
    selection_path = Path(selection_path)
    if not selection_path.exists():
        raise ValueError(f"missing frozen conventional selection: {selection_path}")
    selection = json.loads(selection_path.read_text())
    selected = selection.get("selected_tasks")
    expected_count = int(config["conventional_task_count"])
    candidates = set(config["conventional_candidates"])
    if not isinstance(selected, list) or len(selected) != expected_count:
        raise ValueError("frozen conventional selection must contain exactly two tasks")
    if len(set(selected)) != expected_count or not set(selected).issubset(candidates):
        raise ValueError("frozen conventional selection contains invalid tasks")
    return list(config["fixed_scientific_tasks"]) + selected


def _source_manifest_hash(raw_root: Path) -> str | None:
    path = Path(raw_root) / "source_manifest.json"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _compact_metadata(metadata: dict) -> dict:
    compact = {}
    for key, value in metadata.items():
        if (key.endswith("_point_ids") or key.endswith("_row_ids")) and isinstance(value, list):
            encoded = json.dumps(value, separators=(",", ":")).encode()
            compact[f"{key}_count"] = len(value)
            compact[f"{key}_sha256"] = hashlib.sha256(encoded).hexdigest()
        elif isinstance(value, dict):
            compact[key] = _compact_metadata(value)
        else:
            compact[key] = value
    return compact


def make_task_bundle(
    task: str,
    data_seed: int,
    config: dict,
    raw_root: Path,
    *,
    include_test: bool,
) -> DatasetBundle:
    task_config = config["tasks"][task]
    if task == "controlled_fano":
        bundle = make_controlled_fano_bundle(data_seed, task_config)
    elif task in {"microwave_fano", "microstrip_resonator", "aluminium_frf"}:
        bundle = make_measured_bundle(
            task, data_seed, task_config, raw_root,
            task_config.get("split_mode", "within_curve"),
        )
    elif task.endswith("_group_transfer"):
        bundle = make_group_transfer_bundle(
            task, data_seed, task_config, raw_root,
        )
    elif task in config["conventional_candidates"]:
        return make_conventional_bundle(task, data_seed, raw_root, include_test=include_test)
    else:
        raise ValueError(f"unknown downstream task: {task}")
    if include_test:
        return bundle
    return DatasetBundle(
        x_train=bundle.x_train,
        y_train=bundle.y_train,
        x_validation=bundle.x_validation,
        y_validation=bundle.y_validation,
        x_test=np.empty((0, bundle.x_train.shape[1]), dtype=np.float32),
        y_test=np.empty((0, bundle.y_train.shape[1]), dtype=np.float32),
        metadata={**bundle.metadata, "test_access": "sealed_for_validation_only_stage"},
    )


def _classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    truth = np.asarray(y_true).reshape(-1).astype(int)
    probability = np.asarray(probability).reshape(-1)
    if len(np.unique(truth)) < 2 or not np.all(np.isfinite(probability)):
        return {
            "auroc": None,
            "balanced_accuracy": None,
            "accuracy": None,
            "brier": None,
            "status": "numerical_failure",
        }
    predicted = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(truth, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "brier": float(brier_score_loss(truth, probability)),
        "status": "ok",
    }


def _test_metrics(task: str, bundle: DatasetBundle, prediction: np.ndarray, config: dict) -> dict:
    problem_type = config["tasks"][task]["problem_type"]
    if problem_type == "complex_regression":
        frequency = np.asarray(bundle.x_test[:, 0], dtype=np.float64)
        resolution = float(np.median(np.diff(np.sort(frequency))))
        return compute_complex_metrics(bundle.y_test, prediction, frequency, resolution)
    if problem_type == "classification":
        return _classification_metrics(bundle.y_test, prediction)
    metrics = compute_regression_metrics(bundle.y_test, prediction)
    metrics["mae"] = float(mean_absolute_error(bundle.y_test, prediction))
    return metrics


def train_one(
    task: str,
    family: str,
    candidate: dict,
    optimizer: dict,
    *,
    data_seed: int,
    init_seed: int,
    config: dict,
    raw_root: Path,
    evaluate_test: bool,
    validation_scorer=None,
    global_validation_scorer=None,
) -> dict:
    """Train one frozen configuration and optionally open its held-out test role."""
    started = time.perf_counter()
    full_bundle = make_task_bundle(
        task, data_seed, config, raw_root, include_test=evaluate_test
    )
    task_config = config["tasks"][task]
    model, model_metadata = seeded_build(
        init_seed,
        lambda: build_model(
            family,
            int(task_config["input_dim"]),
            int(task_config["output_dim"]),
            candidate["config"],
            seed=init_seed,
        ),
    )
    model = model.to(DEVICE)
    prepared = prepare_bundle(full_bundle, task_config["problem_type"], DEVICE)
    settings = config["training"]
    trained = fit_task_model(
        model,
        prepared,
        task_config["problem_type"],
        learning_rate=optimizer["lr"],
        weight_decay=optimizer["weight_decay"],
        max_steps=settings["max_steps"],
        min_steps=settings["min_steps"],
        patience=settings["patience"],
        checkpoints=settings["checkpoints"],
        gradient_clip=settings["gradient_clip"],
        batch_size=task_config.get("batch_size"),
        batch_seed=int(data_seed) + 49979687,
        validation_interval=settings.get("validation_interval", 1),
        validation_scorer=validation_scorer,
        global_validation_scorer=global_validation_scorer,
    )
    record = {
        "task": task,
        "task_role": task_config["role"],
        "problem_type": task_config["problem_type"],
        "family": family,
        "data_seed": int(data_seed),
        "init_seed": int(init_seed),
        "candidate_key": candidate["candidate_key"],
        "config": candidate["config"],
        "actual_parameters": int(candidate["actual_parameters"]),
        "optimizer": optimizer,
        "best_step": trained.best_step,
        "optimizer_steps": trained.optimizer_steps,
        "validation_score": trained.best_validation_score,
        "best_global_validation_score": trained.best_global_validation_score,
        "status": trained.status,
        "curve": trained.curve,
        "training_wall_seconds": trained.wall_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_memory_bytes": trained.peak_memory_bytes,
        "training_batch_size": int(min(
            task_config.get("batch_size") or len(prepared.x_train),
            len(prepared.x_train),
        )),
        "batch_seed": int(data_seed) + 49979687,
        "validation_interval": int(settings.get("validation_interval", 1)),
        "validation_evaluations": trained.validation_evaluations,
        "gradient_clip_events": trained.gradient_clip_events,
        "gradient_clip_rate": trained.gradient_clip_rate,
        "device": str(DEVICE),
        "dataset": _compact_metadata(full_bundle.metadata),
        "model_metadata": model_metadata,
        "source_manifest_sha256": _source_manifest_hash(raw_root),
        "stage_role": "confirmatory_test" if evaluate_test else "validation_only",
    }
    if evaluate_test:
        metrics = _test_metrics(task, full_bundle, trained.prediction, config)
        record["test_metrics"] = metrics
        primary_metric_name = {
            "complex_regression": "complex_nrmse",
            "classification": "auroc",
            "regression": "nrmse",
        }[task_config["problem_type"]]
        record["primary_metric_name"] = primary_metric_name
        record["primary_metric_value"] = metrics.get(primary_metric_name)
        if trained.status == "ok" and metrics.get("status") != "ok":
            record["status"] = metrics["status"]
    return record


def _screening_path(output_root: Path, task: str, family: str, budget: int) -> Path:
    return Path(output_root) / "tuning_screen" / _slug(task) / f"{_slug(family)}__{budget}.jsonl"


def _tuning_path(output_root: Path, task: str, family: str, budget: int) -> Path:
    return Path(output_root) / "tuning_final" / _slug(task) / f"{_slug(family)}__{budget}.jsonl"


def _selection_path(output_root: Path, task: str, family: str, budget: int) -> Path:
    return Path(output_root) / "selections" / _slug(task) / f"{_slug(family)}__{budget}.json"


def _rank_candidates(records: list[dict], limit: int) -> list[str]:
    grouped = {}
    for record in records:
        grouped.setdefault(record["candidate_key"], []).append(_score_or_inf(record))
    return sorted(
        grouped,
        key=lambda key: (
            sum(not math.isfinite(score) for score in grouped[key]),
            float(np.median(grouped[key])),
            key,
        ),
    )[: int(limit)]


def _score_or_inf(record: dict) -> float:
    score = record.get("validation_score")
    if record.get("status") != "ok" or score is None:
        return float("inf")
    score = float(score)
    return score if math.isfinite(score) else float("inf")


def _tensor_validation_values(value, *, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim == 1:
        tensor = tensor.reshape(-1)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    return tensor.detach().to(dtype=torch.float64)


def _trapezoid_weights(axis: np.ndarray) -> np.ndarray:
    if len(axis) == 1:
        return np.ones(1, dtype=np.float64)
    differences = np.diff(axis)
    if np.any(differences <= 0.0):
        raise ValueError("validation frequency must be strictly increasing")
    weights = np.empty(len(axis), dtype=np.float64)
    weights[0] = 0.5 * differences[0]
    weights[-1] = 0.5 * differences[-1]
    weights[1:-1] = 0.5 * (differences[:-1] + differences[1:])
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("validation frequency has no positive measure")
    return weights / total


def _frequency_weights_in_original_order(axis: np.ndarray) -> np.ndarray:
    """Compute frequency quadrature weights without changing prediction row order."""
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_axis = values[order]
    if len(sorted_axis) > 1 and np.any(np.diff(sorted_axis) <= 0.0):
        raise ValueError("validation frequency must be strictly increasing")
    sorted_weights = _trapezoid_weights(sorted_axis)
    weights = np.empty(len(values), dtype=np.float64)
    weights[order] = sorted_weights
    return weights


def build_v2_validation_scorers(
    window_spec: WindowSpec,
    validation_frequency,
    *,
    log_weighted: bool = False,
    output_mean=None,
    output_std=None,
):
    """Build tensor scorers from one frozen validation window specification.

    Both callbacks receive the model's raw ``(n, 2)`` tensor output and the
    prepared target tensor. Lower scores are better. A no-feature instance
    uses the global score for checkpointing; a required but empty window is
    explicitly assigned infinity so ranking cannot treat it as a valid peak
    result.
    """
    if not isinstance(window_spec, WindowSpec):
        raise ValueError("window_spec must be a WindowSpec")
    frequencies = _tensor_validation_values(validation_frequency, name="validation_frequency")
    axis = frequencies.numpy()
    if window_spec.axis_space == "log10_frequency":
        if np.any(axis <= 0.0):
            raise ValueError("log10-frequency validation requires positive frequencies")
        axis = np.log10(axis)
    window_mask = np.zeros(len(axis), dtype=bool)
    for left, right in window_spec.intervals:
        window_mask |= (axis >= float(left)) & (axis <= float(right))
    if log_weighted:
        global_weights = _frequency_weights_in_original_order(axis)
    else:
        global_weights = np.full(len(axis), 1.0 / len(axis), dtype=np.float64)
    global_weights_tensor = torch.from_numpy(global_weights)
    window_indices = np.flatnonzero(window_mask)
    if log_weighted and len(window_indices):
        local_weights = _frequency_weights_in_original_order(axis[window_mask])
    elif len(window_indices):
        local_weights = np.full(len(window_indices), 1.0 / len(window_indices), dtype=np.float64)
    else:
        local_weights = np.empty(0, dtype=np.float64)
    local_weights_tensor = torch.from_numpy(local_weights)

    def restore(values: torch.Tensor) -> torch.Tensor:
        if output_mean is None and output_std is None:
            return values
        if output_mean is None or output_std is None:
            raise ValueError("output_mean and output_std must be supplied together")
        mean = torch.as_tensor(output_mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(output_std, dtype=values.dtype, device=values.device)
        return values * std + mean

    def nrmse(prediction: torch.Tensor, target: torch.Tensor, indices, weights) -> float:
        if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
            raise ValueError("complex validation scorers require matching two-dimensional tensors")
        if prediction.shape[1] != 2:
            raise ValueError("complex validation scorers require real and imaginary columns")
        prediction = restore(prediction)
        target = restore(target)
        if indices is not None:
            prediction = prediction[indices]
            target = target[indices]
        if prediction.shape[0] == 0:
            return float("inf")
        weights = weights.to(device=prediction.device, dtype=prediction.dtype)
        weights = weights / weights.sum()
        residual = torch.sum((prediction - target) ** 2, dim=1)
        centered_target = target - torch.sum(weights[:, None] * target, dim=0)
        denominator = torch.sqrt(torch.sum(weights * torch.sum(centered_target ** 2, dim=1)))
        if denominator <= torch.finfo(prediction.dtype).eps:
            denominator = torch.sqrt(torch.sum(weights * torch.sum(target ** 2, dim=1)))
        if denominator <= torch.finfo(prediction.dtype).eps:
            denominator = torch.tensor(torch.finfo(prediction.dtype).eps, device=prediction.device)
        value = torch.sqrt(torch.sum(weights * residual)) / denominator
        return float(value.detach().item()) if torch.isfinite(value) else float("inf")

    def global_scorer(prediction: torch.Tensor, target: torch.Tensor) -> float:
        return nrmse(prediction, target, None, global_weights_tensor)

    def resonance_scorer(prediction: torch.Tensor, target: torch.Tensor) -> float:
        if window_spec.status == "no_identifiable_feature":
            return global_scorer(prediction, target)
        if not len(window_indices):
            return float("inf")
        indices = torch.as_tensor(window_indices, dtype=torch.long, device=prediction.device)
        return nrmse(prediction, target, indices, local_weights_tensor)

    return resonance_scorer, global_scorer


make_v2_validation_scorers = build_v2_validation_scorers


def v2_candidate_rank_key(records: list[dict]) -> tuple:
    """Return the frozen v2 candidate ranking tuple; all scores are minimized."""
    if not records:
        raise ValueError("candidate ranking requires at least one record")

    def finite_value(record: dict, *names: str) -> float:
        for name in names:
            value = record.get(name)
            if value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    return value
        return float("inf")

    failure_count = sum(record.get("status") != "ok" for record in records)
    failure_count += sum(
        record.get("status") == "ok"
        and not math.isfinite(finite_value(record, "validation_global_nrmse", "best_global_validation_score"))
        for record in records
    )
    def window_status(record: dict) -> str:
        explicit = record.get("window_status")
        if explicit is not None:
            return str(explicit)
        window = record.get("window_spec")
        if isinstance(window, dict):
            return str(window.get("status", "ok"))
        return "ok"

    missing_window_count = sum(
        window_status(record) in {"empty_window", "missing_validation_window"}
        for record in records
    )
    resonance_values = [
        finite_value(record, "validation_resonance_nrmse")
        for record in records
        if record.get("status") == "ok" and window_status(record) == "ok"
    ]
    global_values = [
        finite_value(record, "validation_global_nrmse", "best_global_validation_score")
        for record in records
        if record.get("status") == "ok"
    ]
    candidate_key = str(records[0]["candidate_key"])
    return (
        int(failure_count),
        int(missing_window_count),
        float(np.median(resonance_values)) if resonance_values else float("inf"),
        float(np.median(global_values)) if global_values else float("inf"),
        candidate_key,
    )


def rank_v2_candidates(records: list[dict], limit: int) -> list[str]:
    """Rank v2 records by failure, window coverage, resonance, and global score."""
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["candidate_key"]), []).append(record)
    return [
        key for key, _ in sorted(
            grouped.items(), key=lambda item: v2_candidate_rank_key(item[1])
        )[: int(limit)]
    ]


_rank_candidates_v2 = rank_v2_candidates


def tune_shard(
    config: dict,
    inventory: dict,
    output_root: Path,
    task: str,
    family: str,
    budget: int,
    raw_root: Path,
    *,
    stage: str,
) -> dict:
    """Screen every frozen candidate, then tune validation-selected finalists."""
    candidates = inventory["tasks"][task][family][str(int(budget))]
    selection_path = _selection_path(output_root, task, family, budget)
    if not candidates:
        selection = {
            "task": task,
            "family": family,
            "target_budget": int(budget),
            "status": "missing_budget_match",
            "stage": stage,
        }
        _write_json(selection_path, selection)
        return selection
    optimizers = config["optimizer_grids"][family]
    screening_optimizer = optimizers[min(1, len(optimizers) - 1)]
    screening_config = deepcopy(config)
    screening_steps = min(
        int(config["training"]["screening_steps"]),
        int(config["training"]["max_steps"]),
    )
    screening_config["training"].update({
        "max_steps": screening_steps,
        "min_steps": min(screening_steps, int(config["training"]["min_steps"])),
        "patience": screening_steps,
        "checkpoints": sorted(set(
            [step for step in config["training"]["checkpoints"] if step <= screening_steps]
            + [screening_steps]
        )),
    })
    screen_path = _screening_path(output_root, task, family, budget)
    screen_records = _load_jsonl(screen_path)
    completed = {
        (record["candidate_key"], int(record["tuning_seed"]))
        for record in screen_records
    }
    for candidate in candidates:
        for tuning_seed in config["tuning_seeds"]:
            key = (candidate["candidate_key"], int(tuning_seed))
            if key in completed:
                continue
            record = train_one(
                task,
                family,
                candidate,
                screening_optimizer,
                data_seed=int(tuning_seed),
                init_seed=int(tuning_seed) + 1_000_003,
                config=screening_config,
                raw_root=raw_root,
                evaluate_test=False,
            )
            record.update({
                "stage": f"{stage}_screen",
                "target_budget": int(budget),
                "tuning_seed": int(tuning_seed),
            })
            _append_jsonl(screen_path, record)
    screen_records = _load_jsonl(screen_path)
    expected_screen = len(candidates) * len(config["tuning_seeds"])
    if len(screen_records) != expected_screen:
        raise ValueError(
            f"incomplete screening for {task}/{family}/{budget}: "
            f"expected {expected_screen}, found {len(screen_records)}"
        )
    finalist_keys = set(_rank_candidates(
        screen_records,
        min(int(config["finalists_per_budget"]), len(candidates)),
    ))
    finalists = [
        candidate for candidate in candidates
        if candidate["candidate_key"] in finalist_keys
    ]
    tuning_path = _tuning_path(output_root, task, family, budget)
    tuning_records = _load_jsonl(tuning_path)
    completed_tuning = {
        (record["candidate_key"], record["optimizer"]["key"], int(record["tuning_seed"]))
        for record in tuning_records
    }
    for candidate in finalists:
        for optimizer in optimizers:
            for tuning_seed in config["tuning_seeds"]:
                key = (candidate["candidate_key"], optimizer["key"], int(tuning_seed))
                if key in completed_tuning:
                    continue
                record = train_one(
                    task,
                    family,
                    candidate,
                    optimizer,
                    data_seed=int(tuning_seed),
                    init_seed=int(tuning_seed) + 1_000_003,
                    config=config,
                    raw_root=raw_root,
                    evaluate_test=False,
                )
                record.update({
                    "stage": stage,
                    "target_budget": int(budget),
                    "tuning_seed": int(tuning_seed),
                })
                _append_jsonl(tuning_path, record)
    tuning_records = _load_jsonl(tuning_path)
    grouped = {}
    for record in tuning_records:
        grouped.setdefault(
            (record["candidate_key"], record["optimizer"]["key"]), []
        ).append(record)
    expected_seeds = len(config["tuning_seeds"])
    complete = {
        key: values for key, values in grouped.items()
        if len(values) == expected_seeds
    }
    expected_groups = len(finalists) * len(optimizers)
    if len(complete) != expected_groups:
        raise ValueError(
            f"incomplete final tuning for {task}/{family}/{budget}: "
            f"expected {expected_groups} groups, found {len(complete)}"
        )
    winner_key = min(
        complete,
        key=lambda key: (
            sum(not math.isfinite(_score_or_inf(record)) for record in complete[key]),
            float(np.median([_score_or_inf(record) for record in complete[key]])),
            key,
        ),
    )
    winner_records = complete[winner_key]
    winner_candidate = next(
        candidate for candidate in candidates
        if candidate["candidate_key"] == winner_key[0]
    )
    winner_optimizer = next(
        optimizer for optimizer in optimizers if optimizer["key"] == winner_key[1]
    )
    selection = {
        "task": task,
        "family": family,
        "target_budget": int(budget),
        "status": "selected",
        "stage": stage,
        "candidate": winner_candidate,
        "optimizer": winner_optimizer,
        "selected_step": int(round(np.median([
            record["best_step"] for record in winner_records
        ]))),
        "median_validation_score": float(np.median([
            _score_or_inf(record) for record in winner_records
        ])),
        "screened_candidate_count": len(candidates),
        "finalist_count": len(finalists),
        "optimizer_count": len(optimizers),
        "tuning_seed_count": expected_seeds,
    }
    _write_json(selection_path, selection)
    return selection


def evaluate_shard(
    config: dict,
    output_root: Path,
    task: str,
    family: str,
    budget: int,
    raw_root: Path,
) -> None:
    """Evaluate a validation-selected configuration on each frozen seed pair once."""
    selection_path = _selection_path(output_root, task, family, budget)
    if not selection_path.exists():
        raise ValueError(f"missing tuning selection: {selection_path}")
    selection = json.loads(selection_path.read_text())
    if selection["status"] != "selected":
        return
    output_path = (
        Path(output_root) / "evaluation" / _slug(task)
        / f"{_slug(family)}__{int(budget)}.jsonl"
    )
    existing = _load_jsonl(output_path)
    completed = {
        (int(record["data_seed"]), int(record["init_seed"]))
        for record in existing
    }
    pairs = [
        (int(data_seed), int(init_seed))
        for data_seed in config["evaluation_data_seeds"]
        for init_seed in config["evaluation_init_seeds"]
    ]
    for data_seed, init_seed in pairs:
        if (data_seed, init_seed) in completed:
            continue
        record = train_one(
            task,
            family,
            selection["candidate"],
            selection["optimizer"],
            data_seed=data_seed,
            init_seed=init_seed,
            config=config,
            raw_root=raw_root,
            evaluate_test=True,
        )
        record.update({
            "stage": "confirmatory_evaluation",
            "target_budget": int(budget),
            "selection_path": str(selection_path),
        })
        _append_jsonl(output_path, record)


def audit_tuning(
    config: dict,
    inventory: dict,
    output_root: Path,
    tasks: list[str],
    families: list[str],
    budgets: list[int],
) -> None:
    errors = []
    for task in tasks:
        for family in families:
            for budget in budgets:
                candidates = inventory["tasks"][task][family][str(int(budget))]
                selection_path = _selection_path(output_root, task, family, budget)
                if not selection_path.exists():
                    errors.append(f"missing selection {task}/{family}/{budget}")
                    continue
                selection = json.loads(selection_path.read_text())
                expected_status = "selected" if candidates else "missing_budget_match"
                if selection.get("status") != expected_status:
                    errors.append(f"wrong selection status {task}/{family}/{budget}")
                    continue
                if not candidates:
                    continue
                screen = _load_jsonl(_screening_path(output_root, task, family, budget))
                expected_screen = len(candidates) * len(config["tuning_seeds"])
                if len(screen) != expected_screen:
                    errors.append(
                        f"incomplete screen {task}/{family}/{budget}: "
                        f"{len(screen)}/{expected_screen}"
                    )
                keys = [(row["candidate_key"], row["tuning_seed"]) for row in screen]
                if len(keys) != len(set(keys)):
                    errors.append(f"duplicate screen keys {task}/{family}/{budget}")
    if errors:
        raise ValueError("tuning audit failed:\n" + "\n".join(errors[:30]))


def _filter(available: list, requested: str | None, cast=str) -> list:
    if not requested:
        return list(available)
    wanted = [cast(item.strip()) for item in requested.split(",") if item.strip()]
    unknown = set(wanted) - set(available)
    if unknown:
        raise ValueError(f"unknown requested values: {sorted(unknown)}")
    return wanted


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _resolve_tasks(config: dict, args) -> list[str]:
    all_tasks = list(config["fixed_scientific_tasks"]) + list(config["conventional_candidates"])
    if args.tasks:
        return _filter(all_tasks, args.tasks)
    if args.stage == "explore":
        return list(config["conventional_candidates"])
    return load_confirmatory_tasks(config, args.selection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "downstream_config.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("inventory", "explore", "tune", "audit-tuning", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--tasks")
    parser.add_argument("--families")
    parser.add_argument("--budgets")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--maximum-candidates", type=int)
    parser.add_argument("--maximum-optimizers", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--min-steps", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--tuning-seed-count", type=int)
    parser.add_argument("--evaluation-data-seed-count", type=int)
    parser.add_argument("--evaluation-init-seed-count", type=int)
    args = parser.parse_args()
    config = _read_json(args.config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.selection is None:
        args.selection = args.output_root / "conventional_selection.json"
    if args.smoke:
        config = deepcopy(config)
        config["tuning_seeds"] = config["tuning_seeds"][:1]
        config["evaluation_data_seeds"] = config["evaluation_data_seeds"][:1]
        config["evaluation_init_seeds"] = config["evaluation_init_seeds"][:1]
        config["finalists_per_budget"] = 1
        config["training"].update({
            "screening_steps": 20,
            "max_steps": 20,
            "min_steps": 5,
            "patience": 10,
            "checkpoints": [5, 20],
        })
        for family in config["families"]:
            config["optimizer_grids"][family] = config["optimizer_grids"][family][:1]
    for key, value in (
        ("max_steps", args.max_steps),
        ("min_steps", args.min_steps),
        ("patience", args.patience),
    ):
        if value is not None:
            config["training"][key] = int(value)
    if args.maximum_optimizers is not None:
        for family in config["families"]:
            config["optimizer_grids"][family] = config["optimizer_grids"][family][
                :args.maximum_optimizers
            ]
    if args.tuning_seed_count is not None:
        config["tuning_seeds"] = config["tuning_seeds"][:args.tuning_seed_count]
    if args.evaluation_data_seed_count is not None:
        config["evaluation_data_seeds"] = config["evaluation_data_seeds"][
            :args.evaluation_data_seed_count
        ]
    if args.evaluation_init_seed_count is not None:
        config["evaluation_init_seeds"] = config["evaluation_init_seeds"][
            :args.evaluation_init_seed_count
        ]
    tasks = _resolve_tasks(config, args)
    families = _filter(config["families"], args.families)
    budgets = _filter(config["budgets"], args.budgets, int)
    code_paths = [
        Path(__file__), ROOT / "downstream_protocol.py", ROOT / "downstream_metrics.py",
        ROOT / "downstream_training.py", ROOT / "confirmatory_models.py",
        ROOT / "fair_baselines.py", args.config,
    ]
    execution_manifest = build_run_manifest(config, sys.argv, code_paths)
    ensure_protocol_manifest(args.output_root / "run_manifest.json", execution_manifest)
    _write_json(
        args.output_root / "execution_manifests"
        / f"{_slug(socket.gethostname())}_{os.getpid()}.json",
        execution_manifest,
    )
    inventory_path = args.output_root / "candidate_manifest.json"
    if args.stage == "inventory" or not inventory_path.exists():
        inventory = build_inventory(
            config,
            args.output_root,
            tasks=tasks,
            families=families,
            budgets=budgets,
            maximum_candidates=(
                3 if args.smoke else args.maximum_candidates
            ),
        )
    else:
        inventory = _read_json(inventory_path)
        for task in tasks:
            if task not in inventory["tasks"]:
                raise ValueError(f"candidate manifest lacks requested task: {task}")
    if args.stage == "inventory":
        return
    if args.stage == "audit-tuning":
        audit_tuning(config, inventory, args.output_root, tasks, families, budgets)
        print("tuning audit passed")
        return
    tuning_stage = "exploration" if args.stage == "explore" else "confirmatory_tuning"
    for task in tasks:
        for family in families:
            for budget in budgets:
                if args.stage in {"explore", "tune", "all"}:
                    tune_shard(
                        config, inventory, args.output_root, task, family, budget,
                        args.raw_root, stage=tuning_stage,
                    )
                if args.stage in {"evaluate", "all"}:
                    evaluate_shard(
                        config, args.output_root, task, family, budget, args.raw_root
                    )


if __name__ == "__main__":
    main()
