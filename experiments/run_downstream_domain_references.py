"""Run validation-selected domain references on measured resonant responses."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from confirmatory_protocol import DatasetBundle
from downstream_benchmark import make_task_bundle
from downstream_domain_baselines import fit_fano_reference, fit_vector_reference
from downstream_metrics import compute_complex_metrics


ROOT = Path(__file__).resolve().parent


def _frequency_and_response(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.asarray(x, dtype=np.float64)[:, 0]
    response = np.asarray(y, dtype=np.float64)
    order = np.argsort(frequency)
    return frequency[order], response[order]


def _metrics(frequency: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> dict:
    order = np.argsort(frequency)
    frequency = np.asarray(frequency, dtype=np.float64)[order]
    truth = np.asarray(truth, dtype=np.float64)[order]
    prediction = np.asarray(prediction, dtype=np.float64)[order]
    resolution = float(np.median(np.diff(frequency)))
    return compute_complex_metrics(truth, prediction, frequency, resolution)


def _score(metrics: dict) -> float:
    value = metrics.get("complex_nrmse")
    return float(value) if value is not None and math.isfinite(float(value)) else float("inf")


def evaluate_vector_reference(
    task: str,
    data_seed: int,
    bundle: DatasetBundle,
    *,
    candidate_orders: tuple[int, ...] = (2, 4, 8, 16),
) -> dict:
    """Select vector-fit order on validation frequencies and evaluate test once."""
    train_frequency, train_response = _frequency_and_response(bundle.x_train, bundle.y_train)
    validation_frequency = np.asarray(bundle.x_validation, dtype=np.float64)[:, 0]
    test_frequency = np.asarray(bundle.x_test, dtype=np.float64)[:, 0]
    candidates = []
    for maximum_poles in candidate_orders:
        fitted = fit_vector_reference(
            train_frequency,
            train_response,
            max_poles=int(maximum_poles),
            evaluation_frequency=validation_frequency,
        )
        metrics = _metrics(validation_frequency, bundle.y_validation, fitted.prediction)
        candidates.append({
            "maximum_poles": int(maximum_poles),
            "status": fitted.status,
            "validation_metrics": metrics,
            "wall_seconds": fitted.wall_seconds,
            "metadata": fitted.metadata,
        })
    selected = min(candidates, key=lambda value: (_score(value["validation_metrics"]), value["maximum_poles"]))
    tested = fit_vector_reference(
        train_frequency,
        train_response,
        max_poles=int(selected["maximum_poles"]),
        evaluation_frequency=test_frequency,
    )
    test_metrics = _metrics(test_frequency, bundle.y_test, tested.prediction)
    status = "ok" if tested.status == "ok" and test_metrics.get("status") == "ok" else "numerical_failure"
    return {
        "task": task,
        "data_seed": int(data_seed),
        "method": "vector_fitting",
        "status": status,
        "selection_role": "validation_only",
        "selected_max_poles": int(selected["maximum_poles"]),
        "validation_candidates": candidates,
        "test_metrics": test_metrics,
        "test_fit_wall_seconds": tested.wall_seconds,
        "test_fit_metadata": tested.metadata,
        "dataset": bundle.metadata,
    }


def evaluate_fano_reference(task: str, data_seed: int, bundle: DatasetBundle) -> dict:
    """Fit the prespecified one-resonance Fano reference and evaluate held-out points."""
    train_frequency, train_response = _frequency_and_response(bundle.x_train, bundle.y_train)
    validation_frequency = np.asarray(bundle.x_validation, dtype=np.float64)[:, 0]
    test_frequency = np.asarray(bundle.x_test, dtype=np.float64)[:, 0]
    validated = fit_fano_reference(
        train_frequency, train_response, evaluation_frequency=validation_frequency
    )
    validation_metrics = _metrics(
        validation_frequency, bundle.y_validation, validated.prediction
    )
    tested = fit_fano_reference(
        train_frequency, train_response, evaluation_frequency=test_frequency
    )
    test_metrics = _metrics(test_frequency, bundle.y_test, tested.prediction)
    status = "ok" if tested.status == "ok" and test_metrics.get("status") == "ok" else "numerical_failure"
    return {
        "task": task,
        "data_seed": int(data_seed),
        "method": "single_resonance_fano_fit",
        "status": status,
        "selection_role": "prespecified_model_validation_reported",
        "validation_metrics": validation_metrics,
        "validation_fit_wall_seconds": validated.wall_seconds,
        "test_metrics": test_metrics,
        "test_fit_wall_seconds": tested.wall_seconds,
        "test_fit_metadata": tested.metadata,
        "dataset": bundle.metadata,
    }


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "downstream_config.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default="microwave_fano,microstrip_resonator,aluminium_frf",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    allowed = {"microwave_fano", "microstrip_resonator", "aluminium_frf"}
    if not set(tasks).issubset(allowed):
        raise ValueError(f"domain references only support {sorted(allowed)}")
    output = args.output_root / "domain_references" / "records.jsonl"
    existing = [] if not output.exists() else [
        json.loads(line) for line in output.read_text().splitlines() if line.strip()
    ]
    completed = {(row["task"], int(row["data_seed"]), row["method"]) for row in existing}
    started = time.perf_counter()
    for task in tasks:
        for data_seed in config["evaluation_data_seeds"]:
            bundle = make_task_bundle(
                task, int(data_seed), config, args.raw_root, include_test=True
            )
            evaluators = [evaluate_vector_reference]
            if task == "microwave_fano":
                evaluators.insert(0, evaluate_fano_reference)
            for evaluator in evaluators:
                method = (
                    "single_resonance_fano_fit"
                    if evaluator is evaluate_fano_reference else "vector_fitting"
                )
                if (task, int(data_seed), method) in completed:
                    continue
                _append_jsonl(output, evaluator(task, int(data_seed), bundle))
    manifest = {
        "command": __import__("sys").argv,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "record_count": sum(1 for line in output.read_text().splitlines() if line.strip()),
        "wall_seconds": time.perf_counter() - started,
    }
    manifest_path = args.output_root / "domain_references" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
