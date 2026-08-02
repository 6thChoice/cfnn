"""Utilities for Microwave Fano sparse-observation parameter recovery."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import numpy as np

from ai4science_fano import FanoReference
from confirmatory_protocol import DatasetBundle
from downstream_protocol import MeasuredCurve


def enumerate_a2_jobs(
    *,
    curve_ids: Iterable[str],
    families: Iterable[str],
    observation_budgets: Iterable[int],
    split_seeds: Iterable[int],
    init_seeds: Iterable[int],
) -> list[dict]:
    """Return deterministic A2 run jobs sorted by curve, family, budget, split, init."""
    return [
        {
            "curve_id": str(curve_id),
            "family": str(family),
            "observation_budget": int(observation_budget),
            "split_seed": int(split_seed),
            "init_seed": int(init_seed),
        }
        for curve_id in sorted({str(value) for value in curve_ids})
        for family in sorted({str(value) for value in families})
        for observation_budget in sorted({int(value) for value in observation_budgets})
        for split_seed in sorted({int(value) for value in split_seeds})
        for init_seed in sorted({int(value) for value in init_seeds})
    ]


def order_a2_jobs(jobs: Iterable[dict], *, order: str = "deterministic") -> list[dict]:
    """Order A2 jobs for execution without changing their resume identity."""
    items = [dict(job) for job in jobs]
    if order == "deterministic":
        return sorted(
            items,
            key=lambda job: (
                str(job["curve_id"]),
                str(job["family"]),
                int(job["observation_budget"]),
                int(job["split_seed"]),
                int(job["init_seed"]),
            ),
        )
    if order == "balanced":
        return sorted(
            items,
            key=lambda job: (
                int(job["observation_budget"]),
                int(job["split_seed"]),
                int(job["init_seed"]),
                str(job["curve_id"]),
                str(job["family"]),
            ),
        )
    if order == "coverage":
        return sorted(
            items,
            key=lambda job: (
                int(job["split_seed"]),
                int(job["init_seed"]),
                str(job["curve_id"]),
                str(job["family"]),
                int(job["observation_budget"]),
            ),
        )
    if order == "science_priority":
        budget_priority = {64: 0, 128: 1, 256: 2, 32: 3, 16: 4}
        return sorted(
            items,
            key=lambda job: (
                budget_priority.get(int(job["observation_budget"]), 100 + int(job["observation_budget"])),
                int(job["init_seed"]),
                str(job["curve_id"]),
                str(job["family"]),
                int(job["split_seed"]),
            ),
        )
    raise ValueError(f"unknown A2 job order: {order}")


def job_key(record: dict) -> tuple[str, str, int, int, int] | None:
    """Return the resume identity for an A2 record, or None if incomplete."""
    required = ("curve_id", "family", "observation_budget", "split_seed", "init_seed")
    if any(field not in record for field in required):
        return None
    return (
        str(record["curve_id"]),
        str(record["family"]),
        int(record["observation_budget"]),
        int(record["split_seed"]),
        int(record["init_seed"]),
    )


def completed_job_keys(records: Iterable[dict]) -> set[tuple[str, str, int, int, int]]:
    """Collect completed A2 job identities from existing records."""
    keys = set()
    for record in records:
        key = job_key(record)
        if key is not None:
            keys.add(key)
    return keys


def _as_complex(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1 and np.iscomplexobj(array):
        response = np.asarray(array, dtype=np.complex128)
    elif array.ndim == 2 and array.shape[1] == 2:
        response = np.asarray(array[:, 0], dtype=np.float64) + 1j * np.asarray(
            array[:, 1], dtype=np.float64
        )
    else:
        raise ValueError(f"{name} must be a complex vector or two-column real array")
    if len(response) == 0 or not np.all(np.isfinite(response)):
        raise ValueError(f"{name} must contain finite values")
    return response


def complex_nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Complex RMSE normalized by the RMS magnitude of the true response."""
    truth_complex = _as_complex(truth, "truth")
    prediction_complex = _as_complex(prediction, "prediction")
    if truth_complex.shape != prediction_complex.shape:
        raise ValueError("truth and prediction shapes differ")
    denominator = max(float(np.sqrt(np.mean(np.abs(truth_complex) ** 2))), 1e-12)
    numerator = float(np.sqrt(np.mean(np.abs(prediction_complex - truth_complex) ** 2)))
    return float(numerator / denominator)


def peak_window_complex_nrmse(
    frequency_hz: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    f0_hz: float | None,
    linewidth_hz: float | None,
    window_linewidths: float = 2.0,
) -> float | None:
    """Complex NRMSE inside the reference resonance window."""
    if f0_hz is None or linewidth_hz is None:
        return None
    if not math.isfinite(float(f0_hz)) or not math.isfinite(float(linewidth_hz)):
        return None
    half_width = abs(float(linewidth_hz)) * float(window_linewidths)
    if half_width <= 0.0:
        return None
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    mask = np.abs(frequency - float(f0_hz)) <= half_width
    if not np.any(mask):
        return None
    return complex_nrmse(np.asarray(truth)[mask], np.asarray(prediction)[mask])


def _point_id(curve_id: str, index: int) -> str:
    return f"{curve_id}:frequency:{int(index)}"


def _seed_for_curve(curve_id: str, split_seed: int, role: str) -> int:
    encoded = f"ai4science-fano-a2:{int(split_seed)}:{role}:{curve_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def make_sparse_curve_split(
    curve: MeasuredCurve,
    *,
    observation_count: int,
    validation_count: int,
    split_seed: int,
) -> DatasetBundle:
    """Create one sparse-observation reconstruction split for a complete curve."""
    frequency = np.asarray(curve.frequency, dtype=np.float64).reshape(-1)
    response = np.asarray(curve.response, dtype=np.float32)
    observation_count = int(observation_count)
    validation_count = int(validation_count)
    if frequency.ndim != 1 or response.shape != (len(frequency), 2):
        raise ValueError("curve frequency/response shape mismatch")
    if observation_count < 2 or validation_count < 1:
        raise ValueError("observation_count and validation_count must be positive")
    if observation_count + validation_count >= len(frequency):
        raise ValueError("sparse split leaves no held-out frequencies")

    rng = np.random.default_rng(_seed_for_curve(curve.curve_id, split_seed, "train-validation"))
    order = rng.permutation(len(frequency))
    train_indices = np.sort(order[:observation_count])
    validation_indices = np.sort(order[observation_count:observation_count + validation_count])

    return DatasetBundle(
        x_train=frequency[train_indices].reshape(-1, 1),
        y_train=response[train_indices],
        x_validation=frequency[validation_indices].reshape(-1, 1),
        y_validation=response[validation_indices],
        x_test=frequency.reshape(-1, 1),
        y_test=response,
        metadata={
            "task": "microwave_fano_parameter_recovery",
            "base_task": "microwave_fano",
            "curve_id": curve.curve_id,
            "conditions": dict(curve.conditions),
            "curve_metadata": dict(curve.metadata),
            "source_files": list(curve.source_files),
            "split_strategy": "within_evaluation_curve_sparse_observation_complete_curve_target",
            "split_seed": int(split_seed),
            "observation_budget": int(observation_count),
            "validation_count": int(validation_count),
            "train_point_ids": [_point_id(curve.curve_id, index) for index in train_indices],
            "validation_point_ids": [_point_id(curve.curve_id, index) for index in validation_indices],
            "test_point_ids": [_point_id(curve.curve_id, index) for index in range(len(frequency))],
            "test_target": "complete_measured_curve",
        },
    )


def _finite_abs_error(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        return None
    return float(abs(float(a) - float(b)))


def _finite_relative_error(reference: float | None, predicted: float | None) -> float | None:
    absolute = _finite_abs_error(reference, predicted)
    if absolute is None or reference is None:
        return None
    denominator = abs(float(reference))
    if denominator <= np.finfo(float).eps:
        return None
    return float(absolute / denominator)


def compare_fano_parameters(reference: FanoReference, predicted: FanoReference) -> dict:
    """Compare predicted Fano/event parameters against an A1 reference label."""
    if reference.status != "ok":
        status = "reference_unidentifiable"
    elif predicted.status != "ok":
        status = "prediction_unidentifiable"
    else:
        status = "ok"

    return {
        "curve_id": reference.curve_id,
        "reference_status": reference.status,
        "prediction_status": predicted.status,
        "parameter_recovery_status": status,
        "reference_failure_reason": reference.failure_reason,
        "prediction_failure_reason": predicted.failure_reason,
        "f0_abs_error_hz": (
            _finite_abs_error(reference.f0_hz, predicted.f0_hz) if status == "ok" else None
        ),
        "linewidth_relative_error": (
            _finite_relative_error(reference.linewidth_hz, predicted.linewidth_hz)
            if status == "ok" else None
        ),
        "quality_factor_relative_error": (
            _finite_relative_error(reference.quality_factor, predicted.quality_factor)
            if status == "ok" else None
        ),
        "q_abs_error": (
            _finite_abs_error(reference.q, predicted.q) if status == "ok" else None
        ),
        "magnitude_peak_abs_error_hz": (
            _finite_abs_error(reference.magnitude_peak_hz, predicted.magnitude_peak_hz)
            if status == "ok" else None
        ),
        "magnitude_valley_abs_error_hz": (
            _finite_abs_error(reference.magnitude_valley_hz, predicted.magnitude_valley_hz)
            if status == "ok" else None
        ),
        "phase_transition_abs_error_hz": (
            _finite_abs_error(reference.phase_transition_hz, predicted.phase_transition_hz)
            if status == "ok" else None
        ),
        "prediction_relative_magnitude_rmse": predicted.relative_magnitude_rmse,
    }
