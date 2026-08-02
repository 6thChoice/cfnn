"""Robustness utilities for Microwave Fano A3 experiments."""
from __future__ import annotations

from collections.abc import Iterable
import hashlib
import math

import numpy as np

from ai4science_fano_a2 import make_sparse_curve_split
from confirmatory_protocol import DatasetBundle
from downstream_protocol import MeasuredCurve


def _seed(*parts: object) -> int:
    encoded = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _condition_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "clean", "null"}:
        return None
    return float(value)


def enumerate_a3_jobs(
    *,
    curve_ids: Iterable[str],
    families: Iterable[str],
    observation_budgets: Iterable[int],
    split_seeds: Iterable[int],
    init_seeds: Iterable[int],
    noise_snr_dbs: Iterable[float | None],
    frequency_drift_ppms: Iterable[float],
    missing_fractions: Iterable[float],
    perturbation_seeds: Iterable[int],
) -> list[dict]:
    """Return deterministic A3 robustness jobs sorted by scientific condition."""
    return [
        {
            "curve_id": str(curve_id),
            "family": str(family),
            "observation_budget": int(observation_budget),
            "split_seed": int(split_seed),
            "init_seed": int(init_seed),
            "noise_snr_db": _condition_value(noise_snr_db),
            "frequency_drift_ppm": float(frequency_drift_ppm),
            "missing_fraction": float(missing_fraction),
            "perturbation_seed": int(perturbation_seed),
        }
        for curve_id in sorted({str(value) for value in curve_ids})
        for family in sorted({str(value) for value in families})
        for observation_budget in sorted({int(value) for value in observation_budgets})
        for split_seed in sorted({int(value) for value in split_seeds})
        for init_seed in sorted({int(value) for value in init_seeds})
        for noise_snr_db in sorted({_condition_value(value) for value in noise_snr_dbs}, key=lambda x: (x is not None, x if x is not None else math.inf))
        for frequency_drift_ppm in sorted({float(value) for value in frequency_drift_ppms})
        for missing_fraction in sorted({float(value) for value in missing_fractions})
        for perturbation_seed in sorted({int(value) for value in perturbation_seeds})
    ]


def a3_job_key(record: dict) -> tuple[str, str, int, int, int, float | None, float, float, int] | None:
    """Return the A3 resume identity, including robustness perturbation settings."""
    required = (
        "curve_id",
        "family",
        "observation_budget",
        "split_seed",
        "init_seed",
        "noise_snr_db",
        "frequency_drift_ppm",
        "missing_fraction",
        "perturbation_seed",
    )
    if any(field not in record for field in required):
        return None
    return (
        str(record["curve_id"]),
        str(record["family"]),
        int(record["observation_budget"]),
        int(record["split_seed"]),
        int(record["init_seed"]),
        _condition_value(record["noise_snr_db"]),
        float(record["frequency_drift_ppm"]),
        float(record["missing_fraction"]),
        int(record["perturbation_seed"]),
    )


def a3_completed_job_keys(
    records: Iterable[dict],
) -> set[tuple[str, str, int, int, int, float | None, float, float, int]]:
    """Collect completed A3 job identities from existing records."""
    keys = set()
    for record in records:
        key = a3_job_key(record)
        if key is not None:
            keys.add(key)
    return keys


def add_complex_gaussian_noise(
    response: np.ndarray,
    *,
    snr_db: float | None,
    seed: int,
) -> np.ndarray:
    """Add deterministic two-channel complex Gaussian noise at the requested SNR."""
    values = np.asarray(response)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("response must be a two-column real/imag array")
    if snr_db is None or math.isinf(float(snr_db)):
        return values.copy()

    signal_rms = math.sqrt(float(np.mean(np.sum(values.astype(np.float64) ** 2, axis=1))))
    if signal_rms <= 0.0:
        return values.copy()
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(size=values.shape)
    noise_rms = math.sqrt(float(np.mean(np.sum(noise ** 2, axis=1))))
    if noise_rms <= 0.0:
        return values.copy()
    target_noise_rms = signal_rms / (10.0 ** (float(snr_db) / 20.0))
    noisy = values.astype(np.float64) + noise * (target_noise_rms / noise_rms)
    return noisy.astype(values.dtype, copy=False)


def _apply_frequency_drift(
    x: np.ndarray,
    *,
    frequency_drift_ppm: float,
    seed: int,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    drift_ppm = float(frequency_drift_ppm)
    if drift_ppm == 0.0:
        return values.copy()
    rng = np.random.default_rng(int(seed))
    relative = rng.normal(loc=0.0, scale=abs(drift_ppm) * 1e-6, size=values.shape)
    return values * (1.0 + relative)


def _keep_after_missing(
    n_items: int,
    *,
    missing_fraction: float,
    seed: int,
) -> np.ndarray:
    missing = float(missing_fraction)
    if missing < 0.0 or missing >= 1.0:
        raise ValueError("missing_fraction must be in [0, 1)")
    keep_count = max(2, int(round(int(n_items) * (1.0 - missing))))
    keep_count = min(int(n_items), keep_count)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(n_items), size=keep_count, replace=False))


def make_robust_sparse_curve_split(
    curve: MeasuredCurve,
    *,
    observation_count: int,
    validation_count: int,
    split_seed: int,
    perturbation_seed: int,
    noise_snr_db: float | None = None,
    frequency_drift_ppm: float = 0.0,
    missing_fraction: float = 0.0,
) -> DatasetBundle:
    """Create an A3 split with noisy/drifted observed points and clean full-curve target."""
    base = make_sparse_curve_split(
        curve,
        observation_count=int(observation_count),
        validation_count=int(validation_count),
        split_seed=int(split_seed),
    )
    keep_indices = _keep_after_missing(
        len(base.x_train),
        missing_fraction=float(missing_fraction),
        seed=_seed("a3-missing", curve.curve_id, split_seed, perturbation_seed),
    )

    x_train = _apply_frequency_drift(
        base.x_train[keep_indices],
        frequency_drift_ppm=float(frequency_drift_ppm),
        seed=_seed("a3-train-frequency", curve.curve_id, split_seed, perturbation_seed),
    )
    x_validation = _apply_frequency_drift(
        base.x_validation,
        frequency_drift_ppm=float(frequency_drift_ppm),
        seed=_seed("a3-validation-frequency", curve.curve_id, split_seed, perturbation_seed),
    )
    y_train = add_complex_gaussian_noise(
        base.y_train[keep_indices],
        snr_db=noise_snr_db,
        seed=_seed("a3-train-noise", curve.curve_id, split_seed, perturbation_seed),
    )
    y_validation = add_complex_gaussian_noise(
        base.y_validation,
        snr_db=noise_snr_db,
        seed=_seed("a3-validation-noise", curve.curve_id, split_seed, perturbation_seed),
    )

    metadata = dict(base.metadata)
    train_point_ids = [metadata["train_point_ids"][int(index)] for index in keep_indices]
    metadata.update({
        "task": "microwave_fano_noise_mismatch_robustness",
        "split_strategy": "within_evaluation_curve_sparse_observation_noisy_drift_missing_complete_curve_target",
        "requested_observation_budget": int(observation_count),
        "actual_observation_count": int(len(keep_indices)),
        "noise_snr_db": _condition_value(noise_snr_db),
        "frequency_drift_ppm": float(frequency_drift_ppm),
        "missing_fraction": float(missing_fraction),
        "perturbation_seed": int(perturbation_seed),
        "train_point_ids": train_point_ids,
        "test_target": "clean_complete_measured_curve",
    })

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=base.x_test.copy(),
        y_test=base.y_test.copy(),
        metadata=metadata,
    )
