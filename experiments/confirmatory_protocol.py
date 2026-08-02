"""Protocol primitives for confirmatory PARN experiments.

This module is intentionally independent from legacy runners. It centralizes
randomness, budget eligibility, validation-only model selection, and dataset
roles so the confirmatory evidence has one auditable implementation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeVar

import numpy as np
import torch

from budget_matching import Candidate, within_budget


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


T = TypeVar("T")


def seeded_build(seed: int, builder: Callable[[], T]) -> T:
    """Construct an object after seeding, independent of prior RNG use."""
    seed_everything(seed)
    return builder()


def eligible_candidates_by_budget(
    candidates: Sequence[Candidate],
    budgets: Iterable[int],
    tolerance: float = 0.05,
) -> dict[int, list[Candidate]]:
    """Return every candidate inside each realized-parameter window."""
    return {
        int(budget): sorted(
            (c for c in candidates if within_budget(c.actual_parameters, int(budget), tolerance)),
            key=lambda c: (c.actual_parameters, c.key),
        )
        for budget in budgets
    }


def select_candidate_by_validation(
    candidates: Sequence[Candidate],
    tuning_seeds: Sequence[int],
    evaluate: Callable[[Candidate, int], float],
) -> tuple[Candidate, list[dict]]:
    """Evaluate all candidates and select by median validation score.

    ``evaluate`` returns a loss, so lower is better. Ties are resolved by the
    stable candidate key, never by candidate enumeration order.
    """
    if not candidates:
        raise ValueError("at least one budget-eligible candidate is required")
    if not tuning_seeds:
        raise ValueError("at least one tuning seed is required")
    records = []
    scores = {}
    for candidate in sorted(candidates, key=lambda c: c.key):
        values = []
        for tuning_seed in tuning_seeds:
            value = float(evaluate(candidate, int(tuning_seed)))
            records.append({
                "candidate_key": candidate.key,
                "tuning_seed": int(tuning_seed),
                "validation_loss": value,
            })
            values.append(value)
        scores[candidate.key] = float(np.median(values))
    selected = min(candidates, key=lambda c: (scores[c.key], c.key))
    return selected, records


def select_search_configuration(
    candidates: Sequence[Candidate],
    optimizer_configs: Sequence[dict],
    tuning_seeds: Sequence[int],
    evaluate: Callable[[Candidate, dict, int], dict],
) -> tuple[dict, list[dict]]:
    """Select architecture and optimizer after evaluating the full product."""
    if not candidates or not optimizer_configs or not tuning_seeds:
        raise ValueError("candidates, optimizer configs, and tuning seeds are required")
    records = []
    grouped = {}
    for candidate in sorted(candidates, key=lambda c: c.key):
        for optimizer in sorted(optimizer_configs, key=lambda item: item["key"]):
            key = (candidate.key, optimizer["key"])
            grouped[key] = []
            for tuning_seed in tuning_seeds:
                result = dict(evaluate(candidate, optimizer, int(tuning_seed)))
                record = {
                    "candidate_key": candidate.key,
                    "optimizer_key": optimizer["key"],
                    "tuning_seed": int(tuning_seed),
                    **result,
                }
                records.append(record)
                grouped[key].append(record)
    winner_key = min(
        grouped,
        key=lambda key: (
            float(np.median([r["validation_nrmse"] for r in grouped[key]])),
            key,
        ),
    )
    candidate = next(c for c in candidates if c.key == winner_key[0])
    optimizer = next(o for o in optimizer_configs if o["key"] == winner_key[1])
    selected_epoch = int(round(np.median([r["best_epoch"] for r in grouped[winner_key]])))
    return {
        "candidate": candidate,
        "optimizer": optimizer,
        "selected_epoch": selected_epoch,
        "median_validation_nrmse": float(np.median([
            r["validation_nrmse"] for r in grouped[winner_key]
        ])),
    }, records


def rank_finalists_by_validation(records: Sequence[dict], limit: int) -> list[str]:
    """Rank screened architectures by median validation NRMSE."""
    grouped = {}
    for record in records:
        grouped.setdefault(record["candidate_key"], []).append(record["validation_nrmse"])
    ranked = sorted(
        grouped,
        key=lambda key: (
            float(np.median([
                value if value is not None and np.isfinite(value) else float("inf")
                for value in grouped[key]
            ])),
            key,
        ),
    )
    return ranked[:limit]


def _analytic_target(name: str, x: np.ndarray, data_seed: int) -> np.ndarray:
    rng = np.random.default_rng(data_seed + 7919)
    if name in {"pole_sharp", "pole_broad"}:
        center = rng.uniform(-0.2, 0.2)
        epsilon = 1e-3 if name == "pole_sharp" else 1e-2
        y = 1.0 / ((x[:, 0] - center) ** 2 + epsilon)
    elif name == "lorentzian_two":
        centers = np.sort(rng.uniform(-0.65, 0.65, size=2))
        widths = rng.uniform(0.035, 0.08, size=2)
        y = sum(1.0 / ((x[:, 0] - c) ** 2 + w ** 2) for c, w in zip(centers, widths))
    elif name == "cusp":
        center = rng.uniform(-0.25, 0.25)
        exponent = rng.uniform(0.25, 0.55)
        y = np.abs(x[:, 0] - center) ** exponent
    elif name == "compact_bump":
        center = rng.uniform(-0.3, 0.3)
        width = rng.uniform(0.22, 0.4)
        z = (x[:, 0] - center) / width
        y = np.where(np.abs(z) < 1.0, (1.0 - z * z) ** 3, 0.0)
    elif name == "asymmetric_peak":
        center = rng.uniform(-0.25, 0.25)
        width = rng.uniform(0.07, 0.16)
        z = x[:, 0] - center
        y = np.exp(-(z / width) ** 2) * (1.0 + 1.5 * np.tanh(5.0 * z))
    elif name == "manifold_bump":
        if x.shape[1] < 3:
            raise ValueError("manifold_bump requires three input dimensions")
        shift = rng.uniform(-0.2, 0.2, size=2)
        u = x[:, 0] + 0.45 * x[:, 1] ** 2 - shift[0]
        v = x[:, 2] - 0.35 * np.sin(3.0 * x[:, 0]) - shift[1]
        y = np.exp(-(u * u / 0.06 + v * v / 0.12))
    else:
        raise ValueError(f"unknown analytic target: {name}")
    return np.asarray(y, dtype=np.float32).reshape(-1, 1)


def make_analytic_bundle(
    task: str,
    data_seed: int,
    n_train: int = 256,
    n_validation: int = 128,
    n_test: int = 2048,
    noise: float = 0.03,
) -> DatasetBundle:
    """Generate paired random observations and an independent clean test set."""
    dimensions = 3 if task == "manifold_bump" else 1
    rng = np.random.default_rng(data_seed)
    x_train = rng.uniform(-1.0, 1.0, size=(n_train, dimensions)).astype(np.float32)
    x_validation = rng.uniform(-1.0, 1.0, size=(n_validation, dimensions)).astype(np.float32)
    if dimensions == 1:
        x_test = np.linspace(-1.0, 1.0, n_test, dtype=np.float32).reshape(-1, 1)
        test_role = "clean_dense_grid"
    else:
        x_test = rng.uniform(-1.0, 1.0, size=(n_test, dimensions)).astype(np.float32)
        test_role = "clean_independent_sample"
    y_train_clean = _analytic_target(task, x_train, data_seed)
    y_validation_clean = _analytic_target(task, x_validation, data_seed)
    y_test = _analytic_target(task, x_test, data_seed)
    scale = np.maximum(np.abs(y_train_clean), np.std(y_train_clean) + 1e-6)
    y_train = y_train_clean + noise * scale * rng.standard_normal(y_train_clean.shape)
    y_validation = y_validation_clean + noise * np.maximum(
        np.abs(y_validation_clean), np.std(y_train_clean) + 1e-6
    ) * rng.standard_normal(y_validation_clean.shape)
    return DatasetBundle(
        x_train=x_train,
        y_train=y_train.astype(np.float32),
        x_validation=x_validation,
        y_validation=y_validation.astype(np.float32),
        x_test=x_test,
        y_test=y_test,
        metadata={
            "task": task,
            "data_seed": int(data_seed),
            "dimensions": dimensions,
            "split_strategy": "independent_random_observations",
            "test_target": test_role,
            "noise": float(noise),
        },
    )


def make_nmr_bundle(
    molecule: str,
    n_observed: int,
    noise: float,
    data_seed: int,
    n_dense: int = 4000,
    validation_fraction: float = 0.25,
) -> DatasetBundle:
    """Build random observed-point splits with a clean full-spectrum test."""
    import nmr_data

    data = nmr_data.make_nmr_dataset(
        molecule, n_points=n_observed, noise=noise, seed=data_seed, n_dense=n_dense
    )
    rng = np.random.default_rng(data_seed + 104729)
    order = rng.permutation(n_observed)
    n_validation = max(1, int(round(n_observed * validation_fraction)))
    validation_idx = order[:n_validation]
    train_idx = order[n_validation:]
    return DatasetBundle(
        x_train=data["X"][train_idx],
        y_train=data["Y"][train_idx],
        x_validation=data["X"][validation_idx],
        y_validation=data["Y"][validation_idx],
        x_test=data["Xeval"],
        y_test=data["Yeval"],
        metadata={
            "task": f"nmr_{molecule}",
            "data_seed": int(data_seed),
            "split_strategy": "random_observed_points",
            "test_target": "clean_dense_grid",
            "n_observed": int(n_observed),
            "n_dense": int(n_dense),
            "noise": float(noise),
            "parameter_source": data["parameter_source"],
        },
    )


def make_energy_bundle(data_seed: int) -> DatasetBundle:
    """Create a repeated train/validation/test split for Energy Efficiency."""
    import pandas as pd
    from pathlib import Path

    frame = pd.read_csv(Path(__file__).with_name("energy_efficiency.csv"))
    feature_columns = [column for column in frame.columns if column.upper().startswith("X")][:8]
    target_column = next(column for column in frame.columns if column.upper() == "Y1")
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    y = frame[[target_column]].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(data_seed)
    order = rng.permutation(len(x))
    train_end = int(0.63 * len(x))
    validation_end = int(0.80 * len(x))
    train_idx, validation_idx, test_idx = (
        order[:train_end], order[train_end:validation_end], order[validation_end:]
    )
    return DatasetBundle(
        x_train=x[train_idx], y_train=y[train_idx],
        x_validation=x[validation_idx], y_validation=y[validation_idx],
        x_test=x[test_idx], y_test=y[test_idx],
        metadata={
            "task": "energy",
            "data_seed": int(data_seed),
            "split_strategy": "repeated_random_63_17_20",
            "test_target": "held_out_observations",
        },
    )
