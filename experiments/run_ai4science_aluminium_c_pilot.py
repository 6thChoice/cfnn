#!/usr/bin/env python3
"""Run Package C Aluminium FRF active modal-analysis pilot experiments."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ai4science_aluminium_c import (  # noqa: E402
    aluminium_response_metrics,
    build_active_schedule,
    complex_modal_fit_response,
    extract_modal_events,
    global_complex_modal_prior_fit,
    half_power_recovery_metrics,
    interpolate_response,
    inverse_signed_log_complex,
    load_aluminium_frf_full,
    low_order_global_complex_modal_prior_fit,
    modal_peak_fit_response,
    signed_log_complex,
)
from confirmatory_models import build_model  # noqa: E402
from confirmatory_protocol import DatasetBundle, seeded_build  # noqa: E402
from downstream_training import fit_task_model, prepare_bundle  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_OUT = ROOT / "ai4science_results" / "aluminium_c_active_modal_pilot"
DEFAULT_SELECTION_ROOT = (
    ROOT / "downstream_group_transfer_results" / "selections"
    / "aluminium_frf_group_transfer"
)
FAMILIES = ("CFNN", "MLP", "Local nested CF control")
INTERPOLATION_BASELINES = (
    "Complex linear interpolation",
    "Signed-log linear interpolation",
    "Modal peak fit interpolation",
    "Complex modal fit interpolation",
    "Global complex modal prior",
    "Low-order global modal prior",
)
FAMILY_SLUGS = {
    "CFNN": "cfnn",
    "MLP": "mlp",
    "Local nested CF control": "local_nested_cf_control",
}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _load_selection(selection_root: Path, family: str) -> dict:
    path = Path(selection_root) / f"{FAMILY_SLUGS[family]}__256.json"
    if not path.exists():
        raise FileNotFoundError(path)
    selection = json.loads(path.read_text())
    if selection.get("status") != "selected":
        raise ValueError(f"selection is not usable for {family}: {path}")
    return selection


def _parse_ints(values: str) -> list[int]:
    return [int(item.strip()) for item in values.split(",") if item.strip()]


def _parse_strings(values: str) -> list[str]:
    return [item.strip() for item in values.split(",") if item.strip()]


def _select_families(
    *,
    include_interpolation_baselines: bool,
    only_interpolation_baselines: bool = False,
) -> list[str]:
    if only_interpolation_baselines:
        return list(INTERPOLATION_BASELINES)
    families = list(FAMILIES)
    if include_interpolation_baselines:
        families.extend(INTERPOLATION_BASELINES)
    return families


def _jobs(
    curve_points: list[int],
    strategies: list[str],
    budgets: list[int],
    split_seeds: list[int],
    init_seeds: list[int],
    families: list[str],
) -> list[dict]:
    return [
        {
            "curve_id": f"point:{point:02d}",
            "point": int(point),
            "sampling_strategy": strategy,
            "budget": int(budget),
            "split_seed": int(split_seed),
            "init_seed": None if family in INTERPOLATION_BASELINES else int(init_seed),
            "family": family,
        }
        for point in curve_points
        for strategy in strategies
        for budget in budgets
        for family in families
        for split_seed in split_seeds
        for init_seed in ([None] if family in INTERPOLATION_BASELINES else init_seeds)
    ]


def _job_key(record: dict) -> tuple:
    return (
        str(record["curve_id"]),
        str(record["sampling_strategy"]),
        int(record["budget"]),
        str(record["family"]),
        int(record["split_seed"]),
        None if record.get("init_seed") is None else int(record["init_seed"]),
    )


def _point_id(curve_id: str, index: int) -> str:
    return f"{curve_id}:frequency:{int(index)}"


def _make_observed_bundle(curve, observed_indices: np.ndarray, seed: int) -> DatasetBundle:
    observed = np.sort(np.asarray(observed_indices, dtype=np.int64))
    if len(observed) < 8:
        raise ValueError("at least eight observed points are required for train/validation split")
    rng = np.random.default_rng(int(seed) + 24281291)
    validation_count = max(4, min(len(observed) // 5, len(observed) - 4))
    validation_positions = np.sort(rng.choice(len(observed), size=validation_count, replace=False))
    validation_indices = observed[validation_positions]
    train_indices = np.delete(observed, validation_positions)
    all_indices = np.arange(len(curve.frequency), dtype=np.int64)
    return DatasetBundle(
        x_train=curve.frequency[train_indices].reshape(-1, 1).astype(np.float64),
        y_train=curve.response[train_indices].astype(np.float32),
        x_validation=curve.frequency[validation_indices].reshape(-1, 1).astype(np.float64),
        y_validation=curve.response[validation_indices].astype(np.float32),
        x_test=curve.frequency[all_indices].reshape(-1, 1).astype(np.float64),
        y_test=curve.response[all_indices].astype(np.float32),
        metadata={
            "task": "aluminium_frf_c_active_modal_analysis",
            "base_task": "aluminium_frf",
            "split_mode": "same_curve_active_frequency_acquisition",
            "curve_id": curve.curve_id,
            "conditions": curve.conditions,
            "curve_metadata": curve.metadata,
            "observed_frequency_count": int(len(observed)),
            "train_frequency_count": int(len(train_indices)),
            "validation_frequency_count": int(len(validation_indices)),
            "observed_point_ids": [_point_id(curve.curve_id, index) for index in observed],
            "train_point_ids": [_point_id(curve.curve_id, index) for index in train_indices],
            "validation_point_ids": [_point_id(curve.curve_id, index) for index in validation_indices],
            "test_point_ids": [_point_id(curve.curve_id, index) for index in all_indices],
            "test_target": "complete_full_grid_h1_frf_reference",
            "validation_note": "validation points are part of the acquired measurement budget and are withheld only for checkpoint selection",
        },
    )


def _transform_bundle_targets(bundle: DatasetBundle, target_transform: str) -> DatasetBundle:
    if target_transform == "complex":
        metadata = {**bundle.metadata, "target_transform": "complex"}
        return DatasetBundle(
            x_train=bundle.x_train,
            y_train=bundle.y_train,
            x_validation=bundle.x_validation,
            y_validation=bundle.y_validation,
            x_test=bundle.x_test,
            y_test=bundle.y_test,
            metadata=metadata,
        )
    if target_transform != "signed_log_complex":
        raise ValueError(f"unknown target transform: {target_transform}")
    return DatasetBundle(
        x_train=bundle.x_train,
        y_train=signed_log_complex(bundle.y_train),
        x_validation=bundle.x_validation,
        y_validation=signed_log_complex(bundle.y_validation),
        x_test=bundle.x_test,
        y_test=signed_log_complex(bundle.y_test),
        metadata={
            **bundle.metadata,
            "target_transform": "signed_log_complex",
            "target_transform_note": (
                "training and checkpoint selection use signed log1p real/imaginary columns; "
                "reported metrics invert predictions back to physical complex FRF"
            ),
        },
    )


def _inverse_prediction(prediction: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "complex":
        return np.asarray(prediction, dtype=np.float32)
    if target_transform == "signed_log_complex":
        return inverse_signed_log_complex(prediction)
    raise ValueError(f"unknown target transform: {target_transform}")


def _curve_lookup(curves) -> dict[str, object]:
    return {curve.curve_id: curve for curve in curves}


def _catalogue_frequencies_for_curve(curve, *, max_modes: int) -> list[float]:
    events = extract_modal_events(
        curve.frequency,
        curve.response,
        max_modes=int(max_modes),
        min_frequency_hz=5.0,
    )
    return [float(mode["frequency_hz"]) for mode in events.get("modes", [])[: int(max_modes)]]


def _schedule_for_job(curve, job: dict, args) -> np.ndarray:
    catalogue_frequencies = _catalogue_frequencies_for_curve(
        curve,
        max_modes=int(getattr(args, "catalogue_max_modes", 5)),
    )
    schedule = build_active_schedule(
        len(curve.frequency),
        initial_count=int(args.initial_count),
        batch_size=int(args.batch_size),
        budgets=_parse_ints(args.budgets),
        strategy=str(job["sampling_strategy"]),
        seed=int(job["split_seed"]),
        frequency=curve.frequency,
        response=curve.response,
        catalogue_frequencies_hz=catalogue_frequencies,
        catalogue_band_half_width_hz=float(getattr(args, "catalogue_band_half_width", 16.0)),
        catalogue_min_band_points=int(getattr(args, "catalogue_min_band_points", 7)),
        catalogue_max_modes=int(getattr(args, "catalogue_max_modes", 5)),
    )
    return schedule[int(job["budget"])]


def _aluminium_metrics(curve, prediction: np.ndarray, args) -> dict:
    metrics = aluminium_response_metrics(
        curve.frequency,
        curve.response,
        prediction,
    )
    catalogue_frequencies = _catalogue_frequencies_for_curve(
        curve,
        max_modes=int(getattr(args, "catalogue_max_modes", 5)),
    )
    metrics["half_power"] = half_power_recovery_metrics(
        curve.frequency,
        curve.response,
        prediction,
        candidate_frequencies_hz=catalogue_frequencies,
        band_half_width_hz=float(getattr(args, "half_power_band_half_width", 8.0)),
        frequency_threshold_hz=float(getattr(args, "half_power_frequency_threshold", 1.0)),
        width_relative_threshold=float(getattr(args, "half_power_width_relative_threshold", 0.20)),
    )
    return metrics


def run_job(curves_by_id: dict[str, object], selections: dict[str, dict], job: dict, args) -> dict:
    started = time.perf_counter()
    curve = curves_by_id[job["curve_id"]]
    observed_indices = _schedule_for_job(curve, job, args)
    if job["family"] in INTERPOLATION_BASELINES:
        if job["family"] == "Complex linear interpolation":
            mode = "complex_linear"
            prediction = interpolate_response(
                curve.frequency,
                curve.response,
                observed_indices,
                mode=mode,
            )
        elif job["family"] == "Signed-log linear interpolation":
            mode = "signed_log_linear"
            prediction = interpolate_response(
                curve.frequency,
                curve.response,
                observed_indices,
                mode=mode,
            )
        elif job["family"] == "Modal peak fit interpolation":
            mode = "modal_peak_fit"
            prediction = modal_peak_fit_response(
                curve.frequency,
                curve.response,
                observed_indices,
                max_modes=5,
            )
        elif job["family"] == "Complex modal fit interpolation":
            mode = "complex_modal_fit"
            prediction = complex_modal_fit_response(
                curve.frequency,
                curve.response,
                observed_indices,
                max_modes=5,
            )
        elif job["family"] == "Global complex modal prior":
            mode = "global_complex_modal_prior"
            fit = global_complex_modal_prior_fit(
                curve.frequency,
                curve.response,
                observed_indices,
                max_modes=5,
            )
            prediction = fit["prediction"]
            model_diagnostics = fit["diagnostics"]
        elif job["family"] == "Low-order global modal prior":
            mode = "low_order_global_modal_prior"
            fit = low_order_global_complex_modal_prior_fit(
                curve.frequency,
                curve.response,
                observed_indices,
                max_modes=5,
            )
            prediction = fit["prediction"]
            model_diagnostics = fit["diagnostics"]
        else:
            raise ValueError(f"unknown interpolation baseline: {job['family']}")
        if job["family"] not in {"Global complex modal prior", "Low-order global modal prior"}:
            model_diagnostics = None
        return {
            **job,
            "status": "ok",
            "device": "none",
            "target_budget": None,
            "actual_parameters": 0,
            "candidate_key": mode,
            "candidate_config": {"mode": mode},
            "optimizer": None,
            "model_metadata": {
                "family": job["family"],
                "implementation": "deterministic interpolation or modal-prior baseline",
                "fit_diagnostics": model_diagnostics,
            },
            "dataset": {
                "task": "aluminium_frf_c_active_modal_analysis",
                "base_task": "aluminium_frf",
                "split_mode": "same_curve_active_frequency_acquisition",
                "curve_id": curve.curve_id,
                "conditions": curve.conditions,
                "curve_metadata": curve.metadata,
                "observed_frequency_count": int(len(observed_indices)),
                "target_transform": mode,
                "test_target": "complete_full_grid_h1_frf_reference",
            },
            "target_transform": mode,
            "observed_indices": [int(index) for index in observed_indices],
            "best_step": None,
            "optimizer_steps": 0,
            "validation_score": None,
            "training_wall_seconds": 0.0,
            "total_wall_seconds": time.perf_counter() - started,
            "curve": [],
            "metrics": _aluminium_metrics(curve, prediction, args),
        }
    physical_bundle = _make_observed_bundle(curve, observed_indices, int(job["split_seed"]))
    bundle = _transform_bundle_targets(physical_bundle, str(args.target_transform))
    family = job["family"]
    selection = selections[family]
    candidate = selection["candidate"]
    optimizer = selection["optimizer"]
    model, model_metadata = seeded_build(
        int(job["init_seed"]),
        lambda: build_model(
            family,
            input_dim=1,
            output_dim=2,
            config=candidate["config"],
            seed=int(job["init_seed"]),
        ),
    )
    model = model.to(DEVICE)
    prepared = prepare_bundle(bundle, "complex_regression", DEVICE)
    trained = fit_task_model(
        model,
        prepared,
        "complex_regression",
        learning_rate=optimizer["lr"],
        weight_decay=optimizer["weight_decay"],
        max_steps=int(args.max_steps),
        min_steps=min(int(args.min_steps), int(args.max_steps)),
        patience=int(args.patience),
        checkpoints=[step for step in (20, 80, 150, 300, 600) if step <= int(args.max_steps)],
        gradient_clip=5.0,
        batch_size=None,
        batch_seed=int(job["split_seed"]) + 49979687,
        validation_interval=10,
    )
    record = {
        **job,
        "status": trained.status,
        "device": str(DEVICE),
        "target_budget": 256,
        "actual_parameters": int(model_metadata.get("actual_parameters", candidate["actual_parameters"])),
        "candidate_key": candidate["candidate_key"],
        "candidate_config": candidate["config"],
        "optimizer": optimizer,
        "model_metadata": model_metadata,
        "dataset": bundle.metadata,
        "target_transform": str(args.target_transform),
        "observed_indices": [int(index) for index in observed_indices],
        "best_step": trained.best_step,
        "optimizer_steps": trained.optimizer_steps,
        "validation_score": trained.best_validation_score,
        "training_wall_seconds": trained.wall_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "curve": trained.curve,
        "metrics": {},
    }
    if trained.status == "ok":
        physical_prediction = _inverse_prediction(trained.prediction, str(args.target_transform))
        record["metrics"] = _aluminium_metrics(curve, physical_prediction, args)
    return record


def _median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _summarize(
    records: list[dict],
    *,
    frequency_threshold_hz: float = 1.0,
    width_relative_threshold: float = 0.10,
) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[(record["family"], record["sampling_strategy"], int(record["budget"]))].append(record)
    rows = []
    for key, items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        ok_metrics = [
            record.get("metrics", {})
            for record in items
            if record.get("status") == "ok"
            and record.get("metrics", {})
            and record.get("metrics", {}).get("status", "ok") == "ok"
        ]
        rows.append({
            "family": key[0],
            "sampling_strategy": key[1],
            "budget": key[2],
            "record_count": len(items),
            "ok_count": sum(record.get("status") == "ok" for record in items),
            "metric_count": len(ok_metrics),
            "median_complex_nrmse": _median([metric.get("complex_nrmse") for metric in ok_metrics]),
            "median_peak_window_complex_nrmse": _median([metric.get("peak_window_complex_nrmse") for metric in ok_metrics]),
            "median_modal_frequency_mae_hz": _median([metric.get("modal_frequency_mae_hz") for metric in ok_metrics]),
            "median_modal_width_relative_mae": _median([metric.get("modal_width_relative_mae") for metric in ok_metrics]),
            "median_modal_peak_amplitude_relative_mae": _median([metric.get("modal_peak_amplitude_relative_mae") for metric in ok_metrics]),
            "median_matched_mode_count": _median([metric.get("matched_mode_count") for metric in ok_metrics]),
            "median_truth_mode_count": _median([metric.get("truth_mode_count") for metric in ok_metrics]),
            "median_half_power_frequency_error_hz": _median([
                metric.get("half_power", {}).get("median_frequency_error_hz") for metric in ok_metrics
            ]),
            "median_half_power_width_relative_error": _median([
                metric.get("half_power", {}).get("median_width_relative_error") for metric in ok_metrics
            ]),
            "median_half_power_passed_mode_count": _median([
                metric.get("half_power", {}).get("passed_mode_count") for metric in ok_metrics
            ]),
            "median_half_power_predicted_ok_mode_count": _median([
                metric.get("half_power", {}).get("predicted_ok_mode_count") for metric in ok_metrics
            ]),
        })
    by_family_strategy = defaultdict(list)
    for row in rows:
        by_family_strategy[(row["family"], row["sampling_strategy"])].append(row)
    threshold_rows = []
    for key, items in sorted(by_family_strategy.items()):
        minimum_budget = None
        for row in sorted(items, key=lambda item: int(item["budget"])):
            frequency_error = row["median_modal_frequency_mae_hz"]
            width_error = row["median_modal_width_relative_mae"]
            if (
                frequency_error is not None
                and width_error is not None
                and float(frequency_error) <= float(frequency_threshold_hz)
                and float(width_error) <= float(width_relative_threshold)
            ):
                minimum_budget = int(row["budget"])
                break
        threshold_rows.append({
            "family": key[0],
            "sampling_strategy": key[1],
            "minimum_budget_meeting_modal_thresholds": minimum_budget,
            "frequency_threshold_hz": float(frequency_threshold_hz),
            "width_relative_threshold": float(width_relative_threshold),
        })
    return {
        "analysis_type": "aluminium_c_active_modal_pilot",
        "record_count": len(records),
        "ok_count": sum(record.get("status") == "ok" for record in records),
        "summary_rows": rows,
        "budget_threshold_rows": threshold_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--curve-points", default="1,2")
    parser.add_argument("--strategies", default="uniform,random,curvature")
    parser.add_argument("--budgets", default="24,48")
    parser.add_argument("--split-seeds", default="233")
    parser.add_argument("--init-seeds", default="1009")
    parser.add_argument("--initial-count", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--catalogue-band-half-width", type=float, default=16.0)
    parser.add_argument("--catalogue-min-band-points", type=int, default=7)
    parser.add_argument("--catalogue-max-modes", type=int, default=5)
    parser.add_argument("--half-power-band-half-width", type=float, default=8.0)
    parser.add_argument("--half-power-frequency-threshold", type=float, default=1.0)
    parser.add_argument("--half-power-width-relative-threshold", type=float, default=0.20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--min-steps", type=int, default=80)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument(
        "--target-transform",
        choices=("complex", "signed_log_complex"),
        default="complex",
    )
    parser.add_argument("--include-interpolation-baselines", action="store_true")
    parser.add_argument("--only-interpolation-baselines", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args()

    curve_points = _parse_ints(args.curve_points)
    strategies = _parse_strings(args.strategies)
    budgets = _parse_ints(args.budgets)
    split_seeds = _parse_ints(args.split_seeds)
    init_seeds = _parse_ints(args.init_seeds)
    families = _select_families(
        include_interpolation_baselines=bool(args.include_interpolation_baselines),
        only_interpolation_baselines=bool(args.only_interpolation_baselines),
    )
    jobs = sorted(
        _jobs(curve_points, strategies, budgets, split_seeds, init_seeds, families),
        key=_job_key,
    )
    if args.list_jobs:
        for index, job in enumerate(jobs):
            print(index, json.dumps(job, sort_keys=True))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_root / "raw_records.jsonl"
    existing = _load_jsonl(raw_path)
    completed = {_job_key(record) for record in existing}
    remaining = [job for job in jobs if _job_key(job) not in completed]
    if args.max_jobs is not None:
        remaining = remaining[: int(args.max_jobs)]

    curves_by_id = _curve_lookup(load_aluminium_frf_full(args.raw_root))
    selections = {family: _load_selection(args.selection_root, family) for family in FAMILIES}
    started_count = 0
    for job in remaining:
        started_count += 1
        print(f"[{started_count}/{len(remaining)}] {job}", flush=True)
        record = run_job(curves_by_id, selections, job, args)
        _append_jsonl(raw_path, record)

    records = _load_jsonl(raw_path)
    all_keys = {_job_key(job) for job in jobs}
    done_keys = {_job_key(record) for record in records}
    summary = {
        "analysis_type": "aluminium_c_active_modal_pilot_status",
        "total_job_count": len(jobs),
        "record_count": len(records),
        "remaining_job_count": len(all_keys - done_keys),
        "ran_this_invocation": len(remaining),
        "families": families,
        "curve_points": curve_points,
        "strategies": strategies,
        "budgets": budgets,
        "settings": {
            "initial_count": int(args.initial_count),
            "batch_size": int(args.batch_size),
            "max_steps": int(args.max_steps),
            "min_steps": int(args.min_steps),
            "patience": int(args.patience),
            "reference_grid": "full_2049_bin_h1_frf",
            "target_transform": str(args.target_transform),
            "catalogue_band_half_width": float(args.catalogue_band_half_width),
            "catalogue_min_band_points": int(args.catalogue_min_band_points),
            "catalogue_max_modes": int(args.catalogue_max_modes),
            "half_power_band_half_width": float(args.half_power_band_half_width),
            "half_power_frequency_threshold": float(args.half_power_frequency_threshold),
            "half_power_width_relative_threshold": float(args.half_power_width_relative_threshold),
        },
    }
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "analysis_summary.json", _summarize(records))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
