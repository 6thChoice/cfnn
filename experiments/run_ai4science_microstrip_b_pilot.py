#!/usr/bin/env python3
"""Run Package B microstrip digital-twin pilot experiments."""
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

from ai4science_microstrip_b import (  # noqa: E402
    make_b1_condition_split,
    make_b2_calibration_split,
    microstrip_response_metrics,
)
from confirmatory_models import build_model  # noqa: E402
from confirmatory_protocol import seeded_build  # noqa: E402
from downstream_protocol import load_microstrip_resonator  # noqa: E402
from downstream_training import fit_task_model, prepare_bundle  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_OUT = ROOT / "ai4science_results" / "microstrip_b1_b2_condition_pilot"
DEFAULT_SELECTION_ROOT = (
    ROOT / "downstream_group_transfer_results" / "selections"
    / "microstrip_resonator_group_transfer"
)
FAMILIES = ("CFNN", "MLP", "Local nested CF control")
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
    if isinstance(value, (np.integer,)):
        return int(value)
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
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(path)


def _load_selection(selection_root: Path, family: str) -> dict:
    path = Path(selection_root) / f"{FAMILY_SLUGS[family]}__256.json"
    if not path.exists():
        raise FileNotFoundError(path)
    selection = json.loads(path.read_text())
    if selection.get("status") != "selected":
        raise ValueError(f"selection is not usable for {family}: {path}")
    return selection


def _b1_jobs(samples: list[str], split_seeds: list[int], init_seeds: list[int]) -> list[dict]:
    scenarios = [
        {
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "holdout_temperature_c": None,
            "holdout_relative_humidity_percent": 50.0,
        },
        {
            "experiment": "B1",
            "scenario": "temperature_holdout",
            "holdout_temperature_c": 25.0,
            "holdout_relative_humidity_percent": None,
        },
        {
            "experiment": "B1",
            "scenario": "condition_combo_holdout",
            "holdout_temperature_c": 25.0,
            "holdout_relative_humidity_percent": 50.0,
        },
    ]
    return [
        {
            **scenario,
            "sample_id": sample,
            "calibration_frequency_count": None,
            "split_seed": int(split_seed),
            "init_seed": int(init_seed),
            "family": family,
        }
        for sample in samples
        for scenario in scenarios
        for family in FAMILIES
        for split_seed in split_seeds
        for init_seed in init_seeds
    ]


def _b2_jobs(
    samples: list[str],
    split_seeds: list[int],
    init_seeds: list[int],
    calibration_counts: list[int],
    calibration_sampling_modes: list[str],
) -> list[dict]:
    return [
        {
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "sample_id": sample,
            "target_temperature_c": 25.0,
            "target_relative_humidity_percent": 50.0,
            "calibration_frequency_count": int(calibration),
            "calibration_sampling_mode": str(calibration_sampling_mode),
            "split_seed": int(split_seed),
            "init_seed": int(init_seed),
            "family": family,
        }
        for sample in samples
        for calibration in calibration_counts
        for calibration_sampling_mode in calibration_sampling_modes
        for family in FAMILIES
        for split_seed in split_seeds
        for init_seed in init_seeds
    ]


def _job_key(record: dict) -> tuple:
    return (
        str(record["experiment"]),
        str(record["scenario"]),
        str(record["sample_id"]),
        str(record["family"]),
        int(record["split_seed"]),
        int(record["init_seed"]),
        None if record.get("calibration_frequency_count") is None
        else int(record["calibration_frequency_count"]),
        str(record.get("calibration_sampling_mode", "random")),
    )


def _bundle_for_job(curves, job: dict, args):
    if job["experiment"] == "B1":
        return make_b1_condition_split(
            curves,
            scenario=job["scenario"],
            sample_id=job["sample_id"],
            holdout_temperature_c=job["holdout_temperature_c"],
            holdout_relative_humidity_percent=job["holdout_relative_humidity_percent"],
            train_frequency_count=args.train_frequency_count,
            validation_frequency_count=args.validation_frequency_count,
            seed=job["split_seed"],
        )
    return make_b2_calibration_split(
        curves,
        sample_id=job["sample_id"],
        target_temperature_c=job["target_temperature_c"],
        target_relative_humidity_percent=job["target_relative_humidity_percent"],
        calibration_frequency_count=job["calibration_frequency_count"],
        support_frequency_count=args.train_frequency_count,
        validation_frequency_count=args.validation_frequency_count,
        seed=job["split_seed"],
        calibration_sampling_mode=job.get("calibration_sampling_mode", "random"),
        calibration_excluded_band_linewidths=args.calibration_excluded_band_linewidths,
    )


def _per_curve_metrics(bundle, prediction: np.ndarray) -> list[dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row_index, point_id in enumerate(bundle.metadata["test_point_ids"]):
        grouped[point_id.split(":frequency:")[0]].append(row_index)
    rows = []
    for curve_id, indices in sorted(grouped.items()):
        index_array = np.asarray(indices, dtype=np.int64)
        frequency = np.asarray(bundle.x_test[index_array, 0], dtype=np.float64)
        order = np.argsort(frequency, kind="mergesort")
        index_array = index_array[order]
        frequency = frequency[order]
        metrics = microstrip_response_metrics(
            frequency,
            np.asarray(bundle.y_test[index_array]),
            np.asarray(prediction[index_array]),
        )
        condition_index = bundle.metadata["test_curve_ids"].index(curve_id)
        rows.append({
            "curve_id": curve_id,
            "conditions": bundle.metadata["test_conditions"][condition_index],
            **metrics,
        })
    return rows


def _median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _event_identifiable(metric: dict) -> bool:
    if "truth_event_identifiable" in metric:
        return bool(metric["truth_event_identifiable"])
    truth = metric.get("truth_events", {})
    q_value = truth.get("quality_factor")
    f0_value = truth.get("resonance_frequency_hz")
    q_available = q_value is not None and math.isfinite(float(q_value))
    boundary = (
        f0_value is not None
        and math.isfinite(float(f0_value))
        and float(f0_value) >= 2.20e9
    )
    return bool(q_available and not boundary)


def _summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[(
            record["experiment"],
            record["scenario"],
            record["family"],
            record.get("calibration_frequency_count"),
            record.get("calibration_sampling_mode", "random"),
        )].append(record)
    rows = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        per_curve = [
            metric
            for record in items if record.get("status") == "ok"
            for metric in record.get("curve_metrics", [])
        ]
        identifiable = [
            metric for metric in per_curve if _event_identifiable(metric)
        ]
        q_available = [
            metric for metric in per_curve
            if bool(metric.get("truth_q_available"))
            or (
                metric.get("truth_events", {}).get("quality_factor") is not None
                and math.isfinite(float(metric.get("truth_events", {}).get("quality_factor")))
            )
        ]
        boundary = [
            metric for metric in per_curve
            if bool(metric.get("truth_high_frequency_boundary_notch"))
            or (
                metric.get("truth_events", {}).get("resonance_frequency_hz") is not None
                and math.isfinite(float(metric.get("truth_events", {}).get("resonance_frequency_hz")))
                and float(metric.get("truth_events", {}).get("resonance_frequency_hz")) >= 2.20e9
            )
        ]
        rows.append({
            "experiment": key[0],
            "scenario": key[1],
            "family": key[2],
            "calibration_frequency_count": key[3],
            "calibration_sampling_mode": key[4],
            "record_count": len(items),
            "ok_count": sum(record.get("status") == "ok" for record in items),
            "curve_metric_count": len(per_curve),
            "q_available_count": len(q_available),
            "high_frequency_boundary_notch_count": len(boundary),
            "event_identifiable_count": len(identifiable),
            "event_identifiable_rate": (
                float(len(identifiable) / len(per_curve)) if per_curve else None
            ),
            "median_all_curve_complex_nrmse": _median([
                row.get("complex_nrmse") for row in per_curve
            ]),
            "median_all_curve_peak_window_complex_nrmse": _median([
                row.get("peak_window_complex_nrmse") for row in per_curve
            ]),
            "median_identifiable_resonance_frequency_abs_error_hz": _median([
                row.get("resonance_frequency_abs_error_hz") for row in identifiable
            ]),
            "median_identifiable_quality_factor_relative_error": _median([
                row.get("quality_factor_relative_error") for row in identifiable
            ]),
            "median_identifiable_phase_transition_abs_error_hz": _median([
                row.get("phase_transition_abs_error_hz") for row in identifiable
            ]),
            "median_identifiable_local_phase_transition_abs_error_hz": _median([
                row.get("local_phase_transition_abs_error_hz") for row in identifiable
            ]),
            "median_complex_nrmse": _median([row.get("complex_nrmse") for row in per_curve]),
            "median_peak_window_complex_nrmse": _median([
                row.get("peak_window_complex_nrmse") for row in per_curve
            ]),
            "median_resonance_frequency_abs_error_hz": _median([
                row.get("resonance_frequency_abs_error_hz") for row in per_curve
            ]),
            "median_quality_factor_relative_error": _median([
                row.get("quality_factor_relative_error") for row in per_curve
            ]),
            "median_phase_transition_abs_error_hz": _median([
                row.get("phase_transition_abs_error_hz") for row in per_curve
            ]),
            "median_local_phase_transition_abs_error_hz": _median([
                row.get("local_phase_transition_abs_error_hz") for row in per_curve
            ]),
        })
    return {
        "analysis_type": "microstrip_b1_b2_condition_pilot",
        "record_count": len(records),
        "ok_count": sum(record.get("status") == "ok" for record in records),
        "summary_rows": rows,
    }


def run_job(curves, selections: dict[str, dict], job: dict, args) -> dict:
    started = time.perf_counter()
    bundle = _bundle_for_job(curves, job, args)
    family = job["family"]
    selection = selections[family]
    candidate = selection["candidate"]
    optimizer = selection["optimizer"]
    model, model_metadata = seeded_build(
        job["init_seed"],
        lambda: build_model(
            family,
            input_dim=3,
            output_dim=2,
            config=candidate["config"],
            seed=job["init_seed"],
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
        checkpoints=[step for step in (20, 80, 200, 500, 800, 1200) if step <= int(args.max_steps)],
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
        "actual_parameters": int(candidate["actual_parameters"]),
        "candidate_key": candidate["candidate_key"],
        "candidate_config": candidate["config"],
        "optimizer": optimizer,
        "model_metadata": model_metadata,
        "dataset": bundle.metadata,
        "best_step": trained.best_step,
        "optimizer_steps": trained.optimizer_steps,
        "validation_score": trained.best_validation_score,
        "training_wall_seconds": trained.wall_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "curve": trained.curve,
        "curve_metrics": [],
    }
    if trained.status == "ok":
        record["curve_metrics"] = _per_curve_metrics(bundle, trained.prediction)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--samples", default="s31,s33,s34")
    parser.add_argument("--experiments", default="B1,B2")
    parser.add_argument("--split-seeds", default="233")
    parser.add_argument("--init-seeds", default="1009")
    parser.add_argument("--calibration-counts", default="0,8,32")
    parser.add_argument("--b2-calibration-sampling-modes", default="random")
    parser.add_argument("--calibration-excluded-band-linewidths", type=float, default=1.0)
    parser.add_argument("--train-frequency-count", type=int, default=64)
    parser.add_argument("--validation-frequency-count", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--min-steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args()

    samples = [item.strip() for item in args.samples.split(",") if item.strip()]
    split_seeds = [int(item) for item in args.split_seeds.split(",") if item.strip()]
    init_seeds = [int(item) for item in args.init_seeds.split(",") if item.strip()]
    calibration_counts = [int(item) for item in args.calibration_counts.split(",") if item.strip()]
    calibration_sampling_modes = [
        item.strip() for item in args.b2_calibration_sampling_modes.split(",") if item.strip()
    ]
    experiments = {item.strip() for item in args.experiments.split(",") if item.strip()}
    jobs = []
    if "B1" in experiments:
        jobs.extend(_b1_jobs(samples, split_seeds, init_seeds))
    if "B2" in experiments:
        jobs.extend(_b2_jobs(samples, split_seeds, init_seeds, calibration_counts, calibration_sampling_modes))
    jobs = sorted(jobs, key=_job_key)
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

    selections = {family: _load_selection(args.selection_root, family) for family in FAMILIES}
    curves = load_microstrip_resonator(args.raw_root)
    started_count = 0
    for job in remaining:
        started_count += 1
        print(f"[{started_count}/{len(remaining)}] {job}", flush=True)
        record = run_job(curves, selections, job, args)
        _append_jsonl(raw_path, record)

    records = _load_jsonl(raw_path)
    all_keys = {_job_key(job) for job in jobs}
    done_keys = {_job_key(record) for record in records}
    summary = {
        "analysis_type": "microstrip_b1_b2_condition_pilot_status",
        "total_job_count": len(jobs),
        "record_count": len(records),
        "remaining_job_count": len(all_keys - done_keys),
        "ran_this_invocation": len(remaining),
        "families": list(FAMILIES),
        "samples": samples,
        "experiments": sorted(experiments),
        "settings": {
            "train_frequency_count": int(args.train_frequency_count),
            "validation_frequency_count": int(args.validation_frequency_count),
            "max_steps": int(args.max_steps),
            "min_steps": int(args.min_steps),
            "patience": int(args.patience),
            "b2_calibration_sampling_modes": calibration_sampling_modes,
            "calibration_excluded_band_linewidths": float(args.calibration_excluded_band_linewidths),
        },
    }
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "analysis_summary.json", _summarize(records))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
