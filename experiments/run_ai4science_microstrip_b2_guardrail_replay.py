#!/usr/bin/env python3
"""Replay representative Microstrip B2 guardrail failures and plot full curves."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ai4science_microstrip_b import (  # noqa: E402
    microstrip_instance_effect_rows,
    microstrip_response_metrics,
    select_microstrip_b2_guardrail_replay_cases,
)
from confirmatory_models import build_model  # noqa: E402
from confirmatory_protocol import seeded_build  # noqa: E402
from downstream_protocol import load_microstrip_resonator  # noqa: E402
from downstream_training import fit_task_model, prepare_bundle  # noqa: E402
from run_ai4science_microstrip_b_pilot import (  # noqa: E402
    DEFAULT_SELECTION_ROOT,
    FAMILIES,
    _bundle_for_job,
    _load_jsonl,
    _load_selection,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_INPUT = (
    ROOT / "ai4science_results"
    / "microstrip_b2_missed_band_negative_control_seed_expansion"
    / "raw_records.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "ai4science_results"
    / "microstrip_b2_guardrail_replay_diagnostics"
)


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


def _write_json(path: Path, value: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(path)


def _job_from_case(case: dict, family: str) -> dict:
    return {
        "experiment": "B2",
        "scenario": str(case["scenario"]),
        "sample_id": str(case["sample_id"]),
        "target_temperature_c": 25.0,
        "target_relative_humidity_percent": 50.0,
        "calibration_frequency_count": int(case["calibration_frequency_count"]),
        "calibration_sampling_mode": str(case.get("calibration_sampling_mode", "missed_band")),
        "split_seed": int(case["split_seed"]),
        "init_seed": int(case["init_seed"]),
        "family": str(family),
    }


def _fit_prediction(curves, selections: dict[str, dict], job: dict, args) -> tuple[object, np.ndarray, dict]:
    bundle = _bundle_for_job(curves, job, args)
    selection = selections[job["family"]]
    candidate = selection["candidate"]
    optimizer = selection["optimizer"]
    model, model_metadata = seeded_build(
        job["init_seed"],
        lambda: build_model(
            job["family"],
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
        checkpoints=[step for step in (20, 80, 140) if step <= int(args.max_steps)],
        gradient_clip=5.0,
        batch_size=None,
        batch_seed=int(job["split_seed"]) + 49979687,
        validation_interval=10,
    )
    metadata = {
        "status": trained.status,
        "best_step": trained.best_step,
        "optimizer_steps": trained.optimizer_steps,
        "validation_score": trained.best_validation_score,
        "actual_parameters": int(candidate["actual_parameters"]),
        "candidate_config": candidate["config"],
        "optimizer": optimizer,
        "model_metadata": model_metadata,
    }
    return bundle, trained.prediction, metadata


def _curve_arrays(bundle, prediction: np.ndarray, curve_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = [
        row_index for row_index, point_id in enumerate(bundle.metadata["test_point_ids"])
        if point_id.split(":frequency:")[0] == curve_id
    ]
    if not indices:
        raise ValueError(f"curve {curve_id} not found in replay bundle")
    index_array = np.asarray(indices, dtype=np.int64)
    frequency = np.asarray(bundle.x_test[index_array, 0], dtype=np.float64)
    order = np.argsort(frequency, kind="mergesort")
    index_array = index_array[order]
    frequency = frequency[order]
    return frequency, np.asarray(bundle.y_test[index_array]), np.asarray(prediction[index_array])


def _complex(response: np.ndarray) -> np.ndarray:
    values = np.asarray(response, dtype=np.float64)
    return values[:, 0] + 1j * values[:, 1]


def _calibration_points_for_curve(curves_by_id: dict[str, object], bundle, curve_id: str) -> tuple[np.ndarray, np.ndarray]:
    curve = curves_by_id[curve_id]
    frequencies = []
    responses = []
    for point_id in bundle.metadata.get("calibration_point_ids", []):
        point_curve_id, raw_index = point_id.rsplit(":frequency:", 1)
        if point_curve_id != curve_id:
            continue
        index = int(raw_index)
        frequencies.append(float(curve.frequency[index]))
        responses.append(curve.response[index])
    if not frequencies:
        return np.empty(0, dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.asarray(frequencies, dtype=np.float64), np.asarray(responses, dtype=np.float64)


def _plot_case(
    path: Path,
    *,
    case: dict,
    frequency: np.ndarray,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    calibration_frequency: np.ndarray,
    calibration_response: np.ndarray,
) -> None:
    truth_complex = _complex(truth)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    x = frequency / 1e9
    axes[0].plot(x, np.abs(truth_complex), color="black", linewidth=2.0, label="truth")
    axes[1].plot(x, np.unwrap(np.angle(truth_complex)), color="black", linewidth=2.0, label="truth")
    colors = {"CFNN": "#c43c39", "MLP": "#3d65a5", "Local nested CF control": "#3b7f55"}
    for family, prediction in predictions.items():
        z = _complex(prediction)
        axes[0].plot(x, np.abs(z), color=colors.get(family, None), linewidth=1.4, label=family)
        axes[1].plot(x, np.unwrap(np.angle(z)), color=colors.get(family, None), linewidth=1.4, label=family)
    if len(calibration_frequency):
        axes[0].scatter(
            calibration_frequency / 1e9,
            np.abs(_complex(calibration_response)),
            marker="o",
            s=24,
            color="#d99000",
            edgecolor="black",
            linewidth=0.4,
            label="calibration",
            zorder=5,
        )
    truth_events = microstrip_response_metrics(frequency, truth, truth)["truth_events"]
    local_truth_events = microstrip_response_metrics(frequency, truth, truth)["local_truth_events"]
    for axis in axes:
        axis.axvline(float(truth_events["resonance_frequency_hz"]) / 1e9, color="black", linestyle="--", linewidth=0.9)
        axis.axvline(float(local_truth_events["phase_transition_hz"]) / 1e9, color="gray", linestyle=":", linewidth=0.9)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("|S21|")
    axes[1].set_ylabel("unwrapped phase")
    axes[1].set_xlabel("frequency (GHz)")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"{case['failure_type']} | {case['curve_id']} | cal={case['calibration_frequency_count']} | "
        f"baseline={case['comparison_family']}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-raw-records", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--max-cases-per-failure-type", type=int, default=1)
    parser.add_argument("--train-frequency-count", type=int, default=16)
    parser.add_argument("--validation-frequency-count", type=int, default=24)
    parser.add_argument("--calibration-excluded-band-linewidths", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--min-steps", type=int, default=50)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    records = _load_jsonl(args.input_raw_records)
    effect_rows = microstrip_instance_effect_rows(
        records,
        comparison_families={"MLP", "Local nested CF control"},
        metrics={
            "quality_factor_relative_error",
            "resonance_frequency_abs_error_hz",
            "local_phase_transition_abs_error_hz",
        },
    )
    cases = select_microstrip_b2_guardrail_replay_cases(
        effect_rows,
        max_cases_per_failure_type=int(args.max_cases_per_failure_type),
    )
    selections = {family: _load_selection(args.selection_root, family) for family in FAMILIES}
    curves = load_microstrip_resonator(args.raw_root)
    curves_by_id = {curve.curve_id: curve for curve in curves}

    case_summaries = []
    for index, case in enumerate(cases, start=1):
        families = ["CFNN", str(case["comparison_family"])]
        family_predictions = {}
        family_metrics = {}
        family_training = {}
        bundle = None
        for family in families:
            job = _job_from_case(case, family)
            bundle, prediction, training = _fit_prediction(curves, selections, job, args)
            frequency, truth, curve_prediction = _curve_arrays(bundle, prediction, str(case["curve_id"]))
            family_predictions[family] = curve_prediction
            family_metrics[family] = microstrip_response_metrics(frequency, truth, curve_prediction)
            family_training[family] = training
        if bundle is None:
            raise RuntimeError("no replay bundle generated")
        calibration_frequency, calibration_response = _calibration_points_for_curve(
            curves_by_id,
            bundle,
            str(case["curve_id"]),
        )
        figure_path = args.output_root / "figures" / f"case_{index:02d}_{case['failure_type']}_{case['curve_id'].replace(':', '_')}.png"
        _plot_case(
            figure_path,
            case=case,
            frequency=frequency,
            truth=truth,
            predictions=family_predictions,
            calibration_frequency=calibration_frequency,
            calibration_response=calibration_response,
        )
        case_summaries.append({
            **case,
            "figure_path": str(figure_path),
            "device": str(DEVICE),
            "training": family_training,
            "metrics": family_metrics,
            "calibration_frequency_count_for_curve": int(len(calibration_frequency)),
        })

    summary = {
        "analysis_type": "microstrip_b2_guardrail_replay_diagnostics",
        "input_raw_records": str(args.input_raw_records),
        "case_count": len(case_summaries),
        "settings": {
            "max_cases_per_failure_type": int(args.max_cases_per_failure_type),
            "train_frequency_count": int(args.train_frequency_count),
            "validation_frequency_count": int(args.validation_frequency_count),
            "calibration_excluded_band_linewidths": float(args.calibration_excluded_band_linewidths),
            "max_steps": int(args.max_steps),
            "min_steps": int(args.min_steps),
            "patience": int(args.patience),
        },
        "cases": case_summaries,
    }
    _write_json(args.output_root / "summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
