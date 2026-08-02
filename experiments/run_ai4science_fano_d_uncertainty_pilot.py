#!/usr/bin/env python3
"""Run Package D ensemble uncertainty pilot for Microwave Fano."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ai4science_fano import FanoReference, fit_complex_fano_curve
from ai4science_fano_a2 import make_sparse_curve_split
from ai4science_uncertainty_d import (
    calibration_scale_from_predictions,
    curve_stratified_conformal_scale,
    parameter_interval_metrics,
    response_interval_metrics,
    scaled_response_interval_metrics,
    split_conformal_scale,
    validation_calibrated_response_interval_metrics,
)
from confirmatory_models import build_model
from confirmatory_protocol import seeded_build
from downstream_protocol import load_microwave_fano
from downstream_training import fit_task_model, prepare_bundle
from run_ai4science_fano_a2_pilot import (
    canonical_family_name,
    _load_evaluation_curve_ids,
    _load_reference_map,
    _load_selection,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "microwave_fano_d_ensemble_uncertainty_pilot"


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


def _median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def summarize_family_ensembles(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], int(row["observation_budget"]))].append(row)
    family_rows = []
    for (family, budget), items in sorted(grouped.items()):
        family_rows.append({
            "family": family,
            "observation_budget": int(budget),
            "ensemble_count": len(items),
            "median_joint_coverage": _median([
                item.get("response_interval", {}).get("joint_coverage") for item in items
            ]),
            "median_peak_window_joint_coverage": _median([
                item.get("response_interval", {}).get("peak_window_joint_coverage") for item in items
            ]),
            "median_dangerous_point_failure_rate": _median([
                item.get("response_interval", {}).get("dangerous_point_failure_rate") for item in items
            ]),
            "median_error_uncertainty_rank_correlation": _median([
                item.get("response_interval", {}).get("error_uncertainty_rank_correlation") for item in items
            ]),
            "median_parameter_coverage_rate": _median([
                item.get("parameter_interval", {}).get("coverage_rate") for item in items
            ]),
            "median_dangerous_parameter_failure_rate": _median([
                item.get("parameter_interval", {}).get("dangerous_parameter_failure_rate") for item in items
            ]),
            "median_validation_calibrated_joint_coverage": _median([
                item.get("validation_calibrated_response_interval", {}).get("joint_coverage") for item in items
            ]),
            "median_validation_calibration_scale": _median([
                item.get("validation_calibrated_response_interval", {}).get("calibration_scale") for item in items
            ]),
            "median_cross_curve_calibrated_joint_coverage": _median([
                item.get("cross_curve_calibrated_response_interval", {}).get("joint_coverage") for item in items
            ]),
            "median_cross_curve_calibrated_peak_window_joint_coverage": _median([
                item.get("cross_curve_calibrated_response_interval", {}).get("peak_window_joint_coverage") for item in items
            ]),
            "median_cross_curve_calibration_scale": _median([
                item.get("cross_curve_calibrated_response_interval", {}).get("calibration_scale") for item in items
            ]),
            "median_cross_curve_dangerous_point_failure_rate": _median([
                item.get("cross_curve_calibrated_response_interval", {}).get("dangerous_point_failure_rate") for item in items
            ]),
            "median_global_split_conformal_joint_coverage": _median([
                item.get("global_split_conformal_response_interval", {}).get("joint_coverage") for item in items
            ]),
            "median_global_split_conformal_peak_window_joint_coverage": _median([
                item.get("global_split_conformal_response_interval", {}).get("peak_window_joint_coverage") for item in items
            ]),
            "median_global_split_conformal_calibration_scale": _median([
                item.get("global_split_conformal_response_interval", {}).get("calibration_scale") for item in items
            ]),
            "median_global_split_conformal_dangerous_point_failure_rate": _median([
                item.get("global_split_conformal_response_interval", {}).get("dangerous_point_failure_rate") for item in items
            ]),
            "median_curve_stratified_split_conformal_joint_coverage": _median([
                item.get("curve_stratified_split_conformal_response_interval", {}).get("joint_coverage") for item in items
            ]),
            "median_curve_stratified_split_conformal_peak_window_joint_coverage": _median([
                item.get("curve_stratified_split_conformal_response_interval", {}).get("peak_window_joint_coverage") for item in items
            ]),
            "median_curve_stratified_split_conformal_calibration_scale": _median([
                item.get("curve_stratified_split_conformal_response_interval", {}).get("calibration_scale") for item in items
            ]),
            "median_curve_stratified_split_conformal_dangerous_point_failure_rate": _median([
                item.get("curve_stratified_split_conformal_response_interval", {}).get("dangerous_point_failure_rate") for item in items
            ]),
        })
    return {
        "analysis_type": "microwave_fano_d_ensemble_uncertainty_pilot",
        "ensemble_count": len(rows),
        "family_rows": family_rows,
    }


def observation_budgets_from_args(args: argparse.Namespace) -> list[int]:
    """Return the ordered observation budgets requested by the CLI."""
    raw_budgets = getattr(args, "observation_budgets", None)
    if raw_budgets is None:
        raw_budgets = [getattr(args, "observation_budget")]
    budgets = []
    seen = set()
    for value in raw_budgets:
        budget = int(value)
        if budget <= 0:
            raise ValueError("observation budgets must be positive")
        if budget in seen:
            continue
        seen.add(budget)
        budgets.append(budget)
    if not budgets:
        raise ValueError("at least one observation budget is required")
    return budgets


def attach_cross_curve_calibrated_response_intervals(
    ensembles: list[dict],
    payloads: list[dict],
    *,
    confidence: float,
) -> None:
    """Attach leave-one-curve validation-scaled response metrics to ensembles."""
    payload_by_key = {
        (payload["family"], payload["curve_id"], int(payload["observation_budget"])): payload
        for payload in payloads
    }
    grouped = defaultdict(list)
    for payload in payloads:
        grouped[(payload["family"], int(payload["observation_budget"]))].append(payload)

    for ensemble in ensembles:
        family = ensemble["family"]
        curve_id = ensemble["curve_id"]
        budget = int(ensemble["observation_budget"])
        target = payload_by_key[(family, curve_id, budget)]
        calibration_payloads = [
            payload for payload in grouped[(family, budget)]
            if payload["curve_id"] != curve_id
        ]
        if not calibration_payloads:
            ensemble["cross_curve_calibrated_response_interval"] = {
                "status": "no_external_calibration_curves",
                "confidence": float(confidence),
                "calibration_curve_count": 0,
                "calibration_curve_ids": [],
            }
            continue

        calibration_scales = [
            calibration_scale_from_predictions(
                payload["validation_truth"],
                payload["validation_predictions"],
                confidence=float(confidence),
            )
            for payload in calibration_payloads
        ]
        scale = float(np.median(calibration_scales))
        calibration_point_count = int(sum(len(np.asarray(payload["validation_truth"])) for payload in calibration_payloads))
        metrics = scaled_response_interval_metrics(
            target["truth"],
            target["predictions"],
            scale=scale,
            frequency_hz=target.get("frequency_hz"),
            peak_center_hz=target.get("peak_center_hz"),
            peak_width_hz=target.get("peak_width_hz"),
            confidence=float(confidence),
            calibration_source="leave_one_curve_validation",
            calibration_point_count=calibration_point_count,
        )
        metrics.update({
            "calibration_curve_count": int(len(calibration_payloads)),
            "calibration_curve_ids": [payload["curve_id"] for payload in calibration_payloads],
            "calibration_scale_statistic": "median",
            "calibration_scales": [float(value) for value in calibration_scales],
        })
        ensemble["cross_curve_calibrated_response_interval"] = metrics


def _curve_stratum(curve_id: str) -> str:
    """Return the coarse coupling stratum encoded in Microwave Fano curve ids."""
    return str(curve_id).split(":", 1)[0]


def _scaled_prediction_metrics(target: dict, *, calibration: dict, calibration_source: str, confidence: float) -> dict:
    if calibration.get("status") != "ok":
        return dict(calibration)
    metrics = scaled_response_interval_metrics(
        target["truth"],
        target["predictions"],
        scale=float(calibration["scale"]),
        frequency_hz=target.get("frequency_hz"),
        peak_center_hz=target.get("peak_center_hz"),
        peak_width_hz=target.get("peak_width_hz"),
        confidence=float(confidence),
        calibration_source=calibration_source,
        calibration_point_count=int(calibration.get("calibration_point_count", 0)),
    )
    metrics.update({
        "calibration_curve_count": int(calibration["calibration_curve_count"]),
        "calibration_curve_ids": list(calibration["calibration_curve_ids"]),
        "calibration_strata": list(calibration.get("calibration_strata", [])),
        "calibration_scores": [float(value) for value in calibration.get("calibration_scores", [])],
        "target_stratum": calibration.get("target_stratum"),
        "calibration_scale_statistic": "finite_sample_split_conformal_quantile",
    })
    return metrics


def attach_split_conformal_response_intervals(
    ensembles: list[dict],
    payloads: list[dict],
    *,
    confidence: float,
    min_stratum_count: int = 2,
) -> None:
    """Attach prediction-level split-conformal response metrics without target leakage."""
    payload_by_key = {
        (payload["family"], payload["curve_id"], int(payload["observation_budget"])): payload
        for payload in payloads
    }
    grouped = defaultdict(list)
    for payload in payloads:
        grouped[(payload["family"], int(payload["observation_budget"]))].append(payload)

    calibration_rows_by_key = defaultdict(list)
    for payload in payloads:
        scale = calibration_scale_from_predictions(
            payload["validation_truth"],
            payload["validation_predictions"],
            confidence=float(confidence),
        )
        calibration_rows_by_key[(payload["family"], int(payload["observation_budget"]))].append({
            "curve_id": payload["curve_id"],
            "stratum": _curve_stratum(payload["curve_id"]),
            "scale": float(scale),
            "calibration_point_count": int(len(np.asarray(payload["validation_truth"]))),
        })

    for ensemble in ensembles:
        family = ensemble["family"]
        curve_id = ensemble["curve_id"]
        budget = int(ensemble["observation_budget"])
        target = payload_by_key[(family, curve_id, budget)]
        target_stratum = _curve_stratum(curve_id)
        candidate_rows = [
            row for row in calibration_rows_by_key[(family, budget)]
            if row["curve_id"] != curve_id
        ]

        if not candidate_rows:
            no_calibration = {
                "status": "no_external_calibration_curves",
                "confidence": float(confidence),
                "target_curve_id": curve_id,
                "target_stratum": target_stratum,
                "calibration_curve_count": 0,
                "calibration_curve_ids": [],
            }
            ensemble["global_split_conformal_response_interval"] = dict(no_calibration)
            ensemble["curve_stratified_split_conformal_response_interval"] = dict(no_calibration)
            continue

        global_calibration = {
            "status": "ok",
            "confidence": float(confidence),
            "target_curve_id": curve_id,
            "target_stratum": target_stratum,
            "scale": split_conformal_scale(
                [float(row["scale"]) for row in candidate_rows],
                confidence=float(confidence),
            ),
            "calibration_curve_count": int(len(candidate_rows)),
            "calibration_curve_ids": [str(row["curve_id"]) for row in candidate_rows],
            "calibration_strata": [str(row["stratum"]) for row in candidate_rows],
            "calibration_scores": [float(row["scale"]) for row in candidate_rows],
            "calibration_point_count": int(sum(row["calibration_point_count"] for row in candidate_rows)),
        }
        ensemble["global_split_conformal_response_interval"] = _scaled_prediction_metrics(
            target,
            calibration=global_calibration,
            calibration_source="global_split_conformal",
            confidence=float(confidence),
        )

        stratified_calibration = curve_stratified_conformal_scale(
            calibration_rows_by_key[(family, budget)],
            target_curve_id=curve_id,
            target_stratum=target_stratum,
            confidence=float(confidence),
            min_stratum_count=int(min_stratum_count),
        )
        if stratified_calibration.get("status") == "ok":
            selected_counts = {
                row["curve_id"]: int(row["calibration_point_count"])
                for row in candidate_rows
            }
            stratified_calibration["calibration_point_count"] = int(sum(
                selected_counts.get(curve_id, 0)
                for curve_id in stratified_calibration["calibration_curve_ids"]
            ))
        ensemble["curve_stratified_split_conformal_response_interval"] = _scaled_prediction_metrics(
            target,
            calibration=stratified_calibration,
            calibration_source=stratified_calibration.get("calibration_source", "curve_stratified_split_conformal"),
            confidence=float(confidence),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument(
        "--a1-reference-json",
        type=Path,
        default=ROOT / "ai4science_results" / "microwave_fano_a1" / "reference_parameters.json",
    )
    parser.add_argument(
        "--role-manifest",
        type=Path,
        default=ROOT / "peak_sensitive_results" / "provenance" / "microwave_fano_role_manifest.json",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=ROOT / "peak_sensitive_results" / "selections" / "microwave_fano",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--curve-ids", nargs="+", default=None)
    parser.add_argument("--families", nargs="+", default=["CFNN", "MLP", "Local nested CF control"])
    parser.add_argument("--observation-budget", type=int, default=64)
    parser.add_argument("--observation-budgets", nargs="+", type=int, default=None)
    parser.add_argument("--model-budget", type=int, default=512)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=719)
    parser.add_argument("--init-seeds", nargs="+", type=int, default=[11003, 12007, 13001])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--min-steps", type=int, default=80)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--validation-interval", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.80)
    return parser.parse_args()


def _reference_dict(reference: FanoReference) -> dict:
    return {
        "f0_hz": reference.f0_hz,
        "quality_factor": reference.quality_factor,
        "q": reference.q,
        "magnitude_peak_hz": reference.magnitude_peak_hz,
        "magnitude_valley_hz": reference.magnitude_valley_hz,
        "phase_transition_hz": reference.phase_transition_hz,
    }


def _fit_prediction_reference(curve, prediction: np.ndarray) -> FanoReference:
    return fit_complex_fano_curve(
        curve.frequency,
        prediction,
        curve_id=curve.curve_id,
        residual_threshold=0.05,
        conditions=curve.conditions,
        metadata={**curve.metadata, "prediction_source": "d_ensemble_member"},
    )


def _predict_physical(model: torch.nn.Module, x: torch.Tensor, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        raw = model(x).detach().cpu().numpy()
    return np.asarray(raw * y_std + y_mean, dtype=np.float32)


def run_member(args, *, curve, family: str, observation_budget: int, init_seed: int, device: torch.device) -> dict:
    selection = _load_selection(args.selection_root, family, int(args.model_budget))
    candidate = selection["candidate"]
    optimizer = selection["optimizer"]
    bundle = make_sparse_curve_split(
        curve,
        observation_count=int(observation_budget),
        validation_count=int(args.validation_count),
        split_seed=int(args.split_seed),
    )
    model, model_metadata = seeded_build(
        int(init_seed),
        lambda: build_model(
            family,
            input_dim=1,
            output_dim=2,
            config=candidate["config"],
            seed=int(init_seed),
        ),
    )
    model = model.to(device)
    prepared = prepare_bundle(bundle, "complex_regression", device)
    started = time.perf_counter()
    trained = fit_task_model(
        model,
        prepared,
        "complex_regression",
        learning_rate=float(optimizer["lr"]),
        weight_decay=float(optimizer["weight_decay"]),
        max_steps=int(args.max_steps),
        min_steps=min(int(args.min_steps), int(args.max_steps)),
        patience=int(args.patience),
        checkpoints=[step for step in (20, 80, 150, 300) if step <= int(args.max_steps)],
        gradient_clip=5.0,
        batch_seed=int(args.split_seed) + 49979687,
        validation_interval=int(args.validation_interval),
    )
    prediction = np.asarray(trained.prediction, dtype=np.float32)
    validation_prediction = _predict_physical(
        trained.model,
        prepared.x_validation,
        prepared.y_mean,
        prepared.y_std,
    )
    predicted_reference = _fit_prediction_reference(curve, prediction)
    return {
        "family": family,
        "curve_id": curve.curve_id,
        "observation_budget": int(observation_budget),
        "split_seed": int(args.split_seed),
        "init_seed": int(init_seed),
        "training_status": trained.status,
        "best_step": int(trained.best_step),
        "validation_score": float(trained.best_validation_score),
        "actual_parameters": int(model_metadata["actual_parameters"]),
        "candidate_key": candidate["candidate_key"],
        "training_wall_seconds": float(trained.wall_seconds),
        "total_wall_seconds": float(time.perf_counter() - started),
        "prediction": prediction,
        "validation_prediction": validation_prediction,
        "validation_truth": bundle.y_validation.astype(np.float32),
        "predicted_reference": predicted_reference.to_dict(),
    }


def run_family_ensemble(args, *, curve, reference: FanoReference, family: str, observation_budget: int, device: torch.device) -> tuple[dict, list[dict], dict]:
    members = [
        run_member(args, curve=curve, family=family, observation_budget=int(observation_budget), init_seed=seed, device=device)
        for seed in args.init_seeds
    ]
    predictions = np.asarray([member["prediction"] for member in members], dtype=np.float32)
    validation_predictions = np.asarray([member["validation_prediction"] for member in members], dtype=np.float32)
    validation_truth = np.asarray(members[0]["validation_truth"], dtype=np.float32)
    parameter_predictions = [
        _reference_dict(FanoReference(**member["predicted_reference"]))
        for member in members
        if member["predicted_reference"].get("status") == "ok"
    ]
    response_metrics = response_interval_metrics(
        curve.response,
        predictions,
        frequency_hz=curve.frequency,
        peak_center_hz=reference.f0_hz,
        peak_width_hz=reference.linewidth_hz,
        confidence=float(args.confidence),
    )
    parameter_metrics = parameter_interval_metrics(
        _reference_dict(reference),
        parameter_predictions,
        parameters=(
            "f0_hz",
            "quality_factor",
            "q",
            "magnitude_peak_hz",
            "magnitude_valley_hz",
            "phase_transition_hz",
        ),
        confidence=float(args.confidence),
        danger_thresholds={
            "f0_hz": {"max_interval_width": 2.0e3, "max_abs_error": 2.0e3},
            "quality_factor": {"max_interval_width": 5.0e3, "max_abs_error": 5.0e3},
            "q": {"max_interval_width": 0.05, "max_abs_error": 0.02},
            "magnitude_peak_hz": {"max_interval_width": 2.0e3, "max_abs_error": 2.0e3},
            "magnitude_valley_hz": {"max_interval_width": 1.0e3, "max_abs_error": 5.0e2},
            "phase_transition_hz": {"max_interval_width": 1.0e3, "max_abs_error": 1.0e3},
        },
    )
    validation_calibrated_metrics = validation_calibrated_response_interval_metrics(
        curve.response,
        predictions,
        calibration_truth=validation_truth,
        calibration_predictions=validation_predictions,
        frequency_hz=curve.frequency,
        peak_center_hz=reference.f0_hz,
        peak_width_hz=reference.linewidth_hz,
        confidence=float(args.confidence),
    )
    ensemble = {
        "family": family,
        "curve_id": curve.curve_id,
        "observation_budget": int(observation_budget),
        "split_seed": int(args.split_seed),
        "init_seeds": [int(seed) for seed in args.init_seeds],
        "confidence": float(args.confidence),
        "response_interval": response_metrics,
        "validation_calibrated_response_interval": validation_calibrated_metrics,
        "parameter_interval": parameter_metrics,
    }
    member_rows = []
    for member in members:
        cleaned = dict(member)
        cleaned.pop("prediction", None)
        cleaned.pop("validation_prediction", None)
        cleaned.pop("validation_truth", None)
        member_rows.append(cleaned)
    calibration_payload = {
        "family": family,
        "curve_id": curve.curve_id,
        "observation_budget": int(observation_budget),
        "truth": curve.response.astype(np.float32),
        "predictions": predictions,
        "validation_truth": validation_truth,
        "validation_predictions": validation_predictions,
        "frequency_hz": curve.frequency,
        "peak_center_hz": reference.f0_hz,
        "peak_width_hz": reference.linewidth_hz,
    }
    return ensemble, member_rows, calibration_payload


def main() -> None:
    args = parse_args()
    references = _load_reference_map(args.a1_reference_json)
    curve_ids = args.curve_ids or _load_evaluation_curve_ids(args.role_manifest)[:1]
    families = [canonical_family_name(family) for family in args.families]
    observation_budgets = observation_budgets_from_args(args)
    curves = {curve.curve_id: curve for curve in load_microwave_fano(args.raw_root)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)

    ensembles = []
    members = []
    calibration_payloads = []
    for curve_id in curve_ids:
        curve = curves[curve_id]
        reference = references[curve_id]
        for observation_budget in observation_budgets:
            for family in families:
                print(json.dumps({
                    "event": "start_ensemble",
                    "curve_id": curve_id,
                    "observation_budget": int(observation_budget),
                    "family": family,
                }), flush=True)
                ensemble, member_rows, calibration_payload = run_family_ensemble(
                    args,
                    curve=curve,
                    reference=reference,
                    family=family,
                    observation_budget=int(observation_budget),
                    device=device,
                )
                ensembles.append(ensemble)
                members.extend(member_rows)
                calibration_payloads.append(calibration_payload)

    attach_cross_curve_calibrated_response_intervals(
        ensembles,
        calibration_payloads,
        confidence=float(args.confidence),
    )
    attach_split_conformal_response_intervals(
        ensembles,
        calibration_payloads,
        confidence=float(args.confidence),
    )

    (args.output_root / "raw_members.jsonl").write_text(
        "".join(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n" for row in members)
    )
    summary = {
        **summarize_family_ensembles(ensembles),
        "curve_ids": list(curve_ids),
        "families": families,
        "observation_budget": int(observation_budgets[0]) if len(observation_budgets) == 1 else None,
        "observation_budgets": [int(budget) for budget in observation_budgets],
        "split_seed": int(args.split_seed),
        "init_seeds": [int(seed) for seed in args.init_seeds],
        "confidence": float(args.confidence),
        "ensemble_rows": ensembles,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(_json_safe({
        "record_count": len(ensembles),
        "member_count": len(members),
        "output_root": str(args.output_root),
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
