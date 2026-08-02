"""Boundary descriptors for Microwave Fano D/E applicability analysis."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def _as_complex_response(response: np.ndarray) -> np.ndarray:
    array = np.asarray(response)
    if array.ndim == 1 and np.iscomplexobj(array):
        values = np.asarray(array, dtype=np.complex128)
    elif array.ndim == 2 and array.shape[1] == 2:
        values = np.asarray(array[:, 0], dtype=np.float64) + 1j * np.asarray(array[:, 1], dtype=np.float64)
    else:
        raise ValueError("response must be complex-valued or have shape (n, 2)")
    if len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("response must contain at least three finite points")
    return values


def _finite_float(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)) or abs(float(denominator)) <= np.finfo(float).eps:
        return None
    return float(numerator) / float(denominator)


def fano_spectral_descriptors(
    curve_id: str,
    frequency_hz: np.ndarray,
    response: np.ndarray,
    *,
    reference: dict | None = None,
) -> dict:
    """Compute curve-level spectral descriptors for Fano applicability analysis."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    y = _as_complex_response(response)
    if len(frequency) != len(y):
        raise ValueError("frequency and response lengths differ")
    if np.any(np.diff(frequency) <= 0.0):
        raise ValueError("frequency must be strictly increasing")

    span_hz = float(np.ptp(frequency))
    if span_hz <= 0.0:
        raise ValueError("frequency span must be positive")
    magnitude = np.abs(y)
    magnitude_rms = max(float(np.sqrt(np.mean(magnitude ** 2))), np.finfo(float).eps)
    magnitude_ptp = float(np.ptp(magnitude))
    magnitude_median = float(np.median(magnitude))
    phase = np.unwrap(np.angle(y))
    phase_slope = np.gradient(phase, frequency)
    x = (frequency - float(frequency[0])) / span_hz
    normalized_magnitude = (magnitude - magnitude_median) / max(magnitude_ptp, np.finfo(float).eps)
    first_derivative = np.gradient(normalized_magnitude, x)
    second_derivative = np.gradient(first_derivative, x)

    reference = dict(reference or {})
    f0_hz = _finite_float(reference.get("f0_hz"))
    linewidth_hz = _finite_float(reference.get("linewidth_hz"))
    peak_hz = _finite_float(reference.get("magnitude_peak_hz"))
    valley_hz = _finite_float(reference.get("magnitude_valley_hz"))
    q = _finite_float(reference.get("q"))
    fano_rmse = _finite_float(reference.get("relative_magnitude_rmse"))

    return {
        "curve_id": str(curve_id),
        "point_count": int(len(frequency)),
        "frequency_min_hz": float(frequency[0]),
        "frequency_max_hz": float(frequency[-1]),
        "frequency_span_hz": span_hz,
        "magnitude_dynamic_range_ratio": float(magnitude_ptp / magnitude_rms),
        "magnitude_peak_to_median_ratio": float((float(np.max(magnitude)) - magnitude_median) / magnitude_rms),
        "magnitude_valley_to_median_ratio": float((magnitude_median - float(np.min(magnitude))) / magnitude_rms),
        "phase_total_turns": float(np.ptp(phase) / (2.0 * math.pi)),
        "max_phase_slope_times_span": float(np.max(np.abs(phase_slope)) * span_hz),
        "max_normalized_magnitude_slope": float(np.max(np.abs(first_derivative))),
        "max_normalized_magnitude_curvature": float(np.max(np.abs(second_derivative))),
        "reference_status": reference.get("status"),
        "f0_position_fraction": _safe_ratio((f0_hz - float(frequency[0])) if f0_hz is not None else None, span_hz),
        "linewidth_fraction": _safe_ratio(linewidth_hz, span_hz),
        "peak_valley_separation_fraction": _safe_ratio(abs(peak_hz - valley_hz) if peak_hz is not None and valley_hz is not None else None, span_hz),
        "abs_q": abs(q) if q is not None else None,
        "fano_relative_magnitude_rmse": fano_rmse,
    }


def curve_family_budget_boundary_rows(summary: dict, descriptors_by_curve: dict[str, dict]) -> list[dict]:
    """Join D uncertainty rows with curve descriptors."""
    confidence = float(summary.get("confidence", 0.80))
    rows = []
    for item in summary.get("ensemble_rows", []):
        curve_id = item["curve_id"]
        raw = item.get("response_interval", {})
        same_curve = item.get("validation_calibrated_response_interval", {})
        cross_curve = item.get("cross_curve_calibrated_response_interval", {})
        same_scale = _finite_float(same_curve.get("calibration_scale"))
        cross_scale = _finite_float(cross_curve.get("calibration_scale"))
        cross_joint = _finite_float(cross_curve.get("joint_coverage"))
        same_dangerous = _finite_float(same_curve.get("dangerous_point_failure_rate"))
        dangerous = _finite_float(cross_curve.get("dangerous_point_failure_rate"))
        row = {
            **dict(descriptors_by_curve.get(curve_id, {"curve_id": curve_id})),
            "family": item["family"],
            "observation_budget": int(item["observation_budget"]),
            "raw_joint_coverage": _finite_float(raw.get("joint_coverage")),
            "same_curve_joint_coverage": _finite_float(same_curve.get("joint_coverage")),
            "same_curve_calibration_scale": same_scale,
            "same_curve_dangerous_point_failure_rate": same_dangerous,
            "cross_curve_joint_coverage": cross_joint,
            "cross_curve_peak_window_coverage": _finite_float(cross_curve.get("peak_window_joint_coverage")),
            "cross_curve_calibration_scale": cross_scale,
            "cross_curve_dangerous_point_failure_rate": dangerous,
            "same_to_cross_scale_ratio": _safe_ratio(same_scale, cross_scale),
            "cross_curve_coverage_deficit": max(0.0, confidence - cross_joint) if cross_joint is not None else None,
        }
        deficit = row["cross_curve_coverage_deficit"]
        row["boundary_score"] = (float(deficit or 0.0) + float(dangerous or 0.0))
        rows.append(row)
    return rows


def split_conformal_family_budget_boundary_rows(summary: dict, descriptors_by_curve: dict[str, dict]) -> list[dict]:
    """Join prediction-level split-conformal D rows with curve descriptors."""
    confidence = float(summary.get("confidence", 0.80))
    interval_fields = (
        ("global_split_conformal", "global_split_conformal_response_interval"),
        ("curve_stratified_split_conformal", "curve_stratified_split_conformal_response_interval"),
    )
    rows = []
    for item in summary.get("ensemble_rows", []):
        curve_id = item["curve_id"]
        for layer, field in interval_fields:
            interval = item.get(field, {})
            if interval.get("status") != "ok":
                continue
            joint = _finite_float(interval.get("joint_coverage"))
            dangerous = _finite_float(interval.get("dangerous_point_failure_rate"))
            deficit = max(0.0, confidence - joint) if joint is not None else None
            rows.append({
                **dict(descriptors_by_curve.get(curve_id, {"curve_id": curve_id})),
                "family": item["family"],
                "observation_budget": int(item["observation_budget"]),
                "conformal_layer": layer,
                "calibrated_joint_coverage": joint,
                "calibrated_peak_window_coverage": _finite_float(interval.get("peak_window_joint_coverage")),
                "calibration_scale": _finite_float(interval.get("calibration_scale")),
                "calibrated_dangerous_point_failure_rate": dangerous,
                "calibrated_coverage_deficit": deficit,
                "calibration_curve_count": int(interval.get("calibration_curve_count") or 0),
                "boundary_score": float(deficit or 0.0) + float(dangerous or 0.0),
            })
    return rows


def curve_boundary_summary(rows: Iterable[dict]) -> list[dict]:
    """Aggregate family-budget boundary rows to one row per curve."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["curve_id"]].append(row)
    curve_rows = []
    descriptor_keys = (
        "reference_status",
        "magnitude_dynamic_range_ratio",
        "phase_total_turns",
        "max_phase_slope_times_span",
        "max_normalized_magnitude_curvature",
        "linewidth_fraction",
        "peak_valley_separation_fraction",
        "abs_q",
        "fano_relative_magnitude_rmse",
    )
    for curve_id, items in sorted(grouped.items()):
        base = {"curve_id": curve_id, "row_count": len(items)}
        for key in descriptor_keys:
            base[key] = items[0].get(key)
        base.update({
            "median_raw_joint_coverage": _median(row.get("raw_joint_coverage") for row in items),
            "median_same_curve_joint_coverage": _median(row.get("same_curve_joint_coverage") for row in items),
            "median_cross_curve_joint_coverage": _median(row.get("cross_curve_joint_coverage") for row in items),
            "median_cross_curve_peak_window_coverage": _median(row.get("cross_curve_peak_window_coverage") for row in items),
            "median_same_curve_calibration_scale": _median(row.get("same_curve_calibration_scale") for row in items),
            "median_cross_curve_calibration_scale": _median(row.get("cross_curve_calibration_scale") for row in items),
            "median_same_to_cross_scale_ratio": _median(row.get("same_to_cross_scale_ratio") for row in items),
            "median_same_curve_dangerous_point_failure_rate": _median(row.get("same_curve_dangerous_point_failure_rate") for row in items),
            "median_cross_curve_dangerous_point_failure_rate": _median(row.get("cross_curve_dangerous_point_failure_rate") for row in items),
            "median_boundary_score": _median(row.get("boundary_score") for row in items),
        })
        curve_rows.append(base)
    return curve_rows


def split_conformal_boundary_summary(rows: Iterable[dict]) -> list[dict]:
    """Aggregate prediction-level split-conformal boundary rows by curve and layer."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["curve_id"], row["conformal_layer"])].append(row)
    curve_rows = []
    descriptor_keys = (
        "reference_status",
        "magnitude_dynamic_range_ratio",
        "phase_total_turns",
        "max_phase_slope_times_span",
        "max_normalized_magnitude_curvature",
        "linewidth_fraction",
        "peak_valley_separation_fraction",
        "abs_q",
        "fano_relative_magnitude_rmse",
    )
    for (curve_id, layer), items in sorted(grouped.items()):
        base = {
            "curve_id": curve_id,
            "conformal_layer": layer,
            "row_count": len(items),
        }
        for key in descriptor_keys:
            base[key] = items[0].get(key)
        base.update({
            "median_calibrated_joint_coverage": _median(row.get("calibrated_joint_coverage") for row in items),
            "median_calibrated_peak_window_coverage": _median(row.get("calibrated_peak_window_coverage") for row in items),
            "median_calibration_scale": _median(row.get("calibration_scale") for row in items),
            "median_calibrated_dangerous_point_failure_rate": _median(row.get("calibrated_dangerous_point_failure_rate") for row in items),
            "median_calibrated_coverage_deficit": _median(row.get("calibrated_coverage_deficit") for row in items),
            "median_boundary_score": _median(row.get("boundary_score") for row in items),
        })
        curve_rows.append(base)
    return curve_rows


def rank_boundary_curves(curve_rows: Iterable[dict]) -> list[dict]:
    """Sort curves from strongest boundary/failure case to easiest case."""
    def key(row: dict) -> tuple:
        coverage = row.get("median_cross_curve_joint_coverage")
        score = row.get("median_boundary_score")
        ratio = row.get("median_same_to_cross_scale_ratio")
        return (
            float(coverage) if coverage is not None else float("inf"),
            -float(score or 0.0),
            -float(ratio or 0.0),
            row["curve_id"],
        )

    return sorted((dict(row) for row in curve_rows), key=key)


def rank_split_conformal_boundary_curves(curve_rows: Iterable[dict]) -> list[dict]:
    """Sort split-conformal curve/layer rows from strongest boundary to easiest."""
    def key(row: dict) -> tuple:
        coverage = row.get("median_calibrated_joint_coverage")
        score = row.get("median_boundary_score")
        layer_priority = 0 if row.get("conformal_layer") == "global_split_conformal" else 1
        return (
            float(coverage) if coverage is not None else float("inf"),
            -float(score or 0.0),
            layer_priority,
            row["curve_id"],
        )

    return sorted((dict(row) for row in curve_rows), key=key)


def _finite_values(rows: Iterable[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _finite_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _threshold_or_quantile(
    rows: list[dict],
    key: str,
    supplied: float | None,
    *,
    quantile: float,
) -> float | None:
    if supplied is not None:
        return float(supplied)
    values = _finite_values(rows, key)
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(quantile)))


def descriptor_risk_rule_audit(
    conformal_curve_rows: Iterable[dict],
    *,
    conformal_layer: str = "global_split_conformal",
    confidence: float = 0.80,
    phase_turn_threshold: float | None = None,
    curvature_threshold: float | None = None,
    phase_turn_quantile: float = 0.20,
    curvature_quantile: float = 0.80,
) -> dict:
    """Evaluate a simple descriptor rule for split-conformal reliability boundaries."""
    rows = [
        dict(row) for row in conformal_curve_rows
        if str(row.get("conformal_layer")) == str(conformal_layer)
    ]
    if not rows:
        return {
            "status": "no_rows",
            "conformal_layer": str(conformal_layer),
            "curve_count": 0,
        }
    phase_threshold = _threshold_or_quantile(
        rows,
        "phase_total_turns",
        phase_turn_threshold,
        quantile=float(phase_turn_quantile),
    )
    curvature_cutoff = _threshold_or_quantile(
        rows,
        "max_normalized_magnitude_curvature",
        curvature_threshold,
        quantile=float(curvature_quantile),
    )
    if phase_threshold is None or curvature_cutoff is None:
        return {
            "status": "missing_descriptor_values",
            "conformal_layer": str(conformal_layer),
            "curve_count": len(rows),
        }

    curve_rows = []
    confusion = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }
    for row in rows:
        phase_turns = _finite_float(row.get("phase_total_turns"))
        curvature = _finite_float(row.get("max_normalized_magnitude_curvature"))
        low_phase = phase_turns is not None and phase_turns <= phase_threshold
        high_curvature = curvature is not None and curvature >= curvature_cutoff
        predicted = bool(low_phase and high_curvature)
        coverage = _finite_float(row.get("median_calibrated_joint_coverage"))
        dangerous = _finite_float(row.get("median_calibrated_dangerous_point_failure_rate"))
        boundary_score = _finite_float(row.get("median_boundary_score"))
        observed = bool(
            (coverage is not None and coverage < float(confidence))
            or (dangerous is not None and dangerous > 0.10)
            or (boundary_score is not None and boundary_score > 0.10)
        )
        if predicted and observed:
            confusion["true_positive"] += 1
        elif predicted and not observed:
            confusion["false_positive"] += 1
        elif not predicted and observed:
            confusion["false_negative"] += 1
        else:
            confusion["true_negative"] += 1
        if predicted:
            risk_stratum = "weak_phase_high_curvature"
        elif low_phase:
            risk_stratum = "weak_phase_only"
        elif high_curvature:
            risk_stratum = "high_curvature_only"
        else:
            risk_stratum = "ordinary_resonant"
        curve_rows.append({
            **row,
            "low_phase": bool(low_phase),
            "high_curvature": bool(high_curvature),
            "predicted_high_risk": predicted,
            "observed_conformal_failure": observed,
            "risk_stratum": risk_stratum,
        })

    tp = confusion["true_positive"]
    fp = confusion["false_positive"]
    tn = confusion["true_negative"]
    fn = confusion["false_negative"]
    return {
        "status": "ok",
        "conformal_layer": str(conformal_layer),
        "curve_count": len(rows),
        "rule": {
            "risk_stratum": "weak_phase_high_curvature",
            "phase_turn_threshold": float(phase_threshold),
            "curvature_threshold": float(curvature_cutoff),
            "phase_turn_quantile": float(phase_turn_quantile),
            "curvature_quantile": float(curvature_quantile),
            "failure_confidence_threshold": float(confidence),
            "failure_dangerous_rate_threshold": 0.10,
            "failure_boundary_score_threshold": 0.10,
        },
        "confusion": confusion,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "curve_rows": sorted(
            curve_rows,
            key=lambda row: (
                not bool(row.get("predicted_high_risk")),
                not bool(row.get("observed_conformal_failure")),
                str(row.get("curve_id")),
            ),
        ),
    }


def descriptor_guardrail_policy_audit(risk_audit: dict) -> dict:
    """Evaluate an accept/reject policy derived from descriptor risk rows."""
    if risk_audit.get("status") != "ok":
        return {
            "status": "risk_audit_not_ok",
            "conformal_layer": risk_audit.get("conformal_layer"),
            "risk_audit_status": risk_audit.get("status"),
        }
    rows = [dict(row) for row in risk_audit.get("curve_rows", [])]
    if not rows:
        return {
            "status": "no_rows",
            "conformal_layer": risk_audit.get("conformal_layer"),
        }
    accepted = [row for row in rows if not bool(row.get("predicted_high_risk"))]
    rejected = [row for row in rows if bool(row.get("predicted_high_risk"))]
    failures = [row for row in rows if bool(row.get("observed_conformal_failure"))]
    accepted_failures = [row for row in accepted if bool(row.get("observed_conformal_failure"))]
    rejected_failures = [row for row in rejected if bool(row.get("observed_conformal_failure"))]
    accepted_coverages = _finite_values(accepted, "median_calibrated_joint_coverage")
    rejected_coverages = _finite_values(rejected, "median_calibrated_joint_coverage")
    accepted_danger = _finite_values(accepted, "median_calibrated_dangerous_point_failure_rate")
    rejected_danger = _finite_values(rejected, "median_calibrated_dangerous_point_failure_rate")

    total = len(rows)
    return {
        "status": "ok",
        "conformal_layer": risk_audit.get("conformal_layer"),
        "policy": "accept_ordinary_reject_weak_phase_high_curvature",
        "curve_count": total,
        "accepted_curve_count": len(accepted),
        "rejected_curve_count": len(rejected),
        "rejection_rate": float(len(rejected) / total) if total else None,
        "observed_failure_count": len(failures),
        "accepted_failure_count": len(accepted_failures),
        "rejected_failure_count": len(rejected_failures),
        "accepted_failure_rate": float(len(accepted_failures) / len(accepted)) if accepted else None,
        "rejected_failure_rate": float(len(rejected_failures) / len(rejected)) if rejected else None,
        "failure_capture_rate": float(len(rejected_failures) / len(failures)) if failures else None,
        "accepted_median_joint_coverage": float(np.median(accepted_coverages)) if accepted_coverages else None,
        "rejected_median_joint_coverage": float(np.median(rejected_coverages)) if rejected_coverages else None,
        "accepted_median_dangerous_failure_rate": float(np.median(accepted_danger)) if accepted_danger else None,
        "rejected_median_dangerous_failure_rate": float(np.median(rejected_danger)) if rejected_danger else None,
        "accepted_curve_ids": [str(row.get("curve_id")) for row in accepted],
        "rejected_curve_ids": [str(row.get("curve_id")) for row in rejected],
    }


def descriptor_risk_rule_sensitivity_audit(
    conformal_curve_rows: Iterable[dict],
    *,
    conformal_layer: str = "global_split_conformal",
    confidence: float = 0.80,
    phase_turn_quantiles: Iterable[float] = (0.10, 0.20, 0.30, 0.40),
    curvature_quantiles: Iterable[float] = (0.60, 0.70, 0.80, 0.90),
) -> dict:
    """Sweep descriptor quantiles to assess risk-rule threshold sensitivity."""
    rows = [
        dict(row) for row in conformal_curve_rows
        if str(row.get("conformal_layer")) == str(conformal_layer)
    ]
    if not rows:
        return {
            "status": "no_rows",
            "conformal_layer": str(conformal_layer),
            "curve_count": 0,
        }

    grid_rows = []
    for phase_quantile in phase_turn_quantiles:
        for curvature_quantile in curvature_quantiles:
            risk = descriptor_risk_rule_audit(
                rows,
                conformal_layer=conformal_layer,
                confidence=float(confidence),
                phase_turn_quantile=float(phase_quantile),
                curvature_quantile=float(curvature_quantile),
            )
            if risk.get("status") != "ok":
                continue
            policy = descriptor_guardrail_policy_audit(risk)
            confusion = risk.get("confusion", {})
            tp = int(confusion.get("true_positive", 0))
            fp = int(confusion.get("false_positive", 0))
            fn = int(confusion.get("false_negative", 0))
            rejected_curve_ids = list(policy.get("rejected_curve_ids", [])) if policy.get("status") == "ok" else []
            exact_isolating = bool(tp >= 1 and fp == 0 and fn == 0)
            accepted_zero_failure = bool(
                policy.get("status") == "ok"
                and _finite_float(policy.get("accepted_failure_rate")) == 0.0
            )
            grid_rows.append({
                "conformal_layer": str(conformal_layer),
                "phase_turn_quantile": float(phase_quantile),
                "curvature_quantile": float(curvature_quantile),
                "phase_turn_threshold": risk.get("rule", {}).get("phase_turn_threshold"),
                "curvature_threshold": risk.get("rule", {}).get("curvature_threshold"),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": int(confusion.get("true_negative", 0)),
                "false_negative": fn,
                "precision": risk.get("precision"),
                "recall": risk.get("recall"),
                "specificity": risk.get("specificity"),
                "rejection_rate": policy.get("rejection_rate") if policy.get("status") == "ok" else None,
                "accepted_failure_rate": policy.get("accepted_failure_rate") if policy.get("status") == "ok" else None,
                "failure_capture_rate": policy.get("failure_capture_rate") if policy.get("status") == "ok" else None,
                "rejected_curve_ids": rejected_curve_ids,
                "exact_isolating_rule": exact_isolating,
                "accepted_zero_failure_rule": accepted_zero_failure,
            })

    if not grid_rows:
        return {
            "status": "no_valid_grid_rows",
            "conformal_layer": str(conformal_layer),
            "curve_count": len(rows),
        }

    def best_key(row: dict) -> tuple:
        return (
            not bool(row.get("exact_isolating_rule")),
            not bool(row.get("accepted_zero_failure_rule")),
            int(row.get("false_positive", 0)) + int(row.get("false_negative", 0)),
            -float(row.get("true_positive", 0)),
            float(row.get("rejection_rate") or 0.0),
            -float(row.get("curvature_quantile") or 0.0),
            float(row.get("phase_turn_quantile") or 0.0),
        )

    exact_rows = [row for row in grid_rows if bool(row.get("exact_isolating_rule"))]
    zero_failure_rows = [row for row in grid_rows if bool(row.get("accepted_zero_failure_rule"))]
    best = sorted(grid_rows, key=best_key)[0]
    return {
        "status": "ok",
        "conformal_layer": str(conformal_layer),
        "curve_count": len(rows),
        "grid_count": len(grid_rows),
        "exact_isolating_rule_count": len(exact_rows),
        "accepted_zero_failure_rule_count": len(zero_failure_rows),
        "exact_isolating_rule_fraction": float(len(exact_rows) / len(grid_rows)),
        "accepted_zero_failure_rule_fraction": float(len(zero_failure_rows) / len(grid_rows)),
        "phase_turn_quantiles": [float(value) for value in phase_turn_quantiles],
        "curvature_quantiles": [float(value) for value in curvature_quantiles],
        "best_rule": dict(best),
        "grid_rows": sorted(
            grid_rows,
            key=lambda row: (
                float(row.get("phase_turn_quantile") or 0.0),
                float(row.get("curvature_quantile") or 0.0),
            ),
        ),
    }


def descriptor_same_curve_escalation_policy_audit(
    risk_audit: dict,
    same_curve_rows: Iterable[dict],
    *,
    confidence: float = 0.80,
) -> dict:
    """Evaluate accepting ordinary curves while escalating high-risk curves to same-curve calibration."""
    if risk_audit.get("status") != "ok":
        return {
            "status": "risk_audit_not_ok",
            "conformal_layer": risk_audit.get("conformal_layer"),
            "risk_audit_status": risk_audit.get("status"),
        }
    rows = [dict(row) for row in risk_audit.get("curve_rows", [])]
    if not rows:
        return {
            "status": "no_rows",
            "conformal_layer": risk_audit.get("conformal_layer"),
        }
    same_by_curve = {str(row.get("curve_id")): dict(row) for row in same_curve_rows}
    policy_rows = []
    missing_same_curve = []
    for row in rows:
        curve_id = str(row.get("curve_id"))
        high_risk = bool(row.get("predicted_high_risk"))
        if high_risk:
            same = same_by_curve.get(curve_id)
            if same is None:
                missing_same_curve.append(curve_id)
                joint = None
                dangerous = None
                scale = None
            else:
                joint = _finite_float(same.get("median_same_curve_joint_coverage"))
                dangerous = _finite_float(same.get("median_same_curve_dangerous_point_failure_rate"))
                scale = _finite_float(same.get("median_same_curve_calibration_scale"))
            source = "same_curve_calibration"
        else:
            joint = _finite_float(row.get("median_calibrated_joint_coverage"))
            dangerous = _finite_float(row.get("median_calibrated_dangerous_point_failure_rate"))
            scale = _finite_float(row.get("median_calibration_scale"))
            source = "split_conformal"
        failure = bool(
            (joint is not None and joint < float(confidence))
            or (dangerous is not None and dangerous > 0.10)
        )
        policy_rows.append({
            "curve_id": curve_id,
            "risk_stratum": row.get("risk_stratum"),
            "predicted_high_risk": high_risk,
            "policy_source": source,
            "post_policy_joint_coverage": joint,
            "post_policy_dangerous_point_failure_rate": dangerous,
            "post_policy_calibration_scale": scale,
            "post_policy_failure": failure,
        })

    accepted = [row for row in policy_rows if row["policy_source"] == "split_conformal"]
    escalated = [row for row in policy_rows if row["policy_source"] == "same_curve_calibration"]
    failures = [row for row in policy_rows if bool(row.get("post_policy_failure"))]
    post_joint = _finite_values(policy_rows, "post_policy_joint_coverage")
    post_danger = _finite_values(policy_rows, "post_policy_dangerous_point_failure_rate")
    escalated_joint = _finite_values(escalated, "post_policy_joint_coverage")
    escalated_danger = _finite_values(escalated, "post_policy_dangerous_point_failure_rate")
    total = len(policy_rows)
    return {
        "status": "ok",
        "conformal_layer": risk_audit.get("conformal_layer"),
        "policy": "accept_ordinary_escalate_weak_phase_high_curvature_to_same_curve",
        "curve_count": total,
        "accepted_curve_count": len(accepted),
        "escalated_curve_count": len(escalated),
        "escalation_rate": float(len(escalated) / total) if total else None,
        "post_policy_failure_count": len(failures),
        "post_policy_failure_rate": float(len(failures) / total) if total else None,
        "post_policy_median_joint_coverage": float(np.median(post_joint)) if post_joint else None,
        "post_policy_median_dangerous_failure_rate": float(np.median(post_danger)) if post_danger else None,
        "escalated_median_same_curve_joint_coverage": float(np.median(escalated_joint)) if escalated_joint else None,
        "escalated_median_same_curve_dangerous_failure_rate": float(np.median(escalated_danger)) if escalated_danger else None,
        "accepted_curve_ids": [str(row.get("curve_id")) for row in accepted],
        "escalated_curve_ids": [str(row.get("curve_id")) for row in escalated],
        "missing_same_curve_curve_ids": sorted(set(missing_same_curve)),
        "curve_rows": sorted(
            policy_rows,
            key=lambda row: (
                row.get("policy_source") != "same_curve_calibration",
                str(row.get("curve_id")),
            ),
        ),
    }
