"""Package D uncertainty metrics for AI4Science response reconstruction."""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _as_complex(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 2:
        return array[:, 0] + 1j * array[:, 1]
    raise ValueError(f"{name} must have shape (n, 2)")


def _as_prediction_array(predictions: np.ndarray) -> np.ndarray:
    array = np.asarray(predictions, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError("predictions must have shape (members, n, 2)")
    if array.shape[0] < 2:
        raise ValueError("at least two ensemble members are required")
    if not np.all(np.isfinite(array)):
        raise ValueError("predictions must contain finite values")
    return array


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return None
    ar = _rankdata(a[mask])
    br = _rankdata(b[mask])
    if float(np.std(ar)) <= np.finfo(float).eps or float(np.std(br)) <= np.finfo(float).eps:
        return None
    return float(np.corrcoef(ar, br)[0, 1])


def _nominal_joint_scale(truth: np.ndarray, median: np.ndarray, lower: np.ndarray, upper: np.ndarray, confidence: float) -> float | None:
    half_width = np.maximum((upper - lower) / 2.0, np.finfo(float).eps)
    ratios = np.max(np.abs(truth - median) / half_width, axis=1)
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        return None
    return float(max(1.0, np.quantile(ratios, float(confidence))))


def response_interval_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    frequency_hz: np.ndarray | None = None,
    peak_center_hz: float | None = None,
    peak_width_hz: float | None = None,
    confidence: float = 0.95,
) -> dict:
    """Compute empirical ensemble interval coverage for complex responses."""
    prediction_array = _as_prediction_array(predictions)
    truth_array = np.asarray(truth, dtype=np.float64)
    if truth_array.shape != prediction_array.shape[1:]:
        raise ValueError("truth and prediction shapes differ")
    if not np.all(np.isfinite(truth_array)):
        raise ValueError("truth must contain finite values")
    alpha = 1.0 - float(confidence)
    if not 0.0 < alpha < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    lower = np.quantile(prediction_array, alpha / 2.0, axis=0)
    upper = np.quantile(prediction_array, 1.0 - alpha / 2.0, axis=0)
    median = np.median(prediction_array, axis=0)
    real_covered = (truth_array[:, 0] >= lower[:, 0]) & (truth_array[:, 0] <= upper[:, 0])
    imag_covered = (truth_array[:, 1] >= lower[:, 1]) & (truth_array[:, 1] <= upper[:, 1])
    joint_covered = real_covered & imag_covered
    widths = upper - lower
    radius = np.sqrt(widths[:, 0] ** 2 + widths[:, 1] ** 2)
    truth_complex = _as_complex(truth_array, name="truth")
    median_complex = _as_complex(median, name="median_prediction")
    point_error = np.abs(median_complex - truth_complex)
    narrow_threshold = float(np.quantile(radius, 0.25))
    dangerous = (radius <= narrow_threshold) & (~joint_covered)
    nominal_scale = _nominal_joint_scale(truth_array, median, lower, upper, float(confidence))

    peak_coverage = None
    peak_scale = None
    peak_count = 0
    if frequency_hz is not None and peak_center_hz is not None and peak_width_hz is not None:
        frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
        if len(frequency) != len(truth_array):
            raise ValueError("frequency length differs from response length")
        half_width = max(float(abs(peak_width_hz)), np.finfo(float).eps)
        peak_mask = np.abs(frequency - float(peak_center_hz)) <= half_width
        peak_count = int(np.sum(peak_mask))
        if peak_count:
            peak_coverage = float(np.mean(joint_covered[peak_mask]))
            peak_scale = _nominal_joint_scale(
                truth_array[peak_mask],
                median[peak_mask],
                lower[peak_mask],
                upper[peak_mask],
                float(confidence),
            )

    return {
        "status": "ok",
        "member_count": int(prediction_array.shape[0]),
        "point_count": int(prediction_array.shape[1]),
        "confidence": float(confidence),
        "real_coverage": float(np.mean(real_covered)),
        "imag_coverage": float(np.mean(imag_covered)),
        "joint_coverage": float(np.mean(joint_covered)),
        "peak_window_point_count": peak_count,
        "peak_window_joint_coverage": peak_coverage,
        "median_real_interval_width": float(np.median(widths[:, 0])),
        "median_imag_interval_width": float(np.median(widths[:, 1])),
        "median_complex_interval_radius": float(np.median(radius)),
        "nominal_joint_coverage_scale": nominal_scale,
        "peak_window_nominal_joint_coverage_scale": peak_scale,
        "error_uncertainty_rank_correlation": _rank_correlation(point_error, radius),
        "dangerous_point_failure_count": int(np.sum(dangerous)),
        "dangerous_point_failure_rate": float(np.mean(dangerous)),
    }


def _joint_coverage_from_scaled_interval(
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction_array = _as_prediction_array(predictions)
    truth_array = np.asarray(truth, dtype=np.float64)
    if truth_array.shape != prediction_array.shape[1:]:
        raise ValueError("truth and prediction shapes differ")
    median = np.median(prediction_array, axis=0)
    lower = np.min(prediction_array, axis=0)
    upper = np.max(prediction_array, axis=0)
    half_width = np.maximum((upper - lower) / 2.0, np.finfo(float).eps)
    scaled_lower = median - float(scale) * half_width
    scaled_upper = median + float(scale) * half_width
    covered = (
        (truth_array[:, 0] >= scaled_lower[:, 0])
        & (truth_array[:, 0] <= scaled_upper[:, 0])
        & (truth_array[:, 1] >= scaled_lower[:, 1])
        & (truth_array[:, 1] <= scaled_upper[:, 1])
    )
    radius = np.sqrt(np.sum((float(scale) * half_width) ** 2, axis=1))
    return covered, radius, median, truth_array


def calibration_scale_from_predictions(
    calibration_truth: np.ndarray,
    calibration_predictions: np.ndarray,
    *,
    confidence: float = 0.95,
) -> float:
    """Estimate a scalar interval expansion from calibration predictions."""
    calibration_array = _as_prediction_array(calibration_predictions)
    calibration_truth_array = np.asarray(calibration_truth, dtype=np.float64)
    if calibration_truth_array.shape != calibration_array.shape[1:]:
        raise ValueError("calibration truth and predictions differ in shape")
    calibration_lower = np.min(calibration_array, axis=0)
    calibration_upper = np.max(calibration_array, axis=0)
    calibration_median = np.median(calibration_array, axis=0)
    scale = _nominal_joint_scale(
        calibration_truth_array,
        calibration_median,
        calibration_lower,
        calibration_upper,
        float(confidence),
    )
    if scale is None:
        raise ValueError("no valid calibration points")
    return float(scale)


def split_conformal_scale(scores: Iterable[float], *, confidence: float = 0.95) -> float:
    """Return the finite-sample split-conformal quantile for scalar calibration scores."""
    values = sorted(
        float(value)
        for value in scores
        if value is not None and math.isfinite(float(value))
    )
    if not values:
        raise ValueError("at least one finite calibration score is required")
    alpha = 1.0 - float(confidence)
    if not 0.0 < alpha < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    index = min(max(rank, 1), len(values)) - 1
    return float(values[index])


def curve_stratified_conformal_scale(
    calibration_rows: Iterable[dict],
    *,
    target_curve_id: str,
    target_stratum: str,
    confidence: float = 0.95,
    min_stratum_count: int = 2,
) -> dict:
    """Estimate a leave-target-out conformal scale, preferring the target stratum."""
    rows = [
        row for row in calibration_rows
        if str(row.get("curve_id")) != str(target_curve_id)
        and row.get("scale") is not None
        and math.isfinite(float(row.get("scale")))
    ]
    stratum_rows = [
        row for row in rows
        if str(row.get("stratum")) == str(target_stratum)
    ]
    if len(stratum_rows) >= int(min_stratum_count):
        selected = stratum_rows
        source = "curve_stratified_split_conformal"
    else:
        selected = rows
        source = "global_split_conformal_fallback"
    if not selected:
        return {
            "status": "no_valid_calibration_curves",
            "confidence": float(confidence),
            "target_curve_id": str(target_curve_id),
            "target_stratum": str(target_stratum),
            "calibration_curve_count": 0,
            "calibration_curve_ids": [],
        }
    return {
        "status": "ok",
        "confidence": float(confidence),
        "target_curve_id": str(target_curve_id),
        "target_stratum": str(target_stratum),
        "calibration_source": source,
        "scale": split_conformal_scale(
            [float(row["scale"]) for row in selected],
            confidence=float(confidence),
        ),
        "calibration_curve_count": int(len(selected)),
        "calibration_curve_ids": [str(row["curve_id"]) for row in selected],
        "calibration_strata": [str(row.get("stratum")) for row in selected],
        "calibration_scores": [float(row["scale"]) for row in selected],
    }


def scaled_response_interval_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    scale: float,
    frequency_hz: np.ndarray | None = None,
    peak_center_hz: float | None = None,
    peak_width_hz: float | None = None,
    confidence: float = 0.95,
    calibration_source: str = "external",
    calibration_point_count: int | None = None,
) -> dict:
    """Evaluate response coverage after applying an externally supplied scale."""
    scale_number = float(scale)
    if not math.isfinite(scale_number) or scale_number <= 0.0:
        raise ValueError("scale must be positive and finite")
    covered, radius, median, truth_array = _joint_coverage_from_scaled_interval(
        truth,
        predictions,
        scale=scale_number,
    )
    truth_complex = _as_complex(truth_array, name="truth")
    median_complex = _as_complex(median, name="median_prediction")
    point_error = np.abs(median_complex - truth_complex)
    narrow_threshold = float(np.quantile(radius, 0.25))
    dangerous = (radius <= narrow_threshold) & (~covered)

    peak_coverage = None
    peak_count = 0
    if frequency_hz is not None and peak_center_hz is not None and peak_width_hz is not None:
        frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
        if len(frequency) != len(truth_array):
            raise ValueError("frequency length differs from response length")
        peak_mask = np.abs(frequency - float(peak_center_hz)) <= max(float(abs(peak_width_hz)), np.finfo(float).eps)
        peak_count = int(np.sum(peak_mask))
        if peak_count:
            peak_coverage = float(np.mean(covered[peak_mask]))

    return {
        "status": "ok",
        "member_count": int(np.asarray(predictions).shape[0]),
        "point_count": int(len(truth_array)),
        "calibration_point_count": calibration_point_count,
        "confidence": float(confidence),
        "calibration_scale": scale_number,
        "calibration_source": str(calibration_source),
        "joint_coverage": float(np.mean(covered)),
        "peak_window_point_count": peak_count,
        "peak_window_joint_coverage": peak_coverage,
        "median_scaled_complex_interval_radius": float(np.median(radius)),
        "error_uncertainty_rank_correlation": _rank_correlation(point_error, radius),
        "dangerous_point_failure_count": int(np.sum(dangerous)),
        "dangerous_point_failure_rate": float(np.mean(dangerous)),
    }


def validation_calibrated_response_interval_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    calibration_truth: np.ndarray,
    calibration_predictions: np.ndarray,
    frequency_hz: np.ndarray | None = None,
    peak_center_hz: float | None = None,
    peak_width_hz: float | None = None,
    confidence: float = 0.95,
) -> dict:
    """Evaluate test coverage after scaling ensemble intervals on validation data."""
    try:
        scale = calibration_scale_from_predictions(
            calibration_truth,
            calibration_predictions,
            confidence=float(confidence),
        )
    except ValueError:
        return {"status": "no_valid_calibration_points"}

    metrics = scaled_response_interval_metrics(
        truth,
        predictions,
        scale=float(scale),
        frequency_hz=frequency_hz,
        peak_center_hz=peak_center_hz,
        peak_width_hz=peak_width_hz,
        confidence=float(confidence),
        calibration_source="same_curve_validation",
        calibration_point_count=int(len(np.asarray(calibration_truth))),
    )
    return metrics


def _finite_prediction_values(predictions: Iterable[dict], parameter: str) -> list[float]:
    values = []
    for prediction in predictions:
        value = prediction.get(parameter)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def parameter_interval_metrics(
    reference: dict,
    predictions: Iterable[dict],
    *,
    parameters: Iterable[str],
    confidence: float = 0.95,
    danger_thresholds: dict | None = None,
) -> dict:
    """Compute empirical ensemble intervals for derived scientific parameters."""
    prediction_rows = list(predictions)
    alpha = 1.0 - float(confidence)
    if not 0.0 < alpha < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rows = []
    for parameter in parameters:
        reference_value = reference.get(parameter)
        if reference_value is None:
            continue
        reference_number = float(reference_value)
        if not math.isfinite(reference_number):
            continue
        values = _finite_prediction_values(prediction_rows, str(parameter))
        if len(values) < 2:
            continue
        array = np.asarray(values, dtype=np.float64)
        lower = float(np.quantile(array, alpha / 2.0))
        upper = float(np.quantile(array, 1.0 - alpha / 2.0))
        median = float(np.median(array))
        width = float(upper - lower)
        abs_error = abs(median - reference_number)
        covered = bool(lower <= reference_number <= upper)
        thresholds = (danger_thresholds or {}).get(parameter, {})
        max_width = thresholds.get("max_interval_width")
        max_error = thresholds.get("max_abs_error")
        dangerous = (
            max_width is not None
            and max_error is not None
            and width <= float(max_width)
            and abs_error > float(max_error)
        )
        rows.append({
            "parameter": str(parameter),
            "reference": reference_number,
            "member_count": len(values),
            "median_prediction": median,
            "lower": lower,
            "upper": upper,
            "interval_width": width,
            "covered": covered,
            "median_abs_error": float(abs_error),
            "dangerous_failure": bool(dangerous),
        })
    covered_count = sum(1 for row in rows if row["covered"])
    dangerous_count = sum(1 for row in rows if row["dangerous_failure"])
    return {
        "status": "ok" if rows else "no_valid_parameters",
        "member_count": len(prediction_rows),
        "confidence": float(confidence),
        "parameter_count": len(rows),
        "covered_count": int(covered_count),
        "coverage_rate": float(covered_count / len(rows)) if rows else None,
        "dangerous_parameter_failure_count": int(dangerous_count),
        "dangerous_parameter_failure_rate": float(dangerous_count / len(rows)) if rows else None,
        "parameter_rows": rows,
    }
