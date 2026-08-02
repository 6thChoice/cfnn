"""Package B utilities for microstrip condition generalization experiments."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import numpy as np
from scipy.signal import find_peaks

from confirmatory_protocol import DatasetBundle
from downstream_protocol import MeasuredCurve


def _seed(*parts: object) -> int:
    encoded = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _point_id(curve: MeasuredCurve, index: int) -> str:
    return f"{curve.curve_id}:frequency:{int(index)}"


def _sample_indices(curve: MeasuredCurve, count: int | None, seed: int, role: str) -> np.ndarray:
    if count is None or int(count) >= len(curve.frequency):
        return np.arange(len(curve.frequency), dtype=np.int64)
    if int(count) < 0:
        raise ValueError("frequency count must be non-negative")
    if int(count) == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(_seed("microstrip-b", seed, role, curve.curve_id))
    return np.sort(rng.choice(len(curve.frequency), size=int(count), replace=False))


def _sample_indices_outside_resonance_band(
    curve: MeasuredCurve,
    count: int,
    seed: int,
    role: str,
    *,
    excluded_band_linewidths: float,
) -> np.ndarray:
    if int(count) < 0:
        raise ValueError("frequency count must be non-negative")
    if int(count) == 0:
        return np.empty(0, dtype=np.int64)
    events = extract_microstrip_events(curve.frequency, curve.response)
    f0 = events.get("resonance_frequency_hz")
    linewidth = events.get("linewidth_hz")
    if f0 is None or linewidth is None or not math.isfinite(float(linewidth)) or float(linewidth) <= 0.0:
        raise ValueError(f"cannot build missed-band calibration for {curve.curve_id}: missing linewidth")
    excluded_half_width = float(excluded_band_linewidths) * float(linewidth)
    eligible = np.flatnonzero(np.abs(np.asarray(curve.frequency, dtype=np.float64) - float(f0)) > excluded_half_width)
    if len(eligible) < int(count):
        raise ValueError(
            f"insufficient missed-band frequencies for {curve.curve_id}: "
            f"need {int(count)}, have {len(eligible)}"
        )
    rng = np.random.default_rng(_seed("microstrip-b", seed, role, curve.curve_id))
    return np.sort(rng.choice(eligible, size=int(count), replace=False))


def _input_rows(curve: MeasuredCurve, indices: np.ndarray) -> np.ndarray:
    temperature = float(curve.conditions["temperature_c"])
    humidity = float(curve.conditions["relative_humidity_percent"])
    conditions = np.repeat([[temperature, humidity]], len(indices), axis=0)
    return np.column_stack((curve.frequency[indices], conditions)).astype(np.float64)


def _stack_curves(
    curves: Iterable[MeasuredCurve],
    *,
    frequency_count: int | None,
    seed: int,
    role: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_parts = []
    y_parts = []
    point_ids = []
    for curve in curves:
        indices = _sample_indices(curve, frequency_count, seed, role)
        if not len(indices):
            continue
        x_parts.append(_input_rows(curve, indices))
        y_parts.append(curve.response[indices].astype(np.float32))
        point_ids.extend(_point_id(curve, index) for index in indices)
    if not x_parts:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float32),
            [],
        )
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        point_ids,
    )


def _stack_curves_from_indices(
    curves: Iterable[MeasuredCurve],
    indices_by_curve: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_parts = []
    y_parts = []
    point_ids = []
    for curve in curves:
        indices = np.asarray(indices_by_curve.get(curve.curve_id, np.empty(0, dtype=np.int64)), dtype=np.int64)
        if not len(indices):
            continue
        x_parts.append(_input_rows(curve, indices))
        y_parts.append(curve.response[indices].astype(np.float32))
        point_ids.extend(_point_id(curve, index) for index in indices)
    if not x_parts:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float32),
            [],
        )
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        point_ids,
    )


def _curve_conditions(curves: Iterable[MeasuredCurve]) -> list[dict]:
    return [dict(curve.conditions) for curve in curves]


def _same_sample_curves(curves: Iterable[MeasuredCurve], sample_id: str) -> list[MeasuredCurve]:
    selected = [
        curve for curve in curves
        if str(curve.conditions["sample_id"]) == str(sample_id)
    ]
    if not selected:
        raise ValueError(f"no microstrip curves found for sample_id={sample_id}")
    return sorted(selected, key=lambda curve: curve.curve_id)


def _matches_holdout(
    curve: MeasuredCurve,
    *,
    scenario: str,
    holdout_temperature_c: float | None,
    holdout_relative_humidity_percent: float | None,
) -> bool:
    temperature = float(curve.conditions["temperature_c"])
    humidity = float(curve.conditions["relative_humidity_percent"])
    if scenario == "humidity_holdout":
        if holdout_relative_humidity_percent is None:
            raise ValueError("humidity_holdout requires holdout_relative_humidity_percent")
        return humidity == float(holdout_relative_humidity_percent)
    if scenario == "temperature_holdout":
        if holdout_temperature_c is None:
            raise ValueError("temperature_holdout requires holdout_temperature_c")
        return temperature == float(holdout_temperature_c)
    if scenario == "condition_combo_holdout":
        if holdout_temperature_c is None or holdout_relative_humidity_percent is None:
            raise ValueError("condition_combo_holdout requires temperature and humidity")
        return (
            temperature == float(holdout_temperature_c)
            and humidity == float(holdout_relative_humidity_percent)
        )
    raise ValueError(f"unknown B1 scenario: {scenario}")


def make_b1_condition_split(
    curves: Iterable[MeasuredCurve],
    *,
    scenario: str,
    sample_id: str,
    holdout_temperature_c: float | None,
    holdout_relative_humidity_percent: float | None,
    train_frequency_count: int,
    validation_frequency_count: int,
    seed: int,
    validation_curve_count: int = 4,
) -> DatasetBundle:
    """Hold out complete condition curves for same-device digital-twin testing."""
    sample_curves = _same_sample_curves(curves, sample_id)
    test_curves = [
        curve for curve in sample_curves
        if _matches_holdout(
            curve,
            scenario=scenario,
            holdout_temperature_c=holdout_temperature_c,
            holdout_relative_humidity_percent=holdout_relative_humidity_percent,
        )
    ]
    if not test_curves:
        raise ValueError(f"no test curves for {sample_id}/{scenario}")
    pool = [curve for curve in sample_curves if curve.curve_id not in {c.curve_id for c in test_curves}]
    if len(pool) < 2:
        raise ValueError(f"insufficient non-holdout curves for {sample_id}/{scenario}")
    rng = np.random.default_rng(_seed("microstrip-b1-validation", seed, sample_id, scenario))
    order = rng.permutation(len(pool))
    n_validation = max(1, min(int(validation_curve_count), len(pool) - 1))
    validation_curves = [pool[int(index)] for index in order[:n_validation]]
    train_curves = [pool[int(index)] for index in order[n_validation:]]

    x_train, y_train, train_ids = _stack_curves(
        train_curves, frequency_count=int(train_frequency_count), seed=seed, role="b1-train"
    )
    x_validation, y_validation, validation_ids = _stack_curves(
        validation_curves,
        frequency_count=int(validation_frequency_count),
        seed=seed,
        role="b1-validation",
    )
    x_test, y_test, test_ids = _stack_curves(
        test_curves, frequency_count=None, seed=seed, role="b1-test"
    )
    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "task": "microstrip_resonator_b1_condition_generalization",
            "base_task": "microstrip_resonator",
            "split_mode": "microstrip_b1_condition_holdout",
            "split_strategy": "same_sample_complete_environment_condition_holdout",
            "scenario": scenario,
            "sample_id": str(sample_id),
            "holdout_temperature_c": holdout_temperature_c,
            "holdout_relative_humidity_percent": holdout_relative_humidity_percent,
            "condition_encoding": ["temperature_c", "relative_humidity_percent"],
            "train_curve_ids": [curve.curve_id for curve in train_curves],
            "validation_curve_ids": [curve.curve_id for curve in validation_curves],
            "test_curve_ids": [curve.curve_id for curve in test_curves],
            "train_conditions": _curve_conditions(train_curves),
            "validation_conditions": _curve_conditions(validation_curves),
            "test_conditions": _curve_conditions(test_curves),
            "train_point_ids": train_ids,
            "validation_point_ids": validation_ids,
            "test_point_ids": test_ids,
            "train_frequency_count_per_curve": int(train_frequency_count),
            "validation_frequency_count_per_curve": int(validation_frequency_count),
            "test_target": "complete_held_out_environment_condition_curves",
        },
    )


def make_b2_calibration_split(
    curves: Iterable[MeasuredCurve],
    *,
    sample_id: str,
    target_temperature_c: float,
    target_relative_humidity_percent: float,
    calibration_frequency_count: int,
    support_frequency_count: int,
    validation_frequency_count: int,
    seed: int,
    validation_curve_count: int = 4,
    calibration_sampling_mode: str = "random",
    calibration_excluded_band_linewidths: float = 1.0,
) -> DatasetBundle:
    """Expose sparse calibration points from the target condition and test full curves."""
    sample_curves = _same_sample_curves(curves, sample_id)
    test_curves = [
        curve for curve in sample_curves
        if float(curve.conditions["temperature_c"]) == float(target_temperature_c)
        and float(curve.conditions["relative_humidity_percent"]) == float(target_relative_humidity_percent)
    ]
    if not test_curves:
        raise ValueError(f"no target condition curves for {sample_id}")
    support_pool = [
        curve for curve in sample_curves
        if curve.curve_id not in {target.curve_id for target in test_curves}
    ]
    if len(support_pool) < 2:
        raise ValueError(f"insufficient support curves for {sample_id}")
    rng = np.random.default_rng(_seed("microstrip-b2-validation", seed, sample_id))
    order = rng.permutation(len(support_pool))
    n_validation = max(1, min(int(validation_curve_count), len(support_pool) - 1))
    validation_curves = [support_pool[int(index)] for index in order[:n_validation]]
    support_curves = [support_pool[int(index)] for index in order[n_validation:]]

    x_support, y_support, support_ids = _stack_curves(
        support_curves,
        frequency_count=int(support_frequency_count),
        seed=seed,
        role="b2-support",
    )
    if calibration_sampling_mode == "random":
        x_calibration, y_calibration, calibration_ids = _stack_curves(
            test_curves,
            frequency_count=int(calibration_frequency_count),
            seed=seed,
            role="b2-calibration",
        )
    elif calibration_sampling_mode == "missed_band":
        calibration_indices = {
            curve.curve_id: _sample_indices_outside_resonance_band(
                curve,
                int(calibration_frequency_count),
                seed,
                "b2-calibration-missed-band",
                excluded_band_linewidths=float(calibration_excluded_band_linewidths),
            )
            for curve in test_curves
        }
        x_calibration, y_calibration, calibration_ids = _stack_curves_from_indices(
            test_curves,
            calibration_indices,
        )
    else:
        raise ValueError(f"unknown B2 calibration_sampling_mode: {calibration_sampling_mode}")
    x_validation, y_validation, validation_ids = _stack_curves(
        validation_curves,
        frequency_count=int(validation_frequency_count),
        seed=seed,
        role="b2-validation",
    )
    x_test, y_test, test_ids = _stack_curves(
        test_curves, frequency_count=None, seed=seed, role="b2-test"
    )
    x_train = np.concatenate([x_support, x_calibration], axis=0)
    y_train = np.concatenate([y_support, y_calibration], axis=0)
    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "task": "microstrip_resonator_b2_low_calibration_adaptation",
            "base_task": "microstrip_resonator",
            "split_mode": "microstrip_b2_low_calibration_adaptation",
            "split_strategy": (
                "same_sample_sparse_target_condition_missed_band_calibration_complete_curve_test"
                if calibration_sampling_mode == "missed_band"
                else "same_sample_sparse_target_condition_calibration_complete_curve_test"
            ),
            "sample_id": str(sample_id),
            "target_temperature_c": float(target_temperature_c),
            "target_relative_humidity_percent": float(target_relative_humidity_percent),
            "calibration_sampling_mode": str(calibration_sampling_mode),
            "calibration_excluded_band_linewidths": float(calibration_excluded_band_linewidths),
            "condition_encoding": ["temperature_c", "relative_humidity_percent"],
            "support_curve_ids": [curve.curve_id for curve in support_curves],
            "validation_curve_ids": [curve.curve_id for curve in validation_curves],
            "test_curve_ids": [curve.curve_id for curve in test_curves],
            "support_conditions": _curve_conditions(support_curves),
            "validation_conditions": _curve_conditions(validation_curves),
            "test_conditions": _curve_conditions(test_curves),
            "train_point_ids": support_ids + calibration_ids,
            "support_point_ids": support_ids,
            "calibration_point_ids": calibration_ids,
            "validation_point_ids": validation_ids,
            "test_point_ids": test_ids,
            "support_frequency_count_per_curve": int(support_frequency_count),
            "calibration_frequency_count_per_target_curve": int(calibration_frequency_count),
            "validation_frequency_count_per_curve": int(validation_frequency_count),
            "test_target": "complete_sparse_calibrated_target_condition_curves",
        },
    )


def complex_nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction shapes differ")
    residual = np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=1)))
    centered = truth - np.mean(truth, axis=0, keepdims=True)
    denom = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
    if denom <= np.finfo(float).eps:
        denom = max(np.sqrt(np.mean(np.sum(truth ** 2, axis=1))), np.finfo(float).eps)
    return float(residual / denom)


def _as_complex(response: np.ndarray) -> np.ndarray:
    values = np.asarray(response, dtype=np.float64)
    return values[:, 0] + 1j * values[:, 1]


def extract_microstrip_events(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    *,
    center_hz: float = 2.0e9,
    search_half_width_hz: float = 2.5e8,
) -> dict:
    """Extract a notch-centered event summary around the reported 2 GHz resonator."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    complex_response = _as_complex(response)
    magnitude = np.abs(complex_response)
    phase = np.unwrap(np.angle(complex_response))
    mask = np.abs(frequency - float(center_hz)) <= float(search_half_width_hz)
    if not np.any(mask):
        mask = np.ones(len(frequency), dtype=bool)
    local_indices = np.flatnonzero(mask)
    local_magnitude = magnitude[mask]
    dip_index = int(local_indices[int(np.argmin(local_magnitude))])
    baseline = float(np.percentile(local_magnitude, 90.0))
    dip = float(magnitude[dip_index])
    half_level = dip + 0.5 * max(baseline - dip, 0.0)
    left = np.flatnonzero(magnitude[:dip_index] >= half_level)
    right = np.flatnonzero(magnitude[dip_index + 1:] >= half_level)
    linewidth_hz = None
    quality_factor = None
    if len(left) and len(right):
        left_index = int(left[-1])
        right_index = int(dip_index + 1 + right[0])
        linewidth_hz = float(frequency[right_index] - frequency[left_index])
        if linewidth_hz > 0.0:
            quality_factor = float(abs(frequency[dip_index]) / linewidth_hz)
    phase_gradient = np.abs(np.gradient(phase, frequency))
    phase_index = int(local_indices[int(np.argmax(phase_gradient[mask]))])
    peak_indices, _ = find_peaks(-local_magnitude, prominence=max(np.ptp(local_magnitude) * 0.05, 1e-12))
    return {
        "resonance_frequency_hz": float(frequency[dip_index]),
        "notch_magnitude": dip,
        "linewidth_hz": linewidth_hz,
        "quality_factor": quality_factor,
        "phase_transition_hz": float(frequency[phase_index]),
        "notch_count": int(len(peak_indices)),
    }


def local_phase_transition_hz(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    *,
    resonance_frequency_hz: float,
    linewidth_hz: float | None,
    window_linewidths: float = 1.0,
    min_half_width_hz: float = 5.0e6,
) -> float:
    """Find the largest phase slope near the main resonance instead of globally."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    phase = np.unwrap(np.angle(_as_complex(response)))
    if linewidth_hz is None or not math.isfinite(float(linewidth_hz)) or float(linewidth_hz) <= 0.0:
        half_width = float(min_half_width_hz)
    else:
        half_width = max(float(window_linewidths) * float(linewidth_hz), float(min_half_width_hz))
    mask = np.abs(frequency - float(resonance_frequency_hz)) <= half_width
    if not np.any(mask):
        mask = np.ones(len(frequency), dtype=bool)
    local_indices = np.flatnonzero(mask)
    phase_gradient = np.abs(np.gradient(phase, frequency))
    return float(frequency[int(local_indices[int(np.argmax(phase_gradient[mask]))])])


def extract_microstrip_events_local_phase(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    *,
    center_hz: float = 2.0e9,
    search_half_width_hz: float = 2.5e8,
    window_linewidths: float = 1.0,
) -> dict:
    """Extract events with phase transition constrained to the main resonance window."""
    events = extract_microstrip_events(
        frequency_hz,
        response,
        center_hz=center_hz,
        search_half_width_hz=search_half_width_hz,
    )
    events = dict(events)
    events["global_phase_transition_hz"] = events["phase_transition_hz"]
    events["phase_transition_hz"] = local_phase_transition_hz(
        frequency_hz,
        response,
        resonance_frequency_hz=events["resonance_frequency_hz"],
        linewidth_hz=events.get("linewidth_hz"),
        window_linewidths=window_linewidths,
    )
    events["phase_transition_rule"] = f"max_phase_slope_within_{window_linewidths:g}_linewidths_of_f0"
    return events


def event_identifiability_flags(
    truth_events: dict,
    *,
    high_frequency_boundary_hz: float = 2.20e9,
) -> dict:
    """Return truth-side event availability flags for guarded scientific metrics."""
    q_value = truth_events.get("quality_factor")
    f0_value = truth_events.get("resonance_frequency_hz")
    q_available = (
        q_value is not None
        and math.isfinite(float(q_value))
    )
    high_frequency_boundary = (
        f0_value is not None
        and math.isfinite(float(f0_value))
        and float(f0_value) >= float(high_frequency_boundary_hz)
    )
    return {
        "truth_q_available": bool(q_available),
        "truth_high_frequency_boundary_notch": bool(high_frequency_boundary),
        "truth_event_identifiable": bool(q_available and not high_frequency_boundary),
        "event_identifiability_rule": (
            "finite_truth_q_and_truth_f0_below_2p20ghz_boundary"
        ),
    }


def microstrip_response_metrics(
    frequency_hz: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict:
    truth_events = extract_microstrip_events(frequency_hz, truth)
    predicted_events = extract_microstrip_events(frequency_hz, prediction)
    local_truth_events = extract_microstrip_events_local_phase(frequency_hz, truth)
    local_predicted_events = extract_microstrip_events_local_phase(frequency_hz, prediction)
    event_flags = event_identifiability_flags(truth_events)
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    f0 = truth_events["resonance_frequency_hz"]
    linewidth = truth_events["linewidth_hz"]
    if linewidth is None or not math.isfinite(float(linewidth)) or linewidth <= 0.0:
        half_width = 5.0e7
    else:
        half_width = max(float(linewidth) * 2.0, 5.0e6)
    peak_mask = np.abs(frequency - f0) <= half_width
    if not np.any(peak_mask):
        peak_mask = np.ones(len(frequency), dtype=bool)
    phase_difference = np.angle(_as_complex(prediction) * np.conjugate(_as_complex(truth)))

    def abs_error(name: str) -> float | None:
        a = truth_events.get(name)
        b = predicted_events.get(name)
        if a is None or b is None:
            return None
        if not math.isfinite(float(a)) or not math.isfinite(float(b)):
            return None
        return float(abs(float(a) - float(b)))

    def local_abs_error(name: str) -> float | None:
        a = local_truth_events.get(name)
        b = local_predicted_events.get(name)
        if a is None or b is None:
            return None
        if not math.isfinite(float(a)) or not math.isfinite(float(b)):
            return None
        return float(abs(float(a) - float(b)))

    q_error = None
    if truth_events["quality_factor"] is not None and predicted_events["quality_factor"] is not None:
        denom = max(abs(float(truth_events["quality_factor"])), np.finfo(float).eps)
        q_error = float(abs(float(predicted_events["quality_factor"]) - float(truth_events["quality_factor"])) / denom)
    return {
        "complex_nrmse": complex_nrmse(truth, prediction),
        "peak_window_complex_nrmse": complex_nrmse(truth[peak_mask], prediction[peak_mask]),
        "phase_mae_rad": float(np.mean(np.abs(phase_difference))),
        "resonance_frequency_abs_error_hz": abs_error("resonance_frequency_hz"),
        "quality_factor_relative_error": q_error,
        "linewidth_abs_error_hz": abs_error("linewidth_hz"),
        "phase_transition_abs_error_hz": abs_error("phase_transition_hz"),
        "local_phase_transition_abs_error_hz": local_abs_error("phase_transition_hz"),
        "notch_magnitude_abs_error": abs_error("notch_magnitude"),
        "truth_events": truth_events,
        "predicted_events": predicted_events,
        "local_truth_events": local_truth_events,
        "local_predicted_events": local_predicted_events,
        **event_flags,
        "status": "ok",
    }


def _finite_metric(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median_metric(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.median(finite)) if finite else None


def _curve_pair_key(record: dict, curve_metric: dict) -> tuple:
    return (
        record.get("experiment"),
        record.get("scenario"),
        record.get("sample_id"),
        record.get("calibration_frequency_count"),
        record.get("calibration_sampling_mode", "random"),
        record.get("split_seed"),
        record.get("init_seed"),
        curve_metric.get("curve_id"),
    )


def microstrip_instance_effect_rows(
    records: list[dict],
    *,
    comparison_families: set[str] | None = None,
    metrics: set[str] | None = None,
) -> list[dict]:
    """Pair CFNN and baseline per test curve instance for event-level B audits."""
    comparison_families = set(comparison_families or {"MLP", "Local nested CF control"})
    metrics = set(metrics or {
        "quality_factor_relative_error",
        "phase_transition_abs_error_hz",
        "local_phase_transition_abs_error_hz",
        "resonance_frequency_abs_error_hz",
        "complex_nrmse",
        "peak_window_complex_nrmse",
    })
    by_key: dict[tuple, dict[str, dict]] = {}
    metadata_by_key: dict[tuple, dict] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        family = str(record.get("family"))
        if family != "CFNN" and family not in comparison_families:
            continue
        for curve_metric in record.get("curve_metrics", []):
            if not bool(curve_metric.get("truth_event_identifiable", True)):
                continue
            key = _curve_pair_key(record, curve_metric)
            by_key.setdefault(key, {})[family] = curve_metric
            metadata_by_key[key] = {
                "experiment": record.get("experiment"),
                "scenario": record.get("scenario"),
                "sample_id": record.get("sample_id"),
                "calibration_frequency_count": record.get("calibration_frequency_count"),
                "calibration_sampling_mode": record.get("calibration_sampling_mode", "random"),
                "split_seed": record.get("split_seed"),
                "init_seed": record.get("init_seed"),
                "curve_id": curve_metric.get("curve_id"),
                "conditions": curve_metric.get("conditions", {}),
                "truth_events": curve_metric.get("truth_events", {}),
                "truth_event_identifiable": bool(curve_metric.get("truth_event_identifiable", True)),
            }

    rows = []
    for key, families in sorted(by_key.items(), key=lambda item: tuple(str(part) for part in item[0])):
        cfnn = families.get("CFNN")
        if not cfnn:
            continue
        meta = metadata_by_key[key]
        for comparison in sorted(comparison_families):
            baseline = families.get(comparison)
            if not baseline:
                continue
            for metric in sorted(metrics):
                cfnn_value = _finite_metric(cfnn.get(metric))
                baseline_value = _finite_metric(baseline.get(metric))
                if cfnn_value is None or baseline_value in (None, 0.0):
                    continue
                effect = (float(baseline_value) - float(cfnn_value)) / float(baseline_value)
                rows.append({
                    **meta,
                    "comparison_family": comparison,
                    "metric": metric,
                    "cfnn_metric": cfnn_value,
                    "baseline_metric": baseline_value,
                    "cfnn_effect": float(effect),
                    "cfnn_wins": bool(cfnn_value < baseline_value),
                })
    return rows


def summarize_microstrip_instance_effects(rows: list[dict]) -> list[dict]:
    """Summarize paired Microstrip B instance effects by experiment/scenario/baseline/metric."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row.get("experiment"),
            row.get("scenario"),
            row.get("calibration_frequency_count"),
            row.get("calibration_sampling_mode", "random"),
            row.get("comparison_family"),
            row.get("metric"),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        effects = [float(row["cfnn_effect"]) for row in items if _finite_metric(row.get("cfnn_effect")) is not None]
        if not effects:
            continue
        experiment, scenario, calibration, calibration_sampling_mode, comparison, metric = key
        summaries.append({
            "experiment": experiment,
            "scenario": scenario,
            "calibration_frequency_count": calibration,
            "calibration_sampling_mode": calibration_sampling_mode,
            "comparison_family": comparison,
            "metric": metric,
            "pair_count": len(effects),
            "cfnn_win_count": int(sum(1 for row in items if bool(row.get("cfnn_wins")))),
            "cfnn_win_rate": float(sum(1 for row in items if bool(row.get("cfnn_wins"))) / len(effects)),
            "median_cfnn_effect": float(np.median(effects)),
            "positive_effect_count": int(sum(1 for value in effects if value > 0.0)),
            "negative_effect_count": int(sum(1 for value in effects if value < 0.0)),
        })
    return summaries


def microstrip_phase_geometry_rows(
    records: list[dict],
    *,
    far_threshold_linewidths: float = 1.0,
) -> list[dict]:
    """Deduplicate truth curves and measure whether phase transition is near f0."""
    by_curve: dict[str, dict] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        for curve_metric in record.get("curve_metrics", []):
            curve_id = curve_metric.get("curve_id")
            if not curve_id or curve_id in by_curve:
                continue
            truth = curve_metric.get("truth_events", {})
            f0 = _finite_metric(truth.get("resonance_frequency_hz"))
            phase = _finite_metric(truth.get("phase_transition_hz"))
            linewidth = _finite_metric(truth.get("linewidth_hz"))
            if f0 is None or phase is None or linewidth in (None, 0.0):
                continue
            gap = abs(float(phase) - float(f0))
            ratio = gap / float(linewidth)
            by_curve[str(curve_id)] = {
                "curve_id": str(curve_id),
                "sample_id": record.get("sample_id") or curve_metric.get("conditions", {}).get("sample_id"),
                "conditions": curve_metric.get("conditions", {}),
                "resonance_frequency_hz": f0,
                "phase_transition_hz": phase,
                "linewidth_hz": linewidth,
                "phase_f0_gap_hz": float(gap),
                "phase_f0_gap_linewidth_ratio": float(ratio),
                "phase_far_from_resonance": bool(ratio > float(far_threshold_linewidths)),
                "notch_count": truth.get("notch_count"),
                "truth_event_identifiable": bool(curve_metric.get("truth_event_identifiable", True)),
            }
    return [by_curve[key] for key in sorted(by_curve)]


def summarize_microstrip_phase_geometry(rows: list[dict]) -> dict:
    ratios = [
        float(row["phase_f0_gap_linewidth_ratio"])
        for row in rows
        if _finite_metric(row.get("phase_f0_gap_linewidth_ratio")) is not None
    ]
    far_count = int(sum(1 for row in rows if bool(row.get("phase_far_from_resonance"))))
    return {
        "curve_count": len(rows),
        "far_phase_count": far_count,
        "far_phase_fraction": float(far_count / len(rows)) if rows else None,
        "median_phase_f0_gap_linewidth_ratio": float(np.median(ratios)) if ratios else None,
        "max_phase_f0_gap_linewidth_ratio": float(np.max(ratios)) if ratios else None,
        "count_gap_gt_0p5_linewidth": int(sum(1 for value in ratios if value > 0.5)),
        "count_gap_gt_1p0_linewidth": int(sum(1 for value in ratios if value > 1.0)),
    }


def summarize_microstrip_phase_effect_instability(
    rows: list[dict],
    *,
    extreme_effect_abs: float = 1.0,
) -> list[dict]:
    """Summarize phase-transition paired effects and count large sign flips."""
    phase_rows = [
        row for row in rows
        if row.get("metric") == "phase_transition_abs_error_hz"
        and _finite_metric(row.get("cfnn_effect")) is not None
    ]
    groups: dict[tuple, list[dict]] = {}
    for row in phase_rows:
        key = (
            row.get("experiment"),
            row.get("scenario"),
            row.get("calibration_frequency_count"),
            row.get("comparison_family"),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        effects = [float(row["cfnn_effect"]) for row in items]
        wins = [bool(row.get("cfnn_wins")) for row in items]
        experiment, scenario, calibration, comparison = key
        summaries.append({
            "experiment": experiment,
            "scenario": scenario,
            "calibration_frequency_count": calibration,
            "comparison_family": comparison,
            "pair_count": len(effects),
            "cfnn_win_count": int(sum(wins)),
            "cfnn_win_rate": float(sum(wins) / len(wins)) if wins else None,
            "median_cfnn_effect": float(np.median(effects)),
            "positive_effect_count": int(sum(1 for value in effects if value > 0.0)),
            "negative_effect_count": int(sum(1 for value in effects if value < 0.0)),
            "positive_extreme_count": int(sum(1 for value in effects if value > float(extreme_effect_abs))),
            "negative_extreme_count": int(sum(1 for value in effects if value < -float(extreme_effect_abs))),
            "max_cfnn_effect": float(np.max(effects)),
            "min_cfnn_effect": float(np.min(effects)),
        })
    return summaries


def microstrip_local_phase_geometry_rows(
    curves: Iterable[MeasuredCurve],
    *,
    window_linewidths: float = 1.0,
    far_threshold_linewidths: float = 1.0,
) -> list[dict]:
    """Compare global and f0-local phase transition geometry on raw truth curves."""
    rows = []
    for curve in curves:
        global_events = extract_microstrip_events(curve.frequency, curve.response)
        local_events = extract_microstrip_events_local_phase(
            curve.frequency,
            curve.response,
            window_linewidths=window_linewidths,
        )
        f0 = _finite_metric(global_events.get("resonance_frequency_hz"))
        linewidth = _finite_metric(global_events.get("linewidth_hz"))
        global_phase = _finite_metric(global_events.get("phase_transition_hz"))
        local_phase = _finite_metric(local_events.get("phase_transition_hz"))
        if f0 is None or linewidth in (None, 0.0) or global_phase is None or local_phase is None:
            continue
        global_gap = abs(float(global_phase) - float(f0))
        local_gap = abs(float(local_phase) - float(f0))
        global_ratio = global_gap / float(linewidth)
        local_ratio = local_gap / float(linewidth)
        rows.append({
            "curve_id": curve.curve_id,
            "conditions": dict(curve.conditions),
            "resonance_frequency_hz": f0,
            "linewidth_hz": linewidth,
            "global_phase_transition_hz": global_phase,
            "local_phase_transition_hz": local_phase,
            "global_phase_f0_gap_hz": float(global_gap),
            "local_phase_f0_gap_hz": float(local_gap),
            "global_phase_f0_gap_linewidth_ratio": float(global_ratio),
            "local_phase_f0_gap_linewidth_ratio": float(local_ratio),
            "global_phase_far_from_resonance": bool(global_ratio > float(far_threshold_linewidths)),
            "local_phase_far_from_resonance": bool(local_ratio > float(far_threshold_linewidths)),
            "local_improves_gap": bool(local_gap < global_gap),
            "notch_count": global_events.get("notch_count"),
            "phase_window_linewidths": float(window_linewidths),
        })
    return sorted(rows, key=lambda row: row["curve_id"])


def summarize_microstrip_local_phase_geometry(rows: list[dict]) -> dict:
    global_ratios = [
        float(row["global_phase_f0_gap_linewidth_ratio"])
        for row in rows
        if _finite_metric(row.get("global_phase_f0_gap_linewidth_ratio")) is not None
    ]
    local_ratios = [
        float(row["local_phase_f0_gap_linewidth_ratio"])
        for row in rows
        if _finite_metric(row.get("local_phase_f0_gap_linewidth_ratio")) is not None
    ]
    improved = int(sum(1 for row in rows if bool(row.get("local_improves_gap"))))
    return {
        "curve_count": len(rows),
        "local_improves_gap_count": improved,
        "local_improves_gap_fraction": float(improved / len(rows)) if rows else None,
        "global_far_phase_count": int(sum(1 for row in rows if bool(row.get("global_phase_far_from_resonance")))),
        "local_far_phase_count": int(sum(1 for row in rows if bool(row.get("local_phase_far_from_resonance")))),
        "median_global_phase_f0_gap_linewidth_ratio": float(np.median(global_ratios)) if global_ratios else None,
        "median_local_phase_f0_gap_linewidth_ratio": float(np.median(local_ratios)) if local_ratios else None,
        "max_global_phase_f0_gap_linewidth_ratio": float(np.max(global_ratios)) if global_ratios else None,
        "max_local_phase_f0_gap_linewidth_ratio": float(np.max(local_ratios)) if local_ratios else None,
    }


def microstrip_phase_effect_geometry_strata_rows(
    effect_rows: list[dict],
    geometry_rows: list[dict],
) -> list[dict]:
    """Join phase paired effects with truth-side local/global phase geometry."""
    geometry_by_curve = {
        str(row["curve_id"]): row
        for row in geometry_rows
        if row.get("curve_id")
    }
    rows = []
    for effect in effect_rows:
        if effect.get("metric") != "phase_transition_abs_error_hz":
            continue
        curve_id = effect.get("curve_id")
        geometry = geometry_by_curve.get(str(curve_id))
        if not geometry:
            continue
        global_far = bool(geometry.get("global_phase_far_from_resonance"))
        local_far = bool(geometry.get("local_phase_far_from_resonance"))
        if global_far and not local_far:
            stratum = "global_far_local_near"
        elif global_far and local_far:
            stratum = "global_far_local_far"
        elif not global_far and local_far:
            stratum = "global_near_local_far"
        else:
            stratum = "global_near_local_near"
        rows.append({
            **effect,
            "truth_phase_geometry_stratum": stratum,
            "global_phase_far_from_resonance": global_far,
            "local_phase_far_from_resonance": local_far,
            "global_phase_f0_gap_linewidth_ratio": _finite_metric(
                geometry.get("global_phase_f0_gap_linewidth_ratio")
            ),
            "local_phase_f0_gap_linewidth_ratio": _finite_metric(
                geometry.get("local_phase_f0_gap_linewidth_ratio")
            ),
            "local_improves_gap": bool(geometry.get("local_improves_gap")),
        })
    return rows


def summarize_microstrip_phase_effect_geometry_strata(
    rows: list[dict],
    *,
    extreme_effect_abs: float = 1.0,
) -> list[dict]:
    """Summarize phase effect stability by truth-side phase geometry stratum."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        effect = _finite_metric(row.get("cfnn_effect"))
        if effect is None:
            continue
        key = (
            row.get("experiment"),
            row.get("scenario"),
            row.get("calibration_frequency_count"),
            row.get("comparison_family"),
            row.get("truth_phase_geometry_stratum"),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        effects = [float(row["cfnn_effect"]) for row in items]
        wins = [bool(row.get("cfnn_wins")) for row in items]
        global_ratios = [
            float(row["global_phase_f0_gap_linewidth_ratio"])
            for row in items
            if _finite_metric(row.get("global_phase_f0_gap_linewidth_ratio")) is not None
        ]
        local_ratios = [
            float(row["local_phase_f0_gap_linewidth_ratio"])
            for row in items
            if _finite_metric(row.get("local_phase_f0_gap_linewidth_ratio")) is not None
        ]
        experiment, scenario, calibration, comparison, stratum = key
        summaries.append({
            "experiment": experiment,
            "scenario": scenario,
            "calibration_frequency_count": calibration,
            "comparison_family": comparison,
            "truth_phase_geometry_stratum": stratum,
            "pair_count": len(effects),
            "cfnn_win_count": int(sum(wins)),
            "cfnn_win_rate": float(sum(wins) / len(wins)) if wins else None,
            "median_cfnn_effect": float(np.median(effects)),
            "positive_effect_count": int(sum(1 for value in effects if value > 0.0)),
            "negative_effect_count": int(sum(1 for value in effects if value < 0.0)),
            "positive_extreme_count": int(sum(1 for value in effects if value > float(extreme_effect_abs))),
            "negative_extreme_count": int(sum(1 for value in effects if value < -float(extreme_effect_abs))),
            "max_cfnn_effect": float(np.max(effects)),
            "min_cfnn_effect": float(np.min(effects)),
            "median_global_phase_f0_gap_linewidth_ratio": (
                float(np.median(global_ratios)) if global_ratios else None
            ),
            "median_local_phase_f0_gap_linewidth_ratio": (
                float(np.median(local_ratios)) if local_ratios else None
            ),
        })
    return summaries


def _frequency_index_by_curve(curves: Iterable[MeasuredCurve]) -> dict[str, np.ndarray]:
    return {
        str(curve.curve_id): np.asarray(curve.frequency, dtype=np.float64).reshape(-1)
        for curve in curves
    }


def _parse_frequency_point_id(point_id: str) -> tuple[str, int] | None:
    marker = ":frequency:"
    if marker not in str(point_id):
        return None
    curve_id, raw_index = str(point_id).rsplit(marker, 1)
    try:
        return curve_id, int(raw_index)
    except ValueError:
        return None


def microstrip_b2_calibration_coverage_rows(
    records: list[dict],
    curves: Iterable[MeasuredCurve],
    *,
    band_linewidths: float = 1.0,
) -> list[dict]:
    """Measure whether sparse B2 calibration points cover the target resonance band."""
    frequency_by_curve = _frequency_index_by_curve(curves)
    rows_by_key: dict[tuple, dict] = {}
    for record in records:
        if record.get("status") != "ok" or record.get("experiment") != "B2":
            continue
        calibration_ids = list((record.get("dataset") or {}).get("calibration_point_ids") or [])
        calibration_by_curve: dict[str, list[float]] = {}
        for point_id in calibration_ids:
            parsed = _parse_frequency_point_id(str(point_id))
            if parsed is None:
                continue
            curve_id, index = parsed
            frequency = frequency_by_curve.get(curve_id)
            if frequency is None or index < 0 or index >= len(frequency):
                continue
            calibration_by_curve.setdefault(curve_id, []).append(float(frequency[index]))

        for curve_metric in record.get("curve_metrics", []):
            curve_id = str(curve_metric.get("curve_id"))
            truth = curve_metric.get("truth_events", {})
            f0 = _finite_metric(truth.get("resonance_frequency_hz"))
            linewidth = _finite_metric(truth.get("linewidth_hz"))
            if f0 is None or linewidth in (None, 0.0):
                continue
            key = (
                record.get("experiment"),
                record.get("scenario"),
                record.get("sample_id"),
                record.get("calibration_frequency_count"),
                record.get("calibration_sampling_mode", "random"),
                record.get("split_seed"),
                record.get("init_seed"),
                curve_id,
            )
            if key in rows_by_key:
                continue
            calibration_frequencies = sorted(calibration_by_curve.get(curve_id, []))
            distances = [abs(float(value) - float(f0)) for value in calibration_frequencies]
            band_half_width = float(band_linewidths) * float(linewidth)
            within_one = int(sum(1 for value in distances if value <= band_half_width))
            within_half = int(sum(1 for value in distances if value <= 0.5 * float(linewidth)))
            if not calibration_frequencies:
                stratum = "no_calibration"
                nearest_distance = None
                nearest_ratio = None
            else:
                nearest_distance = float(min(distances))
                nearest_ratio = float(nearest_distance / float(linewidth))
                stratum = "resonance_band_covered" if within_one > 0 else "resonance_band_missed"
            rows_by_key[key] = {
                "experiment": record.get("experiment"),
                "scenario": record.get("scenario"),
                "sample_id": record.get("sample_id"),
                "calibration_frequency_count": record.get("calibration_frequency_count"),
                "calibration_sampling_mode": record.get("calibration_sampling_mode", "random"),
                "split_seed": record.get("split_seed"),
                "init_seed": record.get("init_seed"),
                "curve_id": curve_id,
                "conditions": curve_metric.get("conditions", {}),
                "truth_event_identifiable": bool(curve_metric.get("truth_event_identifiable", True)),
                "resonance_frequency_hz": f0,
                "linewidth_hz": linewidth,
                "calibration_point_count_for_curve": int(len(calibration_frequencies)),
                "calibration_points_within_one_linewidth": within_one,
                "calibration_points_within_half_linewidth": within_half,
                "nearest_calibration_distance_hz": nearest_distance,
                "nearest_calibration_distance_linewidth_ratio": nearest_ratio,
                "calibration_coverage_stratum": stratum,
                "calibration_band_linewidths": float(band_linewidths),
            }
    return [rows_by_key[key] for key in sorted(rows_by_key, key=lambda item: tuple(str(part) for part in item))]


def microstrip_b2_calibration_coverage_effect_rows(
    effect_rows: list[dict],
    coverage_rows: list[dict],
    *,
    metrics: set[str] | None = None,
) -> list[dict]:
    """Join B2 paired effects to sparse calibration resonance-band coverage."""
    metrics = set(metrics or {"quality_factor_relative_error"})
    coverage_by_key = {
        (
            row.get("experiment"),
            row.get("scenario"),
            row.get("sample_id"),
            row.get("calibration_frequency_count"),
            row.get("calibration_sampling_mode", "random"),
            row.get("split_seed"),
            row.get("init_seed"),
            row.get("curve_id"),
        ): row
        for row in coverage_rows
    }
    joined = []
    for effect in effect_rows:
        if effect.get("experiment") != "B2" or effect.get("metric") not in metrics:
            continue
        key = (
            effect.get("experiment"),
            effect.get("scenario"),
            effect.get("sample_id"),
            effect.get("calibration_frequency_count"),
            effect.get("calibration_sampling_mode", "random"),
            effect.get("split_seed"),
            effect.get("init_seed"),
            effect.get("curve_id"),
        )
        coverage = coverage_by_key.get(key)
        if coverage is None:
            continue
        joined.append({
            **effect,
            "calibration_sampling_mode": coverage.get("calibration_sampling_mode", effect.get("calibration_sampling_mode", "random")),
            "calibration_coverage_stratum": coverage.get("calibration_coverage_stratum"),
            "calibration_point_count_for_curve": coverage.get("calibration_point_count_for_curve"),
            "calibration_points_within_one_linewidth": coverage.get("calibration_points_within_one_linewidth"),
            "calibration_points_within_half_linewidth": coverage.get("calibration_points_within_half_linewidth"),
            "nearest_calibration_distance_linewidth_ratio": coverage.get("nearest_calibration_distance_linewidth_ratio"),
            "calibration_band_linewidths": coverage.get("calibration_band_linewidths"),
        })
    return joined


def summarize_microstrip_b2_calibration_coverage_effects(rows: list[dict]) -> list[dict]:
    """Summarize B2 Q effects by whether sparse calibration covered the resonance band."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if _finite_metric(row.get("cfnn_effect")) is None:
            continue
        key = (
            row.get("scenario"),
            row.get("calibration_frequency_count"),
            row.get("calibration_sampling_mode", "random"),
            row.get("comparison_family"),
            row.get("metric"),
            row.get("calibration_coverage_stratum"),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        effects = [float(row["cfnn_effect"]) for row in items]
        wins = [bool(row.get("cfnn_wins")) for row in items]
        distances = [
            float(row["nearest_calibration_distance_linewidth_ratio"])
            for row in items
            if _finite_metric(row.get("nearest_calibration_distance_linewidth_ratio")) is not None
        ]
        scenario, calibration, calibration_sampling_mode, comparison, metric, stratum = key
        summaries.append({
            "experiment": "B2",
            "scenario": scenario,
            "calibration_frequency_count": calibration,
            "calibration_sampling_mode": calibration_sampling_mode,
            "comparison_family": comparison,
            "metric": metric,
            "calibration_coverage_stratum": stratum,
            "pair_count": len(effects),
            "cfnn_win_count": int(sum(wins)),
            "cfnn_win_rate": float(sum(wins) / len(wins)) if wins else None,
            "median_cfnn_effect": float(np.median(effects)),
            "positive_effect_count": int(sum(1 for value in effects if value > 0.0)),
            "negative_effect_count": int(sum(1 for value in effects if value < 0.0)),
            "median_nearest_calibration_distance_linewidth_ratio": (
                float(np.median(distances)) if distances else None
            ),
            "median_points_within_one_linewidth": _median_metric([
                row.get("calibration_points_within_one_linewidth") for row in items
            ]),
            "median_points_within_half_linewidth": _median_metric([
                row.get("calibration_points_within_half_linewidth") for row in items
            ]),
        })
    return summaries


def select_microstrip_b2_guardrail_replay_cases(
    effect_rows: list[dict],
    *,
    max_cases_per_failure_type: int = 2,
) -> list[dict]:
    """Select B2 missed-band cases where Q improves but f0/local phase guardrails fail."""
    metrics = {
        "quality_factor_relative_error",
        "resonance_frequency_abs_error_hz",
        "local_phase_transition_abs_error_hz",
    }
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in effect_rows:
        if row.get("experiment") != "B2":
            continue
        if row.get("calibration_sampling_mode", "random") != "missed_band":
            continue
        metric = row.get("metric")
        if metric not in metrics:
            continue
        key = (
            row.get("scenario"),
            row.get("sample_id"),
            row.get("calibration_frequency_count"),
            row.get("calibration_sampling_mode", "random"),
            row.get("comparison_family"),
            row.get("split_seed"),
            row.get("init_seed"),
            row.get("curve_id"),
        )
        grouped.setdefault(key, {})[str(metric)] = row

    candidates: dict[str, list[dict]] = {
        "f0_guardrail_failure": [],
        "local_phase_guardrail_failure": [],
    }
    guardrail_specs = [
        ("f0_guardrail_failure", "resonance_frequency_abs_error_hz"),
        ("local_phase_guardrail_failure", "local_phase_transition_abs_error_hz"),
    ]
    for rows_by_metric in grouped.values():
        q_row = rows_by_metric.get("quality_factor_relative_error")
        if q_row is None:
            continue
        q_effect = _finite_metric(q_row.get("cfnn_effect"))
        if q_effect is None or q_effect <= 0.0:
            continue
        for failure_type, metric in guardrail_specs:
            guardrail_row = rows_by_metric.get(metric)
            if guardrail_row is None:
                continue
            guardrail_effect = _finite_metric(guardrail_row.get("cfnn_effect"))
            if guardrail_effect is None or guardrail_effect >= 0.0:
                continue
            candidates[failure_type].append({
                "failure_type": failure_type,
                "experiment": q_row.get("experiment"),
                "scenario": q_row.get("scenario"),
                "sample_id": q_row.get("sample_id"),
                "curve_id": q_row.get("curve_id"),
                "calibration_frequency_count": q_row.get("calibration_frequency_count"),
                "calibration_sampling_mode": q_row.get("calibration_sampling_mode", "random"),
                "comparison_family": q_row.get("comparison_family"),
                "split_seed": q_row.get("split_seed"),
                "init_seed": q_row.get("init_seed"),
                "quality_factor_effect": q_effect,
                "guardrail_metric": metric,
                "guardrail_effect": guardrail_effect,
                "selection_score": float(abs(guardrail_effect) + 1e-3 * max(q_effect, 0.0)),
            })

    selected = []
    for failure_type in sorted(candidates):
        ordered = sorted(
            candidates[failure_type],
            key=lambda row: (
                -float(row["selection_score"]),
                str(row.get("sample_id")),
                str(row.get("curve_id")),
                str(row.get("comparison_family")),
                str(row.get("calibration_frequency_count")),
            ),
        )
        selected.extend(ordered[: int(max_cases_per_failure_type)])
    return selected


def microstrip_truth_notch_descriptor_rows(
    curves: Iterable[MeasuredCurve],
    *,
    center_hz: float = 2.0e9,
    search_half_width_hz: float = 2.5e8,
    prominence_fraction: float = 0.05,
    same_notch_linewidths: float = 0.5,
    boundary_margin_linewidths: float = 1.0,
) -> list[dict]:
    """Describe truth-side notch geometry that can make B2 f0 extraction ambiguous."""
    rows = []
    lower_edge = float(center_hz) - float(search_half_width_hz)
    upper_edge = float(center_hz) + float(search_half_width_hz)
    for curve in curves:
        frequency = np.asarray(curve.frequency, dtype=np.float64).reshape(-1)
        response = np.asarray(curve.response, dtype=np.float64)
        events = extract_microstrip_events(
            frequency,
            response,
            center_hz=center_hz,
            search_half_width_hz=search_half_width_hz,
        )
        f0 = _finite_metric(events.get("resonance_frequency_hz"))
        linewidth = _finite_metric(events.get("linewidth_hz"))
        if f0 is None or linewidth in (None, 0.0):
            continue
        magnitude = np.abs(_as_complex(response))
        mask = np.abs(frequency - float(center_hz)) <= float(search_half_width_hz)
        if not np.any(mask):
            mask = np.ones(len(frequency), dtype=bool)
        local_indices = np.flatnonzero(mask)
        local_magnitude = magnitude[mask]
        prominence = max(float(np.ptp(local_magnitude)) * float(prominence_fraction), 1e-12)
        peak_indices, properties = find_peaks(-local_magnitude, prominence=prominence)
        notch_indices = [int(local_indices[int(index)]) for index in peak_indices]
        if int(np.argmin(np.abs(frequency - float(f0)))) not in notch_indices:
            notch_indices.append(int(np.argmin(np.abs(frequency - float(f0)))))
        notch_indices = sorted(set(notch_indices), key=lambda index: float(frequency[index]))
        notch_frequencies = [float(frequency[index]) for index in notch_indices]
        notch_magnitudes = [float(magnitude[index]) for index in notch_indices]
        same_notch_half_width = float(same_notch_linewidths) * float(linewidth)
        competing = [
            abs(float(value) - float(f0))
            for value in notch_frequencies
            if abs(float(value) - float(f0)) > same_notch_half_width
        ]
        nearest_competing = float(min(competing)) if competing else None
        nearest_edge_distance = float(min(abs(float(f0) - lower_edge), abs(upper_edge - float(f0))))
        edge_ratio = float(nearest_edge_distance / float(linewidth))
        near_boundary = nearest_edge_distance <= float(boundary_margin_linewidths) * float(linewidth)
        multi_notch = len(competing) > 0
        if edge_ratio <= 0.65:
            boundary_stratum = "edge_proximal"
        elif edge_ratio <= float(boundary_margin_linewidths):
            boundary_stratum = "edge_intermediate"
        else:
            boundary_stratum = "edge_interior"
        if multi_notch or near_boundary:
            stratum = "multi_notch_or_boundary"
        else:
            stratum = "single_notch_interior"
        rows.append({
            "curve_id": curve.curve_id,
            "sample_id": curve.conditions.get("sample_id"),
            "conditions": dict(curve.conditions),
            "resonance_frequency_hz": f0,
            "linewidth_hz": linewidth,
            "quality_factor": _finite_metric(events.get("quality_factor")),
            "notch_count": int(len(notch_frequencies)),
            "notch_frequencies_hz": notch_frequencies,
            "notch_magnitudes": notch_magnitudes,
            "competing_notch_count": int(len(competing)),
            "nearest_competing_notch_gap_hz": nearest_competing,
            "nearest_competing_notch_gap_linewidth_ratio": (
                float(nearest_competing / float(linewidth)) if nearest_competing is not None else None
            ),
            "nearest_search_edge_distance_hz": nearest_edge_distance,
            "nearest_search_edge_distance_linewidth_ratio": edge_ratio,
            "truth_f0_near_search_boundary": bool(near_boundary),
            "truth_multi_notch": bool(multi_notch),
            "notch_geometry_stratum": stratum,
            "boundary_proximity_stratum": boundary_stratum,
            "search_window_lower_hz": lower_edge,
            "search_window_upper_hz": upper_edge,
            "notch_descriptor_rule": (
                f"multi_prominent_notch_or_f0_within_{boundary_margin_linewidths:g}_linewidths_of_search_edge"
            ),
        })
    return sorted(rows, key=lambda row: str(row["curve_id"]))


def microstrip_b2_f0_guardrail_descriptor_rows(
    effect_rows: list[dict],
    descriptor_rows: list[dict],
    coverage_rows: list[dict] | None = None,
) -> list[dict]:
    """Join missed-band B2 Q/f0 paired effects with truth-side notch descriptors."""
    descriptors_by_curve = {
        str(row["curve_id"]): row
        for row in descriptor_rows
        if row.get("curve_id")
    }
    coverage_by_key = {
        (
            row.get("experiment"),
            row.get("scenario"),
            row.get("sample_id"),
            row.get("calibration_frequency_count"),
            row.get("calibration_sampling_mode", "random"),
            row.get("split_seed"),
            row.get("init_seed"),
            row.get("curve_id"),
        ): row
        for row in (coverage_rows or [])
    }
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in effect_rows:
        if row.get("experiment") != "B2":
            continue
        if row.get("calibration_sampling_mode", "random") != "missed_band":
            continue
        if row.get("metric") not in {"quality_factor_relative_error", "resonance_frequency_abs_error_hz"}:
            continue
        key = (
            row.get("experiment"),
            row.get("scenario"),
            row.get("sample_id"),
            row.get("calibration_frequency_count"),
            row.get("calibration_sampling_mode", "random"),
            row.get("comparison_family"),
            row.get("split_seed"),
            row.get("init_seed"),
            row.get("curve_id"),
        )
        grouped.setdefault(key, {})[str(row.get("metric"))] = row

    joined = []
    for key, rows_by_metric in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        q_row = rows_by_metric.get("quality_factor_relative_error")
        f0_row = rows_by_metric.get("resonance_frequency_abs_error_hz")
        if q_row is None or f0_row is None:
            continue
        q_effect = _finite_metric(q_row.get("cfnn_effect"))
        f0_effect = _finite_metric(f0_row.get("cfnn_effect"))
        if q_effect is None or f0_effect is None:
            continue
        curve_id = str(q_row.get("curve_id"))
        descriptor = descriptors_by_curve.get(curve_id)
        if descriptor is None:
            continue
        coverage_key = (
            q_row.get("experiment"),
            q_row.get("scenario"),
            q_row.get("sample_id"),
            q_row.get("calibration_frequency_count"),
            q_row.get("calibration_sampling_mode", "random"),
            q_row.get("split_seed"),
            q_row.get("init_seed"),
            q_row.get("curve_id"),
        )
        coverage = coverage_by_key.get(coverage_key, {})
        joined.append({
            "experiment": q_row.get("experiment"),
            "scenario": q_row.get("scenario"),
            "sample_id": q_row.get("sample_id"),
            "calibration_frequency_count": q_row.get("calibration_frequency_count"),
            "calibration_sampling_mode": q_row.get("calibration_sampling_mode", "random"),
            "comparison_family": q_row.get("comparison_family"),
            "split_seed": q_row.get("split_seed"),
            "init_seed": q_row.get("init_seed"),
            "curve_id": curve_id,
            "quality_factor_effect": q_effect,
            "f0_effect": f0_effect,
            "q_positive_f0_negative": bool(q_effect > 0.0 and f0_effect < 0.0),
            "f0_cfnn_wins": bool(f0_row.get("cfnn_wins")),
            "quality_factor_cfnn_wins": bool(q_row.get("cfnn_wins")),
            "calibration_coverage_stratum": coverage.get("calibration_coverage_stratum"),
            "nearest_calibration_distance_linewidth_ratio": coverage.get(
                "nearest_calibration_distance_linewidth_ratio"
            ),
            "calibration_points_within_one_linewidth": coverage.get(
                "calibration_points_within_one_linewidth"
            ),
            **{
                name: descriptor.get(name)
                for name in (
                    "resonance_frequency_hz",
                    "linewidth_hz",
                    "quality_factor",
                    "notch_count",
                    "competing_notch_count",
                    "nearest_competing_notch_gap_linewidth_ratio",
                    "nearest_search_edge_distance_linewidth_ratio",
                    "truth_f0_near_search_boundary",
                    "truth_multi_notch",
                    "notch_geometry_stratum",
                    "boundary_proximity_stratum",
                )
            },
        })
    return joined


def summarize_microstrip_b2_f0_guardrail_descriptors(rows: list[dict]) -> list[dict]:
    """Summarize missed-band f0 failures by truth-side notch geometry stratum."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row.get("comparison_family"),
            row.get("calibration_frequency_count"),
            row.get("notch_geometry_stratum"),
            row.get("boundary_proximity_stratum"),
            row.get("calibration_coverage_stratum"),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        q_effects = [
            float(row["quality_factor_effect"]) for row in items
            if _finite_metric(row.get("quality_factor_effect")) is not None
        ]
        f0_effects = [
            float(row["f0_effect"]) for row in items
            if _finite_metric(row.get("f0_effect")) is not None
        ]
        if not f0_effects:
            continue
        comparison, calibration, stratum, boundary_stratum, coverage = key
        qpos_f0neg = [row for row in items if bool(row.get("q_positive_f0_negative"))]
        summaries.append({
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "comparison_family": comparison,
            "calibration_frequency_count": calibration,
            "notch_geometry_stratum": stratum,
            "boundary_proximity_stratum": boundary_stratum,
            "calibration_coverage_stratum": coverage,
            "pair_count": len(f0_effects),
            "f0_negative_count": int(sum(1 for value in f0_effects if value < 0.0)),
            "f0_negative_rate": float(sum(1 for value in f0_effects if value < 0.0) / len(f0_effects)),
            "q_positive_f0_negative_count": int(len(qpos_f0neg)),
            "q_positive_f0_negative_rate": float(len(qpos_f0neg) / len(f0_effects)),
            "median_f0_effect": float(np.median(f0_effects)),
            "median_quality_factor_effect": float(np.median(q_effects)) if q_effects else None,
        })
    return summaries


def microstrip_b2_confirmatory_guardrail_stratum_effect_rows(
    effect_rows: list[dict],
    descriptor_rows: list[dict],
    *,
    metrics: set[str] | None = None,
) -> list[dict]:
    """Join B2 missed-band paired effects to pre-registered guardrail strata."""
    metrics = set(metrics or {
        "quality_factor_relative_error",
        "resonance_frequency_abs_error_hz",
        "local_phase_transition_abs_error_hz",
        "complex_nrmse",
    })
    descriptor_by_key = {
        (
            row.get("experiment"),
            row.get("scenario"),
            row.get("sample_id"),
            row.get("calibration_frequency_count"),
            row.get("calibration_sampling_mode", "random"),
            row.get("comparison_family"),
            row.get("split_seed"),
            row.get("init_seed"),
            row.get("curve_id"),
        ): row
        for row in descriptor_rows
        if row.get("calibration_sampling_mode", "random") == "missed_band"
        and row.get("calibration_coverage_stratum") == "resonance_band_missed"
        and row.get("boundary_proximity_stratum") in {"edge_proximal", "edge_intermediate"}
    }
    rows = []
    for effect in effect_rows:
        if effect.get("experiment") != "B2":
            continue
        if effect.get("calibration_sampling_mode", "random") != "missed_band":
            continue
        if effect.get("metric") not in metrics:
            continue
        effect_value = _finite_metric(effect.get("cfnn_effect"))
        if effect_value is None:
            continue
        key = (
            effect.get("experiment"),
            effect.get("scenario"),
            effect.get("sample_id"),
            effect.get("calibration_frequency_count"),
            effect.get("calibration_sampling_mode", "random"),
            effect.get("comparison_family"),
            effect.get("split_seed"),
            effect.get("init_seed"),
            effect.get("curve_id"),
        )
        descriptor = descriptor_by_key.get(key)
        if descriptor is None:
            continue
        boundary = str(descriptor.get("boundary_proximity_stratum"))
        rows.append({
            **effect,
            "cfnn_effect": effect_value,
            "stratum_id": f"{boundary}_missed_band",
            "stratum_role": (
                "primary_guardrail_stratum"
                if boundary == "edge_proximal"
                else "contrast_stratum"
            ),
            "boundary_proximity_stratum": boundary,
            "calibration_coverage_stratum": descriptor.get("calibration_coverage_stratum"),
            "notch_geometry_stratum": descriptor.get("notch_geometry_stratum"),
            "q_positive_f0_negative": bool(descriptor.get("q_positive_f0_negative")),
            "descriptor_quality_factor_effect": _finite_metric(descriptor.get("quality_factor_effect")),
            "descriptor_f0_effect": _finite_metric(descriptor.get("f0_effect")),
            "nearest_search_edge_distance_linewidth_ratio": descriptor.get(
                "nearest_search_edge_distance_linewidth_ratio"
            ),
            "nearest_calibration_distance_linewidth_ratio": descriptor.get(
                "nearest_calibration_distance_linewidth_ratio"
            ),
        })
    return rows


def summarize_microstrip_b2_confirmatory_guardrail_stratum_effects(rows: list[dict]) -> dict:
    """Summarize B2 confirmatory guardrail strata by metric and claim gate."""
    metric_groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if _finite_metric(row.get("cfnn_effect")) is None:
            continue
        key = (
            row.get("stratum_id"),
            row.get("stratum_role"),
            row.get("comparison_family"),
            row.get("calibration_frequency_count"),
            row.get("metric"),
        )
        metric_groups.setdefault(key, []).append(row)

    metric_rows = []
    for key, items in sorted(metric_groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        effects = [float(row["cfnn_effect"]) for row in items]
        wins = [bool(row.get("cfnn_wins")) for row in items]
        stratum_id, role, comparison, calibration, metric = key
        metric_rows.append({
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "stratum_id": stratum_id,
            "stratum_role": role,
            "comparison_family": comparison,
            "calibration_frequency_count": calibration,
            "metric": metric,
            "pair_count": len(effects),
            "cfnn_win_count": int(sum(wins)),
            "cfnn_win_rate": float(sum(wins) / len(wins)) if wins else None,
            "median_cfnn_effect": float(np.median(effects)),
            "positive_effect_count": int(sum(1 for value in effects if value > 0.0)),
            "negative_effect_count": int(sum(1 for value in effects if value < 0.0)),
        })

    gate_groups: dict[tuple, list[dict]] = {}
    for row in metric_rows:
        key = (
            row.get("stratum_id"),
            row.get("stratum_role"),
            row.get("comparison_family"),
            row.get("calibration_frequency_count"),
        )
        gate_groups.setdefault(key, []).append(row)

    def metric_effect(items: list[dict], metric: str) -> float | None:
        values = [
            float(row["median_cfnn_effect"])
            for row in items
            if row.get("metric") == metric
            and _finite_metric(row.get("median_cfnn_effect")) is not None
        ]
        return float(np.median(values)) if values else None

    gate_rows = []
    for key, items in sorted(gate_groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        stratum_id, role, comparison, calibration = key
        q_effect = metric_effect(items, "quality_factor_relative_error")
        f0_effect = metric_effect(items, "resonance_frequency_abs_error_hz")
        phase_effect = metric_effect(items, "local_phase_transition_abs_error_hz")
        global_effect = metric_effect(items, "complex_nrmse")
        q_positive = q_effect is not None and q_effect > 0.0
        f0_pass = f0_effect is not None and f0_effect >= 0.0
        phase_pass = phase_effect is not None and phase_effect >= 0.0
        global_pass = global_effect is not None and global_effect >= 0.0
        all_guardrails_pass = bool(f0_pass and phase_pass and global_pass)
        if q_positive and all_guardrails_pass:
            status = "passes_q_with_event_guardrails"
        elif q_positive:
            status = "localized_q_signal_blocked_by_guardrail"
        else:
            status = "no_confirmed_q_signal"
        gate_rows.append({
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "stratum_id": stratum_id,
            "stratum_role": role,
            "comparison_family": comparison,
            "calibration_frequency_count": calibration,
            "quality_factor_effect": q_effect,
            "f0_effect": f0_effect,
            "local_phase_effect": phase_effect,
            "global_complex_nrmse_effect": global_effect,
            "q_gain_positive": bool(q_positive),
            "f0_guardrail_pass": bool(f0_pass),
            "phase_guardrail_pass": bool(phase_pass),
            "global_guardrail_pass": bool(global_pass),
            "all_guardrails_pass": all_guardrails_pass,
            "claim_gate_status": status,
        })

    return {
        "analysis_type": "microstrip_b2_confirmatory_guardrail_stratum_effects",
        "primary_claim_gate": "q_gain_requires_f0_and_phase_guardrails",
        "metric_summary_rows": metric_rows,
        "claim_gate_rows": gate_rows,
    }
