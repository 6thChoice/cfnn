"""Leakage-safe feature windows and complex response metrics.

Window discovery deliberately accepts only observed roles and generator metadata.
The hidden response is used by :func:`complex_window_metrics` only after a window
has been frozen, never by :func:`derive_window_spec`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter


_TASKS = {
    "controlled_fano",
    "microwave_fano",
    "microstrip_resonator",
    "aluminium_frf",
    "figshare_battery_eis",
    "hybrid_supercapacitor_eis",
}
_EIS_TASKS = {"figshare_battery_eis", "hybrid_supercapacitor_eis"}
_DIP_TASKS = {"microwave_fano", "microstrip_resonator"}
_AXIS_SPACES = {"linear", "log10_frequency"}
_SOURCE_ROLES = {"train_validation", "generator_metadata", "no_identifiable_feature"}
_STATUSES = {"ok", "no_identifiable_feature"}
_EPS = np.finfo(float).eps


def _freeze_detector(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_detector(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_detector(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_detector(item) for item in value)
    return value


def _serializable_detector(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _serializable_detector(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable_detector(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_serializable_detector(item) for item in value]
        return sorted(items, key=repr)
    return value


@dataclass(frozen=True)
class WindowSpec:
    """Frozen, serializable description of a task's resonance intervals."""

    task: str
    axis_space: str
    intervals: tuple[tuple[float, float], ...]
    source_role: str
    status: str
    detector: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.task not in _TASKS:
            raise ValueError(f"unsupported task: {self.task!r}")
        if self.axis_space not in _AXIS_SPACES:
            raise ValueError(f"unsupported axis space: {self.axis_space!r}")
        if self.source_role not in _SOURCE_ROLES:
            raise ValueError(f"unsupported window source role: {self.source_role!r}")
        if self.status not in _STATUSES:
            raise ValueError(f"unsupported window status: {self.status!r}")
        if not isinstance(self.detector, Mapping):
            raise ValueError("detector metadata must be a mapping")

        normalized = []
        for interval in self.intervals:
            if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)):
                raise ValueError("each interval must contain two numeric endpoints")
            if len(interval) != 2:
                raise ValueError("each interval must contain two numeric endpoints")
            left, right = float(interval[0]), float(interval[1])
            if not np.isfinite(left) or not np.isfinite(right) or not left < right:
                raise ValueError("window interval endpoints must be finite and increasing")
            normalized.append((left, right))

        merged = []
        for left, right in sorted(normalized):
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))
        if self.status == "no_identifiable_feature" and merged:
            raise ValueError("a no-identifiable-feature window cannot contain intervals")
        if self.source_role == "no_identifiable_feature" and self.status != "no_identifiable_feature":
            raise ValueError("no-identifiable-feature source role requires matching status")
        object.__setattr__(self, "intervals", tuple(merged))
        object.__setattr__(self, "detector", _freeze_detector(self.detector))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible copy of this immutable window specification."""
        return {
            "task": self.task,
            "axis_space": self.axis_space,
            "intervals": [list(interval) for interval in self.intervals],
            "source_role": self.source_role,
            "status": self.status,
            "detector": _serializable_detector(self.detector),
        }


def _rule(rules: Mapping[str, Any], name: str, default: Any) -> Any:
    value = rules.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"rule {name!r} must be numeric, not boolean")
    return value


def _validate_rules(rules: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(rules, Mapping):
        raise ValueError("rules must be a mapping")
    smoothing = rules.get("smoothing", {})
    if not isinstance(smoothing, Mapping):
        raise ValueError("smoothing rules must be a mapping")
    method = smoothing.get("method", "savgol")
    if method not in {"savgol", "none"}:
        raise ValueError(f"unsupported smoothing method: {method!r}")
    window_length = int(smoothing.get("window_length", 5))
    polyorder = int(smoothing.get("polyorder", 2))
    if window_length < 3 or window_length % 2 == 0 or polyorder < 0 or polyorder >= window_length:
        raise ValueError("savgol window_length must be odd and exceed polyorder")
    prominence_fraction = float(_rule(rules, "prominence_fraction", 0.05))
    minimum_distance = int(_rule(rules, "minimum_distance_observed_spacings", 2))
    max_features = int(_rule(rules, "max_features", 3))
    minimum_half_width = float(_rule(rules, "minimum_half_width_observed_spacings", 2))
    if not 0.0 < prominence_fraction:
        raise ValueError("prominence_fraction must be positive")
    if minimum_distance < 1 or max_features < 1 or minimum_half_width <= 0.0:
        raise ValueError("detector spacing and feature limits must be positive")
    return {
        "smoothing_method": method,
        "window_length": window_length,
        "polyorder": polyorder,
        "prominence_fraction": prominence_fraction,
        "minimum_distance_observed_spacings": minimum_distance,
        "max_features": max_features,
        "minimum_half_width_observed_spacings": minimum_half_width,
    }


def _observed_x(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty vector or (n, 1) array")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _complex_response(values: Any, name: str, *, require_finite: bool = True) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1 and np.iscomplexobj(array):
        response = np.asarray(array, dtype=np.complex128)
    elif array.ndim == 2 and array.shape[1] == 2:
        if np.iscomplexobj(array):
            raise ValueError(f"{name} two-column responses must be real-valued")
        real = np.asarray(array[:, 0], dtype=np.float64)
        imaginary = np.asarray(array[:, 1], dtype=np.float64)
        response = real + 1j * imaginary
    else:
        raise ValueError(f"{name} must have shape (n, 2) or be a complex vector")
    if len(response) == 0:
        raise ValueError(f"{name} must contain finite values")
    if require_finite and not np.all(np.isfinite(response)):
        raise ValueError(f"{name} must contain finite values")
    return response


def _axis_values(task: str, frequency: np.ndarray) -> tuple[str, np.ndarray]:
    if task in _EIS_TASKS:
        if np.any(frequency <= 0.0):
            raise ValueError("EIS frequencies must be positive")
        return "log10_frequency", np.log10(frequency)
    return "linear", frequency.copy()


def _combined_observations(
    task: str,
    x_train: Any,
    y_train: Any,
    x_validation: Any,
    y_validation: Any,
) -> tuple[str, np.ndarray, np.ndarray]:
    train_x = _observed_x(x_train, "x_train")
    validation_x = _observed_x(x_validation, "x_validation")
    train_y = _complex_response(y_train, "y_train")
    validation_y = _complex_response(y_validation, "y_validation")
    if len(train_x) != len(train_y) or len(validation_x) != len(validation_y):
        raise ValueError("each frequency and response role must have matching lengths")
    frequency = np.concatenate((train_x, validation_x))
    response = np.concatenate((train_y, validation_y))
    axis_space, axis = _axis_values(task, frequency)
    order = np.argsort(axis, kind="mergesort")
    axis = axis[order]
    response = response[order]
    if len(axis) < 2 or np.any(np.diff(axis) <= 0.0):
        raise ValueError("observed frequencies must be distinct after combining roles")
    return axis_space, axis, response


def _robust_span(values: np.ndarray, *, central: bool = False) -> float:
    finite = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(finite, [45.0, 55.0] if central else [5.0, 95.0])
    span = float(high - low)
    if span <= _EPS:
        span = float(np.ptp(finite))
    return span


def _smooth(signal: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    if config["smoothing_method"] == "none" or len(signal) < config["window_length"]:
        return signal.copy()
    return np.asarray(
        savgol_filter(signal, config["window_length"], config["polyorder"], mode="interp"),
        dtype=np.float64,
    )


def _signal_candidates(task: str, response: np.ndarray) -> list[tuple[str, np.ndarray]]:
    magnitude = np.abs(response)
    if task in _DIP_TASKS:
        return [("magnitude", magnitude), ("inverse_magnitude", -magnitude)]
    if task in _EIS_TASKS:
        return [("negative_imaginary_impedance", -response.imag)]
    return [("magnitude", magnitude)]


def _feature_candidates(
    task: str,
    axis: np.ndarray,
    response: np.ndarray,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    spacing = float(np.median(np.diff(axis)))
    distance = max(1, int(config["minimum_distance_observed_spacings"]))
    candidates: list[dict[str, Any]] = []
    for signal_name, raw_signal in _signal_candidates(task, response):
        signal = _smooth(np.asarray(raw_signal, dtype=np.float64), config)
        span = _robust_span(signal, central=signal_name == "inverse_magnitude")
        if span <= _EPS:
            continue
        peaks, properties = find_peaks(
            signal,
            prominence=max(config["prominence_fraction"] * span, _EPS),
            distance=distance,
        )
        prominences = properties.get("prominences", np.zeros(len(peaks)))
        if not len(peaks):
            continue
        widths = peak_widths(signal, peaks, rel_height=0.5)[0]
        for peak, prominence, width in zip(peaks, prominences, widths):
            half_width = max(
                config["minimum_half_width_observed_spacings"] * spacing,
                0.5 * float(width) * spacing,
            )
            candidates.append({
                "center": float(axis[peak]),
                "half_width": float(half_width),
                "prominence": float(prominence),
                "signal": signal_name,
                "index": int(peak),
            })
    candidates.sort(key=lambda item: (-item["prominence"], item["center"], item["index"], item["signal"]))
    selected = []
    for candidate in candidates:
        if any(abs(candidate["center"] - item["center"]) < spacing for item in selected):
            continue
        selected.append(candidate)
        if len(selected) == config["max_features"]:
            break
    return sorted(selected, key=lambda item: (item["center"], item["signal"]))


def _metadata_candidates(metadata: Mapping[str, Any]) -> list[tuple[float, float]]:
    sources = [metadata]
    for key in ("generator_metadata", "generator", "parameters"):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            sources.append(value)

    centers: Any = None
    widths: Any = None
    for source in sources:
        if centers is None:
            for key in ("feature_centers", "resonance_centers", "generator_centers", "centers"):
                if key in source:
                    centers = source[key]
                    break
        if widths is None:
            for key in ("feature_widths", "resonance_widths", "generator_widths", "widths"):
                if key in source:
                    widths = source[key]
                    break

    parameter_source = next(
        (source for source in sources if "width" in source or "center" in source or "resonance_center" in source),
        None,
    )
    if centers is None and parameter_source is not None:
        center = parameter_source.get("resonance_center", parameter_source.get("center", 0.5))
        separation = parameter_source.get("separation", 0.0)
        if float(separation) == 0.0:
            centers = [center]
        else:
            centers = [float(center) - float(separation) / 2.0, float(center) + float(separation) / 2.0]
    if centers is None:
        return []
    if np.isscalar(centers):
        centers = [centers]
    if widths is None:
        widths = [0.0] * len(centers)
    elif np.isscalar(widths):
        widths = [widths] * len(centers)
    centers = list(centers)
    widths = list(widths)
    if len(widths) != len(centers):
        raise ValueError("generator feature centers and widths must have matching lengths")
    result = []
    for center, width in zip(centers, widths):
        center = float(center)
        width = float(width)
        if not np.isfinite(center) or not np.isfinite(width) or width < 0.0:
            raise ValueError("generator feature centers and widths must be finite and non-negative")
        result.append((center, width))
    return result


def _no_feature_spec(
    task: str,
    axis_space: str,
    detector: Mapping[str, Any],
) -> WindowSpec:
    return WindowSpec(task, axis_space, (), "no_identifiable_feature", "no_identifiable_feature", detector)


def derive_window_spec(
    task: str,
    x_train: Any,
    y_train: Any,
    x_validation: Any,
    y_validation: Any,
    metadata: Mapping[str, Any],
    rules: Mapping[str, Any],
) -> WindowSpec:
    """Derive a window from train/validation observations and allowed metadata only."""
    if task not in _TASKS:
        raise ValueError(f"unsupported task: {task!r}")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    config = _validate_rules(rules)
    axis_space, axis, response = _combined_observations(
        task, x_train, y_train, x_validation, y_validation
    )
    detector = {
        "method": "metadata" if task == "controlled_fano" else "find_peaks",
        "signal": "generator_metadata" if task == "controlled_fano" else _signal_candidates(task, response)[0][0],
        "prominence_fraction": config["prominence_fraction"],
        "minimum_distance_observed_spacings": config["minimum_distance_observed_spacings"],
        "minimum_half_width_observed_spacings": config["minimum_half_width_observed_spacings"],
        "max_features": config["max_features"],
        "smoothing": {
            "method": config["smoothing_method"],
            "window_length": config["window_length"],
            "polyorder": config["polyorder"],
        },
    }

    if task == "controlled_fano":
        metadata_features = _metadata_candidates(metadata)
        if not metadata_features:
            return _no_feature_spec(task, axis_space, detector)
        spacing = float(np.median(np.diff(axis)))
        minimum_half_width = config["minimum_half_width_observed_spacings"] * spacing
        intervals = [
            (center - max(width, minimum_half_width), center + max(width, minimum_half_width))
            for center, width in metadata_features[: config["max_features"]]
        ]
        detector["feature_count"] = len(intervals)
        detector["features"] = [
            {"center": float(center), "half_width": float(max(width, minimum_half_width)), "source": "metadata"}
            for center, width in metadata_features[: config["max_features"]]
        ]
        return WindowSpec(task, axis_space, tuple(intervals), "generator_metadata", "ok", detector)

    features = _feature_candidates(task, axis, response, config)
    if not features:
        detector["feature_count"] = 0
        return _no_feature_spec(task, axis_space, detector)
    lower, upper = float(axis[0]), float(axis[-1])
    intervals = [
        (max(lower, item["center"] - item["half_width"]), min(upper, item["center"] + item["half_width"]))
        for item in features
    ]
    intervals = [interval for interval in intervals if interval[0] < interval[1]]
    if not intervals:
        detector["feature_count"] = 0
        return _no_feature_spec(task, axis_space, detector)
    detector["feature_count"] = len(features)
    detector["signals"] = sorted({item["signal"] for item in features})
    detector["features"] = [
        {
            "center": item["center"],
            "half_width": item["half_width"],
            "prominence": item["prominence"],
            "signal": item["signal"],
        }
        for item in features
    ]
    return WindowSpec(task, axis_space, tuple(intervals), "train_validation", "ok", detector)


def _metric_axis(window_spec: WindowSpec, frequency: Any) -> tuple[np.ndarray, np.ndarray]:
    values = _observed_x(frequency, "frequency")
    if window_spec.axis_space == "log10_frequency":
        if np.any(values <= 0.0):
            raise ValueError("log10-frequency metrics require positive frequencies")
        axis = np.log10(values)
    else:
        axis = values.copy()
    if len(axis) > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError("frequency must be strictly increasing")
    return values, axis


def _trapezoid_weights(axis: np.ndarray) -> np.ndarray:
    """Return normalized trapezoid weights; a singleton axis gets unit mass."""
    if len(axis) == 1:
        return np.ones(1, dtype=np.float64)
    differences = np.diff(axis)
    weights = np.empty(len(axis), dtype=np.float64)
    weights[0] = 0.5 * differences[0]
    weights[-1] = 0.5 * differences[-1]
    weights[1:-1] = 0.5 * (differences[:-1] + differences[1:])
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("frequency axis has no positive trapezoidal measure")
    return weights / total


def _weighted_nrmse(truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / float(np.sum(weights))
    residual = float(np.sqrt(np.sum(weights * np.abs(prediction - truth) ** 2)))
    centered = truth - np.sum(weights * truth)
    denominator = float(np.sqrt(np.sum(weights * np.abs(centered) ** 2)))
    if denominator <= _EPS:
        denominator = max(float(np.sqrt(np.sum(weights * np.abs(truth) ** 2))), _EPS)
    return residual / denominator


def _metric_features(
    task: str,
    axis: np.ndarray,
    response: np.ndarray,
    detector: Mapping[str, Any],
) -> list[dict[str, Any]]:
    config = {
        "smoothing_method": detector.get("smoothing", {}).get("method", "savgol"),
        "window_length": int(detector.get("smoothing", {}).get("window_length", 5)),
        "polyorder": int(detector.get("smoothing", {}).get("polyorder", 2)),
        "prominence_fraction": float(detector.get("prominence_fraction", 0.05)),
        "minimum_distance_observed_spacings": int(detector.get("minimum_distance_observed_spacings", 2)),
        "max_features": int(detector.get("max_features", 3)),
        "minimum_half_width_observed_spacings": 2.0,
    }
    return _feature_candidates(task, axis, response, config)


def _match_features(
    truth: list[dict[str, Any]],
    prediction: list[dict[str, Any]],
    tolerance: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidates = sorted(
        (
            abs(float(left["center"] - right["center"])), left_index, right_index
        )
        for left_index, left in enumerate(truth)
        for right_index, right in enumerate(prediction)
        if abs(float(left["center"] - right["center"])) <= tolerance
    )
    used_truth: set[int] = set()
    used_prediction: set[int] = set()
    matches = []
    for _, left_index, right_index in candidates:
        if left_index in used_truth or right_index in used_prediction:
            continue
        used_truth.add(left_index)
        used_prediction.add(right_index)
        matches.append((truth[left_index], prediction[right_index]))
    return matches


def complex_window_metrics(
    frequency: Any,
    truth: Any,
    prediction: Any,
    window_spec: WindowSpec,
    *,
    log_weighted: bool = False,
) -> dict[str, Any]:
    """Compute global/window complex errors and frozen-window feature diagnostics."""
    if not isinstance(window_spec, WindowSpec):
        raise ValueError("window_spec must be a WindowSpec")
    values, axis = _metric_axis(window_spec, frequency)
    truth_complex = _complex_response(truth, "truth", require_finite=False)
    prediction_complex = _complex_response(prediction, "prediction", require_finite=False)
    if len(truth_complex) != len(prediction_complex) or len(values) != len(truth_complex):
        raise ValueError("truth, prediction, and frequency lengths must match")

    base = {
        "complex_nrmse": None,
        "global_complex_nrmse": None,
        "resonance_window_complex_nrmse": None,
        "phase_mae_rad": None,
        "phase_mae": None,
        "resonance_window_phase_mae_rad": None,
        "feature_center_error": None,
        "feature_center_error_axis": None,
        "strongest_feature_frequency_error": None,
        "missed_feature_rate": None,
        "false_feature_rate": None,
        "missed_peak_rate": None,
        "false_peak_rate": None,
        "effective_hidden_count": 0,
        "effective_hidden_point_count": 0,
        "true_feature_count": 0,
        "predicted_feature_count": 0,
        "matched_feature_count": 0,
        "window_status": window_spec.status,
        "log_frequency_weight_sum": None,
        "status": window_spec.status,
    }
    if not np.all(np.isfinite(truth_complex)) or not np.all(np.isfinite(prediction_complex)):
        base["status"] = "numerical_failure"
        return base

    if log_weighted:
        if window_spec.axis_space != "log10_frequency":
            raise ValueError("log_weighted metrics require a log10_frequency WindowSpec")
        weights = _trapezoid_weights(axis)
        base["log_frequency_weight_sum"] = float(np.sum(weights))
    else:
        weights = np.full(len(axis), 1.0 / len(axis), dtype=np.float64)

    base["global_complex_nrmse"] = _weighted_nrmse(truth_complex, prediction_complex, weights)
    base["complex_nrmse"] = base["global_complex_nrmse"]
    phase = np.abs(np.angle(prediction_complex * np.conjugate(truth_complex)))
    base["phase_mae_rad"] = float(np.sum(weights * phase))
    base["phase_mae"] = base["phase_mae_rad"]

    mask = np.zeros(len(axis), dtype=bool)
    for left, right in window_spec.intervals:
        mask |= (axis >= left) & (axis <= right)
    base["effective_hidden_count"] = int(np.count_nonzero(mask))
    base["effective_hidden_point_count"] = base["effective_hidden_count"]
    if base["effective_hidden_count"]:
        if log_weighted:
            # Recompute the quadrature on the frozen window axis so outside points
            # cannot change an endpoint's local interval contribution.
            window_weights = _trapezoid_weights(axis[mask])
        else:
            window_weights = weights[mask]
        base["resonance_window_complex_nrmse"] = _weighted_nrmse(
            truth_complex[mask], prediction_complex[mask], window_weights
        )
        base["resonance_window_phase_mae_rad"] = float(
            np.sum((window_weights / np.sum(window_weights)) * phase[mask])
        )
    elif window_spec.status == "ok":
        base["window_status"] = "empty_window"
        base["status"] = "empty_window"

    detector = window_spec.detector
    true_features = _metric_features(window_spec.task, axis, truth_complex, detector)
    predicted_features = _metric_features(window_spec.task, axis, prediction_complex, detector)
    base["true_feature_count"] = len(true_features)
    base["predicted_feature_count"] = len(predicted_features)
    if true_features or predicted_features:
        spacing = float(np.median(np.diff(axis))) if len(axis) > 1 else _EPS
        tolerance = float(detector.get("matching_tolerance", 2.0 * spacing))
        matches = _match_features(true_features, predicted_features, tolerance)
        base["matched_feature_count"] = len(matches)
        base["missed_feature_rate"] = float((len(true_features) - len(matches)) / len(true_features)) if true_features else 0.0
        base["false_feature_rate"] = float((len(predicted_features) - len(matches)) / len(predicted_features)) if predicted_features else 0.0
        base["missed_peak_rate"] = base["missed_feature_rate"]
        base["false_peak_rate"] = base["false_feature_rate"]
        errors = [abs(float(left["center"] - right["center"])) for left, right in matches]
        if errors:
            base["feature_center_error_axis"] = float(np.mean(errors))
            base["feature_center_error"] = base["feature_center_error_axis"]
        if true_features and predicted_features:
            strongest_truth = max(true_features, key=lambda item: (item["prominence"], -item["center"]))
            strongest_prediction = max(predicted_features, key=lambda item: (item["prominence"], -item["center"]))
            if window_spec.axis_space == "log10_frequency":
                true_frequency = 10.0 ** strongest_truth["center"]
                predicted_frequency = 10.0 ** strongest_prediction["center"]
            else:
                true_frequency = strongest_truth["center"]
                predicted_frequency = strongest_prediction["center"]
            base["strongest_feature_frequency_error"] = float(abs(predicted_frequency - true_frequency))
    else:
        base["missed_feature_rate"] = 0.0
        base["false_feature_rate"] = 0.0
        base["missed_peak_rate"] = 0.0
        base["false_peak_rate"] = 0.0
    return base
