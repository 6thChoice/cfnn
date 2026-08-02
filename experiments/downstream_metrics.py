"""Task-aware metrics for complex resonant responses."""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import find_peaks


def _to_complex(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("complex responses must have shape (n, 2)")
    return array[:, 0] + 1j * array[:, 1]


def _peak_indices(magnitude: np.ndarray) -> np.ndarray:
    span = float(np.ptp(magnitude))
    if not np.isfinite(span) or span <= np.finfo(float).eps:
        return np.asarray([], dtype=int)
    prominence = max(0.05 * span, np.finfo(float).eps)
    peaks, _ = find_peaks(magnitude, prominence=prominence)
    return peaks.astype(int)


def _match_peaks(
    frequency: np.ndarray,
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
    tolerance: float,
) -> list[tuple[int, int]]:
    candidates = sorted(
        (
            (abs(float(frequency[truth] - frequency[prediction])), int(truth), int(prediction))
            for truth in true_indices
            for prediction in predicted_indices
            if abs(float(frequency[truth] - frequency[prediction])) <= tolerance
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    used_true = set()
    used_predicted = set()
    matches = []
    for _, truth, prediction in candidates:
        if truth in used_true or prediction in used_predicted:
            continue
        used_true.add(truth)
        used_predicted.add(prediction)
        matches.append((truth, prediction))
    return matches


def _quality_factor(frequency: np.ndarray, magnitude: np.ndarray, peak_index: int) -> float | None:
    baseline = float(np.min(magnitude))
    peak = float(magnitude[peak_index])
    level = baseline + (peak - baseline) / math.sqrt(2.0)
    left = np.flatnonzero(magnitude[:peak_index] <= level)
    right = np.flatnonzero(magnitude[peak_index + 1:] <= level)
    if not len(left) or not len(right):
        return None
    left_index = int(left[-1])
    right_index = int(peak_index + 1 + right[0])
    bandwidth = float(frequency[right_index] - frequency[left_index])
    if bandwidth <= 0:
        return None
    return float(abs(frequency[peak_index]) / bandwidth)


def compute_complex_metrics(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    frequency: np.ndarray,
    resolution: float,
) -> dict:
    truth = _to_complex(y_true)
    prediction = _to_complex(y_predicted)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape or len(frequency) != len(truth):
        raise ValueError("truth, prediction, and frequency lengths must match")
    if resolution <= 0:
        raise ValueError("frequency resolution must be positive")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        return {
            "complex_nrmse": None,
            "phase_mae_rad": None,
            "status": "numerical_failure",
        }

    residual_rms = float(np.sqrt(np.mean(np.abs(prediction - truth) ** 2)))
    denominator = float(np.sqrt(np.mean(np.abs(truth - np.mean(truth)) ** 2)))
    if denominator <= np.finfo(float).eps:
        denominator = max(float(np.sqrt(np.mean(np.abs(truth) ** 2))), np.finfo(float).eps)
    complex_nrmse = residual_rms / denominator
    phase_difference = np.angle(prediction * np.conjugate(truth))
    phase_mae = float(np.mean(np.abs(phase_difference)))

    true_magnitude = np.abs(truth)
    predicted_magnitude = np.abs(prediction)
    true_peaks = _peak_indices(true_magnitude)
    predicted_peaks = _peak_indices(predicted_magnitude)
    tolerance = 2.0 * float(resolution)
    matches = _match_peaks(frequency, true_peaks, predicted_peaks, tolerance)
    missed = len(true_peaks) - len(matches)
    false = len(predicted_peaks) - len(matches)
    frequency_errors = [
        abs(float(frequency[truth_index] - frequency[predicted_index]))
        for truth_index, predicted_index in matches
    ]

    q_error = None
    if len(true_peaks) and len(predicted_peaks):
        strongest_true = int(true_peaks[np.argmax(true_magnitude[true_peaks])])
        strongest_predicted = int(predicted_peaks[np.argmax(predicted_magnitude[predicted_peaks])])
        true_q = _quality_factor(frequency, true_magnitude, strongest_true)
        predicted_q = _quality_factor(frequency, predicted_magnitude, strongest_predicted)
        if true_q is not None and predicted_q is not None:
            q_error = float(abs(predicted_q - true_q))

    return {
        "complex_nrmse": float(complex_nrmse),
        "phase_mae_rad": phase_mae,
        "true_peak_count": int(len(true_peaks)),
        "predicted_peak_count": int(len(predicted_peaks)),
        "matched_peak_count": int(len(matches)),
        "missed_peak_rate": float(missed / len(true_peaks)) if len(true_peaks) else 0.0,
        "false_peak_rate": float(false / len(predicted_peaks)) if len(predicted_peaks) else 0.0,
        "peak_frequency_mae": float(np.mean(frequency_errors)) if frequency_errors else None,
        "quality_factor_absolute_error": q_error,
        "peak_matching_tolerance": tolerance,
        "status": "ok",
    }

