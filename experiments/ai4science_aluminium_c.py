"""Package C utilities for Aluminium FRF active modal-analysis experiments."""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_prominences, peak_widths

from downstream_protocol import MeasuredCurve


def _seed(*parts: object) -> int:
    encoded = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _as_two_columns(response: np.ndarray) -> np.ndarray:
    values = np.asarray(response)
    return np.column_stack((values.real, values.imag)).astype(np.float32)


def _as_complex(response: np.ndarray) -> np.ndarray:
    values = np.asarray(response, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("complex response arrays must have shape (n, 2)")
    return values[:, 0] + 1j * values[:, 1]


def signed_log_complex(response: np.ndarray) -> np.ndarray:
    """Compress real/imaginary dynamic range while preserving signs."""
    values = np.asarray(response, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("complex response arrays must have shape (n, 2)")
    return (np.sign(values) * np.log1p(np.abs(values))).astype(np.float32)


def inverse_signed_log_complex(encoded: np.ndarray) -> np.ndarray:
    """Invert :func:`signed_log_complex` back to real/imaginary columns."""
    values = np.asarray(encoded, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("encoded complex arrays must have shape (n, 2)")
    return (np.sign(values) * np.expm1(np.abs(values))).astype(np.float32)


def interpolate_response(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    observed_indices: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Reconstruct a full complex FRF by linear interpolation from measured bins."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    observed = np.sort(np.asarray(observed_indices, dtype=np.int64))
    if len(observed) < 2:
        raise ValueError("interpolation baselines require at least two observed points")
    if np.min(observed) < 0 or np.max(observed) >= len(frequency):
        raise ValueError("observed indices are outside the frequency grid")
    if mode == "complex_linear":
        values = np.asarray(response, dtype=np.float64)
        inverse = lambda array: array
    elif mode == "signed_log_linear":
        values = signed_log_complex(response).astype(np.float64)
        inverse = inverse_signed_log_complex
    else:
        raise ValueError(f"unknown interpolation mode: {mode}")
    predicted = np.column_stack([
        np.interp(frequency, frequency[observed], values[observed, column])
        for column in range(2)
    ]).astype(np.float32)
    return np.asarray(inverse(predicted), dtype=np.float32)


def modal_peak_fit_response(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    observed_indices: np.ndarray,
    *,
    max_modes: int = 5,
) -> np.ndarray:
    """Reconstruct a complex FRF with a sparse, observed-only modal peak prior."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    observed = np.sort(np.asarray(observed_indices, dtype=np.int64))
    if len(observed) < 5:
        return interpolate_response(frequency, response, observed, mode="signed_log_linear")
    if np.min(observed) < 0 or np.max(observed) >= len(frequency):
        raise ValueError("observed indices are outside the frequency grid")

    values = _as_complex(response)
    observed_frequency = frequency[observed]
    observed_values = values[observed]
    observed_log_magnitude = np.log1p(np.abs(observed_values))
    span = float(np.ptp(observed_log_magnitude))
    if not np.isfinite(span) or span <= np.finfo(float).eps:
        return interpolate_response(frequency, response, observed, mode="signed_log_linear")

    base = interpolate_response(frequency, response, observed, mode="signed_log_linear")
    base_complex = _as_complex(base)
    phase = np.interp(
        frequency,
        observed_frequency,
        np.unwrap(np.angle(observed_values)),
    )
    fitted_log_magnitude = np.log1p(np.abs(base_complex))

    distance = max(1, len(observed) // 32)
    peaks, properties = find_peaks(
        observed_log_magnitude,
        prominence=max(0.08 * span, np.finfo(float).eps),
        distance=distance,
    )
    if len(peaks) < 1:
        interior = np.arange(1, len(observed) - 1, dtype=np.int64)
        if not len(interior):
            return base
        peaks = interior[np.argsort(-observed_log_magnitude[interior])[: int(max_modes)]]
        prominences = np.full(len(peaks), span, dtype=np.float64)
    else:
        prominences = np.asarray(properties.get("prominences"))
        if prominences.size != len(peaks):
            prominences = peak_prominences(observed_log_magnitude, peaks)[0]
    order = np.argsort(observed_frequency[peaks])[: int(max_modes)]

    median_spacing = (
        float(np.median(np.diff(observed_frequency)))
        if len(observed_frequency) > 1 else
        float(np.median(np.diff(frequency)))
    )
    grid_spacing = float(np.median(np.diff(frequency))) if len(frequency) > 1 else 1.0
    for peak_position in peaks[order]:
        peak_position = int(peak_position)
        peak_frequency = float(observed_frequency[peak_position])
        peak_height = float(observed_log_magnitude[peak_position])
        neighborhood_left = max(0, peak_position - 4)
        neighborhood_right = min(len(observed_log_magnitude), peak_position + 5)
        baseline = float(np.percentile(observed_log_magnitude[neighborhood_left:neighborhood_right], 20.0))
        amplitude = max(peak_height - baseline, 0.0)
        if amplitude <= np.finfo(float).eps:
            continue

        half_height = baseline + 0.5 * amplitude
        left_position = peak_position
        while left_position > 0 and observed_log_magnitude[left_position] > half_height:
            left_position -= 1
        right_position = peak_position
        while right_position < len(observed_log_magnitude) - 1 and observed_log_magnitude[right_position] > half_height:
            right_position += 1
        left_frequency = float(observed_frequency[left_position])
        right_frequency = float(observed_frequency[right_position])
        width = max(right_frequency - left_frequency, 2.0 * median_spacing, 2.0 * grid_spacing)
        gamma = max(0.5 * width, grid_spacing)
        profile = baseline + amplitude / (1.0 + ((frequency - peak_frequency) / gamma) ** 2)
        fitted_log_magnitude = np.maximum(fitted_log_magnitude, profile)

    magnitude = np.expm1(np.maximum(fitted_log_magnitude, 0.0))
    prediction = magnitude * np.exp(1j * phase)
    return _as_two_columns(prediction)


def complex_modal_fit_response(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    observed_indices: np.ndarray,
    *,
    max_modes: int = 5,
) -> np.ndarray:
    """Reconstruct a complex FRF by locally fitting observed modal pole shapes."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    observed = np.sort(np.asarray(observed_indices, dtype=np.int64))
    if len(observed) < 8:
        return modal_peak_fit_response(frequency, response, observed, max_modes=max_modes)
    if np.min(observed) < 0 or np.max(observed) >= len(frequency):
        raise ValueError("observed indices are outside the frequency grid")

    values = _as_complex(response)
    observed_frequency = frequency[observed]
    observed_values = values[observed]
    observed_log_magnitude = np.log1p(np.abs(observed_values))
    span = float(np.ptp(observed_log_magnitude))
    if not np.isfinite(span) or span <= np.finfo(float).eps:
        return modal_peak_fit_response(frequency, response, observed, max_modes=max_modes)

    prediction = _as_complex(modal_peak_fit_response(frequency, response, observed, max_modes=max_modes))
    median_spacing = (
        float(np.median(np.diff(observed_frequency)))
        if len(observed_frequency) > 1 else
        float(np.median(np.diff(frequency)))
    )
    grid_spacing = float(np.median(np.diff(frequency))) if len(frequency) > 1 else 1.0
    peaks, properties = find_peaks(
        observed_log_magnitude,
        prominence=max(0.08 * span, np.finfo(float).eps),
        distance=max(1, len(observed) // 32),
    )
    if len(peaks) < 1:
        interior = np.arange(1, len(observed) - 1, dtype=np.int64)
        if not len(interior):
            return _as_two_columns(prediction)
        peaks = interior[np.argsort(-observed_log_magnitude[interior])[: int(max_modes)]]
    order = np.argsort(observed_frequency[peaks])[: int(max_modes)]

    for peak_position in peaks[order]:
        peak_position = int(peak_position)
        left = max(0, peak_position - 6)
        right = min(len(observed_frequency), peak_position + 7)
        if right - left < 7:
            continue
        local_frequency = observed_frequency[left:right]
        local_values = observed_values[left:right]
        center_guess = float(observed_frequency[peak_position])
        local_span = max(float(local_frequency[-1] - local_frequency[0]), 4.0 * median_spacing, 4.0 * grid_spacing)
        scale = max(float(np.sqrt(np.mean(np.abs(local_values) ** 2))), np.finfo(float).eps)
        background_guess = complex(np.median(local_values[:2].real), np.median(local_values[:2].imag))
        gamma_guess = max(median_spacing, grid_spacing)
        residue_guess = (observed_values[peak_position] - background_guess) * (1j * gamma_guess * max(abs(center_guess), grid_spacing))

        def evaluate(parameters: np.ndarray, grid: np.ndarray) -> np.ndarray:
            f0 = float(parameters[0])
            gamma = float(np.exp(parameters[1]))
            background = complex(float(parameters[2]), float(parameters[3]))
            slope = complex(float(parameters[4]), float(parameters[5]))
            residue = complex(float(parameters[6]), float(parameters[7]))
            normalized = (grid - f0) / local_span
            denominator = (f0**2 - grid**2) + 1j * gamma * np.maximum(np.abs(grid), grid_spacing)
            floor = (grid_spacing**2) + 1j * gamma * grid_spacing
            denominator = np.where(np.abs(denominator) < abs(floor), floor, denominator)
            return background + slope * normalized + residue / denominator

        def residual(parameters: np.ndarray) -> np.ndarray:
            error = (evaluate(parameters, local_frequency) - local_values) / scale
            return np.concatenate([error.real, error.imag])

        initial = np.asarray([
            center_guess,
            math.log(gamma_guess),
            background_guess.real,
            background_guess.imag,
            0.0,
            0.0,
            residue_guess.real,
            residue_guess.imag,
        ], dtype=np.float64)
        lower = np.asarray([
            float(local_frequency[0]),
            math.log(max(0.25 * grid_spacing, np.finfo(float).eps)),
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
        ], dtype=np.float64)
        upper = np.asarray([
            float(local_frequency[-1]),
            math.log(max(local_span, grid_spacing)),
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ], dtype=np.float64)
        try:
            fitted = least_squares(
                residual,
                initial,
                bounds=(lower, upper),
                max_nfev=300,
                loss="soft_l1",
            )
        except (FloatingPointError, ValueError):
            continue
        if not fitted.success or not np.all(np.isfinite(fitted.x)):
            continue
        local_indices = observed[left:right]
        current_local_error = float(np.mean(np.abs(prediction[local_indices] - local_values) ** 2))
        fitted_local = evaluate(fitted.x, local_frequency)
        fitted_error = float(np.mean(np.abs(fitted_local - local_values) ** 2))
        if not np.isfinite(fitted_error) or fitted_error > current_local_error:
            continue
        blend_width = max(0.5 * local_span, 4.0 * grid_spacing)
        weights = np.exp(-((frequency - float(fitted.x[0])) / blend_width) ** 4)
        fitted_full = evaluate(fitted.x, frequency)
        finite = np.isfinite(fitted_full.real) & np.isfinite(fitted_full.imag)
        prediction[finite] = (1.0 - weights[finite]) * prediction[finite] + weights[finite] * fitted_full[finite]
    return _as_two_columns(prediction)


def global_complex_modal_prior_fit(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    observed_indices: np.ndarray,
    *,
    max_modes: int = 5,
    candidate_order: str = "prominence",
    candidate_frequencies_hz: list[float] | None = None,
    center_window_hz: float | None = None,
) -> dict:
    """Fit a global complex modal prior and return prediction plus fit diagnostics."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    observed = np.sort(np.asarray(observed_indices, dtype=np.int64))
    if len(observed) < max(12, 4 * int(max_modes)):
        prediction = complex_modal_fit_response(frequency, response, observed, max_modes=max_modes)
        return {
            "prediction": prediction,
            "diagnostics": {
                "status": "fallback",
                "fallback_reason": "insufficient_observed_points",
                "initial_candidate_frequencies_hz": [],
                "fitted_candidate_frequencies_hz": [],
                "selected_observed_indices": [],
            },
        }
    if np.min(observed) < 0 or np.max(observed) >= len(frequency):
        raise ValueError("observed indices are outside the frequency grid")

    values = _as_complex(response)
    observed_frequency = frequency[observed]
    observed_values = values[observed]
    grid_spacing = float(np.median(np.diff(frequency))) if len(frequency) > 1 else 1.0
    observed_spacing = float(np.median(np.diff(observed_frequency))) if len(observed_frequency) > 1 else grid_spacing
    span_hz = max(float(np.ptp(frequency)), grid_spacing)
    y_scale = max(float(np.sqrt(np.mean(np.abs(observed_values) ** 2))), np.finfo(float).eps)
    log_magnitude = np.log1p(np.abs(observed_values))
    dynamic_range = float(np.ptp(log_magnitude))
    if not np.isfinite(dynamic_range) or dynamic_range <= np.finfo(float).eps:
        prediction = complex_modal_fit_response(frequency, response, observed, max_modes=max_modes)
        return {
            "prediction": prediction,
            "diagnostics": {
                "status": "fallback",
                "fallback_reason": "no_dynamic_range",
                "initial_candidate_frequencies_hz": [],
                "fitted_candidate_frequencies_hz": [],
                "selected_observed_indices": [],
            },
        }

    if candidate_order not in {"prominence", "low_order_frequency", "catalogue_supplied"}:
        raise ValueError(f"unknown candidate order: {candidate_order}")

    supplied_candidates = [
        float(value)
        for value in (candidate_frequencies_hz or [])
        if value is not None and math.isfinite(float(value))
    ]
    supplied_candidates = sorted(supplied_candidates)[: int(max_modes)]
    if candidate_order == "catalogue_supplied":
        if not supplied_candidates:
            prediction = complex_modal_fit_response(frequency, response, observed, max_modes=max_modes)
            return {
                "prediction": prediction,
                "diagnostics": {
                    "status": "fallback",
                    "fallback_reason": "no_catalogue_candidates",
                    "candidate_order": candidate_order,
                    "initial_candidate_frequencies_hz": [],
                    "fitted_candidate_frequencies_hz": [],
                    "selected_observed_indices": [],
                },
            }
        centers = np.asarray(supplied_candidates, dtype=np.float64)
        selected = np.asarray([
            int(np.argmin(np.abs(observed_frequency - center)))
            for center in centers
        ], dtype=np.int64)
    else:
        prominence_fraction = 0.004 if candidate_order == "low_order_frequency" else 0.04
        peak_distance = (
            max(1, len(observed) // 96)
            if candidate_order == "low_order_frequency" else
            max(2, len(observed) // 24)
        )
        peaks, properties = find_peaks(
            log_magnitude,
            prominence=max(prominence_fraction * dynamic_range, np.finfo(float).eps),
            distance=peak_distance,
        )
        if len(peaks):
            prominences = np.asarray(properties.get("prominences"))
            if prominences.size != len(peaks):
                prominences = peak_prominences(log_magnitude, peaks)[0]
            if candidate_order == "low_order_frequency":
                selected = peaks[np.argsort(observed_frequency[peaks])[: int(max_modes)]]
            else:
                selected = peaks[np.argsort(-prominences)[: int(max_modes)]]
        else:
            interior = np.arange(1, len(observed) - 1, dtype=np.int64)
            if not len(interior):
                prediction = complex_modal_fit_response(frequency, response, observed, max_modes=max_modes)
                return {
                    "prediction": prediction,
                    "diagnostics": {
                        "status": "fallback",
                        "fallback_reason": "no_interior_candidates",
                        "initial_candidate_frequencies_hz": [],
                        "fitted_candidate_frequencies_hz": [],
                        "selected_observed_indices": [],
                    },
                }
            selected = interior[np.argsort(-log_magnitude[interior])[: int(max_modes)]]
        selected = np.asarray(sorted(selected, key=lambda index: float(observed_frequency[int(index)])), dtype=np.int64)
        if not len(selected):
            prediction = complex_modal_fit_response(frequency, response, observed, max_modes=max_modes)
            return {
                "prediction": prediction,
                "diagnostics": {
                    "status": "fallback",
                    "fallback_reason": "no_selected_candidates",
                    "initial_candidate_frequencies_hz": [],
                    "fitted_candidate_frequencies_hz": [],
                    "selected_observed_indices": [],
                },
            }
        centers = observed_frequency[selected]

    mode_count = int(len(selected))
    background = complex(float(np.median(observed_values.real)), float(np.median(observed_values.imag)))
    slope = 0.0 + 0.0j
    gammas = np.full(mode_count, max(2.0 * observed_spacing, grid_spacing), dtype=np.float64)
    residues = []
    for center, gamma, position in zip(centers, gammas, selected):
        denominator = (center**2 - center**2) + 1j * gamma * max(abs(float(center)), grid_spacing)
        residues.append((observed_values[int(position)] - background) * denominator)
    residues = np.asarray(residues, dtype=np.complex128)

    def evaluate(parameters: np.ndarray, grid: np.ndarray) -> np.ndarray:
        bg = complex(float(parameters[0]), float(parameters[1]))
        sl = complex(float(parameters[2]), float(parameters[3]))
        normalized = (grid - float(frequency[0])) / span_hz
        prediction = bg + sl * normalized
        offset = 4
        for _ in range(mode_count):
            f0 = float(parameters[offset])
            gamma = float(np.exp(parameters[offset + 1]))
            residue = complex(float(parameters[offset + 2]), float(parameters[offset + 3]))
            denominator = (f0**2 - grid**2) + 1j * gamma * np.maximum(np.abs(grid), grid_spacing)
            floor = (grid_spacing**2) + 1j * gamma * grid_spacing
            denominator = np.where(np.abs(denominator) < abs(floor), floor, denominator)
            prediction = prediction + residue / denominator
            offset += 4
        return prediction

    def residual(parameters: np.ndarray) -> np.ndarray:
        error = (evaluate(parameters, observed_frequency) - observed_values) / y_scale
        return np.concatenate([error.real, error.imag])

    initial = [background.real, background.imag, slope.real, slope.imag]
    lower = [-np.inf, -np.inf, -np.inf, -np.inf]
    upper = [np.inf, np.inf, np.inf, np.inf]
    for center, gamma, residue in zip(centers, gammas, residues):
        window = (
            max(float(center_window_hz), 2.0 * grid_spacing)
            if center_window_hz is not None else
            max(6.0 * observed_spacing, 2.0 * grid_spacing)
        )
        initial.extend([float(center), math.log(float(gamma)), float(residue.real), float(residue.imag)])
        lower.extend([
            max(float(frequency[0]), float(center - window)),
            math.log(max(0.25 * grid_spacing, np.finfo(float).eps)),
            -np.inf,
            -np.inf,
        ])
        upper.extend([
            min(float(frequency[-1]), float(center + window)),
            math.log(max(0.25 * span_hz, grid_spacing)),
            np.inf,
            np.inf,
        ])

    try:
        fitted = least_squares(
            residual,
            np.asarray(initial, dtype=np.float64),
            bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
            loss="soft_l1",
            max_nfev=1200,
        )
    except (FloatingPointError, ValueError):
        prediction = complex_modal_fit_response(frequency, response, observed, max_modes=max_modes)
        return {
            "prediction": prediction,
            "diagnostics": {
                "status": "fallback",
                "fallback_reason": "least_squares_exception",
                "initial_candidate_frequencies_hz": [float(center) for center in centers],
                "fitted_candidate_frequencies_hz": [],
                "selected_observed_indices": [int(observed[int(index)]) for index in selected],
            },
        }
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        prediction = complex_modal_fit_response(frequency, response, observed, max_modes=max_modes)
        return {
            "prediction": prediction,
            "diagnostics": {
                "status": "fallback",
                "fallback_reason": "least_squares_failed",
                "initial_candidate_frequencies_hz": [float(center) for center in centers],
                "fitted_candidate_frequencies_hz": [],
                "selected_observed_indices": [int(observed[int(index)]) for index in selected],
            },
        }

    prediction = evaluate(fitted.x, frequency)
    if not np.all(np.isfinite(prediction.real)) or not np.all(np.isfinite(prediction.imag)):
        fallback = complex_modal_fit_response(frequency, response, observed, max_modes=max_modes)
        return {
            "prediction": fallback,
            "diagnostics": {
                "status": "fallback",
                "fallback_reason": "nonfinite_prediction",
                "initial_candidate_frequencies_hz": [float(center) for center in centers],
                "fitted_candidate_frequencies_hz": [],
                "selected_observed_indices": [int(observed[int(index)]) for index in selected],
            },
        }
    fitted_centers = [
        float(fitted.x[offset])
        for offset in range(4, 4 + 4 * mode_count, 4)
    ]
    return {
        "prediction": _as_two_columns(prediction),
        "diagnostics": {
                "status": "ok",
                "candidate_source": "observed_log_magnitude_prominence",
                "candidate_order": candidate_order,
                "initial_candidate_frequencies_hz": [float(center) for center in centers],
            "fitted_candidate_frequencies_hz": fitted_centers,
            "initial_candidate_gamma_hz": [float(gamma) for gamma in gammas],
            "fitted_candidate_gamma_hz": [
                float(np.exp(fitted.x[offset + 1]))
                for offset in range(4, 4 + 4 * mode_count, 4)
            ],
            "selected_observed_indices": [int(observed[int(index)]) for index in selected],
            "selected_observed_positions": [int(index) for index in selected],
            "observed_frequency_count": int(len(observed)),
            "mode_count": int(mode_count),
            "cost": float(fitted.cost),
            "optimality": float(fitted.optimality),
            "nfev": int(fitted.nfev),
        },
    }


def global_complex_modal_prior_response(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    observed_indices: np.ndarray,
    *,
    max_modes: int = 5,
) -> np.ndarray:
    """Fit a global complex second-order modal prior from sparse observed FRF bins."""
    return global_complex_modal_prior_fit(
        frequency_hz,
        response,
        observed_indices,
        max_modes=max_modes,
    )["prediction"]


def low_order_global_complex_modal_prior_fit(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    observed_indices: np.ndarray,
    *,
    max_modes: int = 5,
) -> dict:
    """Fit the global modal prior using low-frequency candidate ordering."""
    return global_complex_modal_prior_fit(
        frequency_hz,
        response,
        observed_indices,
        max_modes=max_modes,
        candidate_order="low_order_frequency",
    )


def low_order_global_complex_modal_prior_response(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    observed_indices: np.ndarray,
    *,
    max_modes: int = 5,
) -> np.ndarray:
    """Global complex modal-prior response with low-order candidate selection."""
    return low_order_global_complex_modal_prior_fit(
        frequency_hz,
        response,
        observed_indices,
        max_modes=max_modes,
    )["prediction"]


def catalogue_aware_global_complex_modal_prior_fit(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    observed_indices: np.ndarray,
    *,
    candidate_frequencies_hz: list[float],
    center_window_hz: float | None = None,
    max_modes: int = 5,
) -> dict:
    """Fit the global modal prior from supplied low-order modal candidate centers."""
    return global_complex_modal_prior_fit(
        frequency_hz,
        response,
        observed_indices,
        max_modes=max_modes,
        candidate_order="catalogue_supplied",
        candidate_frequencies_hz=candidate_frequencies_hz,
        center_window_hz=center_window_hz,
    )


def catalogue_aware_global_complex_modal_prior_response(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    observed_indices: np.ndarray,
    *,
    candidate_frequencies_hz: list[float],
    center_window_hz: float | None = None,
    max_modes: int = 5,
) -> np.ndarray:
    """Global modal-prior response initialized from supplied low-order modal centers."""
    return catalogue_aware_global_complex_modal_prior_fit(
        frequency_hz,
        response,
        observed_indices,
        candidate_frequencies_hz=candidate_frequencies_hz,
        center_window_hz=center_window_hz,
        max_modes=max_modes,
    )["prediction"]


def _single_mode_response(
    parameters: np.ndarray,
    grid: np.ndarray,
    *,
    span_hz: float,
    grid_spacing: float,
) -> np.ndarray:
    f0 = float(parameters[0])
    gamma = float(np.exp(parameters[1]))
    background = complex(float(parameters[2]), float(parameters[3]))
    slope = complex(float(parameters[4]), float(parameters[5]))
    residue = complex(float(parameters[6]), float(parameters[7]))
    normalized = (grid - f0) / max(float(span_hz), grid_spacing)
    denominator = (f0**2 - grid**2) + 1j * gamma * np.maximum(np.abs(grid), grid_spacing)
    floor = (grid_spacing**2) + 1j * gamma * grid_spacing
    denominator = np.where(np.abs(denominator) < abs(floor), floor, denominator)
    return background + slope * normalized + residue / denominator


def local_catalogue_band_modal_parameter_fit(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    observed_indices: np.ndarray,
    *,
    candidate_frequencies_hz: list[float],
    band_half_width_hz: float,
    center_window_hz: float,
    frequency_threshold_hz: float = 1.0,
    width_relative_threshold: float = 0.20,
) -> dict:
    """Fit independent local complex modal bands and report per-mode parameters."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    values = _as_complex(response)
    observed = np.sort(np.asarray(observed_indices, dtype=np.int64))
    if len(frequency) != len(values):
        raise ValueError("frequency and response lengths must match")
    if len(observed) and (np.min(observed) < 0 or np.max(observed) >= len(frequency)):
        raise ValueError("observed indices are outside the frequency grid")
    grid_spacing = float(np.median(np.diff(frequency))) if len(frequency) > 1 else 1.0
    mode_rows = []
    for rank, center in enumerate(candidate_frequencies_hz, start=1):
        center = float(center)
        band_mask = np.abs(frequency - center) <= float(band_half_width_hz)
        band_indices = np.flatnonzero(band_mask)
        observed_band = observed[np.abs(frequency[observed] - center) <= float(band_half_width_hz)]
        if len(observed_band) < 5 or len(band_indices) < 5:
            mode_rows.append({
                "mode_rank": int(rank),
                "truth_frequency_hz": center,
                "status": "insufficient_observed_band_points",
                "observed_band_count": int(len(observed_band)),
            })
            continue
        local_frequency = frequency[observed_band]
        local_values = values[observed_band]
        full_band_frequency = frequency[band_indices]
        truth_band_values = values[band_indices]
        local_span = max(float(np.ptp(local_frequency)), 2.0 * grid_spacing)
        scale = max(float(np.sqrt(np.mean(np.abs(local_values) ** 2))), np.finfo(float).eps)
        nearest = int(observed_band[int(np.argmin(np.abs(frequency[observed_band] - center)))])
        background_guess = complex(float(np.median(local_values.real)), float(np.median(local_values.imag)))
        gamma_guess = max(grid_spacing, 0.25 * float(band_half_width_hz))
        residue_guess = (values[nearest] - background_guess) * (1j * gamma_guess * max(abs(center), grid_spacing))

        def residual(parameters: np.ndarray) -> np.ndarray:
            error = (_single_mode_response(
                parameters,
                local_frequency,
                span_hz=local_span,
                grid_spacing=grid_spacing,
            ) - local_values) / scale
            return np.concatenate([error.real, error.imag])

        initial = np.asarray([
            center,
            math.log(gamma_guess),
            background_guess.real,
            background_guess.imag,
            0.0,
            0.0,
            residue_guess.real,
            residue_guess.imag,
        ], dtype=np.float64)
        lower = np.asarray([
            max(float(frequency[0]), center - float(center_window_hz)),
            math.log(max(0.25 * grid_spacing, np.finfo(float).eps)),
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
        ], dtype=np.float64)
        upper = np.asarray([
            min(float(frequency[-1]), center + float(center_window_hz)),
            math.log(max(float(band_half_width_hz), grid_spacing)),
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ], dtype=np.float64)
        try:
            fitted = least_squares(
                residual,
                initial,
                bounds=(lower, upper),
                loss="soft_l1",
                max_nfev=500,
            )
        except (FloatingPointError, ValueError):
            mode_rows.append({
                "mode_rank": int(rank),
                "truth_frequency_hz": center,
                "status": "fit_exception",
                "observed_band_count": int(len(observed_band)),
            })
            continue
        if not fitted.success or not np.all(np.isfinite(fitted.x)):
            mode_rows.append({
                "mode_rank": int(rank),
                "truth_frequency_hz": center,
                "status": "fit_failed",
                "observed_band_count": int(len(observed_band)),
            })
            continue

        predicted_band = _single_mode_response(
            fitted.x,
            full_band_frequency,
            span_hz=local_span,
            grid_spacing=grid_spacing,
        )
        truth_events = extract_modal_events(
            full_band_frequency,
            _as_two_columns(truth_band_values),
            max_modes=1,
            min_frequency_hz=float(full_band_frequency[0]),
        )
        predicted_events = extract_modal_events(
            full_band_frequency,
            _as_two_columns(predicted_band),
            max_modes=1,
            min_frequency_hz=float(full_band_frequency[0]),
        )
        truth_modes = truth_events.get("modes", [])
        predicted_modes = predicted_events.get("modes", [])
        if truth_modes and predicted_modes:
            truth_mode = truth_modes[0]
            predicted_mode = predicted_modes[0]
            truth_frequency = float(truth_mode["frequency_hz"])
            predicted_frequency = float(predicted_mode["frequency_hz"])
            truth_width = _finite_metric(truth_mode.get("width_hz"))
            predicted_width = _finite_metric(predicted_mode.get("width_hz"))
            truth_amplitude = _finite_metric(truth_mode.get("peak_amplitude"))
            predicted_amplitude = _finite_metric(predicted_mode.get("peak_amplitude"))
        else:
            truth_frequency = center
            predicted_frequency = float(fitted.x[0])
            truth_width = None
            predicted_width = float(np.exp(fitted.x[1]))
            truth_amplitude = None
            predicted_amplitude = None

        frequency_error = abs(predicted_frequency - truth_frequency)
        if truth_width is not None and predicted_width is not None and truth_width > 0.0:
            width_error = abs(predicted_width - truth_width) / max(abs(truth_width), np.finfo(float).eps)
        else:
            width_error = None
        if truth_amplitude is not None and predicted_amplitude is not None and truth_amplitude > 0.0:
            amplitude_error = abs(predicted_amplitude - truth_amplitude) / max(abs(truth_amplitude), np.finfo(float).eps)
        else:
            amplitude_error = None
        status = (
            "ok"
            if frequency_error <= float(frequency_threshold_hz)
            and width_error is not None
            and width_error <= float(width_relative_threshold)
            else
            "parameter_failure"
        )
        mode_rows.append({
            "mode_rank": int(rank),
            "status": status,
            "observed_band_count": int(len(observed_band)),
            "truth_frequency_hz": truth_frequency,
            "predicted_frequency_hz": predicted_frequency,
            "frequency_error_hz": float(frequency_error),
            "truth_width_hz": truth_width,
            "predicted_width_hz": predicted_width,
            "width_relative_error": width_error,
            "truth_peak_amplitude": truth_amplitude,
            "predicted_peak_amplitude": predicted_amplitude,
            "peak_amplitude_relative_error": amplitude_error,
            "fitted_center_hz": float(fitted.x[0]),
            "fitted_gamma_hz": float(np.exp(fitted.x[1])),
            "fit_cost": float(fitted.cost),
            "fit_nfev": int(fitted.nfev),
        })
    ok_rows = [row for row in mode_rows if row.get("status") == "ok"]
    finite_frequency_errors = [
        row.get("frequency_error_hz") for row in mode_rows
        if row.get("frequency_error_hz") is not None
    ]
    finite_width_errors = [
        row.get("width_relative_error") for row in mode_rows
        if row.get("width_relative_error") is not None
    ]
    return {
        "status": "ok" if mode_rows else "no_modes",
        "mode_count": int(len(mode_rows)),
        "passed_mode_count": int(len(ok_rows)),
        "pass_fraction": float(len(ok_rows) / len(mode_rows)) if mode_rows else None,
        "median_frequency_error_hz": _median(finite_frequency_errors),
        "median_width_relative_error": _median(finite_width_errors),
        "frequency_threshold_hz": float(frequency_threshold_hz),
        "width_relative_threshold": float(width_relative_threshold),
        "band_half_width_hz": float(band_half_width_hz),
        "center_window_hz": float(center_window_hz),
        "mode_rows": mode_rows,
    }


def _interpolate_threshold_crossing(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    threshold: float,
) -> float:
    denominator = float(y1) - float(y0)
    if abs(denominator) <= np.finfo(float).eps:
        return 0.5 * (float(x0) + float(x1))
    fraction = (float(threshold) - float(y0)) / denominator
    fraction = min(max(float(fraction), 0.0), 1.0)
    return float(x0) + fraction * (float(x1) - float(x0))


def half_power_modal_parameters(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    *,
    candidate_frequencies_hz: list[float],
    band_half_width_hz: float,
    min_band_points: int = 5,
) -> dict:
    """Estimate per-mode half-power bandwidth parameters inside catalogue bands."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    values = _as_complex(response)
    if len(frequency) != len(values):
        raise ValueError("frequency and response lengths must match")
    if len(frequency) < 2:
        raise ValueError("at least two frequency points are required")
    magnitude = np.abs(values)
    band_half_width_hz = max(float(band_half_width_hz), np.finfo(float).eps)
    min_band_points = max(3, int(min_band_points))
    mode_rows = []
    for rank, center in enumerate(candidate_frequencies_hz, start=1):
        center = float(center)
        band_indices = np.flatnonzero(np.abs(frequency - center) <= band_half_width_hz)
        if len(band_indices) < min_band_points:
            mode_rows.append({
                "mode_rank": int(rank),
                "candidate_frequency_hz": center,
                "status": "insufficient_band_points",
                "band_point_count": int(len(band_indices)),
            })
            continue
        local_frequency = frequency[band_indices]
        local_magnitude = magnitude[band_indices]
        peak_position = int(np.argmax(local_magnitude))
        peak_magnitude = float(local_magnitude[peak_position])
        edge_count = max(1, min(3, len(local_magnitude) // 4))
        edge_baseline = float(min(
            np.median(local_magnitude[:edge_count]),
            np.median(local_magnitude[-edge_count:]),
        ))
        if peak_magnitude <= np.finfo(float).eps:
            mode_rows.append({
                "mode_rank": int(rank),
                "candidate_frequency_hz": center,
                "status": "flat_or_no_peak",
                "band_point_count": int(len(band_indices)),
            })
            continue
        threshold = float(peak_magnitude / math.sqrt(2.0))

        left_crossing = None
        for position in range(peak_position, 0, -1):
            if local_magnitude[position - 1] <= threshold <= local_magnitude[position]:
                left_crossing = _interpolate_threshold_crossing(
                    local_frequency[position - 1],
                    local_magnitude[position - 1],
                    local_frequency[position],
                    local_magnitude[position],
                    threshold,
                )
                break
        right_crossing = None
        for position in range(peak_position, len(local_magnitude) - 1):
            if local_magnitude[position] >= threshold >= local_magnitude[position + 1]:
                right_crossing = _interpolate_threshold_crossing(
                    local_frequency[position],
                    local_magnitude[position],
                    local_frequency[position + 1],
                    local_magnitude[position + 1],
                    threshold,
                )
                break
        if left_crossing is None or right_crossing is None or right_crossing <= left_crossing:
            mode_rows.append({
                "mode_rank": int(rank),
                "candidate_frequency_hz": center,
                "frequency_hz": float(local_frequency[peak_position]),
                "status": "half_power_crossing_missing",
                "band_point_count": int(len(band_indices)),
                "peak_magnitude": peak_magnitude,
                "edge_baseline_magnitude": edge_baseline,
                "half_power_magnitude": threshold,
            })
            continue
        width_hz = float(right_crossing - left_crossing)
        frequency_hz_value = float(local_frequency[peak_position])
        mode_rows.append({
            "mode_rank": int(rank),
            "candidate_frequency_hz": center,
            "status": "ok",
            "band_point_count": int(len(band_indices)),
            "frequency_hz": frequency_hz_value,
            "width_hz": width_hz,
            "damping_ratio": float(width_hz / (2.0 * max(abs(frequency_hz_value), np.finfo(float).eps))),
            "left_half_power_frequency_hz": float(left_crossing),
            "right_half_power_frequency_hz": float(right_crossing),
            "peak_magnitude": peak_magnitude,
            "edge_baseline_magnitude": edge_baseline,
            "half_power_magnitude": threshold,
        })
    ok_rows = [row for row in mode_rows if row.get("status") == "ok"]
    return {
        "status": "ok" if mode_rows else "no_modes",
        "estimator": "local_half_power_bandwidth",
        "mode_count": int(len(mode_rows)),
        "ok_mode_count": int(len(ok_rows)),
        "band_half_width_hz": float(band_half_width_hz),
        "mode_rows": mode_rows,
    }


def half_power_recovery_metrics(
    frequency_hz: np.ndarray,
    truth_response: np.ndarray,
    prediction_response: np.ndarray,
    *,
    candidate_frequencies_hz: list[float],
    band_half_width_hz: float,
    frequency_threshold_hz: float = 1.0,
    width_relative_threshold: float = 0.20,
) -> dict:
    """Compare truth and predicted curves with half-power modal parameters."""
    truth = half_power_modal_parameters(
        frequency_hz,
        truth_response,
        candidate_frequencies_hz=candidate_frequencies_hz,
        band_half_width_hz=band_half_width_hz,
    )
    prediction = half_power_modal_parameters(
        frequency_hz,
        prediction_response,
        candidate_frequencies_hz=candidate_frequencies_hz,
        band_half_width_hz=band_half_width_hz,
    )
    mode_rows = []
    for truth_row, predicted_row in zip(truth.get("mode_rows", []), prediction.get("mode_rows", [])):
        row = {
            "mode_rank": truth_row.get("mode_rank"),
            "candidate_frequency_hz": truth_row.get("candidate_frequency_hz"),
            "truth_status": truth_row.get("status"),
            "predicted_status": predicted_row.get("status"),
            "truth_frequency_hz": truth_row.get("frequency_hz"),
            "predicted_frequency_hz": predicted_row.get("frequency_hz"),
            "truth_width_hz": truth_row.get("width_hz"),
            "predicted_width_hz": predicted_row.get("width_hz"),
            "truth_damping_ratio": truth_row.get("damping_ratio"),
            "predicted_damping_ratio": predicted_row.get("damping_ratio"),
        }
        if truth_row.get("status") == "ok" and predicted_row.get("status") == "ok":
            frequency_error = abs(float(predicted_row["frequency_hz"]) - float(truth_row["frequency_hz"]))
            width_error = abs(float(predicted_row["width_hz"]) - float(truth_row["width_hz"])) / max(
                abs(float(truth_row["width_hz"])),
                np.finfo(float).eps,
            )
            damping_error = abs(float(predicted_row["damping_ratio"]) - float(truth_row["damping_ratio"])) / max(
                abs(float(truth_row["damping_ratio"])),
                np.finfo(float).eps,
            )
            row.update({
                "frequency_error_hz": float(frequency_error),
                "width_relative_error": float(width_error),
                "damping_relative_error": float(damping_error),
                "passed": bool(
                    frequency_error <= float(frequency_threshold_hz)
                    and width_error <= float(width_relative_threshold)
                ),
            })
        else:
            row.update({
                "frequency_error_hz": None,
                "width_relative_error": None,
                "damping_relative_error": None,
                "passed": False,
            })
        mode_rows.append(row)
    finite_frequency_errors = [
        row.get("frequency_error_hz") for row in mode_rows
        if row.get("frequency_error_hz") is not None
    ]
    finite_width_errors = [
        row.get("width_relative_error") for row in mode_rows
        if row.get("width_relative_error") is not None
    ]
    finite_damping_errors = [
        row.get("damping_relative_error") for row in mode_rows
        if row.get("damping_relative_error") is not None
    ]
    return {
        "status": "ok" if mode_rows else "no_modes",
        "estimator": "half_power_recovery",
        "mode_count": int(len(mode_rows)),
        "truth_ok_mode_count": int(truth.get("ok_mode_count") or 0),
        "predicted_ok_mode_count": int(prediction.get("ok_mode_count") or 0),
        "passed_mode_count": int(sum(1 for row in mode_rows if row.get("passed"))),
        "pass_fraction": (
            float(sum(1 for row in mode_rows if row.get("passed")) / len(mode_rows))
            if mode_rows else None
        ),
        "median_frequency_error_hz": _median(finite_frequency_errors),
        "median_width_relative_error": _median(finite_width_errors),
        "median_damping_relative_error": _median(finite_damping_errors),
        "frequency_threshold_hz": float(frequency_threshold_hz),
        "width_relative_threshold": float(width_relative_threshold),
        "band_half_width_hz": float(band_half_width_hz),
        "truth_parameters": truth,
        "predicted_parameters": prediction,
        "mode_rows": mode_rows,
    }


@lru_cache(maxsize=2)
def load_aluminium_frf_full(raw_root: Path) -> list[MeasuredCurve]:
    """Construct full 2049-bin H1 accelerance FRFs from the released TDMS files."""
    from nptdms import TdmsFile
    from scipy.signal import csd, welch

    directory = Path(raw_root) / "aluminium_frf"
    paths = sorted(directory.glob("Waveforms Point*.tdms"))
    if len(paths) != 25:
        raise ValueError(f"expected 25 aluminium TDMS files, found {len(paths)}")
    curves = []
    for path in paths:
        point_match = re.search(r"Point(\d+)", path.name)
        point = int(point_match.group(1)) if point_match else -1
        tdms = TdmsFile.read(path)
        channels = [channel for group in tdms.groups() for channel in group.channels()]
        force_channel = next(
            channel for channel in channels
            if "Force" in str(channel.properties.get("NI_SV_SensorType", ""))
        )
        acceleration_channel = next(
            channel for channel in channels
            if "Accelerometer" in str(channel.properties.get("NI_SV_SensorType", ""))
        )
        force = np.asarray(force_channel[:], dtype=np.float64)
        acceleration = np.asarray(acceleration_channel[:], dtype=np.float64) * 9.81
        sample_interval = float(force_channel.properties["wf_increment"])
        sampling_rate = 1.0 / sample_interval
        nperseg = min(4096, len(force))
        noverlap = nperseg // 2
        frequency, cross_spectrum = csd(
            force,
            acceleration,
            fs=sampling_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="spectrum",
        )
        _, force_spectrum = welch(
            force,
            fs=sampling_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="spectrum",
        )
        floor = np.finfo(float).eps * max(float(np.max(force_spectrum)), 1.0)
        response = cross_spectrum / np.maximum(force_spectrum, floor)
        curves.append(MeasuredCurve(
            curve_id=f"point:{point:02d}",
            frequency=frequency.astype(np.float64),
            response=_as_two_columns(response),
            conditions={"point": point},
            source_files=(path.name,),
            metadata={
                "source_record": 7758683,
                "frequency_unit": "Hz",
                "response_unit": "m_per_s2_per_N",
                "frf_estimator": "H1",
                "sampling_rate_hz": sampling_rate,
                "nperseg": nperseg,
                "noverlap": noverlap,
                "window": "hann",
                "reference_grid": "full_2049_bin_h1_frf",
                "reference_point_count": int(len(frequency)),
            },
        ))
    if {len(curve.frequency) for curve in curves} != {2049}:
        raise ValueError("Package C expects 2049-bin aluminium FRF references")
    return curves


def _initial_uniform(length: int, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, int(count), dtype=np.int64))


def _uniform_batch(observed: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    selected = []
    current = np.sort(np.asarray(observed, dtype=np.int64))
    candidate_set = set(int(value) for value in candidates)
    while len(selected) < count and candidate_set:
        gaps = []
        for left, right in zip(current[:-1], current[1:]):
            available = [value for value in candidate_set if int(left) < value < int(right)]
            if available:
                midpoint = (int(left) + int(right)) / 2.0
                value = min(available, key=lambda item: (abs(item - midpoint), item))
                gaps.append((int(right) - int(left), -abs(value - midpoint), value))
        if not gaps:
            value = min(candidate_set)
        else:
            value = max(gaps, key=lambda item: (item[0], item[1], -item[2]))[2]
        selected.append(int(value))
        candidate_set.remove(int(value))
        current = np.sort(np.append(current, int(value)))
    return np.asarray(selected, dtype=np.int64)


def _curvature_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
) -> np.ndarray:
    if frequency is None or response is None or len(observed) < 3:
        return _uniform_batch(observed, candidates, count)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    x_observed = np.asarray(frequency, dtype=np.float64)[observed]
    y_observed = np.log1p(np.abs(_as_complex(response)[observed]))
    curvature = np.zeros(len(observed), dtype=np.float64)
    for index in range(1, len(observed) - 1):
        left_dx = max(float(x_observed[index] - x_observed[index - 1]), np.finfo(float).eps)
        right_dx = max(float(x_observed[index + 1] - x_observed[index]), np.finfo(float).eps)
        left_slope = (y_observed[index] - y_observed[index - 1]) / left_dx
        right_slope = (y_observed[index + 1] - y_observed[index]) / right_dx
        span = max(float(x_observed[index + 1] - x_observed[index - 1]), np.finfo(float).eps)
        curvature[index] = abs(float(right_slope - left_slope)) / span
    max_curvature = max(float(np.max(curvature)), np.finfo(float).eps)
    positions = np.searchsorted(observed, candidates)
    scores = []
    for candidate, position in zip(candidates, positions):
        left_pos = max(0, int(position) - 1)
        right_pos = min(len(observed) - 1, int(position))
        left = int(observed[left_pos])
        right = int(observed[right_pos])
        gap = max(abs(right - left), 1)
        midpoint_weight = 1.0
        if right != left:
            midpoint_weight += min(candidate - left, right - candidate) / gap
        local_curvature = max(float(curvature[left_pos]), float(curvature[right_pos]))
        scores.append(((1.0 + local_curvature / max_curvature) * gap * midpoint_weight, -int(candidate)))
    order = np.argsort([(-score, tie) for score, tie in scores], axis=0)[:, 0]
    return np.sort(candidates[order[:count]].astype(np.int64))


def _peak_refine_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
) -> np.ndarray:
    if frequency is None or response is None or len(observed) < 3:
        return _curvature_batch(observed, candidates, count, frequency, response)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    magnitude = np.log1p(np.abs(_as_complex(response)))
    predicted = np.interp(frequency, frequency[observed], magnitude[observed])
    span = float(np.ptp(predicted))
    if not np.isfinite(span) or span <= np.finfo(float).eps:
        return _curvature_batch(observed, candidates, count, frequency, response)
    candidate_set = set(int(value) for value in candidates)
    selected = []
    gap_targets = []
    for left, right in zip(observed[:-1], observed[1:]):
        available = [value for value in candidate_set if int(left) < value < int(right)]
        if not available:
            continue
        midpoint = (int(left) + int(right)) / 2.0
        candidate = min(available, key=lambda item: (abs(item - midpoint), item))
        endpoint_level = max(float(magnitude[int(left)]), float(magnitude[int(right)]))
        gap = int(right) - int(left)
        midpoint_weight = 1.0 + min(candidate - int(left), int(right) - candidate) / max(gap, 1)
        gap_targets.append((endpoint_level * gap * midpoint_weight, int(candidate)))
    for _, candidate in sorted(gap_targets, key=lambda item: (-item[0], item[1])):
        if len(selected) >= count:
            break
        if candidate in candidate_set:
            selected.append(candidate)
            candidate_set.remove(candidate)
    peaks, properties = find_peaks(
        predicted,
        prominence=max(0.05 * span, np.finfo(float).eps),
        distance=max(2, len(frequency) // 200),
    )
    if len(peaks):
        prominences = np.asarray(properties.get("prominences"))
        if prominences.size != len(peaks):
            prominences = peak_prominences(predicted, peaks)[0]
        target_order = list(peaks[np.argsort(-prominences)])
    else:
        target_order = list(np.argsort(-predicted)[: max(count * 4, count)])
    for target in target_order:
        if len(selected) >= count or not candidate_set:
            break
        nearest = min(candidate_set, key=lambda item: (abs(item - int(target)), item))
        selected.append(int(nearest))
        candidate_set.remove(int(nearest))
    if len(selected) < count and candidate_set:
        filler = _curvature_batch(
            np.sort(np.concatenate([observed, np.asarray(selected, dtype=np.int64)])),
            np.asarray(sorted(candidate_set), dtype=np.int64),
            count - len(selected),
            frequency,
            response,
        )
        selected.extend(int(value) for value in filler)
    return np.sort(np.asarray(selected, dtype=np.int64))


def _coverage_peak_refine_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
) -> np.ndarray:
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    coverage_count = max(1, int(math.ceil(count / 2.0)))
    coverage = _uniform_batch(observed, candidates, min(coverage_count, count))
    remaining_candidates = np.setdiff1d(candidates, coverage, assume_unique=False)
    remaining_count = count - len(coverage)
    if remaining_count <= 0 or not len(remaining_candidates):
        return np.sort(coverage.astype(np.int64))
    refined_observed = np.sort(np.unique(np.concatenate([observed, coverage])))
    refine = _peak_refine_batch(
        refined_observed,
        remaining_candidates,
        remaining_count,
        frequency,
        response,
    )
    return np.sort(np.unique(np.concatenate([coverage, refine])).astype(np.int64))


def _bootstrap_peak_uncertainty_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
) -> np.ndarray:
    if frequency is None or response is None or len(observed) < 5:
        return _coverage_peak_refine_batch(observed, candidates, count, frequency, response)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    magnitude = np.log1p(np.abs(_as_complex(response)))
    base_prediction = np.interp(frequency, frequency[observed], magnitude[observed])
    ensemble = []
    interior_positions = np.arange(1, len(observed) - 1, dtype=np.int64)
    for offset in range(min(8, max(2, len(interior_positions)))):
        keep = np.ones(len(observed), dtype=bool)
        if len(interior_positions):
            removed = interior_positions[offset::max(2, min(4, len(interior_positions)))]
            keep[removed] = False
        subset = observed[keep]
        if len(subset) >= 3:
            ensemble.append(np.interp(frequency, frequency[subset], magnitude[subset]))
    if len(ensemble) < 2:
        return _coverage_peak_refine_batch(observed, candidates, count, frequency, response)
    uncertainty = np.std(np.asarray(ensemble), axis=0)
    uncertainty_span = max(float(np.ptp(uncertainty)), np.finfo(float).eps)
    prediction_span = max(float(np.ptp(base_prediction)), np.finfo(float).eps)

    candidate_set = set(int(value) for value in candidates)
    selected = []
    coverage_count = max(1, int(math.ceil(count / 2.0)))
    coverage = _uniform_batch(observed, candidates, min(coverage_count, count))
    for value in coverage:
        if int(value) in candidate_set:
            selected.append(int(value))
            candidate_set.remove(int(value))
    positions = np.searchsorted(observed, candidates)
    scores = []
    for candidate, position in zip(candidates, positions):
        if int(candidate) not in candidate_set:
            continue
        left_pos = max(0, int(position) - 1)
        right_pos = min(len(observed) - 1, int(position))
        gap = max(int(observed[right_pos]) - int(observed[left_pos]), 1)
        uncertainty_score = float(uncertainty[int(candidate)] / uncertainty_span)
        peak_score = float((base_prediction[int(candidate)] - np.min(base_prediction)) / prediction_span)
        score = (0.65 * uncertainty_score + 0.35 * peak_score) * gap
        scores.append((score, -int(candidate), int(candidate)))
    for _, _, candidate in sorted(scores, key=lambda item: (-item[0], item[1])):
        if len(selected) >= count:
            break
        if candidate in candidate_set:
            selected.append(candidate)
            candidate_set.remove(candidate)
    if len(selected) < count and candidate_set:
        filler = _coverage_peak_refine_batch(
            np.sort(np.concatenate([observed, np.asarray(selected, dtype=np.int64)])),
            np.asarray(sorted(candidate_set), dtype=np.int64),
            count - len(selected),
            frequency,
            response,
        )
        selected.extend(int(value) for value in filler)
    return np.sort(np.asarray(selected, dtype=np.int64))


def _modal_parameter_uncertainty_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
) -> np.ndarray:
    if frequency is None or response is None or len(observed) < 5:
        return _coverage_peak_refine_batch(observed, candidates, count, frequency, response)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    magnitude = np.log1p(np.abs(_as_complex(response)))

    coverage_count = max(1, int(math.ceil(count / 2.0)))
    coverage = _uniform_batch(observed, candidates, min(coverage_count, count))
    selected = [int(value) for value in coverage]
    candidate_set = set(int(value) for value in candidates)
    for value in selected:
        candidate_set.discard(value)
    remaining_count = count - len(selected)
    if remaining_count <= 0 or not candidate_set:
        return np.sort(np.asarray(selected, dtype=np.int64))

    refined_observed = np.sort(np.unique(np.concatenate([observed, np.asarray(selected, dtype=np.int64)])))
    ensemble_peaks: list[np.ndarray] = []
    interior_positions = np.arange(1, len(refined_observed) - 1, dtype=np.int64)
    for offset in range(min(10, max(2, len(interior_positions)))):
        keep = np.ones(len(refined_observed), dtype=bool)
        if len(interior_positions):
            removed = interior_positions[offset::max(2, min(4, len(interior_positions)))]
            keep[removed] = False
        subset = refined_observed[keep]
        if len(subset) < 3:
            continue
        prediction = np.interp(frequency, frequency[subset], magnitude[subset])
        span = float(np.ptp(prediction))
        if not np.isfinite(span) or span <= np.finfo(float).eps:
            continue
        peaks, properties = find_peaks(
            prediction,
            prominence=max(0.05 * span, np.finfo(float).eps),
            distance=max(2, len(frequency) // 200),
        )
        if not len(peaks):
            peaks = np.asarray(np.argsort(-prediction)[:3], dtype=np.int64)
        prominences = np.asarray(properties.get("prominences", np.ones(len(peaks))))
        if prominences.size != len(peaks):
            prominences = np.ones(len(peaks), dtype=np.float64)
        peak_order = np.argsort(frequency[peaks])[:5]
        ensemble_peaks.append(np.asarray(peaks[peak_order], dtype=np.int64))
    if not ensemble_peaks:
        return _coverage_peak_refine_batch(observed, candidates, count, frequency, response)

    target_indices = np.concatenate(ensemble_peaks)
    bandwidth = max(2.0, len(frequency) / 150.0)
    positions = np.searchsorted(refined_observed, candidates)
    scores = []
    for candidate, position in zip(candidates, positions):
        candidate = int(candidate)
        if candidate not in candidate_set:
            continue
        left_pos = max(0, int(position) - 1)
        right_pos = min(len(refined_observed) - 1, int(position))
        gap = max(int(refined_observed[right_pos]) - int(refined_observed[left_pos]), 1)
        distances = np.abs(target_indices - candidate)
        mode_proximity = float(np.sum(np.exp(-(distances / bandwidth) ** 2)))
        score = mode_proximity * gap
        scores.append((score, -candidate, candidate))
    for _, _, candidate in sorted(scores, key=lambda item: (-item[0], item[1])):
        if len(selected) >= count:
            break
        if candidate in candidate_set:
            selected.append(candidate)
            candidate_set.remove(candidate)
    if len(selected) < count and candidate_set:
        filler = _coverage_peak_refine_batch(
            np.sort(np.concatenate([observed, np.asarray(selected, dtype=np.int64)])),
            np.asarray(sorted(candidate_set), dtype=np.int64),
            count - len(selected),
            frequency,
            response,
        )
        selected.extend(int(value) for value in filler)
    return np.sort(np.asarray(selected, dtype=np.int64))


def _coverage_modal_uncertainty_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
) -> np.ndarray:
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    coverage_count = max(1, int(math.ceil(0.75 * count)))
    coverage = _uniform_batch(observed, candidates, min(coverage_count, count))
    selected = [int(value) for value in coverage]
    candidate_set = set(int(value) for value in candidates)
    for value in selected:
        candidate_set.discard(value)
    remaining_count = count - len(selected)
    if remaining_count <= 0 or not candidate_set:
        return np.sort(np.asarray(selected, dtype=np.int64))
    refined_observed = np.sort(np.unique(np.concatenate([observed, np.asarray(selected, dtype=np.int64)])))
    refine = _modal_parameter_uncertainty_batch(
        refined_observed,
        np.asarray(sorted(candidate_set), dtype=np.int64),
        remaining_count,
        frequency,
        response,
    )
    return np.sort(np.unique(np.concatenate([coverage, refine])).astype(np.int64))


def _catalogue_centres_from_response(
    frequency: np.ndarray,
    response: np.ndarray | None,
    *,
    max_modes: int,
) -> list[float]:
    if response is None:
        return []
    events = extract_modal_events(
        frequency,
        response,
        max_modes=max_modes,
        min_frequency_hz=max(5.0, float(frequency[0]) if len(frequency) else 5.0),
    )
    return [float(mode["frequency_hz"]) for mode in events.get("modes", [])[:max_modes]]


def _catalogue_band_coverage_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
    *,
    catalogue_frequencies_hz: list[float] | None,
    band_half_width_hz: float,
    min_band_points: int,
    max_modes: int,
) -> np.ndarray:
    if frequency is None:
        return _uniform_batch(observed, candidates, count)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    centres = [
        float(value) for value in (catalogue_frequencies_hz or _catalogue_centres_from_response(
            frequency,
            response,
            max_modes=max_modes,
        ))
        if np.isfinite(float(value))
    ][:max(1, int(max_modes))]
    if not centres:
        return _uniform_batch(observed, candidates, count)

    band_half_width_hz = max(float(band_half_width_hz), np.finfo(float).eps)
    min_band_points = max(1, int(min_band_points))
    selected: list[int] = []
    candidate_set = set(int(value) for value in candidates)

    while len(selected) < count and candidate_set:
        current = np.sort(np.unique(np.concatenate([
            observed,
            np.asarray(selected, dtype=np.int64),
        ])))
        underfilled = []
        for rank, centre in enumerate(centres):
            observed_band_mask = np.abs(frequency[current] - centre) <= band_half_width_hz
            observed_band_count = int(np.sum(observed_band_mask))
            available = [
                int(index) for index in candidate_set
                if abs(float(frequency[int(index)]) - centre) <= band_half_width_hz
            ]
            if observed_band_count < min_band_points and available:
                underfilled.append((observed_band_count, rank, centre, available))
        if not underfilled:
            break

        _, _, centre, available = sorted(underfilled, key=lambda item: (item[0], item[1]))[0]
        current_band = current[np.abs(frequency[current] - centre) <= band_half_width_hz]
        target_grid = np.linspace(
            centre - band_half_width_hz,
            centre + band_half_width_hz,
            min_band_points,
            dtype=np.float64,
        )
        if len(current_band):
            distances_to_observed = np.min(
                np.abs(target_grid.reshape(-1, 1) - frequency[current_band].reshape(1, -1)),
                axis=1,
            )
            target = float(target_grid[int(np.argmax(distances_to_observed))])
        else:
            target = float(centre)
        candidate = min(available, key=lambda index: (abs(float(frequency[index]) - target), index))
        selected.append(int(candidate))
        candidate_set.remove(int(candidate))

    if len(selected) < count and candidate_set:
        filler = _uniform_batch(
            np.sort(np.unique(np.concatenate([observed, np.asarray(selected, dtype=np.int64)]))),
            np.asarray(sorted(candidate_set), dtype=np.int64),
            count - len(selected),
        )
        selected.extend(int(value) for value in filler)
    return np.sort(np.asarray(selected[:count], dtype=np.int64))


def _catalogue_band_informative_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
    *,
    catalogue_frequencies_hz: list[float] | None,
    band_half_width_hz: float,
    min_band_points: int,
    max_modes: int,
) -> np.ndarray:
    if frequency is None:
        return _uniform_batch(observed, candidates, count)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    centres = [
        float(value) for value in (catalogue_frequencies_hz or _catalogue_centres_from_response(
            frequency,
            response,
            max_modes=max_modes,
        ))
        if np.isfinite(float(value))
    ][:max(1, int(max_modes))]
    if not centres:
        return _uniform_batch(observed, candidates, count)

    band_half_width_hz = max(float(band_half_width_hz), np.finfo(float).eps)
    min_band_points = max(1, int(min_band_points))
    selected: list[int] = []
    candidate_set = set(int(value) for value in candidates)

    magnitude_score = np.zeros(len(frequency), dtype=np.float64)
    curvature_score = np.zeros(len(frequency), dtype=np.float64)
    if response is not None:
        magnitude = np.log1p(np.abs(_as_complex(response)))
        span = float(np.ptp(magnitude))
        if np.isfinite(span) and span > np.finfo(float).eps:
            magnitude_score = (magnitude - float(np.min(magnitude))) / span
        if len(frequency) >= 5:
            first = np.gradient(magnitude, frequency, edge_order=1)
            second = np.abs(np.gradient(first, frequency, edge_order=1))
            curvature_span = float(np.ptp(second))
            if np.isfinite(curvature_span) and curvature_span > np.finfo(float).eps:
                curvature_score = (second - float(np.min(second))) / curvature_span

    while len(selected) < count and candidate_set:
        current = np.sort(np.unique(np.concatenate([
            observed,
            np.asarray(selected, dtype=np.int64),
        ])))
        band_rows = []
        for rank, centre in enumerate(centres):
            current_band = current[np.abs(frequency[current] - centre) <= band_half_width_hz]
            available = [
                int(index) for index in candidate_set
                if abs(float(frequency[int(index)]) - centre) <= band_half_width_hz
            ]
            if not available:
                continue
            if len(current_band) >= 2:
                local_gap = float(np.max(np.diff(frequency[current_band])))
            else:
                local_gap = 2.0 * band_half_width_hz
            band_rows.append((len(current_band), -local_gap, rank, centre, current_band, available))
        if not band_rows:
            break

        band_count, _, _, centre, current_band, available = sorted(band_rows)[0]
        target_grid_count = max(min_band_points, int(band_count) + 1)
        target_grid = np.linspace(
            centre - band_half_width_hz,
            centre + band_half_width_hz,
            target_grid_count,
            dtype=np.float64,
        )
        if len(current_band):
            distances_to_observed = np.min(
                np.abs(target_grid.reshape(-1, 1) - frequency[current_band].reshape(1, -1)),
                axis=1,
            )
            target = float(target_grid[int(np.argmax(distances_to_observed))])
        else:
            target = float(centre)

        scored = []
        for index in available:
            if len(current_band):
                nearest_distance = float(np.min(np.abs(frequency[current_band] - frequency[index])))
            else:
                nearest_distance = band_half_width_hz
            info = 0.45 * float(curvature_score[index]) + 0.35 * float(magnitude_score[index])
            shoulder = 1.0 - min(abs(float(magnitude_score[index]) - 0.5), 0.5) / 0.5
            score = nearest_distance * (1.0 + info + 0.20 * shoulder)
            target_distance = abs(float(frequency[index]) - target)
            centre_distance = abs(float(frequency[index]) - centre)
            scored.append((score, target_distance, centre_distance, index))
        candidate = int(sorted(scored, key=lambda item: (-item[0], item[1], item[2], item[3]))[0][-1])
        selected.append(candidate)
        candidate_set.remove(candidate)

    if len(selected) < count and candidate_set:
        filler = _uniform_batch(
            np.sort(np.unique(np.concatenate([observed, np.asarray(selected, dtype=np.int64)]))),
            np.asarray(sorted(candidate_set), dtype=np.int64),
            count - len(selected),
        )
        selected.extend(int(value) for value in filler)
    return np.sort(np.asarray(selected[:count], dtype=np.int64))


def _catalogue_band_half_power_online_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
    *,
    catalogue_frequencies_hz: list[float] | None,
    band_half_width_hz: float,
    min_band_points: int,
    max_modes: int,
) -> np.ndarray:
    if frequency is None:
        return _uniform_batch(observed, candidates, count)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    if catalogue_frequencies_hz is None:
        return _catalogue_band_coverage_batch(
            observed,
            candidates,
            count,
            frequency,
            response,
            catalogue_frequencies_hz=None,
            band_half_width_hz=band_half_width_hz,
            min_band_points=min_band_points,
            max_modes=max_modes,
        )
    centres = [
        float(value) for value in catalogue_frequencies_hz
        if np.isfinite(float(value))
    ][:max(1, int(max_modes))]
    if not centres:
        return _uniform_batch(observed, candidates, count)

    band_half_width_hz = max(float(band_half_width_hz), np.finfo(float).eps)
    min_band_points = max(1, int(min_band_points))
    selected: list[int] = []
    candidate_set = set(int(value) for value in candidates)
    observed_magnitude = None
    if response is not None:
        observed_magnitude = np.abs(_as_complex(response))[observed]

    while len(selected) < count and candidate_set:
        current = np.sort(np.unique(np.concatenate([
            observed,
            np.asarray(selected, dtype=np.int64),
        ])))
        band_rows = []
        for rank, centre in enumerate(centres):
            current_band = current[np.abs(frequency[current] - centre) <= band_half_width_hz]
            available = [
                int(index) for index in candidate_set
                if abs(float(frequency[int(index)]) - centre) <= band_half_width_hz
            ]
            if not available:
                continue
            local_gap = (
                float(np.max(np.diff(frequency[current_band])))
                if len(current_band) >= 2 else
                2.0 * band_half_width_hz
            )
            band_rows.append((len(current_band), -local_gap, rank, centre, current_band, available))
        if not band_rows:
            break

        band_count, _, _, centre, current_band, available = sorted(band_rows)[0]
        target_locations = []
        if observed_magnitude is not None and len(current_band) >= 3:
            positions = np.searchsorted(observed, current_band)
            local_frequency = frequency[current_band]
            local_magnitude = observed_magnitude[positions]
            peak_position = int(np.argmax(local_magnitude))
            peak_frequency = float(local_frequency[peak_position])
            peak_magnitude = float(local_magnitude[peak_position])
            half_power = peak_magnitude / math.sqrt(2.0)
            target_locations.append(peak_frequency)
            left_candidates = np.flatnonzero(local_frequency < peak_frequency)
            if len(left_candidates):
                left_position = int(left_candidates[np.argmin(np.abs(local_magnitude[left_candidates] - half_power))])
                target_locations.append(float(local_frequency[left_position]))
            else:
                target_locations.append(max(float(frequency[0]), centre - 0.5 * band_half_width_hz))
            right_candidates = np.flatnonzero(local_frequency > peak_frequency)
            if len(right_candidates):
                right_position = int(right_candidates[np.argmin(np.abs(local_magnitude[right_candidates] - half_power))])
                target_locations.append(float(local_frequency[right_position]))
            else:
                target_locations.append(min(float(frequency[-1]), centre + 0.5 * band_half_width_hz))
        if not target_locations:
            target_grid_count = max(min_band_points, int(band_count) + 1)
            target_locations = list(np.linspace(
                centre - band_half_width_hz,
                centre + band_half_width_hz,
                target_grid_count,
                dtype=np.float64,
            ))
        if len(current_band):
            distances = np.min(
                np.abs(np.asarray(target_locations, dtype=np.float64).reshape(-1, 1) - frequency[current_band].reshape(1, -1)),
                axis=1,
            )
            target = float(target_locations[int(np.argmax(distances))])
        else:
            target = float(centre)

        if len(current_band):
            candidate = max(
                available,
                key=lambda index: (
                    np.min(np.abs(frequency[current_band] - frequency[int(index)])),
                    -abs(float(frequency[int(index)]) - target),
                    -abs(float(frequency[int(index)]) - centre),
                    -int(index),
                ),
            )
        else:
            candidate = min(available, key=lambda index: (abs(float(frequency[index]) - target), index))
        selected.append(int(candidate))
        candidate_set.remove(int(candidate))

    if len(selected) < count and candidate_set:
        filler = _uniform_batch(
            np.sort(np.unique(np.concatenate([observed, np.asarray(selected, dtype=np.int64)]))),
            np.asarray(sorted(candidate_set), dtype=np.int64),
            count - len(selected),
        )
        selected.extend(int(value) for value in filler)
    return np.sort(np.asarray(selected[:count], dtype=np.int64))


def _observed_low_frequency_limit(
    frequency: np.ndarray,
    *,
    band_half_width_hz: float,
    max_modes: int,
) -> float:
    span = max(8.0 * float(max_modes) * float(band_half_width_hz), 3.0 * float(band_half_width_hz))
    return float(min(float(frequency[-1]), max(float(frequency[0]), 0.0) + span))


def _observed_peak_centres_from_observed(
    frequency: np.ndarray,
    response: np.ndarray,
    observed: np.ndarray,
    *,
    band_half_width_hz: float,
    max_modes: int,
) -> list[float]:
    values = _as_complex(response)
    limit = _observed_low_frequency_limit(
        frequency,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    )
    usable = observed[(frequency[observed] >= 5.0) & (frequency[observed] <= limit)]
    if len(usable) < 3:
        return []
    local_frequency = frequency[usable]
    magnitude = np.log1p(np.abs(values[usable]))
    span = float(np.ptp(magnitude))
    distance = max(1, int(math.ceil(float(band_half_width_hz) / max(float(np.median(np.diff(local_frequency))), np.finfo(float).eps))))
    peaks, properties = find_peaks(
        magnitude,
        prominence=max(0.05 * span, np.finfo(float).eps),
        distance=distance,
    )
    candidates: list[tuple[float, float]] = []
    if len(peaks):
        prominences = np.asarray(properties.get("prominences", np.ones(len(peaks))), dtype=np.float64)
        if prominences.size != len(peaks):
            prominences = np.ones(len(peaks), dtype=np.float64)
        for peak, prominence in zip(peaks, prominences):
            candidates.append((float(prominence), float(local_frequency[int(peak)])))
    centres: list[float] = []
    for _, centre in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if all(abs(float(centre) - existing) > float(band_half_width_hz) for existing in centres):
            centres.append(float(centre))
        if len(centres) >= int(max_modes):
            break
    return sorted(centres)


def _observed_peak_stratified_centres_from_observed(
    frequency: np.ndarray,
    response: np.ndarray,
    observed: np.ndarray,
    *,
    band_half_width_hz: float,
    max_modes: int,
) -> list[float]:
    values = _as_complex(response)
    limit = _observed_low_frequency_limit(
        frequency,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    )
    usable = observed[(frequency[observed] >= 5.0) & (frequency[observed] <= limit)]
    if len(usable) < 3:
        return []
    local_frequency = frequency[usable]
    magnitude = np.log1p(np.abs(values[usable]))
    span = float(np.ptp(magnitude))
    distance = max(1, int(math.ceil(float(band_half_width_hz) / max(float(np.median(np.diff(local_frequency))), np.finfo(float).eps))))
    peaks, properties = find_peaks(
        magnitude,
        prominence=max(0.05 * span, np.finfo(float).eps),
        distance=distance,
    )
    candidates: list[tuple[float, float]] = []
    if len(peaks):
        prominences = np.asarray(properties.get("prominences", np.ones(len(peaks))), dtype=np.float64)
        if prominences.size != len(peaks):
            prominences = np.ones(len(peaks), dtype=np.float64)
        for peak, prominence in zip(peaks, prominences):
            candidates.append((float(prominence), float(local_frequency[int(peak)])))

    segment_count = max(int(max_modes) * 2, int(max_modes) + 3)
    if segment_count > 1:
        edges = np.linspace(5.0, limit, segment_count + 1, dtype=np.float64)
        magnitude_min = float(np.min(magnitude))
        magnitude_span = max(float(np.ptp(magnitude)), np.finfo(float).eps)
        for left, right in zip(edges[:-1], edges[1:]):
            segment = np.flatnonzero((local_frequency >= left) & (local_frequency <= right))
            if not len(segment):
                continue
            local_peak = int(segment[int(np.argmax(magnitude[segment]))])
            score = 0.5 + float((magnitude[local_peak] - magnitude_min) / magnitude_span)
            candidates.append((score, float(local_frequency[local_peak])))
            midpoint = 0.5 * (float(left) + float(right))
            midpoint_peak = int(segment[int(np.argmin(np.abs(local_frequency[segment] - midpoint)))])
            midpoint_score = 0.25 + 0.5 * float((magnitude[midpoint_peak] - magnitude_min) / magnitude_span)
            candidates.append((midpoint_score, float(local_frequency[midpoint_peak])))

    scored_centres: list[tuple[float, float]] = []
    minimum_separation = 0.5 * float(band_half_width_hz)
    for score, centre in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if all(abs(float(centre) - existing) > minimum_separation for _, existing in scored_centres):
            scored_centres.append((float(score), float(centre)))
        if len(scored_centres) >= 2 * int(max_modes):
            break
    representative_limit = int(max_modes)
    candidate_limit = int(max_modes) + 1
    if len(scored_centres) <= candidate_limit:
        return sorted(float(centre) for _, centre in scored_centres)

    ordered = sorted(scored_centres, key=lambda item: item[1])
    edges = np.linspace(ordered[0][1], ordered[-1][1], representative_limit + 1, dtype=np.float64)
    selected: list[tuple[float, float]] = []
    used: set[float] = set()
    for rank, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        rows = [
            row for row in ordered
            if float(left) <= row[1] <= float(right) and row[1] not in used
        ]
        if not rows:
            continue
        if rank == 0:
            chosen = min(rows, key=lambda item: item[1])
        elif rank >= representative_limit - 2:
            chosen = max(rows, key=lambda item: item[1])
        else:
            midpoint = 0.5 * (float(left) + float(right))
            chosen = sorted(rows, key=lambda item: (-item[0], abs(item[1] - midpoint), item[1]))[0]
        selected.append(chosen)
        used.add(chosen[1])
    if len(selected) < representative_limit:
        for row in sorted(ordered, key=lambda item: (-item[0], item[1])):
            if row[1] in used:
                continue
            selected.append(row)
            used.add(row[1])
            if len(selected) >= representative_limit:
                break
    for row in sorted(ordered, key=lambda item: (-item[0], item[1])):
        if row[1] in used:
            continue
        if all(abs(row[1] - selected_row[1]) > float(band_half_width_hz) for selected_row in selected):
            selected.append(row)
            break
    return sorted(float(centre) for _, centre in selected[:candidate_limit])


def _normalised_observed_features(
    frequency: np.ndarray,
    response: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _as_complex(response)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    local_frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)[observed]
    magnitude = np.log1p(np.abs(values[observed]))
    if len(magnitude) < 2:
        return local_frequency, np.zeros_like(magnitude), np.zeros_like(magnitude)
    magnitude_span = float(np.ptp(magnitude))
    if np.isfinite(magnitude_span) and magnitude_span > np.finfo(float).eps:
        magnitude_score = (magnitude - float(np.min(magnitude))) / magnitude_span
    else:
        magnitude_score = np.zeros_like(magnitude)
    if len(magnitude) >= 5:
        first = np.gradient(magnitude, local_frequency, edge_order=1)
        curvature = np.abs(np.gradient(first, local_frequency, edge_order=1))
        curvature_span = float(np.ptp(curvature))
        if np.isfinite(curvature_span) and curvature_span > np.finfo(float).eps:
            curvature_score = (curvature - float(np.min(curvature))) / curvature_span
        else:
            curvature_score = np.zeros_like(magnitude)
    else:
        curvature_score = np.zeros_like(magnitude)
    return local_frequency, magnitude_score, curvature_score


def _append_observed_edge_candidates(
    rows: list[tuple[float, float, str]],
    frequency: np.ndarray,
    response: np.ndarray,
    observed: np.ndarray,
    *,
    low_limit: float,
    max_modes: int,
) -> None:
    values = _as_complex(response)
    usable = observed[(frequency[observed] >= 5.0) & (frequency[observed] <= low_limit)]
    if len(usable) < 3:
        return
    local_frequency = frequency[usable]
    magnitude = np.log1p(np.abs(values[usable]))
    segment_count = max(2, int(max_modes))
    edges = np.linspace(float(local_frequency[0]), float(local_frequency[-1]), segment_count + 1, dtype=np.float64)
    for rank, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        segment = np.flatnonzero((local_frequency >= left) & (local_frequency <= right))
        if not len(segment):
            continue
        local_peak = int(segment[int(np.argmax(magnitude[segment]))])
        source = "edge_low" if rank == 0 else ("edge_high" if rank == segment_count - 1 else "segment_peak")
        rows.append((0.75, float(local_frequency[local_peak]), source))


def _observed_stability_ranked_centres_from_observed(
    frequency: np.ndarray,
    response: np.ndarray,
    observed: np.ndarray,
    *,
    band_half_width_hz: float,
    max_modes: int,
) -> list[float]:
    """Rank observed-only modal candidates by stability and half-power usefulness."""
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    observed = observed[(observed >= 0) & (observed < len(frequency))]
    if len(observed) < 3:
        return []
    band_half_width_hz = max(float(band_half_width_hz), np.finfo(float).eps)
    max_modes = max(1, int(max_modes))
    low_limit = _observed_low_frequency_limit(
        frequency,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    )
    usable = observed[(frequency[observed] >= 5.0) & (frequency[observed] <= low_limit)]
    if len(usable) < 3:
        return []

    candidate_rows: list[tuple[float, float, str]] = []
    for centre in _observed_peak_centres_from_observed(
        frequency,
        response,
        observed,
        band_half_width_hz=band_half_width_hz,
        max_modes=2 * max_modes,
    ):
        candidate_rows.append((1.25, float(centre), "prominence"))
    for centre in _observed_peak_stratified_centres_from_observed(
        frequency,
        response,
        observed,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    ):
        candidate_rows.append((1.00, float(centre), "stratified"))
    _append_observed_edge_candidates(
        candidate_rows,
        frequency,
        response,
        observed,
        low_limit=low_limit,
        max_modes=max_modes,
    )

    # Deterministic prefix probes reward candidates that persist as acquisition grows.
    usable_positions = np.searchsorted(observed, usable)
    for fraction in (0.60, 0.75, 0.90):
        cutoff = max(3, int(math.ceil(len(usable_positions) * fraction)))
        prefix = observed[: int(usable_positions[min(cutoff - 1, len(usable_positions) - 1)]) + 1]
        if len(prefix) < 3:
            continue
        for centre in _observed_peak_centres_from_observed(
            frequency,
            response,
            prefix,
            band_half_width_hz=band_half_width_hz,
            max_modes=max_modes,
        ):
            candidate_rows.append((0.85, float(centre), "prefix_prominence"))
        for centre in _observed_peak_stratified_centres_from_observed(
            frequency,
            response,
            prefix,
            band_half_width_hz=band_half_width_hz,
            max_modes=max_modes,
        ):
            candidate_rows.append((0.65, float(centre), "prefix_stratified"))

    if not candidate_rows:
        return []

    merge_radius = 0.5 * band_half_width_hz
    groups: list[dict] = []
    for base_score, centre, source in sorted(candidate_rows, key=lambda item: item[1]):
        if not np.isfinite(float(centre)):
            continue
        matched = None
        for group in groups:
            if abs(float(centre) - float(group["centre"])) <= merge_radius:
                matched = group
                break
        if matched is None:
            groups.append({
                "centre": float(centre),
                "scores": [float(base_score)],
                "sources": {str(source)},
                "frequencies": [float(centre)],
            })
        else:
            matched["scores"].append(float(base_score))
            matched["sources"].add(str(source))
            matched["frequencies"].append(float(centre))
            weights = np.asarray(matched["scores"], dtype=np.float64)
            matched["centre"] = float(np.average(np.asarray(matched["frequencies"], dtype=np.float64), weights=weights))

    local_frequency, magnitude_score, curvature_score = _normalised_observed_features(frequency, response, usable)
    values = _as_complex(response)
    observed_magnitude = np.log1p(np.abs(values[usable]))
    magnitude_min = float(np.min(observed_magnitude))
    magnitude_span = max(float(np.ptp(observed_magnitude)), np.finfo(float).eps)
    scored: list[tuple[float, float, dict]] = []
    for group in groups:
        centre = float(group["centre"])
        nearest = int(np.argmin(np.abs(local_frequency - centre)))
        band = np.flatnonzero(np.abs(local_frequency - centre) <= band_half_width_hz)
        if len(band):
            band_frequency = local_frequency[band]
            band_magnitude = observed_magnitude[band]
            peak_position = int(np.argmax(band_magnitude))
            peak_frequency = float(band_frequency[peak_position])
            peak_magnitude = float(band_magnitude[peak_position])
            half_power_log = magnitude_min + 0.5 * (peak_magnitude - magnitude_min)
            left = np.flatnonzero(band_frequency < peak_frequency)
            right = np.flatnonzero(band_frequency > peak_frequency)
            side_support = float(bool(len(left)) + bool(len(right))) / 2.0
            shoulder_support = 0.0
            if len(left):
                shoulder_support += 0.5 * (1.0 - min(abs(float(np.min(np.abs(band_magnitude[left] - half_power_log))) / magnitude_span), 1.0))
            if len(right):
                shoulder_support += 0.5 * (1.0 - min(abs(float(np.min(np.abs(band_magnitude[right] - half_power_log))) / magnitude_span), 1.0))
            band_density = min(float(len(band)) / max(float(max_modes), 1.0), 1.0)
        else:
            side_support = 0.0
            shoulder_support = 0.0
            band_density = 0.0

        source_count = len(group["sources"])
        source_score = min(float(source_count) / 4.0, 1.0)
        base_score = min(float(np.sum(group["scores"])) / 3.0, 1.5)
        edge_score = 0.0
        if centre <= 5.0 + 2.0 * band_half_width_hz:
            edge_score = 0.35
        elif centre >= low_limit - 2.0 * band_half_width_hz:
            edge_score = 0.25
        score = (
            1.00 * base_score
            + 1.10 * source_score
            + 0.85 * float(magnitude_score[nearest])
            + 0.70 * float(curvature_score[nearest])
            + 0.70 * side_support
            + 0.55 * shoulder_support
            + 0.35 * band_density
            + edge_score
        )
        scored.append((score, centre, group))

    selected: list[tuple[float, float, dict]] = []
    for score, centre, group in sorted(scored, key=lambda item: (-item[0], item[1])):
        if any(abs(float(centre) - existing[1]) < band_half_width_hz for existing in selected):
            continue
        selected.append((float(score), float(centre), group))
        if len(selected) >= max_modes:
            break
    if len(selected) < max_modes:
        for score, centre, group in sorted(scored, key=lambda item: (-item[0], item[1])):
            if any(abs(float(centre) - existing[1]) < merge_radius for existing in selected):
                continue
            selected.append((float(score), float(centre), group))
            if len(selected) >= max_modes:
                break
    return sorted(float(centre) for _, centre, _ in selected[:max_modes])


def _frequency_stratum_id(value_hz: float, min_hz: float, max_hz: float) -> str:
    span = max(float(max_hz) - float(min_hz), np.finfo(float).eps)
    position = (float(value_hz) - float(min_hz)) / span
    if position < 1.0 / 3.0:
        return "low"
    if position < 2.0 / 3.0:
        return "mid"
    return "high"


def _observed_stratified_budget_candidate_rows_from_observed(
    frequency: np.ndarray,
    response: np.ndarray,
    observed: np.ndarray,
    *,
    band_half_width_hz: float,
    max_modes: int,
) -> list[dict]:
    """Return observed-only modal candidates with protected frequency strata."""
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    observed = np.sort(np.unique(np.asarray(observed, dtype=np.int64)))
    observed = observed[(observed >= 0) & (observed < len(frequency))]
    if len(observed) < 3:
        return []
    band_half_width_hz = max(float(band_half_width_hz), np.finfo(float).eps)
    max_modes = max(1, int(max_modes))
    min_hz = max(5.0, float(frequency[0]))
    max_hz = _observed_low_frequency_limit(
        frequency,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    )
    usable = observed[(frequency[observed] >= min_hz) & (frequency[observed] <= max_hz)]
    if len(usable) < 3:
        return []

    values = _as_complex(response)
    observed_frequency = frequency[usable]
    magnitude = np.log1p(np.abs(values[usable]))
    magnitude_min = float(np.min(magnitude))
    magnitude_span = max(float(np.ptp(magnitude)), np.finfo(float).eps)
    median_step = max(float(np.median(np.diff(observed_frequency))), np.finfo(float).eps)
    distance = max(1, int(math.ceil(0.5 * band_half_width_hz / median_step)))
    prominence_floor = max(0.02 * magnitude_span, np.finfo(float).eps)

    candidates: list[dict] = []
    peaks, properties = find_peaks(magnitude, prominence=prominence_floor, distance=distance)
    prominences = np.asarray(properties.get("prominences", np.ones(len(peaks))), dtype=np.float64)
    if prominences.size != len(peaks):
        prominences = np.ones(len(peaks), dtype=np.float64)
    for peak, prominence in zip(peaks, prominences):
        peak = int(peak)
        centre = float(observed_frequency[peak])
        band = np.flatnonzero(np.abs(observed_frequency - centre) <= band_half_width_hz)
        left = np.flatnonzero((observed_frequency < centre) & (np.abs(observed_frequency - centre) <= band_half_width_hz))
        right = np.flatnonzero((observed_frequency > centre) & (np.abs(observed_frequency - centre) <= band_half_width_hz))
        shoulder_score = 0.5 * float(bool(len(left))) + 0.5 * float(bool(len(right)))
        sparse_priority = 1.0 / max(len(band), 1)
        if 0 < peak < len(magnitude) - 1:
            left_slope = float(magnitude[peak] - magnitude[peak - 1])
            right_slope = float(magnitude[peak] - magnitude[peak + 1])
            curvature_score = abs(left_slope - right_slope) / magnitude_span
        else:
            curvature_score = 0.0
        score = (
            1.0
            + float(prominence) / magnitude_span
            + 0.50 * curvature_score
            + 0.25 * shoulder_score
            + 0.25 * sparse_priority
        )
        candidates.append({
            "candidate_frequency_hz": centre,
            "stratum_id": _frequency_stratum_id(centre, min_hz, max_hz),
            "score": float(score),
            "observed_band_point_count": int(len(band)),
            "source": "observed_stratified_budget",
        })

    segment_count = max(3, 2 * max_modes)
    edges = np.linspace(min_hz, max_hz, segment_count + 1, dtype=np.float64)
    for left_edge, right_edge in zip(edges[:-1], edges[1:]):
        segment = np.flatnonzero((observed_frequency >= left_edge) & (observed_frequency <= right_edge))
        if not len(segment):
            continue
        local_peak = int(segment[int(np.argmax(magnitude[segment]))])
        centre = float(observed_frequency[local_peak])
        band = np.flatnonzero(np.abs(observed_frequency - centre) <= band_half_width_hz)
        score = 0.55 + float((magnitude[local_peak] - magnitude_min) / magnitude_span) + 0.10 / max(len(band), 1)
        candidates.append({
            "candidate_frequency_hz": centre,
            "stratum_id": _frequency_stratum_id(centre, min_hz, max_hz),
            "score": float(score),
            "observed_band_point_count": int(len(band)),
            "source": "observed_stratified_budget",
        })

    stratum_edges = np.linspace(min_hz, max_hz, 4, dtype=np.float64)
    for left_edge, right_edge in zip(stratum_edges[:-1], stratum_edges[1:]):
        segment = np.flatnonzero((observed_frequency >= left_edge) & (observed_frequency <= right_edge))
        if not len(segment):
            continue
        for local_index in (int(segment[0]), int(segment[-1])):
            centre = float(observed_frequency[local_index])
            band = np.flatnonzero(np.abs(observed_frequency - centre) <= band_half_width_hz)
            score = 0.90 + float((magnitude[local_index] - magnitude_min) / magnitude_span) + 0.10 / max(len(band), 1)
            candidates.append({
                "candidate_frequency_hz": centre,
                "stratum_id": _frequency_stratum_id(centre, min_hz, max_hz),
                "score": float(score),
                "observed_band_point_count": int(len(band)),
                "source": "observed_stratified_budget",
            })
        edge_probes = (
            (max(float(left_edge), float(observed_frequency[int(segment[0])]) - band_half_width_hz), int(segment[0])),
            (min(float(right_edge), float(observed_frequency[int(segment[-1])]) + band_half_width_hz), int(segment[-1])),
        )
        for centre, local_index in edge_probes:
            band = np.flatnonzero(np.abs(observed_frequency - centre) <= band_half_width_hz)
            score = 1.30 + float((magnitude[local_index] - magnitude_min) / magnitude_span) + 0.10 / max(len(band), 1)
            candidates.append({
                "candidate_frequency_hz": float(centre),
                "stratum_id": _frequency_stratum_id(float(centre), min_hz, max_hz),
                "score": float(score),
                "observed_band_point_count": int(len(band)),
                "source": "observed_stratified_budget",
            })

    if not candidates:
        return []

    merge_radius = 0.5 * band_half_width_hz
    deduped: list[dict] = []
    for row in sorted(candidates, key=lambda item: (-float(item["score"]), float(item["candidate_frequency_hz"]))):
        centre = float(row["candidate_frequency_hz"])
        if any(abs(centre - float(existing["candidate_frequency_hz"])) <= merge_radius for existing in deduped):
            continue
        deduped.append(dict(row))

    quota: dict[str, int] = {"low": 1, "mid": 1, "high": 1}
    remaining_slots = max(0, max_modes - sum(quota.values()))
    for stratum in ("mid", "high", "low"):
        if remaining_slots <= 0:
            break
        quota[stratum] += 1
        remaining_slots -= 1

    selected: list[dict] = []
    selected_keys: set[float] = set()
    for stratum in ("low", "mid", "high"):
        rows = [row for row in deduped if row["stratum_id"] == stratum]
        rows = sorted(rows, key=lambda item: (-float(item["score"]), float(item["candidate_frequency_hz"])))
        for row in rows[: quota.get(stratum, 0)]:
            selected.append(row)
            selected_keys.add(float(row["candidate_frequency_hz"]))
    for row in sorted(deduped, key=lambda item: (-float(item["score"]), float(item["candidate_frequency_hz"]))):
        if len(selected) >= max_modes:
            break
        centre = float(row["candidate_frequency_hz"])
        if centre in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(centre)
    return sorted(selected[:max_modes], key=lambda item: float(item["candidate_frequency_hz"]))


def _observed_stratified_budget_centres_from_observed(
    frequency: np.ndarray,
    response: np.ndarray,
    observed: np.ndarray,
    *,
    band_half_width_hz: float,
    max_modes: int,
) -> list[float]:
    return [
        float(row["candidate_frequency_hz"])
        for row in _observed_stratified_budget_candidate_rows_from_observed(
            frequency,
            response,
            observed,
            band_half_width_hz=band_half_width_hz,
            max_modes=max_modes,
        )
    ]


def _observed_peak_half_power_online_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
    *,
    band_half_width_hz: float,
    min_band_points: int,
    max_modes: int,
) -> np.ndarray:
    if frequency is None or response is None:
        return _uniform_batch(observed, candidates, count)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    band_half_width_hz = max(float(band_half_width_hz), np.finfo(float).eps)
    min_band_points = max(1, int(min_band_points))
    max_modes = max(1, int(max_modes))

    low_limit = _observed_low_frequency_limit(
        frequency,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    )
    low_observed = observed[(frequency[observed] >= 0.0) & (frequency[observed] <= low_limit)]
    discovery_target = max(max_modes * min_band_points, 3 * max_modes)
    low_candidates = candidates[(frequency[candidates] >= 0.0) & (frequency[candidates] <= low_limit)]
    if len(low_observed) < discovery_target and len(low_candidates):
        return _uniform_batch(observed, low_candidates, min(count, len(low_candidates)))

    centres = _observed_peak_centres_from_observed(
        frequency,
        response,
        observed,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    )
    extended_discovery_target = min(
        len(low_candidates) + len(low_observed),
        max(discovery_target, 5 * max_modes * min_band_points),
    )
    if len(centres) < max_modes and len(low_observed) < extended_discovery_target and len(low_candidates):
        return _uniform_batch(observed, low_candidates, min(count, len(low_candidates)))
    if not centres:
        return _uniform_batch(observed, low_candidates if len(low_candidates) else candidates, count)
    return _catalogue_band_half_power_online_batch(
        observed,
        candidates,
        count,
        frequency,
        response,
        catalogue_frequencies_hz=centres,
        band_half_width_hz=band_half_width_hz,
        min_band_points=min_band_points,
        max_modes=max(max_modes, len(centres)),
    )


def _observed_peak_stratified_half_power_online_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
    *,
    band_half_width_hz: float,
    min_band_points: int,
    max_modes: int,
) -> np.ndarray:
    if frequency is None or response is None:
        return _uniform_batch(observed, candidates, count)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    band_half_width_hz = max(float(band_half_width_hz), np.finfo(float).eps)
    min_band_points = max(1, int(min_band_points))
    max_modes = max(1, int(max_modes))

    low_limit = _observed_low_frequency_limit(
        frequency,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    )
    low_observed = observed[(frequency[observed] >= 0.0) & (frequency[observed] <= low_limit)]
    discovery_target = max(max_modes * min_band_points, 3 * max_modes)
    low_candidates = candidates[(frequency[candidates] >= 0.0) & (frequency[candidates] <= low_limit)]
    low_coverage_target = min(
        len(low_candidates) + len(low_observed),
        max(discovery_target, 2 * max_modes * min_band_points),
    )
    if len(low_observed) < low_coverage_target and len(low_candidates):
        return _uniform_batch(observed, low_candidates, min(count, len(low_candidates)))
    if len(low_observed) < discovery_target and len(low_candidates):
        return _uniform_batch(observed, low_candidates, min(count, len(low_candidates)))

    centres = _observed_peak_stratified_centres_from_observed(
        frequency,
        response,
        observed,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    )
    extended_discovery_target = min(
        len(low_candidates) + len(low_observed),
        max(discovery_target, 5 * max_modes * min_band_points),
    )
    if len(centres) < max_modes and len(low_observed) < extended_discovery_target and len(low_candidates):
        return _uniform_batch(observed, low_candidates, min(count, len(low_candidates)))
    if not centres:
        return _uniform_batch(observed, low_candidates if len(low_candidates) else candidates, count)
    return _catalogue_band_half_power_online_batch(
        observed,
        candidates,
        count,
        frequency,
        response,
        catalogue_frequencies_hz=centres,
        band_half_width_hz=band_half_width_hz,
        min_band_points=min_band_points,
        max_modes=max(max_modes, len(centres)),
    )


def _observed_stability_ranked_half_power_online_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
    *,
    band_half_width_hz: float,
    min_band_points: int,
    max_modes: int,
) -> np.ndarray:
    if frequency is None or response is None:
        return _uniform_batch(observed, candidates, count)
    observed = np.sort(np.asarray(observed, dtype=np.int64))
    candidates = np.asarray(candidates, dtype=np.int64)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    band_half_width_hz = max(float(band_half_width_hz), np.finfo(float).eps)
    min_band_points = max(1, int(min_band_points))
    max_modes = max(1, int(max_modes))

    low_limit = _observed_low_frequency_limit(
        frequency,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    )
    low_observed = observed[(frequency[observed] >= 0.0) & (frequency[observed] <= low_limit)]
    discovery_target = max(max_modes * min_band_points, 3 * max_modes)
    low_candidates = candidates[(frequency[candidates] >= 0.0) & (frequency[candidates] <= low_limit)]
    if len(low_observed) < discovery_target and len(low_candidates):
        return _uniform_batch(observed, low_candidates, min(count, len(low_candidates)))

    centres = _observed_stability_ranked_centres_from_observed(
        frequency,
        response,
        observed,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    )
    extended_discovery_target = min(
        len(low_candidates) + len(low_observed),
        max(discovery_target, 5 * max_modes * min_band_points),
    )
    if len(centres) < max_modes and len(low_observed) < extended_discovery_target and len(low_candidates):
        return _uniform_batch(observed, low_candidates, min(count, len(low_candidates)))
    if not centres:
        return _uniform_batch(observed, low_candidates if len(low_candidates) else candidates, count)
    return _catalogue_band_half_power_online_batch(
        observed,
        candidates,
        count,
        frequency,
        response,
        catalogue_frequencies_hz=centres,
        band_half_width_hz=band_half_width_hz,
        min_band_points=min_band_points,
        max_modes=max(max_modes, len(centres)),
    )


def _observed_stratified_budget_half_power_online_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    count: int,
    frequency: np.ndarray | None,
    response: np.ndarray | None,
    *,
    band_half_width_hz: float,
    min_band_points: int,
    max_modes: int,
) -> np.ndarray:
    if frequency is None or response is None:
        return _uniform_batch(observed, candidates, count)
    centres = _observed_stratified_budget_centres_from_observed(
        frequency,
        response,
        observed,
        band_half_width_hz=band_half_width_hz,
        max_modes=max_modes,
    )
    if not centres:
        return _uniform_batch(observed, candidates, count)
    return _catalogue_band_half_power_online_batch(
        observed,
        candidates,
        count,
        frequency,
        response,
        catalogue_frequencies_hz=centres,
        band_half_width_hz=band_half_width_hz,
        min_band_points=min_band_points,
        max_modes=max(max_modes, len(centres)),
    )


def build_active_schedule(
    length: int,
    *,
    initial_count: int,
    batch_size: int,
    budgets: list[int],
    strategy: str,
    seed: int,
    frequency: np.ndarray | None = None,
    response: np.ndarray | None = None,
    catalogue_frequencies_hz: list[float] | None = None,
    catalogue_band_half_width_hz: float = 16.0,
    catalogue_min_band_points: int = 7,
    catalogue_max_modes: int = 5,
) -> dict[int, np.ndarray]:
    """Return nested measured-index schedules for a sequential FRF acquisition."""
    length = int(length)
    if length < 2:
        raise ValueError("length must be at least two")
    budgets = sorted({int(budget) for budget in budgets})
    if not budgets:
        raise ValueError("at least one budget is required")
    if min(budgets) < int(initial_count):
        raise ValueError("all budgets must be at least initial_count")
    if max(budgets) > length:
        raise ValueError("budgets cannot exceed the reference-grid length")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if strategy not in {
        "uniform",
        "random",
        "curvature",
        "peak_refine",
        "coverage_peak_refine",
        "bootstrap_peak_uncertainty",
        "modal_parameter_uncertainty",
        "coverage_modal_uncertainty",
        "catalogue_band_coverage",
        "catalogue_band_informative",
        "catalogue_band_half_power_online",
        "observed_peak_half_power_online",
        "observed_peak_stratified_half_power_online",
        "observed_stability_ranked_half_power_online",
        "observed_stratified_budget_half_power_online",
    }:
        raise ValueError(f"unknown active sampling strategy: {strategy}")
    observed = _initial_uniform(length, initial_count)
    rng = np.random.default_rng(_seed("aluminium-c-schedule", strategy, seed, length))
    schedule = {}
    for budget in budgets:
        while len(observed) < budget:
            need = min(int(batch_size), budget - len(observed))
            candidates = np.setdiff1d(np.arange(length, dtype=np.int64), observed, assume_unique=False)
            if strategy == "random":
                batch = np.sort(rng.choice(candidates, size=need, replace=False))
            elif strategy == "coverage_modal_uncertainty":
                batch = _coverage_modal_uncertainty_batch(observed, candidates, need, frequency, response)
            elif strategy == "catalogue_band_coverage":
                batch = _catalogue_band_coverage_batch(
                    observed,
                    candidates,
                    need,
                    frequency,
                    response,
                    catalogue_frequencies_hz=catalogue_frequencies_hz,
                    band_half_width_hz=catalogue_band_half_width_hz,
                    min_band_points=catalogue_min_band_points,
                    max_modes=catalogue_max_modes,
                )
            elif strategy == "catalogue_band_informative":
                batch = _catalogue_band_informative_batch(
                    observed,
                    candidates,
                    need,
                    frequency,
                    response,
                    catalogue_frequencies_hz=catalogue_frequencies_hz,
                    band_half_width_hz=catalogue_band_half_width_hz,
                    min_band_points=catalogue_min_band_points,
                    max_modes=catalogue_max_modes,
                )
            elif strategy == "catalogue_band_half_power_online":
                batch = _catalogue_band_half_power_online_batch(
                    observed,
                    candidates,
                    need,
                    frequency,
                    response,
                    catalogue_frequencies_hz=catalogue_frequencies_hz,
                    band_half_width_hz=catalogue_band_half_width_hz,
                    min_band_points=catalogue_min_band_points,
                    max_modes=catalogue_max_modes,
                )
            elif strategy == "observed_peak_half_power_online":
                batch = _observed_peak_half_power_online_batch(
                    observed,
                    candidates,
                    need,
                    frequency,
                    response,
                    band_half_width_hz=catalogue_band_half_width_hz,
                    min_band_points=catalogue_min_band_points,
                    max_modes=catalogue_max_modes,
                )
            elif strategy == "observed_peak_stratified_half_power_online":
                batch = _observed_peak_stratified_half_power_online_batch(
                    observed,
                    candidates,
                    need,
                    frequency,
                    response,
                    band_half_width_hz=catalogue_band_half_width_hz,
                    min_band_points=catalogue_min_band_points,
                    max_modes=catalogue_max_modes,
                )
            elif strategy == "observed_stability_ranked_half_power_online":
                batch = _observed_stability_ranked_half_power_online_batch(
                    observed,
                    candidates,
                    need,
                    frequency,
                    response,
                    band_half_width_hz=catalogue_band_half_width_hz,
                    min_band_points=catalogue_min_band_points,
                    max_modes=catalogue_max_modes,
                )
            elif strategy == "observed_stratified_budget_half_power_online":
                batch = _observed_stratified_budget_half_power_online_batch(
                    observed,
                    candidates,
                    need,
                    frequency,
                    response,
                    band_half_width_hz=catalogue_band_half_width_hz,
                    min_band_points=catalogue_min_band_points,
                    max_modes=catalogue_max_modes,
                )
            elif strategy == "modal_parameter_uncertainty":
                batch = _modal_parameter_uncertainty_batch(observed, candidates, need, frequency, response)
            elif strategy == "bootstrap_peak_uncertainty":
                batch = _bootstrap_peak_uncertainty_batch(observed, candidates, need, frequency, response)
            elif strategy == "coverage_peak_refine":
                batch = _coverage_peak_refine_batch(observed, candidates, need, frequency, response)
            elif strategy == "peak_refine":
                batch = _peak_refine_batch(observed, candidates, need, frequency, response)
            elif strategy == "curvature":
                batch = _curvature_batch(observed, candidates, need, frequency, response)
            else:
                batch = _uniform_batch(observed, candidates, need)
            observed = np.sort(np.unique(np.concatenate([observed, np.asarray(batch, dtype=np.int64)])))
        schedule[int(budget)] = observed.copy()
    return schedule


def complex_nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_complex = _as_complex(truth)
    prediction_complex = _as_complex(prediction)
    if truth_complex.shape != prediction_complex.shape:
        raise ValueError("truth and prediction shapes differ")
    residual = float(np.sqrt(np.mean(np.abs(prediction_complex - truth_complex) ** 2)))
    centered = truth_complex - np.mean(truth_complex)
    denominator = float(np.sqrt(np.mean(np.abs(centered) ** 2)))
    if denominator <= np.finfo(float).eps:
        denominator = max(float(np.sqrt(np.mean(np.abs(truth_complex) ** 2))), np.finfo(float).eps)
    return residual / denominator


def _finite_metric(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def extract_modal_events(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    *,
    max_modes: int = 5,
    min_frequency_hz: float = 5.0,
) -> dict:
    """Extract dominant FRF modal peaks and antiresonances from a complex response."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    values = _as_complex(response)
    if len(frequency) != len(values):
        raise ValueError("frequency and response lengths must match")
    eligible_mask = frequency >= float(min_frequency_hz)
    eligible_indices = np.flatnonzero(eligible_mask)
    if len(eligible_indices) < 3:
        eligible_indices = np.arange(len(frequency), dtype=np.int64)
    local_frequency = frequency[eligible_indices]
    local_values = values[eligible_indices]
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(local_values), 1e-30))
    span = float(np.ptp(magnitude_db))
    if not np.isfinite(span) or span <= np.finfo(float).eps:
        return {"status": "no_dynamic_range", "mode_count": 0, "modes": [], "antiresonance_frequencies_hz": []}
    distance = max(3, int(len(frequency) / 100))
    prominence_floor = max(0.02 * span, 1.0)
    peaks, properties = find_peaks(magnitude_db, prominence=prominence_floor, distance=distance)
    if len(peaks) < int(max_modes):
        peaks, properties = find_peaks(magnitude_db, prominence=max(0.01 * span, 0.5), distance=distance)
    if len(peaks):
        prominences = np.asarray(properties.get("prominences"))
        if prominences.size != len(peaks):
            prominences = peak_prominences(magnitude_db, peaks)[0]
        widths, _, left_ips, right_ips = peak_widths(magnitude_db, peaks, rel_height=0.5)
        sample_axis = np.arange(len(local_frequency), dtype=np.float64)
        left_hz = np.interp(left_ips, sample_axis, local_frequency)
        right_hz = np.interp(right_ips, sample_axis, local_frequency)
        order = np.argsort(local_frequency[peaks])[: int(max_modes)]
        modes = []
        for item in sorted(order, key=lambda idx: float(local_frequency[int(peaks[idx])])):
            peak_index = int(eligible_indices[int(peaks[item])])
            width_hz = max(float(right_hz[item] - left_hz[item]), 0.0)
            modes.append({
                "index": peak_index,
                "frequency_hz": float(frequency[peak_index]),
                "peak_amplitude": float(np.abs(values[peak_index])),
                "peak_magnitude_db": float(20.0 * np.log10(max(float(np.abs(values[peak_index])), 1e-30))),
                "prominence_db": float(prominences[item]),
                "width_hz": width_hz,
                "damping_ratio_proxy": (
                    float(width_hz / (2.0 * abs(frequency[peak_index])))
                    if frequency[peak_index] != 0.0 else None
                ),
            })
    else:
        modes = []
    valleys, _ = find_peaks(-magnitude_db, prominence=max(0.05 * span, 1.0), distance=distance)
    return {
        "status": "ok",
        "mode_count": int(len(modes)),
        "modes": modes,
        "antiresonance_frequencies_hz": [
            float(frequency[int(eligible_indices[index])])
            for index in valleys[: int(max_modes)]
        ],
        "detector": "log_magnitude_prominence_with_half_prominence_width",
        "min_frequency_hz": float(min_frequency_hz),
    }


def _match_modes(truth_modes: list[dict], predicted_modes: list[dict], tolerance_hz: float) -> list[tuple[dict, dict]]:
    candidates = sorted(
        (
            abs(float(truth["frequency_hz"]) - float(predicted["frequency_hz"])),
            truth_index,
            predicted_index,
        )
        for truth_index, truth in enumerate(truth_modes)
        for predicted_index, predicted in enumerate(predicted_modes)
        if abs(float(truth["frequency_hz"]) - float(predicted["frequency_hz"])) <= tolerance_hz
    )
    used_truth = set()
    used_prediction = set()
    matches = []
    for _, truth_index, predicted_index in candidates:
        if truth_index in used_truth or predicted_index in used_prediction:
            continue
        used_truth.add(truth_index)
        used_prediction.add(predicted_index)
        matches.append((truth_modes[truth_index], predicted_modes[predicted_index]))
    return matches


def aluminium_response_metrics(
    frequency_hz: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict:
    """Compute response- and modal-level metrics for Package C."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    if len(frequency) != len(truth) or len(frequency) != len(prediction):
        raise ValueError("frequency, truth, and prediction lengths must match")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        return {"status": "numerical_failure"}
    truth_events = extract_modal_events(frequency, truth)
    predicted_events = extract_modal_events(frequency, prediction)
    truth_modes = truth_events.get("modes", [])
    predicted_modes = predicted_events.get("modes", [])
    resolution = float(np.median(np.diff(frequency))) if len(frequency) > 1 else 1.0
    truth_widths = [
        float(mode["width_hz"]) for mode in truth_modes
        if mode.get("width_hz") is not None and math.isfinite(float(mode["width_hz"])) and float(mode["width_hz"]) > 0.0
    ]
    tolerance_hz = max(5.0 * resolution, 2.0 * (float(np.median(truth_widths)) if truth_widths else resolution))
    matches = _match_modes(truth_modes, predicted_modes, tolerance_hz)
    peak_mask = np.zeros(len(frequency), dtype=bool)
    for mode in truth_modes:
        width = mode.get("width_hz")
        half_width = max(2.5 * resolution, 1.5 * float(width or resolution))
        peak_mask |= np.abs(frequency - float(mode["frequency_hz"])) <= half_width
    if not np.any(peak_mask):
        peak_mask = np.ones(len(frequency), dtype=bool)

    frequency_errors = [
        abs(float(truth_mode["frequency_hz"]) - float(predicted_mode["frequency_hz"]))
        for truth_mode, predicted_mode in matches
    ]
    width_errors = []
    amplitude_errors = []
    for truth_mode, predicted_mode in matches:
        truth_width = truth_mode.get("width_hz")
        predicted_width = predicted_mode.get("width_hz")
        if truth_width is not None and predicted_width is not None and float(truth_width) > 0.0:
            width_errors.append(abs(float(predicted_width) - float(truth_width)) / max(abs(float(truth_width)), np.finfo(float).eps))
        truth_amplitude = truth_mode.get("peak_amplitude")
        predicted_amplitude = predicted_mode.get("peak_amplitude")
        if truth_amplitude is not None and predicted_amplitude is not None and float(truth_amplitude) > 0.0:
            amplitude_errors.append(abs(float(predicted_amplitude) - float(truth_amplitude)) / max(abs(float(truth_amplitude)), np.finfo(float).eps))
    return {
        "status": "ok",
        "complex_nrmse": complex_nrmse(truth, prediction),
        "peak_window_complex_nrmse": complex_nrmse(np.asarray(truth)[peak_mask], np.asarray(prediction)[peak_mask]),
        "truth_mode_count": int(len(truth_modes)),
        "predicted_mode_count": int(len(predicted_modes)),
        "matched_mode_count": int(len(matches)),
        "modal_matching_tolerance_hz": float(tolerance_hz),
        "modal_frequency_mae_hz": float(np.mean(frequency_errors)) if frequency_errors else None,
        "modal_width_relative_mae": float(np.mean(width_errors)) if width_errors else None,
        "modal_peak_amplitude_relative_mae": float(np.mean(amplitude_errors)) if amplitude_errors else None,
        "truth_events": truth_events,
        "predicted_events": predicted_events,
    }


def aluminium_truth_modal_catalogue(
    curves: list[MeasuredCurve],
    *,
    max_modes: int = 5,
    min_frequency_hz: float = 5.0,
) -> list[dict]:
    """Build a stable full-curve modal catalogue for Aluminium FRF audits."""
    rows = []
    for curve in curves:
        events = extract_modal_events(
            curve.frequency,
            curve.response,
            max_modes=max_modes,
            min_frequency_hz=min_frequency_hz,
        )
        for rank, mode in enumerate(events.get("modes", []), start=1):
            rows.append({
                "curve_id": curve.curve_id,
                "point": curve.conditions.get("point"),
                "mode_rank": int(rank),
                "frequency_hz": float(mode["frequency_hz"]),
                "index": int(mode["index"]),
                "width_hz": _finite_metric(mode.get("width_hz")),
                "damping_ratio_proxy": _finite_metric(mode.get("damping_ratio_proxy")),
                "peak_amplitude": _finite_metric(mode.get("peak_amplitude")),
                "peak_magnitude_db": _finite_metric(mode.get("peak_magnitude_db")),
                "prominence_db": _finite_metric(mode.get("prominence_db")),
                "mode_count": int(events.get("mode_count") or 0),
                "detector": events.get("detector"),
                "min_frequency_hz": float(events.get("min_frequency_hz", min_frequency_hz)),
            })
    return rows


def aluminium_modal_recovery_diagnostics(
    records: list[dict],
    *,
    frequency_by_curve: dict[str, np.ndarray] | None = None,
    required_modes: int = 5,
    frequency_threshold_hz: float = 1.0,
    width_relative_threshold: float = 0.10,
) -> list[dict]:
    """Separate sparse-observation, missed-mode, false-mode, and width failures."""
    diagnostics = []
    frequency_by_curve = frequency_by_curve or {}
    for record in records:
        if record.get("status") != "ok":
            continue
        metrics = record.get("metrics", {})
        if not metrics or metrics.get("status", "ok") != "ok":
            continue
        truth_modes = list(metrics.get("truth_events", {}).get("modes", []))
        predicted_modes = list(metrics.get("predicted_events", {}).get("modes", []))
        if not truth_modes:
            continue
        target_modes = max(1, min(int(required_modes), len(truth_modes)))
        truth_modes = truth_modes[:target_modes]
        tolerance_hz = _finite_metric(metrics.get("modal_matching_tolerance_hz"))
        if tolerance_hz is None:
            widths = [
                float(mode["width_hz"]) for mode in truth_modes
                if mode.get("width_hz") is not None and math.isfinite(float(mode["width_hz"]))
            ]
            tolerance_hz = max(1.0, 2.0 * (float(np.median(widths)) if widths else 1.0))
        matches = _match_modes(truth_modes, predicted_modes, float(tolerance_hz))
        matched_truth_ids = {id(truth_mode) for truth_mode, _ in matches}
        matched_predicted_ids = {id(predicted_mode) for _, predicted_mode in matches}
        missed_truth = [
            mode for mode in truth_modes
            if id(mode) not in matched_truth_ids
        ]
        false_predicted = [
            mode for mode in predicted_modes
            if id(mode) not in matched_predicted_ids
        ]

        observed_distances = []
        observed_covered = 0
        curve_frequency = frequency_by_curve.get(str(record.get("curve_id")))
        observed_indices = record.get("observed_indices")
        if curve_frequency is not None and observed_indices is not None:
            frequency = np.asarray(curve_frequency, dtype=np.float64).reshape(-1)
            observed = np.asarray(observed_indices, dtype=np.int64)
            observed = observed[(observed >= 0) & (observed < len(frequency))]
            observed_frequency = frequency[np.unique(observed)] if len(observed) else np.asarray([], dtype=np.float64)
            for mode in truth_modes:
                width = _finite_metric(mode.get("width_hz")) or 0.0
                coverage_radius = max(float(tolerance_hz), 0.5 * float(width))
                if len(observed_frequency):
                    distance = float(np.min(np.abs(observed_frequency - float(mode["frequency_hz"]))))
                    observed_distances.append(distance)
                    if distance <= coverage_radius:
                        observed_covered += 1

        width_errors = []
        frequency_errors = []
        for truth_mode, predicted_mode in matches:
            frequency_errors.append(abs(float(predicted_mode["frequency_hz"]) - float(truth_mode["frequency_hz"])))
            truth_width = _finite_metric(truth_mode.get("width_hz"))
            predicted_width = _finite_metric(predicted_mode.get("width_hz"))
            if truth_width is not None and predicted_width is not None and truth_width > 0.0:
                width_errors.append(abs(predicted_width - truth_width) / max(abs(truth_width), np.finfo(float).eps))

        if observed_distances and observed_covered < target_modes:
            failure_stage = "sparse_observation_miss"
        elif len(matches) < target_modes:
            failure_stage = "prediction_mode_miss"
        elif frequency_errors and float(np.median(frequency_errors)) > float(frequency_threshold_hz):
            failure_stage = "frequency_estimation_failure"
        elif width_errors and float(np.median(width_errors)) > float(width_relative_threshold):
            failure_stage = "width_estimation_failure"
        elif false_predicted:
            failure_stage = "false_mode_guardrail"
        else:
            failure_stage = "pass"

        diagnostics.append({
            "curve_id": record.get("curve_id"),
            "point": record.get("point"),
            "family": record.get("family"),
            "sampling_strategy": record.get("sampling_strategy"),
            "budget": record.get("budget"),
            "split_seed": record.get("split_seed"),
            "init_seed": record.get("init_seed"),
            "truth_mode_count": int(len(truth_modes)),
            "predicted_mode_count": int(len(predicted_modes)),
            "required_mode_count": int(target_modes),
            "observed_covered_mode_count": int(observed_covered) if observed_distances else None,
            "observed_covered_mode_fraction": (
                float(observed_covered / target_modes) if observed_distances else None
            ),
            "median_nearest_observed_distance_hz": _median(observed_distances),
            "max_nearest_observed_distance_hz": (
                float(np.max(observed_distances)) if observed_distances else None
            ),
            "predicted_matched_mode_count": int(len(matches)),
            "predicted_missed_mode_count": int(len(missed_truth)),
            "false_predicted_mode_count": int(len(false_predicted)),
            "matched_mode_fraction": float(len(matches) / target_modes),
            "median_matched_frequency_error_hz": _median(frequency_errors),
            "median_matched_width_relative_error": _median(width_errors),
            "frequency_threshold_hz": float(frequency_threshold_hz),
            "width_relative_threshold": float(width_relative_threshold),
            "missed_truth_frequencies_hz": [
                float(mode["frequency_hz"]) for mode in missed_truth
            ],
            "false_predicted_frequencies_hz": [
                float(mode["frequency_hz"]) for mode in false_predicted
            ],
            "truth_frequencies_hz": [
                float(mode["frequency_hz"]) for mode in truth_modes
            ],
            "predicted_frequencies_hz": [
                float(mode["frequency_hz"]) for mode in predicted_modes
            ],
            "modal_matching_tolerance_hz": float(tolerance_hz),
            "failure_stage": failure_stage,
        })
    return diagnostics


def summarize_aluminium_modal_recovery_diagnostics(rows: list[dict]) -> list[dict]:
    """Summarize modal recovery diagnostics by family, sampling strategy, and budget."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row.get("family"),
            row.get("sampling_strategy"),
            row.get("budget"),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        stage_counts: dict[str, int] = {}
        for row in items:
            stage = str(row.get("failure_stage"))
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        dominant_stage = sorted(
            stage_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        family, strategy, budget = key
        summaries.append({
            "family": family,
            "sampling_strategy": strategy,
            "budget": budget,
            "record_count": len(items),
            "dominant_failure_stage": dominant_stage,
            "stage_counts": stage_counts,
            "median_observed_covered_mode_fraction": _median([
                row.get("observed_covered_mode_fraction") for row in items
            ]),
            "median_matched_mode_fraction": _median([
                row.get("matched_mode_fraction") for row in items
            ]),
            "median_predicted_missed_mode_count": _median([
                row.get("predicted_missed_mode_count") for row in items
            ]),
            "median_false_predicted_mode_count": _median([
                row.get("false_predicted_mode_count") for row in items
            ]),
            "median_nearest_observed_distance_hz": _median([
                row.get("median_nearest_observed_distance_hz") for row in items
            ]),
            "median_matched_frequency_error_hz": _median([
                row.get("median_matched_frequency_error_hz") for row in items
            ]),
            "median_matched_width_relative_error": _median([
                row.get("median_matched_width_relative_error") for row in items
            ]),
        })
    return summaries


def _match_frequency_candidates(
    truth_frequencies: list[float],
    candidate_frequencies: list[float],
    tolerance_hz: float,
) -> tuple[list[tuple[float, float]], list[float], list[float]]:
    pairs = sorted(
        (
            abs(float(truth) - float(candidate)),
            truth_index,
            candidate_index,
        )
        for truth_index, truth in enumerate(truth_frequencies)
        for candidate_index, candidate in enumerate(candidate_frequencies)
        if abs(float(truth) - float(candidate)) <= float(tolerance_hz)
    )
    used_truth = set()
    used_candidate = set()
    matches = []
    for _, truth_index, candidate_index in pairs:
        if truth_index in used_truth or candidate_index in used_candidate:
            continue
        used_truth.add(truth_index)
        used_candidate.add(candidate_index)
        matches.append((
            float(truth_frequencies[truth_index]),
            float(candidate_frequencies[candidate_index]),
        ))
    missed_truth = [
        float(frequency)
        for index, frequency in enumerate(truth_frequencies)
        if index not in used_truth
    ]
    false_candidates = [
        float(frequency)
        for index, frequency in enumerate(candidate_frequencies)
        if index not in used_candidate
    ]
    return matches, missed_truth, false_candidates


def _truth_aligned_half_power_counts(
    *,
    truth_frequencies_hz: list[float],
    candidate_frequencies_hz: list[float],
    half_power_rows: list[dict],
    tolerance_hz: float,
) -> dict:
    """Count half-power success only for candidates aligned to truth modes."""
    pairs = sorted(
        (
            abs(float(truth) - float(candidate)),
            truth_index,
            candidate_index,
        )
        for truth_index, truth in enumerate(truth_frequencies_hz)
        for candidate_index, candidate in enumerate(candidate_frequencies_hz)
        if abs(float(truth) - float(candidate)) <= float(tolerance_hz)
    )
    used_truth = set()
    used_candidate = set()
    covered_count = 0
    predicted_ok_count = 0
    passed_count = 0
    for _, truth_index, candidate_index in pairs:
        if truth_index in used_truth or candidate_index in used_candidate:
            continue
        used_truth.add(truth_index)
        used_candidate.add(candidate_index)
        covered_count += 1
        row = half_power_rows[candidate_index] if candidate_index < len(half_power_rows) else {}
        if row.get("predicted_status") == "ok":
            predicted_ok_count += 1
        if bool(row.get("passed")):
            passed_count += 1
    target_count = max(1, len(truth_frequencies_hz))
    return {
        "covered_mode_count": int(covered_count),
        "covered_mode_fraction": float(covered_count / target_count),
        "predicted_ok_mode_count": int(predicted_ok_count),
        "passed_mode_count": int(passed_count),
        "pass_fraction": float(passed_count / target_count),
    }


def aluminium_global_modal_candidate_audit(
    records: list[dict],
    truth_catalogue_rows: list[dict],
    *,
    required_modes: int = 5,
    tolerance_hz: float = 5.0,
) -> list[dict]:
    """Audit whether global modal prior initial and fitted centers cover truth modes."""
    truth_by_curve: dict[str, list[dict]] = {}
    for row in truth_catalogue_rows:
        truth_by_curve.setdefault(str(row.get("curve_id")), []).append(row)
    for rows in truth_by_curve.values():
        rows.sort(key=lambda row: int(row.get("mode_rank", 10**9)))

    audit_rows = []
    for record in records:
        if record.get("status") != "ok":
            continue
        if record.get("family") not in {
            "Global complex modal prior",
            "Low-order global modal prior",
            "Catalogue-aware global modal prior",
        }:
            continue
        diagnostics = record.get("model_metadata", {}).get("fit_diagnostics", {})
        if not diagnostics:
            continue
        curve_id = str(record.get("curve_id"))
        truth_rows = truth_by_curve.get(curve_id, [])[: int(required_modes)]
        if not truth_rows:
            continue
        truth_frequencies = [float(row["frequency_hz"]) for row in truth_rows]
        initial_candidates = [
            float(value) for value in diagnostics.get("initial_candidate_frequencies_hz", [])
            if value is not None and math.isfinite(float(value))
        ]
        fitted_candidates = [
            float(value) for value in diagnostics.get("fitted_candidate_frequencies_hz", [])
            if value is not None and math.isfinite(float(value))
        ]
        initial_matches, initial_missed, initial_false = _match_frequency_candidates(
            truth_frequencies,
            initial_candidates,
            tolerance_hz,
        )
        fitted_matches, fitted_missed, fitted_false = _match_frequency_candidates(
            truth_frequencies,
            fitted_candidates,
            tolerance_hz,
        )
        target_modes = max(1, min(int(required_modes), len(truth_frequencies)))
        initial_stage = (
            "initial_candidates_cover_truth"
            if len(initial_matches) >= target_modes else
            "initial_candidate_miss"
        )
        if len(fitted_matches) >= target_modes:
            fitted_stage = "fitted_candidates_cover_truth"
        elif len(initial_matches) >= target_modes:
            fitted_stage = "fitted_center_drift"
        else:
            fitted_stage = "candidate_pipeline_miss"
        audit_rows.append({
            "curve_id": record.get("curve_id"),
            "point": record.get("point"),
            "family": record.get("family"),
            "sampling_strategy": record.get("sampling_strategy"),
            "budget": record.get("budget"),
            "split_seed": record.get("split_seed"),
            "init_seed": record.get("init_seed"),
            "fit_status": diagnostics.get("status"),
            "required_mode_count": int(target_modes),
            "truth_frequencies_hz": truth_frequencies,
            "initial_candidate_frequencies_hz": initial_candidates,
            "fitted_candidate_frequencies_hz": fitted_candidates,
            "initial_candidate_covered_mode_count": int(len(initial_matches)),
            "fitted_candidate_covered_mode_count": int(len(fitted_matches)),
            "initial_candidate_covered_mode_fraction": float(len(initial_matches) / target_modes),
            "fitted_candidate_covered_mode_fraction": float(len(fitted_matches) / target_modes),
            "initial_missed_truth_frequencies_hz": initial_missed,
            "fitted_missed_truth_frequencies_hz": fitted_missed,
            "initial_false_candidate_frequencies_hz": initial_false,
            "fitted_false_candidate_frequencies_hz": fitted_false,
            "initial_candidate_failure_stage": initial_stage,
            "fitted_candidate_failure_stage": fitted_stage,
            "candidate_tolerance_hz": float(tolerance_hz),
        })
    return audit_rows


def aluminium_observed_candidate_diagnostics(
    records: list[dict],
    curves: list[MeasuredCurve],
    *,
    required_modes: int = 5,
    band_half_width_hz: float = 8.0,
) -> list[dict]:
    """Audit observed-only candidate discovery against truth modes and half-power status."""
    curves_by_id = {str(curve.curve_id): curve for curve in curves}
    rows = []
    for record in records:
        if record.get("status") != "ok":
            continue
        strategy = str(record.get("sampling_strategy"))
        if strategy not in {
            "observed_peak_half_power_online",
            "observed_peak_stratified_half_power_online",
            "observed_stability_ranked_half_power_online",
            "observed_stratified_budget_half_power_online",
        }:
            continue
        curve = curves_by_id.get(str(record.get("curve_id")))
        observed_indices = record.get("observed_indices")
        if curve is None or observed_indices is None:
            continue
        observed = np.asarray(observed_indices, dtype=np.int64)
        observed = np.sort(np.unique(observed[(observed >= 0) & (observed < len(curve.frequency))]))
        target_modes = max(1, int(required_modes))
        truth_events = extract_modal_events(
            curve.frequency,
            curve.response,
            max_modes=target_modes,
            min_frequency_hz=5.0,
        )
        truth_frequencies = [
            float(mode["frequency_hz"])
            for mode in truth_events.get("modes", [])[:target_modes]
        ]
        if not truth_frequencies:
            continue
        target_modes = min(target_modes, len(truth_frequencies))
        if strategy == "observed_peak_stratified_half_power_online":
            candidate_source = "observed_peak_stratified"
            candidate_frequencies = _observed_peak_stratified_centres_from_observed(
                curve.frequency,
                curve.response,
                observed,
                band_half_width_hz=band_half_width_hz,
                max_modes=target_modes,
            )
        elif strategy == "observed_stability_ranked_half_power_online":
            candidate_source = "observed_stability_ranked"
            candidate_frequencies = _observed_stability_ranked_centres_from_observed(
                curve.frequency,
                curve.response,
                observed,
                band_half_width_hz=band_half_width_hz,
                max_modes=target_modes,
            )
        elif strategy == "observed_stratified_budget_half_power_online":
            candidate_source = "observed_stratified_budget"
            candidate_frequencies = _observed_stratified_budget_centres_from_observed(
                curve.frequency,
                curve.response,
                observed,
                band_half_width_hz=band_half_width_hz,
                max_modes=target_modes,
            )
        else:
            candidate_source = "observed_peak_prominence"
            candidate_frequencies = _observed_peak_centres_from_observed(
                curve.frequency,
                curve.response,
                observed,
                band_half_width_hz=band_half_width_hz,
                max_modes=target_modes,
            )
        matches, missed_truth, false_candidates = _match_frequency_candidates(
            truth_frequencies,
            candidate_frequencies,
            float(band_half_width_hz),
        )
        sorted_candidates = sorted(float(value) for value in candidate_frequencies)
        candidate_separations = [
            float(right - left) for left, right in zip(sorted_candidates[:-1], sorted_candidates[1:])
        ]
        conflict_count = int(sum(1 for value in candidate_separations if value < float(band_half_width_hz)))
        half_power = (record.get("metrics") or {}).get("half_power") or {}
        half_power_rows = list(half_power.get("mode_rows") or [])
        predicted_ok_count = int(half_power.get("predicted_ok_mode_count") or 0)
        passed_count = int(half_power.get("passed_mode_count") or 0)
        truth_aligned_counts = _truth_aligned_half_power_counts(
            truth_frequencies_hz=truth_frequencies,
            candidate_frequencies_hz=[float(value) for value in candidate_frequencies],
            half_power_rows=half_power_rows,
            tolerance_hz=float(band_half_width_hz),
        )
        crossing_missing_count = int(sum(
            1 for row in half_power_rows
            if row.get("predicted_status") != "ok"
        ))
        if len(matches) < target_modes:
            failure_stage = "candidate_miss"
        elif conflict_count > 0:
            failure_stage = "candidate_conflict"
        elif predicted_ok_count < target_modes:
            failure_stage = "crossing_missing"
        elif passed_count < target_modes:
            failure_stage = "parameter_failure"
        else:
            failure_stage = "pass"
        rows.append({
            "curve_id": record.get("curve_id"),
            "point": record.get("point"),
            "family": record.get("family"),
            "sampling_strategy": strategy,
            "budget": record.get("budget"),
            "split_seed": record.get("split_seed"),
            "init_seed": record.get("init_seed"),
            "candidate_source": candidate_source,
            "required_mode_count": int(target_modes),
            "truth_frequencies_hz": truth_frequencies,
            "candidate_frequencies_hz": [float(value) for value in candidate_frequencies],
            "candidate_count": int(len(candidate_frequencies)),
            "candidate_covered_mode_count": int(len(matches)),
            "candidate_covered_mode_fraction": float(len(matches) / target_modes),
            "missed_truth_frequencies_hz": missed_truth,
            "false_candidate_frequencies_hz": false_candidates,
            "false_candidate_count": int(len(false_candidates)),
            "candidate_conflict_count": conflict_count,
            "min_candidate_separation_hz": (
                float(np.min(candidate_separations)) if candidate_separations else None
            ),
            "median_candidate_separation_hz": _median(candidate_separations),
            "half_power_passed_mode_count": passed_count,
            "half_power_predicted_ok_mode_count": predicted_ok_count,
            "half_power_crossing_missing_count": crossing_missing_count,
            "truth_aligned_half_power_covered_mode_count": int(truth_aligned_counts["covered_mode_count"]),
            "truth_aligned_half_power_covered_mode_fraction": float(truth_aligned_counts["covered_mode_fraction"]),
            "truth_aligned_half_power_predicted_ok_mode_count": int(truth_aligned_counts["predicted_ok_mode_count"]),
            "truth_aligned_half_power_passed_mode_count": int(truth_aligned_counts["passed_mode_count"]),
            "truth_aligned_half_power_pass_fraction": float(truth_aligned_counts["pass_fraction"]),
            "candidate_failure_stage": failure_stage,
            "band_half_width_hz": float(band_half_width_hz),
        })
    return rows


def summarize_aluminium_observed_candidate_diagnostics(rows: list[dict]) -> list[dict]:
    """Summarize observed-only candidate diagnostics by family, strategy, and budget."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row.get("family"),
            row.get("sampling_strategy"),
            row.get("budget"),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        stage_counts: dict[str, int] = {}
        for row in items:
            stage = str(row.get("candidate_failure_stage"))
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        dominant_stage = sorted(
            stage_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        family, strategy, budget = key
        summaries.append({
            "family": family,
            "sampling_strategy": strategy,
            "budget": budget,
            "record_count": len(items),
            "dominant_candidate_failure_stage": dominant_stage,
            "candidate_failure_stage_counts": stage_counts,
            "median_candidate_covered_mode_fraction": _median([
                row.get("candidate_covered_mode_fraction") for row in items
            ]),
            "median_candidate_covered_mode_count": _median([
                row.get("candidate_covered_mode_count") for row in items
            ]),
            "median_false_candidate_count": _median([
                row.get("false_candidate_count") for row in items
            ]),
            "median_candidate_conflict_count": _median([
                row.get("candidate_conflict_count") for row in items
            ]),
            "median_half_power_passed_mode_count": _median([
                row.get("half_power_passed_mode_count") for row in items
            ]),
            "median_half_power_predicted_ok_mode_count": _median([
                row.get("half_power_predicted_ok_mode_count") for row in items
            ]),
            "median_half_power_crossing_missing_count": _median([
                row.get("half_power_crossing_missing_count") for row in items
            ]),
            "median_truth_aligned_half_power_covered_mode_count": _median([
                row.get("truth_aligned_half_power_covered_mode_count") for row in items
            ]),
            "median_truth_aligned_half_power_predicted_ok_mode_count": _median([
                row.get("truth_aligned_half_power_predicted_ok_mode_count") for row in items
            ]),
            "median_truth_aligned_half_power_passed_mode_count": _median([
                row.get("truth_aligned_half_power_passed_mode_count") for row in items
            ]),
            "median_truth_aligned_half_power_pass_fraction": _median([
                row.get("truth_aligned_half_power_pass_fraction") for row in items
            ]),
        })
    return summaries


def aluminium_modal_failure_rows(
    records: list[dict],
    *,
    required_modes: int = 5,
    frequency_threshold_hz: float = 1.0,
    width_relative_threshold: float = 0.10,
) -> list[dict]:
    """Classify why each Aluminium C record misses the modal-analysis target."""
    rows = []
    required_modes = int(required_modes)
    for record in records:
        if record.get("status") != "ok":
            continue
        metrics = record.get("metrics", {})
        if not metrics or metrics.get("status", "ok") != "ok":
            continue
        truth_count = int(metrics.get("truth_mode_count") or 0)
        matched_count = int(metrics.get("matched_mode_count") or 0)
        target_modes = max(1, min(required_modes, truth_count or required_modes))
        frequency_error = _finite_metric(metrics.get("modal_frequency_mae_hz"))
        width_error = _finite_metric(metrics.get("modal_width_relative_mae"))
        if matched_count < target_modes:
            reason = "coverage_failure"
        elif frequency_error is None or frequency_error > float(frequency_threshold_hz):
            reason = "frequency_failure"
        elif width_error is None or width_error > float(width_relative_threshold):
            reason = "width_failure"
        else:
            reason = "pass"
        rows.append({
            "curve_id": record.get("curve_id"),
            "point": record.get("point"),
            "family": record.get("family"),
            "sampling_strategy": record.get("sampling_strategy"),
            "budget": record.get("budget"),
            "split_seed": record.get("split_seed"),
            "init_seed": record.get("init_seed"),
            "truth_mode_count": truth_count,
            "matched_mode_count": matched_count,
            "required_mode_count": target_modes,
            "matched_mode_fraction": float(matched_count / target_modes),
            "modal_frequency_mae_hz": frequency_error,
            "modal_width_relative_mae": width_error,
            "modal_peak_amplitude_relative_mae": _finite_metric(
                metrics.get("modal_peak_amplitude_relative_mae")
            ),
            "complex_nrmse": _finite_metric(metrics.get("complex_nrmse")),
            "peak_window_complex_nrmse": _finite_metric(metrics.get("peak_window_complex_nrmse")),
            "frequency_threshold_hz": float(frequency_threshold_hz),
            "width_relative_threshold": float(width_relative_threshold),
            "failure_reason": reason,
        })
    return rows


def summarize_aluminium_modal_failures(rows: list[dict]) -> list[dict]:
    """Summarize Aluminium C modal failure classifications by family/strategy/budget."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row.get("family"),
            row.get("sampling_strategy"),
            row.get("budget"),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        reason_counts: dict[str, int] = {}
        for row in items:
            reason = str(row.get("failure_reason"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        dominant_reason = sorted(
            reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        family, strategy, budget = key
        summaries.append({
            "family": family,
            "sampling_strategy": strategy,
            "budget": budget,
            "record_count": len(items),
            "dominant_failure_reason": dominant_reason,
            "failure_reason_counts": reason_counts,
            "pass_count": int(reason_counts.get("pass", 0)),
            "coverage_failure_count": int(reason_counts.get("coverage_failure", 0)),
            "frequency_failure_count": int(reason_counts.get("frequency_failure", 0)),
            "width_failure_count": int(reason_counts.get("width_failure", 0)),
            "median_matched_mode_fraction": _median([
                row.get("matched_mode_fraction") for row in items
            ]),
            "median_matched_mode_count": _median([
                row.get("matched_mode_count") for row in items
            ]),
            "median_modal_frequency_mae_hz": _median([
                row.get("modal_frequency_mae_hz") for row in items
            ]),
            "median_modal_width_relative_mae": _median([
                row.get("modal_width_relative_mae") for row in items
            ]),
            "median_peak_window_complex_nrmse": _median([
                row.get("peak_window_complex_nrmse") for row in items
            ]),
        })
    return summaries


def _finite_or_none(value) -> float | None:
    return _finite_metric(value)


def _best_half_power_interpolation_row(rows: list[dict]) -> dict | None:
    candidates = []
    for row in rows:
        family = str(row.get("family"))
        if "interpolation" not in family.lower():
            continue
        passed = _finite_or_none(
            row.get("median_hp_passed_mode_count", row.get("median_half_power_passed_mode_count"))
        )
        budget = row.get("budget")
        if passed is None or budget is None:
            continue
        candidates.append((float(passed), -int(budget), row))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-item[0], item[1], str(item[2].get("family"))))[0][2]


def audit_aluminium_active_measurement_value(
    interpolation_rollup: dict,
    same_schedule_summary: dict,
    candidate_diagnostics_summary: dict,
    *,
    required_modes: int = 5,
) -> dict:
    """Separate Aluminium C measurement-efficiency signal from CFNN advantage claims."""
    interpolation_rows = list(interpolation_rollup.get("summary_rows", []))
    same_schedule_rows = list(same_schedule_summary.get("summary_rows", []))
    candidate_rows = list(candidate_diagnostics_summary.get("summary_rows", []))

    best_interpolation = _best_half_power_interpolation_row(interpolation_rows)
    interpolation_passed = (
        _finite_or_none(best_interpolation.get("median_hp_passed_mode_count"))
        if best_interpolation else None
    )
    interpolation_all_pass = (
        int(best_interpolation.get("all_five_pass_count") or 0)
        if best_interpolation else 0
    )
    interpolation_budget = int(best_interpolation.get("budget")) if best_interpolation else None
    interpolation_supports = bool(
        best_interpolation
        and (
            (interpolation_passed is not None and interpolation_passed >= int(required_modes))
            or interpolation_all_pass > 0
        )
    )

    cfnn_rows = [
        row for row in same_schedule_rows
        if str(row.get("family")) == "CFNN"
    ]
    cfnn_best = None
    if cfnn_rows:
        cfnn_best = max(
            cfnn_rows,
            key=lambda row: _finite_or_none(row.get("median_half_power_passed_mode_count")) or 0.0,
        )
    cfnn_passed = (
        _finite_or_none(cfnn_best.get("median_half_power_passed_mode_count"))
        if cfnn_best else None
    )
    cfnn_matched = (
        _finite_or_none(cfnn_best.get("median_matched_mode_count"))
        if cfnn_best else None
    )
    cfnn_blocks = bool(cfnn_best and (cfnn_passed or 0.0) < int(required_modes))

    candidate_best = None
    if candidate_rows:
        candidate_best = max(
            candidate_rows,
            key=lambda row: _finite_or_none(row.get("median_truth_aligned_half_power_passed_mode_count")) or 0.0,
        )
    candidate_stage = str(candidate_best.get("dominant_candidate_failure_stage")) if candidate_best else None
    candidate_coverage = (
        _finite_or_none(candidate_best.get("median_candidate_covered_mode_fraction"))
        if candidate_best else None
    )
    truth_aligned_passed = (
        _finite_or_none(candidate_best.get("median_truth_aligned_half_power_passed_mode_count"))
        if candidate_best else None
    )

    decision_rows = [
        {
            "decision_id": "observed_interpolation_measurement_signal",
            "evidence_scope": "observed_only_half_power_interpolation",
            "family": best_interpolation.get("family") if best_interpolation else None,
            "minimum_budget": interpolation_budget,
            "median_passed_modes": interpolation_passed,
            "all_five_pass_count": interpolation_all_pass,
            "median_peak_window_complex_nrmse": (
                _finite_or_none(best_interpolation.get("median_peak_window_complex_nrmse"))
                if best_interpolation else None
            ),
            "verdict": (
                "supports_measurement_efficiency_signal"
                if interpolation_supports else
                "no_measurement_efficiency_signal"
            ),
        },
        {
            "decision_id": "same_schedule_cfnn_boundary",
            "evidence_scope": "same_schedule_neural_model_check",
            "family": "CFNN",
            "minimum_budget": int(cfnn_best.get("budget")) if cfnn_best and cfnn_best.get("budget") is not None else None,
            "median_passed_modes": cfnn_passed,
            "median_matched_modes": cfnn_matched,
            "verdict": (
                "blocks_cfnn_advantage_claim"
                if cfnn_blocks else
                "cfnn_candidate_signal"
            ),
        },
        {
            "decision_id": "candidate_discovery_boundary",
            "evidence_scope": "observed_only_candidate_diagnostics",
            "family": candidate_best.get("family") if candidate_best else None,
            "budget": int(candidate_best.get("budget")) if candidate_best and candidate_best.get("budget") is not None else None,
            "dominant_failure_stage": candidate_stage,
            "median_candidate_covered_mode_fraction": candidate_coverage,
            "median_truth_aligned_half_power_passed_mode_count": truth_aligned_passed,
            "verdict": (
                "candidate_discovery_limits_full_modal_recovery"
                if candidate_stage and candidate_stage != "pass" else
                "candidate_discovery_passes"
            ),
        },
    ]
    if interpolation_supports and cfnn_blocks:
        claim_level = "measurement_strategy_signal_not_cfnn_advantage"
    elif interpolation_supports:
        claim_level = "measurement_strategy_signal"
    else:
        claim_level = "active_modal_boundary"
    return {
        "analysis_type": "aluminium_c_active_measurement_value_audit",
        "required_modes": int(required_modes),
        "claim_level": claim_level,
        "decision_rows": decision_rows,
    }
