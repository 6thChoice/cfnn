"""AI4Science Microwave Fano reference-parameter extraction.

This module fits reference physical labels from complete measured curves. It is
intentionally independent of model predictions, so downstream parameter-recovery
experiments can compare neural reconstructions against an external fit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from downstream_protocol import MeasuredCurve, load_microwave_fano


@dataclass(frozen=True)
class FanoReference:
    curve_id: str
    reference_model: str
    status: str
    failure_reason: str | None
    f0_hz: float | None
    gamma_hz: float | None
    linewidth_hz: float | None
    quality_factor: float | None
    q: float | None
    background_real: float | None
    background_imag: float | None
    background_slope_real_per_hz: float | None
    background_slope_imag_per_hz: float | None
    amplitude_real: float | None
    amplitude_imag: float | None
    magnitude_peak_hz: float | None
    magnitude_valley_hz: float | None
    phase_transition_hz: float | None
    relative_complex_rmse: float | None
    relative_magnitude_rmse: float | None
    points: int
    frequency_min_hz: float
    frequency_max_hz: float
    conditions: dict
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _as_complex_response(response: np.ndarray) -> np.ndarray:
    array = np.asarray(response)
    if array.ndim == 1 and np.iscomplexobj(array):
        values = np.asarray(array, dtype=np.complex128)
    elif array.ndim == 2 and array.shape[1] == 2:
        values = np.asarray(array[:, 0], dtype=np.float64) + 1j * np.asarray(
            array[:, 1], dtype=np.float64
        )
    else:
        raise ValueError("response must be a complex vector or a two-column real array")
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("response must contain finite values")
    return values


def complex_fano_response(
    frequency_hz: np.ndarray,
    *,
    f0_hz: float,
    gamma_hz: float,
    q: float,
    background: complex,
    background_slope_per_hz: complex,
    amplitude: complex,
) -> np.ndarray:
    """Evaluate a complex Fano line shape on a physical frequency axis.

    `gamma_hz` is the half-width scale in `eps = (f - f0) / gamma`; the reported
    linewidth is `2 * gamma_hz`.
    """
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if gamma_hz <= 0.0:
        raise ValueError("gamma_hz must be positive")
    eps = (frequency - float(f0_hz)) / float(gamma_hz)
    return (
        complex(background)
        + complex(background_slope_per_hz) * (frequency - float(f0_hz))
        + complex(amplitude) * (((float(q) + eps) / (eps + 1j)) - 1.0)
    )


def magnitude_fano_response(
    frequency_hz: np.ndarray,
    *,
    f0_hz: float,
    gamma_hz: float,
    q: float,
    background: float,
    background_slope_per_hz: float,
    amplitude: float,
) -> np.ndarray:
    """Evaluate a real-valued Fano magnitude line shape."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if gamma_hz <= 0.0:
        raise ValueError("gamma_hz must be positive")
    eps = (frequency - float(f0_hz)) / float(gamma_hz)
    return (
        float(background)
        + float(background_slope_per_hz) * (frequency - float(f0_hz))
        + float(amplitude) * ((float(q) + eps) ** 2 / (1.0 + eps ** 2))
    )


def _normalized_magnitude(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    f0, log_gamma, q, b0, b1, amplitude = params
    gamma = math.exp(float(log_gamma))
    eps = (x - f0) / gamma
    return b0 + b1 * (x - f0) + amplitude * ((q + eps) ** 2 / (1.0 + eps ** 2))


def _response_structure_is_sufficient(response: np.ndarray) -> bool:
    magnitude = np.abs(response)
    level = max(float(np.median(magnitude)), float(np.sqrt(np.mean(magnitude ** 2))), 1e-12)
    return float(np.ptp(magnitude)) / level >= 1e-3


def _candidate_indices(response: np.ndarray) -> list[int]:
    magnitude = np.abs(response)
    phase = np.unwrap(np.angle(response))
    gradient_score = np.abs(np.gradient(magnitude)) + 0.1 * np.abs(np.gradient(phase))
    indices = {
        int(np.argmax(magnitude)),
        int(np.argmin(magnitude)),
        int(np.argmax(gradient_score)),
        len(response) // 2,
    }
    return sorted(indices)


def _initial_magnitude_parameters(x: np.ndarray, magnitude: np.ndarray) -> Iterable[np.ndarray]:
    b0 = float(np.percentile(magnitude, 10.0))
    amplitude_scale = max(float(np.ptp(magnitude)), float(np.std(magnitude)), 1e-8)
    for index in _candidate_indices(magnitude.astype(np.complex128)):
        f0 = float(x[index])
        for gamma in (0.002, 0.01, 0.04, 0.16):
            for q in (-5.0, -1.0, -0.2, 0.2, 1.0, 5.0):
                yield np.asarray([
                    f0,
                    math.log(gamma),
                    q,
                    b0,
                    0.0,
                    amplitude_scale,
                ], dtype=np.float64)


def _fit_failure(
    curve_id: str,
    reason: str,
    frequency: np.ndarray,
    *,
    conditions: dict | None,
    metadata: dict | None,
) -> FanoReference:
    return FanoReference(
        curve_id=curve_id,
        reference_model="magnitude_fano",
        status="unidentifiable",
        failure_reason=reason,
        f0_hz=None,
        gamma_hz=None,
        linewidth_hz=None,
        quality_factor=None,
        q=None,
        background_real=None,
        background_imag=None,
        background_slope_real_per_hz=None,
        background_slope_imag_per_hz=None,
        amplitude_real=None,
        amplitude_imag=None,
        magnitude_peak_hz=None,
        magnitude_valley_hz=None,
        phase_transition_hz=None,
        relative_complex_rmse=None,
        relative_magnitude_rmse=None,
        points=int(len(frequency)),
        frequency_min_hz=float(frequency[0]),
        frequency_max_hz=float(frequency[-1]),
        conditions=dict(conditions or {}),
        metadata=dict(metadata or {}),
    )


def fit_complex_fano_curve(
    frequency_hz: np.ndarray,
    response: np.ndarray,
    *,
    curve_id: str,
    residual_threshold: float = 0.15,
    conditions: dict | None = None,
    metadata: dict | None = None,
) -> FanoReference:
    """Fit a single Fano magnitude reference label to a complete measured curve.

    The measured complex phase is still used to extract a phase-transition event,
    but `f0`, `Gamma`, `Q`, and `q` come from the amplitude-domain Fano fit. This
    avoids treating arbitrary complex calibration phase as part of the Fano q
    definition.
    """
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    y = _as_complex_response(response)
    if len(frequency) != len(y):
        raise ValueError("frequency and response lengths differ")
    if len(frequency) < 16 or not np.all(np.isfinite(frequency)):
        raise ValueError("frequency must contain at least 16 finite points")
    if np.any(np.diff(frequency) <= 0.0):
        raise ValueError("frequency must be strictly increasing")
    if not _response_structure_is_sufficient(y):
        return _fit_failure(
            curve_id, "insufficient_response_structure", frequency,
            conditions=conditions, metadata=metadata,
        )

    center_hz = float(np.median(frequency))
    span_hz = float(np.ptp(frequency))
    x = (frequency - center_hz) / span_hz
    magnitude = np.abs(y)
    residual_scale = max(float(np.ptp(magnitude)), 1e-12)
    min_spacing = max(float(np.min(np.diff(x))), 1e-9)
    lower = np.asarray([
        float(x[0]) - 0.5,
        math.log(0.2 * min_spacing),
        -50.0,
        -np.inf,
        -np.inf,
        -np.inf,
    ])
    upper = np.asarray([
        float(x[-1]) + 0.5,
        math.log(2.0),
        50.0,
        np.inf,
        np.inf,
        np.inf,
    ])

    def residual(params: np.ndarray) -> np.ndarray:
        return (_normalized_magnitude(x, params) - magnitude) / residual_scale

    best = None
    for initial in _initial_magnitude_parameters(x, magnitude):
        candidate = np.minimum(np.maximum(initial, lower), upper)
        result = least_squares(
            residual,
            candidate,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.1,
            max_nfev=3000,
        )
        if best is None or result.cost < best.cost:
            best = result
    if best is None or not best.success:
        return _fit_failure(
            curve_id, "least_squares_failed", frequency,
            conditions=conditions, metadata=metadata,
        )

    params = np.asarray(best.x, dtype=np.float64)
    fitted_magnitude = _normalized_magnitude(x, params)
    absolute_rmse = float(np.sqrt(np.mean((fitted_magnitude - magnitude) ** 2)))
    denominator = max(float(np.sqrt(np.mean(magnitude ** 2))), 1e-12)
    relative_magnitude_rmse = absolute_rmse / denominator
    status = "ok" if relative_magnitude_rmse <= float(residual_threshold) else "high_residual"
    failure_reason = None if status == "ok" else "relative_complex_rmse_exceeds_threshold"

    f0_hz = float(center_hz + params[0] * span_hz)
    gamma_hz = float(math.exp(float(params[1])) * span_hz)
    linewidth_hz = float(2.0 * gamma_hz)
    quality_factor = float(f0_hz / linewidth_hz) if linewidth_hz > 0.0 else None
    background = float(params[3])
    slope_per_hz = float(params[4]) / span_hz
    amplitude = float(params[5])

    dense_frequency = np.linspace(float(frequency[0]), float(frequency[-1]), max(4096, 4 * len(frequency)))
    dense_magnitude = magnitude_fano_response(
        dense_frequency,
        f0_hz=f0_hz,
        gamma_hz=gamma_hz,
        q=float(params[2]),
        background=background,
        background_slope_per_hz=slope_per_hz,
        amplitude=amplitude,
    )
    phase = np.unwrap(np.angle(y))
    phase_slope = np.abs(np.gradient(phase, frequency))
    edge = max(1, len(dense_frequency) // 100)
    valid_slice = slice(edge, -edge)
    local_offset = edge
    peak_index = int(np.argmax(dense_magnitude[valid_slice]) + local_offset)
    valley_index = int(np.argmin(dense_magnitude[valid_slice]) + local_offset)
    phase_edge = max(1, len(frequency) // 100)
    phase_index = int(np.argmax(phase_slope[phase_edge:-phase_edge]) + phase_edge)

    return FanoReference(
        curve_id=curve_id,
        reference_model="magnitude_fano",
        status=status,
        failure_reason=failure_reason,
        f0_hz=f0_hz,
        gamma_hz=gamma_hz,
        linewidth_hz=linewidth_hz,
        quality_factor=quality_factor,
        q=float(params[2]),
        background_real=float(background),
        background_imag=None,
        background_slope_real_per_hz=float(slope_per_hz),
        background_slope_imag_per_hz=None,
        amplitude_real=float(amplitude),
        amplitude_imag=None,
        magnitude_peak_hz=float(dense_frequency[peak_index]),
        magnitude_valley_hz=float(dense_frequency[valley_index]),
        phase_transition_hz=float(frequency[phase_index]),
        relative_complex_rmse=None,
        relative_magnitude_rmse=float(relative_magnitude_rmse),
        points=int(len(frequency)),
        frequency_min_hz=float(frequency[0]),
        frequency_max_hz=float(frequency[-1]),
        conditions=dict(conditions or {}),
        metadata=dict(metadata or {}),
    )


def selected_microwave_fano_curve_ids(role_manifest_path: Path) -> set[str]:
    manifest = json.loads(Path(role_manifest_path).read_text())
    return set(manifest["selected_curve_point_counts"])


def build_microwave_fano_reference_catalog(
    raw_root: Path,
    role_manifest_path: Path,
    *,
    selected_only: bool = True,
    residual_threshold: float = 0.15,
) -> list[FanoReference]:
    """Build A1 reference labels for frozen or all released Microwave Fano curves."""
    raw_root = Path(raw_root)
    curves: list[MeasuredCurve] = load_microwave_fano(raw_root)
    if selected_only:
        selected = selected_microwave_fano_curve_ids(role_manifest_path)
        curves = [curve for curve in curves if curve.curve_id in selected]
    references = [
        fit_complex_fano_curve(
            curve.frequency,
            curve.response,
            curve_id=curve.curve_id,
            residual_threshold=residual_threshold,
            conditions=curve.conditions,
            metadata=curve.metadata,
        )
        for curve in sorted(curves, key=lambda item: item.curve_id)
    ]
    return references


def references_to_csv_rows(references: Iterable[FanoReference]) -> list[dict]:
    rows = []
    for reference in references:
        row = reference.to_dict()
        row["conditions"] = json.dumps(row["conditions"], sort_keys=True)
        row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
        rows.append(row)
    return rows
