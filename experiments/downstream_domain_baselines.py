"""Domain-reference fits for measured resonant responses."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class DomainFitResult:
    prediction: np.ndarray
    parameters: dict
    status: str
    wall_seconds: float
    metadata: dict


def _to_complex(response: np.ndarray) -> np.ndarray:
    values = np.asarray(response, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("response must have shape (n, 2)")
    return values[:, 0] + 1j * values[:, 1]


def _two_columns(response: np.ndarray) -> np.ndarray:
    return np.column_stack([response.real, response.imag]).astype(np.float64)


def _fano_model(frequency: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    center, width, q, amplitude_real, amplitude_imag, background_real, background_imag = parameters
    z = (frequency - center) / width
    term = (q + z) / (z + 1j) - 1.0
    return (background_real + 1j * background_imag) + (
        amplitude_real + 1j * amplitude_imag
    ) * term


def _complex_residual(parameters: np.ndarray, frequency: np.ndarray, response: np.ndarray) -> np.ndarray:
    residual = _fano_model(frequency, parameters) - response
    return np.concatenate([residual.real, residual.imag])


def fit_fano_reference(
    frequency: np.ndarray,
    response: np.ndarray,
    evaluation_frequency: np.ndarray | None = None,
) -> DomainFitResult:
    """Fit a one-resonance complex Fano model with bounded multistart optimization."""
    started = time.perf_counter()
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    target = _to_complex(response)
    if len(frequency) < 8 or len(frequency) != len(target):
        raise ValueError("Fano fitting requires at least eight aligned samples")
    if np.any(np.diff(frequency) <= 0):
        raise ValueError("frequency must be strictly increasing")
    span = float(frequency[-1] - frequency[0])
    resolution = float(np.median(np.diff(frequency)))
    edge_count = max(2, len(frequency) // 20)
    edge_values = np.concatenate([target[:edge_count], target[-edge_count:]])
    background = complex(np.median(edge_values.real), np.median(edge_values.imag))
    centered = target - background
    scale = max(float(np.max(np.abs(centered))), 1e-6)
    center_candidates = {
        float((frequency[0] + frequency[-1]) / 2.0),
        float(frequency[int(np.argmax(np.abs(centered)))]),
        float(frequency[int(np.argmin(np.abs(centered)))]),
    }
    width_candidates = (max(2.0 * resolution, span / 100.0), span / 30.0, span / 10.0)
    lower = np.asarray([
        frequency[0], resolution, -10.0,
        -100.0 * scale, -100.0 * scale,
        background.real - 100.0 * scale, background.imag - 100.0 * scale,
    ])
    upper = np.asarray([
        frequency[-1], span / 2.0, 10.0,
        100.0 * scale, 100.0 * scale,
        background.real + 100.0 * scale, background.imag + 100.0 * scale,
    ])
    best = None
    attempts = 0
    for center in sorted(center_candidates):
        for width in width_candidates:
            for q in (-2.0, -0.5, 0.5, 2.0):
                z = (frequency - center) / width
                term = (q + z) / (z + 1j) - 1.0
                design = np.column_stack([term, np.ones_like(term)])
                amplitude, initial_background = np.linalg.lstsq(design, target, rcond=None)[0]
                initial = np.asarray([
                    center, width, q, amplitude.real, amplitude.imag,
                    initial_background.real, initial_background.imag,
                ])
                initial = np.minimum(np.maximum(initial, lower + 1e-12), upper - 1e-12)
                result = least_squares(
                    _complex_residual,
                    initial,
                    bounds=(lower, upper),
                    args=(frequency, target),
                    max_nfev=3000,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12,
                )
                attempts += 1
                if best is None or result.cost < best.cost:
                    best = result
    assert best is not None
    prediction_frequency = (
        frequency if evaluation_frequency is None
        else np.asarray(evaluation_frequency, dtype=np.float64).reshape(-1)
    )
    prediction = _fano_model(prediction_frequency, best.x)
    names = (
        "center", "width", "q", "amplitude_real", "amplitude_imag",
        "background_real", "background_imag",
    )
    status = "ok" if best.success and np.all(np.isfinite(prediction)) else "numerical_failure"
    return DomainFitResult(
        prediction=_two_columns(prediction),
        parameters={name: float(value) for name, value in zip(names, best.x)},
        status=status,
        wall_seconds=time.perf_counter() - started,
        metadata={
            "method": "bounded_complex_fano_least_squares",
            "attempts": attempts,
            "optimizer_status": int(best.status),
            "optimizer_message": str(best.message),
            "cost": float(best.cost),
            "jacobian_rank": int(np.linalg.matrix_rank(best.jac)),
        },
    )


def fit_vector_reference(
    frequency: np.ndarray,
    response: np.ndarray,
    max_poles: int,
    evaluation_frequency: np.ndarray | None = None,
) -> DomainFitResult:
    """Fit a stable complex transfer reference with scikit-rf vector fitting."""
    import skrf as rf
    from skrf.vectorFitting import VectorFitting

    started = time.perf_counter()
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    target = _to_complex(response)
    if max_poles < 2:
        raise ValueError("vector fitting requires at least two poles")
    try:
        convergence_warnings = []
        network_frequency = rf.Frequency.from_f(frequency, unit="hz")
        network = rf.Network(
            frequency=network_frequency,
            s=target.reshape(-1, 1, 1),
            z0=50.0,
        )
        fitter = VectorFitting(network)
        complex_pairs = max(1, int(max_poles) // 2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitter.vector_fit(
                n_poles_real=0,
                n_poles_cmplx=complex_pairs,
                init_pole_spacing="log",
                parameter_type="s",
                fit_constant=True,
                fit_proportional=False,
                enforce_dc=False,
            )
        convergence_warnings = [str(item.message) for item in caught]
        prediction_frequency = (
            frequency if evaluation_frequency is None
            else np.asarray(evaluation_frequency, dtype=np.float64).reshape(-1)
        )
        prediction = fitter.get_model_response(0, 0, prediction_frequency)
        poles = np.asarray(fitter.poles)
        finite = np.all(np.isfinite(prediction)) and np.all(np.isfinite(poles))
        stable = bool(np.all(np.real(poles) < 0))
        return DomainFitResult(
            prediction=_two_columns(prediction),
            parameters={
                "poles_real": np.real(poles).tolist(),
                "poles_imag": np.imag(poles).tolist(),
            },
            status="ok" if finite else "numerical_failure",
            wall_seconds=time.perf_counter() - started,
            metadata={
                "method": "skrf_vector_fitting",
                "maximum_poles": int(max_poles),
                "complex_pole_pairs": complex_pairs,
                "fitted_pole_count": int(len(poles)),
                "stable_poles": stable,
                "scikit_rf_version": rf.__version__,
                "convergence_warning_count": len(convergence_warnings),
                "convergence_warnings": convergence_warnings,
            },
        )
    except Exception as error:
        return DomainFitResult(
            prediction=np.full_like(np.asarray(response, dtype=np.float64), np.nan),
            parameters={},
            status="numerical_failure",
            wall_seconds=time.perf_counter() - started,
            metadata={
                "method": "skrf_vector_fitting",
                "maximum_poles": int(max_poles),
                "error": f"{type(error).__name__}: {error}",
            },
        )
