"""EIS boundary descriptors and CFNN effect summaries for Package E."""
from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def _as_complex(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1 and np.iscomplexobj(array):
        out = np.asarray(array, dtype=np.complex128)
    elif array.ndim == 2 and array.shape[1] == 2:
        out = np.asarray(array[:, 0], dtype=np.float64) + 1j * np.asarray(array[:, 1], dtype=np.float64)
    else:
        raise ValueError("values must be a complex vector or shape (n, 2)")
    if len(out) < 3 or not np.all(np.isfinite(out)):
        raise ValueError("values must contain at least three finite points")
    return out


def _finite(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _local_maxima(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.asarray([], dtype=int)
    return np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])) + 1


def eis_spectral_descriptors(
    spectrum_id: str,
    frequency_hz: np.ndarray,
    impedance: np.ndarray,
    *,
    metadata: dict | None = None,
) -> dict:
    """Compute structure descriptors for an EIS impedance spectrum."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    z = _as_complex(impedance)
    if len(frequency) != len(z):
        raise ValueError("frequency and impedance lengths differ")
    if np.any(frequency <= 0.0) or np.any(np.diff(frequency) <= 0.0):
        raise ValueError("frequency must be positive and strictly increasing")
    metadata = dict(metadata or {})
    logf = np.log10(frequency)
    span_decades = float(np.ptp(logf))
    if span_decades <= 0.0:
        raise ValueError("frequency span must be positive")

    neg_imag = -np.imag(z)
    scale = max(float(np.ptp(neg_imag)), float(np.std(neg_imag)), np.finfo(float).eps)
    normalized = (neg_imag - float(np.median(neg_imag))) / scale
    slope = np.gradient(normalized, logf)
    curvature = np.gradient(slope, logf)
    peaks = _local_maxima(neg_imag)
    prominence_threshold = 0.05 * max(float(np.ptp(neg_imag)), np.finfo(float).eps)
    prominent = []
    for index in peaks:
        left = np.min(neg_imag[: index + 1])
        right = np.min(neg_imag[index:])
        prominence = float(neg_imag[index] - max(left, right))
        if prominence >= prominence_threshold:
            prominent.append((int(index), prominence))
    peak_logs = [float(logf[index]) for index, _ in prominent]
    spacing = None
    if len(peak_logs) >= 2:
        spacing = float(np.min(np.diff(sorted(peak_logs))))
    phase = np.unwrap(np.angle(z))
    magnitude = np.abs(z)

    return {
        "spectrum_id": str(spectrum_id),
        "point_count": int(len(frequency)),
        "frequency_min_hz": float(frequency[0]),
        "frequency_max_hz": float(frequency[-1]),
        "frequency_span_decades": span_decades,
        "frequency_points_per_decade": float(len(frequency) / span_decades),
        "relaxation_peak_count": int(len(prominent)),
        "max_relaxation_prominence_ratio": float(max((p for _, p in prominent), default=0.0) / max(float(np.max(np.abs(neg_imag))), np.finfo(float).eps)),
        "min_log_peak_spacing_decades": spacing,
        "max_normalized_slope": float(np.max(np.abs(slope))),
        "max_normalized_curvature": float(np.max(np.abs(curvature))),
        "phase_total_turns": float(np.ptp(phase) / (2.0 * math.pi)),
        "impedance_dynamic_range_ratio": float(np.ptp(magnitude) / max(float(np.sqrt(np.mean(magnitude ** 2))), np.finfo(float).eps)),
        "known_pole_count": metadata.get("known_pole_count"),
        "distributed_relaxation": bool(metadata.get("distributed_relaxation", False)),
        "task_family": metadata.get("task_family"),
        "noise": metadata.get("noise"),
        "n_points": metadata.get("n_points"),
    }


def loss_effect_row(
    *,
    rows: Iterable[dict],
    metric: str,
    cfnn_models: set[str],
    baseline_filter=None,
) -> dict:
    """Return signed CFNN effect for a lower-is-better metric."""
    row_list = [row for row in rows if _finite(row.get(metric)) is not None]
    cfnn = [row for row in row_list if row.get("model") in cfnn_models or row.get("family") in cfnn_models]
    baselines = [
        row for row in row_list
        if row not in cfnn and (baseline_filter(row) if baseline_filter is not None else True)
    ]
    if not cfnn or not baselines:
        return {"status": "missing_model_family", "metric": metric}
    best_cfnn = min(cfnn, key=lambda row: float(row[metric]))
    best_baseline = min(baselines, key=lambda row: float(row[metric]))
    cfnn_metric = float(best_cfnn[metric])
    baseline_metric = float(best_baseline[metric])
    denominator = max(abs(baseline_metric), np.finfo(float).eps)
    return {
        "status": "ok",
        "metric": metric,
        "cfnn_model": best_cfnn.get("model", best_cfnn.get("family")),
        "cfnn_metric": cfnn_metric,
        "best_baseline_model": best_baseline.get("model", best_baseline.get("family")),
        "best_baseline_metric": baseline_metric,
        "signed_cfnn_effect": float((baseline_metric - cfnn_metric) / denominator),
    }


def layered_loss_effect_rows(
    *,
    rows: Iterable[dict],
    metric: str,
    cfnn_models: set[str],
    neural_baseline_models: set[str],
    physical_prior_models: set[str],
) -> dict[str, dict]:
    """Compute all/neural-only/physical-prior CFNN effects for one cell."""
    neural_models = set(neural_baseline_models)
    physical_models = set(physical_prior_models)
    all_models = neural_models | physical_models
    return {
        "all_baselines": loss_effect_row(
            rows=rows,
            metric=metric,
            cfnn_models=cfnn_models,
            baseline_filter=lambda row: row.get("model") in all_models or row.get("family") in all_models,
        ),
        "neural_only": loss_effect_row(
            rows=rows,
            metric=metric,
            cfnn_models=cfnn_models,
            baseline_filter=lambda row: row.get("model") in neural_models or row.get("family") in neural_models,
        ),
        "physical_prior_only": loss_effect_row(
            rows=rows,
            metric=metric,
            cfnn_models=cfnn_models,
            baseline_filter=lambda row: row.get("model") in physical_models or row.get("family") in physical_models,
        ),
    }


def grouped_loss_effects(
    rows: Iterable[dict],
    *,
    group_keys: tuple[str, ...],
    metric: str,
    cfnn_models: set[str],
    baseline_filter=None,
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    effects = []
    for key_values, items in sorted(grouped.items(), key=lambda item: item[0]):
        effect = loss_effect_row(
            rows=items,
            metric=metric,
            cfnn_models=cfnn_models,
            baseline_filter=baseline_filter,
        )
        for key, value in zip(group_keys, key_values):
            effect[key] = value
        effect["row_count"] = len(items)
        effects.append(effect)
    return effects


def summarize_effects(effects: Iterable[dict]) -> dict:
    valid = [row for row in effects if row.get("status") == "ok"]
    return {
        "effect_count": len(valid),
        "median_signed_cfnn_effect": _median(row.get("signed_cfnn_effect") for row in valid),
        "positive_effect_count": int(sum(1 for row in valid if float(row["signed_cfnn_effect"]) > 0.0)),
        "negative_effect_count": int(sum(1 for row in valid if float(row["signed_cfnn_effect"]) < 0.0)),
    }


def _eis_descriptor_stratum(descriptor: dict) -> str:
    if bool(descriptor.get("distributed_relaxation")):
        return "distributed_relaxation"
    pole_count = descriptor.get("known_pole_count")
    peak_count = descriptor.get("relaxation_peak_count")
    if pole_count is not None and int(pole_count) >= 4:
        return "multi_pole_relaxation"
    if peak_count is not None and int(peak_count) >= 3:
        return "multi_peak_relaxation"
    return "low_order_relaxation"


def audit_eis_neural_positive_edges(layered_rows: Iterable[dict]) -> dict:
    """Audit neural-only positive EIS cells against all/physical-prior layers."""
    rows = [
        dict(row) for row in layered_rows
        if row.get("status") == "ok" and row.get("task_id") is not None
    ]
    by_task_layer: dict[tuple[str, str], dict] = {
        (str(row.get("task_id")), str(row.get("effect_layer"))): row
        for row in rows
    }
    positive_edges = [
        row for row in rows
        if row.get("effect_layer") == "neural_only"
        and _finite(row.get("signed_cfnn_effect")) is not None
        and float(row["signed_cfnn_effect"]) > 0.0
    ]
    audited = []
    for row in sorted(positive_edges, key=lambda item: str(item.get("task_id"))):
        task_id = str(row["task_id"])
        physical = by_task_layer.get((task_id, "physical_prior_only"))
        all_layer = by_task_layer.get((task_id, "all_baselines"))
        physical_effect = _finite(physical.get("signed_cfnn_effect")) if physical else None
        all_effect = _finite(all_layer.get("signed_cfnn_effect")) if all_layer else None
        if physical_effect is None:
            status = "missing_physical_prior_comparison"
        elif physical_effect < 0.0 or (all_effect is not None and all_effect < 0.0):
            status = "physical_prior_eliminates_neural_edge"
        else:
            status = "candidate_neural_structural_edge"
        descriptor = dict(row.get("descriptor") or {})
        audited.append({
            "task_id": task_id,
            "edge_status": status,
            "neural_effect": float(row["signed_cfnn_effect"]),
            "physical_prior_effect": physical_effect,
            "all_baseline_effect": all_effect,
            "cfnn_metric": _finite(row.get("cfnn_metric")),
            "neural_best_baseline_model": row.get("best_baseline_model"),
            "physical_best_baseline_model": physical.get("best_baseline_model") if physical else None,
            "all_best_baseline_model": all_layer.get("best_baseline_model") if all_layer else None,
            "descriptor_stratum": _eis_descriptor_stratum(descriptor),
            "known_pole_count": descriptor.get("known_pole_count"),
            "relaxation_peak_count": descriptor.get("relaxation_peak_count"),
            "distributed_relaxation": bool(descriptor.get("distributed_relaxation", False)),
            "n_points": descriptor.get("n_points"),
            "noise": descriptor.get("noise"),
            "phase_total_turns": _finite(descriptor.get("phase_total_turns")),
            "max_normalized_curvature": _finite(descriptor.get("max_normalized_curvature")),
        })
    status_counts: dict[str, int] = {}
    for row in audited:
        status_counts[row["edge_status"]] = status_counts.get(row["edge_status"], 0) + 1
    return {
        "analysis_type": "eis_neural_positive_edge_audit",
        "positive_neural_edge_count": len(audited),
        "status_counts": status_counts,
        "positive_edge_rows": audited,
    }


def eis_descriptor_stratum_summary(descriptor_rows: Iterable[dict]) -> list[dict]:
    """Summarize EIS descriptor coverage by spectral stratum."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in descriptor_rows:
        descriptor = dict(row or {})
        if not descriptor:
            continue
        grouped[_eis_descriptor_stratum(descriptor)].append(descriptor)
    summaries = []
    for stratum, items in sorted(grouped.items()):
        summaries.append({
            "descriptor_stratum": stratum,
            "descriptor_count": int(len(items)),
            "median_known_pole_count": _median(row.get("known_pole_count") for row in items),
            "median_relaxation_peak_count": _median(row.get("relaxation_peak_count") for row in items),
            "median_frequency_points_per_decade": _median(row.get("frequency_points_per_decade") for row in items),
            "distributed_relaxation_count": int(sum(bool(row.get("distributed_relaxation")) for row in items)),
        })
    return summaries


def external_spectrum_boundary_gate(
    layered_rows: Iterable[dict],
    descriptor_rows: Iterable[dict],
    *,
    elimination_threshold: float = 0.80,
) -> dict:
    """Gate whether EIS remains a physical-prior boundary under external spectra."""
    rows = [
        dict(row) for row in layered_rows
        if row.get("status") == "ok" and row.get("task_id") is not None
    ]
    by_task_layer: dict[tuple[str, str], dict] = {
        (str(row.get("task_id")), str(row.get("effect_layer"))): row
        for row in rows
    }
    neural_positive = [
        row for row in rows
        if row.get("effect_layer") == "neural_only"
        and _finite(row.get("signed_cfnn_effect")) is not None
        and float(row["signed_cfnn_effect"]) > 0.0
    ]
    eliminated = []
    missing_physical = []
    surviving = []
    for row in neural_positive:
        task_id = str(row["task_id"])
        physical = by_task_layer.get((task_id, "physical_prior_only"))
        all_layer = by_task_layer.get((task_id, "all_baselines"))
        physical_effect = _finite(physical.get("signed_cfnn_effect")) if physical else None
        all_effect = _finite(all_layer.get("signed_cfnn_effect")) if all_layer else None
        if physical_effect is None:
            missing_physical.append(task_id)
            continue
        if physical_effect < 0.0 or (all_effect is not None and all_effect < 0.0):
            eliminated.append(task_id)
        if all_effect is not None and all_effect > 0.0:
            surviving.append(task_id)

    positive_count = len(neural_positive)
    elimination_rate = (
        float(len(eliminated) / positive_count)
        if positive_count else
        1.0
    )
    all_effects = [
        float(row["signed_cfnn_effect"]) for row in rows
        if row.get("effect_layer") == "all_baselines"
        and _finite(row.get("signed_cfnn_effect")) is not None
    ]
    if surviving:
        claim_status = "boundary_challenged_by_surviving_edge"
    elif positive_count and missing_physical and elimination_rate < float(elimination_threshold):
        claim_status = "insufficient_physical_prior_coverage"
    elif elimination_rate >= float(elimination_threshold) and _median(all_effects) is not None and float(_median(all_effects)) < 0.0:
        claim_status = "physical_prior_boundary_supported"
    else:
        claim_status = "boundary_inconclusive"

    return {
        "claim_status": claim_status,
        "neural_positive_edge_count": int(positive_count),
        "physical_prior_eliminated_edge_count": int(len(eliminated)),
        "physical_prior_missing_edge_count": int(len(missing_physical)),
        "all_baseline_surviving_positive_edge_count": int(len(surviving)),
        "physical_prior_elimination_rate": elimination_rate,
        "median_all_baseline_signed_cfnn_effect": _median(all_effects),
        "descriptor_stratum_rows": eis_descriptor_stratum_summary(descriptor_rows),
        "eliminated_task_ids": sorted(eliminated),
        "missing_physical_prior_task_ids": sorted(missing_physical),
        "surviving_all_baseline_positive_task_ids": sorted(surviving),
        "elimination_threshold": float(elimination_threshold),
    }


def hsc_condition_descriptor_rows(curves: Iterable[object]) -> list[dict]:
    """Build descriptor rows for measured hybrid-supercapacitor EIS conditions."""
    rows = []
    for curve in curves:
        conditions = dict(getattr(curve, "conditions", {}) or {})
        curve_id = str(getattr(curve, "curve_id"))
        frequency = np.asarray(getattr(curve, "frequency"), dtype=np.float64)
        response = np.asarray(getattr(curve, "response"))
        descriptor = eis_spectral_descriptors(
            f"hsc:{curve_id}",
            frequency,
            response,
            metadata={
                "task_family": "hybrid_supercapacitor_eis",
                "known_pole_count": None,
                "distributed_relaxation": False,
                "n_points": int(len(frequency)),
                "noise": None,
            },
        )
        descriptor.update({
            "curve_id": curve_id,
            "soc_percent": conditions.get("soc_percent"),
            "temperature_c": conditions.get("temperature_c"),
            "source_key": (getattr(curve, "metadata", {}) or {}).get("source_key"),
        })
        rows.append(descriptor)
    return rows


def hsc_voigt_physical_prior_rows(curves: Iterable[object], *, max_order: int = 4) -> list[dict]:
    """Fit deterministic Voigt physical-prior baselines to HSC condition curves."""
    import eis_utils as U

    max_order = max(1, int(max_order))
    rows = []
    for curve in curves:
        conditions = dict(getattr(curve, "conditions", {}) or {})
        curve_id = str(getattr(curve, "curve_id"))
        frequency = np.asarray(getattr(curve, "frequency"), dtype=np.float64).reshape(-1)
        response = _as_complex(np.asarray(getattr(curve, "response")))
        fits = []
        for order in range(1, max_order + 1):
            fit = U.fit_voigt(frequency, response, order)
            params = fit["params"]
            prediction = U.voigt_impedance(frequency, params["Rs"], params["R_list"], params["C_list"])
            curve_mse = float(np.mean(np.abs(prediction - response) ** 2))
            n = 2 * len(frequency)
            parameter_count = 1 + 2 * order
            rss = max(float(fit.get("rss", curve_mse * len(frequency))), 1e-12)
            aic = float(n * np.log(rss / n) + 2 * parameter_count)
            row = {
                "task_id": f"hsc:{curve_id}",
                "curve_id": curve_id,
                "model": f"Classical-Voigt-K{order}",
                "family": "physical_prior",
                "effect_layer": "physical_prior_only",
                "metric": "curve_mse",
                "curve_mse": curve_mse,
                "complex_r2": float(U.complex_r2(response, prediction)),
                "params_count": int(parameter_count),
                "fit_K": int(order),
                "fit_success": bool(fit.get("success")),
                "fit_cost": _finite(fit.get("cost")),
                "fit_rss": _finite(fit.get("rss")),
                "fit_aic": aic,
                "soc_percent": conditions.get("soc_percent"),
                "temperature_c": conditions.get("temperature_c"),
                "n_points": int(len(frequency)),
            }
            rows.append(row)
            fits.append(row)
        best = min(fits, key=lambda row: float(row["fit_aic"]))
        rows.append({
            **best,
            "model": "Classical-AIC",
            "fit_K": int(best["fit_K"]),
            "params_count": int(best["params_count"]),
            "selected_model": best["model"],
        })
    return rows


def _point_id_sha256(point_ids: Iterable[str] | None) -> str | None:
    if point_ids is None:
        return None
    payload = "\n".join(str(item) for item in point_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _complex_nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = _as_complex(np.asarray(truth))
    yhat = _as_complex(np.asarray(prediction))
    if len(y) != len(yhat):
        raise ValueError("truth and prediction lengths differ")
    numerator = float(np.sqrt(np.mean(np.abs(yhat - y) ** 2)))
    denominator = max(float(np.sqrt(np.mean(np.abs(y) ** 2))), np.finfo(float).eps)
    return float(numerator / denominator)


def hsc_same_split_voigt_prior_rows(bundles: Iterable[object], *, max_order: int = 4) -> list[dict]:
    """Fit Voigt priors on HSC train+validation points and score frozen test points."""
    import eis_utils as U

    max_order = max(1, int(max_order))
    rows = []
    for bundle in bundles:
        metadata = dict(getattr(bundle, "metadata", {}) or {})
        curve_id = str(metadata.get("curve_id"))
        data_seed = int(metadata.get("data_seed"))
        task_id = f"hsc:{curve_id}:data_seed:{data_seed}"
        conditions = dict(metadata.get("conditions", {}) or {})
        x_observed = np.concatenate([
            np.asarray(getattr(bundle, "x_train"), dtype=np.float64).reshape(-1),
            np.asarray(getattr(bundle, "x_validation"), dtype=np.float64).reshape(-1),
        ])
        y_observed = _as_complex(np.concatenate([
            np.asarray(getattr(bundle, "y_train")),
            np.asarray(getattr(bundle, "y_validation")),
        ], axis=0))
        x_test = np.asarray(getattr(bundle, "x_test"), dtype=np.float64).reshape(-1)
        y_test = _as_complex(np.asarray(getattr(bundle, "y_test")))
        if len(x_observed) < 3 or len(x_test) < 3:
            raise ValueError("same-split Voigt prior requires at least three observed and test points")
        fits = []
        for order in range(1, max_order + 1):
            fit = U.fit_voigt(x_observed, y_observed, order)
            params = fit["params"]
            observed_prediction = U.voigt_impedance(
                x_observed, params["Rs"], params["R_list"], params["C_list"]
            )
            test_prediction = U.voigt_impedance(
                x_test, params["Rs"], params["R_list"], params["C_list"]
            )
            observed_mse = float(np.mean(np.abs(observed_prediction - y_observed) ** 2))
            test_mse = float(np.mean(np.abs(test_prediction - y_test) ** 2))
            n = 2 * len(x_observed)
            parameter_count = 1 + 2 * order
            rss = max(float(np.sum(np.abs(observed_prediction - y_observed) ** 2)), 1e-12)
            aic = float(n * np.log(rss / n) + 2 * parameter_count)
            row = {
                "task_id": task_id,
                "curve_id": curve_id,
                "data_seed": data_seed,
                "model": f"Classical-Voigt-SameSplit-K{order}",
                "family": "physical_prior",
                "effect_layer": "physical_prior_only",
                "metric": "global_complex_nrmse",
                "global_complex_nrmse": _complex_nrmse(y_test, test_prediction),
                "test_curve_mse": test_mse,
                "fit_observed_curve_mse": observed_mse,
                "fit_aic": aic,
                "fit_K": int(order),
                "params_count": int(parameter_count),
                "fit_success": bool(fit.get("success")),
                "fit_cost": _finite(fit.get("cost")),
                "fit_rss": _finite(fit.get("rss")),
                "fit_observation_count": int(len(x_observed)),
                "test_point_count": int(len(x_test)),
                "train_point_count": int(len(np.asarray(getattr(bundle, "x_train")).reshape(-1))),
                "validation_point_count": int(len(np.asarray(getattr(bundle, "x_validation")).reshape(-1))),
                "train_point_ids_sha256": _point_id_sha256(metadata.get("train_point_ids")),
                "validation_point_ids_sha256": _point_id_sha256(metadata.get("validation_point_ids")),
                "test_point_ids_sha256": _point_id_sha256(metadata.get("test_point_ids")),
                "split_strategy": metadata.get("split_strategy"),
                "soc_percent": conditions.get("soc_percent"),
                "temperature_c": conditions.get("temperature_c"),
            }
            rows.append(row)
            fits.append(row)
        best = min(fits, key=lambda row: float(row["fit_aic"]))
        rows.append({
            **best,
            "model": "Classical-AIC-SameSplit",
            "selected_model": best["model"],
            "fit_K": int(best["fit_K"]),
            "params_count": int(best["params_count"]),
        })
    return rows


def hsc_neural_metric_rows(records: Iterable[dict], *, metric: str = "global_complex_nrmse") -> list[dict]:
    """Extract comparable HSC neural metric rows from frozen evaluation records."""
    rows = []
    for record in records:
        if record.get("task") != "hybrid_supercapacitor_eis" or record.get("status") != "ok":
            continue
        value = _finite((record.get("test_metrics") or {}).get(metric))
        if value is None:
            continue
        dataset = dict(record.get("dataset") or {})
        curve_id = dataset.get("curve_id")
        data_seed = record.get("data_seed", dataset.get("data_seed"))
        if curve_id is None or data_seed is None:
            continue
        conditions = dict(dataset.get("conditions") or {})
        row = {
            "task_id": f"hsc:{curve_id}:data_seed:{int(data_seed)}",
            "curve_id": str(curve_id),
            "data_seed": int(data_seed),
            "model": record.get("family"),
            "family": record.get("family"),
            "metric": metric,
            metric: value,
            "target_budget": record.get("target_budget"),
            "selected_common_budget": record.get("selected_common_budget"),
            "init_seed": record.get("init_seed"),
            "train_point_count": dataset.get("train_point_ids_count"),
            "validation_point_count": dataset.get("validation_point_ids_count"),
            "test_point_count": dataset.get("test_point_ids_count"),
            "test_point_ids_sha256": dataset.get("test_point_ids_sha256"),
            "split_strategy": dataset.get("split_strategy"),
            "soc_percent": conditions.get("soc_percent"),
            "temperature_c": conditions.get("temperature_c"),
        }
        rows.append(row)
    return rows


def _median_metric_rows(rows: Iterable[dict], *, metric: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get(metric))
        if value is None:
            continue
        grouped[(row.get("task_id"), row.get("model"))].append(row)
    aggregated = []
    for (task_id, model), items in sorted(grouped.items(), key=lambda item: item[0]):
        template = dict(items[0])
        template[metric] = float(np.median([float(item[metric]) for item in items]))
        template["source_row_count"] = int(len(items))
        template["aggregation"] = "median_by_task_id_model"
        aggregated.append(template)
    return aggregated


def hsc_same_split_layered_effect_rows(
    *,
    neural_rows: Iterable[dict],
    physical_rows: Iterable[dict],
    metric: str = "global_complex_nrmse",
) -> list[dict]:
    """Compare CFNN with neural and same-split physical-prior HSC baselines."""
    neural_aggregated = _median_metric_rows(neural_rows, metric=metric)
    physical_aggregated = [
        dict(row) for row in _median_metric_rows(physical_rows, metric=metric)
        if row.get("model") == "Classical-AIC-SameSplit"
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in neural_aggregated + physical_aggregated:
        if row.get("task_id") is not None:
            grouped[str(row["task_id"])].append(row)
    output = []
    neural_baselines = {
        "Fourier-feature MLP",
        "Gaussian RBF",
        "KAN",
        "Local nested CF control",
        "MLP",
        "Rational activation NN",
        "SIREN",
    }
    for task_id, items in sorted(grouped.items()):
        effects = layered_loss_effect_rows(
            rows=items,
            metric=metric,
            cfnn_models={"CFNN"},
            neural_baseline_models=neural_baselines,
            physical_prior_models={"Classical-AIC-SameSplit"},
        )
        for layer, effect in effects.items():
            row = dict(effect)
            row.update({
                "task_id": task_id,
                "effect_layer": layer,
                "metric": metric,
                "row_count": int(len(items)),
            })
            output.append(row)
    return output
