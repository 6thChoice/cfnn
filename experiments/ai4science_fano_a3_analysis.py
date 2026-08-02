"""Preregistered robustness audit for Microwave Fano A3 summaries."""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


A3_PREREG_REQUIRED_METRICS = (
    "median_global_complex_nrmse",
    "median_peak_window_complex_nrmse",
    "median_f0_abs_error_hz",
    "median_quality_factor_relative_error",
    "median_q_abs_error",
    "parameter_recovery_failure_rate",
)


def _finite(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _condition_key(row: dict) -> tuple:
    return (
        int(row.get("observation_budget")),
        float(row.get("noise_snr_db", 0.0) or 0.0),
        float(row.get("missing_fraction", 0.0) or 0.0),
        float(row.get("frequency_drift_ppm", 0.0) or 0.0),
    )


def _metric_effect(cfnn_value: float, baseline_value: float) -> float:
    """Positive means CFNN has lower error/failure than the baseline."""
    if abs(float(baseline_value)) > np.finfo(float).eps:
        return float((float(baseline_value) - float(cfnn_value)) / abs(float(baseline_value)))
    return float(float(baseline_value) - float(cfnn_value))


def _group_by_condition(a3_summary: dict) -> dict[tuple, dict[str, dict]]:
    by_condition: dict[tuple, dict[str, dict]] = {}
    for row in a3_summary.get("by_family_budget_condition", []):
        by_condition.setdefault(_condition_key(row), {})[str(row.get("family"))] = dict(row)
    return by_condition


def summarize_preregistered_a3_robustness(
    a3_summaries: Iterable[tuple[str, dict]],
    *,
    baseline_family: str = "CFNN",
    comparison_families: Iterable[str] = ("MLP", "Local nested CF control"),
    required_metrics: Iterable[str] = A3_PREREG_REQUIRED_METRICS,
) -> dict:
    """Evaluate A3 robustness under response, parameter, and failure guardrails.

    This operates on production A3 summary rows. A condition passes only if CFNN
    is non-inferior or better on all required metrics against every requested
    comparison family under the same budget/noise/missing/drift condition.
    """
    comparison_families = tuple(str(value) for value in comparison_families)
    required_metrics = tuple(str(value) for value in required_metrics)
    comparison_rows = []
    condition_gate_rows = []

    for source_label, summary in a3_summaries:
        by_condition = _group_by_condition(summary)
        for condition, family_rows in sorted(by_condition.items()):
            cfnn = family_rows.get(baseline_family)
            if not cfnn:
                continue
            budget, snr, missing, drift = condition
            passed_comparisons = set()
            for comparison in comparison_families:
                baseline = family_rows.get(comparison)
                metric_effects = {}
                failed = []
                missing_metrics = []
                if not baseline:
                    missing_metrics = list(required_metrics)
                else:
                    for metric in required_metrics:
                        cfnn_value = _finite(cfnn.get(metric))
                        baseline_value = _finite(baseline.get(metric))
                        if cfnn_value is None or baseline_value is None:
                            missing_metrics.append(metric)
                            continue
                        effect = _metric_effect(cfnn_value, baseline_value)
                        metric_effects[metric] = effect
                        if effect < 0.0:
                            failed.append(metric)
                passes = bool(not failed and not missing_metrics)
                if passes:
                    passed_comparisons.add(comparison)
                comparison_rows.append({
                    "source_label": str(source_label),
                    "observation_budget": budget,
                    "noise_snr_db": snr,
                    "missing_fraction": missing,
                    "frequency_drift_ppm": drift,
                    "comparison_family": comparison,
                    "required_metrics": list(required_metrics),
                    "metric_effects": metric_effects,
                    "failed_metrics": failed,
                    "missing_metrics": missing_metrics,
                    "comparison_status": (
                        "passes_preregistered_robustness"
                        if passes else
                        "fails_preregistered_robustness"
                    ),
                })
            condition_passes = all(comparison in passed_comparisons for comparison in comparison_families)
            condition_gate_rows.append({
                "source_label": str(source_label),
                "observation_budget": budget,
                "noise_snr_db": snr,
                "missing_fraction": missing,
                "frequency_drift_ppm": drift,
                "passed_comparison_families": sorted(passed_comparisons),
                "required_comparison_families": list(comparison_families),
                "condition_status": (
                    "passes_preregistered_robustness"
                    if condition_passes else
                    "blocked_by_baseline_guardrail"
                ),
            })

    passed_conditions = [
        row for row in condition_gate_rows
        if row["condition_status"] == "passes_preregistered_robustness"
    ]
    mlp_only_or_partial = [
        row for row in condition_gate_rows
        if row["condition_status"] != "passes_preregistered_robustness"
        and "MLP" in row["passed_comparison_families"]
    ]
    return {
        "analysis_type": "microwave_fano_a3_preregistered_robustness_audit",
        "baseline_family": baseline_family,
        "comparison_families": list(comparison_families),
        "required_metrics": list(required_metrics),
        "condition_comparison_rows": comparison_rows,
        "condition_gate_rows": condition_gate_rows,
        "condition_count": len(condition_gate_rows),
        "passed_condition_count": len(passed_conditions),
        "mlp_only_or_partial_condition_count": len(mlp_only_or_partial),
        "overall_status": (
            "passes_some_conditions"
            if passed_conditions else
            "does_not_pass_local_control_guardrail"
        ),
    }
