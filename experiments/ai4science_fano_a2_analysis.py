"""Enhanced aggregation for Microwave Fano A2 sparse-observation runs."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


A2_METRICS = (
    "global_complex_nrmse",
    "peak_window_complex_nrmse",
    "f0_abs_error_hz",
    "quality_factor_relative_error",
    "q_abs_error",
    "magnitude_peak_abs_error_hz",
    "magnitude_valley_abs_error_hz",
    "phase_transition_abs_error_hz",
)

PREREG_A2_REQUIRED_METRICS = (
    "peak_window_complex_nrmse",
    "f0_abs_error_hz",
    "quality_factor_relative_error",
    "q_abs_error",
)
PREREG_A2_GLOBAL_GUARDRAIL_METRIC = "global_complex_nrmse"


def _to_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def normalize_record(record: dict) -> dict:
    normalized = dict(record)
    for field in ("observation_budget", "split_seed", "init_seed", "model_budget"):
        if field in normalized:
            normalized[field] = _to_int(normalized[field])
    for field in A2_METRICS:
        if field in normalized:
            normalized[field] = _to_float(normalized[field])
    return normalized


def load_a2_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return [normalize_record(row) for row in csv.DictReader(handle)]


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _metric_summary(records: list[dict], metric: str) -> dict:
    values = [record.get(metric) for record in records if record.get(metric) is not None]
    return {
        "metric": metric,
        "valid_count": len(values),
        "median": _quantile(values, 0.50),
        "q25": _quantile(values, 0.25),
        "q75": _quantile(values, 0.75),
    }


def _job_identity(record: dict) -> tuple:
    return (
        record.get("curve_id"),
        int(record.get("observation_budget")),
        int(record.get("split_seed")),
        int(record.get("init_seed")),
    )


def _paired_metric_summary(
    baseline_records: list[dict],
    comparison_records: list[dict],
    *,
    baseline_family: str,
    comparison_family: str,
    observation_budget: int,
    metric: str,
) -> dict:
    comparison_by_identity = {_job_identity(record): record for record in comparison_records}
    deltas = []
    relative_deltas = []
    baseline_wins = 0
    ties = 0
    for baseline in baseline_records:
        comparison = comparison_by_identity.get(_job_identity(baseline))
        if comparison is None:
            continue
        baseline_value = baseline.get(metric)
        comparison_value = comparison.get(metric)
        if baseline_value is None or comparison_value is None:
            continue
        delta = float(baseline_value) - float(comparison_value)
        deltas.append(delta)
        if abs(float(comparison_value)) > np.finfo(float).eps:
            relative_deltas.append(delta / abs(float(comparison_value)))
        if baseline_value < comparison_value:
            baseline_wins += 1
        elif baseline_value == comparison_value:
            ties += 1

    valid_count = len(deltas)
    return {
        "baseline_family": baseline_family,
        "comparison_family": comparison_family,
        "observation_budget": int(observation_budget),
        "metric": metric,
        "valid_pair_count": valid_count,
        "baseline_win_count": baseline_wins,
        "tie_count": ties,
        "baseline_win_rate": float(baseline_wins / valid_count) if valid_count else None,
        "median_paired_delta": _quantile(deltas, 0.50),
        "q25_paired_delta": _quantile(deltas, 0.25),
        "q75_paired_delta": _quantile(deltas, 0.75),
        "median_relative_delta": _quantile(relative_deltas, 0.50),
    }


def build_enhanced_summary(
    records: Iterable[dict],
    *,
    baseline_family: str = "CFNN",
    metrics: Iterable[str] = A2_METRICS,
) -> dict:
    rows = [normalize_record(record) for record in records]
    metrics = tuple(metrics)
    families = sorted({str(row["family"]) for row in rows})
    budgets = sorted({int(row["observation_budget"]) for row in rows})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], int(row["observation_budget"]))].append(row)

    family_budget_status = []
    family_budget_metrics = []
    for family in families:
        for budget in budgets:
            items = grouped.get((family, budget), [])
            if not items:
                continue
            ok_count = sum(1 for item in items if item.get("parameter_recovery_status") == "ok")
            family_budget_status.append({
                "family": family,
                "observation_budget": int(budget),
                "record_count": len(items),
                "parameter_recovery_ok_count": ok_count,
                "parameter_recovery_failure_count": len(items) - ok_count,
                "parameter_recovery_failure_rate": float((len(items) - ok_count) / len(items)),
            })
            for metric in metrics:
                if any(metric in item for item in items):
                    family_budget_metrics.append({
                        "family": family,
                        "observation_budget": int(budget),
                        **_metric_summary(items, metric),
                    })

    pairwise = []
    for budget in budgets:
        baseline_records = grouped.get((baseline_family, budget), [])
        if not baseline_records:
            continue
        for family in families:
            if family == baseline_family:
                continue
            comparison_records = grouped.get((family, budget), [])
            if not comparison_records:
                continue
            for metric in metrics:
                pairwise.append(_paired_metric_summary(
                    baseline_records,
                    comparison_records,
                    baseline_family=baseline_family,
                    comparison_family=family,
                    observation_budget=budget,
                    metric=metric,
                ))

    return {
        "task": "microwave_fano_a2_enhanced_analysis",
        "record_count": len(rows),
        "baseline_family": baseline_family,
        "families": families,
        "observation_budgets": budgets,
        "curve_ids": sorted({str(row["curve_id"]) for row in rows}),
        "split_seeds": sorted({int(row["split_seed"]) for row in rows if row.get("split_seed") is not None}),
        "init_seeds": sorted({int(row["init_seed"]) for row in rows if row.get("init_seed") is not None}),
        "metrics": list(metrics),
        "family_budget_status": family_budget_status,
        "family_budget_metrics": family_budget_metrics,
        "pairwise_baseline_comparisons": pairwise,
    }


def write_enhanced_summary(input_csv: Path, output_json: Path, *, baseline_family: str = "CFNN") -> dict:
    summary = build_enhanced_summary(load_a2_csv(input_csv), baseline_family=baseline_family)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def summarize_preregistered_a2_success(
    summary: dict,
    *,
    budgets: Iterable[int] = (32, 64, 128),
    comparison_families: Iterable[str] = ("MLP", "Local nested CF control"),
    required_metrics: Iterable[str] = PREREG_A2_REQUIRED_METRICS,
    global_guardrail_metric: str = PREREG_A2_GLOBAL_GUARDRAIL_METRIC,
) -> dict:
    """Evaluate the A2 preregistered response+parameter success criterion."""
    budgets = tuple(int(value) for value in budgets)
    comparison_families = tuple(str(value) for value in comparison_families)
    required_metrics = tuple(str(value) for value in required_metrics)
    pairwise = {
        (
            int(row.get("observation_budget")),
            str(row.get("comparison_family")),
            str(row.get("metric")),
        ): row
        for row in summary.get("pairwise_baseline_comparisons", [])
    }
    rows = []
    passed_budget_comparisons = set()
    for budget in budgets:
        for comparison in comparison_families:
            metric_effects = {}
            failed = []
            missing = []
            for metric in required_metrics:
                row = pairwise.get((budget, comparison, metric))
                if row is None or row.get("median_relative_delta") is None:
                    missing.append(metric)
                    continue
                effect = -float(row["median_relative_delta"])
                metric_effects[metric] = effect
                if effect <= 0.0:
                    failed.append(metric)
            guardrail_row = pairwise.get((budget, comparison, global_guardrail_metric))
            if guardrail_row is None or guardrail_row.get("median_relative_delta") is None:
                global_effect = None
                global_pass = False
            else:
                global_effect = -float(guardrail_row["median_relative_delta"])
                global_pass = bool(global_effect >= 0.0)
            success = bool(not failed and not missing and global_pass)
            if success:
                passed_budget_comparisons.add((budget, comparison))
            rows.append({
                "observation_budget": budget,
                "comparison_family": comparison,
                "required_metrics": list(required_metrics),
                "metric_effects": metric_effects,
                "global_guardrail_metric": global_guardrail_metric,
                "global_guardrail_effect": global_effect,
                "global_guardrail_pass": global_pass,
                "failed_required_metrics": failed,
                "missing_required_metrics": missing,
                "success_status": (
                    "passes_preregistered_success"
                    if success else
                    "fails_preregistered_success"
                ),
            })
    passed_budgets = sorted({
        budget for budget in budgets
        if all((budget, comparison) in passed_budget_comparisons for comparison in comparison_families)
    })
    return {
        "analysis_type": "microwave_fano_a2_preregistered_success_audit",
        "budgets": list(budgets),
        "comparison_families": list(comparison_families),
        "required_metrics": list(required_metrics),
        "global_guardrail_metric": global_guardrail_metric,
        "budget_comparison_rows": rows,
        "passed_budgets": passed_budgets,
        "passed_budget_count": len(passed_budgets),
        "overall_status": (
            "passes_at_some_main_budgets"
            if passed_budgets else
            "does_not_pass_main_budget_gate"
        ),
    }
