"""Unified Package E applicability table across Fano and EIS evidence."""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _finite(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _boundary_from_coverage(coverage: float | None, dangerous_rate: float | None) -> float | None:
    if coverage is None and dangerous_rate is None:
        return None
    coverage_deficit = max(0.0, 0.80 - float(coverage)) if coverage is not None else 0.0
    return float(coverage_deficit + float(dangerous_rate or 0.0))


def _effect_boundary(effect: float | None) -> float | None:
    if effect is None:
        return None
    return float(max(0.0, -float(effect)))


def fano_applicability_rows(fano_summary: dict) -> list[dict]:
    rows = []
    for row in fano_summary.get("ranked_curve_rows", []):
        coverage = _finite(row.get("median_cross_curve_joint_coverage"))
        dangerous = _finite(row.get("median_cross_curve_dangerous_point_failure_rate"))
        rows.append({
            "domain": "Microwave Fano",
            "task_id": row["curve_id"],
            "evidence_axis": "reliability_boundary",
            "effect_layer": "cross_curve_calibration",
            "metric": "joint_coverage",
            "cfnn_effect": None,
            "best_baseline_model": None,
            "cross_curve_joint_coverage": coverage,
            "cross_curve_peak_coverage": _finite(row.get("median_cross_curve_peak_window_coverage")),
            "dangerous_failure_rate": dangerous,
            "same_to_cross_scale_ratio": _finite(row.get("median_same_to_cross_scale_ratio")),
            "phase_total_turns": _finite(row.get("phase_total_turns")),
            "max_curvature": _finite(row.get("max_normalized_magnitude_curvature")),
            "linewidth_or_peak_width_fraction": _finite(row.get("linewidth_fraction")),
            "relaxation_peak_count": None,
            "known_pole_count": None,
            "distributed_relaxation": False,
            "frequency_points_per_decade": None,
            "noise": None,
            "fit_residual": _finite(row.get("fano_relative_magnitude_rmse")),
            "boundary_score": _boundary_from_coverage(coverage, dangerous),
        })
    for row in fano_summary.get("ranked_conformal_curve_rows", []):
        coverage = _finite(row.get("median_calibrated_joint_coverage"))
        dangerous = _finite(row.get("median_calibrated_dangerous_point_failure_rate"))
        layer = str(row.get("conformal_layer"))
        rows.append({
            "domain": "Microwave Fano",
            "task_id": f"{row['curve_id']}_{layer}",
            "evidence_axis": "reliability_boundary",
            "effect_layer": f"prediction_level_{layer}",
            "metric": "joint_coverage",
            "cfnn_effect": None,
            "best_baseline_model": None,
            "cross_curve_joint_coverage": coverage,
            "cross_curve_peak_coverage": _finite(row.get("median_calibrated_peak_window_coverage")),
            "dangerous_failure_rate": dangerous,
            "same_to_cross_scale_ratio": None,
            "calibration_scale": _finite(row.get("median_calibration_scale")),
            "phase_total_turns": _finite(row.get("phase_total_turns")),
            "max_curvature": _finite(row.get("max_normalized_magnitude_curvature")),
            "linewidth_or_peak_width_fraction": _finite(row.get("linewidth_fraction")),
            "relaxation_peak_count": None,
            "known_pole_count": None,
            "distributed_relaxation": False,
            "frequency_points_per_decade": None,
            "noise": None,
            "fit_residual": _finite(row.get("fano_relative_magnitude_rmse")),
            "boundary_score": _finite(row.get("median_boundary_score")) or _boundary_from_coverage(coverage, dangerous),
        })
    return rows


def _select_policy_row(rows: Iterable[dict], *, conformal_layer: str | None = None) -> dict:
    rows = [dict(row) for row in rows or [] if isinstance(row, dict)]
    if conformal_layer:
        for row in rows:
            if row.get("conformal_layer") == conformal_layer:
                return row
    return rows[0] if rows else {}


def summarize_fano_reliability_claim_gate(package_e_summary: dict) -> dict:
    """Summarize Fano D/E reliability as ordinary support plus high-risk guardrails."""
    fano_summary = package_e_summary.get("fano_boundary_summary") or package_e_summary
    policy = _select_policy_row(
        fano_summary.get("descriptor_guardrail_policy_audits", []),
        conformal_layer="global_split_conformal",
    )
    escalation = _select_policy_row(
        fano_summary.get("descriptor_same_curve_escalation_policy_audits", []),
    )
    accepted_failure_rate = _finite(policy.get("accepted_failure_rate"))
    failure_capture_rate = _finite(policy.get("failure_capture_rate"))
    post_policy_failure_rate = _finite(escalation.get("post_policy_failure_rate"))
    highrisk_curve_ids = list(
        escalation.get("escalated_curve_ids")
        or policy.get("rejected_curve_ids")
        or []
    )
    ordinary_pass = bool(
        accepted_failure_rate is not None
        and accepted_failure_rate <= 0.0
        and failure_capture_rate is not None
        and failure_capture_rate >= 1.0
    )
    full_pass = bool(ordinary_pass and post_policy_failure_rate == 0.0 and not highrisk_curve_ids)
    if full_pass:
        claim_status = "full_reliability_supported"
    elif ordinary_pass and highrisk_curve_ids:
        claim_status = "ordinary_reliability_supported_highrisk_requires_escalation"
    elif ordinary_pass:
        claim_status = "ordinary_reliability_supported_with_unresolved_escalation"
    else:
        claim_status = "reliability_not_supported_or_incomplete"
    return {
        "analysis_type": "fano_reliability_claim_gate",
        "claim_status": claim_status,
        "full_reliability_claim_pass": full_pass,
        "ordinary_curve_reliability_pass": ordinary_pass,
        "policy": policy.get("policy"),
        "conformal_layer": policy.get("conformal_layer"),
        "accepted_curve_count": policy.get("accepted_curve_count"),
        "accepted_failure_rate": accepted_failure_rate,
        "accepted_median_joint_coverage": _finite(policy.get("accepted_median_joint_coverage")),
        "accepted_median_dangerous_failure_rate": _finite(policy.get("accepted_median_dangerous_failure_rate")),
        "failure_capture_rate": failure_capture_rate,
        "highrisk_curve_ids": highrisk_curve_ids,
        "rejected_failure_rate": _finite(policy.get("rejected_failure_rate")),
        "post_policy_failure_rate": post_policy_failure_rate,
        "escalated_median_same_curve_joint_coverage": _finite(
            escalation.get("escalated_median_same_curve_joint_coverage")
        ),
        "escalated_median_same_curve_dangerous_failure_rate": _finite(
            escalation.get("escalated_median_same_curve_dangerous_failure_rate")
        ),
    }


def eis_applicability_rows(eis_summary: dict) -> list[dict]:
    rows = []
    for row in eis_summary.get("layered_effect_descriptor_rows", []):
        if row.get("status") != "ok":
            continue
        descriptor = row.get("descriptor", {})
        effect = _finite(row.get("signed_cfnn_effect"))
        rows.append({
            "domain": "EIS",
            "task_id": row["task_id"],
            "evidence_axis": "accuracy_effect",
            "effect_layer": row.get("effect_layer"),
            "metric": row.get("metric"),
            "cfnn_effect": effect,
            "cfnn_metric": _finite(row.get("cfnn_metric")),
            "best_baseline_model": row.get("best_baseline_model"),
            "best_baseline_metric": _finite(row.get("best_baseline_metric")),
            "cross_curve_joint_coverage": None,
            "cross_curve_peak_coverage": None,
            "dangerous_failure_rate": None,
            "same_to_cross_scale_ratio": None,
            "phase_total_turns": _finite(descriptor.get("phase_total_turns")),
            "max_curvature": _finite(descriptor.get("max_normalized_curvature")),
            "linewidth_or_peak_width_fraction": None,
            "relaxation_peak_count": descriptor.get("relaxation_peak_count"),
            "known_pole_count": descriptor.get("known_pole_count"),
            "distributed_relaxation": bool(descriptor.get("distributed_relaxation", False)),
            "frequency_points_per_decade": _finite(descriptor.get("frequency_points_per_decade")),
            "noise": _finite(descriptor.get("noise")),
            "fit_residual": None,
            "boundary_score": _effect_boundary(effect),
        })
    return rows


def _effect_values(rows: Iterable[dict], *, domain: str, layer: str) -> list[float]:
    values = [
        _finite(row.get("cfnn_effect"))
        for row in rows
        if row.get("domain") == domain and row.get("effect_layer") == layer
    ]
    return [value for value in values if value is not None]


def summarize_eis_applicability_claim_gate(package_e_summary: dict) -> dict:
    """Turn EIS positive/negative cells into an applicability-boundary claim gate."""
    rows = [dict(row) for row in package_e_summary.get("unified_rows", [])]
    all_effects = _effect_values(rows, domain="EIS", layer="all_baselines")
    neural_effects = _effect_values(rows, domain="EIS", layer="neural_only")
    physical_effects = _effect_values(rows, domain="EIS", layer="physical_prior_only")
    audit = package_e_summary.get("eis_neural_positive_edge_audit", {}) or {}
    eliminated = int((audit.get("status_counts") or {}).get("physical_prior_eliminates_neural_edge", 0))
    neural_positive = int(audit.get("positive_neural_edge_count") or sum(1 for value in neural_effects if value > 0.0))
    physical_negative_count = int(sum(1 for value in physical_effects if value < 0.0))
    all_negative_count = int(sum(1 for value in all_effects if value < 0.0))
    general_pass = bool(all_effects and all(value > 0.0 for value in all_effects))
    if physical_negative_count > 0 and neural_positive == eliminated:
        claim_status = "physical_prior_boundary_supported"
    elif general_pass:
        claim_status = "candidate_general_cfnn_advantage"
    else:
        claim_status = "mixed_or_incomplete_boundary"
    return {
        "analysis_type": "eis_applicability_claim_gate",
        "claim_status": claim_status,
        "general_cfnn_advantage_pass": general_pass,
        "all_baseline_count": len(all_effects),
        "all_baseline_negative_count": all_negative_count,
        "all_baseline_median_effect": float(np.median(all_effects)) if all_effects else None,
        "neural_only_count": len(neural_effects),
        "neural_only_positive_count": int(sum(1 for value in neural_effects if value > 0.0)),
        "neural_only_median_effect": float(np.median(neural_effects)) if neural_effects else None,
        "physical_prior_count": len(physical_effects),
        "physical_prior_negative_count": physical_negative_count,
        "physical_prior_median_effect": float(np.median(physical_effects)) if physical_effects else None,
        "neural_positive_edge_count": neural_positive,
        "physical_prior_eliminated_edge_count": eliminated,
        "positive_edge_status_counts": audit.get("status_counts") or {},
    }


def fano_a2_accuracy_effect_rows(
    a2_summary: dict,
    *,
    budgets: set[int] | None = None,
    comparison_families: set[str] | None = None,
    metrics: set[str] | None = None,
) -> list[dict]:
    """Convert A2 paired CFNN-vs-baseline summaries to signed effect rows."""
    budgets = set(budgets or {32, 64, 128})
    comparison_families = set(comparison_families or {"MLP", "Local nested CF control"})
    metrics = set(metrics or {
        "global_complex_nrmse",
        "peak_window_complex_nrmse",
        "f0_abs_error_hz",
        "quality_factor_relative_error",
        "q_abs_error",
    })
    rows = []
    for row in a2_summary.get("pairwise_baseline_comparisons", []):
        budget = int(row.get("observation_budget"))
        comparison = str(row.get("comparison_family"))
        metric = str(row.get("metric"))
        relative_delta = _finite(row.get("median_relative_delta"))
        if budget not in budgets or comparison not in comparison_families or metric not in metrics:
            continue
        if relative_delta is None:
            continue
        effect = -float(relative_delta)
        rows.append({
            "domain": "Microwave Fano",
            "task_id": f"a2_budget{budget}_{metric}_vs_{comparison}",
            "evidence_axis": "accuracy_effect",
            "effect_layer": f"a2_vs_{comparison}",
            "metric": metric,
            "cfnn_effect": effect,
            "cfnn_metric": None,
            "best_baseline_model": comparison,
            "best_baseline_metric": None,
            "cross_curve_joint_coverage": None,
            "cross_curve_peak_coverage": None,
            "dangerous_failure_rate": None,
            "same_to_cross_scale_ratio": None,
            "phase_total_turns": None,
            "max_curvature": None,
            "linewidth_or_peak_width_fraction": None,
            "relaxation_peak_count": None,
            "known_pole_count": None,
            "distributed_relaxation": False,
            "frequency_points_per_decade": None,
            "noise": None,
            "fit_residual": None,
            "observation_budget": budget,
            "valid_pair_count": int(row.get("valid_pair_count") or 0),
            "cfnn_win_rate": _finite(row.get("baseline_win_rate")),
            "boundary_score": _effect_boundary(effect),
        })
    return rows


def _condition_key(row: dict) -> tuple:
    return (
        int(row.get("observation_budget")),
        float(row.get("noise_snr_db", 0.0) or 0.0),
        float(row.get("missing_fraction", 0.0) or 0.0),
        float(row.get("frequency_drift_ppm", 0.0) or 0.0),
    )


def _slug_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number).replace(".", "p")


def _a3_public_metric(metric: str) -> str:
    return metric.removeprefix("median_")


def fano_a3_robustness_effect_rows(
    a3_summary: dict,
    *,
    source_label: str,
    comparison_families: set[str] | None = None,
    metrics: set[str] | None = None,
) -> list[dict]:
    """Convert A3 robustness summaries to signed same-condition CFNN effect rows."""
    comparison_families = set(comparison_families or {"MLP", "Local nested CF control"})
    metrics = set(metrics or {
        "median_global_complex_nrmse",
        "median_peak_window_complex_nrmse",
        "median_f0_abs_error_hz",
        "median_quality_factor_relative_error",
        "median_q_abs_error",
        "parameter_recovery_failure_rate",
    })
    by_condition: dict[tuple, dict[str, dict]] = {}
    for row in a3_summary.get("by_family_budget_condition", []):
        by_condition.setdefault(_condition_key(row), {})[str(row.get("family"))] = row

    rows = []
    for condition, family_rows in sorted(by_condition.items()):
        cfnn = family_rows.get("CFNN")
        if not cfnn:
            continue
        budget, snr, missing, drift = condition
        for comparison in sorted(comparison_families):
            baseline = family_rows.get(comparison)
            if not baseline:
                continue
            for metric in sorted(metrics):
                cfnn_metric = _finite(cfnn.get(metric))
                baseline_metric = _finite(baseline.get(metric))
                if cfnn_metric is None or baseline_metric in (None, 0.0):
                    continue
                effect = (float(baseline_metric) - float(cfnn_metric)) / float(baseline_metric)
                public_metric = _a3_public_metric(metric)
                rows.append({
                    "domain": "Microwave Fano",
                    "task_id": (
                        f"a3_{source_label}_budget{budget}_snr{_slug_number(snr)}"
                        f"_missing{_slug_number(missing)}_drift{_slug_number(drift)}"
                        f"_{public_metric}_vs_{comparison}"
                    ),
                    "evidence_axis": "robustness_effect",
                    "effect_layer": f"a3_{source_label}_vs_{comparison}",
                    "metric": public_metric,
                    "cfnn_effect": float(effect),
                    "cfnn_metric": cfnn_metric,
                    "best_baseline_model": comparison,
                    "best_baseline_metric": baseline_metric,
                    "cross_curve_joint_coverage": None,
                    "cross_curve_peak_coverage": None,
                    "dangerous_failure_rate": None,
                    "same_to_cross_scale_ratio": None,
                    "phase_total_turns": None,
                    "max_curvature": None,
                    "linewidth_or_peak_width_fraction": None,
                    "relaxation_peak_count": None,
                    "known_pole_count": None,
                    "distributed_relaxation": False,
                    "frequency_points_per_decade": None,
                    "noise": snr,
                    "fit_residual": None,
                    "observation_budget": budget,
                    "missing_fraction": missing,
                    "frequency_drift_ppm": drift,
                    "record_count": int(cfnn.get("record_count") or 0),
                    "baseline_record_count": int(baseline.get("record_count") or 0),
                    "boundary_score": _effect_boundary(effect),
                })
    return rows


def _b_condition_key(row: dict) -> tuple:
    return (
        str(row.get("experiment")),
        str(row.get("scenario")),
        row.get("calibration_frequency_count"),
    )


def _public_metric(metric: str) -> str:
    return metric.removeprefix("median_").removeprefix("identifiable_")


def _calibration_slug(value) -> str:
    return "full" if value is None else str(int(value))


def microstrip_b_device_effect_rows(
    microstrip_summary: dict,
    *,
    comparison_families: set[str] | None = None,
    metrics: set[str] | None = None,
) -> list[dict]:
    """Convert Package B same-condition device generalization summaries to signed CFNN effect rows."""
    comparison_families = set(comparison_families or {"MLP", "Local nested CF control"})
    metrics = set(metrics or {
        "median_complex_nrmse",
        "median_peak_window_complex_nrmse",
        "median_resonance_frequency_abs_error_hz",
        "median_quality_factor_relative_error",
        "median_phase_transition_abs_error_hz",
    })
    by_condition: dict[tuple, dict[str, dict]] = {}
    for row in microstrip_summary.get("summary_rows", []):
        by_condition.setdefault(_b_condition_key(row), {})[str(row.get("family"))] = row

    rows = []
    for condition, family_rows in sorted(by_condition.items(), key=lambda item: str(item[0])):
        cfnn = family_rows.get("CFNN")
        if not cfnn:
            continue
        experiment, scenario, calibration = condition
        experiment_layer = str(experiment).lower()
        for comparison in sorted(comparison_families):
            baseline = family_rows.get(comparison)
            if not baseline:
                continue
            for metric in sorted(metrics):
                cfnn_metric = _finite(cfnn.get(metric))
                baseline_metric = _finite(baseline.get(metric))
                if cfnn_metric is None or baseline_metric in (None, 0.0):
                    continue
                effect = (float(baseline_metric) - float(cfnn_metric)) / float(baseline_metric)
                public_metric = _public_metric(metric)
                rows.append({
                    "domain": "Microstrip Resonator",
                    "task_id": (
                        f"{experiment_layer}_{scenario}_cal{_calibration_slug(calibration)}"
                        f"_{public_metric}_vs_{comparison}"
                    ),
                    "evidence_axis": "device_generalization_effect",
                    "effect_layer": f"{experiment_layer}_vs_{comparison}",
                    "metric": public_metric,
                    "cfnn_effect": float(effect),
                    "cfnn_metric": cfnn_metric,
                    "best_baseline_model": comparison,
                    "best_baseline_metric": baseline_metric,
                    "cross_curve_joint_coverage": None,
                    "cross_curve_peak_coverage": None,
                    "dangerous_failure_rate": None,
                    "same_to_cross_scale_ratio": None,
                    "phase_total_turns": None,
                    "max_curvature": None,
                    "linewidth_or_peak_width_fraction": None,
                    "relaxation_peak_count": None,
                    "known_pole_count": None,
                    "distributed_relaxation": False,
                    "frequency_points_per_decade": None,
                    "noise": None,
                    "fit_residual": None,
                    "scenario": scenario,
                    "calibration_frequency_count": calibration,
                    "event_identifiable_rate": _finite(cfnn.get("event_identifiable_rate")),
                    "record_count": int(cfnn.get("record_count") or 0),
                    "baseline_record_count": int(baseline.get("record_count") or 0),
                    "boundary_score": _effect_boundary(effect),
                })
    return rows


def _missed_band_coverage_lookup(missed_band_summary: dict) -> dict[tuple, dict]:
    lookup = {}
    for row in missed_band_summary.get("q_effect_summary_rows", []):
        if row.get("calibration_sampling_mode") != "missed_band":
            continue
        key = (
            row.get("calibration_frequency_count"),
            row.get("comparison_family"),
            row.get("metric"),
        )
        lookup[key] = row
    return lookup


def microstrip_b_missed_band_negative_control_rows(
    missed_band_summary: dict,
    *,
    comparison_families: set[str] | None = None,
    metrics: set[str] | None = None,
) -> list[dict]:
    """Convert B2 missed-band calibration seed expansion rows into Package E evidence."""
    comparison_families = set(comparison_families or {"MLP", "Local nested CF control"})
    metrics = set(metrics or {
        "quality_factor_relative_error",
        "complex_nrmse",
        "peak_window_complex_nrmse",
        "resonance_frequency_abs_error_hz",
        "local_phase_transition_abs_error_hz",
    })
    coverage_by_key = _missed_band_coverage_lookup(missed_band_summary)
    rows = []
    for row in missed_band_summary.get("all_metric_effect_summary_rows", []):
        if row.get("calibration_sampling_mode") != "missed_band":
            continue
        comparison = str(row.get("comparison_family"))
        metric = str(row.get("metric"))
        if comparison not in comparison_families or metric not in metrics:
            continue
        effect = _finite(row.get("median_cfnn_effect"))
        if effect is None:
            continue
        calibration = row.get("calibration_frequency_count")
        coverage = coverage_by_key.get((calibration, comparison, metric), {})
        rows.append({
            "domain": "Microstrip Resonator",
            "task_id": (
                f"b2_missed_band_cal{_calibration_slug(calibration)}"
                f"_{metric}_vs_{comparison}"
            ),
            "evidence_axis": "device_calibration_negative_control_effect",
            "effect_layer": f"b2_missed_band_vs_{comparison}",
            "metric": metric,
            "cfnn_effect": effect,
            "cfnn_metric": None,
            "best_baseline_model": comparison,
            "best_baseline_metric": None,
            "cross_curve_joint_coverage": None,
            "cross_curve_peak_coverage": None,
            "dangerous_failure_rate": None,
            "same_to_cross_scale_ratio": None,
            "phase_total_turns": None,
            "max_curvature": None,
            "linewidth_or_peak_width_fraction": None,
            "relaxation_peak_count": None,
            "known_pole_count": None,
            "distributed_relaxation": False,
            "frequency_points_per_decade": None,
            "noise": None,
            "fit_residual": None,
            "scenario": row.get("scenario"),
            "calibration_frequency_count": calibration,
            "calibration_sampling_mode": "missed_band",
            "pair_count": int(row.get("pair_count") or 0),
            "cfnn_win_rate": _finite(row.get("cfnn_win_rate")),
            "nearest_calibration_distance_linewidth_ratio": _finite(
                coverage.get("median_nearest_calibration_distance_linewidth_ratio")
            ),
            "points_within_one_linewidth": _finite(coverage.get("median_points_within_one_linewidth")),
            "points_within_half_linewidth": _finite(coverage.get("median_points_within_half_linewidth")),
            "boundary_score": _effect_boundary(effect),
        })
    return rows


def microstrip_b2_f0_descriptor_guardrail_rows(
    descriptor_summary: dict,
    *,
    comparison_families: set[str] | None = None,
) -> list[dict]:
    """Convert Microstrip B2 f0 descriptor audit rows into Package E guardrail evidence."""
    comparison_families = set(comparison_families or {"MLP", "Local nested CF control"})
    rows = []
    for row in descriptor_summary.get("descriptor_summary_rows", []):
        comparison = str(row.get("comparison_family"))
        if comparison not in comparison_families:
            continue
        rate = _finite(row.get("q_positive_f0_negative_rate"))
        if rate is None:
            continue
        calibration = row.get("calibration_frequency_count")
        boundary_stratum = str(row.get("boundary_proximity_stratum"))
        coverage_stratum = str(row.get("calibration_coverage_stratum"))
        effect = -float(rate)
        rows.append({
            "domain": "Microstrip Resonator",
            "task_id": (
                f"b2_f0_descriptor_{boundary_stratum}_missed_band"
                f"_cal{_calibration_slug(calibration)}_vs_{comparison}"
            ),
            "evidence_axis": "device_f0_guardrail_descriptor",
            "effect_layer": f"b2_f0_descriptor_vs_{comparison}",
            "metric": "q_positive_f0_negative_rate",
            "cfnn_effect": effect,
            "cfnn_metric": rate,
            "best_baseline_model": comparison,
            "best_baseline_metric": None,
            "cross_curve_joint_coverage": None,
            "cross_curve_peak_coverage": None,
            "dangerous_failure_rate": None,
            "same_to_cross_scale_ratio": None,
            "phase_total_turns": None,
            "max_curvature": None,
            "linewidth_or_peak_width_fraction": None,
            "relaxation_peak_count": None,
            "known_pole_count": None,
            "distributed_relaxation": False,
            "frequency_points_per_decade": None,
            "noise": None,
            "fit_residual": None,
            "scenario": "low_calibration_target_25c_50rh",
            "calibration_frequency_count": calibration,
            "calibration_sampling_mode": "missed_band",
            "notch_geometry_stratum": row.get("notch_geometry_stratum"),
            "boundary_proximity_stratum": boundary_stratum,
            "calibration_coverage_stratum": coverage_stratum,
            "pair_count": int(row.get("pair_count") or 0),
            "f0_negative_rate": _finite(row.get("f0_negative_rate")),
            "q_positive_f0_negative_rate": rate,
            "median_f0_effect": _finite(row.get("median_f0_effect")),
            "median_quality_factor_effect": _finite(row.get("median_quality_factor_effect")),
            "boundary_score": float(rate),
        })
    return rows


def aluminium_c_active_modal_boundary_rows(aluminium_summary: dict) -> list[dict]:
    """Record Package C active modal-analysis boundary rows without turning failures into CFNN wins."""
    rows = []
    for row in aluminium_summary.get("summary_rows", []):
        if row.get("family") != "CFNN":
            continue
        matched = _finite(row.get("median_matched_mode_count"))
        truth = _finite(row.get("median_truth_mode_count"))
        if matched is None or truth in (None, 0.0):
            continue
        matched_fraction = float(matched) / float(truth)
        rows.append({
            "domain": "Aluminium FRF",
            "task_id": f"c_active_modal_{row.get('sampling_strategy')}_budget{int(row.get('budget'))}_matched_mode_fraction",
            "evidence_axis": "active_modal_boundary",
            "effect_layer": "active_modal_sampling",
            "metric": "matched_mode_fraction",
            "cfnn_effect": None,
            "cfnn_metric": matched,
            "best_baseline_model": None,
            "best_baseline_metric": None,
            "cross_curve_joint_coverage": None,
            "cross_curve_peak_coverage": None,
            "dangerous_failure_rate": None,
            "same_to_cross_scale_ratio": None,
            "phase_total_turns": None,
            "max_curvature": None,
            "linewidth_or_peak_width_fraction": None,
            "relaxation_peak_count": None,
            "known_pole_count": int(truth),
            "distributed_relaxation": False,
            "frequency_points_per_decade": None,
            "noise": None,
            "fit_residual": None,
            "sampling_strategy": row.get("sampling_strategy"),
            "observation_budget": int(row.get("budget")),
            "matched_mode_fraction": matched_fraction,
            "median_complex_nrmse": _finite(row.get("median_complex_nrmse")),
            "median_peak_window_complex_nrmse": _finite(row.get("median_peak_window_complex_nrmse")),
            "record_count": int(row.get("record_count") or 0),
            "boundary_score": float(max(0.0, 1.0 - matched_fraction)),
        })

    for row in aluminium_summary.get("budget_threshold_rows", []):
        if row.get("family") != "CFNN":
            continue
        minimum_budget = row.get("minimum_budget_meeting_modal_thresholds")
        rows.append({
            "domain": "Aluminium FRF",
            "task_id": f"c_active_modal_{row.get('sampling_strategy')}_threshold_budget",
            "evidence_axis": "active_modal_boundary",
            "effect_layer": "active_modal_sampling",
            "metric": "minimum_budget_meeting_modal_thresholds",
            "cfnn_effect": None,
            "cfnn_metric": _finite(minimum_budget),
            "best_baseline_model": None,
            "best_baseline_metric": None,
            "cross_curve_joint_coverage": None,
            "cross_curve_peak_coverage": None,
            "dangerous_failure_rate": None,
            "same_to_cross_scale_ratio": None,
            "phase_total_turns": None,
            "max_curvature": None,
            "linewidth_or_peak_width_fraction": None,
            "relaxation_peak_count": None,
            "known_pole_count": None,
            "distributed_relaxation": False,
            "frequency_points_per_decade": None,
            "noise": None,
            "fit_residual": None,
            "sampling_strategy": row.get("sampling_strategy"),
            "observation_budget": _finite(minimum_budget),
            "frequency_threshold_hz": _finite(row.get("frequency_threshold_hz")),
            "width_relative_threshold": _finite(row.get("width_relative_threshold")),
            "threshold_met": minimum_budget is not None,
            "boundary_score": 0.0 if minimum_budget is not None else 1.0,
        })
    return rows


def _decision_by_id(summary: dict) -> dict[str, dict]:
    return {
        str(row.get("decision_id")): dict(row)
        for row in summary.get("decision_rows", []) or []
    }


def summarize_aluminium_active_modal_claim_gate(package_e_summary: dict) -> dict:
    """Separate active measurement value from CFNN modal-recovery advantage."""
    audit = package_e_summary.get("aluminium_active_measurement_value_audit", {}) or {}
    decisions = _decision_by_id(audit)
    measurement = decisions.get("observed_interpolation_measurement_signal", {})
    same_schedule = decisions.get("same_schedule_cfnn_boundary", {})
    candidate = decisions.get("candidate_discovery_boundary", {})

    measurement_supported = measurement.get("verdict") == "supports_measurement_efficiency_signal"
    cfnn_blocked = same_schedule.get("verdict") == "blocks_cfnn_advantage_claim"
    candidate_limited = candidate.get("verdict") == "candidate_discovery_limits_full_modal_recovery"
    if measurement_supported and cfnn_blocked:
        claim_status = "measurement_strategy_value_without_cfnn_advantage"
    elif measurement_supported:
        claim_status = "candidate_measurement_strategy_value"
    else:
        claim_status = "not_supported_or_incomplete"

    return {
        "analysis_type": "aluminium_active_modal_claim_gate",
        "claim_status": claim_status,
        "measurement_strategy_pass": bool(measurement_supported),
        "cfnn_advantage_pass": bool(measurement_supported and not cfnn_blocked),
        "measurement_strategy_minimum_budget": measurement.get("minimum_budget"),
        "measurement_strategy_passed_modes": _finite(measurement.get("median_passed_modes")),
        "same_schedule_cfnn_minimum_budget": same_schedule.get("minimum_budget"),
        "same_schedule_cfnn_passed_modes": _finite(same_schedule.get("median_passed_modes")),
        "same_schedule_cfnn_status": same_schedule.get("verdict"),
        "candidate_discovery_status": candidate.get("verdict"),
        "candidate_discovery_minimum_budget": candidate.get("minimum_budget"),
        "candidate_discovery_passed_modes": _finite(candidate.get("median_passed_modes")),
        "candidate_discovery_limited": bool(candidate_limited),
        "decision_rows": list(decisions.values()),
    }


def build_unified_applicability_rows(
    fano_summary: dict,
    eis_summary: dict,
    a2_summary: dict | None = None,
    a3_summaries: Iterable[tuple[str, dict]] | None = None,
    microstrip_summary: dict | None = None,
    microstrip_missed_band_summary: dict | None = None,
    microstrip_f0_descriptor_summary: dict | None = None,
    aluminium_summary: dict | None = None,
) -> list[dict]:
    rows = fano_applicability_rows(fano_summary) + eis_applicability_rows(eis_summary)
    if a2_summary is not None:
        rows.extend(fano_a2_accuracy_effect_rows(a2_summary))
    for source_label, a3_summary in a3_summaries or []:
        rows.extend(fano_a3_robustness_effect_rows(a3_summary, source_label=source_label))
    if microstrip_summary is not None:
        rows.extend(microstrip_b_device_effect_rows(microstrip_summary))
    if microstrip_missed_band_summary is not None:
        rows.extend(microstrip_b_missed_band_negative_control_rows(microstrip_missed_band_summary))
    if microstrip_f0_descriptor_summary is not None:
        rows.extend(microstrip_b2_f0_descriptor_guardrail_rows(microstrip_f0_descriptor_summary))
    if aluminium_summary is not None:
        rows.extend(aluminium_c_active_modal_boundary_rows(aluminium_summary))
    return rows


def summarize_unified_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    by_domain = {}
    for domain in sorted({row["domain"] for row in rows}):
        items = [row for row in rows if row["domain"] == domain]
        effects = [row["cfnn_effect"] for row in items if row.get("cfnn_effect") is not None]
        boundary_scores = [row["boundary_score"] for row in items if row.get("boundary_score") is not None]
        by_domain[domain] = {
            "row_count": len(items),
            "median_cfnn_effect": float(np.median(effects)) if effects else None,
            "positive_effect_count": int(sum(1 for value in effects if float(value) > 0.0)),
            "negative_effect_count": int(sum(1 for value in effects if float(value) < 0.0)),
            "median_boundary_score": float(np.median(boundary_scores)) if boundary_scores else None,
        }
    return {
        "row_count": len(rows),
        "domain_summaries": by_domain,
    }


def _median_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _quantile_or_none(values: Iterable[float | None], quantile: float) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.quantile(np.asarray(finite, dtype=np.float64), float(quantile))) if finite else None


def _descriptor_stratum(
    row: dict,
    *,
    fano_phase_turn_threshold: float | None,
    fano_curvature_threshold: float | None,
) -> str:
    domain = str(row.get("domain"))
    if domain == "Microwave Fano" and row.get("evidence_axis") == "reliability_boundary":
        phase = _finite(row.get("phase_total_turns"))
        curvature = _finite(row.get("max_curvature"))
        low_phase = (
            phase is not None
            and fano_phase_turn_threshold is not None
            and phase <= float(fano_phase_turn_threshold)
        )
        high_curvature = (
            curvature is not None
            and fano_curvature_threshold is not None
            and curvature >= float(fano_curvature_threshold)
        )
        if low_phase and high_curvature:
            return "weak_phase_high_curvature"
        if low_phase:
            return "weak_phase_only"
        if high_curvature:
            return "high_curvature_only"
        return "ordinary_resonant"
    if domain == "EIS":
        if bool(row.get("distributed_relaxation")):
            return "distributed_relaxation"
        pole_count = row.get("known_pole_count")
        peak_count = row.get("relaxation_peak_count")
        if pole_count is not None and int(pole_count) >= 4:
            return "multi_pole_relaxation"
        if peak_count is not None and int(peak_count) >= 3:
            return "multi_peak_relaxation"
        return "low_order_relaxation"
    if domain == "Aluminium FRF":
        return "modal_active_boundary"
    if domain == "Microstrip Resonator":
        if row.get("evidence_axis") == "device_f0_guardrail_descriptor":
            boundary = str(row.get("boundary_proximity_stratum") or "unknown_boundary")
            coverage = str(row.get("calibration_coverage_stratum") or "unknown_calibration")
            if coverage == "resonance_band_missed":
                coverage = "missed_band"
            return f"{boundary}_{coverage}"
        if row.get("evidence_axis") == "device_calibration_negative_control_effect":
            return "missed_band_calibration"
        return "device_condition_response"
    return "unstratified"


def summarize_descriptor_strata(rows: Iterable[dict]) -> dict:
    """Summarize effects and boundary scores by interpretable spectral strata."""
    rows = [dict(row) for row in rows]
    fano_reliability = [
        row for row in rows
        if row.get("domain") == "Microwave Fano"
        and row.get("evidence_axis") == "reliability_boundary"
    ]
    phase_threshold = _quantile_or_none(
        [row.get("phase_total_turns") for row in fano_reliability],
        0.20,
    )
    curvature_threshold = _quantile_or_none(
        [row.get("max_curvature") for row in fano_reliability],
        0.80,
    )
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        stratum = _descriptor_stratum(
            row,
            fano_phase_turn_threshold=phase_threshold,
            fano_curvature_threshold=curvature_threshold,
        )
        key = (
            str(row.get("domain")),
            str(row.get("evidence_axis")),
            str(row.get("effect_layer")),
            stratum,
        )
        groups.setdefault(key, []).append(row)

    stratum_rows = []
    for key, items in sorted(groups.items(), key=lambda item: item[0]):
        domain, evidence_axis, effect_layer, stratum = key
        effects = [_finite(row.get("cfnn_effect")) for row in items]
        effects = [value for value in effects if value is not None]
        scores = [_finite(row.get("boundary_score")) for row in items]
        scores = [value for value in scores if value is not None]
        stratum_rows.append({
            "domain": domain,
            "evidence_axis": evidence_axis,
            "effect_layer": effect_layer,
            "descriptor_stratum": stratum,
            "row_count": len(items),
            "median_cfnn_effect": float(np.median(effects)) if effects else None,
            "positive_effect_count": int(sum(1 for value in effects if value > 0.0)),
            "negative_effect_count": int(sum(1 for value in effects if value < 0.0)),
            "median_boundary_score": float(np.median(scores)) if scores else None,
            "max_boundary_score": float(np.max(scores)) if scores else None,
            "example_task_id": str(sorted(items, key=lambda row: -(row.get("boundary_score") or -1.0))[0].get("task_id")),
        })
    return {
        "stratum_row_count": len(stratum_rows),
        "fano_phase_turn_threshold": phase_threshold,
        "fano_curvature_threshold": curvature_threshold,
        "stratum_rows": stratum_rows,
    }


def _microstrip_decision_row(
    *,
    decision_id: str,
    decision_question: str,
    rows: list[dict],
    supportive_when_positive: bool,
    guardrail_when_boundary: bool,
    claim_scope: str,
) -> dict:
    effects = [_finite(row.get("cfnn_effect")) for row in rows]
    effects = [value for value in effects if value is not None]
    scores = [_finite(row.get("boundary_score")) for row in rows]
    scores = [value for value in scores if value is not None]
    median_effect = float(np.median(effects)) if effects else None
    median_score = float(np.median(scores)) if scores else None
    positive = int(sum(1 for value in effects if value > 0.0))
    negative = int(sum(1 for value in effects if value < 0.0))
    boundary_present = bool(median_score is not None and median_score > 0.0)
    if not rows:
        verdict = "insufficient_evidence"
    elif guardrail_when_boundary and (boundary_present or negative > positive):
        verdict = "guardrail_required"
    elif supportive_when_positive and median_effect is not None and median_effect > 0.0 and positive >= negative:
        verdict = "supporting_local_event_signal"
    else:
        verdict = "no_current_guardrail_signal"
    return {
        "decision_id": decision_id,
        "decision_question": decision_question,
        "claim_scope": claim_scope,
        "row_count": len(rows),
        "median_cfnn_effect": median_effect,
        "positive_effect_count": positive,
        "negative_effect_count": negative,
        "median_boundary_score": median_score,
        "max_boundary_score": float(np.max(scores)) if scores else None,
        "verdict": verdict,
    }


def summarize_microstrip_guardrail_decisions(rows: Iterable[dict]) -> dict:
    """Summarize Microstrip B evidence as claim-support and guardrail decisions."""
    microstrip_rows = [
        dict(row) for row in rows
        if row.get("domain") == "Microstrip Resonator"
    ]
    missed_band_rows = [
        row for row in microstrip_rows
        if row.get("evidence_axis") == "device_calibration_negative_control_effect"
    ]
    f0_descriptor_rows = [
        row for row in microstrip_rows
        if row.get("evidence_axis") == "device_f0_guardrail_descriptor"
    ]
    decision_rows = [
        _microstrip_decision_row(
            decision_id="missed_band_q_signal",
            decision_question="Does CFNN preserve Q/linewidth gains when calibration avoids the resonance band?",
            rows=[
                row for row in missed_band_rows
                if row.get("metric") == "quality_factor_relative_error"
            ],
            supportive_when_positive=True,
            guardrail_when_boundary=False,
            claim_scope="localized_Q_linewidth_event_transfer",
        ),
        _microstrip_decision_row(
            decision_id="missed_band_phase_guardrail",
            decision_question="Do phase-event errors block a complete device-digital-twin claim?",
            rows=[
                row for row in missed_band_rows
                if row.get("metric") == "local_phase_transition_abs_error_hz"
            ],
            supportive_when_positive=False,
            guardrail_when_boundary=True,
            claim_scope="phase_event_recovery_guardrail",
        ),
        _microstrip_decision_row(
            decision_id="edge_proximal_f0_guardrail",
            decision_question="Do edge-proximal notches show Q-positive/f0-negative co-occurrence?",
            rows=[
                row for row in f0_descriptor_rows
                if row.get("boundary_proximity_stratum") == "edge_proximal"
            ],
            supportive_when_positive=False,
            guardrail_when_boundary=True,
            claim_scope="resonance_frequency_guardrail",
        ),
        _microstrip_decision_row(
            decision_id="edge_intermediate_f0_guardrail",
            decision_question="Do edge-intermediate notches show the same f0 guardrail signal?",
            rows=[
                row for row in f0_descriptor_rows
                if row.get("boundary_proximity_stratum") == "edge_intermediate"
            ],
            supportive_when_positive=False,
            guardrail_when_boundary=True,
            claim_scope="resonance_frequency_guardrail",
        ),
    ]
    q_verdict = next(row["verdict"] for row in decision_rows if row["decision_id"] == "missed_band_q_signal")
    guardrail_verdicts = {
        row["verdict"] for row in decision_rows
        if row["decision_id"] != "missed_band_q_signal"
    }
    if q_verdict == "supporting_local_event_signal" and "guardrail_required" in guardrail_verdicts:
        claim_level = "localized_event_transfer_with_guardrails"
    elif q_verdict == "supporting_local_event_signal":
        claim_level = "candidate_device_event_transfer"
    else:
        claim_level = "mixed_or_boundary_evidence"
    return {
        "analysis_type": "microstrip_guardrail_decision_table",
        "claim_level": claim_level,
        "decision_rows": decision_rows,
    }


def _microstrip_stratum_summary(rows: list[dict], boundary: str) -> dict:
    f0_rows = [
        row for row in rows
        if row.get("domain") == "Microstrip Resonator"
        and row.get("evidence_axis") == "device_f0_guardrail_descriptor"
        and row.get("boundary_proximity_stratum") == boundary
        and row.get("calibration_coverage_stratum") in {"resonance_band_missed", "missed_band"}
    ]
    q_rows = [
        row for row in rows
        if row.get("domain") == "Microstrip Resonator"
        and row.get("evidence_axis") == "device_calibration_negative_control_effect"
        and row.get("metric") == "quality_factor_relative_error"
        and row.get("calibration_sampling_mode") == "missed_band"
    ]
    phase_rows = [
        row for row in rows
        if row.get("domain") == "Microstrip Resonator"
        and row.get("evidence_axis") == "device_calibration_negative_control_effect"
        and row.get("metric") == "local_phase_transition_abs_error_hz"
        and row.get("calibration_sampling_mode") == "missed_band"
    ]
    calibrations = sorted({
        int(row["calibration_frequency_count"])
        for row in f0_rows + q_rows
        if row.get("calibration_frequency_count") is not None
    })
    q_f0_rates = [_finite(row.get("q_positive_f0_negative_rate")) for row in f0_rows]
    q_f0_rates = [value for value in q_f0_rates if value is not None]
    f0_effects = [_finite(row.get("median_f0_effect")) for row in f0_rows]
    f0_effects = [value for value in f0_effects if value is not None]
    q_effects = [_finite(row.get("median_quality_factor_effect")) for row in f0_rows]
    q_effects = [value for value in q_effects if value is not None]
    q_layer_effects = [_finite(row.get("cfnn_effect")) for row in q_rows]
    q_layer_effects = [value for value in q_layer_effects if value is not None]
    phase_effects = [_finite(row.get("cfnn_effect")) for row in phase_rows]
    phase_effects = [value for value in phase_effects if value is not None]
    boundary_scores = [_finite(row.get("boundary_score")) for row in f0_rows]
    boundary_scores = [value for value in boundary_scores if value is not None]
    role = "primary_guardrail_stratum" if boundary == "edge_proximal" else "contrast_stratum"
    median_boundary = float(np.median(boundary_scores)) if boundary_scores else None
    if boundary == "edge_proximal" and median_boundary is not None and median_boundary > 0.0:
        provisional = "guardrail_confirmatory_target"
    elif boundary == "edge_intermediate" and (median_boundary is None or median_boundary == 0.0):
        provisional = "contrast_no_current_f0_guardrail"
    else:
        provisional = "needs_confirmatory_recheck"
    return {
        "stratum_id": f"{boundary}_missed_band",
        "stratum_role": role,
        "include_when": {
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "calibration_sampling_mode": "missed_band",
            "calibration_coverage_stratum": "resonance_band_missed",
            "boundary_proximity_stratum": boundary,
        },
        "calibration_frequency_counts": calibrations,
        "primary_positive_metric": "quality_factor_relative_error",
        "guardrail_metrics": [
            "resonance_frequency_abs_error_hz",
            "local_phase_transition_abs_error_hz",
            "complex_nrmse",
        ],
        "success_rule": (
            "CFNN can support only a localized Q/linewidth transfer claim when Q effect is positive "
            "and f0, local phase, and global complex NRMSE guardrails are non-negative in paired curve-level tests."
        ),
        "failure_rule": (
            "Any recurrent Q-positive/f0-negative or phase-negative pattern keeps the stratum as a guardrail boundary "
            "and blocks a complete device-digital-twin claim."
        ),
        "pilot_row_count": len(f0_rows),
        "pilot_pair_count": int(sum(int(row.get("pair_count") or 0) for row in f0_rows)),
        "pilot_q_positive_f0_negative_rate": float(np.median(q_f0_rates)) if q_f0_rates else None,
        "pilot_median_quality_factor_effect": float(np.median(q_effects)) if q_effects else (
            float(np.median(q_layer_effects)) if q_layer_effects else None
        ),
        "pilot_median_f0_effect": float(np.median(f0_effects)) if f0_effects else None,
        "pilot_global_missed_band_q_effect": float(np.median(q_layer_effects)) if q_layer_effects else None,
        "pilot_global_missed_band_phase_effect": float(np.median(phase_effects)) if phase_effects else None,
        "pilot_median_boundary_score": median_boundary,
        "provisional_status": provisional,
    }


def summarize_microstrip_confirmatory_guardrail_strata(rows: Iterable[dict]) -> dict:
    """Build a pre-registered B2 guardrail-stratum spec from current Package E rows."""
    rows = [dict(row) for row in rows]
    return {
        "analysis_type": "microstrip_b2_confirmatory_guardrail_strata",
        "primary_claim_gate": "q_gain_requires_f0_and_phase_guardrails",
        "stratum_rows": [
            _microstrip_stratum_summary(rows, "edge_proximal"),
            _microstrip_stratum_summary(rows, "edge_intermediate"),
        ],
    }


def summarize_microstrip_digital_twin_claim_gate(package_e_summary: dict) -> dict:
    """Evaluate whether Microstrip B supports a full conditional digital-twin claim."""
    rows = [
        dict(row) for row in package_e_summary.get("unified_rows", [])
        if row.get("domain") == "Microstrip Resonator"
        and row.get("evidence_axis") == "device_generalization_effect"
    ]
    required_metrics = [
        "complex_nrmse",
        "peak_window_complex_nrmse",
        "resonance_frequency_abs_error_hz",
        "quality_factor_relative_error",
        "phase_transition_abs_error_hz",
    ]
    required_layers = [
        "b1_vs_MLP",
        "b1_vs_Local nested CF control",
        "b2_vs_MLP",
        "b2_vs_Local nested CF control",
    ]
    gate_rows = []
    failed_layers = []
    for layer in required_layers:
        layer_rows = [row for row in rows if row.get("effect_layer") == layer]
        metric_effects = {}
        failed_metrics = []
        missing_metrics = []
        for metric in required_metrics:
            values = [
                _finite(row.get("cfnn_effect"))
                for row in layer_rows
                if row.get("metric") == metric
            ]
            values = [value for value in values if value is not None]
            if not values:
                missing_metrics.append(metric)
                continue
            effect = float(np.median(values))
            metric_effects[metric] = effect
            if effect < 0.0:
                failed_metrics.append(metric)
        layer_pass = bool(not failed_metrics and not missing_metrics)
        if not layer_pass:
            failed_layers.append(layer)
        gate_rows.append({
            "effect_layer": layer,
            "required_metrics": required_metrics,
            "metric_effects": metric_effects,
            "failed_metrics": failed_metrics,
            "missing_metrics": missing_metrics,
            "layer_status": "passes_full_response_event_gate" if layer_pass else "fails_full_response_event_gate",
        })

    guardrail_table = package_e_summary.get("microstrip_guardrail_decision_table", {}) or {}
    guardrail_rows = guardrail_table.get("decision_rows", []) or []
    localized_q_signal_status = None
    blocking_decisions = []
    for row in guardrail_rows:
        decision_id = str(row.get("decision_id"))
        verdict = str(row.get("verdict"))
        if decision_id == "missed_band_q_signal":
            localized_q_signal_status = verdict
        elif verdict == "guardrail_required":
            blocking_decisions.append(decision_id)

    confirmatory = package_e_summary.get("microstrip_confirmatory_guardrail_strata", {}) or {}
    blocking_strata = [
        str(row.get("stratum_id"))
        for row in confirmatory.get("stratum_rows", []) or []
        if row.get("provisional_status") == "guardrail_confirmatory_target"
    ]
    full_pass = bool(not failed_layers and not blocking_decisions and not blocking_strata)
    if full_pass:
        claim_status = "supports_full_digital_twin_claim"
    elif localized_q_signal_status == "supporting_local_event_signal":
        claim_status = "blocked_by_global_event_guardrails"
    else:
        claim_status = "not_supported_or_incomplete"
    return {
        "analysis_type": "microstrip_digital_twin_claim_gate",
        "claim_status": claim_status,
        "full_digital_twin_pass": full_pass,
        "localized_q_signal_status": localized_q_signal_status,
        "required_effect_layers": required_layers,
        "required_metrics": required_metrics,
        "gate_rows": gate_rows,
        "failed_effect_layers": failed_layers,
        "blocking_guardrail_decisions": blocking_decisions,
        "blocking_confirmatory_strata": blocking_strata,
    }


def _median_effect_for(rows: list[dict], *, domain: str, layers: set[str], metrics: set[str] | None = None) -> float | None:
    values = [
        _finite(row.get("cfnn_effect"))
        for row in rows
        if row.get("domain") == domain
        and row.get("effect_layer") in layers
        and (metrics is None or row.get("metric") in metrics)
    ]
    values = [value for value in values if value is not None]
    return float(np.median(values)) if values else None


def _max_boundary_for(rows: list[dict], *, domain: str, axis: str | None = None) -> float | None:
    values = [
        _finite(row.get("boundary_score"))
        for row in rows
        if row.get("domain") == domain
        and (axis is None or row.get("evidence_axis") == axis)
    ]
    values = [value for value in values if value is not None]
    return float(np.max(values)) if values else None


def _int_list(values) -> list[int]:
    return [int(value) for value in values or []]


def summarize_ai4science_claim_ledger(package_e_summary: dict) -> dict:
    """Turn Package E evidence and audits into a report-facing CFNN claim ledger."""
    rows = [dict(row) for row in package_e_summary.get("unified_rows", [])]
    fano_a2_mlp = _median_effect_for(
        rows,
        domain="Microwave Fano",
        layers={"a2_vs_MLP"},
        metrics={"f0_abs_error_hz", "quality_factor_relative_error", "q_abs_error", "peak_window_complex_nrmse"},
    )
    fano_a2_local = _median_effect_for(
        rows,
        domain="Microwave Fano",
        layers={"a2_vs_Local nested CF control"},
        metrics={"f0_abs_error_hz", "quality_factor_relative_error", "q_abs_error", "peak_window_complex_nrmse"},
    )
    fano_boundary = _max_boundary_for(rows, domain="Microwave Fano", axis="reliability_boundary")
    fano_supported = (
        fano_a2_mlp is not None
        and fano_a2_local is not None
        and fano_a2_mlp > 0.0
        and fano_a2_local > 0.0
    )
    fano_prereg = package_e_summary.get("fano_a2_preregistered_success_audit")
    fano_prereg_budgets = _int_list((fano_prereg or {}).get("budgets")) if fano_prereg else []
    fano_prereg_passed = _int_list((fano_prereg or {}).get("passed_budgets")) if fano_prereg else []
    fano_prereg_failed = [
        budget for budget in fano_prereg_budgets
        if budget not in set(fano_prereg_passed)
    ]
    if fano_prereg:
        if fano_supported and fano_prereg_passed:
            fano_claim_status = "supported_with_preregistered_budget_guardrails"
            fano_reporting_rule = (
                "Can be written as the strongest current CFNN advantage only at the preregistered passing "
                f"budgets ({'/'.join(str(value) for value in fano_prereg_passed)} observations); "
                "failed budgets must be reported as sparse-budget boundaries and reliability claims must keep Fano D/E guardrails."
            )
        else:
            fano_claim_status = "blocked_by_preregistered_budget_gate"
            fano_reporting_rule = (
                "Do not write a CFNN scientific inversion advantage unless the preregistered A2 response+parameter budget gate passes."
            )
    else:
        fano_claim_status = "supported_with_guardrails" if fano_supported else "not_supported_or_incomplete"
        fano_reporting_rule = (
            "Can be written as the strongest current CFNN advantage, but reliability claims must keep Fano D/E guardrails."
        )
    fano_a3 = package_e_summary.get("fano_a3_preregistered_robustness_audit") or {}
    fano_a3_status = fano_a3.get("overall_status")
    fano_a3_passed = fano_a3.get("passed_condition_count")
    fano_a3_partial = fano_a3.get("mlp_only_or_partial_condition_count")
    if fano_a3_status:
        fano_reporting_rule = (
            f"{fano_reporting_rule} A3 robustness must be reported as a guardrail: "
            f"{fano_a3_passed} conditions passed the MLP+Local gate and "
            f"{fano_a3_partial} conditions were MLP-only or partial."
        )
    fano_reliability = package_e_summary.get("fano_reliability_claim_gate") or {}
    fano_reliability_status = fano_reliability.get("claim_status")
    if fano_reliability_status:
        fano_reporting_rule = (
            f"{fano_reporting_rule} Reliability language is limited to ordinary Fano reliability "
            f"with high-risk escalation: {','.join(fano_reliability.get('highrisk_curve_ids') or [])}."
        )

    microstrip_level = str(package_e_summary.get("microstrip_guardrail_decision_table", {}).get("claim_level"))
    microstrip_gate = package_e_summary.get("microstrip_digital_twin_claim_gate") or {}
    microstrip_gate_status = microstrip_gate.get("claim_status")
    aluminium_level = str(package_e_summary.get("aluminium_active_measurement_value_audit", {}).get("claim_level"))
    aluminium_gate = package_e_summary.get("aluminium_active_modal_claim_gate") or {}
    aluminium_gate_status = aluminium_gate.get("claim_status")
    eis_audit = package_e_summary.get("eis_neural_positive_edge_audit", {}) or {}
    eis_eliminated = int((eis_audit.get("status_counts") or {}).get("physical_prior_eliminates_neural_edge", 0))
    eis_gate = package_e_summary.get("eis_applicability_claim_gate") or {}
    eis_gate_status = eis_gate.get("claim_status")
    eis_physical_negative_count = eis_gate.get("physical_prior_negative_count")
    eis_neural_positive_count = eis_gate.get("neural_positive_edge_count")
    eis_gate_eliminated = eis_gate.get("physical_prior_eliminated_edge_count")

    claim_rows = [
        {
            "claim_id": "fano_sparse_parameter_recovery",
            "domain": "Microwave Fano",
            "claim_type": "cfnn_advantage",
            "claim_status": fano_claim_status,
            "evidence_scope": "A2 sparse response and parameter recovery against MLP and Local nested CF control",
            "primary_positive_effect": fano_a2_local,
            "secondary_positive_effect": fano_a2_mlp,
            "guardrail_boundary_score": fano_boundary,
            "preregistered_passed_budgets": fano_prereg_passed,
            "preregistered_failed_budgets": fano_prereg_failed,
            "a3_robustness_guardrail_status": fano_a3_status,
            "a3_passed_condition_count": fano_a3_passed,
            "a3_mlp_only_or_partial_condition_count": fano_a3_partial,
            "reliability_claim_status": fano_reliability_status,
            "highrisk_curve_ids": fano_reliability.get("highrisk_curve_ids"),
            "post_policy_failure_rate": fano_reliability.get("post_policy_failure_rate"),
            "reporting_rule": fano_reporting_rule,
        },
        {
            "claim_id": "microstrip_device_digital_twin",
            "domain": "Microstrip Resonator",
            "claim_type": "boundary_or_localized_signal",
            "claim_status": (
                str(microstrip_gate_status)
                if microstrip_gate_status else
                (
                    "blocked_by_guardrails"
                    if microstrip_level == "localized_event_transfer_with_guardrails"
                    else "candidate_or_incomplete"
                )
            ),
            "evidence_scope": "B2 low-calibration Q/linewidth signal plus f0/phase guardrails",
            "claim_level": microstrip_level,
            "localized_q_signal_status": microstrip_gate.get("localized_q_signal_status"),
            "failed_effect_layers": microstrip_gate.get("failed_effect_layers"),
            "blocking_guardrail_decisions": microstrip_gate.get("blocking_guardrail_decisions"),
            "reporting_rule": (
                "Report only localized Q/linewidth transfer; do not write a full digital twin claim "
                "unless global response, f0, Q, and phase gates pass against MLP and Local nested CF control. "
                f"Current failed layers: {','.join(microstrip_gate.get('failed_effect_layers') or [])}; "
                f"blocking guardrails: {','.join(microstrip_gate.get('blocking_guardrail_decisions') or [])}."
            ),
        },
        {
            "claim_id": "aluminium_active_modal_cfnn_advantage",
            "domain": "Aluminium FRF",
            "claim_type": "measurement_strategy_boundary",
            "claim_status": (
                str(aluminium_gate_status)
                if aluminium_gate_status else
                (
                    "not_supported_measurement_strategy_only"
                    if aluminium_level == "measurement_strategy_signal_not_cfnn_advantage"
                    else "candidate_or_incomplete"
                )
            ),
            "evidence_scope": "C active half-power/modal recovery under observed-only interpolation and same-schedule CFNN",
            "claim_level": aluminium_level,
            "measurement_strategy_minimum_budget": aluminium_gate.get("measurement_strategy_minimum_budget"),
            "measurement_strategy_passed_modes": aluminium_gate.get("measurement_strategy_passed_modes"),
            "same_schedule_cfnn_passed_modes": aluminium_gate.get("same_schedule_cfnn_passed_modes"),
            "candidate_discovery_status": aluminium_gate.get("candidate_discovery_status"),
            "reporting_rule": (
                "Use as active measurement strategy value, not as CFNN advantage, unless same-schedule neural modal recovery passes. "
                f"Current measurement budget: {aluminium_gate.get('measurement_strategy_minimum_budget')}; "
                f"same-schedule CFNN passed modes: {aluminium_gate.get('same_schedule_cfnn_passed_modes')}."
            ),
        },
        {
            "claim_id": "eis_general_advantage",
            "domain": "EIS",
            "claim_type": "negative_applicability_boundary",
            "claim_status": (
                str(eis_gate_status)
                if eis_gate_status else
                (
                    "not_supported_physical_priors_dominate"
                    if eis_eliminated > 0 else
                    "candidate_or_incomplete"
                )
            ),
            "evidence_scope": "EIS neural-only positive cells checked against physical-prior baselines",
            "physical_prior_negative_count": eis_physical_negative_count,
            "neural_positive_edge_count": eis_neural_positive_count,
            "physical_prior_eliminated_edge_count": (
                eis_gate_eliminated
                if eis_gate_eliminated is not None else
                eis_eliminated
            ),
            "physical_prior_median_effect": eis_gate.get("physical_prior_median_effect"),
            "reporting_rule": (
                "Do not present neural-only edges as structural CFNN advantages when physical priors eliminate them; "
                "write EIS as an applicability boundary tied to circuit-prior baselines and spectral descriptors."
                if eis_gate_status else
                "Do not present neural-only positive cells as EIS structural advantages when physical priors dominate."
            ),
        },
    ]
    advantage_count = int(sum(1 for row in claim_rows if row["claim_type"] == "cfnn_advantage" and row["claim_status"].startswith("supported")))
    blocked_count = int(sum(1 for row in claim_rows if row["claim_type"] != "cfnn_advantage" or "blocked" in row["claim_status"] or row["claim_status"].startswith("not_supported")))
    return {
        "analysis_type": "ai4science_claim_ledger",
        "source_row_count": package_e_summary.get("unified_summary", {}).get("row_count", len(rows)),
        "claim_rows": claim_rows,
        "claim_summary": {
            "claim_count": len(claim_rows),
            "cfnn_advantage_claim_count": advantage_count,
            "blocked_or_boundary_claim_count": blocked_count,
        },
    }
