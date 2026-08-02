from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_fano_spectral_descriptors_report_shape_and_reference_fields():
    from ai4science_fano_boundary_e import fano_spectral_descriptors

    frequency = np.linspace(10.0, 20.0, 101)
    phase = np.linspace(0.0, np.pi, len(frequency))
    magnitude = 1.0 + 0.4 * np.exp(-((frequency - 15.0) / 0.5) ** 2)
    response = np.column_stack((magnitude * np.cos(phase), magnitude * np.sin(phase)))
    reference = {
        "curve_id": "curve:a",
        "status": "ok",
        "f0_hz": 15.0,
        "linewidth_hz": 1.0,
        "q": -2.0,
        "relative_magnitude_rmse": 0.01,
        "magnitude_peak_hz": 15.0,
        "magnitude_valley_hz": 12.0,
    }

    row = fano_spectral_descriptors("curve:a", frequency, response, reference=reference)

    assert row["curve_id"] == "curve:a"
    assert row["point_count"] == 101
    assert row["reference_status"] == "ok"
    assert row["linewidth_fraction"] == 0.1
    assert row["f0_position_fraction"] == 0.5
    assert row["abs_q"] == 2.0
    assert row["fano_relative_magnitude_rmse"] == 0.01
    assert 0.45 < row["phase_total_turns"] < 0.55
    assert row["magnitude_dynamic_range_ratio"] > 0.0
    assert row["max_normalized_magnitude_curvature"] > 0.0


def test_boundary_rows_rank_same_scale_cross_scale_mismatch():
    from ai4science_fano_boundary_e import (
        curve_boundary_summary,
        curve_family_budget_boundary_rows,
        rank_boundary_curves,
    )

    summary = {
        "confidence": 0.8,
        "ensemble_rows": [
            {
                "curve_id": "curve:hard",
                "family": "CFNN",
                "observation_budget": 64,
                "response_interval": {"joint_coverage": 0.1},
                "validation_calibrated_response_interval": {
                    "joint_coverage": 0.8,
                    "calibration_scale": 20.0,
                },
                "cross_curve_calibrated_response_interval": {
                    "status": "ok",
                    "joint_coverage": 0.2,
                    "peak_window_joint_coverage": 0.3,
                    "calibration_scale": 5.0,
                    "dangerous_point_failure_rate": 0.25,
                },
            },
            {
                "curve_id": "curve:easy",
                "family": "CFNN",
                "observation_budget": 64,
                "response_interval": {"joint_coverage": 0.1},
                "validation_calibrated_response_interval": {
                    "joint_coverage": 0.8,
                    "calibration_scale": 5.0,
                },
                "cross_curve_calibrated_response_interval": {
                    "status": "ok",
                    "joint_coverage": 0.82,
                    "peak_window_joint_coverage": 0.8,
                    "calibration_scale": 6.0,
                    "dangerous_point_failure_rate": 0.02,
                },
            },
        ],
    }
    descriptors = {
        "curve:hard": {"curve_id": "curve:hard", "linewidth_fraction": 0.02},
        "curve:easy": {"curve_id": "curve:easy", "linewidth_fraction": 0.10},
    }

    rows = curve_family_budget_boundary_rows(summary, descriptors)
    hard = next(row for row in rows if row["curve_id"] == "curve:hard")
    assert hard["same_to_cross_scale_ratio"] == 4.0
    assert np.isclose(hard["cross_curve_coverage_deficit"], 0.6)

    curve_rows = curve_boundary_summary(rows)
    ranked = rank_boundary_curves(curve_rows)
    assert ranked[0]["curve_id"] == "curve:hard"
    assert ranked[0]["median_same_to_cross_scale_ratio"] == 4.0


def test_split_conformal_boundary_rows_preserve_prediction_level_failure_layer():
    from ai4science_fano_boundary_e import (
        split_conformal_boundary_summary,
        split_conformal_family_budget_boundary_rows,
        rank_split_conformal_boundary_curves,
    )

    summary = {
        "confidence": 0.8,
        "ensemble_rows": [
            {
                "curve_id": "curve:hard",
                "family": "CFNN",
                "observation_budget": 64,
                "global_split_conformal_response_interval": {
                    "status": "ok",
                    "joint_coverage": 0.35,
                    "peak_window_joint_coverage": 0.55,
                    "calibration_scale": 6.8,
                    "dangerous_point_failure_rate": 0.22,
                },
                "curve_stratified_split_conformal_response_interval": {
                    "status": "ok",
                    "joint_coverage": 0.35,
                    "peak_window_joint_coverage": 0.55,
                    "calibration_scale": 6.8,
                    "dangerous_point_failure_rate": 0.22,
                    "calibration_source": "curve_stratified_split_conformal",
                },
            },
            {
                "curve_id": "curve:easy",
                "family": "CFNN",
                "observation_budget": 64,
                "global_split_conformal_response_interval": {
                    "status": "ok",
                    "joint_coverage": 0.98,
                    "peak_window_joint_coverage": 0.95,
                    "calibration_scale": 20.0,
                    "dangerous_point_failure_rate": 0.01,
                },
            },
        ],
    }
    descriptors = {
        "curve:hard": {
            "curve_id": "curve:hard",
            "phase_total_turns": 0.1,
            "max_normalized_magnitude_curvature": 100.0,
        },
        "curve:easy": {
            "curve_id": "curve:easy",
            "phase_total_turns": 1.0,
            "max_normalized_magnitude_curvature": 10.0,
        },
    }

    rows = split_conformal_family_budget_boundary_rows(summary, descriptors)

    hard_global = next(
        row for row in rows
        if row["curve_id"] == "curve:hard" and row["conformal_layer"] == "global_split_conformal"
    )
    assert hard_global["calibrated_joint_coverage"] == 0.35
    assert np.isclose(hard_global["calibrated_coverage_deficit"], 0.45)
    assert np.isclose(hard_global["boundary_score"], 0.67)
    assert hard_global["phase_total_turns"] == 0.1

    curve_rows = split_conformal_boundary_summary(rows)
    ranked = rank_split_conformal_boundary_curves(curve_rows)
    assert ranked[0]["curve_id"] == "curve:hard"
    assert ranked[0]["conformal_layer"] == "global_split_conformal"
    assert ranked[0]["median_calibrated_joint_coverage"] == 0.35


def test_descriptor_risk_rule_audit_flags_low_phase_high_curvature_boundary():
    from ai4science_fano_boundary_e import descriptor_risk_rule_audit

    rows = [
        {
            "curve_id": "curve:hard",
            "conformal_layer": "global_split_conformal",
            "phase_total_turns": 0.10,
            "max_normalized_magnitude_curvature": 100.0,
            "median_calibrated_joint_coverage": 0.35,
            "median_calibrated_dangerous_point_failure_rate": 0.22,
            "median_boundary_score": 0.67,
        },
        {
            "curve_id": "curve:easy1",
            "conformal_layer": "global_split_conformal",
            "phase_total_turns": 0.90,
            "max_normalized_magnitude_curvature": 30.0,
            "median_calibrated_joint_coverage": 0.98,
            "median_calibrated_dangerous_point_failure_rate": 0.01,
            "median_boundary_score": 0.01,
        },
        {
            "curve_id": "curve:easy2",
            "conformal_layer": "global_split_conformal",
            "phase_total_turns": 0.95,
            "max_normalized_magnitude_curvature": 20.0,
            "median_calibrated_joint_coverage": 0.99,
            "median_calibrated_dangerous_point_failure_rate": 0.00,
            "median_boundary_score": 0.00,
        },
    ]

    audit = descriptor_risk_rule_audit(
        rows,
        conformal_layer="global_split_conformal",
        confidence=0.80,
        phase_turn_threshold=0.25,
        curvature_threshold=75.0,
    )

    assert audit["status"] == "ok"
    assert audit["rule"]["phase_turn_threshold"] == 0.25
    assert audit["rule"]["curvature_threshold"] == 75.0
    assert audit["confusion"] == {
        "true_positive": 1,
        "false_positive": 0,
        "true_negative": 2,
        "false_negative": 0,
    }
    hard = next(row for row in audit["curve_rows"] if row["curve_id"] == "curve:hard")
    assert hard["risk_stratum"] == "weak_phase_high_curvature"
    assert hard["observed_conformal_failure"] is True
    assert hard["predicted_high_risk"] is True


def test_descriptor_guardrail_policy_audit_reports_acceptance_and_failure_capture():
    from ai4science_fano_boundary_e import descriptor_guardrail_policy_audit

    risk_audit = {
        "status": "ok",
        "conformal_layer": "global_split_conformal",
        "curve_rows": [
            {
                "curve_id": "curve:hard",
                "predicted_high_risk": True,
                "observed_conformal_failure": True,
                "median_calibrated_joint_coverage": 0.35,
                "median_calibrated_dangerous_point_failure_rate": 0.22,
                "risk_stratum": "weak_phase_high_curvature",
            },
            {
                "curve_id": "curve:easy1",
                "predicted_high_risk": False,
                "observed_conformal_failure": False,
                "median_calibrated_joint_coverage": 0.98,
                "median_calibrated_dangerous_point_failure_rate": 0.01,
                "risk_stratum": "ordinary_resonant",
            },
            {
                "curve_id": "curve:easy2",
                "predicted_high_risk": False,
                "observed_conformal_failure": False,
                "median_calibrated_joint_coverage": 0.99,
                "median_calibrated_dangerous_point_failure_rate": 0.00,
                "risk_stratum": "ordinary_resonant",
            },
        ],
    }

    policy = descriptor_guardrail_policy_audit(risk_audit)

    assert policy["status"] == "ok"
    assert policy["policy"] == "accept_ordinary_reject_weak_phase_high_curvature"
    assert policy["accepted_curve_count"] == 2
    assert policy["rejected_curve_count"] == 1
    assert policy["rejection_rate"] == 1 / 3
    assert policy["failure_capture_rate"] == 1.0
    assert policy["accepted_failure_rate"] == 0.0
    assert policy["accepted_median_joint_coverage"] == 0.985
    assert policy["rejected_curve_ids"] == ["curve:hard"]


def test_descriptor_same_curve_escalation_policy_audit_uses_same_curve_for_rejected_risk():
    from ai4science_fano_boundary_e import descriptor_same_curve_escalation_policy_audit

    risk_audit = {
        "status": "ok",
        "conformal_layer": "global_split_conformal",
        "curve_rows": [
            {
                "curve_id": "curve:hard",
                "predicted_high_risk": True,
                "observed_conformal_failure": True,
                "median_calibrated_joint_coverage": 0.35,
                "median_calibrated_dangerous_point_failure_rate": 0.22,
                "risk_stratum": "weak_phase_high_curvature",
            },
            {
                "curve_id": "curve:easy",
                "predicted_high_risk": False,
                "observed_conformal_failure": False,
                "median_calibrated_joint_coverage": 0.98,
                "median_calibrated_dangerous_point_failure_rate": 0.01,
                "risk_stratum": "ordinary_resonant",
            },
        ],
    }
    same_curve_rows = [
        {
            "curve_id": "curve:hard",
            "median_same_curve_joint_coverage": 0.86,
            "median_cross_curve_joint_coverage": 0.35,
            "median_same_curve_calibration_scale": 11.0,
            "median_cross_curve_calibration_scale": 5.0,
            "median_cross_curve_dangerous_point_failure_rate": 0.22,
        },
        {
            "curve_id": "curve:easy",
            "median_same_curve_joint_coverage": 0.90,
            "median_cross_curve_joint_coverage": 0.98,
            "median_same_curve_calibration_scale": 7.0,
            "median_cross_curve_calibration_scale": 8.0,
            "median_cross_curve_dangerous_point_failure_rate": 0.01,
        },
    ]

    audit = descriptor_same_curve_escalation_policy_audit(risk_audit, same_curve_rows)

    assert audit["status"] == "ok"
    assert audit["policy"] == "accept_ordinary_escalate_weak_phase_high_curvature_to_same_curve"
    assert audit["accepted_curve_count"] == 1
    assert audit["escalated_curve_count"] == 1
    assert audit["escalated_curve_ids"] == ["curve:hard"]
    assert audit["escalated_median_same_curve_joint_coverage"] == 0.86
    assert np.isclose(audit["post_policy_median_joint_coverage"], 0.92)
    assert audit["post_policy_failure_count"] == 0
    assert audit["post_policy_failure_rate"] == 0.0


def test_descriptor_risk_rule_sensitivity_audit_counts_exact_isolating_thresholds():
    from ai4science_fano_boundary_e import descriptor_risk_rule_sensitivity_audit

    rows = [
        {
            "curve_id": "curve:hard",
            "conformal_layer": "global_split_conformal",
            "phase_total_turns": 0.10,
            "max_normalized_magnitude_curvature": 100.0,
            "median_calibrated_joint_coverage": 0.35,
            "median_calibrated_dangerous_point_failure_rate": 0.22,
            "median_boundary_score": 0.67,
        },
        {
            "curve_id": "curve:weak_phase_safe",
            "conformal_layer": "global_split_conformal",
            "phase_total_turns": 0.20,
            "max_normalized_magnitude_curvature": 20.0,
            "median_calibrated_joint_coverage": 0.95,
            "median_calibrated_dangerous_point_failure_rate": 0.01,
            "median_boundary_score": 0.01,
        },
        {
            "curve_id": "curve:high_curvature_safe",
            "conformal_layer": "global_split_conformal",
            "phase_total_turns": 0.95,
            "max_normalized_magnitude_curvature": 90.0,
            "median_calibrated_joint_coverage": 0.97,
            "median_calibrated_dangerous_point_failure_rate": 0.00,
            "median_boundary_score": 0.00,
        },
        {
            "curve_id": "curve:ordinary",
            "conformal_layer": "global_split_conformal",
            "phase_total_turns": 1.00,
            "max_normalized_magnitude_curvature": 10.0,
            "median_calibrated_joint_coverage": 0.98,
            "median_calibrated_dangerous_point_failure_rate": 0.00,
            "median_boundary_score": 0.00,
        },
    ]

    audit = descriptor_risk_rule_sensitivity_audit(
        rows,
        conformal_layer="global_split_conformal",
        phase_turn_quantiles=[0.25, 0.50],
        curvature_quantiles=[0.50, 0.75],
        confidence=0.80,
    )

    assert audit["status"] == "ok"
    assert audit["grid_count"] == 4
    assert audit["exact_isolating_rule_count"] == 4
    assert audit["accepted_zero_failure_rule_count"] == 4
    assert audit["best_rule"]["phase_turn_quantile"] == 0.25
    assert audit["best_rule"]["curvature_quantile"] == 0.75
    assert audit["best_rule"]["rejected_curve_ids"] == ["curve:hard"]
