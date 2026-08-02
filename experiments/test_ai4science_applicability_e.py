from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_unified_applicability_rows_include_fano_reliability_and_eis_effects():
    from ai4science_applicability_e import build_unified_applicability_rows

    fano_summary = {
        "ranked_curve_rows": [
            {
                "curve_id": "fano:hard",
                "median_cross_curve_joint_coverage": 0.2,
                "median_cross_curve_peak_window_coverage": 0.3,
                "median_cross_curve_dangerous_point_failure_rate": 0.25,
                "median_same_to_cross_scale_ratio": 5.0,
                "phase_total_turns": 0.1,
                "max_normalized_magnitude_curvature": 100.0,
                "linewidth_fraction": 0.2,
                "fano_relative_magnitude_rmse": 0.03,
            }
        ]
    }
    eis_summary = {
        "layered_effect_descriptor_rows": [
            {
                "task_id": "eis:cell",
                "effect_layer": "neural_only",
                "status": "ok",
                "metric": "curve_mse",
                "signed_cfnn_effect": -0.5,
                "cfnn_metric": 10.0,
                "best_baseline_model": "KAN",
                "best_baseline_metric": 5.0,
                "descriptor": {
                    "phase_total_turns": 0.2,
                    "max_normalized_curvature": 3.0,
                    "relaxation_peak_count": 2,
                    "known_pole_count": 2,
                    "distributed_relaxation": False,
                    "frequency_points_per_decade": 50.0,
                    "noise": 0.02,
                },
            }
        ]
    }

    rows = build_unified_applicability_rows(fano_summary, eis_summary)

    assert len(rows) == 2
    fano = next(row for row in rows if row["domain"] == "Microwave Fano")
    eis = next(row for row in rows if row["domain"] == "EIS")
    assert fano["evidence_axis"] == "reliability_boundary"
    assert fano["boundary_score"] > 0.0
    assert fano["cfnn_effect"] is None
    assert eis["evidence_axis"] == "accuracy_effect"
    assert eis["effect_layer"] == "neural_only"
    assert eis["cfnn_effect"] == -0.5
    assert eis["boundary_score"] == 0.5


def test_eis_applicability_claim_gate_marks_physical_prior_boundary():
    from ai4science_applicability_e import summarize_eis_applicability_claim_gate

    package_summary = {
        "unified_rows": [
            {
                "domain": "EIS",
                "effect_layer": "all_baselines",
                "metric": "curve_mse",
                "cfnn_effect": -9.0,
                "descriptor_stratum": "multi_peak_relaxation",
            },
            {
                "domain": "EIS",
                "effect_layer": "physical_prior_only",
                "metric": "curve_mse",
                "cfnn_effect": -12.0,
                "descriptor_stratum": "multi_peak_relaxation",
            },
            {
                "domain": "EIS",
                "effect_layer": "neural_only",
                "metric": "curve_mse",
                "cfnn_effect": 0.3,
                "descriptor_stratum": "multi_pole_relaxation",
            },
        ],
        "eis_neural_positive_edge_audit": {
            "positive_neural_edge_count": 1,
            "status_counts": {"physical_prior_eliminates_neural_edge": 1},
        },
    }

    gate = summarize_eis_applicability_claim_gate(package_summary)

    assert gate["analysis_type"] == "eis_applicability_claim_gate"
    assert gate["claim_status"] == "physical_prior_boundary_supported"
    assert gate["physical_prior_negative_count"] == 1
    assert gate["neural_positive_edge_count"] == 1
    assert gate["physical_prior_eliminated_edge_count"] == 1
    assert gate["general_cfnn_advantage_pass"] is False


def test_fano_applicability_rows_include_prediction_level_split_conformal_boundaries():
    from ai4science_applicability_e import fano_applicability_rows

    fano_summary = {
        "ranked_curve_rows": [],
        "ranked_conformal_curve_rows": [
            {
                "curve_id": "undercoupled:r8:p03",
                "conformal_layer": "global_split_conformal",
                "median_calibrated_joint_coverage": 0.348,
                "median_calibrated_peak_window_coverage": 0.576,
                "median_calibrated_dangerous_point_failure_rate": 0.219,
                "median_calibration_scale": 6.8,
                "phase_total_turns": 0.117,
                "max_normalized_magnitude_curvature": 91894.0,
                "linewidth_fraction": 0.174,
                "fano_relative_magnitude_rmse": 0.030,
                "median_boundary_score": 0.671,
            }
        ],
    }

    rows = fano_applicability_rows(fano_summary)

    assert len(rows) == 1
    row = rows[0]
    assert row["task_id"] == "undercoupled:r8:p03_global_split_conformal"
    assert row["effect_layer"] == "prediction_level_global_split_conformal"
    assert row["evidence_axis"] == "reliability_boundary"
    assert row["cross_curve_joint_coverage"] == 0.348
    assert row["cross_curve_peak_coverage"] == 0.576
    assert row["dangerous_failure_rate"] == 0.219
    assert row["calibration_scale"] == 6.8
    assert row["phase_total_turns"] == 0.117
    assert row["boundary_score"] == 0.671


def test_fano_reliability_claim_gate_accepts_ordinary_curves_but_blocks_full_reliability():
    from ai4science_applicability_e import summarize_fano_reliability_claim_gate

    fano_summary = {
        "descriptor_guardrail_policy_audits": [
            {
                "conformal_layer": "global_split_conformal",
                "policy": "accept_ordinary_reject_weak_phase_high_curvature",
                "accepted_curve_count": 4,
                "accepted_failure_rate": 0.0,
                "failure_capture_rate": 1.0,
                "rejected_curve_ids": ["undercoupled:r8:p03"],
                "rejected_failure_rate": 1.0,
                "accepted_median_joint_coverage": 0.991,
                "accepted_median_dangerous_failure_rate": 0.002,
            }
        ],
        "descriptor_same_curve_escalation_policy_audits": [
            {
                "policy": "accept_ordinary_escalate_weak_phase_high_curvature_to_same_curve",
                "post_policy_failure_rate": 0.2,
                "escalated_curve_ids": ["undercoupled:r8:p03"],
                "escalated_median_same_curve_joint_coverage": 0.796,
                "escalated_median_same_curve_dangerous_failure_rate": 0.106,
            }
        ],
    }

    gate = summarize_fano_reliability_claim_gate({"fano_boundary_summary": fano_summary})

    assert gate["analysis_type"] == "fano_reliability_claim_gate"
    assert gate["claim_status"] == "ordinary_reliability_supported_highrisk_requires_escalation"
    assert gate["full_reliability_claim_pass"] is False
    assert gate["ordinary_curve_reliability_pass"] is True
    assert gate["highrisk_curve_ids"] == ["undercoupled:r8:p03"]
    assert gate["post_policy_failure_rate"] == 0.2


def test_fano_a2_accuracy_rows_convert_paired_relative_delta_to_signed_effect():
    from ai4science_applicability_e import fano_a2_accuracy_effect_rows

    a2_summary = {
        "pairwise_baseline_comparisons": [
            {
                "baseline_family": "CFNN",
                "comparison_family": "MLP",
                "observation_budget": 64,
                "metric": "global_complex_nrmse",
                "valid_pair_count": 10,
                "baseline_win_rate": 0.8,
                "median_relative_delta": -0.25,
            },
            {
                "baseline_family": "CFNN",
                "comparison_family": "Local nested CF control",
                "observation_budget": 64,
                "metric": "f0_abs_error_hz",
                "valid_pair_count": 10,
                "baseline_win_rate": 0.4,
                "median_relative_delta": 0.10,
            },
            {
                "baseline_family": "CFNN",
                "comparison_family": "MLP",
                "observation_budget": 16,
                "metric": "global_complex_nrmse",
                "valid_pair_count": 10,
                "baseline_win_rate": 0.8,
                "median_relative_delta": -0.25,
            },
        ],
    }

    rows = fano_a2_accuracy_effect_rows(
        a2_summary,
        budgets={64},
        comparison_families={"MLP", "Local nested CF control"},
        metrics={"global_complex_nrmse", "f0_abs_error_hz"},
    )

    assert len(rows) == 2
    mlp = next(row for row in rows if row["best_baseline_model"] == "MLP")
    local = next(row for row in rows if row["best_baseline_model"] == "Local nested CF control")
    assert mlp["cfnn_effect"] == 0.25
    assert mlp["boundary_score"] == 0.0
    assert local["cfnn_effect"] == -0.10
    assert local["boundary_score"] == 0.10


def test_fano_a3_robustness_rows_compare_cfnn_against_same_condition_baselines():
    from ai4science_applicability_e import fano_a3_robustness_effect_rows

    a3_summary = {
        "by_family_budget_condition": [
            {
                "family": "CFNN",
                "observation_budget": 64,
                "noise_snr_db": 20.0,
                "missing_fraction": 0.05,
                "frequency_drift_ppm": 5.0,
                "median_global_complex_nrmse": 0.20,
                "median_f0_abs_error_hz": 100.0,
                "parameter_recovery_failure_rate": 0.10,
                "record_count": 12,
            },
            {
                "family": "MLP",
                "observation_budget": 64,
                "noise_snr_db": 20.0,
                "missing_fraction": 0.05,
                "frequency_drift_ppm": 5.0,
                "median_global_complex_nrmse": 0.25,
                "median_f0_abs_error_hz": 80.0,
                "parameter_recovery_failure_rate": 0.20,
                "record_count": 12,
            },
            {
                "family": "Local nested CF control",
                "observation_budget": 64,
                "noise_snr_db": 20.0,
                "missing_fraction": 0.05,
                "frequency_drift_ppm": 5.0,
                "median_global_complex_nrmse": 0.10,
                "median_f0_abs_error_hz": 200.0,
                "parameter_recovery_failure_rate": 0.05,
                "record_count": 12,
            },
        ]
    }

    rows = fano_a3_robustness_effect_rows(
        a3_summary,
        source_label="combined",
        metrics={"median_global_complex_nrmse", "median_f0_abs_error_hz", "parameter_recovery_failure_rate"},
    )

    assert len(rows) == 6
    response_vs_mlp = next(
        row for row in rows
        if row["metric"] == "global_complex_nrmse" and row["best_baseline_model"] == "MLP"
    )
    f0_vs_local = next(
        row for row in rows
        if row["metric"] == "f0_abs_error_hz" and row["best_baseline_model"] == "Local nested CF control"
    )
    failure_vs_local = next(
        row for row in rows
        if row["metric"] == "parameter_recovery_failure_rate" and row["best_baseline_model"] == "Local nested CF control"
    )

    assert response_vs_mlp["evidence_axis"] == "robustness_effect"
    assert response_vs_mlp["effect_layer"] == "a3_combined_vs_MLP"
    assert response_vs_mlp["cfnn_effect"] == pytest.approx(0.20)
    assert response_vs_mlp["boundary_score"] == 0.0
    assert response_vs_mlp["observation_budget"] == 64
    assert response_vs_mlp["noise"] == 20.0
    assert response_vs_mlp["missing_fraction"] == 0.05
    assert response_vs_mlp["frequency_drift_ppm"] == 5.0
    assert f0_vs_local["cfnn_effect"] == pytest.approx(0.50)
    assert failure_vs_local["cfnn_effect"] == pytest.approx(-1.0)
    assert failure_vs_local["boundary_score"] == pytest.approx(1.0)


def test_build_unified_applicability_rows_can_include_a3_robustness_summaries():
    from ai4science_applicability_e import build_unified_applicability_rows

    fano_summary = {"ranked_curve_rows": []}
    eis_summary = {"layered_effect_descriptor_rows": []}
    a3_summary = {
        "by_family_budget_condition": [
            {
                "family": "CFNN",
                "observation_budget": 128,
                "noise_snr_db": 30.0,
                "missing_fraction": 0.0,
                "frequency_drift_ppm": 0.0,
                "median_quality_factor_relative_error": 0.05,
            },
            {
                "family": "MLP",
                "observation_budget": 128,
                "noise_snr_db": 30.0,
                "missing_fraction": 0.0,
                "frequency_drift_ppm": 0.0,
                "median_quality_factor_relative_error": 0.10,
            },
        ]
    }

    rows = build_unified_applicability_rows(
        fano_summary,
        eis_summary,
        a3_summaries=[("noise", a3_summary)],
    )

    assert len(rows) == 1
    assert rows[0]["task_id"] == "a3_noise_budget128_snr30_missing0_drift0_quality_factor_relative_error_vs_MLP"
    assert rows[0]["cfnn_effect"] == 0.5


def test_microstrip_b_rows_compare_cfnn_with_same_scenario_baselines():
    from ai4science_applicability_e import microstrip_b_device_effect_rows

    microstrip_summary = {
        "summary_rows": [
            {
                "experiment": "B1",
                "scenario": "humidity_holdout",
                "family": "CFNN",
                "calibration_frequency_count": None,
                "median_complex_nrmse": 0.10,
                "median_quality_factor_relative_error": 0.04,
                "record_count": 20,
            },
            {
                "experiment": "B1",
                "scenario": "humidity_holdout",
                "family": "MLP",
                "calibration_frequency_count": None,
                "median_complex_nrmse": 0.08,
                "median_quality_factor_relative_error": 0.08,
                "record_count": 20,
            },
            {
                "experiment": "B1",
                "scenario": "humidity_holdout",
                "family": "Local nested CF control",
                "calibration_frequency_count": None,
                "median_complex_nrmse": 0.12,
                "median_quality_factor_relative_error": 0.02,
                "record_count": 20,
            },
        ]
    }

    rows = microstrip_b_device_effect_rows(
        microstrip_summary,
        metrics={"median_complex_nrmse", "median_quality_factor_relative_error"},
    )

    assert len(rows) == 4
    response_vs_mlp = next(
        row for row in rows
        if row["metric"] == "complex_nrmse" and row["best_baseline_model"] == "MLP"
    )
    q_vs_local = next(
        row for row in rows
        if row["metric"] == "quality_factor_relative_error" and row["best_baseline_model"] == "Local nested CF control"
    )
    assert response_vs_mlp["domain"] == "Microstrip Resonator"
    assert response_vs_mlp["evidence_axis"] == "device_generalization_effect"
    assert response_vs_mlp["effect_layer"] == "b1_vs_MLP"
    assert response_vs_mlp["cfnn_effect"] == pytest.approx(-0.25)
    assert response_vs_mlp["boundary_score"] == pytest.approx(0.25)
    assert response_vs_mlp["scenario"] == "humidity_holdout"
    assert response_vs_mlp["calibration_frequency_count"] is None
    assert q_vs_local["cfnn_effect"] == pytest.approx(-1.0)


def test_microstrip_b_missed_band_rows_enter_package_e_as_negative_control_layer():
    from ai4science_applicability_e import (
        microstrip_b_missed_band_negative_control_rows,
        summarize_descriptor_strata,
    )

    missed_summary = {
        "all_metric_effect_summary_rows": [
            {
                "experiment": "B2",
                "scenario": "low_calibration_target_25c_50rh",
                "calibration_frequency_count": 4,
                "calibration_sampling_mode": "missed_band",
                "comparison_family": "MLP",
                "metric": "quality_factor_relative_error",
                "pair_count": 24,
                "cfnn_win_rate": 0.75,
                "median_cfnn_effect": 0.34,
            },
            {
                "experiment": "B2",
                "scenario": "low_calibration_target_25c_50rh",
                "calibration_frequency_count": 4,
                "calibration_sampling_mode": "random",
                "comparison_family": "MLP",
                "metric": "quality_factor_relative_error",
                "pair_count": 24,
                "cfnn_win_rate": 0.87,
                "median_cfnn_effect": 0.45,
            },
        ],
        "q_effect_summary_rows": [
            {
                "calibration_frequency_count": 4,
                "calibration_sampling_mode": "missed_band",
                "comparison_family": "MLP",
                "metric": "quality_factor_relative_error",
                "median_nearest_calibration_distance_linewidth_ratio": 1.07,
                "median_points_within_one_linewidth": 0.0,
                "median_points_within_half_linewidth": 0.0,
            }
        ],
    }

    rows = microstrip_b_missed_band_negative_control_rows(missed_summary)

    assert len(rows) == 1
    row = rows[0]
    assert row["domain"] == "Microstrip Resonator"
    assert row["evidence_axis"] == "device_calibration_negative_control_effect"
    assert row["effect_layer"] == "b2_missed_band_vs_MLP"
    assert row["task_id"] == "b2_missed_band_cal4_quality_factor_relative_error_vs_MLP"
    assert row["cfnn_effect"] == pytest.approx(0.34)
    assert row["calibration_sampling_mode"] == "missed_band"
    assert row["nearest_calibration_distance_linewidth_ratio"] == pytest.approx(1.07)
    assert row["points_within_one_linewidth"] == 0.0

    strata = summarize_descriptor_strata(rows)
    stratum = strata["stratum_rows"][0]
    assert stratum["descriptor_stratum"] == "missed_band_calibration"
    assert stratum["median_cfnn_effect"] == pytest.approx(0.34)


def test_microstrip_b2_f0_descriptor_rows_enter_package_e_as_guardrail_strata():
    from ai4science_applicability_e import (
        microstrip_b2_f0_descriptor_guardrail_rows,
        summarize_descriptor_strata,
    )

    descriptor_summary = {
        "descriptor_summary_rows": [
            {
                "comparison_family": "Local nested CF control",
                "calibration_frequency_count": 16,
                "notch_geometry_stratum": "multi_notch_or_boundary",
                "boundary_proximity_stratum": "edge_proximal",
                "calibration_coverage_stratum": "resonance_band_missed",
                "pair_count": 12,
                "f0_negative_rate": 0.75,
                "q_positive_f0_negative_rate": 0.667,
                "median_f0_effect": -1.30,
                "median_quality_factor_effect": 0.35,
            },
            {
                "comparison_family": "Local nested CF control",
                "calibration_frequency_count": 16,
                "notch_geometry_stratum": "multi_notch_or_boundary",
                "boundary_proximity_stratum": "edge_intermediate",
                "calibration_coverage_stratum": "resonance_band_missed",
                "pair_count": 12,
                "f0_negative_rate": 0.0,
                "q_positive_f0_negative_rate": 0.0,
                "median_f0_effect": 0.93,
                "median_quality_factor_effect": 0.29,
            },
        ]
    }

    rows = microstrip_b2_f0_descriptor_guardrail_rows(descriptor_summary)
    by_stratum = {row["boundary_proximity_stratum"]: row for row in rows}

    assert len(rows) == 2
    proximal = by_stratum["edge_proximal"]
    assert proximal["domain"] == "Microstrip Resonator"
    assert proximal["evidence_axis"] == "device_f0_guardrail_descriptor"
    assert proximal["effect_layer"] == "b2_f0_descriptor_vs_Local nested CF control"
    assert proximal["task_id"] == "b2_f0_descriptor_edge_proximal_missed_band_cal16_vs_Local nested CF control"
    assert proximal["metric"] == "q_positive_f0_negative_rate"
    assert proximal["cfnn_effect"] == pytest.approx(-0.667)
    assert proximal["boundary_score"] == pytest.approx(0.667)
    assert proximal["median_quality_factor_effect"] == pytest.approx(0.35)

    strata = summarize_descriptor_strata(rows)
    by_descriptor = {row["descriptor_stratum"]: row for row in strata["stratum_rows"]}
    assert "edge_proximal_missed_band" in by_descriptor
    assert "edge_intermediate_missed_band" in by_descriptor
    assert by_descriptor["edge_proximal_missed_band"]["median_boundary_score"] == pytest.approx(0.667)


def test_microstrip_guardrail_decision_table_separates_q_signal_from_f0_and_phase_risks():
    from ai4science_applicability_e import summarize_microstrip_guardrail_decisions

    rows = [
        {
            "domain": "Microstrip Resonator",
            "evidence_axis": "device_calibration_negative_control_effect",
            "effect_layer": "b2_missed_band_vs_MLP",
            "metric": "quality_factor_relative_error",
            "cfnn_effect": 0.34,
            "boundary_score": 0.0,
        },
        {
            "domain": "Microstrip Resonator",
            "evidence_axis": "device_calibration_negative_control_effect",
            "effect_layer": "b2_missed_band_vs_Local nested CF control",
            "metric": "quality_factor_relative_error",
            "cfnn_effect": 0.42,
            "boundary_score": 0.0,
        },
        {
            "domain": "Microstrip Resonator",
            "evidence_axis": "device_calibration_negative_control_effect",
            "effect_layer": "b2_missed_band_vs_MLP",
            "metric": "local_phase_transition_abs_error_hz",
            "cfnn_effect": -0.21,
            "boundary_score": 0.21,
        },
        {
            "domain": "Microstrip Resonator",
            "evidence_axis": "device_f0_guardrail_descriptor",
            "effect_layer": "b2_f0_descriptor_vs_MLP",
            "metric": "q_positive_f0_negative_rate",
            "cfnn_effect": -0.667,
            "boundary_score": 0.667,
            "boundary_proximity_stratum": "edge_proximal",
            "calibration_coverage_stratum": "resonance_band_missed",
            "q_positive_f0_negative_rate": 0.667,
        },
        {
            "domain": "Microstrip Resonator",
            "evidence_axis": "device_f0_guardrail_descriptor",
            "effect_layer": "b2_f0_descriptor_vs_MLP",
            "metric": "q_positive_f0_negative_rate",
            "cfnn_effect": -0.0,
            "boundary_score": 0.0,
            "boundary_proximity_stratum": "edge_intermediate",
            "calibration_coverage_stratum": "resonance_band_missed",
            "q_positive_f0_negative_rate": 0.0,
        },
    ]

    table = summarize_microstrip_guardrail_decisions(rows)
    by_id = {row["decision_id"]: row for row in table["decision_rows"]}

    assert table["claim_level"] == "localized_event_transfer_with_guardrails"
    assert by_id["missed_band_q_signal"]["verdict"] == "supporting_local_event_signal"
    assert by_id["missed_band_q_signal"]["median_cfnn_effect"] == pytest.approx(0.38)
    assert by_id["missed_band_phase_guardrail"]["verdict"] == "guardrail_required"
    assert by_id["edge_proximal_f0_guardrail"]["median_boundary_score"] == pytest.approx(0.667)
    assert by_id["edge_proximal_f0_guardrail"]["verdict"] == "guardrail_required"
    assert by_id["edge_intermediate_f0_guardrail"]["verdict"] == "no_current_guardrail_signal"


def test_microstrip_confirmatory_guardrail_strata_define_preregistered_b2_rules():
    from ai4science_applicability_e import summarize_microstrip_confirmatory_guardrail_strata

    rows = [
        {
            "domain": "Microstrip Resonator",
            "evidence_axis": "device_calibration_negative_control_effect",
            "metric": "quality_factor_relative_error",
            "cfnn_effect": 0.34,
            "boundary_score": 0.0,
            "calibration_sampling_mode": "missed_band",
            "calibration_frequency_count": 16,
        },
        {
            "domain": "Microstrip Resonator",
            "evidence_axis": "device_calibration_negative_control_effect",
            "metric": "local_phase_transition_abs_error_hz",
            "cfnn_effect": -0.21,
            "boundary_score": 0.21,
            "calibration_sampling_mode": "missed_band",
            "calibration_frequency_count": 16,
        },
        {
            "domain": "Microstrip Resonator",
            "evidence_axis": "device_f0_guardrail_descriptor",
            "metric": "q_positive_f0_negative_rate",
            "cfnn_effect": -0.667,
            "boundary_score": 0.667,
            "boundary_proximity_stratum": "edge_proximal",
            "calibration_coverage_stratum": "resonance_band_missed",
            "calibration_frequency_count": 16,
            "pair_count": 12,
            "median_quality_factor_effect": 0.35,
            "median_f0_effect": -1.30,
            "q_positive_f0_negative_rate": 0.667,
        },
        {
            "domain": "Microstrip Resonator",
            "evidence_axis": "device_f0_guardrail_descriptor",
            "metric": "q_positive_f0_negative_rate",
            "cfnn_effect": 0.0,
            "boundary_score": 0.0,
            "boundary_proximity_stratum": "edge_intermediate",
            "calibration_coverage_stratum": "resonance_band_missed",
            "calibration_frequency_count": 16,
            "pair_count": 12,
            "median_quality_factor_effect": 0.29,
            "median_f0_effect": 0.93,
            "q_positive_f0_negative_rate": 0.0,
        },
    ]

    summary = summarize_microstrip_confirmatory_guardrail_strata(rows)
    by_stratum = {row["stratum_id"]: row for row in summary["stratum_rows"]}

    assert summary["analysis_type"] == "microstrip_b2_confirmatory_guardrail_strata"
    assert summary["primary_claim_gate"] == "q_gain_requires_f0_and_phase_guardrails"
    assert by_stratum["edge_proximal_missed_band"]["stratum_role"] == "primary_guardrail_stratum"
    assert by_stratum["edge_proximal_missed_band"]["include_when"]["boundary_proximity_stratum"] == "edge_proximal"
    assert by_stratum["edge_proximal_missed_band"]["include_when"]["calibration_coverage_stratum"] == "resonance_band_missed"
    assert by_stratum["edge_proximal_missed_band"]["primary_positive_metric"] == "quality_factor_relative_error"
    assert by_stratum["edge_proximal_missed_band"]["guardrail_metrics"] == [
        "resonance_frequency_abs_error_hz",
        "local_phase_transition_abs_error_hz",
        "complex_nrmse",
    ]
    assert by_stratum["edge_proximal_missed_band"]["pilot_q_positive_f0_negative_rate"] == pytest.approx(0.667)
    assert by_stratum["edge_proximal_missed_band"]["pilot_median_quality_factor_effect"] == pytest.approx(0.35)
    assert by_stratum["edge_proximal_missed_band"]["provisional_status"] == "guardrail_confirmatory_target"
    assert by_stratum["edge_intermediate_missed_band"]["stratum_role"] == "contrast_stratum"
    assert by_stratum["edge_intermediate_missed_band"]["provisional_status"] == "contrast_no_current_f0_guardrail"


def test_microstrip_digital_twin_claim_gate_blocks_q_signal_when_global_f0_or_phase_fail():
    from ai4science_applicability_e import summarize_microstrip_digital_twin_claim_gate

    package_summary = {
        "unified_rows": [
            {
                "domain": "Microstrip Resonator",
                "evidence_axis": "device_generalization_effect",
                "effect_layer": "b1_vs_MLP",
                "metric": "complex_nrmse",
                "cfnn_effect": -0.20,
            },
            {
                "domain": "Microstrip Resonator",
                "evidence_axis": "device_generalization_effect",
                "effect_layer": "b1_vs_MLP",
                "metric": "resonance_frequency_abs_error_hz",
                "cfnn_effect": 0.10,
            },
            {
                "domain": "Microstrip Resonator",
                "evidence_axis": "device_generalization_effect",
                "effect_layer": "b1_vs_Local nested CF control",
                "metric": "complex_nrmse",
                "cfnn_effect": 0.05,
            },
            {
                "domain": "Microstrip Resonator",
                "evidence_axis": "device_generalization_effect",
                "effect_layer": "b1_vs_Local nested CF control",
                "metric": "resonance_frequency_abs_error_hz",
                "cfnn_effect": -0.10,
            },
        ],
        "microstrip_guardrail_decision_table": {
            "claim_level": "localized_event_transfer_with_guardrails",
            "decision_rows": [
                {
                    "decision_id": "missed_band_q_signal",
                    "verdict": "supporting_local_event_signal",
                },
                {
                    "decision_id": "missed_band_phase_guardrail",
                    "verdict": "guardrail_required",
                },
            ],
        },
        "microstrip_confirmatory_guardrail_strata": {
            "stratum_rows": [
                {
                    "stratum_id": "edge_proximal_missed_band",
                    "stratum_role": "primary_guardrail_stratum",
                    "provisional_status": "guardrail_confirmatory_target",
                }
            ],
        },
    }

    gate = summarize_microstrip_digital_twin_claim_gate(package_summary)

    assert gate["analysis_type"] == "microstrip_digital_twin_claim_gate"
    assert gate["claim_status"] == "blocked_by_global_event_guardrails"
    assert gate["localized_q_signal_status"] == "supporting_local_event_signal"
    assert gate["full_digital_twin_pass"] is False
    assert "b1_vs_MLP" in gate["failed_effect_layers"]
    assert "b1_vs_Local nested CF control" in gate["failed_effect_layers"]
    assert "missed_band_phase_guardrail" in gate["blocking_guardrail_decisions"]


def test_package_e_report_includes_microstrip_guardrail_decision_table():
    from run_ai4science_applicability_e import build_report

    summary = {
        "unified_rows": [],
        "unified_summary": {"domain_summaries": {}},
        "microstrip_guardrail_decision_table": {
            "claim_level": "localized_event_transfer_with_guardrails",
            "decision_rows": [
                {
                    "decision_id": "missed_band_q_signal",
                    "claim_scope": "localized_Q_linewidth_event_transfer",
                    "row_count": 2,
                    "median_cfnn_effect": 0.38,
                    "positive_effect_count": 2,
                    "negative_effect_count": 0,
                    "median_boundary_score": 0.0,
                    "verdict": "supporting_local_event_signal",
                }
            ],
        },
    }

    report = build_report(summary)

    assert "## Microstrip Guardrail Decision Table" in report
    assert "localized_event_transfer_with_guardrails" in report
    assert "missed_band_q_signal" in report
    assert "supporting_local_event_signal" in report


def test_package_e_report_includes_microstrip_digital_twin_claim_gate():
    from run_ai4science_applicability_e import build_report

    summary = {
        "unified_rows": [],
        "unified_summary": {"domain_summaries": {}},
        "microstrip_digital_twin_claim_gate": {
            "claim_status": "blocked_by_global_event_guardrails",
            "localized_q_signal_status": "supporting_local_event_signal",
            "failed_effect_layers": ["b1_vs_MLP"],
            "blocking_guardrail_decisions": ["missed_band_phase_guardrail"],
            "blocking_confirmatory_strata": ["edge_proximal_missed_band"],
            "gate_rows": [
                {
                    "effect_layer": "b1_vs_MLP",
                    "layer_status": "fails_full_response_event_gate",
                    "metric_effects": {
                        "complex_nrmse": -0.2,
                        "resonance_frequency_abs_error_hz": 0.1,
                    },
                    "failed_metrics": ["complex_nrmse"],
                    "missing_metrics": [],
                }
            ],
        },
    }

    report = build_report(summary)

    assert "## Microstrip Digital Twin Claim Gate" in report
    assert "blocked_by_global_event_guardrails" in report
    assert "supporting_local_event_signal" in report
    assert "missed_band_phase_guardrail" in report
    assert "edge_proximal_missed_band" in report


def test_package_e_report_includes_fano_a2_preregistered_success_audit():
    from run_ai4science_applicability_e import build_report

    summary = {
        "unified_rows": [],
        "unified_summary": {"domain_summaries": {}},
        "fano_a2_preregistered_success_audit": {
            "overall_status": "passes_at_some_main_budgets",
            "passed_budgets": [64, 128],
            "budget_comparison_rows": [
                {
                    "observation_budget": 32,
                    "comparison_family": "Local nested CF control",
                    "success_status": "fails_preregistered_success",
                    "global_guardrail_effect": -0.064,
                    "metric_effects": {
                        "peak_window_complex_nrmse": -0.076,
                        "f0_abs_error_hz": 0.355,
                        "quality_factor_relative_error": 0.162,
                        "q_abs_error": -0.014,
                    },
                    "failed_required_metrics": ["peak_window_complex_nrmse", "q_abs_error"],
                },
                {
                    "observation_budget": 64,
                    "comparison_family": "Local nested CF control",
                    "success_status": "passes_preregistered_success",
                    "global_guardrail_effect": 0.052,
                    "metric_effects": {
                        "peak_window_complex_nrmse": 0.059,
                        "f0_abs_error_hz": 0.381,
                        "quality_factor_relative_error": 0.347,
                        "q_abs_error": 0.125,
                    },
                    "failed_required_metrics": [],
                },
            ],
        },
    }

    report = build_report(summary)

    assert "## Microwave Fano A2 Preregistered Success Audit" in report
    assert "passes_at_some_main_budgets" in report
    assert "64,128" in report
    assert "fails_preregistered_success" in report
    assert "peak_window_complex_nrmse,q_abs_error" in report


def test_package_e_report_includes_fano_a3_preregistered_robustness_audit():
    from run_ai4science_applicability_e import build_report

    summary = {
        "unified_rows": [],
        "unified_summary": {"domain_summaries": {}},
        "fano_a3_preregistered_robustness_audit": {
            "overall_status": "does_not_pass_local_control_guardrail",
            "condition_count": 32,
            "passed_condition_count": 0,
            "mlp_only_or_partial_condition_count": 9,
            "condition_gate_rows": [
                {
                    "source_label": "noise",
                    "observation_budget": 64,
                    "noise_snr_db": 30.0,
                    "missing_fraction": 0.0,
                    "frequency_drift_ppm": 0.0,
                    "condition_status": "blocked_by_baseline_guardrail",
                    "passed_comparison_families": ["MLP"],
                }
            ],
        },
    }

    report = build_report(summary)

    assert "## Microwave Fano A3 Preregistered Robustness Audit" in report
    assert "does_not_pass_local_control_guardrail" in report
    assert "MLP-only or partial condition count: `9`" in report
    assert "blocked_by_baseline_guardrail" in report


def test_package_e_report_includes_fano_reliability_claim_gate():
    from run_ai4science_applicability_e import build_report

    summary = {
        "unified_rows": [],
        "unified_summary": {"domain_summaries": {}},
        "fano_reliability_claim_gate": {
            "claim_status": "ordinary_reliability_supported_highrisk_requires_escalation",
            "ordinary_curve_reliability_pass": True,
            "full_reliability_claim_pass": False,
            "accepted_failure_rate": 0.0,
            "accepted_median_joint_coverage": 0.991,
            "accepted_median_dangerous_failure_rate": 0.002,
            "failure_capture_rate": 1.0,
            "highrisk_curve_ids": ["undercoupled:r8:p03"],
            "post_policy_failure_rate": 0.2,
        },
    }

    report = build_report(summary)

    assert "## Microwave Fano Reliability Claim Gate" in report
    assert "ordinary_reliability_supported_highrisk_requires_escalation" in report
    assert "undercoupled:r8:p03" in report
    assert "post-policy failure rate" in report


def test_package_e_report_includes_microstrip_confirmatory_guardrail_strata():
    from run_ai4science_applicability_e import build_report

    summary = {
        "unified_rows": [],
        "unified_summary": {"domain_summaries": {}},
        "microstrip_confirmatory_guardrail_strata": {
            "primary_claim_gate": "q_gain_requires_f0_and_phase_guardrails",
            "stratum_rows": [
                {
                    "stratum_id": "edge_proximal_missed_band",
                    "stratum_role": "primary_guardrail_stratum",
                    "calibration_frequency_counts": [16, 32],
                    "pilot_q_positive_f0_negative_rate": 0.375,
                    "pilot_median_quality_factor_effect": 0.43,
                    "pilot_median_f0_effect": -0.82,
                    "pilot_global_missed_band_phase_effect": -0.04,
                    "provisional_status": "guardrail_confirmatory_target",
                }
            ],
        },
    }

    report = build_report(summary)

    assert "## Microstrip Confirmatory Guardrail Strata" in report
    assert "q_gain_requires_f0_and_phase_guardrails" in report
    assert "edge_proximal_missed_band" in report
    assert "guardrail_confirmatory_target" in report


def test_package_e_report_includes_eis_neural_positive_edge_audit_summary():
    from run_ai4science_applicability_e import build_report

    summary = {
        "unified_rows": [],
        "unified_summary": {"domain_summaries": {}},
        "eis_neural_positive_edge_audit": {
            "positive_neural_edge_count": 4,
            "status_counts": {"physical_prior_eliminates_neural_edge": 4},
            "positive_edge_rows": [
                {
                    "task_id": "voigt_K4_n40_noise0.1",
                    "edge_status": "physical_prior_eliminates_neural_edge",
                    "neural_effect": 0.365,
                    "physical_prior_effect": -2.027,
                    "descriptor_stratum": "multi_pole_relaxation",
                }
            ],
        },
    }

    report = build_report(summary)

    assert "## EIS Neural-Only Positive Edge Audit" in report
    assert "physical_prior_eliminates_neural_edge" in report
    assert "voigt_K4_n40_noise0.1" in report


def test_package_e_report_includes_eis_applicability_claim_gate():
    from run_ai4science_applicability_e import build_report

    summary = {
        "unified_rows": [],
        "unified_summary": {"domain_summaries": {}},
        "eis_applicability_claim_gate": {
            "claim_status": "physical_prior_boundary_supported",
            "general_cfnn_advantage_pass": False,
            "all_baseline_count": 25,
            "all_baseline_negative_count": 25,
            "all_baseline_median_effect": -9.94,
            "neural_only_count": 25,
            "neural_only_positive_count": 4,
            "neural_only_median_effect": -0.581,
            "physical_prior_count": 22,
            "physical_prior_negative_count": 22,
            "physical_prior_median_effect": -12.819,
            "neural_positive_edge_count": 4,
            "physical_prior_eliminated_edge_count": 4,
        },
    }

    report = build_report(summary)

    assert "## EIS Applicability Claim Gate" in report
    assert "physical_prior_boundary_supported" in report
    assert "general CFNN advantage pass" in report
    assert "physical-prior eliminated neural edges" in report


def test_package_e_json_safe_preserves_boolean_gate_fields():
    from run_ai4science_applicability_e import _json_safe

    result = _json_safe({"general_cfnn_advantage_pass": False})

    assert result["general_cfnn_advantage_pass"] is False


def test_ai4science_experiment_plan_progress_audit_tracks_main_packages():
    from ai4science_experiment_plan_audit import summarize_ai4science_experiment_plan_progress

    package_e = {
        "unified_summary": {"row_count": 601},
        "fano_a2_preregistered_success_audit": {
            "overall_status": "passes_at_some_main_budgets",
            "budgets": [32, 64, 128],
            "passed_budgets": [64, 128],
        },
        "fano_a3_preregistered_robustness_audit": {
            "overall_status": "does_not_pass_local_control_guardrail",
            "condition_count": 32,
            "passed_condition_count": 0,
            "mlp_only_or_partial_condition_count": 9,
        },
        "fano_reliability_claim_gate": {
            "claim_status": "ordinary_reliability_supported_highrisk_requires_escalation",
            "highrisk_curve_ids": ["undercoupled:r8:p03"],
        },
        "microstrip_digital_twin_claim_gate": {
            "claim_status": "blocked_by_global_event_guardrails",
            "localized_q_signal_status": "supporting_local_event_signal",
        },
        "eis_applicability_claim_gate": {
            "claim_status": "physical_prior_boundary_supported",
            "general_cfnn_advantage_pass": False,
            "neural_positive_edge_count": 4,
            "physical_prior_eliminated_edge_count": 4,
        },
    }
    claim_ledger = {
        "claim_summary": {
            "claim_count": 4,
            "cfnn_advantage_claim_count": 1,
            "blocked_or_boundary_claim_count": 3,
        },
        "claim_rows": [
            {
                "claim_id": "fano_sparse_parameter_recovery",
                "claim_status": "supported_with_preregistered_budget_guardrails",
            },
            {
                "claim_id": "microstrip_device_digital_twin",
                "claim_status": "blocked_by_global_event_guardrails",
            },
            {
                "claim_id": "aluminium_active_modal_cfnn_advantage",
                "claim_status": "measurement_strategy_value_without_cfnn_advantage",
            },
            {
                "claim_id": "eis_general_advantage",
                "claim_status": "physical_prior_boundary_supported",
            },
        ],
    }
    aluminium = {
        "claim_level": "measurement_strategy_signal_not_cfnn_advantage",
        "decision_rows": [
            {
                "decision_id": "observed_interpolation_measurement_signal",
                "minimum_budget": 64,
                "median_passed_modes": 5.0,
            },
            {
                "decision_id": "same_schedule_cfnn_boundary",
                "median_passed_modes": 0.0,
            },
        ],
    }

    audit = summarize_ai4science_experiment_plan_progress(package_e, claim_ledger, aluminium)
    rows = {row["package_id"]: row for row in audit["plan_rows"]}

    assert audit["analysis_type"] == "ai4science_experiment_plan_progress_audit"
    assert audit["core_question"] == "CFNN_sparse_resonant_complex_response_recovery_interpretability_reliability"
    assert audit["summary"]["row_count"] == 8
    assert audit["summary"]["cfnn_advantage_package_count"] == 1
    assert rows["A2"]["status"] == "supported_with_preregistered_budget_guardrails"
    assert rows["A3"]["status"] == "robustness_guardrail_not_full_reliability"
    assert rows["D"]["status"] == "ordinary_reliability_supported_highrisk_requires_escalation"
    assert rows["B"]["status"] == "localized_q_signal_blocked_full_digital_twin"
    assert rows["C"]["status"] == "measurement_strategy_value_without_cfnn_advantage"
    assert rows["E"]["status"] == "physical_prior_boundary_supported"
    assert "64/128" in rows["A2"]["evidence"]
    assert rows["C"]["measurement_strategy_minimum_budget"] == 64


def test_ai4science_experiment_plan_progress_report_lists_next_actions():
    from run_ai4science_experiment_plan_progress_audit import build_report

    audit = {
        "summary": {
            "row_count": 2,
            "cfnn_advantage_package_count": 1,
            "boundary_or_guardrail_package_count": 1,
        },
        "plan_rows": [
            {
                "package_id": "A2",
                "package_name": "Microwave Fano parameter recovery",
                "status": "supported_with_preregistered_budget_guardrails",
                "claim_role": "main_cfnn_advantage",
                "evidence": "passes at 64/128 observations",
                "next_action": "expand confirmatory curves",
            },
            {
                "package_id": "E",
                "package_name": "EIS applicability boundary",
                "status": "physical_prior_boundary_supported",
                "claim_role": "negative_applicability_boundary",
                "evidence": "4/4 neural edges eliminated",
                "next_action": "external spectrum validation",
            },
        ],
    }

    report = build_report(audit)

    assert "# AI4Science Experiment Plan Progress Audit" in report
    assert "Microwave Fano parameter recovery" in report
    assert "physical_prior_boundary_supported" in report
    assert "external spectrum validation" in report


def test_next_experiment_queue_prioritizes_confirmatory_and_boundary_work():
    from ai4science_next_experiment_queue import build_next_experiment_queue

    progress = {
        "plan_rows": [
            {
                "package_id": "A2",
                "status": "supported_with_preregistered_budget_guardrails",
            },
            {
                "package_id": "B",
                "status": "localized_q_signal_blocked_full_digital_twin",
            },
            {
                "package_id": "C",
                "status": "measurement_strategy_value_without_cfnn_advantage",
            },
            {
                "package_id": "D",
                "status": "ordinary_reliability_supported_highrisk_requires_escalation",
            },
            {
                "package_id": "E",
                "status": "physical_prior_boundary_supported",
            },
        ]
    }

    queue = build_next_experiment_queue(progress)
    items = {row["queue_id"]: row for row in queue["queue_rows"]}

    assert queue["analysis_type"] == "ai4science_next_experiment_queue"
    assert queue["summary"]["queue_count"] >= 4
    assert queue["summary"]["runnable_now_count"] >= 2
    assert items["A_fano_event_focused_consolidation"]["priority"] == 1
    assert items["A_fano_event_focused_consolidation"]["package_id"] == "A2"
    assert items["A_fano_event_focused_consolidation"]["execution_status"] == "runnable_now"
    assert "run_ai4science_fano_event_focus_consolidation.py" in items[
        "A_fano_event_focused_consolidation"
    ]["command"]
    assert items["B_microstrip_low_calibration_event_transfer"]["execution_status"] == "runnable_now"
    assert "run_ai4science_microstrip_low_calibration_event_transfer.py" in items[
        "B_microstrip_low_calibration_event_transfer"
    ]["command"]
    assert "--audit-summary" in items["D_highrisk_budget_multisplit_refresh"]["command"]
    assert "microwave_fano_e_highrisk_budget_escalation_r8p03/summary.json" in items["D_highrisk_budget_multisplit_refresh"]["command"]
    assert "microwave_fano_e_highrisk_budget_escalation_r8p03_split733/summary.json" in items["D_highrisk_budget_multisplit_refresh"]["command"]
    assert "microwave_fano_e_highrisk_budget_escalation_r8p03_split751/summary.json" in items["D_highrisk_budget_multisplit_refresh"]["command"]
    assert "C_observed_candidate_modeling" not in items
    assert items["C_aluminium_boundary_retained"]["execution_status"] == "satisfied_current_artifact"
    assert items["E_external_spectrum_boundary"]["execution_status"] == "satisfied_current_artifact"


def test_next_experiment_queue_pivots_from_aluminium_to_fano_microstrip_focus():
    from ai4science_next_experiment_queue import build_next_experiment_queue

    progress = {
        "completed_queue_artifacts": {
            "B_confirmatory_guardrail_strata": {
                "status": "ok",
                "artifact_path": "ai4science_results/microstrip_b2_confirmatory_guardrail_strata/summary.json",
            },
            "D_highrisk_budget_multisplit_refresh": {
                "status": "ok",
                "artifact_path": "ai4science_results/microwave_fano_e_highrisk_budget_escalation_r8p03_multisplit/summary.json",
            },
            "C_aluminium_boundary_retained": {
                "status": "ok",
                "artifact_path": "ai4science_results/aluminium_c_active_measurement_value_audit/summary.json",
            },
        },
        "plan_rows": [
            {"package_id": "A2", "status": "blocked_by_preregistered_budget_gate"},
            {"package_id": "B", "status": "localized_q_signal_blocked_full_digital_twin"},
            {"package_id": "C", "status": "measurement_strategy_value_without_cfnn_advantage"},
            {"package_id": "D", "status": "ordinary_reliability_supported_highrisk_requires_escalation"},
            {"package_id": "E", "status": "physical_prior_boundary_supported"},
        ],
    }

    queue = build_next_experiment_queue(progress)
    items = {row["queue_id"]: row for row in queue["queue_rows"]}

    assert "C_observed_candidate_modeling" not in items
    assert items["A_fano_event_focused_consolidation"]["priority"] == 1
    assert items["A_fano_event_focused_consolidation"]["execution_status"] == "runnable_now"
    assert "run_ai4science_fano_event_focus_consolidation.py" in items[
        "A_fano_event_focused_consolidation"
    ]["command"]
    assert items["B_microstrip_low_calibration_event_transfer"]["priority"] == 2
    assert items["B_microstrip_low_calibration_event_transfer"]["execution_status"] == "runnable_now"
    assert "run_ai4science_microstrip_low_calibration_event_transfer.py" in items[
        "B_microstrip_low_calibration_event_transfer"
    ]["command"]
    assert items["C_aluminium_boundary_retained"]["execution_status"] == "satisfied_current_artifact"
    assert items["C_aluminium_boundary_retained"]["command"] is None
    assert queue["summary"]["design_needed_count"] == 0


def test_next_experiment_queue_marks_verified_artifacts_as_satisfied_not_runnable():
    from ai4science_next_experiment_queue import build_next_experiment_queue

    progress = {
        "completed_queue_artifacts": {
            "B_microstrip_low_calibration_event_transfer": {
                "status": "ok",
                "artifact_path": "ai4science_results/microstrip_low_calibration_event_transfer/summary.json",
                "claim_gate_row_count": 16,
            },
            "C_aluminium_boundary_retained": {
                "status": "ok",
                "artifact_path": "ai4science_results/aluminium_c_active_measurement_value_audit/summary.json",
            },
            "E_external_spectrum_boundary": {
                "status": "ok",
                "artifact_path": "ai4science_results/eis_external_spectrum_boundary/summary.json",
                "claim_status": "physical_prior_boundary_supported",
            },
            "D_highrisk_budget_multisplit_refresh": {
                "status": "ok",
                "artifact_path": "ai4science_results/microwave_fano_e_highrisk_budget_escalation_r8p03_multisplit/summary.json",
                "best_family": "Local nested CF control",
            },
        },
        "plan_rows": [
            {
                "package_id": "B",
                "status": "localized_q_signal_blocked_full_digital_twin",
            },
            {
                "package_id": "C",
                "status": "measurement_strategy_value_without_cfnn_advantage",
            },
            {
                "package_id": "D",
                "status": "ordinary_reliability_supported_highrisk_requires_escalation",
            },
            {
                "package_id": "E",
                "status": "physical_prior_boundary_supported",
            },
        ],
    }

    queue = build_next_experiment_queue(progress)
    items = {row["queue_id"]: row for row in queue["queue_rows"]}

    assert items["B_microstrip_low_calibration_event_transfer"]["execution_status"] == "satisfied_current_artifact"
    assert items["B_microstrip_low_calibration_event_transfer"]["command"] is None
    assert items["B_microstrip_low_calibration_event_transfer"]["artifact_path"].endswith("summary.json")
    assert items["C_aluminium_boundary_retained"]["execution_status"] == "satisfied_current_artifact"
    assert items["C_aluminium_boundary_retained"]["command"] is None
    assert items["E_external_spectrum_boundary"]["execution_status"] == "satisfied_current_artifact"
    assert items["E_external_spectrum_boundary"]["command"] is None
    assert items["D_highrisk_budget_multisplit_refresh"]["execution_status"] == "satisfied_current_artifact"
    assert items["D_highrisk_budget_multisplit_refresh"]["command"] is None
    assert queue["summary"]["satisfied_current_artifact_count"] == 4
    assert queue["summary"]["runnable_now_count"] == 1


def test_next_experiment_queue_report_includes_commands_and_gates():
    from run_ai4science_next_experiment_queue import build_report

    queue = {
        "summary": {"queue_count": 1, "runnable_now_count": 1, "design_needed_count": 0},
        "queue_rows": [
            {
                "queue_id": "A_fano_event_focused_consolidation",
                "priority": 1,
                "package_id": "A2",
                "objective": "consolidate Fano event recovery",
                "execution_status": "runnable_now",
                "command": "python experiment_refine/run_ai4science_fano_event_focus_consolidation.py",
                "success_gate": "event-focused peak-window and f0 gate",
            }
        ],
    }

    report = build_report(queue)

    assert "# AI4Science Next Experiment Queue" in report
    assert "A_fano_event_focused_consolidation" in report
    assert "run_ai4science_fano_event_focus_consolidation.py" in report
    assert "event-focused peak-window and f0 gate" in report


def test_next_experiment_queue_runner_discovers_focus_and_boundary_artifacts(tmp_path, monkeypatch):
    import json
    import run_ai4science_next_experiment_queue as runner

    fano = tmp_path / "fano" / "summary.json"
    microstrip = tmp_path / "microstrip" / "summary.json"
    aluminium = tmp_path / "aluminium" / "summary.json"
    fano.parent.mkdir()
    microstrip.parent.mkdir()
    aluminium.parent.mkdir()
    fano.write_text(json.dumps({
        "analysis_type": "microwave_fano_event_focused_consolidation",
        "claim_gate": {"claim_status": "passes_event_focused_gate_q_blocks_full_inversion"},
    }))
    microstrip.write_text(json.dumps({
        "analysis_type": "microstrip_low_calibration_event_transfer",
        "claim_gate": {"claim_status": "localized_q_transfer_with_guardrails_not_digital_twin"},
        "focused_stratum_rows": [{"stratum_id": "edge_intermediate_missed_band"}],
    }))
    aluminium.write_text(json.dumps({
        "analysis_type": "aluminium_c_active_measurement_value_audit",
        "claim_level": "measurement_strategy_signal_not_cfnn_advantage",
    }))

    monkeypatch.setattr(runner, "FANO_EVENT_FOCUS_SUMMARY", fano, raising=False)
    monkeypatch.setattr(runner, "MICROSTRIP_LOW_CALIBRATION_EVENT_TRANSFER_SUMMARY", microstrip, raising=False)
    monkeypatch.setattr(runner, "ALUMINIUM_ACTIVE_MEASUREMENT_VALUE_SUMMARY", aluminium, raising=False)

    artifacts = runner._completed_queue_artifacts()

    assert artifacts["A_fano_event_focused_consolidation"]["status"] == "ok"
    assert artifacts["B_microstrip_low_calibration_event_transfer"]["status"] == "ok"
    assert artifacts["C_aluminium_boundary_retained"]["status"] == "ok"


def test_package_e_report_includes_aluminium_active_measurement_value_audit():
    from run_ai4science_applicability_e import build_report

    summary = {
        "unified_rows": [],
        "unified_summary": {"domain_summaries": {}},
        "aluminium_active_measurement_value_audit": {
            "claim_level": "measurement_strategy_signal_not_cfnn_advantage",
            "decision_rows": [
                {
                    "decision_id": "observed_interpolation_measurement_signal",
                    "evidence_scope": "observed_only_half_power_interpolation",
                    "family": "Complex linear interpolation",
                    "minimum_budget": 64,
                    "median_passed_modes": 5.0,
                    "verdict": "supports_measurement_efficiency_signal",
                },
                {
                    "decision_id": "same_schedule_cfnn_boundary",
                    "evidence_scope": "same_schedule_neural_model_check",
                    "family": "CFNN",
                    "minimum_budget": 48,
                    "median_passed_modes": 0.0,
                    "verdict": "blocks_cfnn_advantage_claim",
                },
            ],
        },
    }

    report = build_report(summary)

    assert "## Aluminium Active Measurement Value Audit" in report
    assert "measurement_strategy_signal_not_cfnn_advantage" in report
    assert "observed_interpolation_measurement_signal" in report
    assert "blocks_cfnn_advantage_claim" in report


def test_aluminium_active_modal_claim_gate_separates_measurement_value_from_cfnn_advantage():
    from ai4science_applicability_e import summarize_aluminium_active_modal_claim_gate

    package_summary = {
        "aluminium_active_measurement_value_audit": {
            "claim_level": "measurement_strategy_signal_not_cfnn_advantage",
            "decision_rows": [
                {
                    "decision_id": "observed_interpolation_measurement_signal",
                    "minimum_budget": 64,
                    "median_passed_modes": 5.0,
                    "verdict": "supports_measurement_efficiency_signal",
                },
                {
                    "decision_id": "same_schedule_cfnn_boundary",
                    "minimum_budget": 48,
                    "median_passed_modes": 0.0,
                    "verdict": "blocks_cfnn_advantage_claim",
                },
                {
                    "decision_id": "candidate_discovery_boundary",
                    "minimum_budget": 128,
                    "median_passed_modes": 3.0,
                    "verdict": "candidate_discovery_limits_full_modal_recovery",
                },
            ],
        }
    }

    gate = summarize_aluminium_active_modal_claim_gate(package_summary)

    assert gate["analysis_type"] == "aluminium_active_modal_claim_gate"
    assert gate["claim_status"] == "measurement_strategy_value_without_cfnn_advantage"
    assert gate["measurement_strategy_minimum_budget"] == 64
    assert gate["same_schedule_cfnn_passed_modes"] == 0.0
    assert gate["candidate_discovery_status"] == "candidate_discovery_limits_full_modal_recovery"
    assert gate["cfnn_advantage_pass"] is False


def test_package_e_report_includes_aluminium_active_modal_claim_gate():
    from run_ai4science_applicability_e import build_report

    summary = {
        "unified_rows": [],
        "unified_summary": {"domain_summaries": {}},
        "aluminium_active_modal_claim_gate": {
            "claim_status": "measurement_strategy_value_without_cfnn_advantage",
            "measurement_strategy_minimum_budget": 64,
            "same_schedule_cfnn_passed_modes": 0.0,
            "candidate_discovery_status": "candidate_discovery_limits_full_modal_recovery",
        },
    }

    report = build_report(summary)

    assert "## Aluminium Active Modal Claim Gate" in report
    assert "measurement_strategy_value_without_cfnn_advantage" in report
    assert "candidate_discovery_limits_full_modal_recovery" in report


def test_ai4science_claim_ledger_separates_cfnn_advantages_from_boundaries():
    from ai4science_applicability_e import summarize_ai4science_claim_ledger

    package_summary = {
        "unified_summary": {"row_count": 10},
        "unified_rows": [
            {
                "domain": "Microwave Fano",
                "evidence_axis": "accuracy_effect",
                "effect_layer": "a2_vs_MLP",
                "metric": "quality_factor_relative_error",
                "cfnn_effect": 0.50,
                "boundary_score": 0.0,
            },
            {
                "domain": "Microwave Fano",
                "evidence_axis": "accuracy_effect",
                "effect_layer": "a2_vs_Local nested CF control",
                "metric": "f0_abs_error_hz",
                "cfnn_effect": 0.20,
                "boundary_score": 0.0,
            },
            {
                "domain": "Microwave Fano",
                "evidence_axis": "reliability_boundary",
                "effect_layer": "prediction_level_global_split_conformal",
                "metric": "joint_coverage",
                "task_id": "undercoupled:r8:p03_global_split_conformal",
                "boundary_score": 0.78,
            },
            {
                "domain": "EIS",
                "evidence_axis": "accuracy_effect",
                "effect_layer": "physical_prior_only",
                "metric": "curve_mse",
                "cfnn_effect": -2.0,
                "boundary_score": 2.0,
            },
        ],
        "microstrip_guardrail_decision_table": {
            "claim_level": "localized_event_transfer_with_guardrails",
        },
        "aluminium_active_measurement_value_audit": {
            "claim_level": "measurement_strategy_signal_not_cfnn_advantage",
        },
        "eis_neural_positive_edge_audit": {
            "positive_neural_edge_count": 4,
            "status_counts": {"physical_prior_eliminates_neural_edge": 4},
        },
    }

    ledger = summarize_ai4science_claim_ledger(package_summary)
    rows = {row["claim_id"]: row for row in ledger["claim_rows"]}

    assert ledger["analysis_type"] == "ai4science_claim_ledger"
    assert rows["fano_sparse_parameter_recovery"]["claim_status"] == "supported_with_guardrails"
    assert rows["fano_sparse_parameter_recovery"]["claim_type"] == "cfnn_advantage"
    assert rows["microstrip_device_digital_twin"]["claim_status"] == "blocked_by_guardrails"
    assert rows["aluminium_active_modal_cfnn_advantage"]["claim_status"] == "not_supported_measurement_strategy_only"
    assert rows["eis_general_advantage"]["claim_status"] == "not_supported_physical_priors_dominate"
    assert ledger["claim_summary"]["cfnn_advantage_claim_count"] == 1
    assert ledger["claim_summary"]["blocked_or_boundary_claim_count"] == 3


def test_ai4science_claim_ledger_uses_microstrip_digital_twin_gate_when_available():
    from ai4science_applicability_e import summarize_ai4science_claim_ledger

    package_summary = {
        "unified_summary": {"row_count": 10},
        "unified_rows": [],
        "microstrip_guardrail_decision_table": {
            "claim_level": "localized_event_transfer_with_guardrails",
        },
        "microstrip_digital_twin_claim_gate": {
            "claim_status": "blocked_by_global_event_guardrails",
            "localized_q_signal_status": "supporting_local_event_signal",
            "failed_effect_layers": ["b1_vs_MLP", "b2_vs_MLP"],
            "blocking_guardrail_decisions": ["missed_band_phase_guardrail"],
        },
    }

    ledger = summarize_ai4science_claim_ledger(package_summary)
    row = next(item for item in ledger["claim_rows"] if item["claim_id"] == "microstrip_device_digital_twin")

    assert row["claim_status"] == "blocked_by_global_event_guardrails"
    assert row["localized_q_signal_status"] == "supporting_local_event_signal"
    assert row["failed_effect_layers"] == ["b1_vs_MLP", "b2_vs_MLP"]
    assert "full digital twin" in row["reporting_rule"]


def test_ai4science_claim_ledger_uses_aluminium_active_modal_claim_gate_when_available():
    from ai4science_applicability_e import summarize_ai4science_claim_ledger

    package_summary = {
        "unified_summary": {"row_count": 10},
        "unified_rows": [],
        "aluminium_active_measurement_value_audit": {
            "claim_level": "measurement_strategy_signal_not_cfnn_advantage",
        },
        "aluminium_active_modal_claim_gate": {
            "claim_status": "measurement_strategy_value_without_cfnn_advantage",
            "measurement_strategy_minimum_budget": 64,
            "same_schedule_cfnn_passed_modes": 0.0,
            "candidate_discovery_status": "candidate_discovery_limits_full_modal_recovery",
        },
    }

    ledger = summarize_ai4science_claim_ledger(package_summary)
    row = next(item for item in ledger["claim_rows"] if item["claim_id"] == "aluminium_active_modal_cfnn_advantage")

    assert row["claim_status"] == "measurement_strategy_value_without_cfnn_advantage"
    assert row["measurement_strategy_minimum_budget"] == 64
    assert row["same_schedule_cfnn_passed_modes"] == 0.0
    assert "measurement strategy" in row["reporting_rule"]


def test_ai4science_claim_ledger_uses_eis_applicability_claim_gate_when_available():
    from ai4science_applicability_e import summarize_ai4science_claim_ledger

    package_summary = {
        "unified_summary": {"row_count": 10},
        "unified_rows": [],
        "eis_neural_positive_edge_audit": {
            "positive_neural_edge_count": 4,
            "status_counts": {"physical_prior_eliminates_neural_edge": 4},
        },
        "eis_applicability_claim_gate": {
            "claim_status": "physical_prior_boundary_supported",
            "general_cfnn_advantage_pass": False,
            "physical_prior_negative_count": 22,
            "neural_positive_edge_count": 4,
            "physical_prior_eliminated_edge_count": 4,
            "physical_prior_median_effect": -12.819,
        },
    }

    ledger = summarize_ai4science_claim_ledger(package_summary)
    row = next(item for item in ledger["claim_rows"] if item["claim_id"] == "eis_general_advantage")

    assert row["claim_status"] == "physical_prior_boundary_supported"
    assert row["physical_prior_negative_count"] == 22
    assert row["neural_positive_edge_count"] == 4
    assert row["physical_prior_eliminated_edge_count"] == 4
    assert row["physical_prior_median_effect"] == -12.819
    assert "neural-only edges" in row["reporting_rule"]


def test_ai4science_claim_ledger_uses_fano_a2_preregistered_budget_gate_when_available():
    from ai4science_applicability_e import summarize_ai4science_claim_ledger

    package_summary = {
        "unified_summary": {"row_count": 4},
        "unified_rows": [
            {
                "domain": "Microwave Fano",
                "evidence_axis": "accuracy_effect",
                "effect_layer": "a2_vs_MLP",
                "metric": "quality_factor_relative_error",
                "cfnn_effect": 0.50,
                "boundary_score": 0.0,
            },
            {
                "domain": "Microwave Fano",
                "evidence_axis": "accuracy_effect",
                "effect_layer": "a2_vs_Local nested CF control",
                "metric": "f0_abs_error_hz",
                "cfnn_effect": 0.20,
                "boundary_score": 0.0,
            },
        ],
        "fano_a2_preregistered_success_audit": {
            "overall_status": "passes_at_some_main_budgets",
            "budgets": [32, 64, 128],
            "passed_budgets": [64, 128],
        },
    }

    ledger = summarize_ai4science_claim_ledger(package_summary)
    fano = next(row for row in ledger["claim_rows"] if row["claim_id"] == "fano_sparse_parameter_recovery")

    assert fano["claim_status"] == "supported_with_preregistered_budget_guardrails"
    assert fano["preregistered_passed_budgets"] == [64, 128]
    assert fano["preregistered_failed_budgets"] == [32]
    assert "64/128" in fano["reporting_rule"]


def test_ai4science_claim_ledger_records_fano_a3_robustness_guardrail_when_available():
    from ai4science_applicability_e import summarize_ai4science_claim_ledger

    package_summary = {
        "unified_summary": {"row_count": 4},
        "unified_rows": [
            {
                "domain": "Microwave Fano",
                "evidence_axis": "accuracy_effect",
                "effect_layer": "a2_vs_MLP",
                "metric": "quality_factor_relative_error",
                "cfnn_effect": 0.50,
                "boundary_score": 0.0,
            },
            {
                "domain": "Microwave Fano",
                "evidence_axis": "accuracy_effect",
                "effect_layer": "a2_vs_Local nested CF control",
                "metric": "f0_abs_error_hz",
                "cfnn_effect": 0.20,
                "boundary_score": 0.0,
            },
        ],
        "fano_a2_preregistered_success_audit": {
            "overall_status": "passes_at_some_main_budgets",
            "budgets": [32, 64, 128],
            "passed_budgets": [64, 128],
        },
        "fano_a3_preregistered_robustness_audit": {
            "overall_status": "does_not_pass_local_control_guardrail",
            "condition_count": 32,
            "passed_condition_count": 0,
            "mlp_only_or_partial_condition_count": 9,
        },
    }

    ledger = summarize_ai4science_claim_ledger(package_summary)
    fano = next(row for row in ledger["claim_rows"] if row["claim_id"] == "fano_sparse_parameter_recovery")

    assert fano["a3_robustness_guardrail_status"] == "does_not_pass_local_control_guardrail"
    assert fano["a3_passed_condition_count"] == 0
    assert fano["a3_mlp_only_or_partial_condition_count"] == 9
    assert "A3 robustness" in fano["reporting_rule"]


def test_ai4science_claim_ledger_records_fano_reliability_gate_when_available():
    from ai4science_applicability_e import summarize_ai4science_claim_ledger

    package_summary = {
        "unified_summary": {"row_count": 4},
        "unified_rows": [
            {
                "domain": "Microwave Fano",
                "evidence_axis": "accuracy_effect",
                "effect_layer": "a2_vs_MLP",
                "metric": "quality_factor_relative_error",
                "cfnn_effect": 0.50,
            },
            {
                "domain": "Microwave Fano",
                "evidence_axis": "accuracy_effect",
                "effect_layer": "a2_vs_Local nested CF control",
                "metric": "f0_abs_error_hz",
                "cfnn_effect": 0.20,
            },
        ],
        "fano_a2_preregistered_success_audit": {
            "overall_status": "passes_at_some_main_budgets",
            "budgets": [32, 64, 128],
            "passed_budgets": [64, 128],
        },
        "fano_reliability_claim_gate": {
            "claim_status": "ordinary_reliability_supported_highrisk_requires_escalation",
            "highrisk_curve_ids": ["undercoupled:r8:p03"],
            "post_policy_failure_rate": 0.2,
        },
    }

    ledger = summarize_ai4science_claim_ledger(package_summary)
    fano = next(row for row in ledger["claim_rows"] if row["claim_id"] == "fano_sparse_parameter_recovery")

    assert fano["reliability_claim_status"] == "ordinary_reliability_supported_highrisk_requires_escalation"
    assert fano["highrisk_curve_ids"] == ["undercoupled:r8:p03"]
    assert "ordinary Fano reliability" in fano["reporting_rule"]


def test_ai4science_claim_ledger_blocks_fano_claim_when_preregistered_gate_fails():
    from ai4science_applicability_e import summarize_ai4science_claim_ledger

    package_summary = {
        "unified_summary": {"row_count": 4},
        "unified_rows": [
            {
                "domain": "Microwave Fano",
                "evidence_axis": "accuracy_effect",
                "effect_layer": "a2_vs_MLP",
                "metric": "quality_factor_relative_error",
                "cfnn_effect": 0.50,
                "boundary_score": 0.0,
            },
            {
                "domain": "Microwave Fano",
                "evidence_axis": "accuracy_effect",
                "effect_layer": "a2_vs_Local nested CF control",
                "metric": "f0_abs_error_hz",
                "cfnn_effect": 0.20,
                "boundary_score": 0.0,
            },
        ],
        "fano_a2_preregistered_success_audit": {
            "overall_status": "does_not_pass_main_budget_gate",
            "budgets": [32, 64, 128],
            "passed_budgets": [],
        },
    }

    ledger = summarize_ai4science_claim_ledger(package_summary)
    fano = next(row for row in ledger["claim_rows"] if row["claim_id"] == "fano_sparse_parameter_recovery")

    assert fano["claim_status"] == "blocked_by_preregistered_budget_gate"
    assert ledger["claim_summary"]["cfnn_advantage_claim_count"] == 0


def test_experiment_plan_progress_counts_only_supported_advantage_packages():
    from ai4science_experiment_plan_audit import summarize_ai4science_experiment_plan_progress

    package_e_summary = {
        "unified_summary": {"row_count": 10},
        "fano_a2_preregistered_success_audit": {
            "overall_status": "does_not_pass_main_budget_gate",
            "budgets": [32, 64, 128],
            "passed_budgets": [],
        },
    }
    claim_ledger = {
        "claim_summary": {
            "cfnn_advantage_claim_count": 0,
            "blocked_or_boundary_claim_count": 4,
        },
        "claim_rows": [
            {
                "claim_id": "fano_sparse_parameter_recovery",
                "claim_status": "blocked_by_preregistered_budget_gate",
            }
        ],
    }
    aluminium_measurement_audit = {
        "decision_rows": [
            {
                "decision_id": "observed_interpolation_measurement_signal",
                "median_passed_modes": 5.0,
                "minimum_budget": 64,
            },
            {
                "decision_id": "same_schedule_cfnn_boundary",
                "median_passed_modes": 0.0,
            },
        ],
    }

    audit = summarize_ai4science_experiment_plan_progress(
        package_e_summary,
        claim_ledger,
        aluminium_measurement_audit,
    )
    rows = {row["package_id"]: row for row in audit["plan_rows"]}

    assert rows["A2"]["status"] == "blocked_by_preregistered_budget_gate"
    assert rows["Ledger"]["status"] == "zero_cfnn_advantage_four_boundaries"
    assert audit["summary"]["cfnn_advantage_package_count"] == 0


def test_aluminium_c_rows_record_modal_recovery_boundary_without_overclaiming_effect():
    from ai4science_applicability_e import aluminium_c_active_modal_boundary_rows

    aluminium_summary = {
        "summary_rows": [
            {
                "family": "CFNN",
                "sampling_strategy": "uniform",
                "budget": 24,
                "median_matched_mode_count": 0.0,
                "median_truth_mode_count": 5.0,
                "median_complex_nrmse": 1.0,
                "median_peak_window_complex_nrmse": 1.1,
                "record_count": 1,
            },
            {
                "family": "MLP",
                "sampling_strategy": "uniform",
                "budget": 24,
                "median_matched_mode_count": 2.0,
                "median_truth_mode_count": 5.0,
                "median_complex_nrmse": 0.9,
                "median_peak_window_complex_nrmse": 1.0,
                "record_count": 1,
            },
        ],
        "budget_threshold_rows": [
            {
                "family": "CFNN",
                "sampling_strategy": "uniform",
                "minimum_budget_meeting_modal_thresholds": None,
                "frequency_threshold_hz": 1.0,
                "width_relative_threshold": 0.1,
            }
        ],
    }

    rows = aluminium_c_active_modal_boundary_rows(aluminium_summary)

    modal = next(row for row in rows if row["metric"] == "matched_mode_fraction")
    threshold = next(row for row in rows if row["metric"] == "minimum_budget_meeting_modal_thresholds")
    assert modal["domain"] == "Aluminium FRF"
    assert modal["evidence_axis"] == "active_modal_boundary"
    assert modal["cfnn_effect"] is None
    assert modal["cfnn_metric"] == 0.0
    assert modal["boundary_score"] == 1.0
    assert modal["sampling_strategy"] == "uniform"
    assert modal["observation_budget"] == 24
    assert threshold["boundary_score"] == 1.0


def test_build_unified_applicability_rows_can_include_microstrip_and_aluminium():
    from ai4science_applicability_e import build_unified_applicability_rows

    rows = build_unified_applicability_rows(
        {"ranked_curve_rows": []},
        {"layered_effect_descriptor_rows": []},
        microstrip_summary={
            "summary_rows": [
                {
                    "experiment": "B2",
                    "scenario": "low_calibration_target",
                    "family": "CFNN",
                    "calibration_frequency_count": 4,
                    "median_quality_factor_relative_error": 0.05,
                },
                {
                    "experiment": "B2",
                    "scenario": "low_calibration_target",
                    "family": "MLP",
                    "calibration_frequency_count": 4,
                    "median_quality_factor_relative_error": 0.10,
                },
            ]
        },
        aluminium_summary={
            "summary_rows": [
                {
                    "family": "CFNN",
                    "sampling_strategy": "curvature",
                    "budget": 48,
                    "median_matched_mode_count": 1.0,
                    "median_truth_mode_count": 4.0,
                }
            ],
            "budget_threshold_rows": [],
        },
    )

    assert {row["domain"] for row in rows} == {"Microstrip Resonator", "Aluminium FRF"}
    assert any(row["evidence_axis"] == "device_generalization_effect" for row in rows)
    assert any(row["evidence_axis"] == "active_modal_boundary" for row in rows)


def test_descriptor_stratum_summary_links_spectral_structure_to_effects_and_boundaries():
    from ai4science_applicability_e import summarize_descriptor_strata

    rows = [
        {
            "domain": "Microwave Fano",
            "task_id": "fano:hard",
            "evidence_axis": "reliability_boundary",
            "effect_layer": "prediction_level_global_split_conformal",
            "phase_total_turns": 0.10,
            "max_curvature": 100.0,
            "boundary_score": 0.70,
            "cfnn_effect": None,
        },
        {
            "domain": "Microwave Fano",
            "task_id": "fano:ordinary",
            "evidence_axis": "reliability_boundary",
            "effect_layer": "prediction_level_global_split_conformal",
            "phase_total_turns": 0.95,
            "max_curvature": 20.0,
            "boundary_score": 0.01,
            "cfnn_effect": None,
        },
        {
            "domain": "EIS",
            "task_id": "eis:distributed",
            "evidence_axis": "accuracy_effect",
            "effect_layer": "physical_prior_only",
            "distributed_relaxation": True,
            "known_pole_count": 8,
            "relaxation_peak_count": 4,
            "cfnn_effect": -2.0,
            "boundary_score": 2.0,
        },
        {
            "domain": "EIS",
            "task_id": "eis:low_order",
            "evidence_axis": "accuracy_effect",
            "effect_layer": "neural_only",
            "distributed_relaxation": False,
            "known_pole_count": 2,
            "relaxation_peak_count": 1,
            "cfnn_effect": 0.25,
            "boundary_score": 0.0,
        },
    ]

    summary = summarize_descriptor_strata(rows)
    by_key = {
        (row["domain"], row["descriptor_stratum"], row["effect_layer"]): row
        for row in summary["stratum_rows"]
    }

    hard = by_key[("Microwave Fano", "weak_phase_high_curvature", "prediction_level_global_split_conformal")]
    ordinary = by_key[("Microwave Fano", "ordinary_resonant", "prediction_level_global_split_conformal")]
    distributed = by_key[("EIS", "distributed_relaxation", "physical_prior_only")]
    low_order = by_key[("EIS", "low_order_relaxation", "neural_only")]

    assert hard["row_count"] == 1
    assert hard["median_boundary_score"] == pytest.approx(0.70)
    assert ordinary["median_boundary_score"] == pytest.approx(0.01)
    assert distributed["negative_effect_count"] == 1
    assert distributed["median_cfnn_effect"] == pytest.approx(-2.0)
    assert low_order["positive_effect_count"] == 1
    assert low_order["median_cfnn_effect"] == pytest.approx(0.25)
    assert summary["fano_phase_turn_threshold"] == pytest.approx(0.27)
    assert summary["fano_curvature_threshold"] == pytest.approx(84.0)
