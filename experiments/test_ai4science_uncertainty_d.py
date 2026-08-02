from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_response_interval_metrics_report_coverage_and_dangerous_point_failures():
    from ai4science_uncertainty_d import response_interval_metrics

    frequency = np.linspace(0.0, 10.0, 11)
    truth_complex = np.sin(frequency) + 1j * np.cos(frequency)
    truth = np.column_stack((truth_complex.real, truth_complex.imag))
    predictions = []
    for offset in (-0.2, 0.0, 0.2):
        member = truth_complex + offset
        predictions.append(np.column_stack((member.real, member.imag)))
    predictions = np.asarray(predictions)
    predictions[:, 3:8, 0] = truth[3:8, 0] + 5.0
    predictions[:, 3:8, 1] = truth[3:8, 1] + 5.0

    metrics = response_interval_metrics(
        truth,
        predictions,
        frequency_hz=frequency,
        peak_center_hz=5.0,
        peak_width_hz=1.0,
        confidence=0.80,
    )

    assert metrics["status"] == "ok"
    assert metrics["member_count"] == 3
    assert 0.0 <= metrics["joint_coverage"] < 1.0
    assert metrics["peak_window_joint_coverage"] < 1.0
    assert metrics["dangerous_point_failure_count"] >= 1
    assert metrics["error_uncertainty_rank_correlation"] is not None
    assert metrics["nominal_joint_coverage_scale"] > 1.0
    assert metrics["peak_window_nominal_joint_coverage_scale"] > 1.0


def test_parameter_interval_metrics_report_coverage_and_dangerous_failures():
    from ai4science_uncertainty_d import parameter_interval_metrics

    reference = {
        "f0_hz": 100.0,
        "quality_factor": 50.0,
        "q": -1.0,
    }
    predictions = [
        {"f0_hz": 100.1, "quality_factor": 49.8, "q": -1.1},
        {"f0_hz": 100.2, "quality_factor": 50.0, "q": -1.0},
        {"f0_hz": 100.3, "quality_factor": 50.2, "q": -0.9},
    ]

    metrics = parameter_interval_metrics(
        reference,
        predictions,
        parameters=("f0_hz", "quality_factor", "q"),
        confidence=0.80,
        danger_thresholds={
            "f0_hz": {"max_interval_width": 0.5, "max_abs_error": 0.05},
            "quality_factor": {"max_interval_width": 1.0, "max_abs_error": 1.0},
            "q": {"max_interval_width": 0.5, "max_abs_error": 0.5},
        },
    )

    assert metrics["status"] == "ok"
    assert metrics["member_count"] == 3
    assert metrics["parameter_count"] == 3
    assert metrics["covered_count"] == 2
    assert metrics["dangerous_parameter_failure_count"] == 1
    f0 = next(row for row in metrics["parameter_rows"] if row["parameter"] == "f0_hz")
    assert f0["covered"] is False
    assert f0["dangerous_failure"] is True


def test_validation_calibrated_response_interval_uses_validation_scale_on_test_points():
    from ai4science_uncertainty_d import (
        response_interval_metrics,
        validation_calibrated_response_interval_metrics,
    )

    x = np.linspace(0.0, 1.0, 20)
    truth_complex = x + 1j * (1.0 - x)
    truth = np.column_stack((truth_complex.real, truth_complex.imag))
    test_predictions = []
    validation_predictions = []
    for offset in (-0.05, 0.0, 0.05):
        test_member = truth_complex + 0.2 + offset
        validation_member = truth_complex + 0.2 + offset
        test_predictions.append(np.column_stack((test_member.real, test_member.imag)))
        validation_predictions.append(np.column_stack((validation_member.real, validation_member.imag)))
    test_predictions = np.asarray(test_predictions)
    validation_predictions = np.asarray(validation_predictions)

    uncalibrated = response_interval_metrics(truth, test_predictions, confidence=0.80)
    calibrated = validation_calibrated_response_interval_metrics(
        truth,
        test_predictions,
        calibration_truth=truth,
        calibration_predictions=validation_predictions,
        confidence=0.80,
    )

    assert calibrated["status"] == "ok"
    assert calibrated["calibration_scale"] > 1.0
    assert calibrated["joint_coverage"] > uncalibrated["joint_coverage"]
    assert calibrated["calibration_point_count"] == len(truth)


def test_external_scaled_response_interval_uses_supplied_cross_curve_scale():
    from ai4science_uncertainty_d import (
        calibration_scale_from_predictions,
        scaled_response_interval_metrics,
    )

    x = np.linspace(0.0, 1.0, 20)
    calibration_truth_complex = x + 1j * (1.0 - x)
    calibration_truth = np.column_stack((calibration_truth_complex.real, calibration_truth_complex.imag))
    calibration_predictions = []
    for offset in (-0.05, 0.0, 0.05):
        member = calibration_truth_complex + 0.2 + offset
        calibration_predictions.append(np.column_stack((member.real, member.imag)))
    calibration_predictions = np.asarray(calibration_predictions)

    scale = calibration_scale_from_predictions(
        calibration_truth,
        calibration_predictions,
        confidence=0.80,
    )
    assert scale > 1.0

    target_truth_complex = calibration_truth_complex
    target_truth = np.column_stack((target_truth_complex.real, target_truth_complex.imag))
    target_predictions = []
    for offset in (-0.05, 0.0, 0.05):
        member = target_truth_complex + 0.2 + offset
        target_predictions.append(np.column_stack((member.real, member.imag)))
    target_predictions = np.asarray(target_predictions)

    metrics = scaled_response_interval_metrics(
        target_truth,
        target_predictions,
        scale=scale,
        confidence=0.80,
    )

    assert metrics["status"] == "ok"
    assert metrics["calibration_scale"] == scale
    assert metrics["calibration_source"] == "external"
    assert metrics["joint_coverage"] > 0.75
    assert metrics["calibration_point_count"] is None


def test_runner_summary_keeps_uncertainty_fields_for_each_family():
    from run_ai4science_fano_d_uncertainty_pilot import summarize_family_ensembles

    rows = [
        {
            "family": "CFNN",
            "curve_id": "curve:1",
            "observation_budget": 64,
            "split_seed": 719,
            "response_interval": {
                "joint_coverage": 0.9,
                "peak_window_joint_coverage": 0.8,
                "dangerous_point_failure_rate": 0.1,
            },
            "parameter_interval": {
                "coverage_rate": 0.75,
                "dangerous_parameter_failure_rate": 0.25,
            },
            "validation_calibrated_response_interval": {
                "joint_coverage": 0.95,
                "calibration_scale": 3.0,
            },
            "cross_curve_calibrated_response_interval": {
                "joint_coverage": 0.85,
                "calibration_scale": 4.0,
                "calibration_curve_count": 1,
            },
        }
    ]

    summary = summarize_family_ensembles(rows)

    assert summary["ensemble_count"] == 1
    assert summary["family_rows"][0]["family"] == "CFNN"
    assert summary["family_rows"][0]["median_joint_coverage"] == 0.9
    assert summary["family_rows"][0]["median_parameter_coverage_rate"] == 0.75
    assert summary["family_rows"][0]["median_validation_calibrated_joint_coverage"] == 0.95
    assert summary["family_rows"][0]["median_validation_calibration_scale"] == 3.0
    assert summary["family_rows"][0]["median_cross_curve_calibrated_joint_coverage"] == 0.85
    assert summary["family_rows"][0]["median_cross_curve_calibration_scale"] == 4.0


def test_runner_attaches_leave_one_curve_calibration_without_target_curve_leakage():
    from run_ai4science_fano_d_uncertainty_pilot import attach_cross_curve_calibrated_response_intervals

    x = np.linspace(0.0, 1.0, 20)
    truth_complex = x + 1j * (1.0 - x)
    truth = np.column_stack((truth_complex.real, truth_complex.imag)).astype(np.float32)

    def predictions(bias):
        rows = []
        for offset in (-0.05, 0.0, 0.05):
            member = truth_complex + bias + offset
            rows.append(np.column_stack((member.real, member.imag)))
        return np.asarray(rows, dtype=np.float32)

    ensembles = [
        {"family": "CFNN", "curve_id": "curve:a", "observation_budget": 64},
        {"family": "CFNN", "curve_id": "curve:b", "observation_budget": 64},
    ]
    payloads = [
        {
            "family": "CFNN",
            "curve_id": "curve:a",
            "observation_budget": 64,
            "truth": truth,
            "predictions": predictions(0.2),
            "validation_truth": truth,
            "validation_predictions": predictions(0.8),
            "frequency_hz": x,
            "peak_center_hz": 0.5,
            "peak_width_hz": 0.1,
        },
        {
            "family": "CFNN",
            "curve_id": "curve:b",
            "observation_budget": 64,
            "truth": truth,
            "predictions": predictions(0.2),
            "validation_truth": truth,
            "validation_predictions": predictions(0.2),
            "frequency_hz": x,
            "peak_center_hz": 0.5,
            "peak_width_hz": 0.1,
        },
    ]

    attach_cross_curve_calibrated_response_intervals(ensembles, payloads, confidence=0.80)

    curve_a = ensembles[0]["cross_curve_calibrated_response_interval"]
    curve_b = ensembles[1]["cross_curve_calibrated_response_interval"]
    assert curve_a["status"] == "ok"
    assert curve_b["status"] == "ok"
    assert curve_a["calibration_curve_ids"] == ["curve:b"]
    assert curve_b["calibration_curve_ids"] == ["curve:a"]
    assert curve_a["calibration_scale"] < curve_b["calibration_scale"]


def test_runner_attaches_prediction_level_split_conformal_intervals():
    from run_ai4science_fano_d_uncertainty_pilot import attach_split_conformal_response_intervals

    x = np.linspace(0.0, 1.0, 20)
    truth_complex = x + 1j * (1.0 - x)
    truth = np.column_stack((truth_complex.real, truth_complex.imag)).astype(np.float32)

    def predictions(bias):
        rows = []
        for offset in (-0.05, 0.0, 0.05):
            member = truth_complex + bias + offset
            rows.append(np.column_stack((member.real, member.imag)))
        return np.asarray(rows, dtype=np.float32)

    ensembles = [
        {"family": "CFNN", "curve_id": "undercoupled:a", "observation_budget": 64},
        {"family": "CFNN", "curve_id": "undercoupled:b", "observation_budget": 64},
        {"family": "CFNN", "curve_id": "overcoupled:c", "observation_budget": 64},
    ]
    payloads = [
        {
            "family": "CFNN",
            "curve_id": "undercoupled:a",
            "observation_budget": 64,
            "truth": truth,
            "predictions": predictions(0.2),
            "validation_truth": truth,
            "validation_predictions": predictions(0.2),
            "frequency_hz": x,
            "peak_center_hz": 0.5,
            "peak_width_hz": 0.1,
        },
        {
            "family": "CFNN",
            "curve_id": "undercoupled:b",
            "observation_budget": 64,
            "truth": truth,
            "predictions": predictions(0.2),
            "validation_truth": truth,
            "validation_predictions": predictions(0.4),
            "frequency_hz": x,
            "peak_center_hz": 0.5,
            "peak_width_hz": 0.1,
        },
        {
            "family": "CFNN",
            "curve_id": "overcoupled:c",
            "observation_budget": 64,
            "truth": truth,
            "predictions": predictions(0.2),
            "validation_truth": truth,
            "validation_predictions": predictions(0.8),
            "frequency_hz": x,
            "peak_center_hz": 0.5,
            "peak_width_hz": 0.1,
        },
    ]

    attach_split_conformal_response_intervals(
        ensembles,
        payloads,
        confidence=0.80,
        min_stratum_count=1,
    )

    target = ensembles[0]
    global_metrics = target["global_split_conformal_response_interval"]
    stratified_metrics = target["curve_stratified_split_conformal_response_interval"]

    assert global_metrics["status"] == "ok"
    assert global_metrics["calibration_source"] == "global_split_conformal"
    assert set(global_metrics["calibration_curve_ids"]) == {"undercoupled:b", "overcoupled:c"}
    assert global_metrics["calibration_scale"] > stratified_metrics["calibration_scale"]
    assert stratified_metrics["calibration_source"] == "curve_stratified_split_conformal"
    assert stratified_metrics["calibration_curve_ids"] == ["undercoupled:b"]
    assert 0.0 <= stratified_metrics["joint_coverage"] <= 1.0


def test_runner_observation_budgets_are_explicit_or_backward_compatible():
    from run_ai4science_fano_d_uncertainty_pilot import observation_budgets_from_args

    assert observation_budgets_from_args(SimpleNamespace(observation_budget=64, observation_budgets=None)) == [64]
    assert observation_budgets_from_args(SimpleNamespace(observation_budget=64, observation_budgets=[128, 64, 128])) == [128, 64]


def test_split_conformal_scale_uses_finite_sample_quantile():
    from ai4science_uncertainty_d import split_conformal_scale

    # For n=4 and 80% confidence, ceil((n+1)*0.8)=4, so the conservative
    # split-conformal scale is the maximum calibration score.
    scale = split_conformal_scale([1.0, 2.0, 3.0, 10.0], confidence=0.80)

    assert scale == 10.0


def test_curve_stratified_conformal_scale_excludes_target_and_falls_back_to_global():
    from ai4science_uncertainty_d import curve_stratified_conformal_scale

    calibration_rows = [
        {"curve_id": "a", "scale": 1.0, "stratum": "low"},
        {"curve_id": "b", "scale": 2.0, "stratum": "low"},
        {"curve_id": "c", "scale": 8.0, "stratum": "high"},
    ]

    same_stratum = curve_stratified_conformal_scale(
        calibration_rows,
        target_curve_id="a",
        target_stratum="low",
        confidence=0.80,
        min_stratum_count=1,
    )
    fallback = curve_stratified_conformal_scale(
        calibration_rows,
        target_curve_id="c",
        target_stratum="high",
        confidence=0.80,
        min_stratum_count=2,
    )

    assert same_stratum["calibration_curve_ids"] == ["b"]
    assert same_stratum["calibration_source"] == "curve_stratified_split_conformal"
    assert same_stratum["scale"] == 2.0
    assert fallback["calibration_source"] == "global_split_conformal_fallback"
    assert set(fallback["calibration_curve_ids"]) == {"a", "b"}
    assert fallback["scale"] == 2.0
