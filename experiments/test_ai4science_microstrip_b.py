from pathlib import Path
import sys

import numpy as np
import pytest


RAW_ROOT = Path(__file__).resolve().parent / "downstream_data" / "raw"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_b1_holdout_splits_complete_microstrip_conditions():
    from ai4science_microstrip_b import make_b1_condition_split
    from downstream_protocol import load_microstrip_resonator

    curves = load_microstrip_resonator(RAW_ROOT)
    split = make_b1_condition_split(
        curves,
        scenario="humidity_holdout",
        sample_id="s34",
        holdout_temperature_c=None,
        holdout_relative_humidity_percent=50.0,
        train_frequency_count=16,
        validation_frequency_count=24,
        seed=233,
    )

    assert split.metadata["split_mode"] == "microstrip_b1_condition_holdout"
    assert split.x_train.shape[1] == 3
    assert split.x_test.shape[0] == split.y_test.shape[0]
    assert not (
        set(split.metadata["train_curve_ids"])
        & set(split.metadata["validation_curve_ids"])
        | set(split.metadata["train_curve_ids"])
        & set(split.metadata["test_curve_ids"])
        | set(split.metadata["validation_curve_ids"])
        & set(split.metadata["test_curve_ids"])
    )
    assert {
        curve_id.split(":cycle:")[0]
        for curve_id in split.metadata["test_curve_ids"]
    } == {"s34"}
    assert all(
        condition["relative_humidity_percent"] == 50.0
        for condition in split.metadata["test_conditions"]
    )


def test_b2_calibration_adds_sparse_target_condition_points_without_test_leakage():
    from ai4science_microstrip_b import make_b2_calibration_split
    from downstream_protocol import load_microstrip_resonator

    curves = load_microstrip_resonator(RAW_ROOT)
    split = make_b2_calibration_split(
        curves,
        sample_id="s34",
        target_temperature_c=25.0,
        target_relative_humidity_percent=50.0,
        calibration_frequency_count=8,
        support_frequency_count=16,
        validation_frequency_count=24,
        seed=233,
    )

    assert split.metadata["split_mode"] == "microstrip_b2_low_calibration_adaptation"
    assert split.x_train.shape[1] == 3
    assert len(split.metadata["calibration_point_ids"]) == 8 * len(
        split.metadata["test_curve_ids"]
    )
    assert set(split.metadata["calibration_point_ids"]).issubset(
        set(split.metadata["train_point_ids"])
    )
    calibration_curve_ids = {
        point_id.split(":frequency:")[0]
        for point_id in split.metadata["calibration_point_ids"]
    }
    assert calibration_curve_ids == set(split.metadata["test_curve_ids"])
    assert np.isfinite(split.x_train).all()
    assert np.isfinite(split.y_train).all()


def test_b2_missed_band_calibration_excludes_target_resonance_band():
    from ai4science_microstrip_b import (
        extract_microstrip_events,
        make_b2_calibration_split,
    )
    from downstream_protocol import load_microstrip_resonator

    curves = load_microstrip_resonator(RAW_ROOT)
    split = make_b2_calibration_split(
        curves,
        sample_id="s34",
        target_temperature_c=25.0,
        target_relative_humidity_percent=50.0,
        calibration_frequency_count=4,
        support_frequency_count=16,
        validation_frequency_count=24,
        seed=233,
        calibration_sampling_mode="missed_band",
        calibration_excluded_band_linewidths=1.0,
    )
    by_curve = {curve.curve_id: curve for curve in curves}

    assert split.metadata["calibration_sampling_mode"] == "missed_band"
    assert len(split.metadata["calibration_point_ids"]) == 4 * len(
        split.metadata["test_curve_ids"]
    )
    for point_id in split.metadata["calibration_point_ids"]:
        curve_id, raw_index = point_id.rsplit(":frequency:", 1)
        curve = by_curve[curve_id]
        events = extract_microstrip_events(curve.frequency, curve.response)
        f0 = float(events["resonance_frequency_hz"])
        linewidth = float(events["linewidth_hz"])
        calibration_frequency = float(curve.frequency[int(raw_index)])
        assert abs(calibration_frequency - f0) > linewidth


def test_microstrip_b2_job_key_distinguishes_calibration_sampling_mode():
    from run_ai4science_microstrip_b_pilot import _job_key

    base = {
        "experiment": "B2",
        "scenario": "low_calibration_target_25c_50rh",
        "sample_id": "s34",
        "family": "CFNN",
        "split_seed": 233,
        "init_seed": 1009,
        "calibration_frequency_count": 4,
    }
    assert _job_key({**base, "calibration_sampling_mode": "random"}) != _job_key({
        **base,
        "calibration_sampling_mode": "missed_band",
    })


def test_event_identifiability_flags_require_q_and_nonboundary_f0():
    from ai4science_microstrip_b import event_identifiability_flags

    identifiable = event_identifiability_flags({
        "resonance_frequency_hz": 1.95e9,
        "quality_factor": 8.0,
    })
    assert identifiable["truth_q_available"] is True
    assert identifiable["truth_high_frequency_boundary_notch"] is False
    assert identifiable["truth_event_identifiable"] is True

    no_q = event_identifiability_flags({
        "resonance_frequency_hz": 1.95e9,
        "quality_factor": None,
    })
    assert no_q["truth_event_identifiable"] is False

    boundary = event_identifiability_flags({
        "resonance_frequency_hz": 2.25e9,
        "quality_factor": 8.0,
    })
    assert boundary["truth_high_frequency_boundary_notch"] is True
    assert boundary["truth_event_identifiable"] is False


def test_microstrip_summary_separates_all_curve_and_identifiable_event_metrics():
    from run_ai4science_microstrip_b_pilot import _summarize

    records = [{
        "experiment": "B1",
        "scenario": "temperature_holdout",
        "family": "CFNN",
        "calibration_frequency_count": None,
        "status": "ok",
        "curve_metrics": [
            {
                "complex_nrmse": 0.2,
                "peak_window_complex_nrmse": 0.3,
                "resonance_frequency_abs_error_hz": 10.0,
                "quality_factor_relative_error": 0.1,
                "phase_transition_abs_error_hz": 20.0,
                "local_phase_transition_abs_error_hz": 12.0,
                "truth_event_identifiable": True,
            },
            {
                "complex_nrmse": 0.4,
                "peak_window_complex_nrmse": 0.5,
                "resonance_frequency_abs_error_hz": 999.0,
                "quality_factor_relative_error": 9.9,
                "phase_transition_abs_error_hz": 999.0,
                "local_phase_transition_abs_error_hz": 888.0,
                "truth_event_identifiable": False,
            },
        ],
    }]

    row = _summarize(records)["summary_rows"][0]
    assert row["curve_metric_count"] == 2
    assert row["event_identifiable_count"] == 1
    assert row["event_identifiable_rate"] == 0.5
    assert row["median_all_curve_complex_nrmse"] == pytest.approx(0.3)
    assert row["median_all_curve_peak_window_complex_nrmse"] == pytest.approx(0.4)
    assert row["median_identifiable_resonance_frequency_abs_error_hz"] == 10.0
    assert row["median_identifiable_quality_factor_relative_error"] == 0.1
    assert row["median_identifiable_local_phase_transition_abs_error_hz"] == 12.0
    assert row["median_local_phase_transition_abs_error_hz"] == pytest.approx(450.0)


def test_microstrip_instance_effect_audit_pairs_same_curve_and_seed():
    from ai4science_microstrip_b import microstrip_instance_effect_rows

    records = [
        {
            "status": "ok",
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "sample_id": "s31",
            "family": "CFNN",
            "split_seed": 1,
            "init_seed": 7,
            "calibration_frequency_count": None,
            "curve_metrics": [
                {
                    "curve_id": "s31:cycle:001",
                    "quality_factor_relative_error": 0.05,
                    "phase_transition_abs_error_hz": 10.0,
                    "truth_event_identifiable": True,
                },
                {
                    "curve_id": "s31:cycle:002",
                    "quality_factor_relative_error": 0.20,
                    "phase_transition_abs_error_hz": 40.0,
                    "truth_event_identifiable": True,
                },
            ],
        },
        {
            "status": "ok",
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "sample_id": "s31",
            "family": "MLP",
            "split_seed": 1,
            "init_seed": 7,
            "calibration_frequency_count": None,
            "curve_metrics": [
                {
                    "curve_id": "s31:cycle:001",
                    "quality_factor_relative_error": 0.10,
                    "phase_transition_abs_error_hz": 20.0,
                    "truth_event_identifiable": True,
                },
                {
                    "curve_id": "s31:cycle:002",
                    "quality_factor_relative_error": 0.10,
                    "phase_transition_abs_error_hz": 20.0,
                    "truth_event_identifiable": True,
                },
            ],
        },
    ]

    rows = microstrip_instance_effect_rows(
        records,
        comparison_families={"MLP"},
        metrics={"quality_factor_relative_error", "phase_transition_abs_error_hz"},
    )

    assert len(rows) == 4
    q_positive = next(
        row for row in rows
        if row["curve_id"] == "s31:cycle:001" and row["metric"] == "quality_factor_relative_error"
    )
    phase_negative = next(
        row for row in rows
        if row["curve_id"] == "s31:cycle:002" and row["metric"] == "phase_transition_abs_error_hz"
    )
    assert q_positive["cfnn_effect"] == pytest.approx(0.5)
    assert q_positive["cfnn_wins"] is True
    assert q_positive["truth_event_identifiable"] is True
    assert phase_negative["cfnn_effect"] == pytest.approx(-1.0)
    assert phase_negative["cfnn_wins"] is False


def test_microstrip_instance_effect_summary_reports_win_rate_and_median_effect():
    from ai4science_microstrip_b import summarize_microstrip_instance_effects

    rows = [
        {
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "comparison_family": "MLP",
            "metric": "quality_factor_relative_error",
            "cfnn_effect": 0.50,
            "cfnn_wins": True,
        },
        {
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "comparison_family": "MLP",
            "metric": "quality_factor_relative_error",
            "cfnn_effect": -0.25,
            "cfnn_wins": False,
        },
        {
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "comparison_family": "MLP",
            "metric": "quality_factor_relative_error",
            "cfnn_effect": 0.10,
            "cfnn_wins": True,
        },
    ]

    summary = summarize_microstrip_instance_effects(rows)

    assert len(summary) == 1
    row = summary[0]
    assert row["pair_count"] == 3
    assert row["cfnn_win_rate"] == pytest.approx(2 / 3)
    assert row["median_cfnn_effect"] == pytest.approx(0.10)


def test_microstrip_phase_geometry_rows_flag_phase_far_from_resonance():
    from ai4science_microstrip_b import (
        microstrip_phase_geometry_rows,
        summarize_microstrip_phase_geometry,
    )

    records = [{
        "status": "ok",
        "sample_id": "s31",
        "curve_metrics": [
            {
                "curve_id": "s31:cycle:001",
                "conditions": {"temperature_c": 25.0},
                "truth_events": {
                    "resonance_frequency_hz": 2.0e9,
                    "phase_transition_hz": 2.01e9,
                    "linewidth_hz": 100e6,
                    "quality_factor": 20.0,
                },
                "truth_event_identifiable": True,
            },
            {
                "curve_id": "s31:cycle:002",
                "conditions": {"temperature_c": 35.0},
                "truth_events": {
                    "resonance_frequency_hz": 1.9e9,
                    "phase_transition_hz": 2.2e9,
                    "linewidth_hz": 100e6,
                    "quality_factor": 19.0,
                },
                "truth_event_identifiable": True,
            },
        ],
    }]

    rows = microstrip_phase_geometry_rows(records, far_threshold_linewidths=1.0)
    summary = summarize_microstrip_phase_geometry(rows)

    assert len(rows) == 2
    close = next(row for row in rows if row["curve_id"] == "s31:cycle:001")
    far = next(row for row in rows if row["curve_id"] == "s31:cycle:002")
    assert close["phase_f0_gap_linewidth_ratio"] == pytest.approx(0.1)
    assert close["phase_far_from_resonance"] is False
    assert far["phase_f0_gap_linewidth_ratio"] == pytest.approx(3.0)
    assert far["phase_far_from_resonance"] is True
    assert summary["curve_count"] == 2
    assert summary["far_phase_count"] == 1
    assert summary["far_phase_fraction"] == pytest.approx(0.5)


def test_microstrip_phase_effect_instability_summary_counts_extreme_flips():
    from ai4science_microstrip_b import summarize_microstrip_phase_effect_instability

    rows = [
        {
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "comparison_family": "MLP",
            "metric": "phase_transition_abs_error_hz",
            "cfnn_effect": 0.95,
            "cfnn_wins": True,
        },
        {
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "comparison_family": "MLP",
            "metric": "phase_transition_abs_error_hz",
            "cfnn_effect": -3.0,
            "cfnn_wins": False,
        },
        {
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "comparison_family": "MLP",
            "metric": "quality_factor_relative_error",
            "cfnn_effect": 0.5,
            "cfnn_wins": True,
        },
    ]

    summary = summarize_microstrip_phase_effect_instability(rows, extreme_effect_abs=1.0)

    assert len(summary) == 1
    row = summary[0]
    assert row["pair_count"] == 2
    assert row["positive_extreme_count"] == 0
    assert row["negative_extreme_count"] == 1
    assert row["cfnn_win_rate"] == pytest.approx(0.5)


def test_local_phase_event_search_stays_near_resonance_window():
    from ai4science_microstrip_b import local_phase_transition_hz

    frequency = np.linspace(1.6e9, 2.4e9, 1001)
    # One modest phase turn near the resonance and a sharper unrelated turn far away.
    phase = 0.2 * np.tanh((frequency - 1.85e9) / 0.03e9) + 1.0 * np.tanh((frequency - 2.23e9) / 0.01e9)
    response = np.column_stack((np.cos(phase), np.sin(phase)))

    local = local_phase_transition_hz(
        frequency,
        response,
        resonance_frequency_hz=1.85e9,
        linewidth_hz=0.18e9,
        window_linewidths=1.0,
    )

    assert abs(local - 1.85e9) < 0.05e9


def test_microstrip_response_metrics_record_local_phase_events_for_future_audits():
    from ai4science_microstrip_b import microstrip_response_metrics

    frequency = np.linspace(1.6e9, 2.4e9, 1001)
    magnitude = 1.0 - 0.5 / (1.0 + ((frequency - 1.85e9) / 0.09e9) ** 2)
    truth_phase = (
        0.2 * np.tanh((frequency - 1.85e9) / 0.03e9)
        + 1.0 * np.tanh((frequency - 2.23e9) / 0.01e9)
    )
    pred_phase = (
        0.2 * np.tanh((frequency - 1.86e9) / 0.03e9)
        + 1.0 * np.tanh((frequency - 2.23e9) / 0.01e9)
    )
    truth = np.column_stack((magnitude * np.cos(truth_phase), magnitude * np.sin(truth_phase)))
    prediction = np.column_stack((magnitude * np.cos(pred_phase), magnitude * np.sin(pred_phase)))

    metrics = microstrip_response_metrics(frequency, truth, prediction)

    assert "local_truth_events" in metrics
    assert "local_predicted_events" in metrics
    assert "local_phase_transition_abs_error_hz" in metrics
    assert metrics["local_truth_events"]["phase_transition_rule"] == (
        "max_phase_slope_within_1_linewidths_of_f0"
    )
    assert abs(metrics["local_truth_events"]["phase_transition_hz"] - 1.85e9) < 0.05e9
    assert metrics["local_phase_transition_abs_error_hz"] < 0.05e9


def test_microstrip_local_phase_geometry_rows_compare_global_and_local_gaps():
    from types import SimpleNamespace

    from ai4science_microstrip_b import microstrip_local_phase_geometry_rows

    frequency = np.linspace(1.6e9, 2.4e9, 1001)
    magnitude = 1.0 - 0.5 / (1.0 + ((frequency - 1.85e9) / 0.09e9) ** 2)
    phase = 0.2 * np.tanh((frequency - 1.85e9) / 0.03e9) + 1.0 * np.tanh((frequency - 2.23e9) / 0.01e9)
    response = np.column_stack((magnitude * np.cos(phase), magnitude * np.sin(phase)))
    curve = SimpleNamespace(
        curve_id="synthetic:phase",
        frequency=frequency,
        response=response,
        conditions={"sample_id": "synth"},
    )

    rows = microstrip_local_phase_geometry_rows([curve], window_linewidths=1.0)

    assert len(rows) == 1
    row = rows[0]
    assert row["global_phase_f0_gap_linewidth_ratio"] > 1.0
    assert row["local_phase_f0_gap_linewidth_ratio"] < 0.5
    assert row["local_improves_gap"] is True


def test_microstrip_phase_effect_geometry_strata_join_phase_effects_to_truth_geometry():
    from ai4science_microstrip_b import microstrip_phase_effect_geometry_strata_rows

    effect_rows = [
        {
            "curve_id": "s31:cycle:001",
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "calibration_frequency_count": None,
            "comparison_family": "MLP",
            "metric": "phase_transition_abs_error_hz",
            "cfnn_effect": -2.0,
            "cfnn_wins": False,
        },
        {
            "curve_id": "s31:cycle:001",
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "comparison_family": "MLP",
            "metric": "quality_factor_relative_error",
            "cfnn_effect": 0.5,
            "cfnn_wins": True,
        },
    ]
    geometry_rows = [{
        "curve_id": "s31:cycle:001",
        "global_phase_far_from_resonance": True,
        "local_phase_far_from_resonance": False,
        "global_phase_f0_gap_linewidth_ratio": 1.5,
        "local_phase_f0_gap_linewidth_ratio": 0.6,
        "local_improves_gap": True,
    }]

    rows = microstrip_phase_effect_geometry_strata_rows(effect_rows, geometry_rows)

    assert len(rows) == 1
    row = rows[0]
    assert row["truth_phase_geometry_stratum"] == "global_far_local_near"
    assert row["global_phase_f0_gap_linewidth_ratio"] == pytest.approx(1.5)
    assert row["local_phase_f0_gap_linewidth_ratio"] == pytest.approx(0.6)
    assert row["cfnn_effect"] == pytest.approx(-2.0)


def test_summarize_microstrip_phase_effect_geometry_strata_counts_extremes():
    from ai4science_microstrip_b import summarize_microstrip_phase_effect_geometry_strata

    rows = [
        {
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "calibration_frequency_count": None,
            "comparison_family": "MLP",
            "truth_phase_geometry_stratum": "global_far_local_near",
            "cfnn_effect": -2.0,
            "cfnn_wins": False,
        },
        {
            "experiment": "B1",
            "scenario": "humidity_holdout",
            "calibration_frequency_count": None,
            "comparison_family": "MLP",
            "truth_phase_geometry_stratum": "global_far_local_near",
            "cfnn_effect": 0.5,
            "cfnn_wins": True,
        },
    ]

    summary = summarize_microstrip_phase_effect_geometry_strata(rows, extreme_effect_abs=1.0)

    assert len(summary) == 1
    row = summary[0]
    assert row["pair_count"] == 2
    assert row["cfnn_win_rate"] == pytest.approx(0.5)
    assert row["median_cfnn_effect"] == pytest.approx(-0.75)
    assert row["negative_extreme_count"] == 1


def test_microstrip_b2_calibration_coverage_rows_measure_distance_to_resonance_band():
    from ai4science_microstrip_b import microstrip_b2_calibration_coverage_rows
    from downstream_protocol import MeasuredCurve

    frequency = np.array([1.80e9, 1.90e9, 2.00e9, 2.10e9, 2.20e9])
    response = np.column_stack((np.ones_like(frequency), np.zeros_like(frequency))).astype(np.float32)
    curve = MeasuredCurve(
        curve_id="s31:cycle:001",
        frequency=frequency,
        response=response,
        conditions={"sample_id": "s31"},
        source_files=("synthetic",),
        metadata={},
    )
    record = {
        "status": "ok",
        "experiment": "B2",
        "scenario": "low_calibration_target_25c_50rh",
        "sample_id": "s31",
        "family": "CFNN",
        "split_seed": 233,
        "init_seed": 1009,
        "calibration_frequency_count": 2,
        "dataset": {
            "calibration_point_ids": [
                "s31:cycle:001:frequency:2",
                "s31:cycle:001:frequency:4",
            ]
        },
        "curve_metrics": [
            {
                "curve_id": "s31:cycle:001",
                "truth_events": {
                    "resonance_frequency_hz": 2.00e9,
                    "linewidth_hz": 2.00e8,
                },
                "truth_event_identifiable": True,
            }
        ],
    }

    rows = microstrip_b2_calibration_coverage_rows([record], [curve])

    assert len(rows) == 1
    row = rows[0]
    assert row["calibration_point_count_for_curve"] == 2
    assert row["calibration_points_within_one_linewidth"] == 2
    assert row["calibration_points_within_half_linewidth"] == 1
    assert row["nearest_calibration_distance_linewidth_ratio"] == pytest.approx(0.0)
    assert row["calibration_coverage_stratum"] == "resonance_band_covered"


def test_microstrip_b2_calibration_coverage_effect_summary_stratifies_q_effects():
    from ai4science_microstrip_b import (
        microstrip_b2_calibration_coverage_effect_rows,
        summarize_microstrip_b2_calibration_coverage_effects,
    )

    effect_rows = [
        {
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "sample_id": "s31",
            "calibration_frequency_count": 4,
            "split_seed": 233,
            "init_seed": 1009,
            "curve_id": "s31:cycle:001",
            "comparison_family": "Local nested CF control",
            "metric": "quality_factor_relative_error",
            "cfnn_effect": 0.50,
            "cfnn_wins": True,
        },
        {
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "sample_id": "s31",
            "calibration_frequency_count": 4,
            "split_seed": 233,
            "init_seed": 1009,
            "curve_id": "s31:cycle:002",
            "comparison_family": "Local nested CF control",
            "metric": "quality_factor_relative_error",
            "cfnn_effect": -0.25,
            "cfnn_wins": False,
        },
    ]
    coverage_rows = [
        {
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "sample_id": "s31",
            "calibration_frequency_count": 4,
            "split_seed": 233,
            "init_seed": 1009,
            "curve_id": "s31:cycle:001",
            "calibration_coverage_stratum": "resonance_band_covered",
            "nearest_calibration_distance_linewidth_ratio": 0.20,
            "calibration_points_within_one_linewidth": 1,
        },
        {
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "sample_id": "s31",
            "calibration_frequency_count": 4,
            "split_seed": 233,
            "init_seed": 1009,
            "curve_id": "s31:cycle:002",
            "calibration_coverage_stratum": "resonance_band_missed",
            "nearest_calibration_distance_linewidth_ratio": 2.0,
            "calibration_points_within_one_linewidth": 0,
        },
    ]

    rows = microstrip_b2_calibration_coverage_effect_rows(effect_rows, coverage_rows)
    summary = summarize_microstrip_b2_calibration_coverage_effects(rows)
    by_stratum = {row["calibration_coverage_stratum"]: row for row in summary}

    assert len(rows) == 2
    assert by_stratum["resonance_band_covered"]["pair_count"] == 1
    assert by_stratum["resonance_band_covered"]["median_cfnn_effect"] == pytest.approx(0.50)
    assert by_stratum["resonance_band_missed"]["cfnn_win_rate"] == pytest.approx(0.0)
    assert by_stratum["resonance_band_missed"]["median_nearest_calibration_distance_linewidth_ratio"] == pytest.approx(2.0)


def test_microstrip_b2_effect_and_coverage_join_keeps_sampling_modes_separate():
    from ai4science_microstrip_b import (
        microstrip_b2_calibration_coverage_effect_rows,
        microstrip_instance_effect_rows,
    )

    base_record = {
        "status": "ok",
        "experiment": "B2",
        "scenario": "low_calibration_target_25c_50rh",
        "sample_id": "s31",
        "split_seed": 233,
        "init_seed": 1009,
        "calibration_frequency_count": 4,
    }
    records = []
    for mode, cfnn_q, mlp_q in [
        ("random", 0.10, 0.20),
        ("missed_band", 0.30, 0.20),
    ]:
        for family, q_error in [("CFNN", cfnn_q), ("MLP", mlp_q)]:
            records.append({
                **base_record,
                "family": family,
                "calibration_sampling_mode": mode,
                "curve_metrics": [{
                    "curve_id": "s31:cycle:001",
                    "quality_factor_relative_error": q_error,
                    "truth_event_identifiable": True,
                }],
            })
    coverage_rows = [
        {
            **{key: value for key, value in base_record.items() if key != "status"},
            "curve_id": "s31:cycle:001",
            "calibration_sampling_mode": "random",
            "calibration_coverage_stratum": "resonance_band_covered",
        },
        {
            **{key: value for key, value in base_record.items() if key != "status"},
            "curve_id": "s31:cycle:001",
            "calibration_sampling_mode": "missed_band",
            "calibration_coverage_stratum": "resonance_band_missed",
        },
    ]

    effect_rows = microstrip_instance_effect_rows(
        records,
        comparison_families={"MLP"},
        metrics={"quality_factor_relative_error"},
    )
    joined = microstrip_b2_calibration_coverage_effect_rows(effect_rows, coverage_rows)
    by_mode = {row["calibration_sampling_mode"]: row for row in joined}

    assert len(effect_rows) == 2
    assert len(joined) == 2
    assert by_mode["random"]["calibration_coverage_stratum"] == "resonance_band_covered"
    assert by_mode["random"]["cfnn_effect"] == pytest.approx(0.5)
    assert by_mode["missed_band"]["calibration_coverage_stratum"] == "resonance_band_missed"
    assert by_mode["missed_band"]["cfnn_effect"] == pytest.approx(-0.5)


def test_select_microstrip_b2_guardrail_replay_cases_prefers_q_positive_event_failures():
    from ai4science_microstrip_b import select_microstrip_b2_guardrail_replay_cases

    def row(curve_id, metric, effect, baseline="Local nested CF control", calibration=16):
        return {
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "sample_id": curve_id.split(":")[0],
            "calibration_frequency_count": calibration,
            "calibration_sampling_mode": "missed_band",
            "comparison_family": baseline,
            "metric": metric,
            "curve_id": curve_id,
            "split_seed": 233,
            "init_seed": 1009,
            "cfnn_effect": effect,
        }

    rows = [
        row("s34:cycle:008", "quality_factor_relative_error", 0.49),
        row("s34:cycle:008", "resonance_frequency_abs_error_hz", -1.65),
        row("s34:cycle:008", "local_phase_transition_abs_error_hz", 0.10),
        row("s33:cycle:038", "quality_factor_relative_error", 0.20, calibration=8),
        row("s33:cycle:038", "resonance_frequency_abs_error_hz", 0.05, calibration=8),
        row("s33:cycle:038", "local_phase_transition_abs_error_hz", -3.80, calibration=8),
        row("s31:cycle:008", "quality_factor_relative_error", 0.65, calibration=32),
        row("s31:cycle:008", "resonance_frequency_abs_error_hz", 0.10, calibration=32),
        row("s31:cycle:008", "local_phase_transition_abs_error_hz", -0.18, calibration=32),
        row("s31:cycle:023", "quality_factor_relative_error", 0.24, calibration=32),
        row("s31:cycle:023", "resonance_frequency_abs_error_hz", 0.10, calibration=32),
        row("s31:cycle:023", "local_phase_transition_abs_error_hz", -0.36, calibration=32),
        row("s31:cycle:008", "quality_factor_relative_error", -0.10),
        row("s31:cycle:008", "resonance_frequency_abs_error_hz", -2.00),
    ]

    cases = select_microstrip_b2_guardrail_replay_cases(rows, max_cases_per_failure_type=1)
    by_type = {case["failure_type"]: case for case in cases}

    assert sorted(by_type) == ["f0_guardrail_failure", "local_phase_guardrail_failure"]
    assert by_type["f0_guardrail_failure"]["curve_id"] == "s34:cycle:008"
    assert by_type["f0_guardrail_failure"]["quality_factor_effect"] == pytest.approx(0.49)
    assert by_type["f0_guardrail_failure"]["guardrail_effect"] == pytest.approx(-1.65)
    assert by_type["local_phase_guardrail_failure"]["curve_id"] == "s33:cycle:038"
    assert by_type["local_phase_guardrail_failure"]["calibration_frequency_count"] == 8
    assert by_type["local_phase_guardrail_failure"]["guardrail_effect"] == pytest.approx(-3.80)


def test_select_microstrip_b2_guardrail_replay_cases_ranks_guardrail_severity_before_q_gain():
    from ai4science_microstrip_b import select_microstrip_b2_guardrail_replay_cases

    def row(curve_id, metric, effect, calibration):
        return {
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "sample_id": curve_id.split(":")[0],
            "calibration_frequency_count": calibration,
            "calibration_sampling_mode": "missed_band",
            "comparison_family": "MLP",
            "metric": metric,
            "curve_id": curve_id,
            "split_seed": 233,
            "init_seed": 1013,
            "cfnn_effect": effect,
        }

    rows = [
        row("s34:cycle:038", "quality_factor_relative_error", 0.66, 16),
        row("s34:cycle:038", "local_phase_transition_abs_error_hz", -0.18, 16),
        row("s31:cycle:023", "quality_factor_relative_error", 0.24, 32),
        row("s31:cycle:023", "local_phase_transition_abs_error_hz", -0.36, 32),
    ]

    cases = select_microstrip_b2_guardrail_replay_cases(rows, max_cases_per_failure_type=1)

    assert len(cases) == 1
    assert cases[0]["failure_type"] == "local_phase_guardrail_failure"
    assert cases[0]["curve_id"] == "s31:cycle:023"
    assert cases[0]["guardrail_effect"] == pytest.approx(-0.36)


def test_microstrip_truth_notch_descriptor_flags_multi_notch_or_boundary_risk():
    from downstream_protocol import MeasuredCurve
    from ai4science_microstrip_b import microstrip_truth_notch_descriptor_rows

    frequency = np.linspace(1.75e9, 2.25e9, 801)
    safe_magnitude = 1.0 - 0.5 / (1.0 + ((frequency - 1.95e9) / 0.025e9) ** 2)
    risky_magnitude = (
        1.0
        - 0.45 / (1.0 + ((frequency - 1.90e9) / 0.025e9) ** 2)
        - 0.35 / (1.0 + ((frequency - 2.235e9) / 0.018e9) ** 2)
    )
    boundary_magnitude = 1.0 - 0.5 / (1.0 + ((frequency - 1.90e9) / 0.15e9) ** 2)
    curves = [
        MeasuredCurve(
            curve_id="safe",
            frequency=frequency,
            response=np.column_stack((safe_magnitude, np.zeros_like(frequency))).astype(np.float32),
            conditions={"sample_id": "safe"},
            source_files=("synthetic",),
            metadata={},
        ),
        MeasuredCurve(
            curve_id="risky",
            frequency=frequency,
            response=np.column_stack((risky_magnitude, np.zeros_like(frequency))).astype(np.float32),
            conditions={"sample_id": "risky"},
            source_files=("synthetic",),
            metadata={},
        ),
        MeasuredCurve(
            curve_id="boundary",
            frequency=frequency,
            response=np.column_stack((boundary_magnitude, np.zeros_like(frequency))).astype(np.float32),
            conditions={"sample_id": "boundary"},
            source_files=("synthetic",),
            metadata={},
        ),
    ]

    rows = microstrip_truth_notch_descriptor_rows(curves, boundary_margin_linewidths=1.0)
    by_curve = {row["curve_id"]: row for row in rows}

    assert by_curve["safe"]["notch_geometry_stratum"] == "single_notch_interior"
    assert by_curve["safe"]["boundary_proximity_stratum"] == "edge_interior"
    assert by_curve["safe"]["competing_notch_count"] == 0
    assert by_curve["risky"]["notch_geometry_stratum"] == "multi_notch_or_boundary"
    assert by_curve["risky"]["competing_notch_count"] >= 1
    assert by_curve["risky"]["notch_count"] >= 2
    assert by_curve["boundary"]["notch_geometry_stratum"] == "multi_notch_or_boundary"
    assert by_curve["boundary"]["boundary_proximity_stratum"] in {"edge_proximal", "edge_intermediate"}


def test_microstrip_b2_f0_guardrail_descriptor_rows_join_q_effect_and_notch_geometry():
    from ai4science_microstrip_b import microstrip_b2_f0_guardrail_descriptor_rows

    effect_rows = [
        {
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "sample_id": "s34",
            "calibration_frequency_count": 16,
            "calibration_sampling_mode": "missed_band",
            "comparison_family": "Local nested CF control",
            "metric": "quality_factor_relative_error",
            "curve_id": "s34:cycle:038",
            "split_seed": 377,
            "init_seed": 1009,
            "cfnn_effect": 0.54,
            "cfnn_wins": True,
        },
        {
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "sample_id": "s34",
            "calibration_frequency_count": 16,
            "calibration_sampling_mode": "missed_band",
            "comparison_family": "Local nested CF control",
            "metric": "resonance_frequency_abs_error_hz",
            "curve_id": "s34:cycle:038",
            "split_seed": 377,
            "init_seed": 1009,
            "cfnn_effect": -1.65,
            "cfnn_wins": False,
        },
    ]
    descriptor_rows = [{
        "curve_id": "s34:cycle:038",
        "notch_geometry_stratum": "multi_notch_or_boundary",
        "notch_count": 2,
        "competing_notch_count": 1,
        "nearest_competing_notch_gap_linewidth_ratio": 1.4,
        "nearest_search_edge_distance_linewidth_ratio": 0.9,
        "boundary_proximity_stratum": "edge_intermediate",
    }]
    coverage_rows = [{
        "experiment": "B2",
        "scenario": "low_calibration_target_25c_50rh",
        "sample_id": "s34",
        "calibration_frequency_count": 16,
        "calibration_sampling_mode": "missed_band",
        "split_seed": 377,
        "init_seed": 1009,
        "curve_id": "s34:cycle:038",
        "calibration_coverage_stratum": "resonance_band_missed",
        "nearest_calibration_distance_linewidth_ratio": 1.2,
    }]

    rows = microstrip_b2_f0_guardrail_descriptor_rows(
        effect_rows,
        descriptor_rows,
        coverage_rows,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["quality_factor_effect"] == pytest.approx(0.54)
    assert row["f0_effect"] == pytest.approx(-1.65)
    assert row["q_positive_f0_negative"] is True
    assert row["notch_geometry_stratum"] == "multi_notch_or_boundary"
    assert row["boundary_proximity_stratum"] == "edge_intermediate"
    assert row["calibration_coverage_stratum"] == "resonance_band_missed"


def test_microstrip_b2_confirmatory_guardrail_stratum_effect_rows_join_all_gate_metrics():
    from ai4science_microstrip_b import microstrip_b2_confirmatory_guardrail_stratum_effect_rows

    base = {
        "experiment": "B2",
        "scenario": "low_calibration_target_25c_50rh",
        "sample_id": "s34",
        "calibration_frequency_count": 16,
        "calibration_sampling_mode": "missed_band",
        "comparison_family": "MLP",
        "split_seed": 233,
        "init_seed": 1009,
        "curve_id": "s34:cycle:038",
    }
    effect_rows = [
        {**base, "metric": "quality_factor_relative_error", "cfnn_effect": 0.40, "cfnn_wins": True},
        {**base, "metric": "resonance_frequency_abs_error_hz", "cfnn_effect": -0.30, "cfnn_wins": False},
        {**base, "metric": "local_phase_transition_abs_error_hz", "cfnn_effect": 0.10, "cfnn_wins": True},
        {**base, "metric": "complex_nrmse", "cfnn_effect": 0.05, "cfnn_wins": True},
    ]
    descriptor_rows = [{
        **base,
        "boundary_proximity_stratum": "edge_proximal",
        "calibration_coverage_stratum": "resonance_band_missed",
        "notch_geometry_stratum": "multi_notch_or_boundary",
        "q_positive_f0_negative": True,
        "quality_factor_effect": 0.40,
        "f0_effect": -0.30,
    }]

    rows = microstrip_b2_confirmatory_guardrail_stratum_effect_rows(effect_rows, descriptor_rows)
    by_metric = {row["metric"]: row for row in rows}

    assert len(rows) == 4
    assert by_metric["quality_factor_relative_error"]["stratum_id"] == "edge_proximal_missed_band"
    assert by_metric["quality_factor_relative_error"]["stratum_role"] == "primary_guardrail_stratum"
    assert by_metric["resonance_frequency_abs_error_hz"]["boundary_proximity_stratum"] == "edge_proximal"
    assert by_metric["complex_nrmse"]["calibration_coverage_stratum"] == "resonance_band_missed"


def test_summarize_microstrip_b2_confirmatory_guardrail_stratum_effects_reports_gate_status():
    from ai4science_microstrip_b import summarize_microstrip_b2_confirmatory_guardrail_stratum_effects

    def row(stratum, metric, effect, wins=True):
        boundary = stratum.removesuffix("_missed_band")
        return {
            "experiment": "B2",
            "scenario": "low_calibration_target_25c_50rh",
            "calibration_frequency_count": 16,
            "comparison_family": "MLP",
            "stratum_id": stratum,
            "stratum_role": "primary_guardrail_stratum" if boundary == "edge_proximal" else "contrast_stratum",
            "boundary_proximity_stratum": boundary,
            "calibration_coverage_stratum": "resonance_band_missed",
            "metric": metric,
            "cfnn_effect": effect,
            "cfnn_wins": wins,
        }

    rows = [
        row("edge_proximal_missed_band", "quality_factor_relative_error", 0.40),
        row("edge_proximal_missed_band", "quality_factor_relative_error", 0.20),
        row("edge_proximal_missed_band", "resonance_frequency_abs_error_hz", -0.30, False),
        row("edge_proximal_missed_band", "resonance_frequency_abs_error_hz", -0.10, False),
        row("edge_proximal_missed_band", "local_phase_transition_abs_error_hz", 0.05),
        row("edge_proximal_missed_band", "complex_nrmse", 0.02),
        row("edge_intermediate_missed_band", "quality_factor_relative_error", 0.15),
        row("edge_intermediate_missed_band", "resonance_frequency_abs_error_hz", 0.80),
        row("edge_intermediate_missed_band", "local_phase_transition_abs_error_hz", 0.10),
        row("edge_intermediate_missed_band", "complex_nrmse", 0.03),
    ]

    summary = summarize_microstrip_b2_confirmatory_guardrail_stratum_effects(rows)
    metric_rows = {
        (row["stratum_id"], row["metric"]): row
        for row in summary["metric_summary_rows"]
    }
    gates = {row["stratum_id"]: row for row in summary["claim_gate_rows"]}

    assert metric_rows[("edge_proximal_missed_band", "quality_factor_relative_error")]["median_cfnn_effect"] == pytest.approx(0.30)
    assert metric_rows[("edge_proximal_missed_band", "resonance_frequency_abs_error_hz")]["cfnn_win_rate"] == 0.0
    assert gates["edge_proximal_missed_band"]["q_gain_positive"] is True
    assert gates["edge_proximal_missed_band"]["f0_guardrail_pass"] is False
    assert gates["edge_proximal_missed_band"]["phase_guardrail_pass"] is True
    assert gates["edge_proximal_missed_band"]["claim_gate_status"] == "localized_q_signal_blocked_by_guardrail"
    assert gates["edge_intermediate_missed_band"]["claim_gate_status"] == "passes_q_with_event_guardrails"


def test_microstrip_b2_confirmatory_guardrail_runner_report_includes_gate_rows():
    from run_ai4science_microstrip_b2_confirmatory_guardrail_strata import build_report

    summary = {
        "analysis_type": "microstrip_b2_confirmatory_guardrail_stratum_effects",
        "primary_claim_gate": "q_gain_requires_f0_and_phase_guardrails",
        "metric_summary_rows": [
            {
                "stratum_id": "edge_proximal_missed_band",
                "stratum_role": "primary_guardrail_stratum",
                "comparison_family": "MLP",
                "calibration_frequency_count": 16,
                "metric": "quality_factor_relative_error",
                "pair_count": 2,
                "cfnn_win_rate": 1.0,
                "median_cfnn_effect": 0.30,
            }
        ],
        "claim_gate_rows": [
            {
                "stratum_id": "edge_proximal_missed_band",
                "stratum_role": "primary_guardrail_stratum",
                "comparison_family": "MLP",
                "calibration_frequency_count": 16,
                "quality_factor_effect": 0.30,
                "f0_effect": -0.20,
                "local_phase_effect": 0.05,
                "global_complex_nrmse_effect": 0.02,
                "claim_gate_status": "localized_q_signal_blocked_by_guardrail",
            }
        ],
    }

    report = build_report(summary)

    assert "Microstrip B2 Confirmatory Guardrail Strata" in report
    assert "q_gain_requires_f0_and_phase_guardrails" in report
    assert "edge_proximal_missed_band" in report
    assert "localized_q_signal_blocked_by_guardrail" in report


def test_microstrip_low_calibration_event_transfer_separates_localized_q_from_digital_twin():
    from ai4science_microstrip_event_transfer import microstrip_low_calibration_event_transfer

    summary = {
        "analysis_type": "microstrip_b2_confirmatory_guardrail_stratum_effects",
        "claim_gate_rows": [
            {
                "stratum_id": "edge_intermediate_missed_band",
                "stratum_role": "contrast_stratum",
                "comparison_family": "Local nested CF control",
                "calibration_frequency_count": 8,
                "quality_factor_effect": 0.31,
                "f0_effect": 0.84,
                "local_phase_effect": 0.18,
                "global_complex_nrmse_effect": 0.13,
                "q_gain_positive": True,
                "all_guardrails_pass": True,
                "claim_gate_status": "passes_q_with_event_guardrails",
            },
            {
                "stratum_id": "edge_proximal_missed_band",
                "stratum_role": "primary_guardrail_stratum",
                "comparison_family": "Local nested CF control",
                "calibration_frequency_count": 8,
                "quality_factor_effect": 0.52,
                "f0_effect": -0.38,
                "local_phase_effect": 0.48,
                "global_complex_nrmse_effect": 0.05,
                "q_gain_positive": True,
                "all_guardrails_pass": False,
                "claim_gate_status": "localized_q_signal_blocked_by_guardrail",
            },
        ],
    }

    result = microstrip_low_calibration_event_transfer(summary)

    assert result["analysis_type"] == "microstrip_low_calibration_event_transfer"
    assert result["claim_gate"]["claim_status"] == "localized_q_transfer_with_guardrails_not_digital_twin"
    assert result["claim_gate"]["digital_twin_status"] == "blocked_full_digital_twin_language"
    by_stratum = {row["stratum_id"]: row for row in result["focused_stratum_rows"]}
    assert by_stratum["edge_intermediate_missed_band"]["localized_transfer_status"] == "passes_localized_q_transfer"
    assert by_stratum["edge_proximal_missed_band"]["localized_transfer_status"] == "q_signal_blocked_by_event_guardrail"
