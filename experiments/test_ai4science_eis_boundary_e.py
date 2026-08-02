from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_eis_spectral_descriptors_count_relaxation_peaks_and_phase_turns():
    from ai4science_eis_boundary_e import eis_spectral_descriptors

    frequency = np.logspace(-2, 5, 400)
    w = 2.0 * np.pi * frequency
    response = 10.0 + 300.0 / (1.0 + 1j * w * 1e-2) + 800.0 / (1.0 + 1j * w * 1e-5)

    row = eis_spectral_descriptors(
        "two_tc:test",
        frequency,
        response,
        metadata={"known_pole_count": 2, "distributed_relaxation": False},
    )

    assert row["spectrum_id"] == "two_tc:test"
    assert row["point_count"] == 400
    assert row["relaxation_peak_count"] >= 1
    assert row["known_pole_count"] == 2
    assert row["distributed_relaxation"] is False
    assert row["frequency_points_per_decade"] > 50.0
    assert row["phase_total_turns"] > 0.0
    assert row["max_normalized_curvature"] > 0.0


def test_loss_effect_rows_use_signed_cfnn_advantage_against_best_baseline():
    from ai4science_eis_boundary_e import loss_effect_row

    positive = loss_effect_row(
        rows=[
            {"model": "CFNN-Hybrid", "curve_mse": 2.0},
            {"model": "MLP-matched", "curve_mse": 4.0},
            {"model": "KAN", "curve_mse": 3.0},
        ],
        metric="curve_mse",
        cfnn_models={"CFNN-Hybrid"},
    )
    negative = loss_effect_row(
        rows=[
            {"model": "CFNN-Hybrid", "curve_mse": 5.0},
            {"model": "MLP-matched", "curve_mse": 4.0},
            {"model": "KAN", "curve_mse": 3.0},
        ],
        metric="curve_mse",
        cfnn_models={"CFNN-Hybrid"},
    )

    assert positive["cfnn_metric"] == 2.0
    assert positive["best_baseline_model"] == "KAN"
    assert positive["signed_cfnn_effect"] > 0.0
    assert negative["best_baseline_model"] == "KAN"
    assert negative["signed_cfnn_effect"] < 0.0


def test_layered_loss_effects_separate_neural_and_physical_priors():
    from ai4science_eis_boundary_e import layered_loss_effect_rows

    effects = layered_loss_effect_rows(
        rows=[
            {"model": "CFNN-Hybrid", "curve_mse": 10.0},
            {"model": "MLP-matched", "curve_mse": 12.0},
            {"model": "KAN", "curve_mse": 9.0},
            {"model": "Classical-AIC", "curve_mse": 1.0},
        ],
        metric="curve_mse",
        cfnn_models={"CFNN-Hybrid"},
        neural_baseline_models={"MLP-matched", "KAN"},
        physical_prior_models={"Classical-AIC"},
    )

    assert effects["all_baselines"]["best_baseline_model"] == "Classical-AIC"
    assert effects["neural_only"]["best_baseline_model"] == "KAN"
    assert effects["physical_prior_only"]["best_baseline_model"] == "Classical-AIC"
    assert effects["neural_only"]["signed_cfnn_effect"] < 0.0
    assert effects["physical_prior_only"]["signed_cfnn_effect"] < effects["neural_only"]["signed_cfnn_effect"]


def test_eis_neural_positive_edge_audit_marks_physical_prior_elimination():
    from ai4science_eis_boundary_e import audit_eis_neural_positive_edges

    rows = [
        {
            "task_id": "voigt_K4_n40_noise0.1",
            "effect_layer": "neural_only",
            "status": "ok",
            "signed_cfnn_effect": 0.36,
            "cfnn_metric": 9.0,
            "best_baseline_model": "KAN",
            "best_baseline_metric": 14.0,
            "descriptor": {
                "known_pole_count": 4,
                "relaxation_peak_count": 2,
                "distributed_relaxation": False,
                "n_points": 40,
                "noise": 0.1,
            },
        },
        {
            "task_id": "voigt_K4_n40_noise0.1",
            "effect_layer": "physical_prior_only",
            "status": "ok",
            "signed_cfnn_effect": -4.0,
            "best_baseline_model": "Classical-AIC",
        },
        {
            "task_id": "voigt_K4_n40_noise0.1",
            "effect_layer": "all_baselines",
            "status": "ok",
            "signed_cfnn_effect": -4.0,
            "best_baseline_model": "Classical-AIC",
        },
        {
            "task_id": "voigt_K3_n20_noise0.1",
            "effect_layer": "neural_only",
            "status": "ok",
            "signed_cfnn_effect": 0.10,
            "best_baseline_model": "MLP-matched",
            "descriptor": {"known_pole_count": 3, "relaxation_peak_count": 3, "n_points": 20, "noise": 0.1},
        },
    ]

    summary = audit_eis_neural_positive_edges(rows)
    audited = {row["task_id"]: row for row in summary["positive_edge_rows"]}

    assert summary["positive_neural_edge_count"] == 2
    assert audited["voigt_K4_n40_noise0.1"]["edge_status"] == "physical_prior_eliminates_neural_edge"
    assert audited["voigt_K4_n40_noise0.1"]["physical_prior_effect"] == -4.0
    assert audited["voigt_K4_n40_noise0.1"]["all_baseline_effect"] == -4.0
    assert audited["voigt_K4_n40_noise0.1"]["descriptor_stratum"] == "multi_pole_relaxation"
    assert audited["voigt_K3_n20_noise0.1"]["edge_status"] == "missing_physical_prior_comparison"
    assert summary["status_counts"]["physical_prior_eliminates_neural_edge"] == 1


def test_external_spectrum_boundary_gate_requires_physical_prior_elimination():
    from ai4science_eis_boundary_e import external_spectrum_boundary_gate

    rows = [
        {
            "task_id": "a",
            "effect_layer": "neural_only",
            "status": "ok",
            "signed_cfnn_effect": 0.2,
            "descriptor": {"known_pole_count": 4, "relaxation_peak_count": 2},
        },
        {
            "task_id": "a",
            "effect_layer": "physical_prior_only",
            "status": "ok",
            "signed_cfnn_effect": -1.0,
        },
        {
            "task_id": "a",
            "effect_layer": "all_baselines",
            "status": "ok",
            "signed_cfnn_effect": -1.0,
        },
        {
            "task_id": "b",
            "effect_layer": "neural_only",
            "status": "ok",
            "signed_cfnn_effect": 0.1,
            "descriptor": {"known_pole_count": 3, "relaxation_peak_count": 3},
        },
        {
            "task_id": "b",
            "effect_layer": "physical_prior_only",
            "status": "ok",
            "signed_cfnn_effect": 0.1,
        },
        {
            "task_id": "b",
            "effect_layer": "all_baselines",
            "status": "ok",
            "signed_cfnn_effect": 0.1,
        },
    ]

    gate = external_spectrum_boundary_gate(rows, [], elimination_threshold=0.80)

    assert gate["neural_positive_edge_count"] == 2
    assert gate["physical_prior_eliminated_edge_count"] == 1
    assert gate["all_baseline_surviving_positive_edge_count"] == 1
    assert gate["physical_prior_elimination_rate"] == 0.5
    assert gate["claim_status"] == "boundary_challenged_by_surviving_edge"


def test_eis_external_spectrum_boundary_report_states_claim_gate():
    from run_ai4science_eis_external_spectrum_boundary import build_report

    report = build_report({
        "gate": {
            "claim_status": "physical_prior_boundary_supported",
            "neural_positive_edge_count": 4,
            "physical_prior_eliminated_edge_count": 4,
            "physical_prior_elimination_rate": 1.0,
            "all_baseline_surviving_positive_edge_count": 0,
            "median_all_baseline_signed_cfnn_effect": -9.9,
        },
        "descriptor_strata": [
            {
                "descriptor_stratum": "multi_pole_relaxation",
                "descriptor_count": 3,
                "median_known_pole_count": 4,
                "median_relaxation_peak_count": 2,
                "distributed_relaxation_count": 0,
            }
        ],
        "hsc_same_split_status": "ok",
        "hsc_same_split_layered_effect_row_count": 15,
        "hsc_same_split_audit": {
            "positive_neural_edge_count": 2,
            "status_counts": {"physical_prior_eliminates_neural_edge": 2},
        },
    })

    assert "physical_prior_boundary_supported" in report
    assert "Physical-prior elimination rate" in report
    assert "HSC same-split status" in report
    assert "same-split physical-prior" in report
    assert "Neural-only positive edges are not structural CFNN advantages" in report


def test_hsc_condition_descriptor_rows_preserve_conditions_and_spectrum_descriptors():
    from ai4science_eis_boundary_e import hsc_condition_descriptor_rows
    from downstream_protocol import MeasuredCurve

    frequency = np.logspace(-1, 4, 81)
    w = 2.0 * np.pi * frequency
    impedance = 2.0 + 30.0 / (1.0 + 1j * w * 1e-2)
    curve = MeasuredCurve(
        curve_id="soc:020:temperature:+20",
        frequency=frequency,
        response=np.column_stack((impedance.real, impedance.imag)).astype(np.float32),
        conditions={"soc_percent": 20, "temperature_c": 20},
        source_files=("Supercap_EIS.mat",),
        metadata={"source_key": "hybrid_supercapacitor_eis"},
    )

    rows = hsc_condition_descriptor_rows([curve])

    assert len(rows) == 1
    row = rows[0]
    assert row["spectrum_id"] == "hsc:soc:020:temperature:+20"
    assert row["curve_id"] == "soc:020:temperature:+20"
    assert row["task_family"] == "hybrid_supercapacitor_eis"
    assert row["soc_percent"] == 20
    assert row["temperature_c"] == 20
    assert row["n_points"] == 81
    assert row["point_count"] == 81
    assert row["frequency_points_per_decade"] > 10.0


def test_hsc_voigt_physical_prior_rows_fit_condition_curve_with_aic_selection():
    from ai4science_eis_boundary_e import hsc_voigt_physical_prior_rows
    from downstream_protocol import MeasuredCurve
    import eis_utils as U

    frequency = np.logspace(-2, 5, 81)
    impedance = U.voigt_impedance(
        frequency,
        Rs=5.0,
        R_list=[30.0, 80.0],
        C_list=[1e-3, 1e-5],
    )
    curve = MeasuredCurve(
        curve_id="soc:020:temperature:+20",
        frequency=frequency,
        response=np.column_stack((impedance.real, impedance.imag)).astype(np.float32),
        conditions={"soc_percent": 20, "temperature_c": 20},
        source_files=("Supercap_EIS.mat",),
        metadata={"source_key": "hybrid_supercapacitor_eis"},
    )

    rows = hsc_voigt_physical_prior_rows([curve], max_order=3)
    by_model = {row["model"]: row for row in rows}

    assert {"Classical-Voigt-K1", "Classical-Voigt-K2", "Classical-Voigt-K3", "Classical-AIC"} <= set(by_model)
    assert by_model["Classical-AIC"]["task_id"] == "hsc:soc:020:temperature:+20"
    assert by_model["Classical-AIC"]["fit_K"] in {2, 3}
    assert by_model["Classical-AIC"]["fit_success"] is True
    assert by_model["Classical-AIC"]["curve_mse"] < 1e-6
    assert by_model["Classical-AIC"]["complex_r2"] > 0.999
    assert by_model["Classical-AIC"]["soc_percent"] == 20
    assert by_model["Classical-AIC"]["temperature_c"] == 20


def test_hsc_same_split_voigt_prior_rows_fit_observed_points_and_score_test_split():
    from ai4science_eis_boundary_e import hsc_same_split_voigt_prior_rows
    from confirmatory_protocol import DatasetBundle
    import eis_utils as U

    frequency = np.logspace(-2, 5, 81)
    impedance = U.voigt_impedance(
        frequency,
        Rs=5.0,
        R_list=[30.0, 80.0],
        C_list=[1e-3, 1e-5],
    )
    train = np.arange(0, len(frequency), 3)
    validation = np.arange(1, len(frequency), 3)
    test = np.arange(2, len(frequency), 3)
    bundle = DatasetBundle(
        x_train=frequency[train].reshape(-1, 1),
        y_train=np.column_stack((impedance[train].real, impedance[train].imag)).astype(np.float32),
        x_validation=frequency[validation].reshape(-1, 1),
        y_validation=np.column_stack((impedance[validation].real, impedance[validation].imag)).astype(np.float32),
        x_test=frequency[test].reshape(-1, 1),
        y_test=np.column_stack((impedance[test].real, impedance[test].imag)).astype(np.float32),
        metadata={
            "task": "hybrid_supercapacitor_eis",
            "curve_id": "soc:020:temperature:+20",
            "data_seed": 719,
            "conditions": {"soc_percent": 20, "temperature_c": 20},
            "train_point_ids": [f"p:{int(i)}" for i in train],
            "validation_point_ids": [f"p:{int(i)}" for i in validation],
            "test_point_ids": [f"p:{int(i)}" for i in test],
            "split_strategy": "within_curve_peak_agnostic_random_frequency_mask",
        },
    )

    rows = hsc_same_split_voigt_prior_rows([bundle], max_order=3)
    by_model = {row["model"]: row for row in rows}

    assert {"Classical-Voigt-SameSplit-K1", "Classical-Voigt-SameSplit-K2", "Classical-Voigt-SameSplit-K3", "Classical-AIC-SameSplit"} <= set(by_model)
    assert by_model["Classical-AIC-SameSplit"]["task_id"] == "hsc:soc:020:temperature:+20:data_seed:719"
    assert by_model["Classical-AIC-SameSplit"]["metric"] == "global_complex_nrmse"
    assert by_model["Classical-AIC-SameSplit"]["fit_observation_count"] == len(train) + len(validation)
    assert by_model["Classical-AIC-SameSplit"]["test_point_count"] == len(test)
    assert by_model["Classical-AIC-SameSplit"]["global_complex_nrmse"] < 1e-3
    assert by_model["Classical-AIC-SameSplit"]["fit_K"] in {2, 3}


def test_hsc_neural_metric_rows_preserve_frozen_split_identity():
    from ai4science_eis_boundary_e import hsc_neural_metric_rows

    rows = hsc_neural_metric_rows([
        {
            "task": "hybrid_supercapacitor_eis",
            "family": "CFNN",
            "status": "ok",
            "target_budget": 128,
            "data_seed": 719,
            "init_seed": 11003,
            "dataset": {
                "curve_id": "soc:080:temperature:+50",
                "conditions": {"soc_percent": 80, "temperature_c": 50},
                "split_strategy": "within_curve_peak_agnostic_random_frequency_mask",
                "train_point_ids_count": 7,
                "validation_point_ids_count": 7,
                "test_point_ids_count": 67,
                "test_point_ids_sha256": "abc",
            },
            "test_metrics": {"global_complex_nrmse": 1.25},
        },
        {
            "task": "hybrid_supercapacitor_eis",
            "family": "MLP",
            "status": "ok",
            "target_budget": 128,
            "data_seed": 719,
            "init_seed": 11003,
            "dataset": {"curve_id": "soc:080:temperature:+50"},
            "test_metrics": {"global_complex_nrmse": None},
        },
    ])

    assert len(rows) == 1
    assert rows[0]["task_id"] == "hsc:soc:080:temperature:+50:data_seed:719"
    assert rows[0]["model"] == "CFNN"
    assert rows[0]["metric"] == "global_complex_nrmse"
    assert rows[0]["global_complex_nrmse"] == 1.25
    assert rows[0]["train_point_count"] == 7
    assert rows[0]["validation_point_count"] == 7
    assert rows[0]["test_point_ids_sha256"] == "abc"


def test_hsc_same_split_layered_effect_rows_compare_neural_and_physical_priors():
    from ai4science_eis_boundary_e import hsc_same_split_layered_effect_rows

    rows = hsc_same_split_layered_effect_rows(
        neural_rows=[
            {"task_id": "hsc:a:data_seed:1", "model": "CFNN", "global_complex_nrmse": 0.8},
            {"task_id": "hsc:a:data_seed:1", "model": "CFNN", "global_complex_nrmse": 1.0},
            {"task_id": "hsc:a:data_seed:1", "model": "MLP", "global_complex_nrmse": 1.2},
            {"task_id": "hsc:a:data_seed:1", "model": "KAN", "global_complex_nrmse": 0.7},
        ],
        physical_rows=[
            {
                "task_id": "hsc:a:data_seed:1",
                "model": "Classical-AIC-SameSplit",
                "global_complex_nrmse": 0.2,
            }
        ],
        metric="global_complex_nrmse",
    )

    by_layer = {row["effect_layer"]: row for row in rows}
    assert by_layer["neural_only"]["best_baseline_model"] == "KAN"
    assert by_layer["physical_prior_only"]["best_baseline_model"] == "Classical-AIC-SameSplit"
    assert by_layer["neural_only"]["signed_cfnn_effect"] < 0.0
    assert by_layer["physical_prior_only"]["signed_cfnn_effect"] < 0.0
    assert by_layer["all_baselines"]["task_id"] == "hsc:a:data_seed:1"
    assert by_layer["all_baselines"]["row_count"] == 4
