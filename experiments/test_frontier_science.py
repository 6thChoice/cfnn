import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def test_frontier_config_freezes_scale_and_budgets():
    from frontier_science.protocol import load_config

    config = load_config(ROOT / "frontier_science" / "config.json")
    assert config["budgets"] == [256, 512, 1024]
    assert config["budget_tolerance"] == 0.05
    assert len(config["evaluation_instance_ids"]) == 5
    assert len(config["evaluation_init_seeds"]) == 5


def test_frontier_seed_precedes_model_construction():
    from frontier_science.protocol import seeded_build

    first = seeded_build(1009, lambda: torch.nn.Linear(2, 3)).state_dict()
    torch.rand(100)
    second = seeded_build(1009, lambda: torch.nn.Linear(2, 3)).state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_invalid_record_is_rejected(tmp_path):
    from frontier_science.protocol import append_record

    schema = json.loads(
        (ROOT / "frontier_science" / "results_schema.json").read_text()
    )
    with pytest.raises(ValueError, match="schema"):
        append_record(tmp_path / "raw.jsonl", {"task": "qbic"}, schema)


def test_append_record_is_atomic_and_rejects_duplicate_ids(tmp_path):
    from frontier_science.protocol import append_record

    schema = json.loads(
        (ROOT / "frontier_science" / "results_schema.json").read_text()
    )
    record = {
        "record_id": "quantum:CFNN:256:eval-0:1009",
        "protocol_version": "1.0.0",
        "stage": "evaluation",
        "task": "quantum",
        "family": "CFNN",
        "target_budget": 256,
        "actual_parameters": 254,
        "instance_id": "eval-0",
        "init_seed": 1009,
        "status": "ok",
        "metrics": {"primary": 0.1},
        "provenance": {"config_sha256": "a" * 64},
    }
    path = tmp_path / "raw.jsonl"
    append_record(path, record, schema)
    assert json.loads(path.read_text().strip()) == record
    with pytest.raises(ValueError, match="duplicate record_id"):
        append_record(path, record, schema)


def test_qbic_audit_requires_experimental_angle_csv(tmp_path):
    from frontier_science.sources import audit_qbic_archive

    (tmp_path / "Fig2_SiC100nm_AngleSweep_Exp.csv").write_text(
        "angle,wavelength,R\n0,10,0.5\n"
    )
    audit = audit_qbic_archive(tmp_path)
    assert audit["status"] == "usable"
    assert audit["experimental_angle_files"] == 1


def test_nasa_audit_blocks_scalar_only_impedance(tmp_path):
    from frontier_science.sources import audit_nasa_eis_archive

    scipy.io.savemat(
        tmp_path / "battery.mat",
        {"Re": torch.tensor([1.0]).numpy(), "Rct": torch.tensor([2.0]).numpy()},
    )
    audit = audit_nasa_eis_archive(tmp_path)
    assert audit["status"] == "blocked"
    assert "frequency" in audit["reason"].lower()


def test_nasa_audit_traverses_matlab_cell_lists(tmp_path):
    from frontier_science.sources import audit_nasa_eis_archive

    scipy.io.savemat(
        tmp_path / "battery.mat",
        {
            "cycles": torch.tensor(
                [[(1.0 + 1.0j), (2.0 + 0.5j), (3.0 + 0.25j)]]
            ).numpy(),
        },
    )
    # scipy's simplify_cells output is represented directly to exercise list traversal.
    from frontier_science import sources

    names = list(
        sources._walk_named_arrays(
            {"cycle": [{"data": {"Battery_impedance": torch.ones(3).numpy()}}]}
        )
    )
    assert names[0][0] == "cycle[0].data.Battery_impedance"


def test_safe_extraction_rejects_parent_traversal(tmp_path):
    import zipfile

    from frontier_science.sources import safe_extract_zip

    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "out")


def test_nested_nasa_archives_are_expanded_before_audit(tmp_path):
    import io
    import zipfile

    from frontier_science.sources import (
        audit_nasa_eis_archive,
        extract_nested_zips,
        safe_extract_zip,
    )

    mat_buffer = io.BytesIO()
    scipy.io.savemat(
        mat_buffer,
        {
            "frequency": torch.tensor([1.0, 10.0, 100.0]).numpy(),
            "Battery_impedance": torch.tensor(
                [1.0 + 1.0j, 2.0 + 0.5j, 3.0 + 0.25j]
            ).numpy(),
        },
    )
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w") as inner:
        inner.writestr("cell/B0005.mat", mat_buffer.getvalue())
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as handle:
        handle.writestr("batch.zip", inner_buffer.getvalue())

    extract_root = tmp_path / "expanded"
    safe_extract_zip(outer, extract_root)
    extract_nested_zips(extract_root)
    audit = audit_nasa_eis_archive(extract_root)
    assert audit["status"] == "usable"
    assert len(audit["matched_frequency_impedance_fields"]) == 1


def test_eis_replacement_requires_documented_primary_block():
    from frontier_science.sources import select_eis_source

    selected = select_eis_source(
        {
            "status": "blocked",
            "reason": "no explicit frequency-resolved impedance pair was found",
        },
        {"provider": "figshare", "article_id": 23736582},
    )
    assert selected["status"] == "replacement_selected"
    assert selected["primary_block_reason"].startswith("no explicit frequency")
    assert selected["selected_source"]["article_id"] == 23736582


def test_eis_replacement_is_refused_when_primary_is_usable():
    from frontier_science.sources import select_eis_source

    with pytest.raises(ValueError, match="primary EIS source is usable"):
        select_eis_source(
            {"status": "usable"},
            {"provider": "figshare", "article_id": 23736582},
        )


def test_figshare_eis_audit_accepts_frequency_r_x_columns(tmp_path):
    from frontier_science.sources import audit_figshare_eis_archive

    path = tmp_path / "B01" / "EIS measurements" / "curve.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "Frequency(Hz),R(ohm),X(ohm),V(V),T(deg C)\n"
        "0.01,0.22,-0.31,3.4,20\n"
    )
    audit = audit_figshare_eis_archive(tmp_path)
    assert audit["status"] == "usable"
    assert audit["frequency_impedance_files"] == 1


def test_quantum_forward_matches_stored_clean_green():
    from frontier_science.quantum import make_quantum_instance, matsubara_forward

    instance = make_quantum_instance("test-single-pole", 0.0, 17)
    reconstructed = matsubara_forward(
        instance.spectrum, instance.omega, instance.matsubara, instance.weights
    )
    assert torch.max(torch.abs(reconstructed - instance.clean_green)) < 1e-6


def test_positive_spectrum_obeys_sum_rule():
    from frontier_science.quantum import normalized_positive_spectrum

    raw = torch.linspace(-2, 2, 401)
    weights = torch.full((401,), 0.025)
    spectrum = normalized_positive_spectrum(raw, weights)
    assert torch.all(spectrum >= 0)
    assert torch.isclose(torch.sum(spectrum * weights), torch.tensor(1.0), atol=1e-6)


def test_quantum_training_view_excludes_hidden_spectrum():
    from frontier_science.quantum import make_quantum_instance

    instance = make_quantum_instance("eval-0", 1e-3, 233)
    view = instance.training_view()
    assert "spectrum" not in view
    assert "clean_green" not in view
    assert set(view) == {
        "omega",
        "weights",
        "matsubara",
        "observed_green",
        "train_indices",
        "validation_indices",
    }


def test_quantum_templates_are_distinct_and_include_continua():
    from frontier_science.quantum import make_quantum_instance

    instances = [make_quantum_instance(f"eval-{index}", 0.0, 17) for index in range(5)]
    assert len({instance.template_name for instance in instances}) == 5
    assert sum(instance.has_non_lorentzian_continuum for instance in instances) >= 2
    pairwise = [
        torch.mean(torch.abs(instances[index].spectrum - instances[index + 1].spectrum)).item()
        for index in range(4)
    ]
    assert min(pairwise) > 1e-3


def test_quantum_metrics_are_exact_for_truth():
    from frontier_science.quantum import make_quantum_instance, quantum_metrics

    instance = make_quantum_instance("eval-1", 0.0, 17)
    metrics = quantum_metrics(instance, instance.spectrum.numpy())
    assert metrics["wasserstein_1"] < 1e-10
    assert metrics["integrated_absolute_error"] < 1e-10
    assert metrics["missed_peak_count"] == 0
    assert metrics["spurious_peak_count"] == 0


def test_record_id_can_be_computed_before_training():
    from frontier_science.runner import candidate_record_id

    candidate = {"actual_parameters": 254, "candidate_key": '{"units":10}'}
    optimizer = {"key": "lr3e-3_wd0"}
    record_id = candidate_record_id(
        "tuning", "quantum", "CFNN", candidate, "tune-1", 7001, optimizer
    )
    assert record_id.startswith("tuning:quantum:CFNN:254:")
    assert record_id.endswith(":tune-1:7001:lr3e-3_wd0")


def test_record_id_distinguishes_equal_size_architectures():
    from frontier_science.runner import candidate_record_id

    optimizer = {"key": "lr3e-3_wd0"}
    first = {"actual_parameters": 254, "candidate_key": '{"units":10,"degree":2}'}
    second = {"actual_parameters": 254, "candidate_key": '{"units":7,"degree":3}'}
    args = ("tuning", "quantum", "CFNN")
    assert candidate_record_id(*args, first, "tune-0", 7001, optimizer) != candidate_record_id(
        *args, second, "tune-0", 7001, optimizer
    )


def test_hierarchical_bootstrap_resamples_instances_then_seeds():
    import pandas as pd

    from frontier_science.artifacts import hierarchical_bootstrap

    rows = [
        {"instance_id": f"eval-{instance}", "init_seed": seed, "effect": instance + seed / 100.0}
        for instance in range(5)
        for seed in range(5)
    ]
    result = hierarchical_bootstrap(pd.DataFrame(rows), "effect", seed=17, replicates=200)
    assert result["resampling_levels"] == ["instance_id", "init_seed"]
    assert result["n_instances"] == 5
    assert result["n_records"] == 25
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_advancement_requires_measured_task_for_portfolio():
    import pandas as pd

    from frontier_science.artifacts import evaluate_advancement

    rows = []
    for budget, value in [(256, 0.08), (512, 0.04), (1024, 0.03)]:
        rows.extend(
            [
                {"task": "quantum", "family": "CFNN", "target_budget": budget, "median": value, "failure_rate": 0.0},
                {"task": "quantum", "family": "MLP", "target_budget": budget, "median": value * 1.2, "failure_rate": 0.0},
            ]
        )
    decision = evaluate_advancement(pd.DataFrame(rows), {"quantum_wasserstein_1": 0.1})
    assert decision["tasks"]["quantum"]["advanced"] is True
    assert decision["portfolio_promoted"] is False


def test_advancement_requires_consistency_against_best_comparator():
    import pandas as pd

    from frontier_science.artifacts import evaluate_advancement

    frame = pd.DataFrame(
        [
            {"task": "quantum", "family": "CFNN", "target_budget": 512, "median": 0.11, "failure_rate": 0.0, "best_pair_consistency": 2},
            {"task": "quantum", "family": "KAN", "target_budget": 512, "median": 0.12, "failure_rate": 0.0, "best_pair_consistency": 0},
        ]
    )
    decision = evaluate_advancement(frame, {"quantum_wasserstein_1": 0.1})
    assert decision["tasks"]["quantum"]["criterion_1"] is False


def test_quantum_inverse_loss_has_finite_gradients():
    from frontier_science.quantum import make_quantum_instance, quantum_loss

    instance = make_quantum_instance("tune-0", 1e-3, 17)
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    )
    loss = quantum_loss(model, instance)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )


def test_sparse_mask_is_nested_and_keeps_endpoints():
    from frontier_science.qbic import sparse_mask

    small = sparse_mask(401, 0.125, 17)
    medium = sparse_mask(401, 0.25, 17)
    large = sparse_mask(401, 0.5, 17)
    assert small[0] and small[-1]
    assert medium[0] and medium[-1]
    assert np.all(medium[small])
    assert np.all(large[medium])


def test_qbic_selection_uses_fixed_control_coordinates():
    from frontier_science.qbic import load_qbic_curves, select_qbic_instances

    raw_root = ROOT / "frontier_science" / "data" / "raw" / "qbic"
    curves = load_qbic_curves(raw_root)
    selected = select_qbic_instances(curves)
    assert list(selected) == [
        "tune-0",
        "tune-1",
        "tune-2",
        "eval-0",
        "eval-1",
        "eval-2",
        "eval-3",
        "eval-4",
    ]
    assert [selected[key].metadata["angle_deg"] for key in selected] == [
        10.0,
        25.0,
        40.0,
        0.0,
        15.0,
        20.0,
        30.0,
        45.0,
    ]
    assert all(np.all(np.diff(curve.axis) > 0) for curve in selected.values())


def test_qbic_metrics_are_exact_for_truth():
    from frontier_science.qbic import qbic_metrics, sparse_mask

    wavelength = np.linspace(10.0, 11.0, 401)
    truth = 0.8 - 0.5 / (1.0 + ((wavelength - 10.45) / 0.02) ** 2)
    observed = sparse_mask(len(wavelength), 0.25, 17)
    metrics = qbic_metrics(wavelength, truth, truth.copy(), observed)
    assert metrics["global_hidden_nrmse"] < 1e-12
    assert metrics["resonance_window_nrmse"] < 1e-12
    assert metrics["resonance_center_error"] < 1e-12
    assert metrics["quality_factor_relative_error"] < 1e-12


def test_eis_loader_uses_explicit_frequency_axis():
    from frontier_science.eis import assert_frequency_resolved, load_figshare_eis_curves

    raw_root = ROOT / "frontier_science" / "data" / "raw" / "figshare_eis"
    curves = load_figshare_eis_curves(raw_root)
    assert len(curves) == 440
    assert_frequency_resolved(curves)
    assert all(len(curve.axis) == 28 for curve in curves)
    assert all(np.all(np.diff(curve.axis) > 0) for curve in curves)
    assert all(curve.response.shape == (28, 2) for curve in curves)


def test_eis_selection_uses_distinct_cells_and_fixed_soc():
    from frontier_science.eis import load_figshare_eis_curves, select_eis_instances

    raw_root = ROOT / "frontier_science" / "data" / "raw" / "figshare_eis"
    selected = select_eis_instances(load_figshare_eis_curves(raw_root))
    assert list(selected) == [
        "tune-0",
        "tune-1",
        "tune-2",
        "eval-0",
        "eval-1",
        "eval-2",
        "eval-3",
        "eval-4",
    ]
    assert len({curve.metadata["cell_id"] for curve in selected.values()}) == 8
    assert [selected[key].metadata["soc_percent"] for key in selected] == [
        25,
        50,
        75,
        10,
        30,
        50,
        70,
        90,
    ]


def test_eis_rejects_impedance_without_frequency():
    from frontier_science.eis import DataAuditError, assert_frequency_resolved
    from frontier_science.protocol import MeasuredCurve

    with pytest.raises(DataAuditError, match="frequency-resolved"):
        assert_frequency_resolved(
            [
                MeasuredCurve(
                    axis=None,
                    response=np.ones((8, 2)),
                    metadata={"cell_id": "bad"},
                )
            ]
        )


def test_eis_metrics_are_exact_for_truth():
    from frontier_science.eis import eis_metrics
    from frontier_science.qbic import sparse_mask

    frequency = np.logspace(-2, 3, 101)
    omega = 2 * np.pi * frequency
    impedance = 0.05 + 0.2 / (1.0 + 1j * omega * 0.5)
    truth = np.column_stack([impedance.real, impedance.imag])
    observed = sparse_mask(len(frequency), 0.25, 17)
    metrics = eis_metrics(frequency, truth, truth.copy(), observed)
    assert metrics["complex_nrmse"] < 1e-12
    assert metrics["nyquist_nrmse"] < 1e-12
    assert metrics["magnitude_mae"] < 1e-12
    assert metrics["phase_mae_rad"] < 1e-12


def test_frontier_inventory_has_three_unique_cfnn_candidates():
    from frontier_science.protocol import build_inventory, load_config

    config = load_config(ROOT / "frontier_science" / "config.json")
    inventory = build_inventory(config, ["qbic"], ["CFNN"], [256])
    candidates = inventory["qbic"]["CFNN"]["256"]
    assert len(candidates) == 3
    assert len({item["candidate_key"] for item in candidates}) == 3
    assert all(
        abs(item["actual_parameters"] - 256) / 256 <= 0.05
        for item in candidates
    )


def test_inventory_never_assigns_candidate_to_two_budgets():
    from frontier_science.protocol import build_inventory, load_config

    config = load_config(ROOT / "frontier_science" / "config.json")
    inventory = build_inventory(config, ["quantum"], ["CFNN"], config["budgets"])
    keys = [
        item["candidate_key"]
        for candidates in inventory["quantum"]["CFNN"].values()
        for item in candidates
    ]
    assert len(keys) == len(set(keys))


def test_freeze_selection_requires_complete_tuning_matrix():
    from frontier_science.protocol import freeze_selection

    records = [
        {
            "candidate_key": "candidate-a",
            "optimizer_key": "optimizer-a",
            "instance_id": "tune-0",
            "validation_score": 0.2,
            "status": "ok",
            "best_step": 300,
        }
    ]
    with pytest.raises(ValueError, match="incomplete tuning matrix"):
        freeze_selection(
            records,
            candidate_keys=["candidate-a", "candidate-b"],
            optimizer_keys=["optimizer-a"],
            instance_ids=["tune-0"],
        )


def test_freeze_selection_uses_median_validation_score():
    from frontier_science.protocol import freeze_selection

    records = []
    for candidate, values in {"a": [0.4, 0.2, 0.3], "b": [0.1, 0.5, 0.6]}.items():
        for index, value in enumerate(values):
            records.append(
                {
                    "candidate_key": candidate,
                    "optimizer_key": "adam",
                    "instance_id": f"tune-{index}",
                    "validation_score": value,
                    "status": "ok",
                    "best_step": 200 + 10 * index,
                }
            )
    selected = freeze_selection(
        records,
        candidate_keys=["a", "b"],
        optimizer_keys=["adam"],
        instance_ids=["tune-0", "tune-1", "tune-2"],
    )
    assert selected["candidate_key"] == "a"
    assert selected["selected_steps"] == 210


@pytest.mark.parametrize("task", ["quantum", "qbic", "eis"])
def test_one_cell_training_produces_finite_task_metric(task, tmp_path):
    from frontier_science.protocol import build_inventory, load_config
    from frontier_science.runner import train_candidate

    config = load_config(ROOT / "frontier_science" / "config.json")
    config["training"] = {
        **config["training"],
        "minimum_steps": 2,
        "maximum_steps": 5,
        "patience": 5,
        "validation_interval": 1,
        "checkpoints": [2, 5],
    }
    candidate = build_inventory(config, [task], ["CFNN"], [256])[task]["CFNN"]["256"][0]
    optimizer = config["optimizer_grids"]["CFNN"][1]
    record = train_candidate(
        task=task,
        family="CFNN",
        candidate=candidate,
        optimizer=optimizer,
        instance_id="tune-0",
        init_seed=1009,
        stage="smoke",
        config=config,
        raw_root=ROOT / "frontier_science" / "data" / "raw",
        evaluate_hidden=True,
    )
    assert record["status"] == "ok"
    assert record["actual_parameters"] == candidate["actual_parameters"]
    assert np.isfinite(record["validation_score"])
    assert np.isfinite(record["metrics"][record["primary_metric_name"]])
    assert record["optimizer_steps"] >= 2


def test_one_cell_training_is_seed_reproducible():
    from frontier_science.protocol import build_inventory, load_config
    from frontier_science.runner import train_candidate

    config = load_config(ROOT / "frontier_science" / "config.json")
    config["training"] = {
        **config["training"],
        "minimum_steps": 2,
        "maximum_steps": 3,
        "patience": 3,
        "validation_interval": 1,
        "checkpoints": [2, 3],
    }
    candidate = build_inventory(config, ["quantum"], ["CFNN"], [256])["quantum"]["CFNN"]["256"][0]
    arguments = dict(
        task="quantum",
        family="CFNN",
        candidate=candidate,
        optimizer=config["optimizer_grids"]["CFNN"][0],
        instance_id="tune-0",
        init_seed=1009,
        stage="smoke",
        config=config,
        raw_root=ROOT / "frontier_science" / "data" / "raw",
        evaluate_hidden=True,
    )
    first = train_candidate(**arguments)
    torch.rand(50)
    second = train_candidate(**arguments)
    assert first["validation_score"] == second["validation_score"]
    assert first["metrics"] == second["metrics"]


def test_tikhonov_returns_normalized_nonnegative_spectrum():
    from frontier_science.domain_baselines import quantum_tikhonov
    from frontier_science.quantum import make_quantum_instance

    instance = make_quantum_instance("test-single-pole", 0.0, 17, omega_points=401)
    result = quantum_tikhonov(instance, alpha=1e-4)
    assert result.status == "ok"
    assert np.all(result.prediction >= 0)
    assert abs(np.sum(result.prediction * instance.weights.numpy()) - 1.0) < 1e-6


def test_fano_fit_reports_invalid_when_points_are_insufficient():
    from frontier_science.domain_baselines import fit_fano

    axis = np.arange(4.0)
    response = np.ones(4)
    observed = np.array([True, False, False, True])
    result = fit_fano(axis, response, observed)
    assert result.status == "insufficient_points"
    assert result.prediction is None


def test_fano_fit_reconstructs_noiseless_line_shape():
    from frontier_science.domain_baselines import fano_line, fit_fano
    from frontier_science.qbic import sparse_mask

    axis = np.linspace(10.0, 11.0, 401)
    truth = fano_line(axis, 0.4, -0.3, 10.45, 0.04, 0.7, 0.01)
    observed = sparse_mask(len(axis), 0.25, 17)
    result = fit_fano(axis, truth, observed)
    assert result.status == "ok"
    assert np.sqrt(np.mean((result.prediction - truth) ** 2)) < 1e-3


def test_vector_response_and_drt_return_finite_predictions():
    from frontier_science.domain_baselines import estimate_drt, fit_vector_response
    from frontier_science.qbic import sparse_mask

    frequency = np.logspace(-2, 3, 101)
    impedance = 0.05 + 0.2 / (1.0 + 1j * 2 * np.pi * frequency * 0.5)
    response = np.column_stack([impedance.real, impedance.imag])
    observed = sparse_mask(len(frequency), 0.25, 17)
    vector = fit_vector_response(frequency, response, observed, poles=6)
    drt = estimate_drt(frequency, response, observed, alpha=1e-4)
    assert vector.status == "ok"
    assert vector.prediction.shape == response.shape
    assert np.all(np.isfinite(vector.prediction))
    assert drt["status"] == "ok"
    assert np.all(np.asarray(drt["gamma"]) >= 0)


def test_frontier_launcher_dry_run_lists_all_tasks():
    import subprocess

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "frontier_science" / "run_frontier_pilot.sh"),
            "--stage",
            "evaluate",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--tasks quantum" in result.stdout
    assert "--tasks qbic" in result.stdout
    assert "--tasks eis" in result.stdout


def test_smoke_record_validates_against_schema(tmp_path):
    import jsonschema

    from frontier_science.protocol import load_config
    from frontier_science.runner import run_smoke

    config = load_config(ROOT / "frontier_science" / "config.json")
    config["training"] = {
        **config["training"],
        "minimum_steps": 2,
        "maximum_steps": 3,
        "patience": 3,
        "validation_interval": 1,
        "checkpoints": [2, 3],
    }
    record = run_smoke(
        task="quantum",
        family="CFNN",
        budget=256,
        output_root=tmp_path,
        config=config,
        raw_root=ROOT / "frontier_science" / "data" / "raw",
    )
    schema = json.loads(
        (ROOT / "frontier_science" / "results_schema.json").read_text()
    )
    jsonschema.validate(record, schema)
    assert (tmp_path / "smoke" / "quantum" / "CFNN__256.jsonl").exists()


def test_evaluation_training_uses_all_observations_for_fixed_steps():
    from frontier_science.protocol import build_inventory, load_config
    from frontier_science.runner import train_candidate

    config = load_config(ROOT / "frontier_science" / "config.json")
    candidate = build_inventory(config, ["eis"], ["CFNN"], [256])["eis"]["CFNN"]["256"][0]
    record = train_candidate(
        task="eis",
        family="CFNN",
        candidate=candidate,
        optimizer=config["optimizer_grids"]["CFNN"][0],
        instance_id="eval-0",
        init_seed=1009,
        stage="evaluation",
        config=config,
        raw_root=ROOT / "frontier_science" / "data" / "raw",
        evaluate_hidden=True,
        fixed_steps=3,
    )
    assert record["optimizer_steps"] == 3
    assert record["best_step"] == 3
    assert record["training_role"] == "all_observed_fixed_steps"


@pytest.mark.parametrize(
    "task,method",
    [("quantum", "tikhonov"), ("qbic", "fano"), ("eis", "vector")],
)
def test_domain_record_validates_and_uses_task_metric(task, method):
    import jsonschema

    from frontier_science.protocol import load_config
    from frontier_science.runner import run_domain_record

    config = load_config(ROOT / "frontier_science" / "config.json")
    record = run_domain_record(
        task=task,
        method=method,
        instance_id="eval-0",
        config=config,
        raw_root=ROOT / "frontier_science" / "data" / "raw",
    )
    schema = json.loads(
        (ROOT / "frontier_science" / "results_schema.json").read_text()
    )
    jsonschema.validate(record, schema)
    assert record["status"] == "ok"
    assert np.isfinite(record["metrics"][record["primary_metric_name"]])
    if task != "quantum":
        assert len(record["provenance"]["observed_mask_sha256"]) == 64
