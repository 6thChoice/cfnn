import hashlib
import inspect
import json
import multiprocessing
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from jsonschema import ValidationError

from confirmatory_protocol import DatasetBundle
from confirmatory_models import VectorizedPARN, architecture_grid, build_model, enumerate_candidates
from confirmatory_protocol import seeded_build
from fair_baselines import parameter_count


ROOT = Path(__file__).resolve().parent
DIAGNOSTIC_PATH = ROOT / "peak_sensitive_results" / "diagnostics" / "initialization_scale.json"


def generate_initialization_diagnostic(path=DIAGNOSTIC_PATH):
    dimensions = [1, 8, 23]
    units = [2, 8, 32]
    seeds = list(range(30))
    records = []
    for input_dim in dimensions:
        base = torch.linspace(-1.0, 1.0, 257).unsqueeze(1)
        feature_scale = torch.linspace(1.0, 0.25, input_dim).unsqueeze(0)
        inputs = base * feature_scale
        for unit_count in units:
            for seed in seeds:
                common = {
                    "units": unit_count,
                    "degree": 3,
                    "epsilon": 0.1,
                    "denominator": "squared",
                    "shared_projection": False,
                    "skip": True,
                }
                fixed, _ = seeded_build(
                    seed,
                    lambda: build_model(
                        "CFNN", input_dim, 1,
                        {**common, "initialization_mode": "fixed"}, seed=seed,
                    ),
                )
                scaled, _ = seeded_build(
                    seed,
                    lambda: build_model(
                        "CFNN", input_dim, 1,
                        {**common, "initialization_mode": "numerator_scaled"}, seed=seed,
                    ),
                )
                with torch.no_grad():
                    fixed_output_std = fixed(inputs).std(unbiased=False).item()
                    scaled_output_std = scaled(inputs).std(unbiased=False).item()
                records.append({
                    "input_dim": input_dim,
                    "units": unit_count,
                    "seed": seed,
                    "fixed_output_std": fixed_output_std,
                    "scaled_output_std": scaled_output_std,
                })
    diagnostic = {
        "protocol": "cfnn-peak-sensitive-extension-2.0.0",
        "dimensions": dimensions,
        "units": units,
        "seeds": seeds,
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
    return diagnostic


class InitializationTests(unittest.TestCase):
    def _config(self, **overrides):
        return {
            "units": 16,
            "degree": 3,
            "epsilon": 0.1,
            "denominator": "squared",
            "shared_projection": False,
            "skip": True,
            **overrides,
        }

    def test_scaled_mode_changes_only_numerator_coefficients(self):
        fixed, fixed_meta = seeded_build(
            17,
            lambda: build_model("CFNN", 1, 1, self._config(initialization_mode="fixed"), seed=17),
        )
        scaled, scaled_meta = seeded_build(
            17,
            lambda: build_model(
                "CFNN", 1, 1, self._config(initialization_mode="numerator_scaled"), seed=17,
            ),
        )

        self.assertAlmostEqual(fixed_meta["numerator_init_std"], 0.05)
        self.assertAlmostEqual(fixed_meta["denominator_init_std"], 0.05)
        self.assertAlmostEqual(scaled_meta["numerator_init_std"], 0.05 / 4)
        self.assertAlmostEqual(scaled_meta["denominator_init_std"], 0.05)
        self.assertEqual(fixed_meta["initialization_mode"], "fixed")
        self.assertEqual(scaled_meta["initialization_mode"], "numerator_scaled")
        self.assertTrue(torch.equal(fixed.q_coeffs, scaled.q_coeffs))
        self.assertFalse(torch.equal(fixed.p_coeffs, scaled.p_coeffs))

    def test_seeded_cfnn_build_is_deterministic_and_counts_match(self):
        config = self._config(initialization_mode="numerator_scaled")
        first_model, first_meta = seeded_build(
            23, lambda: build_model("CFNN", 8, 2, config, seed=23)
        )
        second_model, second_meta = seeded_build(
            23, lambda: build_model("CFNN", 8, 2, config, seed=23)
        )

        self.assertEqual(first_meta, second_meta)
        self.assertEqual(parameter_count(first_model), parameter_count(second_model))
        self.assertEqual(first_model.state_dict().keys(), second_model.state_dict().keys())
        self.assertTrue(all(
            torch.equal(first_model.state_dict()[key], second_model.state_dict()[key])
            for key in first_model.state_dict()
        ))

    def test_forward_source_is_unchanged(self):
        source = inspect.getsource(VectorizedPARN.forward)
        self.assertEqual(
            hashlib.sha256(source.encode()).hexdigest(),
            "9ef4eb03b0b556b09502201d3c48fb0584e7b37ebf7e791111864f7ea75b21fc",
        )
        self.assertIn("rational_sum = torch.sum", source)
        self.assertNotIn("sqrt", source)

    def test_v1_cfnn_architecture_grid_is_unchanged(self):
        configs = list(architecture_grid("CFNN"))
        digest = hashlib.sha256(
            json.dumps(configs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(len(configs), 2880)
        self.assertEqual(
            digest,
            "63c8829829826e2af3099c650a50a6d00708dddb470da5ac0d9a4e52b8fc63f8",
        )

    def test_v2_grid_has_frozen_epsilon_and_initialization_axes(self):
        configs = list(architecture_grid("CFNN", protocol_grid="v2"))
        self.assertEqual(
            {config["epsilon"] for config in configs},
            {0.05, 0.1, 0.2, 0.3, 0.5},
        )
        self.assertEqual(
            {config["initialization_mode"] for config in configs},
            {"fixed", "numerator_scaled"},
        )
        self.assertEqual(
            len(configs),
            160 * 6 * 5 * 2,
        )

    def test_v2_candidate_configs_include_effective_initialization_metadata(self):
        candidates = enumerate_candidates("CFNN", 1, 1, protocol_grid="v2")
        selected = next(
            candidate for candidate in candidates
            if candidate.config["units"] == 16
            and candidate.config["degree"] == 3
            and candidate.config["epsilon"] == 0.1
            and candidate.config["initialization_mode"] == "numerator_scaled"
        )
        _, metadata = build_model("CFNN", 1, 1, selected.config, seed=17)
        self.assertEqual(metadata["initialization_mode"], "numerator_scaled")
        self.assertAlmostEqual(metadata["numerator_init_std"], 0.05 / 4)
        self.assertAlmostEqual(metadata["denominator_init_std"], 0.05)
        self.assertAlmostEqual(selected.config["numerator_init_std"], 0.05 / 4)
        self.assertAlmostEqual(selected.config["denominator_init_std"], 0.05)


class InitializationDiagnosticTests(unittest.TestCase):
    def test_diagnostic_contains_raw_values_and_scaled_spread_guardrail(self):
        self.assertTrue(DIAGNOSTIC_PATH.exists())
        diagnostic = json.loads(DIAGNOSTIC_PATH.read_text())
        self.assertEqual(diagnostic["dimensions"], [1, 8, 23])
        self.assertEqual(diagnostic["units"], [2, 8, 32])
        self.assertEqual(diagnostic["seeds"], list(range(30)))

        records = diagnostic["records"]
        self.assertEqual(len(records), 3 * 3 * 30)
        self.assertTrue(all("fixed_output_std" in record for record in records))
        self.assertTrue(all("scaled_output_std" in record for record in records))
        for dimension in diagnostic["dimensions"]:
            scaled_by_units = {
                units: [
                    record["scaled_output_std"]
                    for record in records
                    if record["input_dim"] == dimension and record["units"] == units
                ]
                for units in diagnostic["units"]
            }
            self.assertLessEqual(
                torch.tensor(scaled_by_units[32]).median().item(),
                2 * torch.tensor(scaled_by_units[2]).median().item(),
            )


class TrainingTests(unittest.TestCase):
    @staticmethod
    def _prepared_bundle():
        from downstream_training import prepare_bundle

        bundle = DatasetBundle(
            x_train=np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
            y_train=np.asarray([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32),
            x_validation=np.asarray([[1.5], [2.5]], dtype=np.float32),
            y_validation=np.asarray([[0.0], [0.0]], dtype=np.float32),
            x_test=np.asarray([[1.75]], dtype=np.float32),
            y_test=np.asarray([[0.0]], dtype=np.float32),
            metadata={},
        )
        return prepare_bundle(bundle, "regression", torch.device("cpu"))

    def test_custom_tensor_validation_scorer_controls_checkpoint_and_global_score(self):
        from downstream_training import fit_task_model

        prepared = self._prepared_bundle()
        primary_calls = []
        global_calls = []

        def primary(prediction, target):
            primary_calls.append((prediction.detach().clone(), target.detach().clone()))
            return torch.mean((prediction[:, :1] - target[:, :1]) ** 2)

        def global_score(prediction, target):
            global_calls.append(prediction.detach().clone())
            return torch.mean((prediction - target) ** 2)

        result = fit_task_model(
            torch.nn.Linear(1, 1),
            prepared,
            "regression",
            learning_rate=0.05,
            weight_decay=0.0,
            max_steps=6,
            min_steps=2,
            patience=100,
            checkpoints=(2, 6),
            gradient_clip=5.0,
            validation_scorer=primary,
            global_validation_scorer=global_score,
        )

        self.assertEqual(result.validation_evaluations, 6)
        self.assertEqual(len(primary_calls), result.validation_evaluations)
        self.assertEqual(len(global_calls), result.validation_evaluations)
        self.assertTrue(np.isfinite(result.best_validation_score))
        self.assertTrue(np.isfinite(result.best_global_validation_score))
        self.assertEqual(result.best_global_validation_score, min(
            float(global_score(prediction, target))
            for prediction, target in primary_calls
        ))

    def test_clip_telemetry_counts_preclip_norm_exceedances(self):
        from downstream_training import fit_task_model

        prepared = self._prepared_bundle()
        result = fit_task_model(
            torch.nn.Linear(1, 1),
            prepared,
            "regression",
            learning_rate=0.01,
            weight_decay=0.0,
            max_steps=4,
            min_steps=1,
            patience=100,
            checkpoints=(1, 4),
            gradient_clip=0.01,
        )

        self.assertGreater(result.gradient_clip_events, 0)
        self.assertEqual(result.gradient_clip_rate, result.gradient_clip_events / result.optimizer_steps)

    def test_global_scorer_does_not_stop_primary_checkpoint_selection(self):
        from downstream_training import fit_task_model

        result = fit_task_model(
            torch.nn.Linear(1, 1),
            self._prepared_bundle(),
            "regression",
            learning_rate=0.01,
            weight_decay=0.0,
            max_steps=4,
            min_steps=1,
            patience=100,
            checkpoints=(1, 4),
            gradient_clip=5.0,
            validation_scorer=lambda prediction, target: 0.0,
            global_validation_scorer=lambda prediction, target: float("inf"),
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.optimizer_steps, 4)
        self.assertTrue(np.isinf(result.best_global_validation_score))


class V2ScorerTests(unittest.TestCase):
    def test_v2_scorers_restore_unsorted_frequency_weights_to_original_rows(self):
        from downstream_benchmark import build_v2_validation_scorers
        from resonance_windows import WindowSpec

        spec = WindowSpec(
            "aluminium_frf", "log10_frequency", ((0.0, 1.0),),
            "train_validation", "ok", {},
        )
        frequencies = torch.tensor([10.0, 1.0, 100.0])
        resonance, global_score = build_v2_validation_scorers(
            spec, frequencies, log_weighted=True,
        )
        prediction = torch.ones((3, 2))
        target = torch.zeros_like(prediction)
        self.assertTrue(np.isfinite(resonance(prediction, target)))
        self.assertTrue(np.isfinite(global_score(prediction, target)))

    def test_v2_scorers_use_tensor_outputs_and_empty_required_windows_are_infinite(self):
        from downstream_benchmark import build_v2_validation_scorers
        from resonance_windows import WindowSpec

        spec = WindowSpec(
            "controlled_fano", "linear", ((0.4, 0.6),),
            "generator_metadata", "ok", {},
        )
        resonance, global_score = build_v2_validation_scorers(
            spec, torch.tensor([0.1, 0.9]),
        )
        prediction = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        target = torch.zeros_like(prediction)
        self.assertTrue(np.isinf(resonance(prediction, target)))
        self.assertTrue(np.isfinite(global_score(prediction, target)))

    def test_v2_no_feature_uses_global_checkpoint_score_but_is_excluded_from_peak_rank(self):
        from downstream_benchmark import build_v2_validation_scorers, rank_v2_candidates
        from resonance_windows import WindowSpec

        spec = WindowSpec(
            "controlled_fano", "linear", (),
            "no_identifiable_feature", "no_identifiable_feature", {},
        )
        resonance, global_score = build_v2_validation_scorers(
            spec, torch.tensor([0.1, 0.9]),
        )
        prediction = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        target = torch.zeros_like(prediction)
        self.assertEqual(resonance(prediction, target), global_score(prediction, target))

        records = [
            {"candidate_key": "a", "status": "ok", "window_status": "no_identifiable_feature",
             "validation_resonance_nrmse": 0.0, "validation_global_nrmse": 0.3},
            {"candidate_key": "a", "status": "ok", "window_status": "ok",
             "validation_resonance_nrmse": 0.4, "validation_global_nrmse": 0.2},
            {"candidate_key": "b", "status": "ok", "window_status": "ok",
             "validation_resonance_nrmse": 0.5, "validation_global_nrmse": 0.1},
            {"candidate_key": "b", "status": "ok", "window_status": "ok",
             "validation_resonance_nrmse": 0.5, "validation_global_nrmse": 0.1},
        ]
        self.assertEqual(rank_v2_candidates(records, 2), ["a", "b"])

    def test_v2_rank_key_orders_failures_then_missing_windows_then_scores(self):
        from downstream_benchmark import v2_candidate_rank_key

        key = v2_candidate_rank_key([
            {"candidate_key": "candidate", "status": "ok", "window_status": "empty_window",
             "validation_resonance_nrmse": 0.1, "validation_global_nrmse": 0.3},
            {"candidate_key": "candidate", "status": "nonfinite_train_loss", "window_status": "ok",
             "validation_resonance_nrmse": 0.0, "validation_global_nrmse": 0.0},
        ])
        self.assertEqual(key, (1, 1, float("inf"), 0.3, "candidate"))


class RunnerV2Tests(unittest.TestCase):
    def _synthetic_complete_tuning_root(self, output_root):
        from peak_sensitive_benchmark import (
            _append_jsonl,
            _config_hash,
            _file_hash,
            _final_path,
            _inventory_path,
            _protocol_hash,
            _screen_path,
            _selection_path,
            _write_json,
            audit_tuning,
            summarize_selection_rows,
        )
        from test_peak_sensitive_protocol import _base_record

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        task = "controlled_fano"
        families = [
            "MLP", "Fourier-feature MLP", "Gaussian RBF", "KAN",
            "Local nested CF control", "Rational activation NN",
        ]
        budgets = list(config["budgets"])
        densities = list(config["frf_densities"])
        protocol_sha256 = _protocol_hash(
            ROOT / "peak_sensitive_config.json", ROOT / "peak_sensitive_results_schema.json"
        )
        scope = {
            "tasks": [task], "families": families, "budgets": budgets,
            "frf_densities": densities,
        }
        inventory = {
            "protocol_key": config["protocol_key"],
            "protocol_mode": "production",
            "protocol_sha256": protocol_sha256,
            "config_sha256": _config_hash(config),
            "scope": scope,
            "task_shards": [{"task": task, "frf_density": None}],
            "family_eligibility": {family: {"status": "eligible", "reason": None} for family in families},
            "tasks": {task: {}},
        }
        for family in families:
            inventory["tasks"][task][family] = {}
            for budget in budgets:
                candidate_config = {"synthetic_budget": budget}
                candidate = {
                    "candidate_key": json.dumps(candidate_config, sort_keys=True, separators=(",", ":")),
                    "actual_parameters": budget,
                    "budget_relative_error": 0.0,
                    "config": candidate_config,
                }
                inventory["tasks"][task][family][str(budget)] = [candidate]
        _write_json(_inventory_path(output_root), inventory)
        inventory_hash = _file_hash(_inventory_path(output_root))
        budget_loss = {128: 1.2, 256: 1.0, 512: 1.02, 1024: 1.1}

        def record_for(family, budget, candidate, optimizer, seed, phase):
            from downstream_benchmark import _compact_metadata
            from peak_sensitive_benchmark import make_peak_sensitive_bundle

            record = _base_record("peak_sensitive_tuning")
            record.pop("selected_common_budget")
            index = config["tuning_data_seeds"].index(seed)
            record.update(
                family=family,
                target_budget=budget,
                data_seed=seed,
                init_seed=config["evaluation_init_seeds"][index],
                candidate_key=candidate["candidate_key"],
                config=candidate["config"],
                actual_parameters=candidate["actual_parameters"],
                optimizer=optimizer,
                phase=phase,
                validation_score=budget_loss[budget],
                validation_resonance_nrmse=budget_loss[budget],
                validation_global_nrmse=budget_loss[budget] + 0.1,
                tuning_seed=seed,
                batch_seed=seed + 49979687,
                candidate_manifest_sha256=inventory_hash,
                dataset=_compact_metadata(make_peak_sensitive_bundle(
                    task, seed, config, ROOT / "downstream_data" / "raw",
                    scientific_role="tuning", visibility="sealed",
                ).metadata),
            )
            if phase == "screening":
                record.update(
                    training_min_steps=500,
                    training_max_steps=500,
                    training_patience=500,
                    training_checkpoints=[20, 80, 200, 500],
                    optimizer_steps=500,
                    best_step=500,
                )
            return record

        for family in families:
            for budget in budgets:
                candidate = inventory["tasks"][task][family][str(budget)][0]
                screen_path = _screen_path(output_root, task, family, budget)
                final_path = _final_path(output_root, task, family, budget)
                screening_optimizer = config["screening"]["optimizer_choice_by_family"][family]
                for seed in config["tuning_data_seeds"]:
                    _append_jsonl(
                        screen_path,
                        record_for(family, budget, candidate, screening_optimizer, seed, "screening"),
                    )
                final_records = []
                for optimizer in config["optimizer_grids"][family]:
                    for seed in config["tuning_data_seeds"]:
                        record = record_for(family, budget, candidate, optimizer, seed, "convergence")
                        _append_jsonl(final_path, record)
                        final_records.append(record)
                selected_optimizer = min(
                    config["optimizer_grids"][family], key=lambda item: item["key"]
                )
                selected_rows = [row for row in final_records if row["optimizer"] == selected_optimizer]
                selection = {
                    "task": task,
                    "family": family,
                    "target_budget": budget,
                    "frf_density": None,
                    "status": "selected",
                    "candidate": {**candidate, "target_budget": budget},
                    "optimizer": selected_optimizer,
                    "selected_step": 500,
                    **summarize_selection_rows(selected_rows),
                    "screened_candidate_count": 1,
                    "finalist_count": 1,
                    "optimizer_count": len(config["optimizer_grids"][family]),
                    "tuning_seed_count": len(config["tuning_data_seeds"]),
                    "protocol_sha256": protocol_sha256,
                    "config_sha256": _config_hash(config),
                    "candidate_manifest_sha256": inventory_hash,
                    "screen_jsonl_sha256": _file_hash(screen_path),
                    "final_jsonl_sha256": _file_hash(final_path),
                }
                _write_json(_selection_path(output_root, task, family, budget), selection)
        audit = audit_tuning(
            config, inventory, output_root, tasks=[task], families=families,
            budgets=budgets, protocol_sha256=protocol_sha256, frf_densities=densities,
        )
        return config, inventory, protocol_sha256, families, budgets, audit

    def test_real_six_task_dispatch_and_all_frf_densities(self):
        from peak_sensitive_benchmark import make_peak_sensitive_bundle

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        raw_root = ROOT / "downstream_data" / "raw"
        for task in config["fixed_scientific_tasks"]:
            densities = config["frf_densities"] if task == "aluminium_frf" else [None]
            for density in densities:
                with self.subTest(task=task, density=density):
                    bundle = make_peak_sensitive_bundle(
                        task,
                        config["tuning_data_seeds"][0],
                        config,
                        raw_root,
                        scientific_role="tuning",
                        visibility="sealed",
                        frf_density=density,
                    )
                    self.assertGreater(len(bundle.x_train), 0)
                    self.assertEqual(len(bundle.y_test), 0)
                    if density is not None:
                        self.assertEqual(bundle.metadata["density"], density)

    def test_measured_seed_index_maps_to_explicit_distinct_curve_ids(self):
        from peak_sensitive_benchmark import make_peak_sensitive_bundle

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        raw_root = ROOT / "downstream_data" / "raw"
        measured = [
            "microwave_fano", "microstrip_resonator", "aluminium_frf",
            "figshare_battery_eis", "hybrid_supercapacitor_eis",
        ]
        for task in measured:
            roles = (("tuning", "tuning_data_seeds"), ("evaluation", "evaluation_data_seeds"))
            for role, seed_key in roles:
                curve_ids = []
                for seed in config[seed_key]:
                    bundle = make_peak_sensitive_bundle(
                        task, seed, config, raw_root,
                        scientific_role=role, visibility="sealed",
                        frf_density=24 if task == "aluminium_frf" else None,
                    )
                    curve_ids.append(bundle.metadata["curve_id"])
                with self.subTest(task=task, role=role):
                    self.assertEqual(len(curve_ids), len(config[seed_key]))
                    self.assertEqual(len(curve_ids), len(set(curve_ids)))

    def test_public_bundles_match_independent_record_identity_contract(self):
        from downstream_benchmark import _compact_metadata
        from peak_sensitive_benchmark import make_peak_sensitive_bundle
        from peak_sensitive_result_validation import _expected_dataset_identity

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        raw_root = ROOT / "downstream_data" / "raw"
        for task in config["fixed_scientific_tasks"]:
            for role, seed_key in (
                ("tuning", "tuning_data_seeds"),
                ("evaluation", "evaluation_data_seeds"),
            ):
                seed = config[seed_key][0]
                densities = config["frf_densities"] if task == "aluminium_frf" else [None]
                for density in densities:
                    with self.subTest(task=task, role=role, density=density):
                        bundle = make_peak_sensitive_bundle(
                            task, seed, config, raw_root,
                            scientific_role=role,
                            visibility="sealed" if role == "tuning" else "opened",
                            frf_density=density,
                        )
                        dataset = _compact_metadata(bundle.metadata)
                        expected, source_manifest_sha256 = _expected_dataset_identity({
                            "task": task,
                            "data_seed": seed,
                            "scientific_role": role,
                            "frf_density": density,
                        })
                        for field, value in expected.items():
                            self.assertEqual(dataset.get(field), value, field)
                        self.assertEqual(
                            dataset.get("source_manifest_sha256"), source_manifest_sha256
                        )

    def test_microwave_and_microstrip_verify_role_manifest_source_before_loading(self):
        from peak_sensitive_benchmark import make_peak_sensitive_bundle

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        cases = {
            "microwave_fano": (
                "microwave_fano/Fano Interference Data and Code.zip",
                "peak_sensitive_benchmark.load_microwave_fano",
            ),
            "microstrip_resonator": (
                "microstrip_resonator/Dataset.zip",
                "peak_sensitive_benchmark.load_microstrip_resonator",
            ),
        }
        for task, (relative_path, loader_name) in cases.items():
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary_directory:
                raw_root = Path(temporary_directory)
                source_path = raw_root / relative_path
                source_path.parent.mkdir(parents=True)
                source_path.write_bytes(b"wrong archive bytes")
                with patch(loader_name, side_effect=AssertionError("loader must not run")):
                    with self.assertRaisesRegex(ValueError, "source SHA-256 mismatch"):
                        make_peak_sensitive_bundle(
                            task, config["tuning_data_seeds"][0], config, raw_root,
                            scientific_role="tuning", visibility="sealed",
                        )

    def test_evaluation_window_bundle_and_opened_bundle_have_same_identity(self):
        from peak_sensitive_benchmark import assert_same_bundle_instance, make_peak_sensitive_bundle

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        raw_root = ROOT / "downstream_data" / "raw"
        tasks = ("aluminium_frf", "hybrid_supercapacitor_eis", "figshare_battery_eis")
        for task in tasks:
            kwargs = {"frf_density": 24} if task == "aluminium_frf" else {}
            seed = config["evaluation_data_seeds"][0]
            sealed = make_peak_sensitive_bundle(
                task, seed, config, raw_root,
                scientific_role="evaluation", visibility="sealed", **kwargs,
            )
            opened = make_peak_sensitive_bundle(
                task, seed, config, raw_root,
                scientific_role="evaluation", visibility="opened", **kwargs,
            )
            assert_same_bundle_instance(sealed, opened)
            self.assertEqual(sealed.metadata["curve_id"], opened.metadata["curve_id"])
            self.assertEqual(len(sealed.y_test), 0)
            self.assertGreater(len(opened.y_test), 0)

    def test_smoke_inventory_does_not_enumerate_full_candidate_grid(self):
        from peak_sensitive_benchmark import build_inventory

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "peak_sensitive_benchmark.enumerate_candidates",
                side_effect=AssertionError("full enumeration is forbidden in smoke"),
            ):
                inventory = build_inventory(
                    config,
                    Path(temporary_directory),
                    tasks=["controlled_fano"],
                    families=["CFNN"],
                    budgets=[128],
                    smoke=True,
                )
        self.assertGreaterEqual(
            len(inventory["tasks"]["controlled_fano"]["CFNN"]["128"]), 1
        )

    def test_inventory_persists_frf_density_shards_and_exact_scope(self):
        from peak_sensitive_benchmark import build_inventory

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        with tempfile.TemporaryDirectory() as temporary_directory:
            inventory = build_inventory(
                config, Path(temporary_directory),
                tasks=["controlled_fano", "aluminium_frf"], families=["MLP"],
                budgets=[128], frf_densities=[24, 48], smoke=True,
                protocol_sha256="a" * 64,
            )
        self.assertEqual(inventory["scope"]["frf_densities"], [24, 48])
        self.assertEqual(
            inventory["task_shards"],
            [
                {"task": "controlled_fano", "frf_density": None},
                {"task": "aluminium_frf", "frf_density": 24},
                {"task": "aluminium_frf", "frf_density": 48},
            ],
        )

    def test_sealed_split_never_indexes_hidden_response_rows(self):
        from peak_sensitive_benchmark import _split_curve

        data_seed = 547
        frequency = np.linspace(1.0, 40.0, 40)
        values = np.column_stack([frequency, -frequency])
        order = np.random.default_rng(data_seed + 32452843).permutation(len(frequency))
        allowed = set(order[:14])

        class GuardedResponse:
            def __getitem__(self, indices):
                requested = set(np.asarray(indices, dtype=int).reshape(-1).tolist())
                if not requested.issubset(allowed):
                    raise AssertionError("hidden test response was materialized")
                return values[indices]

        curve = type("Curve", (), {
            "curve_id": "guarded", "frequency": frequency,
            "response": GuardedResponse(), "conditions": {},
            "source_files": (), "metadata": {},
        })()
        bundle = _split_curve(
            curve, "figshare_battery_eis", data_seed,
            {"n_train": 7, "n_validation": 7},
            scientific_role="tuning", visibility="sealed",
        )
        self.assertEqual(len(bundle.y_test), 0)

    def test_public_sealed_adapters_never_materialize_hidden_target_rows(self):
        from peak_sensitive_benchmark import make_peak_sensitive_bundle
        from peak_sensitive_protocol import MeasuredCurve
        import downstream_protocol

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        raw_root = ROOT / "downstream_data" / "raw"
        seed = config["tuning_data_seeds"][0]

        controlled_lengths = []
        original_fano_response = downstream_protocol._fano_response

        def guarded_fano_response(frequency, parameters):
            controlled_lengths.append(len(frequency))
            if len(frequency) == config["tasks"]["controlled_fano"]["n_test"]:
                raise AssertionError("controlled hidden target was materialized")
            return original_fano_response(frequency, parameters)

        with patch("downstream_protocol._fano_response", side_effect=guarded_fano_response):
            bundle = make_peak_sensitive_bundle(
                "controlled_fano", seed, config, raw_root,
                scientific_role="tuning", visibility="sealed",
            )
        self.assertEqual(len(bundle.y_test), 0)
        self.assertNotIn(config["tasks"]["controlled_fano"]["n_test"], controlled_lengths)

        cases = {
            "microwave_fano": (1001, "overcoupled:r4:p28", "peak_sensitive_benchmark.load_microwave_fano"),
            "microstrip_resonator": (1882, "s16:cycle:008", "peak_sensitive_benchmark.load_microstrip_resonator"),
            "figshare_battery_eis": (28, "cell:B02:test:1:soc:025", "peak_sensitive_benchmark._battery_curve"),
            "hybrid_supercapacitor_eis": (81, "soc:000:temperature:-20", "peak_sensitive_benchmark.load_hybrid_supercapacitor_eis"),
        }
        for task, (point_count, curve_id, target) in cases.items():
            spec = config["tasks"][task]
            order = np.random.default_rng(seed + 32452843).permutation(point_count)
            allowed = set(order[:int(spec["n_train"]) + int(spec["n_validation"])])
            values = np.zeros((point_count, 2), dtype=np.float32)

            class GuardedResponse:
                def __getitem__(self, indices):
                    requested = set(np.asarray(indices, dtype=int).reshape(-1).tolist())
                    if not requested.issubset(allowed):
                        raise AssertionError(f"{task} hidden target was materialized")
                    return values[indices]

            curve = MeasuredCurve(
                curve_id=curve_id,
                frequency=np.linspace(1.0, float(point_count), point_count),
                response=GuardedResponse(),
                conditions={}, source_files=("source",), metadata={},
            )
            returned = curve if task == "figshare_battery_eis" else [curve]
            with self.subTest(task=task), patch(target, return_value=returned):
                bundle = make_peak_sensitive_bundle(
                    task, seed, config, raw_root,
                    scientific_role="tuning", visibility="sealed",
                )
                self.assertEqual(len(bundle.y_test), 0)

        frf_requests = []
        frf_values = np.zeros((2048, 2), dtype=np.float32)

        class GuardedFrfResponse:
            def __getitem__(self, indices):
                requested = np.asarray(indices, dtype=int).reshape(-1)
                frf_requests.append(len(requested))
                if len(requested) > 96:
                    raise AssertionError("FRF hidden target was materialized")
                return frf_values[indices]

        frf_curve = MeasuredCurve(
            curve_id="point:03", frequency=np.arange(1, 2049, dtype=np.float64),
            response=GuardedFrfResponse(), conditions={},
            source_files=("Waveforms Point03.tdms",), metadata={},
        )
        frf_audit = {
            "eligible_curve_ids": ["point:03"], "excluded_curves": [],
            "source_manifest_sha256": "0" * 64,
        }
        with patch("peak_sensitive_protocol.audit_frf_density_sources", return_value=frf_audit), patch(
            "peak_sensitive_protocol.load_dense_aluminium_frf", return_value=[frf_curve]
        ):
            bundle = make_peak_sensitive_bundle(
                "aluminium_frf", seed, config, raw_root,
                scientific_role="tuning", visibility="sealed", frf_density=24,
            )
        self.assertEqual(len(bundle.y_test), 0)
        self.assertEqual(frf_requests, [24, 12])

    def test_evaluate_shard_rejects_mismatched_opened_adapter_before_training(self):
        from peak_sensitive_benchmark import evaluate_shard, freeze_budgets
        from resonance_windows import WindowSpec

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, inventory, protocol_sha256, families, budgets, _ = (
                self._synthetic_complete_tuning_root(root)
            )
            freeze_budgets(
                config, inventory, root, tasks=["controlled_fano"], families=families,
                budgets=budgets, protocol_sha256=protocol_sha256,
                frf_densities=config["frf_densities"],
            )

            def bundle_for_visibility(task, data_seed, config, raw_root, *,
                                      scientific_role, visibility, frf_density=None):
                curve_id = "sealed-curve" if visibility == "sealed" else "opened-curve"
                empty = np.empty((0, 1), dtype=np.float32)
                return DatasetBundle(
                    x_train=np.ones((2, 1), dtype=np.float32),
                    y_train=np.ones((2, 2), dtype=np.float32),
                    x_validation=np.ones((2, 1), dtype=np.float32),
                    y_validation=np.ones((2, 2), dtype=np.float32),
                    x_test=empty if visibility == "sealed" else np.ones((2, 1), dtype=np.float32),
                    y_test=np.empty((0, 2), dtype=np.float32) if visibility == "sealed" else np.ones((2, 2), dtype=np.float32),
                    metadata={
                        "task": task, "curve_id": curve_id, "data_seed": int(data_seed),
                        "scientific_role": scientific_role, "density": frf_density,
                        "train_point_ids": ["train:0", "train:1"],
                        "validation_point_ids": ["validation:0", "validation:1"],
                        "test_point_ids": ["test:0", "test:1"],
                    },
                )

            window = WindowSpec(
                "controlled_fano", "linear", ((0.4, 0.6),),
                "generator_metadata", "ok", {},
            )
            with patch(
                "peak_sensitive_benchmark.make_peak_sensitive_bundle",
                side_effect=bundle_for_visibility,
            ), patch("peak_sensitive_benchmark._window_for_bundle", return_value=window), patch(
                "peak_sensitive_benchmark.train_one",
                side_effect=AssertionError("training must not start"),
            ):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    evaluate_shard(
                        config, inventory, root, "controlled_fano", families[0],
                        ROOT / "downstream_data" / "raw", protocol_sha256,
                    )

    def test_protocol_root_rejects_changed_config_code_or_input_hashes(self):
        from peak_sensitive_benchmark import ensure_protocol_root

        manifest = {
            "protocol_key": "key", "protocol_sha256": "a" * 64,
            "config_sha256": "b" * 64, "protocol_mode": "production",
            "smoke_root_marker": None,
            "code_files": [{"path": "code.py", "sha256": "c" * 64}],
            "input_files": [{"path": "input.json", "sha256": "d" * 64}],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ensure_protocol_root(root, manifest)
            ensure_protocol_root(root, dict(manifest))
            for field, value in (
                ("config_sha256", "0" * 64),
                ("code_files", [{"path": "code.py", "sha256": "0" * 64}]),
                ("input_files", [{"path": "input.json", "sha256": "0" * 64}]),
            ):
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "config, code, or input"):
                        ensure_protocol_root(root, {**manifest, field: value})

    def test_protocol_root_rejects_distinct_canonical_raw_roots(self):
        from peak_sensitive_benchmark import _protocol_manifest, ensure_protocol_root

        config_path = ROOT / "peak_sensitive_config.json"
        schema_path = ROOT / "peak_sensitive_results_schema.json"
        config = json.loads(config_path.read_text())
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            first_raw, second_raw = base / "raw-a", base / "raw-b"
            for raw_root in (first_raw, second_raw):
                raw_root.mkdir()
                (raw_root / "source_manifest.json").write_text('{"files": []}\n')
            first = _protocol_manifest(
                config, config_path, schema_path, ["runner"],
                raw_root=first_raw, tasks=["controlled_fano"],
            )
            second = _protocol_manifest(
                config, config_path, schema_path, ["runner"],
                raw_root=second_raw, tasks=["controlled_fano"],
            )
            output_root = base / "output"
            output_root.mkdir()
            ensure_protocol_root(output_root, first)
            with self.assertRaisesRegex(ValueError, "raw root"):
                ensure_protocol_root(output_root, second)

    def test_production_manifest_raw_inputs_are_invariant_to_shard_task_filter(self):
        from peak_sensitive_benchmark import _manifest_tasks

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        self.assertEqual(
            _manifest_tasks(config, ["controlled_fano"]),
            config["fixed_scientific_tasks"],
        )
        smoke = {**config, "protocol_mode": "smoke"}
        self.assertEqual(_manifest_tasks(smoke, ["controlled_fano"]), ["controlled_fano"])

    def test_common_budget_uses_pooled_family_relative_loss(self):
        from peak_sensitive_benchmark import select_common_budget

        losses = {
            "CFNN": {128: 2.0, 256: 1.0, 512: 0.9, 1024: 0.9},
            "MLP": {128: 2.2, 256: 1.1, 512: 1.0, 1024: 0.99},
            "KAN": {128: 2.4, 256: 1.2, 512: 1.0, 1024: 1.0},
            "SIREN": {128: 2.0, 256: 1.1, 512: 1.0, 1024: 1.0},
            "A": {128: 2.0, 256: 1.1, 512: 1.0, 1024: 1.0},
            "B": {128: 2.0, 256: 1.1, 512: 1.0, 1024: 1.0},
        }
        self.assertEqual(
            select_common_budget(losses, tolerance=0.05, minimum_families=6), 512
        )

    def test_common_budget_uses_each_familys_available_budgets(self):
        from peak_sensitive_benchmark import analyze_common_budget

        losses = {
            "A": {128: 1.0, 256: 0.8},
            "B": {128: 1.0, 256: 0.8},
            "C": {128: 1.0, 256: 0.8},
            "D": {128: 1.0, 256: 0.8},
            "E": {128: 1.0, 256: 0.8},
            "F": {128: 1.0, 256: 0.8},
            "G": {256: 0.7, 512: 0.6},
        }
        analysis = analyze_common_budget(
            losses, budgets=[128, 256, 512], minimum_families=6
        )
        self.assertEqual(analysis["selected_common_budget"], 256)
        self.assertEqual(analysis["available_family_count"][512], 1)
        self.assertEqual(analysis["relative_losses"]["G"][512], 1.0)

    def test_common_budget_excludes_no_feature_peak_but_retains_audit_state(self):
        from peak_sensitive_benchmark import summarize_selection_rows

        rows = [
            {"status": "ok", "window_spec": {"status": "ok"},
             "validation_resonance_nrmse": 0.4, "validation_global_nrmse": 0.5},
            {"status": "ok", "window_spec": {"status": "no_identifiable_feature"},
             "validation_resonance_nrmse": 9.0, "validation_global_nrmse": 0.6},
            {"status": "nonfinite_gradient", "window_spec": {"status": "ok"},
             "validation_resonance_nrmse": None, "validation_global_nrmse": None},
        ]
        summary = summarize_selection_rows(rows)
        self.assertEqual(summary["median_validation_resonance_nrmse"], 0.4)
        self.assertEqual(summary["no_feature_count"], 1)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["global_available_count"], 2)

    def test_tuning_record_rejects_test_access_and_duplicate_keys(self):
        from peak_sensitive_benchmark import assert_unique_records, validate_tuning_record

        record = {
            "stage": "peak_sensitive_tuning",
            "stage_role": "validation_only",
            "dataset": {"test_access": "sealed_for_validation_only_stage"},
            "candidate_key": "candidate",
            "optimizer": {"key": "opt"},
            "tuning_seed": 547,
        }
        validate_tuning_record(record)
        assert_unique_records([record])
        with self.assertRaisesRegex(ValueError, "test metrics"):
            validate_tuning_record({**record, "test_metrics": {}})
        with self.assertRaisesRegex(ValueError, "test access"):
            validate_tuning_record({
                **record, "dataset": {"test_access": "opened_confirmatory_test"},
            })
        with self.assertRaisesRegex(ValueError, "duplicate"):
            assert_unique_records([record, dict(record)])

    def test_evaluation_requires_frozen_budget_and_tuning_audit(self):
        from peak_sensitive_benchmark import require_evaluation_prerequisites

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "frozen task budgets"):
                require_evaluation_prerequisites(root, "controlled_fano")
            (root / "frozen_task_budgets.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "tuning audit"):
                require_evaluation_prerequisites(root, "controlled_fano")

    def test_evaluation_rejects_partial_or_hash_mismatched_audit(self):
        from peak_sensitive_benchmark import require_evaluation_prerequisites

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inventory = {"tasks": {}}
            (root / "candidate_manifest.json").write_text(json.dumps(inventory))
            (root / "tuning_audit.json").write_text(json.dumps({
                "status": "passed", "tasks": ["other"], "families": ["CFNN"],
                "budgets": [128], "config_sha256": "0" * 64,
            }))
            (root / "frozen_task_budgets.json").write_text(json.dumps({
                "tasks": {"controlled_fano": {"selected_common_budget": 128}}
            }))
            with self.assertRaises(ValueError):
                require_evaluation_prerequisites(
                    root, "controlled_fano", config=config, inventory=inventory
                )

    def test_synthetic_complete_audit_freeze_and_evaluation_gate(self):
        from peak_sensitive_benchmark import freeze_budgets, require_evaluation_prerequisites

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, inventory, protocol_sha256, families, budgets, audit = (
                self._synthetic_complete_tuning_root(root)
            )
            self.assertEqual(audit["status"], "passed")
            frozen = freeze_budgets(
                config, inventory, root, tasks=["controlled_fano"], families=families,
                budgets=budgets, protocol_sha256=protocol_sha256,
                frf_densities=config["frf_densities"],
            )
            task = frozen["tasks"]["controlled_fano"]
            self.assertEqual(task["selected_common_budget"], 256)
            self.assertEqual(task["available_family_count"][256], 6)
            self.assertTrue(frozen["input_jsonl_sha256"])
            self.assertTrue(frozen["selection_file_sha256"])
            self.assertEqual(
                require_evaluation_prerequisites(
                    root, "controlled_fano", config=config, inventory=inventory,
                    protocol_sha256=protocol_sha256,
                ),
                256,
            )

    def test_audit_rejects_semantically_invalid_loaded_record(self):
        from peak_sensitive_benchmark import _final_path, audit_tuning

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, inventory, protocol_sha256, families, budgets, _ = (
                self._synthetic_complete_tuning_root(root)
            )
            path = _final_path(root, "controlled_fano", families[0], budgets[0])
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]["actual_parameters"] = 1
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            with self.assertRaisesRegex(ValueError, "tuning audit failed"):
                audit_tuning(
                    config, inventory, root, tasks=["controlled_fano"], families=families,
                    budgets=budgets, protocol_sha256=protocol_sha256,
                    frf_densities=config["frf_densities"],
                )

    def test_full_audit_rejects_curve_and_candidate_manifest_identity_mutations(self):
        from peak_sensitive_benchmark import (
            _file_hash,
            _final_path,
            _selection_path,
            _write_json,
            audit_tuning,
        )

        mutations = {
            "curve_id": lambda record: record["dataset"].__setitem__(
                "curve_id", "controlled_fano:seed:999999"
            ),
            "candidate_manifest_sha256": lambda record: record.__setitem__(
                "candidate_manifest_sha256", "0" * 64
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                config, inventory, protocol_sha256, families, budgets, _ = (
                    self._synthetic_complete_tuning_root(root)
                )
                task, family, budget = "controlled_fano", families[0], budgets[0]
                final_path = _final_path(root, task, family, budget)
                records = [json.loads(line) for line in final_path.read_text().splitlines()]
                mutate(records[0])
                final_path.write_text(
                    "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
                )
                selection_path = _selection_path(root, task, family, budget)
                selection = json.loads(selection_path.read_text())
                selection["final_jsonl_sha256"] = _file_hash(final_path)
                _write_json(selection_path, selection)

                with self.assertRaisesRegex(ValueError, "tuning audit failed"):
                    audit_tuning(
                        config, inventory, root, tasks=[task], families=families,
                        budgets=budgets, protocol_sha256=protocol_sha256,
                        frf_densities=config["frf_densities"],
                    )

    def test_audit_rejects_valid_non_winning_optimizer_selection(self):
        from peak_sensitive_benchmark import (
            _file_hash,
            _final_path,
            _selection_path,
            _write_json,
            audit_tuning,
            summarize_selection_rows,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, inventory, protocol_sha256, families, budgets, _ = (
                self._synthetic_complete_tuning_root(root)
            )
            task, family, budget = "controlled_fano", families[0], budgets[0]
            selection_path = _selection_path(root, task, family, budget)
            final_path = _final_path(root, task, family, budget)
            selection = json.loads(selection_path.read_text())
            non_winner = config["optimizer_grids"][family][-1]
            self.assertNotEqual(selection["optimizer"], non_winner)
            rows = [json.loads(line) for line in final_path.read_text().splitlines()]
            selected_rows = [row for row in rows if row["optimizer"] == non_winner]
            selection.update(
                optimizer=non_winner,
                selected_step=int(round(np.median([row["best_step"] for row in selected_rows]))),
                final_jsonl_sha256=_file_hash(final_path),
                **summarize_selection_rows(selected_rows),
            )
            _write_json(selection_path, selection)

            with self.assertRaisesRegex(ValueError, "tuning audit failed"):
                audit_tuning(
                    config, inventory, root, tasks=[task], families=families,
                    budgets=budgets, protocol_sha256=protocol_sha256,
                    frf_densities=config["frf_densities"],
                )

    def test_evaluation_gate_rejects_one_field_budget_mutation(self):
        from peak_sensitive_benchmark import (
            _write_json,
            freeze_budgets,
            require_evaluation_prerequisites,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, inventory, protocol_sha256, families, budgets, _ = (
                self._synthetic_complete_tuning_root(root)
            )
            frozen = freeze_budgets(
                config, inventory, root, tasks=["controlled_fano"], families=families,
                budgets=budgets, protocol_sha256=protocol_sha256,
                frf_densities=config["frf_densities"],
            )
            self.assertEqual(frozen["tasks"]["controlled_fano"]["selected_common_budget"], 256)
            frozen["tasks"]["controlled_fano"]["selected_common_budget"] = 512
            _write_json(root / "frozen_task_budgets.json", frozen)

            with self.assertRaisesRegex(ValueError, "frozen budget artifact"):
                require_evaluation_prerequisites(
                    root, "controlled_fano", config=config, inventory=inventory,
                    protocol_sha256=protocol_sha256,
                )

    def test_evaluation_resume_rejects_wrong_candidate_before_training(self):
        from peak_sensitive_benchmark import (
            _evaluation_path,
            _file_hash,
            _inventory_path,
            _selection_path,
            evaluate_shard,
            freeze_budgets,
        )
        from test_peak_sensitive_protocol import _base_record

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, inventory, protocol_sha256, families, budgets, _ = (
                self._synthetic_complete_tuning_root(root)
            )
            freeze_budgets(
                config, inventory, root, tasks=["controlled_fano"], families=families,
                budgets=budgets, protocol_sha256=protocol_sha256,
                frf_densities=config["frf_densities"],
            )
            family = families[0]
            budget = 256
            selection_path = _selection_path(root, "controlled_fano", family, budget)
            selection = json.loads(selection_path.read_text())
            record = _base_record("peak_sensitive_evaluation")
            candidate = selection["candidate"]
            record.update(
                family=family,
                target_budget=budget,
                selected_common_budget=budget,
                candidate_key="{}",
                config=candidate["config"],
                actual_parameters=candidate["actual_parameters"],
                optimizer=selection["optimizer"],
                candidate_manifest_sha256=_file_hash(_inventory_path(root)),
                selection_path=str(selection_path),
                selection_sha256=_file_hash(selection_path),
            )
            path = _evaluation_path(root, "controlled_fano", family, budget)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record) + "\n")
            with patch("peak_sensitive_benchmark.train_one", side_effect=AssertionError("must not train")):
                with self.assertRaisesRegex(ValueError, "candidate_key"):
                    evaluate_shard(
                        config, inventory, root, "controlled_fano", family,
                        ROOT / "downstream_data" / "raw", protocol_sha256,
                    )

    def test_evaluation_resume_rejects_identity_hash_and_metric_mutations(self):
        from peak_sensitive_benchmark import (
            _evaluation_path,
            _file_hash,
            _inventory_path,
            _selection_path,
            evaluate_shard,
            freeze_budgets,
        )
        from test_peak_sensitive_protocol import _base_record

        def set_nested(record, *path_and_value):
            *path, value = path_and_value
            target = record
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value

        mutations = {
            "curve_id": lambda row: set_nested(
                row, "dataset", "curve_id", "controlled_fano:seed:999999"
            ),
            "point_identity": lambda row: set_nested(
                row, "dataset", "train_point_ids_sha256", "0" * 64
            ),
            "candidate_manifest": lambda row: set_nested(
                row, "candidate_manifest_sha256", "0" * 64
            ),
            "role_manifest": lambda row: set_nested(
                row, "dataset", "role_manifest_sha256", "0" * 64
            ),
            "source_manifest": lambda row: set_nested(
                row, "source_manifest_sha256", "0" * 64
            ),
            "rate_above_one": lambda row: set_nested(
                row, "test_metrics", "underfit_rate", 7.0
            ),
            "rate_below_zero": lambda row: set_nested(
                row, "test_metrics", "numerical_failure_rate", -1.0
            ),
            "non_eis_metric": lambda row: set_nested(
                row, "test_metrics", "kramers_kronig_consistency_proxy", 0.2
            ),
            "underfit_consistency": lambda row: set_nested(
                row, "test_metrics", "underfit_rate", 1.0
            ),
            "failure_consistency": lambda row: set_nested(
                row, "test_metrics", "numerical_failure_rate", 1.0
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                config, inventory, protocol_sha256, families, budgets, _ = (
                    self._synthetic_complete_tuning_root(root)
                )
                freeze_budgets(
                    config, inventory, root, tasks=["controlled_fano"], families=families,
                    budgets=budgets, protocol_sha256=protocol_sha256,
                    frf_densities=config["frf_densities"],
                )
                family, budget = families[0], 256
                selection_path = _selection_path(root, "controlled_fano", family, budget)
                selection = json.loads(selection_path.read_text())
                candidate = selection["candidate"]
                record = _base_record("peak_sensitive_evaluation")
                record.update(
                    family=family,
                    target_budget=budget,
                    selected_common_budget=budget,
                    candidate_key=candidate["candidate_key"],
                    config=candidate["config"],
                    actual_parameters=candidate["actual_parameters"],
                    optimizer=selection["optimizer"],
                    candidate_manifest_sha256=_file_hash(_inventory_path(root)),
                    selection_path=str(selection_path),
                    selection_sha256=_file_hash(selection_path),
                )
                mutate(record)
                path = _evaluation_path(root, "controlled_fano", family, budget)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, sort_keys=True) + "\n")

                with patch(
                    "peak_sensitive_benchmark.train_one",
                    side_effect=AssertionError("must not train"),
                ):
                    with self.assertRaises((ValueError, ValidationError)):
                        evaluate_shard(
                            config, inventory, root, "controlled_fano", family,
                            ROOT / "downstream_data" / "raw", protocol_sha256,
                        )

    def test_corrupt_existing_evaluation_jsonl_is_rejected_before_resume(self):
        from peak_sensitive_benchmark import _load_jsonl

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "evaluation.jsonl"
            path.write_text('{"valid": true}\n{"truncated":')
            with self.assertRaisesRegex(ValueError, "line 2"):
                _load_jsonl(path)

    def test_per_shard_lock_prevents_duplicate_read_check_append(self):
        from peak_sensitive_benchmark import locked_append_once

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "records.jsonl"
            record = {
                "candidate_key": "same", "optimizer": {"key": "opt"},
                "tuning_seed": 547, "phase": "screening",
            }
            processes = [
                multiprocessing.Process(
                    target=locked_append_once,
                    args=(path, record),
                    kwargs={"delay_seconds": 0.1},
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 1)

    def test_scientific_metrics_report_q_kk_and_frozen_underfit_applicability(self):
        from peak_sensitive_benchmark import scientific_metrics
        from resonance_windows import WindowSpec

        frequency = np.linspace(1.0, 10.0, 401)
        truth_complex = 1.0 / (1.0 + 1j * (frequency - 5.0) / 0.25)
        pred_complex = 1.0 / (1.0 + 1j * (frequency - 5.0) / 0.5)
        truth = np.column_stack([truth_complex.real, truth_complex.imag])
        prediction = np.column_stack([pred_complex.real, pred_complex.imag])
        detector = {
            "smoothing": {"method": "savgol", "window_length": 5, "polyorder": 2},
            "prominence_fraction": 0.05, "minimum_distance_observed_spacings": 2,
            "max_features": 3,
        }
        window = WindowSpec(
            "aluminium_frf", "linear", ((3.0, 7.0),),
            "train_validation", "ok", detector,
        )
        metrics = scientific_metrics(
            "aluminium_frf", frequency, truth, prediction, window,
            training_status="ok", validation_global_nrmse=1.2,
        )
        self.assertIsNotNone(metrics["quality_factor_or_bandwidth_error"])
        self.assertIsNone(metrics["kramers_kronig_consistency_proxy"])
        self.assertEqual(metrics["underfit_rate"], 1.0)
        failed = scientific_metrics(
            "aluminium_frf", frequency, truth, prediction, window,
            training_status="nonfinite_gradient", validation_global_nrmse=None,
        )
        self.assertIsNone(failed["underfit_rate"])

        eis_frequency = np.logspace(-2, 3, 401)
        eis_window = WindowSpec(
            "figshare_battery_eis", "log10_frequency", ((-1.0, 2.0),),
            "train_validation", "ok", detector,
        )
        eis = scientific_metrics(
            "figshare_battery_eis", eis_frequency, truth, prediction, eis_window,
            training_status="ok", validation_global_nrmse=0.2,
        )
        self.assertIsNotNone(eis["kramers_kronig_consistency_proxy"])

    def test_schema_valid_smoke_record(self):
        from peak_sensitive_benchmark import make_smoke_record
        from peak_sensitive_result_validation import validate_peak_sensitive_record

        validate_peak_sensitive_record(make_smoke_record())


class ArtifactV2Tests(unittest.TestCase):
    """Fast Task8 unit contracts; canonical integration lives in its focused module."""

    def test_advantage_requires_peak_superiority_and_global_noninferiority(self):
        from build_peak_sensitive_artifacts import advantage_decision

        self.assertEqual(
            advantage_decision((-0.2, -0.05), (-0.03, 0.08), 0.0, 4),
            "supported",
        )
        self.assertEqual(
            advantage_decision((-0.2, -0.05), (-0.03, 0.12), 0.0, 4),
            "peak_only_global_guardrail_failed",
        )

    def test_builder_rejects_test_derived_window_through_production_validation(self):
        from build_peak_sensitive_artifacts import validate_scientific_record

        record = json.loads((ROOT / "test_fixtures" / "peak_sensitive_valid_record.json").read_text())
        record["window_spec"]["source_role"] = "test"
        with self.assertRaisesRegex(ValueError, "test-derived window"):
            validate_scientific_record(record)

    def test_hierarchical_bootstrap_resamples_instances_then_initializations(self):
        from build_peak_sensitive_artifacts import hierarchical_paired_bootstrap

        values = {719: {11003: -0.2, 12007: -0.1}, 773: {11003: 0.1, 12007: 0.2}}
        first = hierarchical_paired_bootstrap(values, seed_parts=("unit", "peak"), replicates=200)
        second = hierarchical_paired_bootstrap(values, seed_parts=("unit", "peak"), replicates=200)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 0.0)
        self.assertLess(first["ci_low"], 0.0)
        self.assertGreater(first["ci_high"], 0.0)

    def test_frf_interaction_fits_effect_per_log2_train_points_with_bootstrap(self):
        from build_peak_sensitive_artifacts import frf_density_interaction

        values = {
            24: {719: {11003: 0.0, 12007: 0.0}, 773: {11003: 0.0, 12007: 0.0}},
            48: {719: {11003: -0.1, 12007: -0.1}, 773: {11003: -0.1, 12007: -0.1}},
            96: {719: {11003: -0.2, 12007: -0.2}, 773: {11003: -0.2, 12007: -0.2}},
            192: {719: {11003: -0.3, 12007: -0.3}, 773: {11003: -0.3, 12007: -0.3}},
        }
        interaction = frf_density_interaction(values, seed_parts=("unit", "FRF"), replicates=200)
        self.assertAlmostEqual(interaction["effect_per_log2_train_points"], -0.1, places=12)
        self.assertAlmostEqual(interaction["ci_low"], -0.1, places=12)
        self.assertAlmostEqual(interaction["ci_high"], -0.1, places=12)

    def test_generated_smoke_records_are_schema_valid_and_validation_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = subprocess.run(
                [
                    "bash", "run_peak_sensitive_experiments.sh", "--stage", "tune",
                    "--tasks", "controlled_fano", "--families", "CFNN,MLP",
                    "--budgets", "128", "--jobs", "1", "--smoke",
                    "--output-root", temporary_directory,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            from peak_sensitive_benchmark import _load_jsonl
            from peak_sensitive_result_validation import validate_peak_sensitive_record

            paths = Path(temporary_directory, "smoke", "tuning_final", "controlled_fano").glob("*.jsonl")
            records = [record for path in paths for record in _load_jsonl(path)]
            self.assertTrue(records)
            for record in records:
                validate_peak_sensitive_record(record)
                self.assertNotIn("test_metrics", record)
                self.assertEqual(record["dataset"]["test_access"], "sealed_for_validation_only_stage")

    def test_resume_does_not_append_duplicate_tuning_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            command = [
                "bash", "run_peak_sensitive_experiments.sh", "--stage", "tune",
                "--tasks", "controlled_fano", "--families", "MLP", "--budgets", "128",
                "--jobs", "1", "--smoke", "--output-root", temporary_directory,
            ]
            for _ in range(2):
                subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            path = Path(temporary_directory, "smoke", "tuning_final", "controlled_fano", "mlp__128.jsonl")
            records = [json.loads(line) for line in path.read_text().splitlines() if line]
            keys = [(row["candidate_key"], row["optimizer"]["key"], row["tuning_seed"]) for row in records]
            self.assertEqual(len(keys), len(set(keys)))

    def test_tune_stage_does_not_create_evaluation_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            subprocess.run(
                [
                    "bash", "run_peak_sensitive_experiments.sh", "--stage", "tune",
                    "--tasks", "controlled_fano", "--families", "MLP", "--budgets", "128",
                    "--jobs", "1", "--smoke", "--output-root", temporary_directory,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(Path(temporary_directory, "smoke", "evaluation").exists())

    def test_launcher_dry_run_uses_v2_roots_and_all_families(self):
        result = subprocess.run(
            ["bash", "run_peak_sensitive_experiments.sh", "--dry-run", "--stage", "tune"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("peak_sensitive_config.json", result.stdout)
        self.assertIn("hybrid_supercapacitor_eis", result.stdout)
        self.assertIn("Local\\ nested\\ CF\\ control", result.stdout)
        self.assertIn("peak_sensitive_results", result.stdout)
        self.assertIn("peak_sensitive_preflight.py verify", result.stdout)

    def test_launcher_rejects_unknown_filters_and_prints_replayable_density_commands(self):
        unknown = subprocess.run(
            ["bash", "run_peak_sensitive_experiments.sh", "--dry-run", "--tasks", "unknown"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(unknown.returncode, 0)
        dry = subprocess.run(
            [
                "bash", "run_peak_sensitive_experiments.sh", "--dry-run", "--stage", "tune",
                "--tasks", "aluminium_frf", "--families", "Local nested CF control",
                "--budgets", "128", "--densities", "24,48,96,192",
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        commands = [line for line in dry.stdout.splitlines() if line.strip()]
        tune_commands = [line for line in commands if "--stage tune" in line]
        self.assertEqual(len(tune_commands), 4)
        self.assertTrue(all("--densities" in line for line in tune_commands))
        self.assertIn("Local\\ nested\\ CF\\ control", dry.stdout)
        for command in commands:
            arguments = shlex.split(command)
            self.assertIn("Local nested CF control", arguments)


class PreflightFreezeTests(unittest.TestCase):
    def _preflight_fixture(self, root: Path):
        from peak_sensitive_benchmark import _config_hash
        from peak_sensitive_preflight import (
            _expected_preflight_test_inventory,
            canonical_preflight_test_log_header,
        )

        raw_root = root / "raw"
        output_root = root / "results"
        raw_root.mkdir()
        output_root.mkdir()
        (raw_root / "source_manifest.json").write_text('{"files": []}\n')
        config_path = ROOT / "peak_sensitive_config.json"
        schema_path = ROOT / "peak_sensitive_results_schema.json"
        config = json.loads(config_path.read_text())
        scope = {
            "tasks": config["fixed_scientific_tasks"],
            "families": config["families"],
            "budgets": config["budgets"],
            "frf_densities": config["frf_densities"],
        }
        manifest = {
            "protocol_key": config["protocol_key"],
            "protocol_sha256": "synthetic-preflight-protocol",
            "config_sha256": _config_hash(config),
            "protocol_mode": "production",
            "smoke_root_marker": config.get("smoke_root_marker"),
            "code_files": [],
            "input_files": [],
            "raw_root": {"synthetic": True},
        }
        (output_root / "protocol_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        tasks = {
            task: {
                family: {
                    str(budget): [{"candidate_key": f"{task}:{family}:{budget}"}]
                    for budget in scope["budgets"]
                }
                for family in scope["families"]
            }
            for task in scope["tasks"]
        }
        inventory = {
            "protocol_key": config["protocol_key"],
            "protocol_mode": "production",
            "protocol_sha256": manifest["protocol_sha256"],
            "config_sha256": manifest["config_sha256"],
            "scope": scope,
            "task_shards": [
                {"task": task, "frf_density": density}
                for task in scope["tasks"]
                for density in (scope["frf_densities"] if task == "aluminium_frf" else [None])
            ],
            "tasks": tasks,
            "family_eligibility": {
                family: {"status": "eligible", "reason": None}
                for family in scope["families"]
            },
        }
        (output_root / "candidate_manifest.json").write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n"
        )
        provenance = output_root / "provenance"
        provenance.mkdir()
        tests_log = provenance / "preflight_tests.log"
        test_inventory = _expected_preflight_test_inventory()
        tests_log.write_text(
            f"{canonical_preflight_test_log_header()}\n\n"
            f"Ran {test_inventory['tests_run']} tests in 0.001s\n\nOK\n"
        )
        environment_log = provenance / "preflight_environment.txt"
        environment_log.write_text("fixture environment\n")
        return (
            config_path, schema_path, raw_root, output_root, scope, tests_log,
            environment_log, manifest,
        )

    def _expected_protocol_patch(self, config_path: Path, manifest: dict):
        config = json.loads(config_path.read_text())
        return patch(
            "peak_sensitive_preflight._load_expected_protocol",
            return_value=(config, manifest),
        )

    def test_preflight_freeze_rejects_changed_inventory_hash(self):
        from peak_sensitive_preflight import create_preflight_freeze, verify_preflight_freeze

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._preflight_fixture(Path(temporary_directory))
            config_path, schema_path, raw_root, output_root, scope, tests_log, environment_log, manifest = paths
            with self._expected_protocol_patch(config_path, manifest):
                freeze = create_preflight_freeze(
                    config_path=config_path, schema_path=schema_path, raw_root=raw_root,
                    output_root=output_root, scope=scope, tests_log=tests_log,
                    environment_log=environment_log,
                )
            self.assertNotIn(
                str((output_root / "preflight_freeze.json").resolve()),
                {record["path"] for record in freeze["files"]},
            )
            with self._expected_protocol_patch(config_path, manifest):
                verify_preflight_freeze(
                    config_path=config_path, schema_path=schema_path, raw_root=raw_root,
                    output_root=output_root, requested_scope={
                        "tasks": ["controlled_fano"], "families": ["CFNN"],
                        "budgets": [128], "frf_densities": [24],
                    },
                )
            inventory_path = output_root / "candidate_manifest.json"
            inventory = json.loads(inventory_path.read_text())
            inventory["tasks"]["controlled_fano"]["CFNN"]["128"].append({"candidate_key": "changed"})
            inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "candidate inventory SHA-256 mismatch"):
                with self._expected_protocol_patch(config_path, manifest):
                    verify_preflight_freeze(
                        config_path=config_path, schema_path=schema_path, raw_root=raw_root,
                        output_root=output_root, requested_scope=scope,
                    )

    def test_preflight_freeze_rejects_each_partial_config_axis(self):
        from peak_sensitive_preflight import create_preflight_freeze

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._preflight_fixture(Path(temporary_directory))
            config_path, schema_path, raw_root, output_root, scope, tests_log, environment_log, manifest = paths
            for axis in ("tasks", "families", "budgets", "frf_densities"):
                partial_scope = {key: list(values) for key, values in scope.items()}
                partial_scope[axis] = partial_scope[axis][:-1]
                with self.assertRaisesRegex(ValueError, "must exactly match frozen config axes"):
                    with self._expected_protocol_patch(config_path, manifest):
                        create_preflight_freeze(
                            config_path=config_path, schema_path=schema_path, raw_root=raw_root,
                            output_root=output_root, scope=partial_scope, tests_log=tests_log,
                            environment_log=environment_log,
                        )

    def test_preflight_freeze_rejects_targeted_passing_test_log(self):
        from peak_sensitive_preflight import (
            create_preflight_freeze,
            canonical_preflight_test_log_header,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._preflight_fixture(Path(temporary_directory))
            config_path, schema_path, raw_root, output_root, scope, tests_log, environment_log, manifest = paths
            header = canonical_preflight_test_log_header()
            tests_log.write_text(f"{header}\n\nRan 2 tests in 0.001s\n\nOK\n")
            with self.assertRaisesRegex(ValueError, "final zero-failure evidence"):
                with self._expected_protocol_patch(config_path, manifest):
                    create_preflight_freeze(
                        config_path=config_path, schema_path=schema_path, raw_root=raw_root,
                        output_root=output_root, scope=scope, tests_log=tests_log,
                        environment_log=environment_log,
                    )
            tests_log.write_text(
                header.replace("test_downstream_domain_baselines", "targeted_test")
                + "\n\nRan 2 tests in 0.001s\n\nOK\n"
            )
            with self.assertRaisesRegex(ValueError, "canonical command marker"):
                with self._expected_protocol_patch(config_path, manifest):
                    create_preflight_freeze(
                        config_path=config_path, schema_path=schema_path, raw_root=raw_root,
                        output_root=output_root, scope=scope, tests_log=tests_log,
                        environment_log=environment_log,
                    )

    def test_preflight_verifier_rejects_incomplete_frozen_scope(self):
        from peak_sensitive_preflight import create_preflight_freeze, verify_preflight_freeze

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._preflight_fixture(Path(temporary_directory))
            config_path, schema_path, raw_root, output_root, scope, tests_log, environment_log, manifest = paths
            with self._expected_protocol_patch(config_path, manifest):
                create_preflight_freeze(
                    config_path=config_path, schema_path=schema_path, raw_root=raw_root,
                    output_root=output_root, scope=scope, tests_log=tests_log,
                    environment_log=environment_log,
                )
            freeze_path = output_root / "preflight_freeze.json"
            freeze = json.loads(freeze_path.read_text())
            del freeze["candidate_inventory"]["scope"]["tasks"]
            freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "does not cover requested production scope"):
                with self._expected_protocol_patch(config_path, manifest):
                    verify_preflight_freeze(
                        config_path=config_path, schema_path=schema_path, raw_root=raw_root,
                        output_root=output_root, requested_scope=scope,
                    )

    def test_production_stage_refuses_before_creating_output_without_preflight_freeze(self):
        from peak_sensitive_benchmark import main

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / "raw"
            output_root = root / "results"
            raw_root.mkdir()
            (raw_root / "source_manifest.json").write_text('{"files": []}\n')
            with patch.object(sys, "argv", [
                "peak_sensitive_benchmark.py", "--stage", "tune",
                "--config", str(ROOT / "peak_sensitive_config.json"),
                "--schema", str(ROOT / "peak_sensitive_results_schema.json"),
                "--raw-root", str(raw_root), "--output-root", str(output_root),
                "--tasks", "controlled_fano", "--families", "CFNN", "--budgets", "128",
                "--densities", "24",
            ]):
                with self.assertRaisesRegex(ValueError, "missing preflight freeze"):
                    main()
            self.assertFalse(output_root.exists())
