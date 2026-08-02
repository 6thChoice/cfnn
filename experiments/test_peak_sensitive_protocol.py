import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from jsonschema import Draft202012Validator, ValidationError


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "downstream_data" / "raw"


def _schema():
    return json.loads((ROOT / "peak_sensitive_results_schema.json").read_text())


def _base_record(stage):
    config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    protocol_digest = hashlib.sha256()
    for path in (ROOT / "peak_sensitive_config.json", ROOT / "peak_sensitive_results_schema.json"):
        protocol_digest.update(path.name.encode())
        protocol_digest.update(path.read_bytes())
    scientific_role = "tuning" if stage == "peak_sensitive_tuning" else "evaluation"
    visibility = "sealed" if stage == "peak_sensitive_tuning" else "opened"
    data_seed = 547 if scientific_role == "tuning" else 719
    try:
        from .downstream_benchmark import _compact_metadata
        from .peak_sensitive_benchmark import make_peak_sensitive_bundle
    except ImportError:
        from downstream_benchmark import _compact_metadata
        from peak_sensitive_benchmark import make_peak_sensitive_bundle
    dataset = _compact_metadata(make_peak_sensitive_bundle(
        "controlled_fano", data_seed, config, ROOT / "downstream_data" / "raw",
        scientific_role=scientific_role, visibility=visibility,
    ).metadata)
    record = {
        "protocol_key": "cfnn-peak-sensitive-extension-2.0.0",
        "protocol_mode": "production",
        "smoke_root_marker": None,
        "config_sha256": hashlib.sha256(canonical_config).hexdigest(),
        "stage": stage,
        "stage_role": "validation_only" if stage == "peak_sensitive_tuning" else "confirmatory_test",
        "phase": "convergence" if stage == "peak_sensitive_tuning" else "evaluation",
        "scientific_role": scientific_role,
        "test_visibility": visibility,
        "task": "controlled_fano",
        "task_role": "controlled_mechanism",
        "problem_type": "complex_regression",
        "family": "CFNN",
        "target_budget": 128,
        "data_seed": data_seed,
        "init_seed": 11003,
        "candidate_key": "{}",
        "config": {},
        "actual_parameters": 128,
        "optimizer": {"key": "lr1e-3_wd0", "lr": 0.001, "weight_decay": 0.0},
        "best_step": 500,
        "optimizer_steps": 500,
        "validation_score": 0.2,
        "status": "ok",
        "curve": [],
        "training_wall_seconds": 1.0,
        "total_wall_seconds": 1.1,
        "device": "cpu",
        "dataset": dataset,
        "source_manifest_sha256": None,
        "protocol_sha256": protocol_digest.hexdigest(),
        "window_spec": {
            "task": "controlled_fano",
            "axis_space": "linear",
            "intervals": [[0.4, 0.6]],
            "source_role": "generator_metadata",
            "status": "ok",
            "detector": {"test_response_used": False},
        },
        "validation_resonance_nrmse": 0.1,
        "validation_global_nrmse": 0.2,
        "initialization_mode": "fixed",
        "gradient_clip_events": 0,
        "gradient_clip_rate": 0.0,
        "gradient_clip_threshold": 5.0,
        "training_min_steps": 500,
        "training_max_steps": 4000,
        "training_patience": 250,
        "training_checkpoints": [20, 80, 200, 500, 1000, 2000, 4000],
        "training_batch_size": 64,
        "batch_seed": data_seed + 49979687,
        "validation_interval": 20,
        "validation_evaluations": 25,
        "tuning_seed": 547 if scientific_role == "tuning" else None,
        "frf_density": None,
        "candidate_manifest_sha256": "a" * 64,
        "selected_common_budget": 128,
    }
    if stage == "peak_sensitive_evaluation":
        record.update(
            {
                "test_metrics": {
                    "status": "ok",
                    "complex_nrmse": 0.2,
                    "global_complex_nrmse": 0.2,
                    "resonance_window_complex_nrmse": 0.1,
                    "phase_mae_rad": 0.01,
                    "strongest_feature_frequency_error": 0.0,
                    "missed_feature_rate": 0.0,
                    "false_feature_rate": 0.0,
                    "quality_factor_or_bandwidth_error": 0.0,
                    "numerical_failure_rate": 0.0,
                    "underfit_rate": 0.0,
                    "dominant_relaxation_frequency_error": None,
                    "kramers_kronig_consistency_proxy": None,
                    "underfit_criterion": "validation_global_nrmse_ge_1.0",
                    "metric_applicability": {
                        "quality_factor_or_bandwidth_error": {
                            "status": "applicable", "reason": "matched_identifiable_feature"
                        },
                        "dominant_relaxation_frequency_error": {
                            "status": "not_applicable", "reason": "task_is_not_eis"
                        },
                        "kramers_kronig_consistency_proxy": {
                            "status": "not_applicable", "reason": "task_is_not_eis"
                        },
                        "underfit_rate": {
                            "status": "applicable", "reason": "frozen_validation_criterion"
                        },
                    },
                },
                "primary_metric_name": "resonance_window_complex_nrmse",
                "primary_metric_value": 0.1,
                "selection_path": "selections/controlled_fano/cfnn__128.json",
                "selection_sha256": "c" * 64,
            }
        )
    return record


def _schema_errors(record):
    return list(Draft202012Validator(_schema()).iter_errors(record))


def _production_validate(record):
    try:
        from .peak_sensitive_result_validation import validate_peak_sensitive_record
    except ImportError:
        from peak_sensitive_result_validation import validate_peak_sensitive_record

    return validate_peak_sensitive_record(record)


def validate_semantic_record(record):
    """Exercise the production validator from the protocol tests."""
    return _production_validate(record)


class SchemaRecordTests(unittest.TestCase):
    def test_draft202012_accepts_validation_only_tuning_without_test_metrics(self):
        record = _base_record("peak_sensitive_tuning")
        record.pop("selected_common_budget")
        validate_semantic_record(record)
        self.assertNotIn("test_metrics", record)

    def test_tuning_selected_common_budget_may_be_null(self):
        record = _base_record("peak_sensitive_tuning")
        record["selected_common_budget"] = None
        validate_semantic_record(record)

    def test_draft202012_accepts_confirmatory_evaluation_with_window_primary_metric(self):
        validate_semantic_record(_base_record("peak_sensitive_evaluation"))

    def test_evaluation_requires_a_frozen_common_budget(self):
        record = _base_record("peak_sensitive_evaluation")
        record.pop("selected_common_budget")
        with self.assertRaises(ValidationError):
            validate_semantic_record(record)

        record = _base_record("peak_sensitive_evaluation")
        record["selected_common_budget"] = None
        with self.assertRaises(ValidationError):
            validate_semantic_record(record)

    def test_tuning_record_cannot_claim_test_derived_window(self):
        record = _base_record("peak_sensitive_tuning")
        record["window_spec"]["source_role"] = "test_response"
        with self.assertRaises((ValidationError, ValueError)):
            validate_semantic_record(record)

    def test_evaluation_record_rejects_invalid_metric_name(self):
        record = _base_record("peak_sensitive_evaluation")
        record["primary_metric_name"] = "complex_nrmse"
        with self.assertRaises((ValidationError, ValueError)):
            validate_semantic_record(record)

    def test_schema_rejects_invalid_protocol_identifiers(self):
        cases = {
            "task": "unapproved_task",
            "family": "unapproved_family",
            "target_budget": 999999,
            "selected_common_budget": 999999,
            "protocol_sha256": "not-a-sha256",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                record = _base_record("peak_sensitive_evaluation")
                record[field] = value
                self.assertTrue(_schema_errors(record))

    def test_semantic_contract_rejects_window_task_mismatch(self):
        record = _base_record("peak_sensitive_evaluation")
        record["window_spec"]["task"] = "microwave_fano"
        with self.assertRaises(ValueError):
            validate_semantic_record(record)

    def test_semantic_contract_rejects_task_axis_or_source_mismatch(self):
        cases = {
            "axis": ("figshare_battery_eis", "linear", "train_validation"),
            "source": ("controlled_fano", "linear", "train_validation"),
        }
        for label, (task, axis_space, source_role) in cases.items():
            with self.subTest(label=label):
                record = _base_record("peak_sensitive_evaluation")
                record["task"] = task
                record["window_spec"].update(
                    task=task, axis_space=axis_space, source_role=source_role
                )
                with self.assertRaises((ValidationError, ValueError)):
                    validate_semantic_record(record)

    def test_semantic_contract_rejects_problem_type_or_init_seed_mismatch(self):
        for field, value in (("problem_type", "regression"), ("init_seed", 12345)):
            with self.subTest(field=field):
                record = _base_record("peak_sensitive_evaluation")
                record[field] = value
                with self.assertRaises((ValidationError, ValueError)):
                    validate_semantic_record(record)

    def test_production_validator_rejects_wrong_role_seed_budget_optimizer_and_controls(self):
        cases = {
            "task_role": "wrong-role",
            "data_seed": 999999,
            "actual_parameters": 1,
            "optimizer": {"key": "invented", "lr": 0.2, "weight_decay": 0.0},
            "optimizer_steps": 1,
            "phase": "invented",
            "config_sha256": "0" * 64,
            "protocol_sha256": "1" * 64,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                record = _base_record("peak_sensitive_tuning")
                record.pop("selected_common_budget")
                record[field] = value
                with self.assertRaises((ValidationError, ValueError)):
                    validate_semantic_record(record)

    def test_production_validator_rejects_stage_role_visibility_and_density_mismatch(self):
        cases = [
            ("scientific_role", "evaluation"),
            ("test_visibility", "opened"),
            ("frf_density", 24),
            ("tuning_seed", 601),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                record = _base_record("peak_sensitive_tuning")
                record.pop("selected_common_budget")
                record[field] = value
                with self.assertRaises((ValidationError, ValueError)):
                    validate_semantic_record(record)

    def test_smoke_controls_require_explicit_protocol_and_root_marker(self):
        record = _base_record("peak_sensitive_tuning")
        record.pop("selected_common_budget")
        record.update(
            protocol_mode="smoke",
            smoke_root_marker="task7-explicit-smoke-root-v1",
            phase="screening",
            optimizer_steps=5,
            best_step=5,
            training_min_steps=2,
            training_max_steps=5,
            training_patience=5,
            training_checkpoints=[2, 5],
            validation_interval=1,
            validation_evaluations=5,
        )
        validate_semantic_record(record)
        record["smoke_root_marker"] = None
        with self.assertRaises((ValidationError, ValueError)):
            validate_semantic_record(record)

    def test_semantic_contract_matches_no_feature_status_to_intervals(self):
        record = _base_record("peak_sensitive_evaluation")
        record["window_spec"].update(
            status="no_identifiable_feature",
            source_role="no_identifiable_feature",
        )
        with self.assertRaises(ValueError):
            validate_semantic_record(record)

        record = _base_record("peak_sensitive_evaluation")
        record["window_spec"].update(
            status="ok", source_role="generator_metadata", intervals=[]
        )
        with self.assertRaises(ValueError):
            validate_semantic_record(record)

    def test_semantic_contract_rejects_unordered_window_intervals(self):
        record = _base_record("peak_sensitive_evaluation")
        record["window_spec"]["intervals"] = [[0.6, 0.8], [0.4, 0.5]]
        with self.assertRaises(ValueError):
            validate_semantic_record(record)


class FrozenConfigTests(unittest.TestCase):
    def test_v2_protocol_is_complete_and_disjoint(self):
        cfg = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        old = json.loads((ROOT / "downstream_config.json").read_text())

        self.assertEqual(cfg["protocol_key"], "cfnn-peak-sensitive-extension-2.0.0")
        self.assertEqual(cfg["protocol_version"], "2.0.0")
        self.assertEqual(cfg["budgets"], [128, 256, 512, 1024])
        self.assertEqual(cfg["budget_tolerance"], 0.05)
        self.assertEqual(cfg["training"]["screening_steps"], 500)
        self.assertEqual(cfg["training"]["max_steps"], 4000)
        self.assertEqual(cfg["training"]["min_steps"], 500)
        self.assertEqual(cfg["training"]["patience"], 250)
        self.assertEqual(cfg["training"]["validation_interval"], 20)
        self.assertEqual(
            cfg["training"]["checkpoints"],
            [20, 80, 200, 500, 1000, 2000, 4000],
        )
        self.assertEqual(cfg["training"]["gradient_clip"], 5.0)
        self.assertEqual(cfg["cfnn_epsilon_grid"], [0.05, 0.1, 0.2, 0.3, 0.5])
        self.assertEqual(
            cfg["cfnn_initialization_modes"], ["fixed", "numerator_scaled"]
        )
        self.assertEqual(cfg["tuning_data_seeds"], [547, 601, 659])
        self.assertEqual(cfg["evaluation_data_seeds"], [719, 773, 829, 887, 947])
        self.assertEqual(
            cfg["evaluation_init_seeds"],
            [11003, 12007, 13001, 14009, 15013, 16001, 17011, 18013, 19001, 20011],
        )
        self.assertTrue(
            set(cfg["evaluation_init_seeds"]).isdisjoint(old["evaluation_init_seeds"])
        )
        self.assertEqual(
            cfg["fixed_scientific_tasks"],
            [
                "controlled_fano",
                "microwave_fano",
                "microstrip_resonator",
                "aluminium_frf",
                "figshare_battery_eis",
                "hybrid_supercapacitor_eis",
            ],
        )
        self.assertEqual(cfg["families"], old["families"])
        self.assertEqual(len(cfg["tasks"]), 6)
        self.assertTrue(set(cfg["fixed_scientific_tasks"]).issubset(cfg["tasks"]))

    def test_screening_and_common_budget_rules_are_frozen(self):
        cfg = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        choices = cfg["screening"]["optimizer_choice_by_family"]
        self.assertEqual(set(choices), set(cfg["families"]))
        self.assertEqual(cfg["screening"]["steps"], 500)
        for family, choice in choices.items():
            with self.subTest(family=family):
                self.assertIn(choice, cfg["optimizer_grids"][family])
        self.assertEqual(cfg["common_budget_selection"]["relative_loss_threshold"], 0.05)
        self.assertEqual(cfg["common_budget_selection"]["minimum_family_eligibility"], 6)

    def test_v2_metric_names_are_frozen(self):
        cfg = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        self.assertEqual(
            cfg["metric_settings"]["primary_validation_metric"],
            "resonance_window_complex_nrmse",
        )
        self.assertIn("global_complex_nrmse", cfg["metric_settings"]["primary_metric_names"])

    def test_v2_inherits_non_cfnn_optimizer_grids_and_freezes_cfnn_grid(self):
        cfg = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        old = json.loads((ROOT / "downstream_config.json").read_text())

        for family in cfg["families"]:
            with self.subTest(family=family):
                if family != "CFNN":
                    self.assertEqual(
                        cfg["optimizer_grids"][family], old["optimizer_grids"][family]
                    )

        self.assertEqual(
            {(entry["lr"], entry["weight_decay"])
             for entry in cfg["optimizer_grids"]["CFNN"]},
            {
                (0.0001, 0.0),
                (0.0001, 0.000001),
                (0.0003, 0.0),
                (0.0003, 0.000001),
                (0.001, 0.0),
                (0.001, 0.000001),
                (0.003, 0.0),
                (0.003, 0.000001),
            },
        )

    def test_v2_task_axes_and_output_roots_are_frozen(self):
        cfg = json.loads((ROOT / "peak_sensitive_config.json").read_text())

        self.assertEqual(cfg["output_root"], "peak_sensitive_results")
        self.assertNotIn("downstream", cfg["output_root"])
        self.assertEqual(cfg["frf_densities"], [24, 48, 96, 192])
        self.assertEqual(
            cfg["frequency_transforms"],
            {
                "controlled_fano": "linear",
                "microwave_fano": "linear",
                "microstrip_resonator": "linear",
                "aluminium_frf": "linear",
                "figshare_battery_eis": "log10_frequency",
                "hybrid_supercapacitor_eis": "log10_frequency",
            },
        )
        self.assertEqual(cfg["peak_detector"]["prominence_fraction"], 0.05)
        self.assertEqual(cfg["peak_detector"]["max_features"], 3)
        self.assertEqual(cfg["peak_detector"]["minimum_half_width_observed_spacings"], 2)
        self.assertEqual(
            cfg["tasks"]["aluminium_frf"]["validation_by_density"],
            {"24": 12, "48": 24, "96": 48, "192": 96},
        )
        self.assertNotIn("tuning_curve_count", cfg["tasks"]["aluminium_frf"])
        self.assertNotIn("evaluation_curve_count", cfg["tasks"]["aluminium_frf"])
        self.assertNotIn("curve_selection_method", cfg["tasks"]["aluminium_frf"])
        self.assertNotIn("replacement_rule", cfg["tasks"]["aluminium_frf"])
        for task in cfg["fixed_scientific_tasks"]:
            with self.subTest(task=task):
                self.assertIn("problem_type", cfg["tasks"][task])
                self.assertIn("input_dim", cfg["tasks"][task])
                self.assertIn("output_dim", cfg["tasks"][task])
                self.assertIn("frequency_transform", cfg["tasks"][task])

    def test_v2_schema_preserves_v1_requirements_and_adds_guardrails(self):
        schema = json.loads((ROOT / "peak_sensitive_results_schema.json").read_text())
        old = json.loads((ROOT / "downstream_results_schema.json").read_text())

        stage_specific_v1_fields = {
            "test_metrics", "primary_metric_name", "primary_metric_value"
        }
        self.assertTrue(
            set(old["required"]) - stage_specific_v1_fields <= set(schema["required"])
        )
        for name, definition in old["properties"].items():
            if name in {
                "stage",
                "stage_role",
                "task",
                "task_role",
                "family",
                "target_budget",
                "problem_type",
                "test_metrics",
                "primary_metric_name",
            }:
                continue
            self.assertIn(name, schema["properties"])
            for key, value in definition.items():
                self.assertEqual(schema["properties"][name].get(key), value)
        self.assertEqual(
            set(schema["properties"]["stage"]["enum"]),
            {"peak_sensitive_evaluation", "peak_sensitive_tuning"},
        )
        self.assertEqual(
            set(schema["properties"]["task_role"]["enum"]),
            {"controlled_mechanism", "fixed_measured", "independent_eis_extension"},
        )
        self.assertTrue(
            {
                "protocol_key",
                "protocol_sha256",
                "window_spec",
                "validation_resonance_nrmse",
                "validation_global_nrmse",
                "initialization_mode",
                "gradient_clip_events",
                "gradient_clip_rate",
            } <= set(schema["required"])
        )
        self.assertIn("test_metrics", schema["allOf"][1]["then"]["required"])
        self.assertIn("selected_common_budget", schema["allOf"][1]["then"]["required"])
        self.assertNotIn("selected_common_budget", schema["required"])
        self.assertNotIn("test_metrics", schema["allOf"][0].get("required", []))
        self.assertEqual(
            schema["properties"]["protocol_sha256"]["pattern"],
            "^[0-9a-f]{64}$",
        )


class SourceAndEISTests(unittest.TestCase):
    def test_hsc_source_is_version_pinned(self):
        sources = json.loads((ROOT / "downstream_sources.json").read_text())
        spec = sources["hybrid_supercapacitor_eis"]
        self.assertEqual(spec["provider"], "figshare")
        self.assertEqual(spec["article_id"], 24321496)
        self.assertEqual(spec["version"], 4)
        self.assertEqual(
            spec["api_url"],
            "https://api.figshare.com/v2/articles/24321496/versions/4",
        )
        self.assertEqual(spec["license"], "CC BY 4.0")
        self.assertEqual(spec["role"], "independent_eis_extension")

    def test_hsc_source_exposes_legacy_url_without_changing_pinned_api_url(self):
        sources = json.loads((ROOT / "downstream_sources.json").read_text())
        spec = sources["hybrid_supercapacitor_eis"]
        self.assertEqual(spec["url"], spec["api_url"])
        self.assertEqual(spec["version"], 4)
        self.assertEqual(
            spec["api_url"],
            "https://api.figshare.com/v2/articles/24321496/versions/4",
        )

    def test_figshare_downloader_refuses_metadata_version_mismatch(self):
        try:
            from .download_downstream_datasets import download_sources
        except ImportError:
            from download_downstream_datasets import download_sources

        specification = {
            "hybrid_supercapacitor_eis": {
                "provider": "figshare",
                "article_id": 24321496,
                "version": 4,
                "api_url": "https://api.figshare.com/v2/articles/24321496/versions/4",
                "license": "CC BY 4.0",
                "role": "independent_eis_extension",
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            config_path = temporary_path / "sources.json"
            config_path.write_text(json.dumps(specification))
            with patch(
                "download_downstream_datasets._read_json_url",
                return_value={"version": 3},
            ):
                with self.assertRaisesRegex(ValueError, "version mismatch"):
                    download_sources(
                        config_path, temporary_path / "raw", metadata_only=True
                    )

    def test_downloader_rejects_provider_checksum_mismatch(self):
        try:
            from .download_downstream_datasets import _verify_provider_checksum
        except ImportError:
            from download_downstream_datasets import _verify_provider_checksum

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.bin"
            path.write_bytes(b"released bytes")
            with self.assertRaisesRegex(ValueError, "provider checksum mismatch"):
                _verify_provider_checksum(
                    path, "00000000000000000000000000000000", "md5"
                )

    def test_hsc_parser_returns_frequency_resolved_complex_curves(self):
        try:
            from .peak_sensitive_protocol import load_hybrid_supercapacitor_eis
        except ImportError:
            from peak_sensitive_protocol import load_hybrid_supercapacitor_eis

        curves = load_hybrid_supercapacitor_eis(RAW_ROOT)
        self.assertEqual(len(curves), 48)
        self.assertGreaterEqual(len({curve.conditions["temperature_c"] for curve in curves}), 8)
        self.assertGreaterEqual(len({curve.conditions["soc_percent"] for curve in curves}), 6)
        self.assertEqual(len({curve.curve_id for curve in curves}), len(curves))
        self.assertTrue(all(np.all(np.diff(curve.frequency) > 0) for curve in curves))
        self.assertTrue(all(
            curve.response.shape == (len(curve.frequency), 2) for curve in curves
        ))
        self.assertTrue(all(np.all(np.isfinite(curve.response)) for curve in curves))
        self.assertTrue(all(
            curve.metadata["imaginary_sign_normalization"]
            == "identity_as_released_negative_imaginary_impedance"
            for curve in curves
        ))

    def test_hsc_parser_rejects_nonmonotonic_frequency_without_sorting(self):
        try:
            from .peak_sensitive_protocol import _strict_hsc_arrays
        except ImportError:
            from peak_sensitive_protocol import _strict_hsc_arrays

        malformed = SimpleNamespace(
            Freq=np.array([1.0, 3.0, 2.0]),
            Real=np.array([0.1, 0.2, 0.3]),
            Imm=np.array([-0.1, -0.2, -0.3]),
        )
        with self.assertRaisesRegex(ValueError, "nonmonotonic or duplicate"):
            _strict_hsc_arrays(malformed, "malformed-fixture")

    def test_hsc_audit_freezes_disjoint_stratified_curve_roles(self):
        try:
            from .peak_sensitive_protocol import audit_peak_sensitive_sources
        except ImportError:
            from peak_sensitive_protocol import audit_peak_sensitive_sources

        config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
        audit = audit_peak_sensitive_sources(RAW_ROOT, config)
        repeated_audit = audit_peak_sensitive_sources(RAW_ROOT, config)
        manifest = audit["curve_manifest"]
        self.assertEqual(manifest, repeated_audit["curve_manifest"])
        self.assertEqual(audit["source"]["article_id"], 24321496)
        self.assertEqual(audit["source"]["version"], 4)
        self.assertEqual(audit["source"]["license"], "CC BY 4.0")
        self.assertEqual(
            audit["source"]["figshare_md5"], "08f50c641f6c649dc08632e56074dfca"
        )
        self.assertEqual(audit["condition_counts"], {
            "temperature_c": 8,
            "soc_percent": 6,
            "temperature_soc_pairs": 48,
        })
        self.assertEqual(audit["point_count_distribution"], {"81": 48})
        tuning = manifest["tuning_curve_ids"]
        evaluation = manifest["evaluation_curve_ids"]
        self.assertEqual(len(tuning), 3)
        self.assertEqual(len(evaluation), 5)
        self.assertFalse(set(tuning) & set(evaluation))
        self.assertEqual(
            manifest["hierarchy_availability"],
            {"curve_id": True, "device_id": False, "repetition_id": False},
        )
        self.assertEqual(
            len({record["temperature_c"] for record in manifest["tuning"] + manifest["evaluation"]}),
            8,
        )


class FRFDensityTests(unittest.TestCase):
    raw_root = RAW_ROOT

    @staticmethod
    def _synthetic_curve(curve_id="point:01", point_count=300):
        from downstream_protocol import MeasuredCurve

        frequency = np.arange(point_count, dtype=np.float64) + 1.0
        response = np.column_stack((
            np.sin(frequency / 17.0),
            np.cos(frequency / 23.0),
        )).astype(np.float32)
        return MeasuredCurve(
            curve_id=curve_id,
            frequency=frequency,
            response=response,
            conditions={"point": 1},
            source_files=("Waveforms Point01.tdms",),
            metadata={"source_record": 7758683, "frf_estimator": "H1"},
        )

    @classmethod
    def _synthetic_curves(cls, point_count=300):
        return [
            cls._synthetic_curve(f"point:{point:02d}", point_count)
            for point in range(1, 10)
        ]

    def _synthetic_bundle(self, curve_id, data_seed, n_train, point_count=300):
        from peak_sensitive_protocol import make_frf_density_bundle

        with patch(
            "peak_sensitive_protocol.load_dense_aluminium_frf",
            return_value=self._synthetic_curves(point_count),
        ):
            return make_frf_density_bundle(curve_id, data_seed, n_train, self.raw_root)

    def test_frf_density_masks_are_nested_and_disjoint(self):
        curve_id = "point:01"
        bundles = {
            n: self._synthetic_bundle(curve_id, 719, n)
            for n in (24, 48, 96, 192)
        }
        train = {
            n: set(bundle.metadata["train_point_ids"])
            for n, bundle in bundles.items()
        }
        self.assertTrue(train[24] < train[48] < train[96] < train[192])
        for bundle in bundles.values():
            roles = [
                set(bundle.metadata[role])
                for role in ("train_point_ids", "validation_point_ids", "test_point_ids")
            ]
            self.assertFalse(roles[0] & roles[1])
            self.assertFalse(roles[0] & roles[2])
            self.assertFalse(roles[1] & roles[2])
            self.assertEqual(sum(map(len, roles)), 300)
            self.assertEqual(
                len(bundle.metadata["validation_point_ids"]),
                {24: 12, 48: 24, 96: 48, 192: 96}[bundle.metadata["density"]],
            )

    def test_frf_density_parent_mask_is_shared_and_cross_call_deterministic(self):
        curve_id = "point:01"
        first = self._synthetic_bundle(curve_id, 719, 48)
        second = self._synthetic_bundle(curve_id, 719, 48)
        wider = self._synthetic_bundle(curve_id, 719, 192)
        np.testing.assert_array_equal(first.x_train, second.x_train)
        np.testing.assert_array_equal(first.y_train, second.y_train)
        self.assertEqual(first.metadata, second.metadata)
        self.assertEqual(first.metadata["parent_mask_hash"], wider.metadata["parent_mask_hash"])
        self.assertEqual(first.metadata["density"], 48)
        self.assertEqual(len(first.metadata["parent_train_point_ids"]), 192)
        self.assertEqual(len(first.metadata["parent_validation_point_ids"]), 96)
        self.assertEqual(len(first.metadata["validation_point_ids"]), 24)
        self.assertEqual(len(first.metadata["parent_train_order_sha256"]), 64)
        self.assertEqual(len(first.metadata["parent_validation_order_sha256"]), 64)
        self.assertTrue(
            set(first.metadata["parent_train_point_ids"])
            .isdisjoint(first.metadata["parent_validation_point_ids"])
        )

    def test_frf_density_seed_changes_parent_order_and_hash(self):
        curve_id = "point:01"
        first = self._synthetic_bundle(curve_id, 719, 24)
        second = self._synthetic_bundle(curve_id, 773, 24)
        self.assertNotEqual(
            first.metadata["parent_mask_hash"], second.metadata["parent_mask_hash"]
        )
        self.assertNotEqual(
            first.metadata["train_point_ids"], second.metadata["train_point_ids"]
        )

    def test_frf_density_frequency_remains_float64_and_standardizer_is_train_only(self):
        from confirmatory_protocol import DatasetBundle
        from downstream_training import prepare_bundle

        bundle = self._synthetic_bundle("point:01", 719, 24)
        self.assertEqual(bundle.x_train.dtype, np.float64)
        self.assertIsInstance(bundle, DatasetBundle)
        prepared = prepare_bundle(bundle, "complex_regression", __import__("torch").device("cpu"))
        expected_mean = bundle.x_train.mean(axis=0, keepdims=True)
        expected_std = np.maximum(bundle.x_train.std(axis=0, keepdims=True), 1e-6)
        np.testing.assert_allclose(prepared.x_mean, expected_mean)
        np.testing.assert_allclose(prepared.x_std, expected_std)
        np.testing.assert_allclose(
            prepared.x_validation.numpy(),
            ((bundle.x_validation - expected_mean) / expected_std).astype(np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_frf_density_rejects_insufficient_frozen_curve(self):
        from peak_sensitive_protocol import make_frf_density_bundle

        curves = self._synthetic_curves()
        curves[0] = self._synthetic_curve("point:01", 288)
        with self.assertRaisesRegex(ValueError, "requires at least 289"):
            with patch(
                "peak_sensitive_protocol.load_dense_aluminium_frf",
                return_value=curves,
            ):
                make_frf_density_bundle("point:01", 719, 24, self.raw_root)

    def test_frf_density_source_only_selection_replaces_insufficient_candidate(self):
        from peak_sensitive_protocol import select_frf_curve_roles

        inventory = [
            {
                "curve_id": f"point:{point:02d}",
                "source_sha256": f"{point:064x}",
                "eligible": True,
                "exclusion_reason": None,
            }
            for point in range(1, 10)
        ]
        initial = select_frf_curve_roles(inventory)
        insufficient_id = initial["tuning_curve_ids"][0]
        for row in inventory:
            if row["curve_id"] == insufficient_id:
                row["eligible"] = False
                row["exclusion_reason"] = "insufficient_points"
        selected = select_frf_curve_roles(inventory)
        self.assertEqual(len(selected["tuning_curve_ids"]), 3)
        self.assertEqual(len(selected["evaluation_curve_ids"]), 5)
        self.assertFalse(
            set(selected["tuning_curve_ids"]) & set(selected["evaluation_curve_ids"])
        )
        self.assertNotIn(insufficient_id, selected["tuning_curve_ids"])
        self.assertTrue(selected["selected_replacements"])
        self.assertTrue(
            all(row["reason"] == "insufficient_points"
                for row in selected["selected_replacements"])
        )

    def test_frf_density_audit_excludes_and_replaces_insufficient_synthetic_curve(self):
        from peak_sensitive_protocol import audit_frf_density_sources

        curves = self._synthetic_curves()
        with patch(
            "peak_sensitive_protocol.load_dense_aluminium_frf", return_value=curves
        ):
            initially_selected = audit_frf_density_sources(self.raw_root)["tuning_curve_ids"][0]
        replacement_curves = [
            self._synthetic_curve(
                curve.curve_id,
                288 if curve.curve_id == initially_selected else 300,
            )
            for curve in curves
        ]
        with patch(
            "peak_sensitive_protocol.load_dense_aluminium_frf",
            return_value=replacement_curves,
        ):
            audit = audit_frf_density_sources(self.raw_root)
        self.assertEqual(
            {row["curve_id"] for row in audit["excluded_curves"]}, {initially_selected}
        )
        self.assertNotIn(initially_selected, audit["tuning_curve_ids"])
        self.assertTrue(audit["replacement_policy"]["selected_replacements"])

    @unittest.skipUnless(
        (RAW_ROOT / "aluminium_frf" / "Waveforms Point01.tdms").exists(),
        "verified Aluminium TDMS raw sources are absent",
    )
    def test_dense_raw_frf_inventory_is_verified_eligible_and_role_frozen(self):
        from peak_sensitive_protocol import audit_frf_density_sources, load_dense_aluminium_frf

        curves = load_dense_aluminium_frf(self.raw_root)
        self.assertEqual(len(curves), 25)
        self.assertTrue(all(len(curve.frequency) == 2048 for curve in curves))
        self.assertTrue(all(np.all(np.isfinite(curve.frequency)) for curve in curves))
        self.assertTrue(all(np.all(curve.frequency > 0.0) for curve in curves))
        self.assertTrue(all(np.all(np.isfinite(curve.response)) for curve in curves))
        self.assertTrue(all(curve.metadata["frf_estimator"] == "H1" for curve in curves))
        self.assertTrue(all(curve.metadata["nperseg"] == 4096 for curve in curves))
        self.assertTrue(all(len(curve.metadata["source_sha256"]) == 64 for curve in curves))

        audit = audit_frf_density_sources(self.raw_root)
        self.assertEqual(audit["minimum_points"], 289)
        self.assertEqual(len(audit["eligible_curve_ids"]), 25)
        self.assertEqual(audit["excluded_curves"], [])
        self.assertEqual(len(audit["tuning_curve_ids"]), 3)
        self.assertEqual(len(audit["evaluation_curve_ids"]), 5)
        self.assertFalse(set(audit["tuning_curve_ids"]) & set(audit["evaluation_curve_ids"]))
        self.assertTrue(
            all(row["source_status"] == "verified" for row in audit["inventory"])
        )
        self.assertTrue(
            all(len(row["source_sha256"]) == 64 for row in audit["inventory"])
        )
        self.assertFalse(audit["replacement_policy"]["performance_replacement_allowed"])

    def test_dense_frf_domain_depends_only_on_frequency_bins(self):
        import inspect

        from peak_sensitive_protocol import dense_frf_domain_indices

        frequency = np.array([0.0, 1.0, np.nan, 2.0, np.inf, 3.0])
        response_a = np.array([0.0, 1.0, np.nan, -4.0, 9.0, 12.0])
        response_b = -1000.0 * response_a
        self.assertEqual(list(inspect.signature(dense_frf_domain_indices).parameters), ["frequency"])
        np.testing.assert_array_equal(
            dense_frf_domain_indices(frequency), np.array([1, 3, 5])
        )
        np.testing.assert_array_equal(
            dense_frf_domain_indices(frequency), np.array([1, 3, 5])
        )
        self.assertFalse(np.array_equal(response_a, response_b))


if __name__ == "__main__":
    unittest.main()
