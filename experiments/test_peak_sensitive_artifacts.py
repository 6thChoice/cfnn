import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class PairedInferenceTests(unittest.TestCase):
    def test_exact_sign_flip_enumerates_instance_summaries(self):
        from build_peak_sensitive_artifacts import exact_sign_flip_p_value

        self.assertEqual(exact_sign_flip_p_value([-1.0] * 5, seed_parts=("exact",)), 0.0625)
        self.assertEqual(exact_sign_flip_p_value([0.0, 0.0], seed_parts=("ties",)), 1.0)

    def test_hierarchical_bootstrap_weights_instances_equally_and_reports_iqr(self):
        from build_peak_sensitive_artifacts import hierarchical_paired_bootstrap

        values = {
            ("curve-a", 719): {11003: -10.0},
            ("curve-b", 773): {seed: 1.0 for seed in range(10)},
        }
        result = hierarchical_paired_bootstrap(
            values, seed_parts=("equal-instance-weight",), replicates=200,
        )
        self.assertEqual(result["state"], "estimable")
        self.assertAlmostEqual(result["estimate"], -4.5)
        self.assertAlmostEqual(result["median"], -4.5)
        self.assertEqual(result["q1"], -7.25)
        self.assertEqual(result["q3"], -1.75)
        self.assertEqual(result["instance_count"], 2)
        self.assertEqual(result["init_count"], 11)

    def test_empty_available_case_is_typed_not_estimable(self):
        from build_peak_sensitive_artifacts import hierarchical_paired_bootstrap

        result = hierarchical_paired_bootstrap({}, seed_parts=("empty",), replicates=20)
        self.assertEqual(result, {
            "state": "not_estimable",
            "reason": "no_paired_available_cases",
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "median": None,
            "q1": None,
            "q3": None,
            "instance_count": 0,
            "init_count": 0,
        })

    def test_frf_bootstrap_preserves_complete_init_trajectories(self):
        from build_peak_sensitive_artifacts import frf_density_interaction

        densities = [24, 48, 96, 192]
        values = {}
        for density_index, density in enumerate(densities):
            values[density] = {
                ("curve-a", 719): {
                    11003: -0.1 * density_index,
                    12007: 100.0 - 0.1 * density_index,
                },
                ("curve-b", 773): {
                    11003: -50.0 - 0.1 * density_index,
                    12007: 50.0 - 0.1 * density_index,
                },
            }
        result = frf_density_interaction(values, seed_parts=("paired-trajectories",), replicates=200)
        self.assertEqual(result["state"], "estimable")
        self.assertAlmostEqual(result["effect_per_log2_train_points"], -0.1, places=12)
        self.assertAlmostEqual(result["ci_low"], -0.1, places=12)
        self.assertAlmostEqual(result["ci_high"], -0.1, places=12)
        self.assertEqual(result["trajectory_count"], 4)

    def test_failures_use_all_pairs_but_peak_and_global_use_available_cases(self):
        from build_peak_sensitive_artifacts import paired_effects
        from peak_sensitive_result_validation import FROZEN_CONFIG

        task, budget = "controlled_fano", 256
        records = []
        cells = {}
        for family in FROZEN_CONFIG["families"]:
            cells[(task, budget, None, family)] = {}
            for instance_index, data_seed in enumerate(FROZEN_CONFIG["evaluation_data_seeds"]):
                for init_seed in FROZEN_CONFIG["evaluation_init_seeds"][:2]:
                    no_feature = family == "KAN"
                    failure = family == "MLP" and instance_index == 0 and init_seed == 11003
                    peak = None if no_feature or failure else 0.2 + 0.01 * instance_index
                    global_score = None if failure else 0.3 + 0.01 * instance_index
                    records.append({
                        "task": task,
                        "selected_common_budget": budget,
                        "frf_density": None,
                        "family": family,
                        "data_seed": data_seed,
                        "init_seed": init_seed,
                        "dataset": {"curve_id": f"curve:{data_seed}"},
                        "window_spec": {
                            "status": "no_identifiable_feature" if no_feature else "ok",
                        },
                        "test_metrics": {
                            "status": "numerical_failure" if failure else "ok",
                            "resonance_window_complex_nrmse": peak,
                            "global_complex_nrmse": global_score,
                        },
                    })
        rows = paired_effects(records, cells)
        mlp = next(row for row in rows if row["baseline"] == "MLP")
        kan = next(row for row in rows if row["baseline"] == "KAN")
        self.assertEqual(mlp["failure_init_count"], 10)
        self.assertEqual(mlp["global_init_count"], 9)
        self.assertEqual(mlp["peak_init_count"], 9)
        self.assertEqual(kan["peak_state"], "not_estimable")
        self.assertIsNone(kan["peak_difference"])
        self.assertIsNone(kan["peak_p_value"])

    def test_holm_is_one_seven_contrast_family_per_task(self):
        from build_peak_sensitive_artifacts import _holm_adjust

        rows = [
            {"task": "aluminium_frf", "baseline": f"baseline-{index}", "p_value": 0.01 * index}
            for index in range(1, 8)
        ]
        _holm_adjust(rows, p_value_field="p_value", output_field="holm_p_value")
        self.assertTrue(all(row["holm_family_size"] == 7 for row in rows))
        self.assertTrue(all(row["holm_family_key"] == "aluminium_frf" for row in rows))


class MacroTests(unittest.TestCase):
    def _compile_all_macros(self, macros: Path) -> None:
        names = re.findall(r"\\newcommand\{\\([^}]+)\}", macros.read_text())
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            local_macros = root / "macros.tex"
            shutil.copy2(macros, local_macros)
            document = [
                r"\documentclass{article}", r"\begin{document}", r"\input{macros.tex}",
                *(f"\\{name}\\par" for name in names), r"\end{document}",
            ]
            (root / "expand.tex").write_text("\n".join(document) + "\n")
            result = subprocess.run(
                ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", "expand.tex"],
                cwd=root, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_generated_macros_are_letter_only_unique_and_all_expand_in_pdflatex(self):
        from build_peak_sensitive_artifacts import _write_macros

        effects = [{
            "task": "aluminium_frf",
            "baseline": "Fourier-feature MLP",
            "frf_density": 192,
            "peak_state": "not_estimable",
            "peak_difference": None,
            "peak_ci_low": None,
            "peak_ci_high": None,
            "peak_p_value": None,
            "holm_peak_p_value": None,
            "advantage_decision": "not_estimable_no_feature",
        }]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            macros = root / "macros.tex"
            _write_macros(macros, effects, [])
            text = macros.read_text()
            names = re.findall(r"\\newcommand\{\\([^}]+)\}", text)
            self.assertTrue(names)
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(re.fullmatch(r"[A-Za-z]+", name) for name in names))
            self.assertIn("not estimable", text)
            self._compile_all_macros(macros)

    def test_committed_fixture_expands_every_generated_macro(self):
        macros = ROOT / "peak_sensitive_results" / "artifact_test" / "latex_macros.tex"
        self.assertTrue(macros.exists())
        self._compile_all_macros(macros)


class CanonicalArtifactIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_fixtures.peak_sensitive_artifacts.generate_fixture import generate_fixture_input

        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.input_root = Path(cls.temporary_directory.name) / "input"
        cls.fixture_manifest = generate_fixture_input(cls.input_root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_real_runner_freeze_is_consumed_as_full_canonical_scope(self):
        from build_peak_sensitive_artifacts import _load_selected_cells
        from peak_sensitive_result_validation import FROZEN_CONFIG

        frozen, _, cells, _, lineage = _load_selected_cells(
            self.input_root, FROZEN_CONFIG, artifact_mode="test_fixture",
        )
        self.assertEqual(frozen["scope"]["tasks"], FROZEN_CONFIG["fixed_scientific_tasks"])
        self.assertEqual(frozen["scope"]["families"], FROZEN_CONFIG["families"])
        self.assertEqual(frozen["scope"]["budgets"], FROZEN_CONFIG["budgets"])
        self.assertEqual(frozen["scope"]["frf_densities"], FROZEN_CONFIG["frf_densities"])
        self.assertEqual(len(lineage), 288)
        self.assertEqual(len(cells), 72)
        self.assertEqual({key[1] for key in cells}, {256})

    def test_fixture_spec_and_manifest_declare_the_exact_materialized_matrix(self):
        from peak_sensitive_result_validation import FROZEN_CONFIG

        spec_path = ROOT / "test_fixtures" / "peak_sensitive_artifacts" / "fixture_spec.json"
        spec = json.loads(spec_path.read_text())
        expected_scope = {
            "tasks": FROZEN_CONFIG["fixed_scientific_tasks"],
            "families": FROZEN_CONFIG["families"],
            "budgets": FROZEN_CONFIG["budgets"],
            "frf_densities": FROZEN_CONFIG["frf_densities"],
        }
        self.assertEqual(spec.get("scope"), expected_scope)
        self.assertEqual(spec.get("expected_selected_cells"), 72)
        self.assertEqual(spec.get("record_count"), 3600)
        self.assertEqual(self.fixture_manifest.get("scope"), expected_scope)
        self.assertEqual(self.fixture_manifest.get("selected_cells"), 72)
        self.assertEqual(self.fixture_manifest["record_count"], 3600)

    def test_production_path_rejects_fixture_mode(self):
        from build_peak_sensitive_artifacts import audit_evaluation_records

        with self.assertRaisesRegex(ValueError, "production.*test_fixture"):
            audit_evaluation_records(self.input_root, artifact_mode="production")

    def test_truncated_frozen_task_scope_is_rejected(self):
        from build_peak_sensitive_artifacts import audit_evaluation_records

        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / "input"
            shutil.copytree(self.input_root, copied)
            frozen_path = copied / "frozen_task_budgets.json"
            frozen = json.loads(frozen_path.read_text())
            del frozen["tasks"]["controlled_fano"]
            frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "exact six-task production scope"):
                audit_evaluation_records(copied, artifact_mode="test_fixture")

    @staticmethod
    def _rehash_selection_lineage(root: Path, *, task: str, selection_path: Path) -> None:
        frozen_path = root / "frozen_task_budgets.json"
        frozen = json.loads(frozen_path.read_text())
        raw_path = str(selection_path)
        digest = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        relative_path = selection_path.relative_to(root)
        original_path = next(
            path for path in frozen["tasks"][task]["selection_file_sha256"]
            if Path(path).parts[-len(relative_path.parts):] == relative_path.parts
        )
        if original_path != raw_path:
            del frozen["tasks"][task]["selection_file_sha256"][original_path]
            del frozen["selection_file_sha256"][original_path]
        frozen["tasks"][task]["selection_file_sha256"][raw_path] = digest
        frozen["selection_file_sha256"][raw_path] = digest
        frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")

    def test_headline_nonselection_without_hash_bound_canonical_exclusion_is_rejected(self):
        from build_peak_sensitive_artifacts import audit_evaluation_records
        from peak_sensitive_benchmark import _selection_path

        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / "input"
            shutil.copytree(self.input_root, copied)
            selection_path = _selection_path(copied, "controlled_fano", "MLP", 256)
            selection = json.loads(selection_path.read_text())
            selection["status"] = "missing_selection"
            selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
            self._rehash_selection_lineage(
                copied, task="controlled_fano", selection_path=selection_path,
            )
            next((copied / "evaluation" / "controlled_fano").glob("MLP__256__*.jsonl")).unlink()

            with self.assertRaisesRegex(ValueError, "non-selected.*canonical exclusion"):
                audit_evaluation_records(copied, artifact_mode="test_fixture")

    def test_rehashed_nonheadline_selection_candidate_must_match_inventory(self):
        from build_peak_sensitive_artifacts import audit_evaluation_records
        from peak_sensitive_benchmark import _selection_path

        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / "input"
            shutil.copytree(self.input_root, copied)
            selection_path = _selection_path(copied, "controlled_fano", "MLP", 128)
            selection = json.loads(selection_path.read_text())
            forged_config = {"forged": True, "target_budget": 999999}
            selection["candidate"] = {
                "candidate_key": json.dumps(forged_config, sort_keys=True, separators=(",", ":")),
                "config": forged_config,
                "actual_parameters": 999999,
                "budget_relative_error": 0.0,
                "target_budget": 128,
            }
            selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
            self._rehash_selection_lineage(
                copied, task="controlled_fano", selection_path=selection_path,
            )

            with self.assertRaisesRegex(ValueError, "selection candidate.*candidate inventory"):
                audit_evaluation_records(copied, artifact_mode="test_fixture")

    def test_full_fixture_build_has_dynamic_nonblank_figures_and_provenance(self):
        from build_peak_sensitive_artifacts import build_artifacts

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "artifact_test"
            manifest = build_artifacts(
                self.input_root, output, artifact_mode="test_fixture",
            )
            self.assertEqual(manifest["artifact_mode"], "test_fixture")
            self.assertEqual(manifest["evaluation_records"], 3600)
            self.assertEqual(manifest["selected_cells"], 72)
            self.assertEqual(manifest["fixture_generator"], self.fixture_manifest["generator"])
            self.assertIn("builder_sha256", manifest["statistical_code"])
            self.assertIn("validator_sha256", manifest["statistical_code"])
            self.assertEqual(
                manifest["governing_analysis_scope"]["frf_primary_comparisons"]["Vector fitting"]["state"],
                "excluded_non_realizable",
            )
            for name in (
                "parameter_curves", "eis_family_comparison", "frf_density_interaction",
                "paired_effects", "clipping_cap_rates", "representative_reconstructions",
            ):
                figure = manifest["figures"][name]
                self.assertEqual(figure["applicability"], "generated")
                self.assertTrue(figure["source_fields"])
                path = output / figure["path"]
                self.assertGreater(path.stat().st_size, 5_000)
                extracted = subprocess.run(
                    ["pdftotext", str(path), "-"], capture_output=True, text=True, check=True,
                ).stdout.strip()
                self.assertTrue(extracted)
            with (output / "budget_curves.csv").open() as handle:
                budget_rows = list(csv.DictReader(handle))
            self.assertEqual({int(row["target_budget"]) for row in budget_rows}, {128, 256, 512, 1024})
            self.assertTrue(all(int(row["actual_parameters"]) > 0 for row in budget_rows))

    def test_reconstruction_is_omitted_when_no_provenance_bound_arrays_exist(self):
        from build_peak_sensitive_artifacts import build_artifacts

        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / "input"
            shutil.copytree(self.input_root, copied)
            (copied / "reconstruction_inputs.json").unlink()
            output = Path(temporary_directory) / "output"
            manifest = build_artifacts(copied, output, artifact_mode="test_fixture")
            reconstruction = manifest["figures"]["representative_reconstructions"]
            self.assertEqual(reconstruction["applicability"], "unavailable")
            self.assertNotIn(reconstruction["path"], manifest["output_hashes"])
            self.assertFalse((output / reconstruction["path"]).exists())


if __name__ == "__main__":
    unittest.main()
