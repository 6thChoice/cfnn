import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from confirmatory_models import build_model, enumerate_candidates
from confirmatory_protocol import eligible_candidates_by_budget
from confirmatory_protocol import seeded_build
from confirmatory_protocol import DatasetBundle
from fair_baselines import parameter_count


class ModelTests(unittest.TestCase):
    def test_cfnn_alias_matches_legacy_parameterization(self):
        config = {
            "units": 3,
            "degree": 3,
            "epsilon": 0.1,
            "denominator": "squared",
            "shared_projection": False,
            "skip": True,
        }
        torch.manual_seed(17)
        cfnn, metadata = build_model("CFNN", 2, 2, config, seed=17)
        torch.manual_seed(17)
        parn, _ = build_model("PARN", 2, 2, config, seed=17)
        self.assertEqual(parameter_count(cfnn), parameter_count(parn))
        self.assertEqual(tuple(cfnn(torch.zeros(5, 2)).shape), (5, 2))
        self.assertEqual(metadata["family"], "CFNN")

    def test_cfnn_budget_inventory_contains_distinct_candidates(self):
        candidates = enumerate_candidates("CFNN", 1, 2)
        eligible = eligible_candidates_by_budget(candidates, [256], 0.05)[256]
        signatures = {json.dumps(candidate.config, sort_keys=True) for candidate in eligible}
        self.assertGreaterEqual(len(signatures), 3)


class RunnerTests(unittest.TestCase):
    def test_evaluate_refuses_unfrozen_conventional_tasks(self):
        from downstream_benchmark import load_confirmatory_tasks

        config = {
            "fixed_scientific_tasks": ["controlled_fano", "microwave_fano"],
            "conventional_candidates": ["energy", "credit", "airfoil", "appliances"],
            "conventional_task_count": 2,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "conventional_selection.json"
            with self.assertRaisesRegex(ValueError, "frozen conventional selection"):
                load_confirmatory_tasks(config, missing)

    def test_seed_is_set_before_every_model_build(self):
        first = seeded_build(17, lambda: torch.nn.Linear(3, 2)).state_dict()
        torch.rand(100)
        second = seeded_build(17, lambda: torch.nn.Linear(3, 2)).state_dict()
        self.assertTrue(all(torch.equal(first[key], second[key]) for key in first))

    def test_protocol_manifest_refuses_mixed_config_hashes(self):
        from downstream_benchmark import ensure_protocol_manifest

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run_manifest.json"
            first = {
                "config_sha256": "config-a",
                "code_files": [{"path": "/first/code.py", "sha256": "code-a"}],
            }
            ensure_protocol_manifest(path, first)
            ensure_protocol_manifest(path, {**first, "command": ["another-shard"]})
            with self.assertRaisesRegex(ValueError, "mixed protocol manifest"):
                ensure_protocol_manifest(path, {**first, "config_sha256": "config-b"})

    def test_launcher_dry_run_lists_exploration_shards(self):
        root = Path(__file__).resolve().parent
        result = subprocess.run(
            [
                "bash", str(root / "run_downstream_experiments.sh"),
                "--dry-run", "--stage", "explore", "--jobs", "2",
                "--output-root", "/tmp/cfnn-downstream-dry-run",
            ],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("--tasks energy", result.stdout)
        self.assertIn("--families CFNN", result.stdout)
        self.assertIn("--families Fourier-feature\\ MLP", result.stdout)
        self.assertIn("--budgets 512", result.stdout)

    def test_extension_launcher_lists_seed_and_group_transfer_shards(self):
        root = Path(__file__).resolve().parent
        seed = subprocess.run(
            [
                "bash", str(root / "run_downstream_extension.sh"),
                "--dry-run", "--protocol", "seed", "--jobs", "2",
                "--output-root", "/tmp/cfnn-seed-extension-dry-run",
            ],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("downstream_extension.py", seed.stdout)
        self.assertIn("--tasks controlled_fano", seed.stdout)
        self.assertIn("--budgets 1024", seed.stdout)
        transfer = subprocess.run(
            [
                "bash", str(root / "run_downstream_extension.sh"),
                "--dry-run", "--protocol", "group-transfer", "--stage", "tune",
                "--output-root", "/tmp/cfnn-group-transfer-dry-run",
            ],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("downstream_group_transfer_config.json", transfer.stdout)
        self.assertIn("--tasks microwave_fano_group_transfer", transfer.stdout)
        self.assertIn("--budgets 256", transfer.stdout)

        subset = subprocess.run(
            [
                "bash", str(root / "run_downstream_extension.sh"),
                "--dry-run", "--protocol", "group-transfer", "--stage", "tune",
                "--families", "CFNN,Gaussian RBF",
                "--output-root", "/tmp/cfnn-group-transfer-subset-dry-run",
            ],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("--families CFNN", subset.stdout)
        self.assertIn("--families Gaussian\\ RBF", subset.stdout)
        self.assertNotIn("--families MLP ", subset.stdout)

        task_subset = subprocess.run(
            [
                "bash", str(root / "run_downstream_extension.sh"),
                "--dry-run", "--protocol", "seed",
                "--tasks", "controlled_fano,energy",
                "--families", "CFNN",
                "--output-root", "/tmp/cfnn-seed-task-subset-dry-run",
            ],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("--tasks controlled_fano", task_subset.stdout)
        self.assertIn("--tasks energy", task_subset.stdout)
        self.assertNotIn("--tasks microwave_fano ", task_subset.stdout)

    def test_nonfinite_failure_record_serializes_and_ranks_last(self):
        from downstream_benchmark import _append_jsonl, _load_jsonl, _rank_candidates

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "records.jsonl"
            failed = {
                "candidate_key": "failed",
                "validation_score": float("inf"),
                "status": "nonfinite_train_loss",
            }
            valid = {
                "candidate_key": "valid",
                "validation_score": 0.25,
                "status": "ok",
            }
            _append_jsonl(path, failed)
            _append_jsonl(path, valid)
            records = _load_jsonl(path)
            self.assertIsNone(records[0]["validation_score"])
            self.assertEqual(_rank_candidates(records, 1), ["valid"])

    def test_config_dimensions_match_conventional_loaders(self):
        from downstream_protocol import make_conventional_bundle

        root = Path(__file__).resolve().parent
        config = json.loads((root / "downstream_config.json").read_text())
        raw_root = root / "downstream_data" / "raw"
        for task in config["conventional_candidates"]:
            with self.subTest(task=task):
                bundle = make_conventional_bundle(task, 101, raw_root, include_test=False)
                self.assertEqual(config["tasks"][task]["input_dim"], bundle.x_train.shape[1])

    def test_group_transfer_config_dimensions_match_bundles(self):
        from downstream_protocol import make_group_transfer_bundle

        root = Path(__file__).resolve().parent
        config = json.loads((root / "downstream_group_transfer_config.json").read_text())
        for task in config["fixed_scientific_tasks"]:
            with self.subTest(task=task):
                task_config = config["tasks"][task]
                bundle = make_group_transfer_bundle(
                    task, config["tuning_seeds"][0], task_config,
                    root / "downstream_data" / "raw",
                )
                self.assertEqual(task_config["input_dim"], bundle.x_train.shape[1])

    def test_group_transfer_inventory_uses_primary_budget(self):
        from downstream_benchmark import build_inventory

        root = Path(__file__).resolve().parent
        config = json.loads((root / "downstream_group_transfer_config.json").read_text())
        with tempfile.TemporaryDirectory() as temporary_directory:
            inventory = build_inventory(
                config, Path(temporary_directory),
                tasks=config["fixed_scientific_tasks"],
                families=config["families"], budgets=[256],
            )
            self.assertEqual(config["budgets"], [256])
            for task in config["fixed_scientific_tasks"]:
                self.assertGreaterEqual(
                    len(inventory["tasks"][task]["CFNN"]["256"]), 3
                )

    def test_group_transfer_task_trains_through_shared_runner(self):
        from downstream_benchmark import build_inventory, train_one

        root = Path(__file__).resolve().parent
        config = json.loads((root / "downstream_group_transfer_config.json").read_text())
        config["training"].update({
            "min_steps": 2, "max_steps": 5, "patience": 5,
            "validation_interval": 1, "checkpoints": [2, 5],
        })
        task = "microwave_fano_group_transfer"
        with tempfile.TemporaryDirectory() as temporary_directory:
            inventory = build_inventory(
                config, Path(temporary_directory), tasks=[task],
                families=["CFNN"], budgets=[256], maximum_candidates=3,
            )
            candidate = inventory["tasks"][task]["CFNN"]["256"][0]
            record = train_one(
                task, "CFNN", candidate, config["optimizer_grids"]["CFNN"][0],
                data_seed=233, init_seed=1009, config=config,
                raw_root=root / "downstream_data" / "raw", evaluate_test=True,
            )
            self.assertEqual(record["dataset"]["split_mode"], "group_transfer")
            self.assertEqual(record["primary_metric_name"], "complex_nrmse")
            self.assertTrue(np.isfinite(record["primary_metric_value"]))

    def test_inventory_uses_realized_parameter_budget(self):
        from downstream_benchmark import build_inventory

        root = Path(__file__).resolve().parent
        config = json.loads((root / "downstream_config.json").read_text())
        with tempfile.TemporaryDirectory() as temporary_directory:
            inventory = build_inventory(
                config,
                Path(temporary_directory),
                tasks=["controlled_fano"],
                families=["CFNN"],
                budgets=[256],
                maximum_candidates=3,
            )
            candidates = inventory["tasks"]["controlled_fano"]["CFNN"]["256"]
            self.assertEqual(len(candidates), 3)
            self.assertTrue(all(
                abs(candidate["actual_parameters"] - 256) / 256 <= 0.05
                for candidate in candidates
            ))

    def test_tuning_trains_every_frozen_candidate_before_selection(self):
        from downstream_benchmark import build_inventory, tune_shard

        root = Path(__file__).resolve().parent
        config = json.loads((root / "downstream_config.json").read_text())
        config["tuning_seeds"] = [101]
        config["finalists_per_budget"] = 1
        config["optimizer_grids"]["CFNN"] = config["optimizer_grids"]["CFNN"][:1]
        config["training"].update({
            "screening_steps": 5, "max_steps": 5, "min_steps": 2,
            "patience": 5, "checkpoints": [2, 5],
        })
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            inventory = build_inventory(
                config, output_root, tasks=["controlled_fano"], families=["CFNN"],
                budgets=[256], maximum_candidates=3,
            )
            selection = tune_shard(
                config, inventory, output_root, "controlled_fano", "CFNN", 256,
                root / "downstream_data" / "raw", stage="exploration",
            )
            self.assertEqual(selection["screened_candidate_count"], 3)
            records = list((output_root / "tuning_screen").rglob("*.jsonl"))
            self.assertEqual(len(records), 1)
            self.assertEqual(len(records[0].read_text().splitlines()), 3)

    def test_evaluation_is_resumable_without_duplicate_seed_pairs(self):
        from downstream_benchmark import build_inventory, evaluate_shard, tune_shard

        root = Path(__file__).resolve().parent
        config = json.loads((root / "downstream_config.json").read_text())
        config["tuning_seeds"] = [101]
        config["evaluation_data_seeds"] = [233]
        config["evaluation_init_seeds"] = [1009]
        config["finalists_per_budget"] = 1
        config["optimizer_grids"]["CFNN"] = config["optimizer_grids"]["CFNN"][:1]
        config["training"].update({
            "screening_steps": 5, "max_steps": 5, "min_steps": 2,
            "patience": 5, "checkpoints": [2, 5],
        })
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            inventory = build_inventory(
                config, output_root, tasks=["controlled_fano"], families=["CFNN"],
                budgets=[256], maximum_candidates=3,
            )
            tune_shard(
                config, inventory, output_root, "controlled_fano", "CFNN", 256,
                root / "downstream_data" / "raw", stage="confirmatory_tuning",
            )
            for _ in range(2):
                evaluate_shard(
                    config, output_root, "controlled_fano", "CFNN", 256,
                    root / "downstream_data" / "raw",
                )
            paths = list((output_root / "evaluation").rglob("*.jsonl"))
            self.assertEqual(len(paths), 1)
            records = [json.loads(line) for line in paths[0].read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["stage_role"], "confirmatory_test")
            self.assertIn("complex_nrmse", records[0]["test_metrics"])


class ExtensionProtocolTests(unittest.TestCase):
    def test_duplicate_retry_cleanup_preserves_audit_and_rejects_disagreement(self):
        from deduplicate_downstream_records import deduplicate_evaluation_root

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "results"
            path = root / "evaluation" / "task" / "model__256.jsonl"
            path.parent.mkdir(parents=True)
            record = {
                "task": "task", "family": "model", "target_budget": 256,
                "data_seed": 1, "init_seed": 2, "primary_metric_value": 0.25,
                "test_metrics": {"score": 0.25}, "optimizer_steps": 20,
                "curve": [{"step": 20, "validation_score": 0.3,
                           "wall_seconds": 1.0}],
                "training_wall_seconds": 1.0, "total_wall_seconds": 1.2,
                "peak_memory_bytes": 100,
            }
            retry = {**record, "training_wall_seconds": 2.0,
                     "total_wall_seconds": 2.2, "peak_memory_bytes": 120,
                     "curve": [{**record["curve"][0], "wall_seconds": 2.0}]}
            path.write_text("\n".join(json.dumps(x) for x in (record, retry)) + "\n")
            audit = deduplicate_evaluation_root(root)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            self.assertEqual(audit["removed_record_count"], 1)
            self.assertTrue(Path(audit["backups"][0]["backup_path"]).exists())

            path.write_text("\n".join(json.dumps(x) for x in (
                record, {**retry, "primary_metric_value": 0.5}
            )) + "\n")
            with self.assertRaisesRegex(ValueError, "scientifically different"):
                deduplicate_evaluation_root(root, audit_name="second_audit.json")

    def _fixture(self, root: Path):
        base_config = root / "downstream_config.json"
        base_config.write_text(json.dumps({"evaluation_init_seeds": [11, 13, 17, 19]}))
        base_root = root / "base"
        (base_root / "selections" / "task").mkdir(parents=True)
        (base_root / "candidate_manifest.json").write_text('{"candidate":"frozen"}')
        (base_root / "conventional_selection.json").write_text('{"selected_tasks":["a","b"]}')
        (base_root / "selections" / "task" / "cfnn__256.json").write_text(
            '{"status":"selected","candidate":{"candidate_key":"c"}}'
        )
        return base_config, base_root

    def _extension(self, base_config: Path, base_root: Path):
        from downstream_extension import sha256_file, selection_tree_sha256

        return {
            "schema_version": "1.0.0",
            "base_config_sha256": sha256_file(base_config),
            "base_candidate_manifest_sha256": sha256_file(
                base_root / "candidate_manifest.json"
            ),
            "base_conventional_selection_sha256": sha256_file(
                base_root / "conventional_selection.json"
            ),
            "base_selection_tree_sha256": selection_tree_sha256(base_root),
            "additional_evaluation_init_seeds": [23, 29, 31, 37, 41, 43],
        }

    def test_extension_rejects_seed_overlap(self):
        from downstream_extension import load_extension

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base_config, base_root = self._fixture(root)
            extension = self._extension(base_config, base_root)
            extension["additional_evaluation_init_seeds"][0] = 11
            path = root / "extension.json"
            path.write_text(json.dumps(extension))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_extension(path, base_config, base_root)

    def test_extension_rejects_changed_base_hash(self):
        from downstream_extension import load_extension

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base_config, base_root = self._fixture(root)
            extension = self._extension(base_config, base_root)
            path = root / "extension.json"
            path.write_text(json.dumps(extension))
            base_config.write_text('{"evaluation_init_seeds":[1]}')
            with self.assertRaisesRegex(ValueError, "base_config_sha256"):
                load_extension(path, base_config, base_root)

    def test_extension_rejects_changed_selection_tree(self):
        from downstream_extension import load_extension

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base_config, base_root = self._fixture(root)
            extension = self._extension(base_config, base_root)
            path = root / "extension.json"
            path.write_text(json.dumps(extension))
            (base_root / "selections" / "task" / "cfnn__256.json").write_text(
                '{"status":"selected","candidate":{"candidate_key":"changed"}}'
            )
            with self.assertRaisesRegex(ValueError, "selection_tree"):
                load_extension(path, base_config, base_root)

    def test_seed_extension_is_append_only_and_resumable(self):
        from downstream_extension import (
            evaluate_seed_extension,
            selection_tree_sha256,
            sha256_file,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base_config = root / "config.json"
            base_config.write_text(json.dumps({
                "evaluation_data_seeds": [233],
                "evaluation_init_seeds": [2, 3, 5, 7, 11, 13, 17, 19, 23],
            }))
            base_root = root / "base"
            selection_path = base_root / "selections" / "controlled_fano" / "cfnn__256.json"
            selection_path.parent.mkdir(parents=True)
            selection_path.write_text(json.dumps({
                "status": "selected",
                "candidate": {"candidate_key": "candidate", "config": {}, "actual_parameters": 250},
                "optimizer": {"key": "adam", "lr": 0.001, "weight_decay": 0.0},
            }))
            (base_root / "candidate_manifest.json").write_text("{}")
            (base_root / "conventional_selection.json").write_text("{}")
            base_evaluation = base_root / "evaluation" / "controlled_fano" / "cfnn__256.jsonl"
            base_evaluation.parent.mkdir(parents=True)
            base_evaluation.write_text('{"base":true}\n')
            original_base_hash = sha256_file(base_evaluation)
            extension_path = root / "extension.json"
            extension_path.write_text(json.dumps({
                "schema_version": "1.0.0",
                "base_config_sha256": sha256_file(base_config),
                "base_candidate_manifest_sha256": sha256_file(base_root / "candidate_manifest.json"),
                "base_conventional_selection_sha256": sha256_file(base_root / "conventional_selection.json"),
                "base_selection_tree_sha256": selection_tree_sha256(base_root),
                "additional_evaluation_init_seeds": [29],
            }))
            calls = []

            def fake_train(task, family, candidate, optimizer, **kwargs):
                calls.append((kwargs["data_seed"], kwargs["init_seed"]))
                return {
                    "task": task, "family": family,
                    "data_seed": kwargs["data_seed"], "init_seed": kwargs["init_seed"],
                }

            output_root = root / "extension-results"
            for _ in range(2):
                evaluate_seed_extension(
                    base_config, extension_path, base_root, output_root,
                    "controlled_fano", "CFNN", 256, root / "raw",
                    train_fn=fake_train,
                )
            self.assertEqual(calls, [(233, 29)])
            self.assertEqual(sha256_file(base_evaluation), original_base_hash)
            output = output_root / "evaluation" / "controlled_fano" / "cfnn__256.jsonl"
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["stage"], "confirmatory_seed_extension")
            self.assertEqual(records[0]["base_selection_sha256"], sha256_file(selection_path))
            self.assertEqual(records[0]["protocol_extension_sha256"], sha256_file(extension_path))


class TrainingTests(unittest.TestCase):
    def test_prepare_bundle_normalizes_large_coordinates_before_float32_cast(self):
        from downstream_training import prepare_bundle

        offset = 9_473_451_000.0
        bundle = DatasetBundle(
            x_train=np.asarray([[offset], [offset + 500.0], [offset + 1000.0]], dtype=np.float64),
            y_train=np.asarray([[0.0], [1.0], [0.0]], dtype=np.float32),
            x_validation=np.asarray([[offset + 250.0]], dtype=np.float64),
            y_validation=np.asarray([[0.5]], dtype=np.float32),
            x_test=np.asarray([[offset + 750.0]], dtype=np.float64),
            y_test=np.asarray([[0.5]], dtype=np.float32),
            metadata={},
        )
        prepared = prepare_bundle(bundle, "regression", torch.device("cpu"))
        self.assertEqual(len(torch.unique(prepared.x_train[:, 0])), 3)
        self.assertAlmostEqual(float(prepared.x_train[0, 0]), -1.2247449, places=5)
        self.assertAlmostEqual(float(prepared.x_train[2, 0]), 1.2247449, places=5)

    def test_prepare_bundle_fits_scaling_on_training_rows_only(self):
        import numpy as np

        from downstream_training import prepare_bundle

        bundle = DatasetBundle(
            x_train=np.asarray([[0.0], [2.0]], dtype=np.float32),
            y_train=np.asarray([[0.0], [2.0]], dtype=np.float32),
            x_validation=np.asarray([[101.0]], dtype=np.float32),
            y_validation=np.asarray([[101.0]], dtype=np.float32),
            x_test=np.asarray([[201.0]], dtype=np.float32),
            y_test=np.asarray([[201.0]], dtype=np.float32),
            metadata={},
        )
        prepared = prepare_bundle(bundle, "regression", torch.device("cpu"))
        self.assertAlmostEqual(float(prepared.x_train.mean()), 0.0, places=6)
        self.assertGreater(float(prepared.x_validation.mean()), 50.0)
        self.assertAlmostEqual(float(prepared.x_mean[0, 0]), 1.0)

    def test_classification_training_produces_finite_validation_score(self):
        import numpy as np

        from downstream_training import fit_task_model, prepare_bundle

        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 2)).astype(np.float32)
        y = (x[:, 0] + x[:, 1] > 0).astype(np.float32).reshape(-1, 1)
        bundle = DatasetBundle(
            x_train=x[:80], y_train=y[:80],
            x_validation=x[80:100], y_validation=y[80:100],
            x_test=x[100:], y_test=y[100:], metadata={},
        )
        prepared = prepare_bundle(bundle, "classification", torch.device("cpu"))
        model = torch.nn.Linear(2, 1)
        result = fit_task_model(
            model, prepared, "classification", learning_rate=0.03,
            weight_decay=0.0, max_steps=80, min_steps=20, patience=30,
            checkpoints=(20, 80), gradient_clip=5.0,
        )
        self.assertTrue(np.isfinite(result.best_validation_score))
        self.assertEqual(result.prediction.shape, (20, 1))
        self.assertTrue(np.all((result.prediction >= 0) & (result.prediction <= 1)))

    def test_tuning_record_does_not_contain_test_metrics(self):
        from downstream_benchmark import train_one

        root = Path(__file__).resolve().parent
        config = json.loads((root / "downstream_config.json").read_text())
        config["training"].update({"max_steps": 20, "min_steps": 5, "patience": 10})
        candidates = enumerate_candidates("CFNN", 1, 2)
        candidate = eligible_candidates_by_budget(candidates, [256], 0.05)[256][0]
        candidate_record = {
            "candidate_key": candidate.key,
            "actual_parameters": candidate.actual_parameters,
            "config": candidate.config,
        }
        record = train_one(
            "controlled_fano", "CFNN", candidate_record,
            config["optimizer_grids"]["CFNN"][0],
            data_seed=101, init_seed=1009, config=config,
            raw_root=root / "downstream_data" / "raw", evaluate_test=False,
        )
        self.assertNotIn("test_metrics", record)
        self.assertEqual(record["stage_role"], "validation_only")

    def test_nonfinite_classification_logits_return_infinite_score(self):
        from downstream_training import _classification_score

        logits = torch.tensor([[float("nan")], [0.0]])
        target = torch.tensor([[0.0], [1.0]])
        self.assertTrue(np.isinf(_classification_score(logits, target)))

    def test_minibatch_training_respects_size_and_explicit_seed(self):
        from downstream_training import fit_task_model, prepare_bundle

        class RecordingLinear(torch.nn.Linear):
            def __init__(self):
                super().__init__(2, 1)
                self.batch_sizes = []

            def forward(self, values):
                if self.training:
                    self.batch_sizes.append(len(values))
                return super().forward(values)

        rng = np.random.default_rng(19)
        x = rng.normal(size=(96, 2)).astype(np.float32)
        y = (x[:, :1] - x[:, 1:]).astype(np.float32)
        bundle = DatasetBundle(
            x_train=x[:64], y_train=y[:64],
            x_validation=x[64:80], y_validation=y[64:80],
            x_test=x[80:], y_test=y[80:], metadata={},
        )
        prepared = prepare_bundle(bundle, "regression", torch.device("cpu"))

        predictions = []
        for _ in range(2):
            torch.manual_seed(23)
            model = RecordingLinear()
            torch.rand(100)
            result = fit_task_model(
                model, prepared, "regression", learning_rate=0.01,
                weight_decay=0.0, max_steps=12, min_steps=4, patience=12,
                checkpoints=(4, 12), gradient_clip=5.0,
                batch_size=8, batch_seed=991,
                validation_interval=4,
            )
            self.assertLessEqual(max(model.batch_sizes), 8)
            self.assertEqual(result.validation_evaluations, 3)
            predictions.append(result.prediction)
        np.testing.assert_array_equal(predictions[0], predictions[1])


class SelectionTests(unittest.TestCase):
    @staticmethod
    def _records(all_fail=False):
        records = []
        tasks = {
            "energy": "regression",
            "credit": "classification",
            "airfoil": "regression",
            "appliances": "regression",
        }
        for task_index, (task, problem_type) in enumerate(tasks.items()):
            for budget in (128, 256, 512):
                for seed in (101, 211, 307):
                    baseline = 0.20 + 0.01 * task_index - 0.0001 * budget
                    cfnn = baseline + (0.03 if all_fail else (-0.02 if task in {"energy", "credit"} else 0.02))
                    for family, score in (("CFNN", cfnn), ("MLP", baseline)):
                        records.append({
                            "task": task,
                            "problem_type": problem_type,
                            "family": family,
                            "target_budget": budget,
                            "actual_parameters": budget,
                            "tuning_seed": seed,
                            "validation_score": score + seed * 1e-7,
                            "training_wall_seconds": 1.0,
                            "status": "ok",
                            "stage_role": "validation_only",
                            "dataset": {"sealed_test_row_ids_sha256": f"sealed-{task}"},
                            "test_metric": 1.0 if task_index % 2 else -1.0,
                        })
        return records

    def test_selection_is_invariant_to_test_metric_fields(self):
        from select_conventional_tasks import select_tasks

        config = {
            "primary_family": "CFNN",
            "conventional_task_count": 2,
            "conventional_candidates": ["energy", "credit", "airfoil", "appliances"],
            "equivalence_margins": {
                "nrmse": 0.01, "auroc": 0.01, "balanced_accuracy": 0.01,
                "parameter_reduction": 0.25, "maximum_runtime_ratio": 10.0,
            },
        }
        first = select_tasks(self._records(), config)
        changed = [{**record, "test_metric": record["test_metric"] * -1000} for record in self._records()]
        second = select_tasks(changed, config)
        self.assertEqual(first["selected_tasks"], second["selected_tasks"])
        self.assertNotIn("test_metric", json.dumps(first))

    def test_fallback_is_labeled_compatibility_only(self):
        from select_conventional_tasks import select_tasks

        config = {
            "primary_family": "CFNN",
            "conventional_task_count": 2,
            "conventional_candidates": ["energy", "credit", "airfoil", "appliances"],
            "equivalence_margins": {
                "nrmse": 0.01, "auroc": 0.01, "balanced_accuracy": 0.01,
                "parameter_reduction": 0.25, "maximum_runtime_ratio": 10.0,
            },
        }
        selected = select_tasks(self._records(all_fail=True), config)
        self.assertEqual(len(selected["selected_tasks"]), 2)
        self.assertEqual(selected["claim_level"], "compatibility_only")

    def test_winner_loader_uses_runner_slug_for_hyphenated_family(self):
        from select_conventional_tasks import _load_winner_records

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            selection_dir = root / "selections" / "energy"
            selection_dir.mkdir(parents=True)
            selection = {
                "status": "selected",
                "task": "energy",
                "family": "Fourier-feature MLP",
                "target_budget": 256,
                "candidate": {"candidate_key": "candidate-a"},
                "optimizer": {"key": "optimizer-a"},
            }
            (selection_dir / "fourier_feature_mlp__256.json").write_text(
                json.dumps(selection)
            )
            tuning_dir = root / "tuning_final" / "energy"
            tuning_dir.mkdir(parents=True)
            record = {
                "candidate_key": "candidate-a",
                "optimizer": {"key": "optimizer-a"},
            }
            (tuning_dir / "fourier_feature_mlp__256.jsonl").write_text(
                json.dumps(record) + "\n"
            )

            self.assertEqual(_load_winner_records(root), [record])


class ArtifactTests(unittest.TestCase):
    def test_retry_audit_inventory_hashes_audit_and_backups(self):
        from build_downstream_extension_artifacts import collect_retry_audits
        from downstream_extension import sha256_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            retry_root = root / "invalidated" / "retry_duplicates"
            backup = retry_root / "evaluation" / "task" / "cfnn__256.jsonl"
            backup.parent.mkdir(parents=True)
            backup.write_text('{"duplicate":true}\n')
            audit = retry_root / "deduplication_audit.json"
            audit.write_text('{"removed":1}\n')

            inventory = collect_retry_audits("seed_extension", root)

            self.assertEqual(len(inventory), 1)
            self.assertEqual(inventory[0]["evidence_root"], "seed_extension")
            self.assertEqual(inventory[0]["audit_sha256"], sha256_file(audit))
            self.assertEqual(inventory[0]["backups"], [{
                "path": "invalidated/retry_duplicates/evaluation/task/cfnn__256.jsonl",
                "sha256": sha256_file(backup),
            }])

    def test_extension_builder_rejects_changed_base_evidence(self):
        from build_downstream_extension_artifacts import validate_base_evidence_unchanged
        from downstream_extension import sha256_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            record = root / "evaluation" / "task" / "cfnn__256.jsonl"
            record.parent.mkdir(parents=True)
            record.write_text('{"frozen":true}\n')
            derived = root / "derived"
            derived.mkdir()
            (derived / "artifact_manifest.json").write_text(json.dumps({
                "raw_inputs": [{
                    "path": "evaluation/task/cfnn__256.jsonl",
                    "sha256": sha256_file(record),
                }]
            }))
            validate_base_evidence_unchanged(root)
            record.write_text('{"frozen":false}\n')
            with self.assertRaisesRegex(ValueError, "base evidence hash"):
                validate_base_evidence_unchanged(root)

    def test_extension_validator_rejects_overlap_and_changed_selection(self):
        from build_downstream_extension_artifacts import validate_seed_extension_records
        from downstream_extension import sha256_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            selection = root / "selections" / "task" / "cfnn__256.json"
            selection.parent.mkdir(parents=True)
            selection.write_text('{"status":"selected"}')
            config = {
                "evaluation_data_seeds": [233],
                "evaluation_init_seeds": [1009],
                "budget_tolerance": 0.05,
            }
            inventory = {"tasks": {"task": {"CFNN": {"256": [{"candidate_key": "a"}]}}}}
            base = [{
                "task": "task", "family": "CFNN", "target_budget": 256,
                "data_seed": 233, "init_seed": 1009, "actual_parameters": 256,
            }]
            extension_config = {
                "additional_evaluation_init_seeds": [2027],
                "protocol_sha256": "a" * 64,
            }
            extended = [{
                "task": "task", "family": "CFNN", "target_budget": 256,
                "data_seed": 233, "init_seed": 2027, "actual_parameters": 256,
                "stage": "confirmatory_seed_extension",
                "protocol_extension_sha256": "a" * 64,
                "base_selection_sha256": sha256_file(selection),
            }]
            validate_seed_extension_records(
                base, extended, config, inventory, extension_config, root,
                tasks=["task"], families=["CFNN"], budgets=[256],
            )
            overlapping = [{**extended[0], "init_seed": 1009}]
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_seed_extension_records(
                    base, overlapping, config, inventory, extension_config, root,
                    tasks=["task"], families=["CFNN"], budgets=[256],
                )
            changed = [{**extended[0], "base_selection_sha256": "b" * 64}]
            with self.assertRaisesRegex(ValueError, "selection hash"):
                validate_seed_extension_records(
                    base, changed, config, inventory, extension_config, root,
                    tasks=["task"], families=["CFNN"], budgets=[256],
                )

    def test_group_transfer_validator_rejects_curve_leakage(self):
        from build_downstream_extension_artifacts import validate_group_transfer_records

        config = {"evaluation_data_seeds": [233], "evaluation_init_seeds": [1009]}
        inventory = {"tasks": {"task_group_transfer": {"CFNN": {"256": [{}]}}}}
        record = {
            "task": "task_group_transfer", "family": "CFNN", "target_budget": 256,
            "data_seed": 233, "init_seed": 1009,
            "dataset": {
                "split_mode": "group_transfer",
                "train_curve_ids": ["train"],
                "validation_curve_ids": ["validation"],
                "test_curve_ids": ["test"],
            },
        }
        validate_group_transfer_records(
            [record], config, inventory,
            tasks=["task_group_transfer"], families=["CFNN"], budgets=[256],
        )
        leaking = [{
            **record,
            "dataset": {**record["dataset"], "test_curve_ids": ["train"]},
        }]
        with self.assertRaisesRegex(ValueError, "curve leakage"):
            validate_group_transfer_records(
                leaking, config, inventory,
                tasks=["task_group_transfer"], families=["CFNN"], budgets=[256],
            )

    def test_latex_macro_names_preserve_distinct_budgets(self):
        from build_downstream_artifacts import render_latex_macros

        rows = [{
            "task": "energy", "family": "CFNN", "target_budget": budget,
            "metric_median": 0.1, "n": 20, "actual_parameters_median": budget,
        } for budget in (128, 256)]
        rendered = render_latex_macros(rows, [])
        self.assertIn("DownstreamEnergyCFNNBOneTwoEightMetric", rendered)
        self.assertIn("DownstreamEnergyCFNNBTwoFiveSixMetric", rendered)

    def test_error_metrics_use_log_scale_but_auroc_does_not(self):
        from build_downstream_artifacts import _metric_uses_log_scale

        self.assertTrue(_metric_uses_log_scale("complex_nrmse"))
        self.assertTrue(_metric_uses_log_scale("nrmse"))
        self.assertFalse(_metric_uses_log_scale("auroc"))

    def test_primary_effect_figure_marks_zero_reference(self):
        from build_downstream_artifacts import render_primary_effects

        comparisons = [{
            "task": "controlled_fano",
            "family_b": "MLP",
            "target_budget": 256,
            "median_difference": -0.04,
            "ci_low": -0.06,
            "ci_high": -0.01,
        }]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "effects.pdf"
            render_primary_effects(comparisons, 256, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 5_000)

    def test_task_example_figure_is_generated_from_frozen_bundles(self):
        from build_downstream_artifacts import render_task_examples

        root = Path(__file__).resolve().parent
        config = json.loads((root / "downstream_config.json").read_text())
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "examples.pdf"
            render_task_examples(
                config, root / "downstream_data" / "raw", output, data_seed=233
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 10_000)

    def test_domain_reference_summary_is_separate_and_robust(self):
        from build_downstream_artifacts import summarize_domain_records

        records = [
            {
                "task": "microstrip_resonator",
                "method": "vector_fitting",
                "status": "ok",
                "selected_max_poles": 4,
                "test_metrics": {"complex_nrmse": 0.02},
                "test_fit_metadata": {"convergence_warning_count": 1},
            },
            {
                "task": "microstrip_resonator",
                "method": "vector_fitting",
                "status": "ok",
                "selected_max_poles": 8,
                "test_metrics": {"complex_nrmse": 0.04},
                "test_fit_metadata": {"convergence_warning_count": 0},
            },
        ]
        summary = summarize_domain_records(records)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["metric_median"], 0.03)
        self.assertEqual(summary[0]["selected_order_median"], 6.0)
        self.assertEqual(summary[0]["failure_rate"], 0.0)
        self.assertEqual(summary[0]["convergence_warning_rate"], 0.5)

    def test_evaluation_record_has_primary_metric_and_compact_ids(self):
        from downstream_benchmark import train_one

        root = Path(__file__).resolve().parent
        config = json.loads((root / "downstream_config.json").read_text())
        config["training"].update({"max_steps": 5, "min_steps": 2, "patience": 5})
        candidate = eligible_candidates_by_budget(
            enumerate_candidates("CFNN", 1, 2), [256], 0.05
        )[256][0]
        record = train_one(
            "controlled_fano", "CFNN",
            {"candidate_key": candidate.key, "actual_parameters": candidate.actual_parameters,
             "config": candidate.config},
            config["optimizer_grids"]["CFNN"][0],
            data_seed=233, init_seed=1009, config=config,
            raw_root=root / "downstream_data" / "raw", evaluate_test=True,
        )
        self.assertEqual(record["primary_metric_name"], "complex_nrmse")
        self.assertTrue(np.isfinite(record["primary_metric_value"]))
        self.assertNotIn("test_point_ids", record["dataset"])
        self.assertEqual(record["dataset"]["test_point_ids_count"], 2048)
        self.assertIn("test_point_ids_sha256", record["dataset"])

    def test_schema_accepts_smoke_evaluation_record(self):
        from jsonschema import Draft202012Validator

        root = Path(__file__).resolve().parent
        schema_path = root / "downstream_results_schema.json"
        self.assertTrue(schema_path.exists())
        schema = json.loads(schema_path.read_text())
        record = json.loads(
            (root / "downstream_smoke" / "evaluation" / "controlled_fano" / "cfnn__256.jsonl")
            .read_text().splitlines()[0]
        )
        errors = list(Draft202012Validator(schema).iter_errors(record))
        self.assertEqual(errors, [])

    def test_schema_accepts_seed_extension_provenance(self):
        from jsonschema import Draft202012Validator

        root = Path(__file__).resolve().parent
        schema = json.loads((root / "downstream_results_schema.json").read_text())
        record = json.loads(
            (root / "downstream_smoke" / "evaluation" / "controlled_fano" / "cfnn__256.jsonl")
            .read_text().splitlines()[0]
        )
        record.update({
            "stage": "confirmatory_seed_extension",
            "protocol_extension_sha256": "a" * 64,
            "base_selection_sha256": "b" * 64,
        })
        errors = list(Draft202012Validator(schema).iter_errors(record))
        self.assertEqual(errors, [])

    def test_primary_comparisons_adjust_only_256_budget(self):
        from build_downstream_artifacts import paired_comparisons

        records = []
        for budget in (128, 256):
            for family, value in (("CFNN", 0.10), ("MLP", 0.15)):
                records.append({
                    "task": "controlled_fano", "family": family,
                    "target_budget": budget, "data_seed": 233, "init_seed": 1009,
                    "problem_type": "complex_regression",
                    "primary_metric_value": value,
                    "test_metrics": {"status": "ok"}, "status": "ok",
                })
        comparisons = paired_comparisons(records, "CFNN", 256, draws=100)
        adjusted = [row for row in comparisons if row["p_holm"] is not None]
        self.assertTrue(adjusted)
        self.assertEqual({row["target_budget"] for row in adjusted}, {256})

    def test_incomplete_matrix_is_rejected(self):
        from build_downstream_artifacts import validate_completeness

        config = {
            "evaluation_data_seeds": [233], "evaluation_init_seeds": [1009],
        }
        inventory = {"tasks": {"controlled_fano": {"CFNN": {"256": [{"candidate_key": "x"}]}}}}
        with self.assertRaisesRegex(ValueError, "incomplete downstream matrix"):
            validate_completeness(
                [], config, inventory, tasks=["controlled_fano"],
                families=["CFNN"], budgets=[256],
            )


if __name__ == "__main__":
    unittest.main()
