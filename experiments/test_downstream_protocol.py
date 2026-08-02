import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FANO_CONFIG = {
    "n_train": 64,
    "n_validation": 32,
    "n_test": 512,
    "noise": 0.01,
    "domain": [0.0, 1.0],
}


class SourceTests(unittest.TestCase):
    def test_zenodo_license_uses_current_metadata_field(self):
        from download_downstream_datasets import _zenodo_license

        self.assertEqual(
            _zenodo_license({"metadata": {"license": {"id": "cc-by-4.0"}}}),
            "cc-by-4.0",
        )

    def test_source_config_has_fixed_primary_records(self):
        config = json.loads((ROOT / "downstream_sources.json").read_text())
        self.assertEqual(config["microwave_fano"]["record_id"], 7767046)
        self.assertEqual(config["microstrip_resonator"]["record_id"], 14175959)
        self.assertEqual(config["aluminium_frf"]["record_id"], 7758683)
        self.assertTrue(all(item["url"].startswith("https://") for item in config.values()))

    def test_manifest_checksum_matches_download(self):
        from download_downstream_datasets import download_sources

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_file = root / "fixture.bin"
            source_file.write_bytes(b"downstream-source-fixture")
            config_path = root / "sources.json"
            config_path.write_text(json.dumps({
                "fixture": {
                    "provider": "direct",
                    "url": source_file.as_uri(),
                    "filename": "fixture.bin",
                    "role": "test_fixture",
                }
            }))
            raw_root = root / "raw"
            manifest = download_sources(config_path, raw_root)
            downloaded = raw_root / manifest["files"][0]["path"]
            expected = hashlib.sha256(downloaded.read_bytes()).hexdigest()
            self.assertEqual(manifest["files"][0]["sha256"], expected)

    def test_download_resumes_existing_partial_file(self):
        from download_downstream_datasets import _download

        payload = b"0123456789" * 100

        class RangeHandler(BaseHTTPRequestHandler):
            observed_range = None

            def do_GET(self):
                type(self).observed_range = self.headers.get("Range")
                offset = int(type(self).observed_range.removeprefix("bytes=").removesuffix("-"))
                self.send_response(206)
                self.send_header("Content-Length", str(len(payload) - offset))
                self.send_header("Content-Range", f"bytes {offset}-{len(payload) - 1}/{len(payload)}")
                self.end_headers()
                self.wfile.write(payload[offset:])

            def log_message(self, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                destination = Path(temporary_directory) / "payload.bin"
                destination.with_name("payload.bin.part").write_bytes(payload[:137])
                _download(f"http://127.0.0.1:{server.server_port}/payload.bin", destination)
                self.assertEqual(RangeHandler.observed_range, "bytes=137-")
                self.assertEqual(destination.read_bytes(), payload)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


class FanoTests(unittest.TestCase):
    def test_controlled_fano_uses_disjoint_tuning_and_evaluation_parameters(self):
        from downstream_protocol import make_controlled_fano_bundle

        tuning = make_controlled_fano_bundle(101, FANO_CONFIG)
        evaluation = make_controlled_fano_bundle(233, FANO_CONFIG)
        self.assertEqual(tuning.y_train.shape[1], 2)
        self.assertNotEqual(tuning.metadata["parameter_key"], evaluation.metadata["parameter_key"])
        self.assertTrue(
            set(tuning.metadata["train_point_ids"]).isdisjoint(
                tuning.metadata["validation_point_ids"]
            )
        )
        self.assertTrue(
            set(tuning.metadata["train_point_ids"]).isdisjoint(
                tuning.metadata["test_point_ids"]
            )
        )

    def test_controlled_fano_is_deterministic_for_one_seed(self):
        from downstream_protocol import make_controlled_fano_bundle

        first = make_controlled_fano_bundle(233, FANO_CONFIG)
        second = make_controlled_fano_bundle(233, FANO_CONFIG)
        self.assertEqual(first.metadata, second.metadata)
        for name in ("x_train", "y_train", "x_validation", "y_validation", "x_test", "y_test"):
            self.assertEqual(getattr(first, name).tobytes(), getattr(second, name).tobytes())


class MetricTests(unittest.TestCase):
    def test_complex_metrics_are_zero_for_identity(self):
        import numpy as np

        from downstream_metrics import compute_complex_metrics

        frequency = np.linspace(1.0, 2.0, 101)
        response = np.column_stack([np.cos(6 * frequency), np.sin(6 * frequency)])
        metrics = compute_complex_metrics(
            response, response, frequency, float(frequency[1] - frequency[0])
        )
        self.assertAlmostEqual(metrics["complex_nrmse"], 0.0)
        self.assertAlmostEqual(metrics["phase_mae_rad"], 0.0)
        self.assertEqual(metrics["status"], "ok")

    def test_flat_prediction_reports_missed_peaks(self):
        import numpy as np

        from downstream_metrics import compute_complex_metrics

        frequency = np.linspace(0.0, 1.0, 501)
        magnitude = 1.0 / ((frequency - 0.5) ** 2 + 0.002)
        response = np.column_stack([magnitude, np.zeros_like(magnitude)])
        prediction = np.zeros_like(response)
        metrics = compute_complex_metrics(
            response, prediction, frequency, float(frequency[1] - frequency[0])
        )
        self.assertGreater(metrics["missed_peak_rate"], 0.0)
        self.assertEqual(metrics["predicted_peak_count"], 0)


class MeasuredParserTests(unittest.TestCase):
    raw_root = ROOT / "downstream_data" / "raw"

    def test_microwave_fano_parser_returns_complex_power_sweeps(self):
        from downstream_protocol import load_microwave_fano

        curves = load_microwave_fano(self.raw_root)
        self.assertGreater(len(curves), 100)
        first = curves[0]
        self.assertEqual(first.response.shape, (len(first.frequency), 2))
        self.assertEqual(len(first.frequency), 1001)
        self.assertIn("power_dbm", first.conditions)
        self.assertTrue(np.all(np.diff(first.frequency) > 0))
        self.assertTrue(np.all(np.isfinite(first.response)))

    def test_microstrip_parser_reads_raw_s12_resonance_window(self):
        from downstream_protocol import load_microstrip_resonator

        curves = load_microstrip_resonator(self.raw_root)
        self.assertEqual(len(curves), 270)
        first = curves[0]
        self.assertEqual(first.response.shape, (len(first.frequency), 2))
        self.assertGreaterEqual(len(first.frequency), 1880)
        self.assertLess(abs(float(first.frequency[0]) - 1.6e9), 5e5)
        self.assertLess(abs(float(first.frequency[-1]) - 2.4e9), 5e5)
        self.assertEqual(first.metadata["s_parameter"], "S12")
        self.assertIn("temperature_c", first.conditions)
        self.assertIn("relative_humidity_percent", first.conditions)
        self.assertTrue(np.all(np.diff(first.frequency) > 0))
        self.assertTrue(np.all(np.isfinite(first.response)))

    def test_aluminium_frf_parser_returns_all_measurement_locations(self):
        from downstream_protocol import load_aluminium_frf

        curves = load_aluminium_frf(self.raw_root)
        self.assertEqual(len(curves), 25)
        self.assertTrue(all(curve.response.shape[1] == 2 for curve in curves))
        self.assertTrue(all(np.all(np.isfinite(curve.response)) for curve in curves))
        self.assertTrue(all(curve.conditions["point"] >= 1 for curve in curves))
        self.assertTrue(all(curve.metadata["frf_estimator"] == "H1" for curve in curves))

    def test_measured_bundle_has_no_frequency_index_leakage(self):
        from downstream_protocol import make_measured_bundle

        config = {"n_train": 128, "n_validation": 128, "test_fraction": 0.5}
        bundle = make_measured_bundle(
            "microwave_fano", 233, config, self.raw_root, "within_curve"
        )
        train = set(bundle.metadata["train_point_ids"])
        validation = set(bundle.metadata["validation_point_ids"])
        test = set(bundle.metadata["test_point_ids"])
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertEqual(bundle.y_train.shape[1], 2)
        self.assertGreater(len(bundle.x_test), len(bundle.x_train))

    def test_measured_bundle_preserves_high_frequency_resolution(self):
        from downstream_protocol import make_measured_bundle

        config = {"n_train": 128, "n_validation": 128}
        bundle = make_measured_bundle(
            "microwave_fano", 389, config, self.raw_root, "within_curve"
        )
        self.assertEqual(bundle.x_train.dtype, np.float64)
        self.assertEqual(len(np.unique(bundle.x_train[:, 0])), len(bundle.x_train))
        self.assertEqual(len(np.unique(bundle.x_validation[:, 0])), len(bundle.x_validation))
        self.assertEqual(len(np.unique(bundle.x_test[:, 0])), len(bundle.x_test))

    def test_measured_source_audit_reports_frozen_inventories(self):
        from downstream_protocol import audit_measured_sources

        audit = audit_measured_sources(self.raw_root)
        self.assertEqual(audit["tasks"]["microwave_fano"]["curve_count"], 432)
        self.assertEqual(audit["tasks"]["microstrip_resonator"]["curve_count"], 270)
        self.assertEqual(audit["tasks"]["aluminium_frf"]["curve_count"], 25)
        self.assertEqual(
            audit["tasks"]["microstrip_resonator"]["point_count_distribution"],
            {"1882": 135, "2561": 135},
        )
        self.assertEqual(len(audit["source_manifest_sha256"]), 64)

    def _group_config(self, task):
        dimensions = {
            "microwave_fano_group_transfer": 2,
            "microstrip_resonator_group_transfer": 3,
            "aluminium_frf_group_transfer": 3,
        }
        return {
            "base_task": task.removesuffix("_group_transfer"),
            "input_dim": dimensions[task],
            "train_curve_count": 12 if task != "aluminium_frf_group_transfer" else 20,
            "validation_curve_count": 4,
            "train_frequency_count": 64,
            "validation_frequency_count": 128,
        }

    def test_group_transfer_holds_out_complete_curves(self):
        from downstream_protocol import make_group_transfer_bundle

        for task in (
            "microwave_fano_group_transfer",
            "microstrip_resonator_group_transfer",
            "aluminium_frf_group_transfer",
        ):
            with self.subTest(task=task):
                config = self._group_config(task)
                bundle = make_group_transfer_bundle(task, 233, config, self.raw_root)
                train = set(bundle.metadata["train_curve_ids"])
                validation = set(bundle.metadata["validation_curve_ids"])
                test = set(bundle.metadata["test_curve_ids"])
                self.assertFalse(train & validation or train & test or validation & test)
                self.assertEqual(len(test), 1)
                self.assertEqual(bundle.x_train.shape[1], config["input_dim"])
                self.assertEqual(bundle.x_test.shape[0], bundle.y_test.shape[0])
                self.assertEqual(bundle.metadata["split_mode"], "group_transfer")

    def test_group_transfer_tuning_and_evaluation_targets_are_disjoint(self):
        from downstream_protocol import make_group_transfer_bundle

        tuning_seeds = [101, 211, 307]
        evaluation_seeds = [233, 277, 331, 389, 443]
        for task in (
            "microwave_fano_group_transfer",
            "microstrip_resonator_group_transfer",
            "aluminium_frf_group_transfer",
        ):
            with self.subTest(task=task):
                config = self._group_config(task)
                tuning = {
                    make_group_transfer_bundle(task, seed, config, self.raw_root)
                    .metadata["test_curve_ids"][0]
                    for seed in tuning_seeds
                }
                evaluation = {
                    make_group_transfer_bundle(task, seed, config, self.raw_root)
                    .metadata["test_curve_ids"][0]
                    for seed in evaluation_seeds
                }
                self.assertEqual(len(tuning), len(tuning_seeds))
                self.assertEqual(len(evaluation), len(evaluation_seeds))
                self.assertTrue(tuning.isdisjoint(evaluation))

    def test_aluminium_group_transfer_uses_released_point_coordinates(self):
        from downstream_protocol import aluminium_point_coordinates

        coordinates = aluminium_point_coordinates()
        self.assertEqual(len(coordinates), 25)
        self.assertEqual(coordinates[1], (0.0, 0.0))
        self.assertEqual(coordinates[5], (0.26, 0.0))
        self.assertEqual(coordinates[21], (0.0, 0.246))
        self.assertEqual(coordinates[25], (0.26, 0.246))


class ConventionalTests(unittest.TestCase):
    raw_root = ROOT / "downstream_data" / "raw"

    def test_exploration_bundle_hides_test_labels(self):
        from downstream_protocol import make_conventional_bundle

        bundle = make_conventional_bundle(
            "airfoil", 101, self.raw_root, include_test=False
        )
        self.assertEqual(bundle.metadata["test_access"], "sealed")
        self.assertEqual(bundle.x_test.shape[0], 0)
        self.assertEqual(bundle.y_test.shape[0], 0)
        self.assertIn("sealed_test_row_ids_sha256", bundle.metadata)

    def test_credit_exploration_hides_test_class_balance(self):
        from downstream_protocol import make_conventional_bundle

        bundle = make_conventional_bundle(
            "credit", 101, self.raw_root, include_test=False
        )
        self.assertNotIn("test_positive_rate", bundle.metadata["class_balance"])

    def test_credit_split_is_stratified(self):
        from downstream_protocol import make_conventional_bundle

        bundle = make_conventional_bundle(
            "credit", 101, self.raw_root, include_test=True
        )
        rates = [float(y.mean()) for y in (
            bundle.y_train, bundle.y_validation, bundle.y_test
        )]
        self.assertLess(max(rates) - min(rates), 0.02)
        self.assertEqual(bundle.metadata["problem_type"], "classification")

    def test_regression_candidate_source_row_counts(self):
        from downstream_protocol import make_conventional_bundle

        expected = {"energy": 768, "airfoil": 1503, "appliances": 19735}
        for task, row_count in expected.items():
            with self.subTest(task=task):
                bundle = make_conventional_bundle(
                    task, 211, self.raw_root, include_test=True
                )
                self.assertEqual(bundle.metadata["source_row_count"], row_count)
                self.assertEqual(bundle.metadata["problem_type"], "regression")


if __name__ == "__main__":
    unittest.main()
