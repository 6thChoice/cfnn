import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from resonance_windows import WindowSpec, complex_window_metrics, derive_window_spec


RULES = {
    "smoothing": {"method": "savgol", "window_length": 5, "polyorder": 2},
    "prominence_fraction": 0.05,
    "minimum_distance_observed_spacings": 2,
    "max_features": 3,
    "minimum_half_width_observed_spacings": 2,
    "include_dips_for_tasks": ["microwave_fano", "microstrip_resonator"],
    "no_feature_status": "no_identifiable_feature",
}


def _splits(x, y):
    return x[::2], y[::2], x[1::2], y[1::2]


def _complex_columns(values):
    return np.column_stack((values.real, values.imag))


FIXTURE_ROOT = Path(__file__).parent / "test_fixtures" / "peak_sensitive"


def _load_fixture(name):
    with (FIXTURE_ROOT / f"{name}.json").open() as handle:
        fixture = json.load(handle)
    return fixture, np.asarray(fixture["x"], dtype=np.float64), np.asarray(fixture["y"], dtype=np.float64)


class WindowContractTests(unittest.TestCase):
    def test_window_derivation_has_no_test_argument(self):
        signature = inspect.signature(derive_window_spec)
        self.assertNotIn("x_test", signature.parameters)
        self.assertNotIn("y_test", signature.parameters)
        self.assertEqual(
            list(signature.parameters),
            ["task", "x_train", "y_train", "x_validation", "y_validation", "metadata", "rules"],
        )

    def test_window_spec_is_immutable_and_normalizes_intervals(self):
        spec = WindowSpec(
            "controlled_fano", "linear", ((0.6, 0.7), (0.1, 0.2)), "generator_metadata", "ok", {}
        )
        self.assertEqual(spec.intervals, ((0.1, 0.2), (0.6, 0.7)))
        with self.assertRaises((AttributeError, TypeError)):
            spec.status = "changed"

    def test_window_spec_rejects_nested_detector_mutation_and_can_be_serialized(self):
        spec = WindowSpec(
            "controlled_fano",
            "linear",
            ((0.1, 0.2),),
            "generator_metadata",
            "ok",
            {"smoothing": {"method": "savgol"}, "signals": ["magnitude"], "tags": {"frozen"}},
        )
        with self.assertRaises((AttributeError, TypeError)):
            spec.detector["smoothing"]["method"] = "none"
        with self.assertRaises((AttributeError, TypeError)):
            spec.detector["signals"].append("phase")
        with self.assertRaises((AttributeError, TypeError)):
            spec.detector["tags"].add("mutable")
        self.assertEqual(spec.detector["smoothing"]["method"], "savgol")
        self.assertEqual(spec.to_dict()["detector"]["signals"], ["magnitude"])
        json.dumps(spec.to_dict(), sort_keys=True)

    def test_peak_sensitive_fixtures_cover_expected_detector_contracts(self):
        expected = {
            "peak": {
                "task": "aluminium_frf",
                "axis_space": "linear",
                "status": "ok",
                "centers": (0.5,),
                "signals": ("magnitude",),
            },
            "dip": {
                "task": "microstrip_resonator",
                "axis_space": "linear",
                "status": "ok",
                "centers": (0.5,),
                "signals": ("inverse_magnitude",),
            },
            "frf_multimode": {
                "task": "aluminium_frf",
                "axis_space": "linear",
                "status": "ok",
                "centers": (1.5, 4.0, 7.5),
                "signals": ("magnitude",),
            },
            "eis_arc": {
                "task": "figshare_battery_eis",
                "axis_space": "log10_frequency",
                "status": "ok",
                "centers": (2.0,),
                "signals": ("negative_imaginary_impedance",),
            },
            "no_feature": {
                "task": "aluminium_frf",
                "axis_space": "linear",
                "status": "no_identifiable_feature",
                "centers": (),
            },
        }
        self.assertEqual(set(expected), {path.stem for path in FIXTURE_ROOT.glob("*.json")})
        for name, contract in expected.items():
            fixture, x, y = _load_fixture(name)
            self.assertEqual(fixture["task"], contract["task"])
            self.assertEqual(fixture["axis_space"], contract["axis_space"])
            spec = derive_window_spec(
                fixture["task"], *(_splits(x, y)), {}, RULES
            )
            self.assertEqual(spec.task, contract["task"])
            self.assertEqual(spec.axis_space, contract["axis_space"])
            self.assertEqual(spec.status, contract["status"])
            self.assertEqual(spec.source_role, "no_identifiable_feature" if not contract["centers"] else "train_validation")
            self.assertEqual(spec.detector["feature_count"], len(contract["centers"]))
            centers = tuple(0.5 * (left + right) for left, right in spec.intervals)
            self.assertEqual(len(centers), len(contract["centers"]))
            for center, expected_center in zip(centers, contract["centers"]):
                self.assertAlmostEqual(center, expected_center, places=12)
            if contract["centers"]:
                self.assertEqual(tuple(spec.detector["signals"]), contract["signals"])

    def test_peak_and_dip_windows_are_detected_from_observed_roles(self):
        x = np.linspace(0.0, 1.0, 201)
        peak = 6.0 * np.exp(-((x - 0.3) / 0.035) ** 2)
        dip = -0.8 * np.exp(-((x - 0.7) / 0.04) ** 2)
        y = _complex_columns(1.0 + peak + dip + 0.02j * np.sin(5 * x))
        spec = derive_window_spec(
            "microstrip_resonator", *(_splits(x, y)), {}, RULES
        )
        centers = [0.5 * (a + b) for a, b in spec.intervals]
        self.assertEqual(spec.source_role, "train_validation")
        self.assertEqual(spec.status, "ok")
        self.assertTrue(any(abs(center - 0.3) < 0.05 for center in centers))
        self.assertTrue(any(abs(center - 0.7) < 0.05 for center in centers))

    def test_frf_detection_keeps_three_deterministic_modes(self):
        x = np.linspace(0.0, 10.0, 401)
        signal = sum(
            amplitude / (1.0 + ((x - center) / width) ** 2)
            for center, width, amplitude in ((1.5, 0.08, 4.0), (4.0, 0.12, 2.5), (7.5, 0.16, 3.0))
        )
        y = _complex_columns(signal + 0.1j * signal)
        first = derive_window_spec("aluminium_frf", *(_splits(x, y)), {}, RULES)
        second = derive_window_spec("aluminium_frf", *(_splits(x, y)), {}, RULES)
        self.assertEqual(first, second)
        centers = [0.5 * (a + b) for a, b in first.intervals]
        self.assertEqual(len(centers), 3)
        for expected in (1.5, 4.0, 7.5):
            self.assertTrue(any(abs(center - expected) < 0.08 for center in centers))

    def test_eis_detection_uses_log_frequency_and_negative_imaginary_signal(self):
        frequency = np.logspace(0.0, 4.0, 161)
        log_frequency = np.log10(frequency)
        arc = 0.25 / (1.0 + ((log_frequency - 2.0) / 0.18) ** 2)
        response = 2.0 - 0.2j - 0.7j * arc
        y = _complex_columns(response)
        spec = derive_window_spec("figshare_battery_eis", *(_splits(frequency, y)), {}, RULES)
        self.assertEqual(spec.axis_space, "log10_frequency")
        self.assertEqual(spec.detector["signal"], "negative_imaginary_impedance")
        centers = [0.5 * (a + b) for a, b in spec.intervals]
        self.assertTrue(any(abs(center - 2.0) < 0.12 for center in centers))

    def test_controlled_fano_uses_generator_metadata_without_response_detection(self):
        x = np.linspace(0.0, 1.0, 33)
        y = np.zeros((len(x), 2))
        metadata = {"parameters": {"width": 0.04, "separation": 0.2}}
        spec = derive_window_spec("controlled_fano", *(_splits(x, y)), metadata, RULES)
        centers = [0.5 * (a + b) for a, b in spec.intervals]
        self.assertEqual(spec.source_role, "generator_metadata")
        self.assertTrue(any(abs(center - 0.4) < 1e-12 for center in centers))
        self.assertTrue(any(abs(center - 0.6) < 1e-12 for center in centers))

    def test_no_identifiable_feature_has_no_intervals(self):
        x = np.linspace(0.0, 1.0, 41)
        y = _complex_columns(np.ones(len(x)) + 0.0j)
        spec = derive_window_spec("aluminium_frf", *(_splits(x, y)), {}, RULES)
        self.assertEqual(spec.intervals, ())
        self.assertEqual(spec.status, "no_identifiable_feature")
        self.assertEqual(spec.source_role, "no_identifiable_feature")

    def test_overlapping_detected_windows_are_merged_and_minimum_width_is_enforced(self):
        x = np.linspace(0.0, 1.0, 101)
        signal = (
            2.0 * np.exp(-((x - 0.45) / 0.035) ** 2)
            + 1.7 * np.exp(-((x - 0.49) / 0.035) ** 2)
        )
        y = _complex_columns(signal)
        rules = {**RULES, "max_features": 2}
        spec = derive_window_spec("aluminium_frf", *(_splits(x, y)), {}, rules)
        self.assertEqual(len(spec.intervals), 1)
        spacing = float(np.median(np.diff(x)))
        left, right = spec.intervals[0]
        self.assertGreaterEqual(0.5 * (right - left), 2.0 * spacing - 1e-12)


class ComplexMetricTests(unittest.TestCase):
    def test_window_metric_uses_complex_residual_and_ignores_background_error(self):
        x = np.linspace(0.0, 1.0, 101)
        truth_complex = np.exp(1j * 2.0 * np.pi * x)
        prediction_complex = truth_complex.copy()
        prediction_complex[(x < 0.4) | (x > 0.6)] += 0.5 + 0.25j
        spec = WindowSpec("controlled_fano", "linear", ((0.4, 0.6),), "generator_metadata", "ok", {})
        metrics = complex_window_metrics(
            x, _complex_columns(truth_complex), _complex_columns(prediction_complex), spec
        )
        self.assertAlmostEqual(metrics["resonance_window_complex_nrmse"], 0.0)
        self.assertGreater(metrics["global_complex_nrmse"], 0.0)
        self.assertEqual(metrics["effective_hidden_count"], 21)
        self.assertEqual(metrics["window_status"], "ok")

    def test_log_frequency_trapezoid_weights_are_normalized(self):
        frequency = np.array([1.0, 10.0, 100.0])
        truth = _complex_columns(np.ones(3, dtype=np.complex128))
        prediction = _complex_columns(np.array([1.0 + 1.0j, 1.0, 1.0], dtype=np.complex128))
        spec = WindowSpec("figshare_battery_eis", "log10_frequency", ((0.0, 2.0),), "train_validation", "ok", {})
        metrics = complex_window_metrics(frequency, truth, prediction, spec, log_weighted=True)
        self.assertAlmostEqual(metrics["global_complex_nrmse"], 0.5, places=12)
        self.assertAlmostEqual(metrics["resonance_window_complex_nrmse"], 0.5, places=12)
        self.assertAlmostEqual(metrics["log_frequency_weight_sum"], 1.0, places=12)

    def test_log_window_metric_ignores_frequency_changes_outside_window(self):
        in_window = np.array([1.0, np.sqrt(10.0), 10.0])
        truth = _complex_columns(np.array([1.0, 2.0, 4.0], dtype=np.complex128))
        prediction = truth.copy()
        prediction[0, 0] += 1.0
        spec = WindowSpec("figshare_battery_eis", "log10_frequency", ((0.0, 1.0),), "train_validation", "ok", {})
        first = complex_window_metrics(
            np.append(in_window, np.sqrt(10.0) * 10.0), truth=np.vstack((truth, truth[-1])),
            prediction=np.vstack((prediction, truth[-1])), window_spec=spec, log_weighted=True
        )
        second = complex_window_metrics(
            np.append(in_window, 1e12), truth=np.vstack((truth, truth[-1])),
            prediction=np.vstack((prediction, truth[-1])), window_spec=spec, log_weighted=True
        )
        self.assertAlmostEqual(
            first["resonance_window_complex_nrmse"],
            second["resonance_window_complex_nrmse"],
            places=12,
        )

    def test_log_window_metric_uses_unit_weight_for_one_selected_point(self):
        frequency = np.array([1.0, 10.0, 100.0])
        truth = _complex_columns(np.ones(3, dtype=np.complex128))
        prediction = truth.copy()
        prediction[0, 0] = 2.0
        spec = WindowSpec("figshare_battery_eis", "log10_frequency", ((0.0, 0.01),), "train_validation", "ok", {})
        metrics = complex_window_metrics(frequency, truth, prediction, spec, log_weighted=True)
        self.assertEqual(metrics["effective_hidden_count"], 1)
        self.assertAlmostEqual(metrics["resonance_window_complex_nrmse"], 1.0, places=12)

    def test_feature_center_error_and_miss_false_rates_are_reported(self):
        x = np.linspace(0.0, 1.0, 201)
        truth_signal = np.exp(-((x - 0.35) / 0.02) ** 2) + 0.8 * np.exp(-((x - 0.72) / 0.025) ** 2)
        predicted_signal = np.exp(-((x - 0.355) / 0.02) ** 2) + 0.2 * np.exp(-((x - 0.9) / 0.01) ** 2)
        spec = WindowSpec(
            "aluminium_frf", "linear", ((0.30, 0.40), (0.65, 0.79)), "train_validation", "ok",
            {"signal": "magnitude", "prominence_fraction": 0.05, "max_features": 3,
             "minimum_distance_observed_spacings": 2},
        )
        metrics = complex_window_metrics(
            x, _complex_columns(truth_signal), _complex_columns(predicted_signal), spec
        )
        self.assertLess(metrics["feature_center_error"], 0.03)
        self.assertGreater(metrics["missed_feature_rate"], 0.0)
        self.assertGreater(metrics["false_feature_rate"], 0.0)
        self.assertLess(metrics["strongest_feature_frequency_error"], 0.03)

    def test_metric_validation_rejects_bad_inputs(self):
        spec = WindowSpec("aluminium_frf", "linear", ((0.2, 0.4),), "train_validation", "ok", {})
        with self.assertRaises(ValueError):
            complex_window_metrics(np.array([0.0, 0.0]), np.zeros((2, 2)), np.zeros((2, 2)), spec)
        with self.assertRaises(ValueError):
            complex_window_metrics(np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros((3, 2)), spec)
        with self.assertRaises(ValueError):
            WindowSpec("unknown", "linear", (), "train_validation", "ok", {})


if __name__ == "__main__":
    unittest.main()
