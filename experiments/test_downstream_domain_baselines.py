import unittest

import numpy as np

from confirmatory_protocol import DatasetBundle
from downstream_metrics import compute_complex_metrics


def _two_columns(response):
    return np.column_stack([response.real, response.imag])


def make_fano_fixture(q=1.4, width=0.03):
    frequency = np.linspace(0.0, 1.0, 1001)
    z = (frequency - 0.48) / width
    response = (0.08 - 0.04j) + (1.2 + 0.25j) * ((q + z) / (z + 1j) - 1.0)
    return frequency, _two_columns(response)


def make_two_mode_fixture():
    frequency = np.linspace(1.0e6, 2.0e6, 801)
    angular = 2.0j * np.pi * frequency
    response = (
        2.0e10 / (angular ** 2 + 2.0 * 0.015 * 2.0 * np.pi * 1.3e6 * angular + (2.0 * np.pi * 1.3e6) ** 2)
        + 1.2e10 / (angular ** 2 + 2.0 * 0.02 * 2.0 * np.pi * 1.7e6 * angular + (2.0 * np.pi * 1.7e6) ** 2)
    )
    return frequency, _two_columns(response)


class DomainBaselineTests(unittest.TestCase):
    def test_fano_reference_recovers_noise_free_response(self):
        from downstream_domain_baselines import fit_fano_reference

        frequency, response = make_fano_fixture()
        result = fit_fano_reference(frequency, response)
        self.assertEqual(result.status, "ok")
        metrics = compute_complex_metrics(
            response, result.prediction, frequency, float(frequency[1] - frequency[0])
        )
        self.assertLess(metrics["complex_nrmse"], 1e-3)

    def test_fano_reference_predicts_at_separate_evaluation_frequencies(self):
        from downstream_domain_baselines import fit_fano_reference

        frequency, response = make_fano_fixture()
        train = np.arange(0, len(frequency), 2)
        evaluate = np.arange(1, len(frequency), 2)
        result = fit_fano_reference(
            frequency[train], response[train],
            evaluation_frequency=frequency[evaluate],
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.prediction.shape, response[evaluate].shape)
        metrics = compute_complex_metrics(
            response[evaluate], result.prediction, frequency[evaluate],
            float(frequency[2] - frequency[0]),
        )
        self.assertLess(metrics["complex_nrmse"], 1e-3)

    def test_vector_fit_reports_requested_order(self):
        from downstream_domain_baselines import fit_vector_reference

        frequency, response = make_two_mode_fixture()
        result = fit_vector_reference(frequency, response, max_poles=4)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.metadata["maximum_poles"], 4)
        self.assertIn("convergence_warning_count", result.metadata)
        self.assertIn("convergence_warnings", result.metadata)
        self.assertEqual(result.prediction.shape, response.shape)
        self.assertTrue(np.all(np.isfinite(result.prediction)))

    def test_vector_fit_predicts_at_separate_evaluation_frequencies(self):
        from downstream_domain_baselines import fit_vector_reference

        frequency, response = make_two_mode_fixture()
        train = np.arange(0, len(frequency), 2)
        evaluate = np.arange(1, len(frequency), 2)
        result = fit_vector_reference(
            frequency[train], response[train], max_poles=4,
            evaluation_frequency=frequency[evaluate],
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.prediction.shape, response[evaluate].shape)
        self.assertTrue(np.all(np.isfinite(result.prediction)))

    def test_reference_runner_selects_order_on_validation_and_reports_test(self):
        from run_downstream_domain_references import evaluate_vector_reference

        frequency, response = make_two_mode_fixture()
        bundle = DatasetBundle(
            x_train=frequency[::4, None].astype(np.float32),
            y_train=response[::4].astype(np.float32),
            x_validation=frequency[1::4, None].astype(np.float32),
            y_validation=response[1::4].astype(np.float32),
            x_test=frequency[2::4, None].astype(np.float32),
            y_test=response[2::4].astype(np.float32),
            metadata={"fixture": "two_mode"},
        )
        record = evaluate_vector_reference(
            "fixture", 17, bundle, candidate_orders=(2, 4)
        )
        self.assertEqual(record["status"], "ok")
        self.assertIn(record["selected_max_poles"], (2, 4))
        self.assertEqual(len(record["validation_candidates"]), 2)
        self.assertGreaterEqual(record["test_metrics"]["complex_nrmse"], 0.0)
        self.assertEqual(record["selection_role"], "validation_only")


if __name__ == "__main__":
    unittest.main()
