import math
import unittest

import numpy as np

from ai4science_fano_a3 import (
    a3_completed_job_keys,
    a3_job_key,
    add_complex_gaussian_noise,
    enumerate_a3_jobs,
    make_robust_sparse_curve_split,
)
from downstream_protocol import MeasuredCurve
from run_ai4science_fano_a3_robustness import select_pending_jobs


class FanoA3RobustnessTests(unittest.TestCase):
    def test_complex_noise_matches_requested_snr_and_is_reproducible(self):
        rng = np.random.default_rng(123)
        response = rng.normal(size=(20000, 2)).astype(np.float64)

        noisy_a = add_complex_gaussian_noise(response, snr_db=20.0, seed=719)
        noisy_b = add_complex_gaussian_noise(response, snr_db=20.0, seed=719)

        self.assertTrue(np.array_equal(noisy_a, noisy_b))
        signal_rms = math.sqrt(float(np.mean(np.sum(response ** 2, axis=1))))
        noise_rms = math.sqrt(float(np.mean(np.sum((noisy_a - response) ** 2, axis=1))))
        measured_snr_db = 20.0 * math.log10(signal_rms / noise_rms)
        self.assertAlmostEqual(measured_snr_db, 20.0, delta=0.35)

    def test_robust_split_perturbs_observations_only_and_preserves_clean_test_target(self):
        frequency = np.linspace(4.995e9, 4.998e9, 101)
        response = np.column_stack((
            np.sin(np.linspace(0.0, 2.0, 101)),
            np.cos(np.linspace(0.0, 2.0, 101)),
        )).astype(np.float32)
        curve = MeasuredCurve(
            curve_id="curve:a3",
            frequency=frequency,
            response=response,
            conditions={"kind": "synthetic"},
            source_files=("synthetic",),
            metadata={"frequency_unit": "Hz"},
        )

        bundle = make_robust_sparse_curve_split(
            curve,
            observation_count=20,
            validation_count=10,
            split_seed=719,
            perturbation_seed=11003,
            noise_snr_db=30.0,
            frequency_drift_ppm=5.0,
            missing_fraction=0.25,
        )

        self.assertEqual(bundle.x_train.shape, (15, 1))
        self.assertEqual(bundle.x_validation.shape, (10, 1))
        self.assertEqual(bundle.x_test.shape, (101, 1))
        np.testing.assert_allclose(bundle.x_test[:, 0], frequency)
        np.testing.assert_allclose(bundle.y_test, response)
        self.assertEqual(bundle.metadata["requested_observation_budget"], 20)
        self.assertEqual(bundle.metadata["actual_observation_count"], 15)
        self.assertEqual(bundle.metadata["noise_snr_db"], 30.0)
        self.assertEqual(bundle.metadata["frequency_drift_ppm"], 5.0)
        self.assertEqual(bundle.metadata["missing_fraction"], 0.25)
        self.assertNotEqual(
            bundle.metadata["split_strategy"],
            "within_evaluation_curve_sparse_observation_complete_curve_target",
        )

    def test_a3_job_identity_includes_robustness_conditions(self):
        jobs = enumerate_a3_jobs(
            curve_ids=["curve:a"],
            families=["CFNN"],
            observation_budgets=[64],
            split_seeds=[719],
            init_seeds=[11003],
            noise_snr_dbs=[40.0, 20.0],
            frequency_drift_ppms=[0.0],
            missing_fractions=[0.0],
            perturbation_seeds=[3001],
        )

        self.assertEqual(len(jobs), 2)
        self.assertNotEqual(a3_job_key(jobs[0]), a3_job_key(jobs[1]))
        self.assertEqual(a3_completed_job_keys([jobs[0], {"curve_id": "curve:a"}]), {a3_job_key(jobs[0])})

    def test_a3_runner_resume_keeps_distinct_noise_conditions_pending(self):
        jobs = enumerate_a3_jobs(
            curve_ids=["curve:a"],
            families=["CFNN"],
            observation_budgets=[64],
            split_seeds=[719],
            init_seeds=[11003],
            noise_snr_dbs=[40.0, 20.0],
            frequency_drift_ppms=[0.0],
            missing_fractions=[0.0],
            perturbation_seeds=[3001],
        )

        pending = select_pending_jobs(jobs, [jobs[0]], max_records=None)

        self.assertEqual([a3_job_key(job) for job in pending], [a3_job_key(jobs[1])])

    def test_preregistered_a3_robustness_requires_response_parameter_and_failure_guardrails(self):
        from ai4science_fano_a3_analysis import summarize_preregistered_a3_robustness

        a3_summary = {
            "by_family_budget_condition": [
                {
                    "family": "CFNN",
                    "observation_budget": 64,
                    "noise_snr_db": 20.0,
                    "frequency_drift_ppm": 0.0,
                    "missing_fraction": 0.0,
                    "median_global_complex_nrmse": 0.10,
                    "median_peak_window_complex_nrmse": 0.12,
                    "median_f0_abs_error_hz": 100.0,
                    "median_quality_factor_relative_error": 0.10,
                    "median_q_abs_error": 1.0,
                    "parameter_recovery_failure_rate": 0.0,
                },
                {
                    "family": "MLP",
                    "observation_budget": 64,
                    "noise_snr_db": 20.0,
                    "frequency_drift_ppm": 0.0,
                    "missing_fraction": 0.0,
                    "median_global_complex_nrmse": 0.20,
                    "median_peak_window_complex_nrmse": 0.18,
                    "median_f0_abs_error_hz": 150.0,
                    "median_quality_factor_relative_error": 0.20,
                    "median_q_abs_error": 2.0,
                    "parameter_recovery_failure_rate": 0.1,
                },
                {
                    "family": "Local nested CF control",
                    "observation_budget": 64,
                    "noise_snr_db": 20.0,
                    "frequency_drift_ppm": 0.0,
                    "missing_fraction": 0.0,
                    "median_global_complex_nrmse": 0.08,
                    "median_peak_window_complex_nrmse": 0.14,
                    "median_f0_abs_error_hz": 120.0,
                    "median_quality_factor_relative_error": 0.11,
                    "median_q_abs_error": 1.2,
                    "parameter_recovery_failure_rate": 0.0,
                },
            ]
        }

        audit = summarize_preregistered_a3_robustness([("noise", a3_summary)])
        rows = {
            row["comparison_family"]: row
            for row in audit["condition_comparison_rows"]
        }

        self.assertEqual(rows["MLP"]["comparison_status"], "passes_preregistered_robustness")
        self.assertEqual(rows["Local nested CF control"]["comparison_status"], "fails_preregistered_robustness")
        self.assertIn("median_global_complex_nrmse", rows["Local nested CF control"]["failed_metrics"])
        self.assertEqual(audit["condition_gate_rows"][0]["condition_status"], "blocked_by_baseline_guardrail")
        self.assertEqual(audit["passed_condition_count"], 0)


if __name__ == "__main__":
    unittest.main()
