import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ai4science_fano import fit_complex_fano_curve, magnitude_fano_response
from ai4science_fano_a2 import (
    completed_job_keys,
    compare_fano_parameters,
    complex_nrmse,
    enumerate_a2_jobs,
    job_key,
    make_sparse_curve_split,
    order_a2_jobs,
    peak_window_complex_nrmse,
)
from ai4science_fano_a2_analysis import build_enhanced_summary
from downstream_protocol import MeasuredCurve
from run_ai4science_fano_a2_pilot import canonical_family_name, read_jsonl_records, select_pending_jobs


def _synthetic_response(frequency, f0, gamma, q):
    magnitude = magnitude_fano_response(
        frequency,
        f0_hz=f0,
        gamma_hz=gamma,
        q=q,
        background=0.02,
        background_slope_per_hz=0.0,
        amplitude=0.01,
    )
    phase = 0.25 * np.arctan((frequency - f0) / gamma)
    return np.column_stack((magnitude * np.cos(phase), magnitude * np.sin(phase)))


class FanoA2UtilityTests(unittest.TestCase):
    def test_balanced_job_order_interleaves_families_before_repeating_init_seeds(self):
        jobs = enumerate_a2_jobs(
            curve_ids=["curve:b", "curve:a"],
            families=["MLP", "CFNN"],
            observation_budgets=[32, 16],
            split_seeds=[773, 719],
            init_seeds=[12007, 11003],
        )

        ordered = order_a2_jobs(jobs, order="balanced")

        self.assertEqual(len(ordered), len(jobs))
        self.assertEqual({job_key(job) for job in ordered}, {job_key(job) for job in jobs})
        self.assertEqual(
            [(job["observation_budget"], job["split_seed"], job["init_seed"], job["curve_id"], job["family"])
             for job in ordered[:4]],
            [
                (16, 719, 11003, "curve:a", "CFNN"),
                (16, 719, 11003, "curve:a", "MLP"),
                (16, 719, 11003, "curve:b", "CFNN"),
                (16, 719, 11003, "curve:b", "MLP"),
            ],
        )

    def test_coverage_job_order_interleaves_observation_budgets_early(self):
        jobs = enumerate_a2_jobs(
            curve_ids=["curve:a"],
            families=["CFNN", "MLP"],
            observation_budgets=[16, 32, 64, 128, 256],
            split_seeds=[719, 773],
            init_seeds=[11003, 12007],
        )

        ordered = order_a2_jobs(jobs, order="coverage")

        self.assertEqual(len(ordered), len(jobs))
        self.assertEqual({job_key(job) for job in ordered}, {job_key(job) for job in jobs})
        self.assertEqual(
            [(job["family"], job["observation_budget"], job["split_seed"], job["init_seed"])
             for job in ordered[:5]],
            [
                ("CFNN", 16, 719, 11003),
                ("CFNN", 32, 719, 11003),
                ("CFNN", 64, 719, 11003),
                ("CFNN", 128, 719, 11003),
                ("CFNN", 256, 719, 11003),
            ],
        )
        self.assertEqual(ordered[5]["family"], "MLP")

    def test_science_priority_job_order_prefers_main_budgets_and_rotates_splits(self):
        jobs = enumerate_a2_jobs(
            curve_ids=["curve:a"],
            families=["CFNN", "MLP"],
            observation_budgets=[16, 32, 64, 128, 256],
            split_seeds=[829, 719, 773],
            init_seeds=[11003],
        )

        ordered = order_a2_jobs(jobs, order="science_priority")

        self.assertEqual(len(ordered), len(jobs))
        self.assertEqual({job_key(job) for job in ordered}, {job_key(job) for job in jobs})
        self.assertEqual(
            [(job["observation_budget"], job["family"], job["split_seed"])
             for job in ordered[:6]],
            [
                (64, "CFNN", 719),
                (64, "CFNN", 773),
                (64, "CFNN", 829),
                (64, "MLP", 719),
                (64, "MLP", 773),
                (64, "MLP", 829),
            ],
        )
        self.assertEqual([job["observation_budget"] for job in ordered[6:12]], [128] * 6)

    def test_enhanced_summary_reports_quantiles_failures_and_pairwise_wins(self):
        records = [
            {
                "curve_id": "curve:1",
                "family": "CFNN",
                "observation_budget": 64,
                "split_seed": 719,
                "init_seed": 11003,
                "parameter_recovery_status": "ok",
                "global_complex_nrmse": 0.10,
                "f0_abs_error_hz": 10.0,
            },
            {
                "curve_id": "curve:2",
                "family": "CFNN",
                "observation_budget": 64,
                "split_seed": 719,
                "init_seed": 11003,
                "parameter_recovery_status": "ok",
                "global_complex_nrmse": 0.20,
                "f0_abs_error_hz": 20.0,
            },
            {
                "curve_id": "curve:1",
                "family": "MLP",
                "observation_budget": 64,
                "split_seed": 719,
                "init_seed": 11003,
                "parameter_recovery_status": "ok",
                "global_complex_nrmse": 0.20,
                "f0_abs_error_hz": 15.0,
            },
            {
                "curve_id": "curve:2",
                "family": "MLP",
                "observation_budget": 64,
                "split_seed": 719,
                "init_seed": 11003,
                "parameter_recovery_status": "prediction_unidentifiable",
                "global_complex_nrmse": 0.30,
                "f0_abs_error_hz": None,
            },
        ]

        summary = build_enhanced_summary(records, baseline_family="CFNN")

        cfnn_global = next(
            item for item in summary["family_budget_metrics"]
            if item["family"] == "CFNN" and item["metric"] == "global_complex_nrmse"
        )
        self.assertEqual(cfnn_global["valid_count"], 2)
        self.assertAlmostEqual(cfnn_global["median"], 0.15)
        self.assertAlmostEqual(cfnn_global["q25"], 0.125)
        self.assertAlmostEqual(cfnn_global["q75"], 0.175)

        mlp_status = next(
            item for item in summary["family_budget_status"]
            if item["family"] == "MLP"
        )
        self.assertEqual(mlp_status["record_count"], 2)
        self.assertEqual(mlp_status["parameter_recovery_ok_count"], 1)
        self.assertAlmostEqual(mlp_status["parameter_recovery_failure_rate"], 0.5)

        pairwise_global = next(
            item for item in summary["pairwise_baseline_comparisons"]
            if item["comparison_family"] == "MLP" and item["metric"] == "global_complex_nrmse"
        )
        self.assertEqual(pairwise_global["valid_pair_count"], 2)
        self.assertEqual(pairwise_global["baseline_win_count"], 2)
        self.assertAlmostEqual(pairwise_global["baseline_win_rate"], 1.0)
        self.assertAlmostEqual(pairwise_global["median_paired_delta"], -0.10)

        pairwise_f0 = next(
            item for item in summary["pairwise_baseline_comparisons"]
            if item["comparison_family"] == "MLP" and item["metric"] == "f0_abs_error_hz"
        )
        self.assertEqual(pairwise_f0["valid_pair_count"], 1)
        self.assertEqual(pairwise_f0["baseline_win_count"], 1)

    def test_preregistered_success_audit_requires_peak_and_f0_q_q_improvements_with_global_guardrail(self):
        from ai4science_fano_a2_analysis import summarize_preregistered_a2_success

        def row(budget, comparison, metric, effect, win_rate=0.7):
            return {
                "baseline_family": "CFNN",
                "comparison_family": comparison,
                "observation_budget": budget,
                "metric": metric,
                "valid_pair_count": 20,
                "baseline_win_rate": win_rate,
                "median_relative_delta": -effect,
            }

        rows = []
        for metric in (
            "global_complex_nrmse",
            "peak_window_complex_nrmse",
            "f0_abs_error_hz",
            "quality_factor_relative_error",
            "q_abs_error",
        ):
            rows.append(row(64, "MLP", metric, 0.10))
            rows.append(row(64, "Local nested CF control", metric, 0.08))
        rows.extend([
            row(32, "Local nested CF control", "global_complex_nrmse", -0.05, 0.4),
            row(32, "Local nested CF control", "peak_window_complex_nrmse", -0.07, 0.4),
            row(32, "Local nested CF control", "f0_abs_error_hz", 0.20),
            row(32, "Local nested CF control", "quality_factor_relative_error", 0.20),
            row(32, "Local nested CF control", "q_abs_error", -0.01, 0.45),
        ])

        audit = summarize_preregistered_a2_success({
            "pairwise_baseline_comparisons": rows,
        })
        by_key = {
            (row["observation_budget"], row["comparison_family"]): row
            for row in audit["budget_comparison_rows"]
        }

        self.assertEqual(
            by_key[(64, "MLP")]["success_status"],
            "passes_preregistered_success",
        )
        self.assertEqual(
            by_key[(64, "Local nested CF control")]["success_status"],
            "passes_preregistered_success",
        )
        self.assertEqual(
            by_key[(32, "Local nested CF control")]["success_status"],
            "fails_preregistered_success",
        )
        self.assertFalse(by_key[(32, "Local nested CF control")]["global_guardrail_pass"])
        self.assertIn("q_abs_error", by_key[(32, "Local nested CF control")]["failed_required_metrics"])
        self.assertEqual(audit["passed_budget_count"], 1)

    def test_pilot_runner_accepts_selection_slug_family_names(self):
        self.assertEqual(canonical_family_name("fourier_feature_mlp"), "Fourier-feature MLP")
        self.assertEqual(canonical_family_name("gaussian_rbf"), "Gaussian RBF")
        self.assertEqual(canonical_family_name("local_nested_cf_control"), "Local nested CF control")
        self.assertEqual(canonical_family_name("rational_activation_nn"), "Rational activation NN")
        self.assertEqual(canonical_family_name("CFNN"), "CFNN")

    def test_pilot_runner_reads_existing_records_and_limits_pending_jobs(self):
        jobs = enumerate_a2_jobs(
            curve_ids=["curve:a"],
            families=["CFNN", "MLP"],
            observation_budgets=[32, 128],
            split_seeds=[719, 773],
            init_seeds=[11003],
        )
        with TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw_records.jsonl"
            raw_path.write_text(
                '{"curve_id": "curve:a", "family": "CFNN", '
                '"observation_budget": 32, "split_seed": 719, "init_seed": 11003}\n'
            )

            existing = read_jsonl_records(raw_path)
            pending = select_pending_jobs(jobs, existing, max_records=2)

        self.assertEqual(len(existing), 1)
        self.assertEqual(len(pending), 2)
        self.assertNotIn(job_key(jobs[0]), {job_key(job) for job in pending})

    def test_enumerate_a2_jobs_includes_split_seed_in_resume_identity(self):
        jobs = enumerate_a2_jobs(
            curve_ids=["curve:b", "curve:a"],
            families=["MLP", "CFNN"],
            observation_budgets=[128, 32],
            split_seeds=[773, 719],
            init_seeds=[12007, 11003],
        )

        self.assertEqual(len(jobs), 32)
        self.assertEqual(jobs[0], {
            "curve_id": "curve:a",
            "family": "CFNN",
            "observation_budget": 32,
            "split_seed": 719,
            "init_seed": 11003,
        })
        self.assertEqual(jobs[-1], {
            "curve_id": "curve:b",
            "family": "MLP",
            "observation_budget": 128,
            "split_seed": 773,
            "init_seed": 12007,
        })
        finished = completed_job_keys([
            {
                "curve_id": "curve:a",
                "family": "CFNN",
                "observation_budget": 32,
                "split_seed": 719,
                "init_seed": 11003,
            },
            {
                "curve_id": "curve:a",
                "family": "CFNN",
                "observation_budget": 32,
                "split_seed": 773,
                "init_seed": 11003,
            },
            {"curve_id": "curve:a", "family": "CFNN"},
        ])
        self.assertEqual(finished, {job_key(jobs[0]), job_key(jobs[2])})

    def test_complex_nrmse_is_zero_for_identical_response_and_positive_for_shift(self):
        truth = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        identical = truth.copy()
        shifted = truth + 0.1

        self.assertEqual(complex_nrmse(truth, identical), 0.0)
        self.assertGreater(complex_nrmse(truth, shifted), 0.0)

    def test_peak_window_complex_nrmse_uses_reference_resonance_window(self):
        frequency = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        truth = np.ones((5, 2), dtype=np.float64)
        prediction = truth.copy()
        prediction[0] += 10.0
        prediction[2] += 0.2

        score = peak_window_complex_nrmse(
            frequency,
            truth,
            prediction,
            f0_hz=2.0,
            linewidth_hz=0.6,
            window_linewidths=1.0,
        )

        self.assertLess(score, complex_nrmse(truth, prediction))
        self.assertAlmostEqual(score, complex_nrmse(truth[2:3], prediction[2:3]))

    def test_sparse_curve_split_uses_requested_budget_and_disjoint_validation(self):
        frequency = np.linspace(1.0, 2.0, 101)
        response = np.column_stack((np.sin(frequency), np.cos(frequency)))
        curve = MeasuredCurve(
            curve_id="curve:001",
            frequency=frequency,
            response=response.astype(np.float32),
            conditions={"kind": "synthetic"},
            source_files=("synthetic",),
            metadata={"frequency_unit": "Hz"},
        )

        bundle = make_sparse_curve_split(
            curve,
            observation_count=16,
            validation_count=12,
            split_seed=719,
        )

        self.assertEqual(bundle.x_train.shape, (16, 1))
        self.assertEqual(bundle.x_validation.shape, (12, 1))
        self.assertEqual(bundle.x_test.shape, (101, 1))
        train_ids = set(bundle.metadata["train_point_ids"])
        validation_ids = set(bundle.metadata["validation_point_ids"])
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(bundle.metadata["observation_budget"], 16)
        self.assertEqual(bundle.metadata["test_target"], "complete_measured_curve")

    def test_compare_fano_parameters_reports_response_and_event_errors(self):
        frequency = np.linspace(4.995e9, 4.998e9, 1001)
        reference_response = _synthetic_response(frequency, 4.99645e9, 1.55e5, -1.25)
        prediction_response = _synthetic_response(frequency, 4.99650e9, 1.65e5, -1.10)
        reference = fit_complex_fano_curve(
            frequency,
            reference_response,
            curve_id="synthetic:fano",
            residual_threshold=1e-3,
        )
        predicted = fit_complex_fano_curve(
            frequency,
            prediction_response,
            curve_id="synthetic:fano",
            residual_threshold=1e-3,
        )

        errors = compare_fano_parameters(reference, predicted)

        self.assertEqual(errors["parameter_recovery_status"], "ok")
        self.assertGreater(errors["f0_abs_error_hz"], 0.0)
        self.assertLess(errors["f0_abs_error_hz"], 7.5e4)
        self.assertGreater(errors["quality_factor_relative_error"], 0.0)
        self.assertAlmostEqual(errors["q_abs_error"], 0.15, delta=0.03)
        self.assertIn("magnitude_peak_abs_error_hz", errors)
        self.assertIn("phase_transition_abs_error_hz", errors)

    def test_compare_fano_parameters_preserves_prediction_failure(self):
        frequency = np.linspace(1.0, 2.0, 101)
        reference_response = _synthetic_response(frequency, 1.45, 0.05, 1.0)
        reference = fit_complex_fano_curve(
            frequency,
            reference_response,
            curve_id="synthetic:fano",
            residual_threshold=1e-3,
        )
        failed = fit_complex_fano_curve(
            frequency,
            np.column_stack((np.ones_like(frequency), np.zeros_like(frequency))),
            curve_id="synthetic:fano",
            residual_threshold=1e-3,
        )

        errors = compare_fano_parameters(reference, failed)

        self.assertEqual(errors["parameter_recovery_status"], "prediction_unidentifiable")
        self.assertEqual(errors["prediction_failure_reason"], "insufficient_response_structure")
        self.assertIsNone(errors["f0_abs_error_hz"])


def test_fano_event_focus_consolidation_passes_narrow_event_gate_and_blocks_q():
    from ai4science_fano_event_focus import fano_event_focus_consolidation

    def pair(budget, comparison, metric, effect, win_rate=0.62, pairs=30):
        return {
            "baseline_family": "CFNN",
            "comparison_family": comparison,
            "observation_budget": budget,
            "metric": metric,
            "valid_pair_count": pairs,
            "baseline_win_count": int(round(win_rate * pairs)),
            "baseline_win_rate": win_rate,
            "median_relative_delta": -effect,
        }

    rows = []
    for comparison in ("MLP", "Local nested CF control"):
        rows.extend([
            pair(128, comparison, "global_complex_nrmse", 0.01, 0.55),
            pair(128, comparison, "peak_window_complex_nrmse", 0.08, 0.70),
            pair(128, comparison, "f0_abs_error_hz", 0.20, 0.72),
            pair(128, comparison, "quality_factor_relative_error", 0.05, 0.58),
            pair(128, comparison, "q_abs_error", -0.03, 0.42),
        ])
    summary = {
        "task": "microwave_fano_a2_enhanced_analysis",
        "record_count": 90,
        "baseline_family": "CFNN",
        "curve_ids": ["ordinary:1", "undercoupled:r8:p03"],
        "observation_budgets": [128],
        "pairwise_baseline_comparisons": rows,
    }

    result = fano_event_focus_consolidation(summary, highrisk_curve_ids=["undercoupled:r8:p03"])

    assert result["analysis_type"] == "microwave_fano_event_focused_consolidation"
    assert result["claim_gate"]["claim_status"] == "passes_event_focused_gate_q_blocks_full_inversion"
    assert result["claim_gate"]["full_inversion_status"] == "blocked_by_q_abs_error"
    assert result["claim_gate"]["event_metrics"] == ["peak_window_complex_nrmse", "f0_abs_error_hz"]
    assert result["claim_gate"]["highrisk_curve_ids"] == ["undercoupled:r8:p03"]


if __name__ == "__main__":
    unittest.main()
