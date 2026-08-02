from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_same_curve_budget_escalation_audit_finds_minimum_passing_budget():
    from ai4science_fano_budget_escalation_e import same_curve_budget_escalation_audit

    summary = {
        "confidence": 0.8,
        "ensemble_rows": [
            {
                "curve_id": "curve:hard",
                "family": "CFNN",
                "observation_budget": 64,
                "validation_calibrated_response_interval": {
                    "joint_coverage": 0.79,
                    "peak_window_joint_coverage": 0.70,
                    "dangerous_point_failure_rate": 0.11,
                    "calibration_scale": 6.0,
                },
            },
            {
                "curve_id": "curve:hard",
                "family": "CFNN",
                "observation_budget": 128,
                "validation_calibrated_response_interval": {
                    "joint_coverage": 0.83,
                    "peak_window_joint_coverage": 0.78,
                    "dangerous_point_failure_rate": 0.05,
                    "calibration_scale": 5.0,
                },
            },
            {
                "curve_id": "curve:hard",
                "family": "MLP",
                "observation_budget": 64,
                "validation_calibrated_response_interval": {
                    "joint_coverage": 0.70,
                    "peak_window_joint_coverage": 0.60,
                    "dangerous_point_failure_rate": 0.20,
                    "calibration_scale": 8.0,
                },
            },
            {
                "curve_id": "curve:hard",
                "family": "MLP",
                "observation_budget": 128,
                "validation_calibrated_response_interval": {
                    "joint_coverage": 0.75,
                    "peak_window_joint_coverage": 0.65,
                    "dangerous_point_failure_rate": 0.15,
                    "calibration_scale": 7.0,
                },
            },
        ],
    }

    audit = same_curve_budget_escalation_audit(summary, target_curve_id="curve:hard")

    assert audit["status"] == "ok"
    assert audit["target_curve_id"] == "curve:hard"
    assert audit["budget_count"] == 2
    assert audit["family_count"] == 2
    cfnn = next(row for row in audit["family_rows"] if row["family"] == "CFNN")
    mlp = next(row for row in audit["family_rows"] if row["family"] == "MLP")
    assert cfnn["minimum_passing_budget"] == 128
    assert cfnn["status"] == "passes_at_budget"
    assert mlp["minimum_passing_budget"] is None
    assert mlp["status"] == "no_passing_budget"
    assert audit["best_family_by_budget"]["family"] == "CFNN"
    assert audit["best_family_by_budget"]["minimum_passing_budget"] == 128


def test_multi_split_budget_escalation_summary_reports_pass_rates_and_min_budget_distribution():
    from ai4science_fano_budget_escalation_e import multi_split_budget_escalation_summary

    split_a = {
        "status": "ok",
        "target_curve_id": "curve:hard",
        "budgets": [32, 64],
        "family_rows": [
            {
                "family": "CFNN",
                "minimum_passing_budget": 32,
                "budget_rows": [
                    {"observation_budget": 32, "passes_reliability": True},
                    {"observation_budget": 64, "passes_reliability": False},
                ],
            },
            {
                "family": "MLP",
                "minimum_passing_budget": None,
                "budget_rows": [
                    {"observation_budget": 32, "passes_reliability": False},
                    {"observation_budget": 64, "passes_reliability": False},
                ],
            },
        ],
    }
    split_b = {
        "status": "ok",
        "target_curve_id": "curve:hard",
        "budgets": [32, 64],
        "family_rows": [
            {
                "family": "CFNN",
                "minimum_passing_budget": 64,
                "budget_rows": [
                    {"observation_budget": 32, "passes_reliability": False},
                    {"observation_budget": 64, "passes_reliability": True},
                ],
            },
            {
                "family": "MLP",
                "minimum_passing_budget": 64,
                "budget_rows": [
                    {"observation_budget": 32, "passes_reliability": False},
                    {"observation_budget": 64, "passes_reliability": True},
                ],
            },
        ],
    }

    summary = multi_split_budget_escalation_summary([
        ("split_a", split_a),
        ("split_b", split_b),
    ])

    assert summary["status"] == "ok"
    assert summary["split_count"] == 2
    cfnn = next(row for row in summary["family_rows"] if row["family"] == "CFNN")
    mlp = next(row for row in summary["family_rows"] if row["family"] == "MLP")
    assert cfnn["split_pass_rate"] == 1.0
    assert cfnn["minimum_passing_budgets"] == [32, 64]
    assert cfnn["median_minimum_passing_budget"] == 48.0
    assert mlp["split_pass_rate"] == 0.5
    assert mlp["minimum_passing_budgets"] == [None, 64]
    cfnn_budget32 = next(
        row for row in summary["budget_rows"]
        if row["family"] == "CFNN" and row["observation_budget"] == 32
    )
    assert cfnn_budget32["pass_rate"] == 0.5


def test_multi_split_budget_report_uses_actual_split_count():
    from run_ai4science_fano_budget_escalation_multisplit_e import build_report

    report = build_report(
        {
            "status": "ok",
            "split_count": 3,
            "family_rows": [],
            "budget_rows": [],
            "best_family_by_split_pass_then_median_budget": None,
        },
        source_summaries=[],
    )

    assert "only 3 split seeds" in report
    assert "only two split seeds" not in report
