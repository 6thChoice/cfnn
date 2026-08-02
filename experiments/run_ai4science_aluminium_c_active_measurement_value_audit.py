#!/usr/bin/env python3
"""Audit whether Aluminium C active value comes from measurement strategy or CFNN."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_aluminium_c import audit_aluminium_active_measurement_value


ROOT = Path(__file__).resolve().parent
DEFAULT_INTERPOLATION_ROLLUP = (
    ROOT / "ai4science_results" / "aluminium_c_online_half_power_multipoint_interpolation_guardrail"
    / "half_power_rollup_summary.json"
)
DEFAULT_SAME_SCHEDULE = (
    ROOT / "ai4science_results" / "aluminium_c_online_half_power_same_schedule_model_pilot"
    / "analysis_summary.json"
)
DEFAULT_CANDIDATE_DIAGNOSTICS = (
    ROOT / "ai4science_results" / "aluminium_c_observed_peak_half_power_multipoint_guardrail"
    / "candidate_diagnostics_summary.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "aluminium_c_active_measurement_value_audit"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def build_report(summary: dict) -> str:
    lines = [
        "# Aluminium C Active Measurement Value Audit",
        "",
        f"Claim level: `{summary.get('claim_level')}`",
        "",
        "This audit separates active measurement value from CFNN advantage. A half-power or modal-parameter signal produced by observed-only interpolation is treated as measurement-strategy evidence unless the neural model also recovers the same event metrics under the same schedule.",
        "",
        "## Decision Rows",
        "",
        "| decision | scope | family | budget | passed modes | matched modes | failure stage | verdict |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in summary.get("decision_rows", []):
        lines.append(
            "| "
            + " | ".join([
                f"`{row.get('decision_id')}`",
                str(row.get("evidence_scope")),
                str(row.get("family")),
                str(row.get("minimum_budget", row.get("budget"))),
                _fmt(row.get("median_passed_modes", row.get("median_truth_aligned_half_power_passed_mode_count"))),
                _fmt(row.get("median_matched_modes")),
                str(row.get("dominant_failure_stage")),
                str(row.get("verdict")),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "If the observed-only interpolation path reaches useful half-power recovery while CFNN remains at zero matched or passed modes, Aluminium C should be written as an active measurement and event-detection boundary task, not as a CFNN advantage task.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpolation-rollup", type=Path, default=DEFAULT_INTERPOLATION_ROLLUP)
    parser.add_argument("--same-schedule-summary", type=Path, default=DEFAULT_SAME_SCHEDULE)
    parser.add_argument("--candidate-diagnostics", type=Path, default=DEFAULT_CANDIDATE_DIAGNOSTICS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    args = parse_args()
    summary = audit_aluminium_active_measurement_value(
        _load(args.interpolation_rollup),
        _load(args.same_schedule_summary),
        _load(args.candidate_diagnostics),
    )
    output = {
        **summary,
        "source_interpolation_rollup": str(args.interpolation_rollup),
        "source_same_schedule_summary": str(args.same_schedule_summary),
        "source_candidate_diagnostics": str(args.candidate_diagnostics),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(output) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "claim_level": output["claim_level"],
        "decision_count": len(output["decision_rows"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
