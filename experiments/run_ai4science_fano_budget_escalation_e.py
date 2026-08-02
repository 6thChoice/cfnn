#!/usr/bin/env python3
"""Build a high-risk Microwave Fano same-curve budget escalation report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_fano_budget_escalation_e import same_curve_budget_escalation_audit


ROOT = Path(__file__).resolve().parent
DEFAULT_D_SUMMARY = ROOT / "ai4science_results" / "microwave_fano_d_highrisk_r8p03_budget_escalation" / "summary.json"
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "microwave_fano_e_highrisk_budget_escalation_r8p03"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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


def build_report(audit: dict, *, source_summary: Path) -> str:
    lines = [
        "# Microwave Fano High-Risk Budget Escalation Audit",
        "",
        f"Source D summary: `{source_summary}`",
        "",
        f"Target curve: `{audit.get('target_curve_id')}`",
        "",
        "This audit asks whether extra sparse observations repair a descriptor-flagged high-risk Fano curve under same-curve calibration. It is a reliability-boundary audit, not a CFNN advantage claim.",
        "",
    ]
    if audit.get("status") != "ok":
        lines.append(f"Status: `{audit.get('status')}`")
        return "\n".join(lines)

    lines.extend([
        "## Family Summary",
        "",
        "| family | status | min passing budget | best joint | lowest danger | budgets |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in audit.get("family_rows", []):
        lines.append(
            "| "
            + " | ".join([
                f"`{row.get('family')}`",
                f"`{row.get('status')}`",
                str(row.get("minimum_passing_budget")) if row.get("minimum_passing_budget") is not None else "NA",
                _fmt(row.get("best_joint_coverage")),
                _fmt(row.get("lowest_dangerous_point_failure_rate")),
                ", ".join(str(value) for value in row.get("budgets", [])),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Budget Rows",
        "",
        "| family | budget | joint | peak joint | danger | scale | pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for family in audit.get("family_rows", []):
        for row in family.get("budget_rows", []):
            lines.append(
                "| "
                + " | ".join([
                    f"`{row.get('family')}`",
                    str(row.get("observation_budget")),
                    _fmt(row.get("same_curve_joint_coverage")),
                    _fmt(row.get("same_curve_peak_window_joint_coverage")),
                    _fmt(row.get("same_curve_dangerous_point_failure_rate")),
                    _fmt(row.get("same_curve_calibration_scale")),
                    "`yes`" if row.get("passes_reliability") else "`no`",
                ])
                + " |"
            )

    best = audit.get("best_family_by_budget")
    lines.extend(["", "## Interpretation", ""])
    if best:
        lines.append(
            f"The earliest passing family is `{best.get('family')}` at budget `{best.get('minimum_passing_budget')}` under the joint coverage >= {_fmt(audit.get('confidence'))} and dangerous rate <= {_fmt(audit.get('danger_threshold'))} rule."
        )
    else:
        lines.append(
            "No family reaches the reliability rule within the tested budgets. The high-risk curve should remain outside automatic scientific interpretation until more measurements or stronger physical priors are added."
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-summary", type=Path, default=DEFAULT_D_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-curve-id", default="undercoupled:r8:p03")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--danger-threshold", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d_summary = json.loads(Path(args.d_summary).read_text())
    audit = same_curve_budget_escalation_audit(
        d_summary,
        target_curve_id=str(args.target_curve_id),
        confidence=args.confidence,
        danger_threshold=float(args.danger_threshold),
    )
    output = {
        "analysis_type": "microwave_fano_e_highrisk_budget_escalation",
        "source_d_summary": str(args.d_summary),
        "audit": audit,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(audit, source_summary=args.d_summary) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "status": audit.get("status"),
        "target_curve_id": audit.get("target_curve_id"),
        "best_family_by_budget": audit.get("best_family_by_budget"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
