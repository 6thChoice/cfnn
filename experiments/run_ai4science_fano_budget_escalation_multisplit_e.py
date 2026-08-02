#!/usr/bin/env python3
"""Aggregate high-risk Microwave Fano budget escalation audits across split seeds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_fano_budget_escalation_e import multi_split_budget_escalation_summary


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "microwave_fano_e_highrisk_budget_escalation_r8p03_multisplit"


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


def build_report(summary: dict, *, source_summaries: list[Path]) -> str:
    lines = [
        "# Microwave Fano High-Risk Budget Escalation Multi-Split Report",
        "",
        "Source summaries:",
        "",
    ]
    for path in source_summaries:
        lines.append(f"- `{path}`")
    lines.extend([
        "",
        "This report aggregates same-curve budget escalation audits across sparse split seeds. It estimates pass rates and minimum reliable budget stability for descriptor-flagged high-risk Fano curves.",
        "",
    ])
    if summary.get("status") != "ok":
        lines.append(f"Status: `{summary.get('status')}`")
        return "\n".join(lines)

    lines.extend([
        "## Family-Level Stability",
        "",
        "| family | split pass rate | min budgets | median min budget | min | max |",
        "|---|---:|---|---:|---:|---:|",
    ])
    for row in summary.get("family_rows", []):
        budgets = [
            "NA" if value is None else str(value)
            for value in row.get("minimum_passing_budgets", [])
        ]
        lines.append(
            "| "
            + " | ".join([
                f"`{row.get('family')}`",
                _fmt(row.get("split_pass_rate")),
                ", ".join(budgets),
                _fmt(row.get("median_minimum_passing_budget")),
                str(row.get("min_minimum_passing_budget")) if row.get("min_minimum_passing_budget") is not None else "NA",
                str(row.get("max_minimum_passing_budget")) if row.get("max_minimum_passing_budget") is not None else "NA",
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Budget-Level Pass Rates",
        "",
        "| family | budget | pass rate | passing splits | median joint | median danger |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary.get("budget_rows", []):
        lines.append(
            "| "
            + " | ".join([
                f"`{row.get('family')}`",
                str(row.get("observation_budget")),
                _fmt(row.get("pass_rate")),
                f"{row.get('passing_split_count')}/{row.get('split_count')}",
                _fmt(row.get("median_joint_coverage")),
                _fmt(row.get("median_dangerous_point_failure_rate")),
            ])
            + " |"
        )

    best = summary.get("best_family_by_split_pass_then_median_budget")
    lines.extend(["", "## Interpretation", ""])
    if best:
        lines.append(
            f"The most stable family by split pass rate and median minimum budget is `{best.get('family')}`: pass rate {_fmt(best.get('split_pass_rate'))}, median minimum budget {_fmt(best.get('median_minimum_passing_budget'))}."
        )
    split_count = summary.get("split_count")
    lines.append(
        f"Because only {split_count} split seeds are included, this remains a stability audit rather than a confirmatory distribution estimate."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-summary",
        type=Path,
        nargs="+",
        required=True,
        help="One or more single-split budget escalation E summary.json files.",
    )
    parser.add_argument("--split-labels", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split_labels is not None and len(args.split_labels) != len(args.audit_summary):
        raise ValueError("--split-labels must match --audit-summary length")
    labels = args.split_labels or [f"split_{index + 1}" for index in range(len(args.audit_summary))]
    split_audits = []
    for label, path in zip(labels, args.audit_summary, strict=True):
        payload = json.loads(Path(path).read_text())
        split_audits.append((str(label), payload["audit"]))

    summary = multi_split_budget_escalation_summary(split_audits)
    output = {
        "analysis_type": "microwave_fano_e_highrisk_budget_escalation_multisplit",
        "source_summaries": [str(path) for path in args.audit_summary],
        "split_labels": labels,
        "summary": summary,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(summary, source_summaries=list(args.audit_summary)) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "status": summary.get("status"),
        "split_count": summary.get("split_count"),
        "best_family": summary.get("best_family_by_split_pass_then_median_budget"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
