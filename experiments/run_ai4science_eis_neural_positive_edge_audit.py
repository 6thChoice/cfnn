#!/usr/bin/env python3
"""Audit EIS neural-only positive CFNN cells against physical-prior baselines."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_eis_boundary_e import audit_eis_neural_positive_edges


ROOT = Path(__file__).resolve().parent
DEFAULT_EIS_SUMMARY = ROOT / "ai4science_results" / "eis_e_applicability_map_starter" / "summary.json"
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "eis_neural_positive_edge_audit"


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
        "# EIS Neural-Only Positive Edge Audit",
        "",
        "This audit isolates cells where CFNN beats neural baselines, then checks whether the same task remains positive once explicit physical-prior baselines are allowed.",
        "",
        "## Status Summary",
        "",
        f"Positive neural-only edge count: `{summary.get('positive_neural_edge_count')}`",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in sorted(summary.get("status_counts", {}).items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend([
        "",
        "## Positive Edge Rows",
        "",
        "| task | status | neural effect | physical-prior effect | all-baseline effect | stratum | poles | peaks | n | noise | neural baseline | physical baseline |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
    ])
    for row in summary.get("positive_edge_rows", []):
        lines.append(
            "| "
            + " | ".join([
                f"`{row.get('task_id')}`",
                str(row.get("edge_status")),
                _fmt(row.get("neural_effect")),
                _fmt(row.get("physical_prior_effect")),
                _fmt(row.get("all_baseline_effect")),
                str(row.get("descriptor_stratum")),
                str(row.get("known_pole_count")),
                str(row.get("relaxation_peak_count")),
                str(row.get("n_points")),
                _fmt(row.get("noise")),
                str(row.get("neural_best_baseline_model")),
                str(row.get("physical_best_baseline_model")),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A neural-only positive edge is not sufficient evidence for CFNN applicability on EIS. If the physical-prior or all-baseline layer is negative, the apparent neural edge is treated as eliminated by stronger scientific priors rather than as a robust CFNN structural advantage.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eis-summary", type=Path, default=DEFAULT_EIS_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eis_summary = json.loads(args.eis_summary.read_text())
    summary = audit_eis_neural_positive_edges(eis_summary.get("layered_effect_descriptor_rows", []))
    output = {
        **summary,
        "source_eis_summary": str(args.eis_summary),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(output) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "positive_neural_edge_count": output["positive_neural_edge_count"],
        "status_counts": output["status_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
