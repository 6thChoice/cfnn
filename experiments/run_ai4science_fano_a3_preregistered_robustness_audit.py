#!/usr/bin/env python3
"""Audit Microwave Fano A3 robustness against preregistered guardrails."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_fano_a3_analysis import summarize_preregistered_a3_robustness


ROOT = Path(__file__).resolve().parent
DEFAULT_A3_SUMMARIES = [
    (
        "noise",
        ROOT / "ai4science_results" / "microwave_fano_a3_noise_heatmap_triad_seed_expansion" / "summary.json",
    ),
    (
        "missing",
        ROOT / "ai4science_results" / "microwave_fano_a3_missing5_10pct_triad_seed_expansion" / "summary.json",
    ),
    (
        "drift",
        ROOT / "ai4science_results" / "microwave_fano_a3_drift5ppm_triad_seed_expansion" / "summary.json",
    ),
]
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "microwave_fano_a3_preregistered_robustness_audit"


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


def _condition_id(row: dict) -> str:
    return (
        f"{row.get('source_label')}:budget{row.get('observation_budget')}:"
        f"snr{_fmt(row.get('noise_snr_db'), 0)}:"
        f"missing{_fmt(row.get('missing_fraction'))}:"
        f"drift{_fmt(row.get('frequency_drift_ppm'))}"
    )


def build_report(summary: dict) -> str:
    lines = [
        "# Microwave Fano A3 Preregistered Robustness Audit",
        "",
        "This audit applies a condition-level guardrail: CFNN must be non-inferior on global response, peak-window response, f0, Q, q, and parameter recovery failure rate against both MLP and Local nested CF control under the same noise/missing/drift condition.",
        "",
        f"Overall status: `{summary.get('overall_status')}`",
        f"Condition count: `{summary.get('condition_count')}`",
        f"Passed condition count: `{summary.get('passed_condition_count')}`",
        f"MLP-only or partial condition count: `{summary.get('mlp_only_or_partial_condition_count')}`",
        "",
        "## Condition Gate Rows",
        "",
        "| condition | status | passed baselines |",
        "|---|---|---|",
    ]
    for row in summary.get("condition_gate_rows", []):
        lines.append(
            "| "
            + " | ".join([
                f"`{_condition_id(row)}`",
                str(row.get("condition_status")),
                ",".join(row.get("passed_comparison_families", [])),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Baseline Comparison Rows",
        "",
        "| condition | baseline | status | global | peak | f0 | Q | q | failure | failed metrics |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in summary.get("condition_comparison_rows", []):
        effects = row.get("metric_effects", {})
        lines.append(
            "| "
            + " | ".join([
                f"`{_condition_id(row)}`",
                str(row.get("comparison_family")),
                str(row.get("comparison_status")),
                _fmt(effects.get("median_global_complex_nrmse")),
                _fmt(effects.get("median_peak_window_complex_nrmse")),
                _fmt(effects.get("median_f0_abs_error_hz")),
                _fmt(effects.get("median_quality_factor_relative_error")),
                _fmt(effects.get("median_q_abs_error")),
                _fmt(effects.get("parameter_recovery_failure_rate")),
                ",".join(row.get("failed_metrics", []) or row.get("missing_metrics", [])),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A3 should be reported as a robustness guardrail, not as a broad CFNN reliability claim, unless conditions pass against Local nested CF control as well as MLP. MLP-only successes remain useful evidence that CFNN improves over an unstructured baseline, but they do not establish a stronger scientific robustness claim.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a3-summary",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        default=None,
        help="Append an A3 robustness summary as LABEL PATH. Defaults to noise/missing/drift production summaries.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [(label, Path(path)) for label, path in args.a3_summary] if args.a3_summary else DEFAULT_A3_SUMMARIES
    loaded = [(label, json.loads(path.read_text())) for label, path in paths if path.exists()]
    summary = summarize_preregistered_a3_robustness(loaded)
    output = {
        **summary,
        "source_a3_summaries": {label: str(path) for label, path in paths if path.exists()},
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(output) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "overall_status": output["overall_status"],
        "condition_count": output["condition_count"],
        "passed_condition_count": output["passed_condition_count"],
        "mlp_only_or_partial_condition_count": output["mlp_only_or_partial_condition_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
