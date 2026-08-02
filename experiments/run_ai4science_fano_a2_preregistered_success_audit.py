#!/usr/bin/env python3
"""Audit Microwave Fano A2 against the preregistered success criterion."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_fano_a2_analysis import summarize_preregistered_a2_success


ROOT = Path(__file__).resolve().parent
DEFAULT_A2_SUMMARY = ROOT / "ai4science_results" / "microwave_fano_a2_production" / "enhanced_summary.json"
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "microwave_fano_a2_preregistered_success_audit"


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
        "# Microwave Fano A2 Preregistered Success Audit",
        "",
        "This audit applies the A2 success rule: CFNN must improve peak-window response recovery and f0/Q/q parameter recovery while passing the global complex NRMSE guardrail.",
        "",
        f"Overall status: `{summary.get('overall_status')}`",
        f"Passed budgets: `{','.join(str(value) for value in summary.get('passed_budgets', []))}`",
        "",
        "| budget | baseline | status | global guardrail | peak | f0 | Q | q | failed metrics |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("budget_comparison_rows", []):
        effects = row.get("metric_effects", {})
        lines.append(
            "| "
            + " | ".join([
                str(row.get("observation_budget")),
                str(row.get("comparison_family")),
                str(row.get("success_status")),
                _fmt(row.get("global_guardrail_effect")),
                _fmt(effects.get("peak_window_complex_nrmse")),
                _fmt(effects.get("f0_abs_error_hz")),
                _fmt(effects.get("quality_factor_relative_error")),
                _fmt(effects.get("q_abs_error")),
                ",".join(row.get("failed_required_metrics", []) or row.get("missing_required_metrics", [])),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A budget passes only when both MLP and Local nested CF control comparisons pass. A failure at 32 observations should be treated as a sparse-budget boundary, not as evidence against the 64/128 point Fano success region.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2-summary", type=Path, default=DEFAULT_A2_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    a2_summary = json.loads(args.a2_summary.read_text())
    summary = summarize_preregistered_a2_success(a2_summary)
    output = {
        **summary,
        "source_a2_summary": str(args.a2_summary),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(output) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "overall_status": output["overall_status"],
        "passed_budgets": output["passed_budgets"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
