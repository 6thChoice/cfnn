#!/usr/bin/env python3
"""Build confirmatory Microstrip B2 guardrail-stratum summaries from existing runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_microstrip_b import (
    microstrip_b2_confirmatory_guardrail_stratum_effect_rows,
    microstrip_instance_effect_rows,
    summarize_microstrip_b2_confirmatory_guardrail_stratum_effects,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_RECORDS = (
    ROOT / "ai4science_results" / "microstrip_b2_missed_band_negative_control_seed_expansion" / "raw_records.jsonl"
)
DEFAULT_DESCRIPTOR_ROWS = (
    ROOT / "ai4science_results" / "microstrip_b2_missed_band_f0_descriptor_audit" / "f0_guardrail_descriptor_rows.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "ai4science_results" / "microstrip_b2_confirmatory_guardrail_strata"
)
METRICS = {
    "quality_factor_relative_error",
    "resonance_frequency_abs_error_hz",
    "local_phase_transition_abs_error_hz",
    "complex_nrmse",
}


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


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def build_report(summary: dict) -> str:
    lines = [
        "# Microstrip B2 Confirmatory Guardrail Strata",
        "",
        f"Primary claim gate: `{summary.get('primary_claim_gate')}`",
        "",
        "This report converts the B2 missed-band pilot into pre-registered stratum-level paired effects. A Q/linewidth gain is not sufficient for a device-digital-twin claim unless f0, local phase, and global complex NRMSE guardrails also pass.",
        "",
        "## Claim Gate Rows",
        "",
        "| stratum | role | baseline | calibration | Q effect | f0 effect | local phase effect | global effect | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("claim_gate_rows", []):
        lines.append(
            "| "
            + " | ".join([
                f"`{row.get('stratum_id')}`",
                str(row.get("stratum_role")),
                str(row.get("comparison_family")),
                str(row.get("calibration_frequency_count")),
                _fmt(row.get("quality_factor_effect")),
                _fmt(row.get("f0_effect")),
                _fmt(row.get("local_phase_effect")),
                _fmt(row.get("global_complex_nrmse_effect")),
                str(row.get("claim_gate_status")),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Metric Summary Rows",
        "",
        "| stratum | baseline | calibration | metric | pairs | win rate | median effect |",
        "|---|---|---:|---|---:|---:|---:|",
    ])
    for row in summary.get("metric_summary_rows", []):
        lines.append(
            "| "
            + " | ".join([
                f"`{row.get('stratum_id')}`",
                str(row.get("comparison_family")),
                str(row.get("calibration_frequency_count")),
                str(row.get("metric")),
                str(row.get("pair_count")),
                _fmt(row.get("cfnn_win_rate")),
                _fmt(row.get("median_cfnn_effect")),
            ])
            + " |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-records", type=Path, default=DEFAULT_RAW_RECORDS)
    parser.add_argument("--descriptor-rows", type=Path, default=DEFAULT_DESCRIPTOR_ROWS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_jsonl(args.raw_records)
    descriptor_rows = json.loads(args.descriptor_rows.read_text())
    effect_rows = microstrip_instance_effect_rows(records, metrics=METRICS)
    stratum_effect_rows = microstrip_b2_confirmatory_guardrail_stratum_effect_rows(
        effect_rows,
        descriptor_rows,
        metrics=METRICS,
    )
    summary = summarize_microstrip_b2_confirmatory_guardrail_stratum_effects(stratum_effect_rows)
    output = {
        **summary,
        "source_raw_records": str(args.raw_records),
        "source_descriptor_rows": str(args.descriptor_rows),
        "effect_row_count": len(effect_rows),
        "stratum_effect_row_count": len(stratum_effect_rows),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "stratum_effect_rows.json").write_text(
        json.dumps(_json_safe(stratum_effect_rows), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(output) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "effect_row_count": len(effect_rows),
        "stratum_effect_row_count": len(stratum_effect_rows),
        "claim_gate_row_count": len(summary["claim_gate_rows"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
