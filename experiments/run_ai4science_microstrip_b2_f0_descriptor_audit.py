#!/usr/bin/env python3
"""Audit whether truth-side notch geometry explains Microstrip B2 f0 guardrail failures."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ai4science_microstrip_b import (  # noqa: E402
    microstrip_b2_calibration_coverage_rows,
    microstrip_b2_f0_guardrail_descriptor_rows,
    microstrip_instance_effect_rows,
    microstrip_truth_notch_descriptor_rows,
    summarize_microstrip_b2_f0_guardrail_descriptors,
)
from downstream_protocol import load_microstrip_resonator  # noqa: E402
from run_ai4science_microstrip_b_pilot import _load_jsonl  # noqa: E402

DEFAULT_INPUT = (
    ROOT / "ai4science_results"
    / "microstrip_b2_missed_band_negative_control_seed_expansion"
    / "raw_records.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "ai4science_results"
    / "microstrip_b2_missed_band_f0_descriptor_audit"
)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _write_json(path: Path, value: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(path)


def _top_failures(rows: list[dict], limit: int = 12) -> list[dict]:
    failures = [row for row in rows if bool(row.get("q_positive_f0_negative"))]
    return sorted(
        failures,
        key=lambda row: (
            float(row.get("f0_effect", 0.0)),
            -float(row.get("quality_factor_effect", 0.0)),
            str(row.get("curve_id")),
        ),
    )[: int(limit)]


def _sample_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row.get("sample_id"), row.get("comparison_family")), []).append(row)
    summaries = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        sample_id, comparison = key
        f0_effects = [float(row["f0_effect"]) for row in items if row.get("f0_effect") is not None]
        qpos_f0neg = [row for row in items if bool(row.get("q_positive_f0_negative"))]
        risky = [
            row for row in items
            if row.get("notch_geometry_stratum") == "multi_notch_or_boundary"
        ]
        summaries.append({
            "sample_id": sample_id,
            "comparison_family": comparison,
            "pair_count": len(f0_effects),
            "q_positive_f0_negative_count": len(qpos_f0neg),
            "q_positive_f0_negative_rate": float(len(qpos_f0neg) / len(f0_effects)) if f0_effects else None,
            "risky_notch_geometry_count": len(risky),
            "risky_notch_geometry_rate": float(len(risky) / len(f0_effects)) if f0_effects else None,
            "median_f0_effect": float(np.median(f0_effects)) if f0_effects else None,
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-raw-records", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    args = parser.parse_args()

    records = _load_jsonl(args.input_raw_records)
    curves = load_microstrip_resonator(args.raw_root)
    descriptor_rows = microstrip_truth_notch_descriptor_rows(curves)
    coverage_rows = microstrip_b2_calibration_coverage_rows(records, curves)
    effect_rows = microstrip_instance_effect_rows(
        records,
        comparison_families={"MLP", "Local nested CF control"},
        metrics={"quality_factor_relative_error", "resonance_frequency_abs_error_hz"},
    )
    joined_rows = microstrip_b2_f0_guardrail_descriptor_rows(
        effect_rows,
        descriptor_rows,
        coverage_rows,
    )
    summary = {
        "analysis_type": "microstrip_b2_missed_band_f0_descriptor_audit",
        "input_raw_records": str(args.input_raw_records),
        "truth_descriptor_curve_count": len(descriptor_rows),
        "joined_pair_count": len(joined_rows),
        "q_positive_f0_negative_count": int(sum(1 for row in joined_rows if row.get("q_positive_f0_negative"))),
        "descriptor_summary_rows": summarize_microstrip_b2_f0_guardrail_descriptors(joined_rows),
        "sample_summary_rows": _sample_summary(joined_rows),
        "top_q_positive_f0_negative_failures": _top_failures(joined_rows),
        "settings": {
            "descriptor_rule": "multi_prominent_notch_or_f0_within_1_linewidth_of_search_edge",
            "calibration_coverage_band_linewidths": 1.0,
        },
    }
    _write_json(args.output_root / "truth_notch_descriptor_rows.json", descriptor_rows)
    _write_json(args.output_root / "f0_guardrail_descriptor_rows.json", joined_rows)
    _write_json(args.output_root / "summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
