#!/usr/bin/env python3
"""Build a progress audit for the complete AI4Science experiment plan."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_experiment_plan_audit import summarize_ai4science_experiment_plan_progress


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_E = ROOT / "ai4science_results" / "package_e_unified_applicability_map" / "summary.json"
DEFAULT_CLAIM_LEDGER = ROOT / "ai4science_results" / "ai4science_claim_ledger" / "summary.json"
DEFAULT_ALUMINIUM_AUDIT = ROOT / "ai4science_results" / "aluminium_c_active_measurement_value_audit" / "summary.json"
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "ai4science_experiment_plan_progress_audit"


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


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def build_report(audit: dict) -> str:
    advantage_count = int(audit.get("summary", {}).get("cfnn_advantage_package_count") or 0)
    if advantage_count > 0:
        interpretation = (
            "The current plan audit supports a narrow CFNN advantage centered on Microwave Fano A2, with Fano A3/D reliability guardrails. "
            "Microstrip and Aluminium remain scientifically useful boundary tasks: Microstrip isolates local Q/linewidth transfer but blocks full digital-twin language, "
            "while Aluminium supports active measurement value without a same-schedule CFNN modal advantage. EIS is a physical-prior applicability boundary."
        )
    else:
        interpretation = (
            "The current plan audit does not support a report-facing CFNN advantage package under the active preregistered gates. "
            "Microwave Fano should be framed as targeted resonance-frequency and peak-window recovery evidence with q/Q and reliability boundaries. "
            "Microstrip, Aluminium, and EIS remain useful boundary or measurement-value tasks."
        )
    lines = [
        "# AI4Science Experiment Plan Progress Audit",
        "",
        f"Core question: `{audit.get('core_question')}`",
        "",
        "## Summary",
        "",
        f"- plan rows: `{audit.get('summary', {}).get('row_count')}`",
        f"- CFNN advantage packages: `{audit.get('summary', {}).get('cfnn_advantage_package_count')}`",
        f"- boundary or guardrail packages: `{audit.get('summary', {}).get('boundary_or_guardrail_package_count')}`",
        "",
        "## Plan Rows",
        "",
        "| package | task | role | status | evidence | next action |",
        "|---|---|---|---|---|---|",
    ]
    for row in audit.get("plan_rows", []):
        lines.append(
            "| "
            + " | ".join([
                f"`{row.get('package_id')}`",
                str(row.get("package_name")),
                str(row.get("claim_role")),
                str(row.get("status")),
                str(row.get("evidence")),
                str(row.get("next_action")),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        interpretation,
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-e-summary", type=Path, default=DEFAULT_PACKAGE_E)
    parser.add_argument("--claim-ledger-summary", type=Path, default=DEFAULT_CLAIM_LEDGER)
    parser.add_argument("--aluminium-measurement-audit", type=Path, default=DEFAULT_ALUMINIUM_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = summarize_ai4science_experiment_plan_progress(
        _load(args.package_e_summary),
        _load(args.claim_ledger_summary),
        _load(args.aluminium_measurement_audit),
    )
    output = {
        **audit,
        "source_package_e_summary": str(args.package_e_summary),
        "source_claim_ledger_summary": str(args.claim_ledger_summary),
        "source_aluminium_measurement_audit": str(args.aluminium_measurement_audit),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(output) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "row_count": output["summary"]["row_count"],
        "cfnn_advantage_package_count": output["summary"]["cfnn_advantage_package_count"],
        "boundary_or_guardrail_package_count": output["summary"]["boundary_or_guardrail_package_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
