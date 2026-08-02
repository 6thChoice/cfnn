#!/usr/bin/env python3
"""Build a report-facing AI4Science CFNN claim ledger from Package E outputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_applicability_e import summarize_ai4science_claim_ledger


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_E = ROOT / "ai4science_results" / "package_e_unified_applicability_map" / "summary.json"
DEFAULT_FANO_A2_PREREGISTERED_AUDIT = (
    ROOT / "ai4science_results" / "microwave_fano_a2_preregistered_success_audit" / "summary.json"
)
DEFAULT_FANO_A3_PREREGISTERED_AUDIT = (
    ROOT / "ai4science_results" / "microwave_fano_a3_preregistered_robustness_audit" / "summary.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "ai4science_claim_ledger"


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


def build_report(ledger: dict) -> str:
    advantage_count = int(ledger.get("claim_summary", {}).get("cfnn_advantage_claim_count") or 0)
    if advantage_count > 0:
        interpretation = (
            "The current ledger supports Microwave Fano as the main CFNN advantage task, with explicit reliability guardrails. "
            "Microstrip is a localized Q/linewidth transfer signal blocked from a full digital-twin claim by f0/phase guardrails. "
            "Aluminium is an active-measurement and event-detection value task rather than a CFNN advantage task. "
            "EIS is a negative applicability boundary because neural-only positive cells are eliminated by physical-prior baselines."
        )
    else:
        interpretation = (
            "The current ledger does not support any report-facing CFNN advantage claim under the active preregistered gates. "
            "Microwave Fano remains scientifically useful as a targeted f0/peak-window recovery signal with q/Q and reliability guardrails. "
            "Microstrip, Aluminium, and EIS should be reported as boundary or measurement-value tasks rather than CFNN advantage tasks."
        )
    lines = [
        "# AI4Science CFNN Claim Ledger",
        "",
        "This ledger separates response recovery, event/parameter recovery, reliability, active-measurement value, and physical-prior guardrails. It is intentionally stricter than a metric table: a claim can be positive on one axis and still blocked by another.",
        "",
        "## Summary",
        "",
        f"- source rows: `{ledger.get('source_row_count')}`",
        f"- CFNN advantage claims: `{ledger.get('claim_summary', {}).get('cfnn_advantage_claim_count')}`",
        f"- blocked or boundary claims: `{ledger.get('claim_summary', {}).get('blocked_or_boundary_claim_count')}`",
        "",
        "## Claim Rows",
        "",
        "| claim | domain | type | status | primary effect | guardrail boundary | prereg pass | prereg fail | A3 guardrail | rule |",
        "|---|---|---|---|---:|---:|---|---|---|---|",
    ]
    for row in ledger.get("claim_rows", []):
        passed = ",".join(str(value) for value in row.get("preregistered_passed_budgets", []))
        failed = ",".join(str(value) for value in row.get("preregistered_failed_budgets", []))
        lines.append(
            "| "
            + " | ".join([
                f"`{row.get('claim_id')}`",
                str(row.get("domain")),
                str(row.get("claim_type")),
                str(row.get("claim_status")),
                _fmt(row.get("primary_positive_effect")),
                _fmt(row.get("guardrail_boundary_score")),
                passed,
                failed,
                str(row.get("a3_robustness_guardrail_status") or ""),
                str(row.get("reporting_rule")),
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
    parser.add_argument("--fano-a2-preregistered-success-audit", type=Path, default=DEFAULT_FANO_A2_PREREGISTERED_AUDIT)
    parser.add_argument("--fano-a3-preregistered-robustness-audit", type=Path, default=DEFAULT_FANO_A3_PREREGISTERED_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_e = json.loads(args.package_e_summary.read_text())
    if (
        "fano_a2_preregistered_success_audit" not in package_e
        and args.fano_a2_preregistered_success_audit.exists()
    ):
        package_e["fano_a2_preregistered_success_audit"] = json.loads(
            args.fano_a2_preregistered_success_audit.read_text()
        )
    if (
        "fano_a3_preregistered_robustness_audit" not in package_e
        and args.fano_a3_preregistered_robustness_audit.exists()
    ):
        package_e["fano_a3_preregistered_robustness_audit"] = json.loads(
            args.fano_a3_preregistered_robustness_audit.read_text()
        )
    ledger = summarize_ai4science_claim_ledger(package_e)
    output = {
        **ledger,
        "source_package_e_summary": str(args.package_e_summary),
        "source_fano_a2_preregistered_success_audit": (
            str(args.fano_a2_preregistered_success_audit)
            if args.fano_a2_preregistered_success_audit.exists() else None
        ),
        "source_fano_a3_preregistered_robustness_audit": (
            str(args.fano_a3_preregistered_robustness_audit)
            if args.fano_a3_preregistered_robustness_audit.exists() else None
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(output) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "claim_count": output["claim_summary"]["claim_count"],
        "cfnn_advantage_claim_count": output["claim_summary"]["cfnn_advantage_claim_count"],
        "blocked_or_boundary_claim_count": output["claim_summary"]["blocked_or_boundary_claim_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
