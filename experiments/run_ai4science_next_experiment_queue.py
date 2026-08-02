#!/usr/bin/env python3
"""Generate the next-experiment queue from the AI4Science progress audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_next_experiment_queue import build_next_experiment_queue


ROOT = Path(__file__).resolve().parent
DEFAULT_PROGRESS = ROOT / "ai4science_results" / "ai4science_experiment_plan_progress_audit" / "summary.json"
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "ai4science_next_experiment_queue"
MICROSTRIP_B_GUARDRAIL_SUMMARY = (
    ROOT / "ai4science_results" / "microstrip_b2_confirmatory_guardrail_strata" / "summary.json"
)
EIS_EXTERNAL_BOUNDARY_SUMMARY = (
    ROOT / "ai4science_results" / "eis_external_spectrum_boundary" / "summary.json"
)
FANO_D_HIGHRISK_MULTISPLIT_SUMMARY = (
    ROOT / "ai4science_results" / "microwave_fano_e_highrisk_budget_escalation_r8p03_multisplit" / "summary.json"
)
FANO_EVENT_FOCUS_SUMMARY = (
    ROOT / "ai4science_results" / "microwave_fano_event_focused_consolidation" / "summary.json"
)
MICROSTRIP_LOW_CALIBRATION_EVENT_TRANSFER_SUMMARY = (
    ROOT / "ai4science_results" / "microstrip_low_calibration_event_transfer" / "summary.json"
)
ALUMINIUM_ACTIVE_MEASUREMENT_VALUE_SUMMARY = (
    ROOT / "ai4science_results" / "aluminium_c_active_measurement_value_audit" / "summary.json"
)


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


def _completed_queue_artifacts() -> dict:
    artifacts = {}
    if FANO_EVENT_FOCUS_SUMMARY.exists():
        summary = json.loads(FANO_EVENT_FOCUS_SUMMARY.read_text())
        claim_status = summary.get("claim_gate", {}).get("claim_status")
        if claim_status in {
            "passes_event_focused_gate_q_blocks_full_inversion",
            "passes_event_and_q_gate",
            "does_not_pass_event_focused_gate",
        }:
            artifacts["A_fano_event_focused_consolidation"] = {
                "status": "ok",
                "artifact_path": str(FANO_EVENT_FOCUS_SUMMARY),
                "claim_status": claim_status,
            }
    if MICROSTRIP_LOW_CALIBRATION_EVENT_TRANSFER_SUMMARY.exists():
        summary = json.loads(MICROSTRIP_LOW_CALIBRATION_EVENT_TRANSFER_SUMMARY.read_text())
        claim_status = summary.get("claim_gate", {}).get("claim_status")
        if claim_status in {
            "localized_q_transfer_with_guardrails_not_digital_twin",
            "no_guardrail_passing_localized_q_transfer",
        }:
            artifacts["B_microstrip_low_calibration_event_transfer"] = {
                "status": "ok",
                "artifact_path": str(MICROSTRIP_LOW_CALIBRATION_EVENT_TRANSFER_SUMMARY),
                "claim_status": claim_status,
                "focused_stratum_row_count": len(summary.get("focused_stratum_rows", [])),
            }
    if ALUMINIUM_ACTIVE_MEASUREMENT_VALUE_SUMMARY.exists():
        summary = json.loads(ALUMINIUM_ACTIVE_MEASUREMENT_VALUE_SUMMARY.read_text())
        if (
            summary.get("analysis_type") == "aluminium_c_active_measurement_value_audit"
            or summary.get("claim_level") == "measurement_strategy_signal_not_cfnn_advantage"
        ):
            artifacts["C_aluminium_boundary_retained"] = {
                "status": "ok",
                "artifact_path": str(ALUMINIUM_ACTIVE_MEASUREMENT_VALUE_SUMMARY),
                "claim_level": summary.get("claim_level"),
            }
    if MICROSTRIP_B_GUARDRAIL_SUMMARY.exists():
        summary = json.loads(MICROSTRIP_B_GUARDRAIL_SUMMARY.read_text())
        if (
            summary.get("primary_claim_gate") == "q_gain_requires_f0_and_phase_guardrails"
            and int(summary.get("claim_gate_row_count", len(summary.get("claim_gate_rows", [])))) >= 1
        ):
            artifacts["B_confirmatory_guardrail_strata"] = {
                "status": "ok",
                "artifact_path": str(MICROSTRIP_B_GUARDRAIL_SUMMARY),
                "claim_gate_row_count": int(
                    summary.get("claim_gate_row_count", len(summary.get("claim_gate_rows", [])))
                ),
            }
    if EIS_EXTERNAL_BOUNDARY_SUMMARY.exists():
        summary = json.loads(EIS_EXTERNAL_BOUNDARY_SUMMARY.read_text())
        if summary.get("gate", {}).get("claim_status") == "physical_prior_boundary_supported":
            artifacts["E_external_spectrum_boundary"] = {
                "status": "ok",
                "artifact_path": str(EIS_EXTERNAL_BOUNDARY_SUMMARY),
                "claim_status": summary.get("gate", {}).get("claim_status"),
            }
    if FANO_D_HIGHRISK_MULTISPLIT_SUMMARY.exists():
        payload = json.loads(FANO_D_HIGHRISK_MULTISPLIT_SUMMARY.read_text())
        summary = payload.get("summary", {})
        if summary.get("status") == "ok" and int(summary.get("split_count", 0)) >= 1:
            best = summary.get("best_family_by_split_pass_then_median_budget") or {}
            artifacts["D_highrisk_budget_multisplit_refresh"] = {
                "status": "ok",
                "artifact_path": str(FANO_D_HIGHRISK_MULTISPLIT_SUMMARY),
                "best_family": best.get("family"),
                "split_count": int(summary.get("split_count", 0)),
            }
    return artifacts


def build_report(queue: dict) -> str:
    lines = [
        "# AI4Science Next Experiment Queue",
        "",
        "This queue translates the plan progress audit into concrete next actions. `runnable_now` rows point to existing repository runners; `design_needed` rows are not commands and should not be launched as experiments.",
        "",
        "## Summary",
        "",
        f"- queue rows: `{queue.get('summary', {}).get('queue_count')}`",
        f"- runnable now: `{queue.get('summary', {}).get('runnable_now_count')}`",
        f"- design needed: `{queue.get('summary', {}).get('design_needed_count')}`",
        "",
        "## Queue",
        "",
        "| priority | queue id | package | status | objective | success gate | command |",
        "|---:|---|---|---|---|---|---|",
    ]
    for row in queue.get("queue_rows", []):
        command = row.get("command") or "NA"
        lines.append(
            "| "
            + " | ".join([
                str(row.get("priority")),
                f"`{row.get('queue_id')}`",
                str(row.get("package_id")),
                str(row.get("execution_status")),
                str(row.get("objective")),
                str(row.get("success_gate")),
                f"`{command}`",
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Use",
        "",
        "Run queued commands one at a time. After any command changes generated artifacts, rerun the ledger refresh queue item before updating narrative reports.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-summary", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress = json.loads(args.progress_summary.read_text())
    progress = {
        **progress,
        "completed_queue_artifacts": {
            **(progress.get("completed_queue_artifacts") or {}),
            **_completed_queue_artifacts(),
        },
    }
    queue = build_next_experiment_queue(progress)
    output = {
        **queue,
        "source_progress_summary": str(args.progress_summary),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(output) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "queue_count": output["summary"]["queue_count"],
        "runnable_now_count": output["summary"]["runnable_now_count"],
        "design_needed_count": output["summary"]["design_needed_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
