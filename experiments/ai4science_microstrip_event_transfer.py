"""Focused low-calibration Microstrip event-transfer consolidation."""
from __future__ import annotations

from collections import defaultdict
import math
import statistics


FOCUS_STRATA = ("edge_intermediate_missed_band", "edge_proximal_missed_band")


def _median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else None


def _status(
    q_effect: float | None,
    f0_effect: float | None,
    phase_effect: float | None,
    global_effect: float | None,
) -> str:
    q_positive = q_effect is not None and q_effect > 0.0
    guardrails = (
        f0_effect is not None and f0_effect >= 0.0
        and phase_effect is not None and phase_effect >= 0.0
        and global_effect is not None and global_effect >= 0.0
    )
    if q_positive and guardrails:
        return "passes_localized_q_transfer"
    if q_positive:
        return "q_signal_blocked_by_event_guardrail"
    return "no_confirmed_q_transfer"


def microstrip_low_calibration_event_transfer(summary: dict) -> dict:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in summary.get("claim_gate_rows", []):
        if row.get("stratum_id") not in FOCUS_STRATA:
            continue
        key = (
            str(row.get("stratum_id")),
            str(row.get("comparison_family")),
            int(row.get("calibration_frequency_count")),
        )
        groups[key].append(row)

    focused = []
    for (stratum, comparison, calibration), rows in sorted(groups.items()):
        q_effect = _median([row.get("quality_factor_effect") for row in rows])
        f0_effect = _median([row.get("f0_effect") for row in rows])
        phase_effect = _median([row.get("local_phase_effect") for row in rows])
        global_effect = _median([row.get("global_complex_nrmse_effect") for row in rows])
        focused.append({
            "stratum_id": stratum,
            "comparison_family": comparison,
            "calibration_frequency_count": calibration,
            "quality_factor_effect": q_effect,
            "f0_effect": f0_effect,
            "local_phase_effect": phase_effect,
            "global_complex_nrmse_effect": global_effect,
            "localized_transfer_status": _status(q_effect, f0_effect, phase_effect, global_effect),
        })

    pass_rows = [row for row in focused if row["localized_transfer_status"] == "passes_localized_q_transfer"]
    blocked_rows = [
        row for row in focused if row["localized_transfer_status"] == "q_signal_blocked_by_event_guardrail"
    ]
    claim_status = (
        "localized_q_transfer_with_guardrails_not_digital_twin"
        if pass_rows else
        "no_guardrail_passing_localized_q_transfer"
    )
    return {
        "analysis_type": "microstrip_low_calibration_event_transfer",
        "source_analysis_type": summary.get("analysis_type"),
        "focused_stratum_rows": focused,
        "claim_gate": {
            "claim_status": claim_status,
            "localized_transfer_pass_count": len(pass_rows),
            "guardrail_blocked_q_count": len(blocked_rows),
            "digital_twin_status": "blocked_full_digital_twin_language",
            "reporting_rule": (
                "Report Microstrip as localized Q/linewidth transfer only; f0, local phase, "
                "and global response remain guardrails."
            ),
        },
    }
