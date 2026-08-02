"""Event-focused consolidation for Microwave Fano A2/D/E evidence."""
from __future__ import annotations


EVENT_METRICS = ("peak_window_complex_nrmse", "f0_abs_error_hz")
SUPPORT_METRICS = ("quality_factor_relative_error",)
BLOCKING_METRICS = ("q_abs_error",)
GLOBAL_GUARDRAIL_METRIC = "global_complex_nrmse"
COMPARISON_FAMILIES = ("MLP", "Local nested CF control")


def _effect(row: dict) -> float | None:
    value = row.get("median_relative_delta")
    if value is None:
        return None
    return -float(value)


def _pairwise_lookup(summary: dict) -> dict[tuple[int, str, str], dict]:
    return {
        (int(row["observation_budget"]), str(row["comparison_family"]), str(row["metric"])): row
        for row in summary.get("pairwise_baseline_comparisons", [])
        if row.get("observation_budget") is not None
        and row.get("comparison_family") is not None
        and row.get("metric") is not None
    }


def fano_event_focus_consolidation(
    summary: dict,
    *,
    budget: int = 128,
    highrisk_curve_ids: list[str] | None = None,
    comparison_families: tuple[str, ...] = COMPARISON_FAMILIES,
) -> dict:
    pairwise = _pairwise_lookup(summary)
    rows = []
    event_passes = []
    global_passes = []
    q_passes = []
    for comparison in comparison_families:
        for metric in (GLOBAL_GUARDRAIL_METRIC, *EVENT_METRICS, *SUPPORT_METRICS, *BLOCKING_METRICS):
            row = pairwise.get((int(budget), comparison, metric))
            effect = _effect(row or {})
            win_rate = None if row is None else row.get("baseline_win_rate")
            passes = bool(effect is not None and effect > 0.0)
            rows.append({
                "observation_budget": int(budget),
                "comparison_family": comparison,
                "metric": metric,
                "effect": effect,
                "baseline_win_rate": win_rate,
                "valid_pair_count": None if row is None else row.get("valid_pair_count"),
                "passes_positive_effect": passes,
            })
            if metric in EVENT_METRICS:
                event_passes.append(passes)
            elif metric == GLOBAL_GUARDRAIL_METRIC:
                global_passes.append(bool(effect is not None and effect >= 0.0))
            elif metric in BLOCKING_METRICS:
                q_passes.append(passes)

    event_gate_pass = bool(event_passes and all(event_passes) and global_passes and all(global_passes))
    q_gate_pass = bool(q_passes and all(q_passes))
    if event_gate_pass and not q_gate_pass:
        claim_status = "passes_event_focused_gate_q_blocks_full_inversion"
    elif event_gate_pass and q_gate_pass:
        claim_status = "passes_event_and_q_gate"
    else:
        claim_status = "does_not_pass_event_focused_gate"

    return {
        "analysis_type": "microwave_fano_event_focused_consolidation",
        "source_task": summary.get("task"),
        "source_record_count": summary.get("record_count"),
        "baseline_family": summary.get("baseline_family", "CFNN"),
        "curve_ids": list(summary.get("curve_ids", [])),
        "event_focus_rows": rows,
        "claim_gate": {
            "observation_budget": int(budget),
            "claim_status": claim_status,
            "event_gate_pass": event_gate_pass,
            "full_inversion_status": "passes_q_abs_error" if q_gate_pass else "blocked_by_q_abs_error",
            "event_metrics": list(EVENT_METRICS),
            "support_metrics": list(SUPPORT_METRICS),
            "blocking_metrics": list(BLOCKING_METRICS),
            "global_guardrail_metric": GLOBAL_GUARDRAIL_METRIC,
            "highrisk_curve_ids": list(highrisk_curve_ids or []),
            "reporting_rule": (
                "Report Fano as event-focused peak-window/f0 evidence unless q_abs_error passes; "
                "high-risk curves remain reliability boundaries."
            ),
        },
    }
