"""Progress audit for the AI4Science CFNN experiment plan."""
from __future__ import annotations


CORE_QUESTION_ID = "CFNN_sparse_resonant_complex_response_recovery_interpretability_reliability"


def _claim_status(claim_ledger: dict, claim_id: str) -> str | None:
    for row in claim_ledger.get("claim_rows", []) or []:
        if row.get("claim_id") == claim_id:
            return row.get("claim_status")
    return None


def _joined(values) -> str:
    items = [str(value) for value in values or []]
    return "/".join(items) if items else "none"


def _decision_row(audit: dict, decision_id: str) -> dict:
    for row in audit.get("decision_rows", []) or []:
        if row.get("decision_id") == decision_id:
            return row
    return {}


def _count_word(value) -> str:
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    number = int(value or 0)
    return words.get(number, str(number))


def _is_supported_advantage_status(status: str | None) -> bool:
    return str(status or "").startswith("supported")


def summarize_ai4science_experiment_plan_progress(
    package_e_summary: dict,
    claim_ledger: dict,
    aluminium_measurement_audit: dict,
) -> dict:
    """Summarize A-E progress against the preregistered AI4Science plan."""
    a2 = package_e_summary.get("fano_a2_preregistered_success_audit") or {}
    a3 = package_e_summary.get("fano_a3_preregistered_robustness_audit") or {}
    reliability = package_e_summary.get("fano_reliability_claim_gate") or {}
    microstrip = package_e_summary.get("microstrip_digital_twin_claim_gate") or {}
    eis = package_e_summary.get("eis_applicability_claim_gate") or {}
    interpolation = _decision_row(aluminium_measurement_audit, "observed_interpolation_measurement_signal")
    same_schedule = _decision_row(aluminium_measurement_audit, "same_schedule_cfnn_boundary")
    candidate = _decision_row(aluminium_measurement_audit, "candidate_discovery_boundary")

    a2_passed = a2.get("passed_budgets") or []
    a2_failed = [
        budget for budget in a2.get("budgets", []) or []
        if budget not in set(a2_passed)
    ]
    a2_status = _claim_status(claim_ledger, "fano_sparse_parameter_recovery") or (
        "supported_with_preregistered_budget_guardrails" if a2_passed else "incomplete"
    )
    a3_status = (
        "robustness_guardrail_not_full_reliability"
        if a3.get("overall_status") == "does_not_pass_local_control_guardrail"
        else str(a3.get("overall_status") or "incomplete")
    )
    microstrip_status = (
        "localized_q_signal_blocked_full_digital_twin"
        if microstrip.get("claim_status") == "blocked_by_global_event_guardrails"
        and microstrip.get("localized_q_signal_status")
        else str(microstrip.get("claim_status") or _claim_status(claim_ledger, "microstrip_device_digital_twin") or "incomplete")
    )
    aluminium_status = _claim_status(claim_ledger, "aluminium_active_modal_cfnn_advantage") or str(
        aluminium_measurement_audit.get("claim_level") or "incomplete"
    )
    eis_status = _claim_status(claim_ledger, "eis_general_advantage") or str(eis.get("claim_status") or "incomplete")
    claim_summary = claim_ledger.get("claim_summary", {}) or {}
    ledger_advantage_count = int(claim_summary.get("cfnn_advantage_claim_count") or 0)
    ledger_boundary_count = int(claim_summary.get("blocked_or_boundary_claim_count") or 0)
    ledger_status = (
        f"{_count_word(ledger_advantage_count)}_cfnn_advantage_"
        f"{_count_word(ledger_boundary_count)}_boundaries"
    )
    a2_next_action = (
        "Do not claim A2 scientific inversion advantage; analyze f0/peak-window positive signals and q/Q failure mechanisms before any renewed claim."
        if not _is_supported_advantage_status(a2_status)
        else "Run confirmatory curve expansion focused on preregistered 64/128 budgets and keep 32-point failure as sparse-budget boundary."
    )

    rows = [
        {
            "package_id": "A1",
            "package_name": "Microwave Fano reference parameter labeling",
            "status": "reference_labels_available_for_downstream_audits",
            "claim_role": "truth_construction",
            "evidence": "Fano A1 reference fit artifacts support A2/A3/D/E parameter extraction and boundary labeling.",
            "next_action": "Keep high-res reference fits independent from neural predictions; expand rejected-fit labels when adding curves.",
        },
        {
            "package_id": "A2",
            "package_name": "Microwave Fano sparse response and parameter recovery",
            "status": a2_status,
            "claim_role": "main_cfnn_advantage",
            "evidence": (
                f"passes at {_joined(a2_passed)} observations; "
                f"fails at {_joined(a2_failed)} observations"
            ),
            "next_action": a2_next_action,
        },
        {
            "package_id": "A3",
            "package_name": "Microwave Fano robustness under noise, missing observations, and drift",
            "status": a3_status,
            "claim_role": "robustness_guardrail",
            "evidence": (
                f"{a3.get('passed_condition_count')} full MLP+Local pass conditions; "
                f"{a3.get('mlp_only_or_partial_condition_count')} MLP-only or partial conditions"
            ),
            "next_action": "Report A3 as MLP-relative robustness with Local-control guardrails; avoid broad reliability language.",
        },
        {
            "package_id": "B",
            "package_name": "Microstrip Resonator cross-condition digital twin",
            "status": microstrip_status,
            "claim_role": "localized_device_transfer_boundary",
            "evidence": (
                f"localized Q status: {microstrip.get('localized_q_signal_status')}; "
                f"blocking layers: {','.join(microstrip.get('failed_effect_layers') or [])}"
            ),
            "next_action": "Move only guardrail-passing Q/linewidth strata into confirmatory B2; keep f0 and phase as blocking digital-twin gates.",
        },
        {
            "package_id": "C",
            "package_name": "Aluminium FRF active modal analysis",
            "status": aluminium_status,
            "claim_role": "measurement_strategy_boundary",
            "evidence": (
                f"observed-only interpolation reaches {interpolation.get('median_passed_modes')} passed modes at "
                f"{interpolation.get('minimum_budget')} points; same-schedule CFNN reaches "
                f"{same_schedule.get('median_passed_modes')} passed modes"
            ),
            "measurement_strategy_minimum_budget": interpolation.get("minimum_budget"),
            "same_schedule_cfnn_passed_modes": same_schedule.get("median_passed_modes"),
            "candidate_discovery_status": candidate.get("verdict"),
            "next_action": "Continue with observed-only candidate detection and uncertainty sampling; do not claim CFNN modal advantage until same-schedule neural recovery passes.",
        },
        {
            "package_id": "D",
            "package_name": "Uncertainty quantification and reliability decision gate",
            "status": str(reliability.get("claim_status") or "incomplete"),
            "claim_role": "reliability_guardrail",
            "evidence": (
                f"ordinary pass: {reliability.get('ordinary_curve_reliability_pass')}; "
                f"high-risk curves: {','.join(reliability.get('highrisk_curve_ids') or [])}"
            ),
            "next_action": "Treat high-risk Fano spectra as escalation cases requiring same-curve calibration and extra measurement budgets.",
        },
        {
            "package_id": "E",
            "package_name": "EIS applicability boundary and spectral descriptor map",
            "status": eis_status,
            "claim_role": "negative_applicability_boundary",
            "evidence": (
                f"{eis.get('physical_prior_eliminated_edge_count')} / "
                f"{eis.get('neural_positive_edge_count')} neural-only positive edges eliminated by physical priors"
            ),
            "next_action": "Validate the physical-prior boundary on broader impedance spectra; do not report neural-only edges as structural advantages.",
        },
        {
            "package_id": "Ledger",
            "package_name": "Cross-task claim ledger",
            "status": ledger_status,
            "claim_role": "reporting_control",
            "evidence": (
                f"{ledger_advantage_count} CFNN advantage claim; "
                f"{ledger_boundary_count} blocked/boundary claims"
            ),
            "next_action": "Update this audit whenever Package E or the claim ledger changes before updating paper/Notion prose.",
        },
    ]
    advantage_count = sum(
        1 for row in rows
        if row["claim_role"] == "main_cfnn_advantage"
        and _is_supported_advantage_status(row.get("status"))
    )
    boundary_count = sum(1 for row in rows if "boundary" in row["claim_role"] or "guardrail" in row["claim_role"])
    return {
        "analysis_type": "ai4science_experiment_plan_progress_audit",
        "core_question": CORE_QUESTION_ID,
        "source_package_e_row_count": package_e_summary.get("unified_summary", {}).get("row_count"),
        "plan_rows": rows,
        "summary": {
            "row_count": len(rows),
            "cfnn_advantage_package_count": advantage_count,
            "boundary_or_guardrail_package_count": boundary_count,
        },
    }
