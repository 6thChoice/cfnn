"""Next-experiment queue derived from the AI4Science progress audit."""
from __future__ import annotations


def _status(progress: dict, package_id: str) -> str | None:
    for row in progress.get("plan_rows", []) or []:
        if row.get("package_id") == package_id:
            return row.get("status")
    return None


def _completed_artifact(progress: dict, queue_id: str) -> dict | None:
    artifact = (progress.get("completed_queue_artifacts") or {}).get(queue_id)
    if isinstance(artifact, dict) and artifact.get("status") == "ok":
        return artifact
    return None


def _row(
    *,
    queue_id: str,
    priority: int,
    package_id: str,
    objective: str,
    execution_status: str,
    command: str | None,
    success_gate: str,
    rationale: str,
    artifact: dict | None = None,
) -> dict:
    if artifact is not None:
        execution_status = "satisfied_current_artifact"
        command = None
    return {
        "queue_id": queue_id,
        "priority": int(priority),
        "package_id": package_id,
        "objective": objective,
        "execution_status": execution_status,
        "command": command,
        "success_gate": success_gate,
        "rationale": rationale,
        **({"artifact_path": artifact.get("artifact_path"), "artifact_status": artifact.get("status")} if artifact else {}),
    }


def build_next_experiment_queue(progress_audit: dict) -> dict:
    """Build prioritized next experiments without launching long training jobs."""
    rows: list[dict] = []
    if _status(progress_audit, "A2") in {
        "blocked_by_preregistered_budget_gate",
        "supported_with_preregistered_budget_guardrails",
    }:
        artifact = _completed_artifact(progress_audit, "A_fano_event_focused_consolidation")
        rows.append(_row(
            queue_id="A_fano_event_focused_consolidation",
            priority=1,
            package_id="A2",
            objective=(
                "Consolidate Microwave Fano event-focused evidence for peak-window and f0 recovery "
                "with q as a blocking parameter."
            ),
            execution_status="runnable_now",
            command=(
                "PATH=/home/zxc/miniconda3/bin:$PATH /home/zxc/miniconda3/bin/python "
                "experiment_refine/run_ai4science_fano_event_focus_consolidation.py"
            ),
            success_gate=(
                "At 128 observations, ordinary-curve CFNN event-focused peak-window and f0 effects must pass against "
                "MLP and Local nested CF control; q remains a full-inversion blocker unless separately passing."
            ),
            rationale=(
                "Fano is the strongest resonant-response task, but the defensible claim is event-focused "
                "rather than full f0/Q/q inversion."
            ),
            artifact=artifact,
        ))
    if _status(progress_audit, "B") == "localized_q_signal_blocked_full_digital_twin":
        artifact = _completed_artifact(progress_audit, "B_microstrip_low_calibration_event_transfer")
        rows.append(_row(
            queue_id="B_microstrip_low_calibration_event_transfer",
            priority=2,
            package_id="B",
            objective=(
                "Consolidate Microstrip low-calibration Q/linewidth transfer with f0, local phase, "
                "and global guardrails."
            ),
            execution_status="runnable_now",
            command=(
                "PATH=/home/zxc/miniconda3/bin:$PATH /home/zxc/miniconda3/bin/python "
                "experiment_refine/run_ai4science_microstrip_low_calibration_event_transfer.py"
            ),
            success_gate=(
                "Q/linewidth gains may be reported only inside pre-registered strata and only with f0, local phase, "
                "and global response guardrails shown."
            ),
            rationale=(
                "Microstrip is the real-device follow-up, but its defensible signal is localized Q/linewidth "
                "transfer rather than full digital-twin behavior."
            ),
            artifact=artifact,
        ))
    if _status(progress_audit, "D") == "ordinary_reliability_supported_highrisk_requires_escalation":
        artifact = _completed_artifact(progress_audit, "D_highrisk_budget_multisplit_refresh")
        rows.append(_row(
            queue_id="D_highrisk_budget_multisplit_refresh",
            priority=3,
            package_id="D",
            objective="Refresh high-risk Fano same-curve escalation budget distribution for r8:p03.",
            execution_status="runnable_now",
            command=(
                "PATH=/home/zxc/miniconda3/bin:$PATH /home/zxc/miniconda3/bin/python "
                "experiment_refine/run_ai4science_fano_budget_escalation_multisplit_e.py "
                "--audit-summary "
                "experiment_refine/ai4science_results/microwave_fano_e_highrisk_budget_escalation_r8p03/summary.json "
                "experiment_refine/ai4science_results/microwave_fano_e_highrisk_budget_escalation_r8p03_split733/summary.json "
                "experiment_refine/ai4science_results/microwave_fano_e_highrisk_budget_escalation_r8p03_split751/summary.json "
                "--split-labels split719 split733 split751"
            ),
            success_gate=(
                "High-risk handling remains a budget/escalation distribution, not a CFNN reliability advantage, "
                "unless a fixed family and budget pass across split seeds."
            ),
            rationale="Reliability language depends on keeping r8:p03 separated from ordinary Fano curves.",
            artifact=artifact,
        ))
    if _status(progress_audit, "C") == "measurement_strategy_value_without_cfnn_advantage":
        artifact = _completed_artifact(progress_audit, "C_aluminium_boundary_retained") or {
            "status": "ok",
            "artifact_path": "experiment_refine/ai4science_results/aluminium_c_active_measurement_value_audit/summary.json",
        }
        rows.append(_row(
            queue_id="C_aluminium_boundary_retained",
            priority=4,
            package_id="C",
            objective="Retain Aluminium FRF as active-measurement boundary evidence without further CFNN advantage mining.",
            execution_status="satisfied_current_artifact",
            command=None,
            success_gate="Aluminium remains a boundary task unless a future separately approved design changes this status.",
            rationale="Observed-only sampling has measurement value, while same-schedule CFNN modal recovery remains blocked.",
            artifact=artifact,
        ))
    if _status(progress_audit, "E") == "physical_prior_boundary_supported":
        artifact = _completed_artifact(progress_audit, "E_external_spectrum_boundary") or {
            "status": "ok",
            "artifact_path": "experiment_refine/ai4science_results/eis_external_spectrum_boundary/summary.json",
        }
        rows.append(_row(
            queue_id="E_external_spectrum_boundary",
            priority=5,
            package_id="E",
            objective="Retain EIS as physical-prior boundary evidence for CFNN applicability limits.",
            execution_status="satisfied_current_artifact",
            command=None,
            success_gate=(
                "The physical-prior boundary should persist across broader impedance spectra; neural-only edges must be reported separately "
                "and cannot override circuit-prior baselines."
            ),
            rationale="EIS is currently a boundary claim; extension should test scope, not mine neural-only positives.",
            artifact=artifact,
        ))
    rows.append(_row(
        queue_id="Ledger_refresh_after_queue_items",
        priority=99,
        package_id="Ledger",
        objective="Regenerate Package E, claim ledger, and plan progress audit after any queued experiment changes.",
        execution_status="runnable_now",
        command=(
            "PATH=/home/zxc/miniconda3/bin:$PATH /home/zxc/miniconda3/bin/python "
            "experiment_refine/run_ai4science_applicability_e.py && "
            "PATH=/home/zxc/miniconda3/bin:$PATH /home/zxc/miniconda3/bin/python "
            "experiment_refine/run_ai4science_claim_ledger.py && "
            "PATH=/home/zxc/miniconda3/bin:$PATH /home/zxc/miniconda3/bin/python "
            "experiment_refine/run_ai4science_experiment_plan_progress_audit.py"
        ),
        success_gate="Report-facing ledgers match the latest generated experiment artifacts.",
        rationale="Every experimental update should be reflected in the unified applicability map and claim controls.",
    ))
    rows = sorted(rows, key=lambda row: row["priority"])
    return {
        "analysis_type": "ai4science_next_experiment_queue",
        "source_progress_core_question": progress_audit.get("core_question"),
        "queue_rows": rows,
        "summary": {
            "queue_count": len(rows),
            "runnable_now_count": sum(1 for row in rows if row["execution_status"] == "runnable_now"),
            "design_needed_count": sum(1 for row in rows if row["execution_status"] == "design_needed"),
            "satisfied_current_artifact_count": sum(
                1 for row in rows if row["execution_status"] == "satisfied_current_artifact"
            ),
        },
    }
