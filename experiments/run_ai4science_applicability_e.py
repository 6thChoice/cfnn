#!/usr/bin/env python3
"""Build a unified Package E applicability map across Fano and EIS."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_applicability_e import (
    build_unified_applicability_rows,
    summarize_aluminium_active_modal_claim_gate,
    summarize_eis_applicability_claim_gate,
    summarize_fano_reliability_claim_gate,
    summarize_microstrip_digital_twin_claim_gate,
    summarize_descriptor_strata,
    summarize_microstrip_confirmatory_guardrail_strata,
    summarize_microstrip_guardrail_decisions,
    summarize_unified_rows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "package_e_unified_applicability_map"
DEFAULT_FANO_E = ROOT / "ai4science_results" / "microwave_fano_e_boundary_descriptors_5curve_d" / "summary.json"
DEFAULT_EIS_E = ROOT / "ai4science_results" / "eis_e_applicability_map_starter" / "summary.json"
DEFAULT_FANO_A2 = ROOT / "ai4science_results" / "microwave_fano_a2_production" / "enhanced_summary.json"
DEFAULT_FANO_A2_PREREGISTERED_AUDIT = (
    ROOT / "ai4science_results" / "microwave_fano_a2_preregistered_success_audit" / "summary.json"
)
DEFAULT_FANO_A3_PREREGISTERED_AUDIT = (
    ROOT / "ai4science_results" / "microwave_fano_a3_preregistered_robustness_audit" / "summary.json"
)
DEFAULT_FANO_A3_SUMMARIES = [
    (
        "noise",
        ROOT / "ai4science_results" / "microwave_fano_a3_noise_heatmap_triad_seed_expansion" / "summary.json",
    ),
    (
        "missing",
        ROOT / "ai4science_results" / "microwave_fano_a3_missing5_10pct_triad_seed_expansion" / "summary.json",
    ),
    (
        "drift",
        ROOT / "ai4science_results" / "microwave_fano_a3_drift5ppm_triad_seed_expansion" / "summary.json",
    ),
]
DEFAULT_MICROSTRIP_B = ROOT / "ai4science_results" / "microstrip_b1_b2_seed_expansion" / "analysis_summary.json"
DEFAULT_MICROSTRIP_B_MISSED_BAND = (
    ROOT / "ai4science_results" / "microstrip_b2_missed_band_negative_control_seed_expansion"
    / "random_vs_missed_band_seed_expansion_summary.json"
)
DEFAULT_MICROSTRIP_B_F0_DESCRIPTOR = (
    ROOT / "ai4science_results" / "microstrip_b2_missed_band_f0_descriptor_audit"
    / "summary.json"
)
DEFAULT_ALUMINIUM_C = ROOT / "ai4science_results" / "aluminium_c_active_modal_guardrail_with_interpolation" / "analysis_summary.json"
DEFAULT_EIS_NEURAL_EDGE_AUDIT = ROOT / "ai4science_results" / "eis_neural_positive_edge_audit" / "summary.json"
DEFAULT_ALUMINIUM_ACTIVE_MEASUREMENT_AUDIT = (
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


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def build_report(summary: dict) -> str:
    rows = summary["unified_rows"]
    domain_summaries = summary["unified_summary"]["domain_summaries"]
    lines = [
        "# Package E Unified Applicability Map",
        "",
        "This report joins Microwave Fano accuracy/reliability evidence, Microstrip device-generalization evidence, Aluminium active-modal boundary evidence, and EIS accuracy-effect evidence. The axes differ by design: Fano A2 rows quantify signed CFNN recovery effects under sparse clean observation, Fano A3 rows quantify signed robustness effects under noise/missing/drift perturbations, Fano D/E rows quantify calibrated reliability failures, Microstrip B rows quantify signed device generalization/calibration effects, Aluminium C rows quantify modal-recovery boundary failures, and EIS rows quantify signed CFNN effects against neural or physical-prior baselines.",
        "",
        "## Domain Summary",
        "",
        "| domain | rows | median CFNN effect | positive | negative | median boundary score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, row in domain_summaries.items():
        lines.append(
            f"| {domain} | {row['row_count']} | {_fmt(row['median_cfnn_effect'])} | {row['positive_effect_count']} | {row['negative_effect_count']} | {_fmt(row['median_boundary_score'])} |"
        )
    layer_groups = {}
    for row in rows:
        group = f"{row['domain']} / {row.get('effect_layer')}"
        layer_groups.setdefault(group, []).append(row)
    lines.extend([
        "",
        "## Layer Summary",
        "",
        "| group | rows | median CFNN effect | positive | negative | median boundary score |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for group, items in sorted(layer_groups.items()):
        effects = [item["cfnn_effect"] for item in items if item.get("cfnn_effect") is not None]
        scores = [item["boundary_score"] for item in items if item.get("boundary_score") is not None]
        lines.append(
            f"| {group} | {len(items)} | {_fmt(float(np.median(effects)) if effects else None)} | {sum(1 for value in effects if value > 0.0)} | {sum(1 for value in effects if value < 0.0)} | {_fmt(float(np.median(scores)) if scores else None)} |"
        )
    fano_prereg = summary.get("fano_a2_preregistered_success_audit", {})
    fano_prereg_rows = fano_prereg.get("budget_comparison_rows", [])
    if fano_prereg_rows:
        passed_budgets = ",".join(str(value) for value in fano_prereg.get("passed_budgets", []))
        lines.extend([
            "",
            "## Microwave Fano A2 Preregistered Success Audit",
            "",
            f"Overall status: `{fano_prereg.get('overall_status')}`",
            f"Passed budgets: `{passed_budgets}`",
            "",
            "| budget | baseline | status | global | peak | f0 | Q | q | failed metrics |",
            "|---:|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for row in fano_prereg_rows:
            effects = row.get("metric_effects", {})
            failed = ",".join(row.get("failed_required_metrics", []) or row.get("missing_required_metrics", []))
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
                    failed,
                ])
                + " |"
            )
    fano_a3_prereg = summary.get("fano_a3_preregistered_robustness_audit", {})
    fano_a3_gate_rows = fano_a3_prereg.get("condition_gate_rows", [])
    if fano_a3_gate_rows:
        lines.extend([
            "",
            "## Microwave Fano A3 Preregistered Robustness Audit",
            "",
            f"Overall status: `{fano_a3_prereg.get('overall_status')}`",
            f"Condition count: `{fano_a3_prereg.get('condition_count')}`",
            f"Passed condition count: `{fano_a3_prereg.get('passed_condition_count')}`",
            f"MLP-only or partial condition count: `{fano_a3_prereg.get('mlp_only_or_partial_condition_count')}`",
            "",
            "| source | budget | SNR | missing | drift | status | passed baselines |",
            "|---|---:|---:|---:|---:|---|---|",
        ])
        for row in fano_a3_gate_rows:
            lines.append(
                "| "
                + " | ".join([
                    str(row.get("source_label")),
                    str(row.get("observation_budget")),
                    _fmt(row.get("noise_snr_db")),
                    _fmt(row.get("missing_fraction")),
                    _fmt(row.get("frequency_drift_ppm")),
                    str(row.get("condition_status")),
                    ",".join(row.get("passed_comparison_families", [])),
                ])
                + " |"
            )
    fano_reliability_gate = summary.get("fano_reliability_claim_gate", {})
    if fano_reliability_gate:
        lines.extend([
            "",
            "## Microwave Fano Reliability Claim Gate",
            "",
            f"Claim status: `{fano_reliability_gate.get('claim_status')}`",
            f"Ordinary curve reliability pass: `{fano_reliability_gate.get('ordinary_curve_reliability_pass')}`",
            f"Full reliability claim pass: `{fano_reliability_gate.get('full_reliability_claim_pass')}`",
            f"Accepted failure rate: `{_fmt(fano_reliability_gate.get('accepted_failure_rate'))}`",
            f"Accepted median joint coverage: `{_fmt(fano_reliability_gate.get('accepted_median_joint_coverage'))}`",
            f"Accepted median dangerous failure rate: `{_fmt(fano_reliability_gate.get('accepted_median_dangerous_failure_rate'))}`",
            f"Failure capture rate: `{_fmt(fano_reliability_gate.get('failure_capture_rate'))}`",
            f"High-risk curves: `{','.join(fano_reliability_gate.get('highrisk_curve_ids', []))}`",
            f"post-policy failure rate: `{_fmt(fano_reliability_gate.get('post_policy_failure_rate'))}`",
            "",
        ])
    descriptor_strata = summary.get("descriptor_strata_summary", {})
    stratum_rows = descriptor_strata.get("stratum_rows", [])
    if stratum_rows:
        lines.extend([
            "",
            "## Descriptor Strata Summary",
            "",
            "| domain | axis | layer | stratum | rows | median CFNN effect | positive | negative | median boundary | example |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ])
        ranked_strata = sorted(
            stratum_rows,
            key=lambda row: (
                row.get("median_boundary_score") is None,
                -(row.get("median_boundary_score") or -1.0),
                row.get("domain", ""),
                row.get("descriptor_stratum", ""),
            ),
        )
        for row in ranked_strata[:20]:
            lines.append(
                "| "
                + " | ".join([
                    str(row.get("domain")),
                    str(row.get("evidence_axis")),
                    str(row.get("effect_layer")),
                    str(row.get("descriptor_stratum")),
                    str(row.get("row_count")),
                    _fmt(row.get("median_cfnn_effect")),
                    str(row.get("positive_effect_count")),
                    str(row.get("negative_effect_count")),
                    _fmt(row.get("median_boundary_score")),
                    f"`{row.get('example_task_id')}`",
                ])
                + " |"
            )
    microstrip_decisions = summary.get("microstrip_guardrail_decision_table", {})
    decision_rows = microstrip_decisions.get("decision_rows", [])
    if decision_rows:
        lines.extend([
            "",
            "## Microstrip Guardrail Decision Table",
            "",
            f"Claim level: `{microstrip_decisions.get('claim_level')}`",
            "",
            "| decision | claim scope | rows | median CFNN effect | positive | negative | median boundary | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for row in decision_rows:
            lines.append(
                "| "
                + " | ".join([
                    f"`{row.get('decision_id')}`",
                    str(row.get("claim_scope")),
                    str(row.get("row_count")),
                    _fmt(row.get("median_cfnn_effect")),
                    str(row.get("positive_effect_count")),
                    str(row.get("negative_effect_count")),
                    _fmt(row.get("median_boundary_score")),
                    str(row.get("verdict")),
                ])
                + " |"
            )
    digital_twin_gate = summary.get("microstrip_digital_twin_claim_gate", {})
    digital_twin_rows = digital_twin_gate.get("gate_rows", [])
    if digital_twin_gate:
        lines.extend([
            "",
            "## Microstrip Digital Twin Claim Gate",
            "",
            f"Claim status: `{digital_twin_gate.get('claim_status')}`",
            f"Localized Q signal: `{digital_twin_gate.get('localized_q_signal_status')}`",
            f"Failed layers: `{','.join(digital_twin_gate.get('failed_effect_layers', []))}`",
            f"Blocking guardrail decisions: `{','.join(digital_twin_gate.get('blocking_guardrail_decisions', []))}`",
            f"Blocking confirmatory strata: `{','.join(digital_twin_gate.get('blocking_confirmatory_strata', []))}`",
            "",
            "| layer | status | global | peak | f0 | Q | phase | failed metrics |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for row in digital_twin_rows:
            effects = row.get("metric_effects", {})
            lines.append(
                "| "
                + " | ".join([
                    f"`{row.get('effect_layer')}`",
                    str(row.get("layer_status")),
                    _fmt(effects.get("complex_nrmse")),
                    _fmt(effects.get("peak_window_complex_nrmse")),
                    _fmt(effects.get("resonance_frequency_abs_error_hz")),
                    _fmt(effects.get("quality_factor_relative_error")),
                    _fmt(effects.get("phase_transition_abs_error_hz")),
                    ",".join(row.get("failed_metrics", []) or row.get("missing_metrics", [])),
                ])
                + " |"
            )
    confirmatory = summary.get("microstrip_confirmatory_guardrail_strata", {})
    confirmatory_rows = confirmatory.get("stratum_rows", [])
    if confirmatory_rows:
        lines.extend([
            "",
            "## Microstrip Confirmatory Guardrail Strata",
            "",
            f"Primary claim gate: `{confirmatory.get('primary_claim_gate')}`",
            "",
            "| stratum | role | calibration counts | Q+/f0- rate | Q effect | f0 effect | phase effect | provisional status |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ])
        for row in confirmatory_rows:
            calibration_counts = ",".join(str(value) for value in row.get("calibration_frequency_counts", []))
            lines.append(
                "| "
                + " | ".join([
                    f"`{row.get('stratum_id')}`",
                    str(row.get("stratum_role")),
                    f"`{calibration_counts}`",
                    _fmt(row.get("pilot_q_positive_f0_negative_rate")),
                    _fmt(row.get("pilot_median_quality_factor_effect")),
                    _fmt(row.get("pilot_median_f0_effect")),
                    _fmt(row.get("pilot_global_missed_band_phase_effect")),
                    str(row.get("provisional_status")),
                ])
                + " |"
            )
    eis_edge_audit = summary.get("eis_neural_positive_edge_audit", {})
    eis_edge_rows = eis_edge_audit.get("positive_edge_rows", [])
    if eis_edge_rows:
        lines.extend([
            "",
            "## EIS Neural-Only Positive Edge Audit",
            "",
            f"Positive neural-only edge count: `{eis_edge_audit.get('positive_neural_edge_count')}`",
            "",
            "| status | count |",
            "|---|---:|",
        ])
        for status, count in sorted(eis_edge_audit.get("status_counts", {}).items()):
            lines.append(f"| `{status}` | {count} |")
        lines.extend([
            "",
            "| task | status | neural effect | physical-prior effect | stratum |",
            "|---|---|---:|---:|---|",
        ])
        for row in eis_edge_rows:
            lines.append(
                "| "
                + " | ".join([
                    f"`{row.get('task_id')}`",
                    str(row.get("edge_status")),
                    _fmt(row.get("neural_effect")),
                    _fmt(row.get("physical_prior_effect")),
                    str(row.get("descriptor_stratum")),
                ])
                + " |"
            )
    eis_claim_gate = summary.get("eis_applicability_claim_gate", {})
    if eis_claim_gate:
        lines.extend([
            "",
            "## EIS Applicability Claim Gate",
            "",
            f"Claim status: `{eis_claim_gate.get('claim_status')}`",
            f"general CFNN advantage pass: `{eis_claim_gate.get('general_cfnn_advantage_pass')}`",
            f"All-baseline effects: `{eis_claim_gate.get('all_baseline_count')}` rows, "
            f"`{eis_claim_gate.get('all_baseline_negative_count')}` negative, "
            f"median `{_fmt(eis_claim_gate.get('all_baseline_median_effect'))}`",
            f"Neural-only effects: `{eis_claim_gate.get('neural_only_count')}` rows, "
            f"`{eis_claim_gate.get('neural_only_positive_count')}` positive, "
            f"median `{_fmt(eis_claim_gate.get('neural_only_median_effect'))}`",
            f"Physical-prior effects: `{eis_claim_gate.get('physical_prior_count')}` rows, "
            f"`{eis_claim_gate.get('physical_prior_negative_count')}` negative, "
            f"median `{_fmt(eis_claim_gate.get('physical_prior_median_effect'))}`",
            f"physical-prior eliminated neural edges: `{eis_claim_gate.get('physical_prior_eliminated_edge_count')}` "
            f"of `{eis_claim_gate.get('neural_positive_edge_count')}`",
            "",
        ])
    aluminium_value_audit = summary.get("aluminium_active_measurement_value_audit", {})
    aluminium_decision_rows = aluminium_value_audit.get("decision_rows", [])
    if aluminium_decision_rows:
        lines.extend([
            "",
            "## Aluminium Active Measurement Value Audit",
            "",
            f"Claim level: `{aluminium_value_audit.get('claim_level')}`",
            "",
            "| decision | scope | family | budget | passed modes | verdict |",
            "|---|---|---|---:|---:|---|",
        ])
        for row in aluminium_decision_rows:
            lines.append(
                "| "
                + " | ".join([
                    f"`{row.get('decision_id')}`",
                    str(row.get("evidence_scope")),
                    str(row.get("family")),
                    str(row.get("minimum_budget", row.get("budget"))),
                    _fmt(row.get("median_passed_modes", row.get("median_truth_aligned_half_power_passed_mode_count"))),
                    str(row.get("verdict")),
                ])
                + " |"
            )
    aluminium_modal_gate = summary.get("aluminium_active_modal_claim_gate", {})
    if aluminium_modal_gate:
        lines.extend([
            "",
            "## Aluminium Active Modal Claim Gate",
            "",
            f"Claim status: `{aluminium_modal_gate.get('claim_status')}`",
            f"Measurement minimum budget: `{aluminium_modal_gate.get('measurement_strategy_minimum_budget')}`",
            f"Measurement passed modes: `{_fmt(aluminium_modal_gate.get('measurement_strategy_passed_modes'))}`",
            f"Same-schedule CFNN passed modes: `{_fmt(aluminium_modal_gate.get('same_schedule_cfnn_passed_modes'))}`",
            f"Candidate discovery status: `{aluminium_modal_gate.get('candidate_discovery_status')}`",
            "",
        ])
    lines.extend([
        "",
        "## Highest Boundary Rows",
        "",
        "| rank | domain | task | layer | metric | CFNN effect | coverage | dangerous | boundary score | phase turns | curvature | peaks/poles | baseline |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    ranked = sorted(rows, key=lambda row: (row.get("boundary_score") is None, -(row.get("boundary_score") or -1.0)))
    for index, row in enumerate(ranked[:15], start=1):
        peaks = row.get("relaxation_peak_count")
        poles = row.get("known_pole_count")
        peak_pole = "NA" if peaks is None and poles is None else f"{peaks}/{poles}"
        lines.append(
            "| "
            + " | ".join([
                str(index),
                row["domain"],
                f"`{row['task_id']}`",
                str(row.get("effect_layer")),
                str(row.get("metric")),
                _fmt(row.get("cfnn_effect")),
                _fmt(row.get("cross_curve_joint_coverage")),
                _fmt(row.get("dangerous_failure_rate")),
                _fmt(row.get("boundary_score")),
                _fmt(row.get("phase_total_turns")),
                _fmt(row.get("max_curvature")),
                peak_pole,
                str(row.get("best_baseline_model")),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Key Layer Representatives",
        "",
        "| group | representative task | reason | value |",
        "|---|---|---|---:|",
    ])
    fano_rows = [row for row in rows if row["domain"] == "Microwave Fano"]
    if fano_rows:
        fano_accuracy_rows = [row for row in fano_rows if row.get("evidence_axis") == "accuracy_effect"]
        if fano_accuracy_rows:
            item = max(fano_accuracy_rows, key=lambda row: row.get("cfnn_effect") or -1e9)
            lines.append(f"| Microwave Fano A2 positive effect | `{item['task_id']}` | largest signed CFNN recovery effect | {_fmt(item.get('cfnn_effect'))} |")
        fano_robustness_rows = [row for row in fano_rows if row.get("evidence_axis") == "robustness_effect"]
        if fano_robustness_rows:
            item = max(fano_robustness_rows, key=lambda row: row.get("cfnn_effect") or -1e9)
            lines.append(f"| Microwave Fano A3 positive robustness | `{item['task_id']}` | largest signed CFNN robustness effect | {_fmt(item.get('cfnn_effect'))} |")
            item = min(fano_robustness_rows, key=lambda row: row.get("cfnn_effect") or 1e9)
            lines.append(f"| Microwave Fano A3 robustness boundary | `{item['task_id']}` | most negative CFNN robustness effect | {_fmt(item.get('cfnn_effect'))} |")
        fano_reliability_rows = [row for row in fano_rows if row.get("evidence_axis") == "reliability_boundary"]
        item = max(fano_reliability_rows or fano_rows, key=lambda row: row.get("boundary_score") or -1.0)
        lines.append(f"| Microwave Fano reliability | `{item['task_id']}` | highest reliability boundary score | {_fmt(item.get('boundary_score'))} |")
        fano_conformal_rows = [
            row for row in fano_reliability_rows
            if str(row.get("effect_layer", "")).startswith("prediction_level_")
        ]
        if fano_conformal_rows:
            item = max(fano_conformal_rows, key=lambda row: row.get("boundary_score") or -1.0)
            lines.append(f"| Microwave Fano conformal boundary | `{item['task_id']}` | highest prediction-level split-conformal boundary score | {_fmt(item.get('boundary_score'))} |")
    microstrip_rows = [row for row in rows if row["domain"] == "Microstrip Resonator"]
    if microstrip_rows:
        item = max(microstrip_rows, key=lambda row: row.get("cfnn_effect") or -1e9)
        lines.append(f"| Microstrip B positive device effect | `{item['task_id']}` | largest signed CFNN device-generalization effect | {_fmt(item.get('cfnn_effect'))} |")
        item = min(microstrip_rows, key=lambda row: row.get("cfnn_effect") or 1e9)
        lines.append(f"| Microstrip B device boundary | `{item['task_id']}` | most negative CFNN device-generalization effect | {_fmt(item.get('cfnn_effect'))} |")
        missed_band_rows = [
            row for row in microstrip_rows
            if row.get("evidence_axis") == "device_calibration_negative_control_effect"
        ]
        missed_band_q_rows = [
            row for row in missed_band_rows
            if row.get("metric") == "quality_factor_relative_error"
        ]
        if missed_band_q_rows:
            item = max(missed_band_q_rows, key=lambda row: row.get("cfnn_effect") or -1e9)
            lines.append(f"| Microstrip B missed-band Q control | `{item['task_id']}` | largest Q effect when calibration avoids f0 ± Gamma | {_fmt(item.get('cfnn_effect'))} |")
        if missed_band_rows:
            item = min(missed_band_rows, key=lambda row: row.get("cfnn_effect") or 1e9)
            lines.append(f"| Microstrip B missed-band guardrail | `{item['task_id']}` | weakest missed-band negative-control effect | {_fmt(item.get('cfnn_effect'))} |")
        f0_descriptor_rows = [
            row for row in microstrip_rows
            if row.get("evidence_axis") == "device_f0_guardrail_descriptor"
        ]
        if f0_descriptor_rows:
            item = max(f0_descriptor_rows, key=lambda row: row.get("boundary_score") or -1.0)
            lines.append(f"| Microstrip B f0 descriptor guardrail | `{item['task_id']}` | highest Q+/f0- risk stratum | {_fmt(item.get('boundary_score'))} |")
    aluminium_rows = [row for row in rows if row["domain"] == "Aluminium FRF"]
    if aluminium_rows:
        item = max(aluminium_rows, key=lambda row: row.get("boundary_score") or -1.0)
        lines.append(f"| Aluminium C active modal boundary | `{item['task_id']}` | largest CFNN modal-recovery boundary score | {_fmt(item.get('boundary_score'))} |")
    neural_rows = [row for row in rows if row["domain"] == "EIS" and row.get("effect_layer") == "neural_only"]
    if neural_rows:
        item = max(neural_rows, key=lambda row: row.get("cfnn_effect") or -1e9)
        lines.append(f"| EIS neural-only positive edge | `{item['task_id']}` | largest positive CFNN effect | {_fmt(item.get('cfnn_effect'))} |")
        item = min(neural_rows, key=lambda row: row.get("cfnn_effect") or 1e9)
        lines.append(f"| EIS neural-only boundary | `{item['task_id']}` | most negative CFNN effect vs neural baseline | {_fmt(item.get('cfnn_effect'))} |")
    physical_rows = [row for row in rows if row["domain"] == "EIS" and row.get("effect_layer") == "physical_prior_only"]
    if physical_rows:
        item = min(physical_rows, key=lambda row: row.get("cfnn_effect") or 1e9)
        lines.append(f"| EIS physical-prior boundary | `{item['task_id']}` | strongest circuit-prior advantage | {_fmt(item.get('cfnn_effect'))} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The unified table supports a bounded claim rather than a universal CFNN claim. Microwave Fano contributes positive A2 sparse recovery effects and A3 robustness effects against MLP in many noise/missing cells, but the local nested CF control and severe 10 dB settings expose guardrails. Fano D/E adds a reliability boundary: weak phase turnover and high local curvature can break cross-curve calibration even when a Fano reference fit is acceptable; prediction-level split conformal substantially repairs ordinary Fano curves, but the same r8:p03 spectrum remains under-covered and dangerous. Microstrip B shows mixed real-device evidence: CFNN helps Q/linewidth cells, and the missed-band negative control shows that this Q signal is not just calibration points landing inside f0 ± Gamma; however it still does not dominate global response, temperature holdout, f0, or phase events. The added f0 descriptor guardrail further separates edge_proximal_missed_band curves, where Q gains can co-occur with f0 failures, from edge_intermediate_missed_band curves, where this failure mode is absent in the current seed expansion. Aluminium C currently contributes a negative active-experiment boundary: the present CFNN active modal pipeline does not recover matched modes. EIS contributes an applicability boundary: against neural baselines CFNN has only local positive cells, and against explicit circuit priors it is consistently worse in the analyzed cells. The descriptor strata table makes this interpretation auditable by tying each row to weak-phase/high-curvature Fano boundaries, distributed or low-order EIS relaxation, device-condition responses, missed-band calibration controls, f0 descriptor guardrails, or active modal boundaries.",
        "",
        "The current cross-task hypothesis is therefore: CFNN is most defensible on sparse, resonant complex responses when the scientific event is narrowband and identifiable; it is not a substitute for task-specific physical priors in broad-band EIS or modal-analysis pipelines, and it needs explicit calibration and task-level event checks before reliability claims.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fano-e-summary", type=Path, default=DEFAULT_FANO_E)
    parser.add_argument("--eis-e-summary", type=Path, default=DEFAULT_EIS_E)
    parser.add_argument("--fano-a2-enhanced-summary", type=Path, default=DEFAULT_FANO_A2)
    parser.add_argument("--fano-a2-preregistered-success-audit", type=Path, default=DEFAULT_FANO_A2_PREREGISTERED_AUDIT)
    parser.add_argument("--fano-a3-preregistered-robustness-audit", type=Path, default=DEFAULT_FANO_A3_PREREGISTERED_AUDIT)
    parser.add_argument("--microstrip-b-analysis-summary", type=Path, default=DEFAULT_MICROSTRIP_B)
    parser.add_argument("--microstrip-b-missed-band-summary", type=Path, default=DEFAULT_MICROSTRIP_B_MISSED_BAND)
    parser.add_argument("--microstrip-b-f0-descriptor-summary", type=Path, default=DEFAULT_MICROSTRIP_B_F0_DESCRIPTOR)
    parser.add_argument("--aluminium-c-analysis-summary", type=Path, default=DEFAULT_ALUMINIUM_C)
    parser.add_argument("--eis-neural-positive-edge-audit", type=Path, default=DEFAULT_EIS_NEURAL_EDGE_AUDIT)
    parser.add_argument("--aluminium-active-measurement-audit", type=Path, default=DEFAULT_ALUMINIUM_ACTIVE_MEASUREMENT_AUDIT)
    parser.add_argument(
        "--fano-a3-summary",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        default=None,
        help="Append an A3 robustness summary as LABEL PATH. Defaults to noise/missing/drift production summaries.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fano = _load(args.fano_e_summary)
    eis = _load(args.eis_e_summary)
    a2 = _load(args.fano_a2_enhanced_summary) if args.fano_a2_enhanced_summary.exists() else None
    fano_a2_preregistered_audit = (
        _load(args.fano_a2_preregistered_success_audit)
        if args.fano_a2_preregistered_success_audit.exists()
        else None
    )
    fano_a3_preregistered_audit = (
        _load(args.fano_a3_preregistered_robustness_audit)
        if args.fano_a3_preregistered_robustness_audit.exists()
        else None
    )
    a3_paths = [(label, Path(path)) for label, path in args.fano_a3_summary] if args.fano_a3_summary else DEFAULT_FANO_A3_SUMMARIES
    a3_summaries = [(label, _load(path)) for label, path in a3_paths if path.exists()]
    microstrip = _load(args.microstrip_b_analysis_summary) if args.microstrip_b_analysis_summary.exists() else None
    microstrip_missed_band = (
        _load(args.microstrip_b_missed_band_summary)
        if args.microstrip_b_missed_band_summary.exists()
        else None
    )
    microstrip_f0_descriptor = (
        _load(args.microstrip_b_f0_descriptor_summary)
        if args.microstrip_b_f0_descriptor_summary.exists()
        else None
    )
    aluminium = _load(args.aluminium_c_analysis_summary) if args.aluminium_c_analysis_summary.exists() else None
    eis_neural_edge_audit = (
        _load(args.eis_neural_positive_edge_audit)
        if args.eis_neural_positive_edge_audit.exists()
        else None
    )
    aluminium_active_measurement_audit = (
        _load(args.aluminium_active_measurement_audit)
        if args.aluminium_active_measurement_audit.exists()
        else None
    )
    rows = build_unified_applicability_rows(
        fano,
        eis,
        a2,
        a3_summaries,
        microstrip,
        microstrip_missed_band,
        microstrip_f0_descriptor,
        aluminium,
    )
    output = {
        "analysis_type": "package_e_unified_applicability_map",
        "source_fano_summary": str(args.fano_e_summary),
        "source_eis_summary": str(args.eis_e_summary),
        "source_fano_a2_summary": str(args.fano_a2_enhanced_summary) if a2 is not None else None,
        "source_fano_a2_preregistered_success_audit": (
            str(args.fano_a2_preregistered_success_audit)
            if fano_a2_preregistered_audit is not None else None
        ),
        "source_fano_a3_preregistered_robustness_audit": (
            str(args.fano_a3_preregistered_robustness_audit)
            if fano_a3_preregistered_audit is not None else None
        ),
        "source_fano_a3_summaries": {label: str(path) for label, path in a3_paths if path.exists()},
        "source_microstrip_b_summary": str(args.microstrip_b_analysis_summary) if microstrip is not None else None,
        "source_microstrip_b_missed_band_summary": (
            str(args.microstrip_b_missed_band_summary)
            if microstrip_missed_band is not None else None
        ),
        "source_microstrip_b_f0_descriptor_summary": (
            str(args.microstrip_b_f0_descriptor_summary)
            if microstrip_f0_descriptor is not None else None
        ),
        "source_aluminium_c_summary": str(args.aluminium_c_analysis_summary) if aluminium is not None else None,
        "source_eis_neural_positive_edge_audit": (
            str(args.eis_neural_positive_edge_audit)
            if eis_neural_edge_audit is not None else None
        ),
        "source_aluminium_active_measurement_audit": (
            str(args.aluminium_active_measurement_audit)
            if aluminium_active_measurement_audit is not None else None
        ),
        "unified_rows": rows,
        "unified_summary": summarize_unified_rows(rows),
        "descriptor_strata_summary": summarize_descriptor_strata(rows),
        "microstrip_guardrail_decision_table": summarize_microstrip_guardrail_decisions(rows),
        "microstrip_confirmatory_guardrail_strata": summarize_microstrip_confirmatory_guardrail_strata(rows),
        "fano_a2_preregistered_success_audit": fano_a2_preregistered_audit,
        "fano_a3_preregistered_robustness_audit": fano_a3_preregistered_audit,
        "eis_neural_positive_edge_audit": eis_neural_edge_audit,
        "aluminium_active_measurement_value_audit": aluminium_active_measurement_audit,
    }
    output["microstrip_digital_twin_claim_gate"] = summarize_microstrip_digital_twin_claim_gate(output)
    output["aluminium_active_modal_claim_gate"] = summarize_aluminium_active_modal_claim_gate(output)
    output["eis_applicability_claim_gate"] = summarize_eis_applicability_claim_gate(output)
    output["fano_reliability_claim_gate"] = summarize_fano_reliability_claim_gate({
        **output,
        "fano_boundary_summary": fano,
    })
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(output) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "row_count": len(rows),
        "domains": sorted(output["unified_summary"]["domain_summaries"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
