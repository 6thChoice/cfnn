#!/usr/bin/env python3
"""Analyze Microwave Fano D/E boundary descriptors."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_fano_boundary_e import (
    curve_boundary_summary,
    curve_family_budget_boundary_rows,
    descriptor_guardrail_policy_audit,
    descriptor_risk_rule_audit,
    descriptor_risk_rule_sensitivity_audit,
    descriptor_same_curve_escalation_policy_audit,
    fano_spectral_descriptors,
    rank_boundary_curves,
    rank_split_conformal_boundary_curves,
    split_conformal_boundary_summary,
    split_conformal_family_budget_boundary_rows,
)
from downstream_protocol import load_microwave_fano


ROOT = Path(__file__).resolve().parent
DEFAULT_D_SUMMARY = ROOT / "ai4science_results" / "microwave_fano_d_5curve_64_128_cross_curve_pilot" / "summary.json"
DEFAULT_PREDICTION_D_SUMMARY = ROOT / "ai4science_results" / "microwave_fano_d_prediction_level_split_conformal_5curve_64_128" / "summary.json"
DEFAULT_A1_REFERENCES = ROOT / "ai4science_results" / "microwave_fano_a1" / "reference_parameters.json"
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "microwave_fano_e_boundary_descriptors_5curve_d"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _load_references(path: Path) -> dict[str, dict]:
    rows = json.loads(Path(path).read_text())
    return {row["curve_id"]: row for row in rows}


def build_boundary_report(summary: dict) -> str:
    ranked = summary["ranked_curve_rows"]
    ranked_conformal = summary.get("ranked_conformal_curve_rows", [])
    lines = [
        "# Microwave Fano D/E Boundary Descriptor Report",
        "",
        "This report joins spectrum-side descriptors with Package D cross-curve calibration outcomes. It is a boundary diagnostic, not a CFNN advantage claim.",
        "",
        "## Curve-Level Boundary Ranking",
        "",
        "| rank | curve | median cross joint | median cross peak | same/cross scale | same scale | cross scale | dangerous rate | linewidth/span | phase turns | Fano mag RMSE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join([
                str(index),
                f"`{row['curve_id']}`",
                _fmt(row.get("median_cross_curve_joint_coverage")),
                _fmt(row.get("median_cross_curve_peak_window_coverage")),
                _fmt(row.get("median_same_to_cross_scale_ratio")),
                _fmt(row.get("median_same_curve_calibration_scale")),
                _fmt(row.get("median_cross_curve_calibration_scale")),
                _fmt(row.get("median_cross_curve_dangerous_point_failure_rate")),
                _fmt(row.get("linewidth_fraction"), digits=4),
                _fmt(row.get("phase_total_turns")),
                _fmt(row.get("fano_relative_magnitude_rmse"), digits=4),
            ])
            + " |"
        )

    if ranked_conformal:
        lines.extend([
            "",
            "## Prediction-Level Split Conformal Boundary Ranking",
            "",
            "| rank | curve | layer | median joint | median peak | scale | dangerous rate | boundary score | phase turns | curvature |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for index, row in enumerate(ranked_conformal, start=1):
            lines.append(
                "| "
                + " | ".join([
                    str(index),
                    f"`{row['curve_id']}`",
                    f"`{row['conformal_layer']}`",
                    _fmt(row.get("median_calibrated_joint_coverage")),
                    _fmt(row.get("median_calibrated_peak_window_coverage")),
                    _fmt(row.get("median_calibration_scale")),
                    _fmt(row.get("median_calibrated_dangerous_point_failure_rate")),
                    _fmt(row.get("median_boundary_score")),
                    _fmt(row.get("phase_total_turns")),
                    _fmt(row.get("max_normalized_magnitude_curvature")),
                ])
                + " |"
            )

    risk_audits = summary.get("descriptor_risk_rule_audits", [])
    if risk_audits:
        lines.extend([
            "",
            "## Descriptor Risk Rule Audit",
            "",
            "| layer | rule | phase threshold | curvature threshold | TP | FP | TN | FN | precision | recall | specificity |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for audit in risk_audits:
            confusion = audit.get("confusion", {})
            rule = audit.get("rule", {})
            lines.append(
                "| "
                + " | ".join([
                    f"`{audit.get('conformal_layer')}`",
                    f"`{rule.get('risk_stratum')}`",
                    _fmt(rule.get("phase_turn_threshold")),
                    _fmt(rule.get("curvature_threshold")),
                    str(confusion.get("true_positive", 0)),
                    str(confusion.get("false_positive", 0)),
                    str(confusion.get("true_negative", 0)),
                    str(confusion.get("false_negative", 0)),
                    _fmt(audit.get("precision")),
                    _fmt(audit.get("recall")),
                    _fmt(audit.get("specificity")),
                ])
                + " |"
            )

    guardrail_audits = summary.get("descriptor_guardrail_policy_audits", [])
    if guardrail_audits:
        lines.extend([
            "",
            "## Descriptor Guardrail Policy Audit",
            "",
            "| layer | policy | accepted | rejected | rejection rate | failure capture | accepted failure | accepted joint | rejected joint |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for audit in guardrail_audits:
            lines.append(
                "| "
                + " | ".join([
                    f"`{audit.get('conformal_layer')}`",
                    f"`{audit.get('policy')}`",
                    str(audit.get("accepted_curve_count")),
                    str(audit.get("rejected_curve_count")),
                    _fmt(audit.get("rejection_rate")),
                    _fmt(audit.get("failure_capture_rate")),
                    _fmt(audit.get("accepted_failure_rate")),
                    _fmt(audit.get("accepted_median_joint_coverage")),
                    _fmt(audit.get("rejected_median_joint_coverage")),
                ])
                + " |"
            )

    sensitivity_audits = summary.get("descriptor_risk_rule_sensitivity_audits", [])
    if sensitivity_audits:
        lines.extend([
            "",
            "## Descriptor Rule Sensitivity Audit",
            "",
            "| layer | grid | exact isolating | exact fraction | zero-failure accepted | best phase q | best curvature q | best rejected |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for audit in sensitivity_audits:
            best = audit.get("best_rule", {})
            lines.append(
                "| "
                + " | ".join([
                    f"`{audit.get('conformal_layer')}`",
                    str(audit.get("grid_count")),
                    str(audit.get("exact_isolating_rule_count")),
                    _fmt(audit.get("exact_isolating_rule_fraction")),
                    str(audit.get("accepted_zero_failure_rule_count")),
                    _fmt(best.get("phase_turn_quantile")),
                    _fmt(best.get("curvature_quantile")),
                    ", ".join(f"`{curve_id}`" for curve_id in best.get("rejected_curve_ids", [])),
                ])
                + " |"
            )

    escalation_audits = summary.get("descriptor_same_curve_escalation_policy_audits", [])
    if escalation_audits:
        lines.extend([
            "",
            "## Same-Curve Escalation Policy Audit",
            "",
            "| layer | policy | accepted | escalated | escalation rate | post-policy failure | post-policy joint | escalated same-curve joint |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for audit in escalation_audits:
            lines.append(
                "| "
                + " | ".join([
                    f"`{audit.get('conformal_layer')}`",
                    f"`{audit.get('policy')}`",
                    str(audit.get("accepted_curve_count")),
                    str(audit.get("escalated_curve_count")),
                    _fmt(audit.get("escalation_rate")),
                    _fmt(audit.get("post_policy_failure_rate")),
                    _fmt(audit.get("post_policy_median_joint_coverage")),
                    _fmt(audit.get("escalated_median_same_curve_joint_coverage")),
                ])
                + " |"
            )

    hardest = ranked[0] if ranked else None
    hardest_conformal = ranked_conformal[0] if ranked_conformal else None
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if hardest:
        lines.append(
            f"The strongest boundary case is `{hardest['curve_id']}`. Its median cross-curve joint coverage is "
            f"{_fmt(hardest.get('median_cross_curve_joint_coverage'))}, while its median same-curve calibration scale is "
            f"{_fmt(hardest.get('median_same_curve_calibration_scale'))} versus a cross-curve scale of "
            f"{_fmt(hardest.get('median_cross_curve_calibration_scale'))}. This identifies a calibration-scale mismatch rather than merely a raw ensemble-spread issue."
        )
    if hardest_conformal:
        lines.append(
            f"The strongest prediction-level conformal boundary is `{hardest_conformal['curve_id']}` under "
            f"`{hardest_conformal['conformal_layer']}`. Its median calibrated joint coverage is "
            f"{_fmt(hardest_conformal.get('median_calibrated_joint_coverage'))} and its median dangerous failure rate is "
            f"{_fmt(hardest_conformal.get('median_calibrated_dangerous_point_failure_rate'))}, showing that conservative split conformal calibration still fails on this spectrum-side boundary."
        )
    if risk_audits:
        best_audit = risk_audits[0]
        rule = best_audit.get("rule", {})
        lines.append(
            f"A simple descriptor rule, `{rule.get('risk_stratum')}`, identifies curves with phase turns below "
            f"{_fmt(rule.get('phase_turn_threshold'))} and normalized curvature above "
            f"{_fmt(rule.get('curvature_threshold'))}. In the current five-curve pilot it isolates the observed conformal boundary, but this is a hypothesis-generating rule rather than a confirmed statistical classifier."
        )
    if guardrail_audits:
        guardrail = guardrail_audits[0]
        lines.append(
            f"As a guardrail policy, accepting ordinary curves and rejecting `weak_phase_high_curvature` curves yields an accepted failure rate of "
            f"{_fmt(guardrail.get('accepted_failure_rate'))} with rejection rate "
            f"{_fmt(guardrail.get('rejection_rate'))}. Rejected curves should receive same-curve calibration or manual review rather than ordinary cross-curve conformal intervals."
        )
    if sensitivity_audits:
        sensitivity = sensitivity_audits[0]
        lines.append(
            f"A quantile sensitivity sweep found {sensitivity.get('exact_isolating_rule_count')} exact-isolating rules out of "
            f"{sensitivity.get('grid_count')} tested threshold pairs for `{sensitivity.get('conformal_layer')}`. This checks whether the descriptor guardrail depends on a single hand-picked threshold."
        )
    if escalation_audits:
        escalation = escalation_audits[0]
        lines.append(
            f"When rejected curves are escalated to same-curve calibration, the post-policy median joint coverage is "
            f"{_fmt(escalation.get('post_policy_median_joint_coverage'))} and the post-policy failure rate is "
            f"{_fmt(escalation.get('post_policy_failure_rate'))}. This audit separates automatic acceptance from cases that need additional curve-specific evidence."
        )
    lines.extend([
        "",
        "Package D therefore supplies a concrete reliability boundary for the AI4Science plan: cross-curve calibration cannot yet be treated as globally transferable across Fano regimes. The current descriptor rule should be treated as a preregisterable boundary hypothesis that needs more curves or split seeds before it is used as a confirmatory classifier.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument("--d-summary", type=Path, default=DEFAULT_D_SUMMARY)
    parser.add_argument("--prediction-d-summary", type=Path, default=DEFAULT_PREDICTION_D_SUMMARY)
    parser.add_argument("--a1-reference-json", type=Path, default=DEFAULT_A1_REFERENCES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d_summary = json.loads(Path(args.d_summary).read_text())
    prediction_d_summary = json.loads(Path(args.prediction_d_summary).read_text()) if Path(args.prediction_d_summary).exists() else None
    references = _load_references(args.a1_reference_json)
    curves = {curve.curve_id: curve for curve in load_microwave_fano(args.raw_root)}
    curve_ids = []
    for curve_id in list(d_summary["curve_ids"]) + list((prediction_d_summary or {}).get("curve_ids", [])):
        if curve_id not in curve_ids:
            curve_ids.append(curve_id)
    descriptors = {
        curve_id: fano_spectral_descriptors(
            curve_id,
            curves[curve_id].frequency,
            curves[curve_id].response,
            reference=references.get(curve_id),
        )
        for curve_id in curve_ids
    }
    family_budget_rows = curve_family_budget_boundary_rows(d_summary, descriptors)
    curve_rows = curve_boundary_summary(family_budget_rows)
    ranked_curve_rows = rank_boundary_curves(curve_rows)
    split_conformal_family_budget_rows = (
        split_conformal_family_budget_boundary_rows(prediction_d_summary, descriptors)
        if prediction_d_summary is not None
        else []
    )
    split_conformal_curve_rows = split_conformal_boundary_summary(split_conformal_family_budget_rows)
    ranked_conformal_curve_rows = rank_split_conformal_boundary_curves(split_conformal_curve_rows)
    descriptor_risk_rule_audits = [
        descriptor_risk_rule_audit(
            split_conformal_curve_rows,
            conformal_layer=layer,
            confidence=float((prediction_d_summary or d_summary).get("confidence", 0.80)),
        )
        for layer in ("global_split_conformal", "curve_stratified_split_conformal")
    ]
    descriptor_guardrail_policy_audits = [
        descriptor_guardrail_policy_audit(audit)
        for audit in descriptor_risk_rule_audits
    ]
    descriptor_risk_rule_sensitivity_audits = [
        descriptor_risk_rule_sensitivity_audit(
            split_conformal_curve_rows,
            conformal_layer=layer,
            confidence=float((prediction_d_summary or d_summary).get("confidence", 0.80)),
            phase_turn_quantiles=(0.10, 0.20, 0.30, 0.40, 0.50),
            curvature_quantiles=(0.50, 0.60, 0.70, 0.80, 0.90),
        )
        for layer in ("global_split_conformal", "curve_stratified_split_conformal")
    ]
    same_curve_source_rows = (
        curve_boundary_summary(curve_family_budget_boundary_rows(prediction_d_summary, descriptors))
        if prediction_d_summary is not None
        else curve_rows
    )
    descriptor_same_curve_escalation_policy_audits = [
        descriptor_same_curve_escalation_policy_audit(
            audit,
            same_curve_source_rows,
            confidence=float((prediction_d_summary or d_summary).get("confidence", 0.80)),
        )
        for audit in descriptor_risk_rule_audits
    ]
    output = {
        "analysis_type": "microwave_fano_e_boundary_descriptors",
        "source_d_summary": str(args.d_summary),
        "source_prediction_d_summary": str(args.prediction_d_summary) if prediction_d_summary is not None else None,
        "curve_ids": curve_ids,
        "descriptor_rows": [descriptors[curve_id] for curve_id in curve_ids],
        "family_budget_boundary_rows": family_budget_rows,
        "curve_rows": curve_rows,
        "ranked_curve_rows": ranked_curve_rows,
        "split_conformal_family_budget_boundary_rows": split_conformal_family_budget_rows,
        "split_conformal_curve_rows": split_conformal_curve_rows,
        "ranked_conformal_curve_rows": ranked_conformal_curve_rows,
        "descriptor_risk_rule_audits": descriptor_risk_rule_audits,
        "descriptor_guardrail_policy_audits": descriptor_guardrail_policy_audits,
        "descriptor_risk_rule_sensitivity_audits": descriptor_risk_rule_sensitivity_audits,
        "descriptor_same_curve_escalation_policy_audits": descriptor_same_curve_escalation_policy_audits,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_boundary_report(output) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "curve_count": len(curve_ids),
        "family_budget_row_count": len(family_budget_rows),
        "split_conformal_family_budget_row_count": len(split_conformal_family_budget_rows),
        "hardest_curve": ranked_curve_rows[0]["curve_id"] if ranked_curve_rows else None,
        "hardest_conformal_curve": ranked_conformal_curve_rows[0]["curve_id"] if ranked_conformal_curve_rows else None,
        "descriptor_risk_rule_statuses": [audit.get("status") for audit in descriptor_risk_rule_audits],
        "descriptor_guardrail_policy_statuses": [audit.get("status") for audit in descriptor_guardrail_policy_audits],
        "descriptor_risk_rule_sensitivity_statuses": [audit.get("status") for audit in descriptor_risk_rule_sensitivity_audits],
        "descriptor_same_curve_escalation_policy_statuses": [audit.get("status") for audit in descriptor_same_curve_escalation_policy_audits],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
