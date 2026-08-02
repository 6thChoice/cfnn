"""Budget-escalation audits for high-risk Microwave Fano reliability cases."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def _finite_float(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _passes_reliability(row: dict, *, confidence: float, danger_threshold: float) -> bool:
    joint = _finite_float(row.get("same_curve_joint_coverage"))
    danger = _finite_float(row.get("same_curve_dangerous_point_failure_rate"))
    return bool(
        joint is not None
        and danger is not None
        and joint >= float(confidence)
        and danger <= float(danger_threshold)
    )


def same_curve_budget_escalation_audit(
    summary: dict,
    *,
    target_curve_id: str,
    confidence: float | None = None,
    danger_threshold: float = 0.10,
) -> dict:
    """Summarize whether additional sparse observations repair a high-risk curve."""
    confidence_value = float(confidence if confidence is not None else summary.get("confidence", 0.80))
    rows = []
    for item in summary.get("ensemble_rows", []):
        if str(item.get("curve_id")) != str(target_curve_id):
            continue
        interval = item.get("validation_calibrated_response_interval", {})
        rows.append({
            "curve_id": str(item.get("curve_id")),
            "family": str(item.get("family")),
            "observation_budget": int(item.get("observation_budget")),
            "same_curve_joint_coverage": _finite_float(interval.get("joint_coverage")),
            "same_curve_peak_window_joint_coverage": _finite_float(interval.get("peak_window_joint_coverage")),
            "same_curve_dangerous_point_failure_rate": _finite_float(interval.get("dangerous_point_failure_rate")),
            "same_curve_calibration_scale": _finite_float(interval.get("calibration_scale")),
        })
    if not rows:
        return {
            "status": "no_rows",
            "target_curve_id": str(target_curve_id),
            "confidence": confidence_value,
            "danger_threshold": float(danger_threshold),
        }

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row["passes_reliability"] = _passes_reliability(
            row,
            confidence=confidence_value,
            danger_threshold=float(danger_threshold),
        )
        grouped[row["family"]].append(row)

    family_rows = []
    for family, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda row: int(row["observation_budget"]))
        passing = [row for row in ordered if bool(row.get("passes_reliability"))]
        minimum = int(passing[0]["observation_budget"]) if passing else None
        family_rows.append({
            "family": family,
            "status": "passes_at_budget" if minimum is not None else "no_passing_budget",
            "minimum_passing_budget": minimum,
            "budget_count": len(ordered),
            "budgets": [int(row["observation_budget"]) for row in ordered],
            "best_joint_coverage": _median(row.get("same_curve_joint_coverage") for row in passing) if passing else max(
                (_finite_float(row.get("same_curve_joint_coverage")) for row in ordered),
                default=None,
            ),
            "lowest_dangerous_point_failure_rate": min(
                (
                    value for value in (
                        _finite_float(row.get("same_curve_dangerous_point_failure_rate"))
                        for row in ordered
                    )
                    if value is not None
                ),
                default=None,
            ),
            "budget_rows": ordered,
        })

    passing_families = [
        row for row in family_rows
        if row.get("minimum_passing_budget") is not None
    ]
    best = None
    if passing_families:
        best = sorted(
            passing_families,
            key=lambda row: (
                int(row["minimum_passing_budget"]),
                str(row["family"]),
            ),
        )[0]

    budgets = sorted({int(row["observation_budget"]) for row in rows})
    return {
        "status": "ok",
        "target_curve_id": str(target_curve_id),
        "confidence": confidence_value,
        "danger_threshold": float(danger_threshold),
        "budget_count": len(budgets),
        "budgets": budgets,
        "family_count": len(family_rows),
        "family_rows": family_rows,
        "best_family_by_budget": {
            key: best.get(key)
            for key in ("family", "status", "minimum_passing_budget")
        } if best is not None else None,
    }


def multi_split_budget_escalation_summary(split_audits: Iterable[tuple[str, dict]]) -> dict:
    """Aggregate high-risk budget escalation audits across sparse split seeds."""
    audits = [(str(label), dict(audit)) for label, audit in split_audits]
    valid = [(label, audit) for label, audit in audits if audit.get("status") == "ok"]
    if not valid:
        return {
            "status": "no_ok_audits",
            "split_count": len(audits),
            "ok_split_count": 0,
        }

    family_records: dict[str, list[dict]] = defaultdict(list)
    budget_records: dict[tuple[str, int], list[dict]] = defaultdict(list)
    targets = sorted({str(audit.get("target_curve_id")) for _, audit in valid})
    budgets = sorted({
        int(budget)
        for _, audit in valid
        for budget in audit.get("budgets", [])
    })
    for split_label, audit in valid:
        for family_row in audit.get("family_rows", []):
            family = str(family_row.get("family"))
            minimum = family_row.get("minimum_passing_budget")
            family_records[family].append({
                "split_label": split_label,
                "minimum_passing_budget": int(minimum) if minimum is not None else None,
                "status": family_row.get("status"),
            })
            for budget_row in family_row.get("budget_rows", []):
                budget = int(budget_row.get("observation_budget"))
                budget_records[(family, budget)].append({
                    "split_label": split_label,
                    "passes_reliability": bool(budget_row.get("passes_reliability")),
                    "same_curve_joint_coverage": _finite_float(budget_row.get("same_curve_joint_coverage")),
                    "same_curve_dangerous_point_failure_rate": _finite_float(
                        budget_row.get("same_curve_dangerous_point_failure_rate")
                    ),
                })

    split_count = len(valid)
    family_rows = []
    for family, records in sorted(family_records.items()):
        minimums = [record["minimum_passing_budget"] for record in records]
        finite_minimums = [int(value) for value in minimums if value is not None]
        family_rows.append({
            "family": family,
            "split_count": len(records),
            "passing_split_count": len(finite_minimums),
            "split_pass_rate": float(len(finite_minimums) / len(records)) if records else None,
            "minimum_passing_budgets": minimums,
            "median_minimum_passing_budget": float(np.median(finite_minimums)) if finite_minimums else None,
            "min_minimum_passing_budget": min(finite_minimums) if finite_minimums else None,
            "max_minimum_passing_budget": max(finite_minimums) if finite_minimums else None,
            "split_rows": sorted(records, key=lambda row: str(row["split_label"])),
        })

    budget_rows = []
    for (family, budget), records in sorted(budget_records.items()):
        pass_count = sum(1 for record in records if bool(record.get("passes_reliability")))
        budget_rows.append({
            "family": family,
            "observation_budget": int(budget),
            "split_count": len(records),
            "passing_split_count": int(pass_count),
            "pass_rate": float(pass_count / len(records)) if records else None,
            "median_joint_coverage": _median(record.get("same_curve_joint_coverage") for record in records),
            "median_dangerous_point_failure_rate": _median(
                record.get("same_curve_dangerous_point_failure_rate") for record in records
            ),
            "split_rows": sorted(records, key=lambda row: str(row["split_label"])),
        })

    best = None
    passing_families = [
        row for row in family_rows
        if row.get("median_minimum_passing_budget") is not None
    ]
    if passing_families:
        best = sorted(
            passing_families,
            key=lambda row: (
                -float(row.get("split_pass_rate") or 0.0),
                float(row.get("median_minimum_passing_budget")),
                str(row.get("family")),
            ),
        )[0]

    return {
        "status": "ok",
        "target_curve_ids": targets,
        "split_count": split_count,
        "input_split_count": len(audits),
        "budgets": budgets,
        "family_count": len(family_rows),
        "family_rows": family_rows,
        "budget_rows": budget_rows,
        "best_family_by_split_pass_then_median_budget": {
            key: best.get(key)
            for key in (
                "family",
                "split_pass_rate",
                "median_minimum_passing_budget",
                "min_minimum_passing_budget",
                "max_minimum_passing_budget",
            )
        } if best is not None else None,
    }
