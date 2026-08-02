"""Validation-only selection and freezing of two conventional downstream tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _slug(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_" for character in value
    ).strip("_")


def _margin(problem_type: str, config: dict) -> float:
    key = "auroc" if problem_type == "classification" else "nrmse"
    return float(config["equivalence_margins"][key])


def _summarize(records: list[dict]) -> dict:
    valid = [
        record for record in records
        if record.get("status") == "ok" and np.isfinite(record["validation_score"])
    ]
    scores = [float(record["validation_score"]) for record in valid]
    runtimes = [float(record["training_wall_seconds"]) for record in valid]
    parameters = [int(record["actual_parameters"]) for record in valid]
    return {
        "n": len(records),
        "valid_n": len(valid),
        "failure_rate": float(1.0 - len(valid) / len(records)) if records else 1.0,
        "median_score": float(np.median(scores)) if scores else float("inf"),
        "score_iqr": float(np.subtract(*np.percentile(scores, [75, 25]))) if scores else float("inf"),
        "median_runtime_seconds": float(np.median(runtimes)) if runtimes else float("inf"),
        "median_parameters": float(np.median(parameters)) if parameters else float("inf"),
    }


def _task_evidence(task: str, task_records: list[dict], config: dict) -> dict:
    primary = config["primary_family"]
    problem_type = next(record["problem_type"] for record in task_records)
    margin = _margin(problem_type, config)
    grouped = {}
    for record in task_records:
        grouped.setdefault((int(record["target_budget"]), record["family"]), []).append(record)
    summaries = {key: _summarize(values) for key, values in grouped.items()}
    gates = []
    budgets = sorted({budget for budget, _ in grouped})
    for budget in budgets:
        cfnn_records = grouped.get((budget, primary))
        baseline_families = sorted(
            family for candidate_budget, family in grouped
            if candidate_budget == budget and family != primary
        )
        if not cfnn_records or not baseline_families:
            continue
        best_family = min(
            baseline_families,
            key=lambda family: summaries[(budget, family)]["median_score"],
        )
        cfnn_summary = summaries[(budget, primary)]
        baseline_summary = summaries[(budget, best_family)]
        by_seed_cfnn = {int(record["tuning_seed"]): record for record in cfnn_records}
        by_seed_baseline = {
            int(record["tuning_seed"]): record
            for record in grouped[(budget, best_family)]
        }
        paired_seeds = sorted(set(by_seed_cfnn) & set(by_seed_baseline))
        paired_differences = [
            float(by_seed_cfnn[seed]["validation_score"])
            - float(by_seed_baseline[seed]["validation_score"])
            for seed in paired_seeds
            if by_seed_cfnn[seed].get("status") == "ok"
            and by_seed_baseline[seed].get("status") == "ok"
        ]
        if (
            paired_differences
            and all(difference < 0 for difference in paired_differences)
            and float(np.median(paired_differences)) < -margin
        ):
            gates.append({
                "gate": "matched_budget_validation_advantage",
                "budget": budget,
                "baseline": best_family,
                "median_score_difference": float(np.median(paired_differences)),
                "paired_seed_count": len(paired_differences),
            })
        runtime_ratio = (
            cfnn_summary["median_runtime_seconds"]
            / max(baseline_summary["median_runtime_seconds"], 1e-12)
        )
        if (
            abs(cfnn_summary["median_score"] - baseline_summary["median_score"]) <= margin
            and runtime_ratio <= float(config["equivalence_margins"]["maximum_runtime_ratio"])
        ):
            gates.append({
                "gate": "matched_budget_practical_equivalence",
                "budget": budget,
                "baseline": best_family,
                "absolute_score_difference": abs(
                    cfnn_summary["median_score"] - baseline_summary["median_score"]
                ),
                "runtime_ratio": runtime_ratio,
            })

    reduction = float(config["equivalence_margins"]["parameter_reduction"])
    cfnn_entries = [
        (budget, summary) for (budget, family), summary in summaries.items()
        if family == primary and np.isfinite(summary["median_score"])
    ]
    for (baseline_budget, family), baseline_summary in summaries.items():
        if family == primary or not np.isfinite(baseline_summary["median_score"]):
            continue
        for cfnn_budget, cfnn_summary in cfnn_entries:
            if (
                cfnn_summary["median_parameters"]
                <= (1.0 - reduction) * baseline_summary["median_parameters"]
                and cfnn_summary["median_score"] <= baseline_summary["median_score"] + margin
            ):
                gates.append({
                    "gate": "validation_equivalence_with_parameter_reduction",
                    "cfnn_budget": cfnn_budget,
                    "baseline_budget": baseline_budget,
                    "baseline": family,
                    "parameter_reduction": float(
                        1.0 - cfnn_summary["median_parameters"]
                        / baseline_summary["median_parameters"]
                    ),
                })
                break
        if any(gate["gate"] == "validation_equivalence_with_parameter_reduction" for gate in gates):
            break
    cfnn_records = [record for record in task_records if record["family"] == primary]
    stability = _summarize(cfnn_records)
    sealed_hashes = sorted({
        record.get("dataset", {}).get("sealed_test_row_ids_sha256")
        for record in task_records
        if record.get("dataset", {}).get("sealed_test_row_ids_sha256")
    })
    return {
        "task": task,
        "problem_type": problem_type,
        "passed": bool(gates),
        "gates": gates,
        "stability": stability,
        "sealed_test_row_ids_sha256": sealed_hashes,
    }


def select_tasks(records: list[dict], config: dict) -> dict:
    """Apply prespecified gates without reading any test outcome."""
    candidates = list(config["conventional_candidates"])
    clean = [
        {
            key: value for key, value in record.items()
            if key not in {"test_metric", "test_metrics", "prediction"}
        }
        for record in records
        if record.get("task") in candidates
        and record.get("stage_role") == "validation_only"
    ]
    by_task = {task: [record for record in clean if record["task"] == task] for task in candidates}
    missing = [task for task, values in by_task.items() if not values]
    if missing:
        raise ValueError(f"missing validation records for tasks: {missing}")
    evidence = {task: _task_evidence(task, values, config) for task, values in by_task.items()}
    passed = [item for item in evidence.values() if item["passed"]]

    def stability_key(item):
        stability = item["stability"]
        return (stability["failure_rate"], stability["score_iqr"], item["task"])

    required = int(config["conventional_task_count"])
    selected = []
    remaining_passed = sorted(passed, key=stability_key)
    while remaining_passed and len(selected) < required:
        if not selected:
            choice = remaining_passed[0]
        else:
            existing_types = {item["problem_type"] for item in selected}
            diverse = [item for item in remaining_passed if item["problem_type"] not in existing_types]
            choice = (diverse or remaining_passed)[0]
        selected.append(choice)
        remaining_passed.remove(choice)
    if len(selected) < required:
        fallback = sorted(
            (item for item in evidence.values() if item not in selected),
            key=stability_key,
        )
        selected.extend(fallback[:required - len(selected)])
    claim_level = (
        "advantage_or_efficiency"
        if len(passed) >= required and all(item["passed"] for item in selected)
        else "compatibility_only"
    )
    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_tasks": [item["task"] for item in selected],
        "claim_level": claim_level,
        "task_evidence": evidence,
        "selection_uses": "validation_records_only",
    }


def _load_winner_records(input_root: Path) -> list[dict]:
    records = []
    for selection_path in sorted((Path(input_root) / "selections").rglob("*.json")):
        selection = json.loads(selection_path.read_text())
        if selection.get("status") != "selected":
            continue
        task = selection["task"]
        family = selection["family"]
        budget = int(selection["target_budget"])
        tuning_path = (
            Path(input_root) / "tuning_final" / _slug(task)
            / f"{_slug(family)}__{budget}.jsonl"
        )
        if not tuning_path.exists():
            candidates = list((Path(input_root) / "tuning_final").rglob(f"*__{budget}.jsonl"))
            matching = [
                path for path in candidates
                if _slug(task) in path.as_posix() and _slug(family) in path.name
            ]
            if len(matching) != 1:
                raise FileNotFoundError(
                    f"expected one tuning record for {task}/{family}/{budget}, "
                    f"found {len(matching)}"
                )
            tuning_path = matching[0]
        for line in tuning_path.read_text().splitlines():
            record = json.loads(line)
            if (
                record["candidate_key"] == selection["candidate"]["candidate_key"]
                and record["optimizer"]["key"] == selection["optimizer"]["key"]
            ):
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("downstream_config.json"))
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        manifest = json.loads(args.audit.read_text())
        if len(manifest.get("selected_tasks", [])) != 2:
            raise ValueError("selection audit requires exactly two tasks")
        if manifest.get("selection_uses") != "validation_records_only":
            raise ValueError("selection manifest lacks validation-only declaration")
        print("conventional selection audit passed")
        return
    if args.input_root is None or args.output is None:
        parser.error("--input-root and --output are required unless --audit is used")
    config = json.loads(args.config.read_text())
    records = _load_winner_records(args.input_root)
    selection = select_tasks(records, config)
    selection["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps({
        "selected_tasks": selection["selected_tasks"],
        "claim_level": selection["claim_level"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
