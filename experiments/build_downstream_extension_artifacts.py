"""Strict validation and artifacts for append-only downstream extensions."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from build_downstream_artifacts import (
    _parameter_curve_figure,
    _write_csv,
    load_records,
    paired_comparisons,
    render_latex_macros,
    render_primary_effects,
    summarize_records,
    validate_completeness,
    validate_records,
)
from downstream_extension import load_extension, sha256_file


def _slug(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_" for character in value
    ).strip("_")


def validate_base_evidence_unchanged(base_root: Path) -> None:
    """Verify every raw input recorded by the strict base artifact build."""
    base_root = Path(base_root)
    manifest_path = base_root / "derived" / "artifact_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing strict base artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    errors = []
    for item in manifest.get("raw_inputs", []):
        path = base_root / item["path"]
        if not path.exists():
            errors.append(f"missing {item['path']}")
        elif sha256_file(path) != item["sha256"]:
            errors.append(f"base evidence hash mismatch for {item['path']}")
    if errors:
        raise ValueError("base evidence hash validation failed:\n" + "\n".join(errors[:40]))


def validate_seed_extension_records(
    base_records: list[dict],
    extension_records: list[dict],
    base_config: dict,
    inventory: dict,
    extension_config: dict,
    base_root: Path,
    *,
    tasks: list[str],
    families: list[str],
    budgets: list[int],
) -> None:
    """Require an exact, selection-bound set of added seed pairs."""
    validate_completeness(
        base_records, base_config, inventory,
        tasks=tasks, families=families, budgets=budgets,
    )
    grouped_base = defaultdict(list)
    grouped_extension = defaultdict(list)
    for record in base_records:
        grouped_base[(record["task"], record["family"], int(record["target_budget"]))].append(record)
    for record in extension_records:
        grouped_extension[(record["task"], record["family"], int(record["target_budget"]))].append(record)
    added_seeds = {int(seed) for seed in extension_config["additional_evaluation_init_seeds"]}
    expected_pairs = {
        (int(data_seed), init_seed)
        for data_seed in base_config["evaluation_data_seeds"]
        for init_seed in added_seeds
    }
    expected_protocol_hash = extension_config["protocol_sha256"]
    errors = []
    allowed = set()
    for task in tasks:
        for family in families:
            for budget in budgets:
                key = (task, family, int(budget))
                allowed.add(key)
                candidates = inventory["tasks"][task][family][str(int(budget))]
                values = grouped_extension.get(key, [])
                expected = expected_pairs if candidates else set()
                actual = {
                    (int(record["data_seed"]), int(record["init_seed"]))
                    for record in values
                }
                if len(actual) != len(values):
                    errors.append(f"duplicate extension pairs {task}/{family}/{budget}")
                base_pairs = {
                    (int(record["data_seed"]), int(record["init_seed"]))
                    for record in grouped_base.get(key, [])
                }
                if actual.intersection(base_pairs):
                    errors.append(f"seed-pair overlap {task}/{family}/{budget}")
                if actual != expected:
                    errors.append(
                        f"extension coverage {task}/{family}/{budget}: "
                        f"expected {len(expected)}, found {len(actual)}"
                    )
                if not candidates:
                    continue
                selection_path = (
                    Path(base_root) / "selections" / _slug(task)
                    / f"{_slug(family)}__{int(budget)}.json"
                )
                expected_selection_hash = sha256_file(selection_path)
                for record in values:
                    if record.get("stage") != "confirmatory_seed_extension":
                        errors.append(f"wrong extension stage {task}/{family}/{budget}")
                    if record.get("protocol_extension_sha256") != expected_protocol_hash:
                        errors.append(f"protocol extension hash mismatch {task}/{family}/{budget}")
                    if record.get("base_selection_sha256") != expected_selection_hash:
                        errors.append(f"selection hash mismatch {task}/{family}/{budget}")
                    relative_error = abs(int(record["actual_parameters"]) - int(budget)) / int(budget)
                    if relative_error > float(base_config["budget_tolerance"]):
                        errors.append(f"parameter-window mismatch {task}/{family}/{budget}")
    extras = sorted(set(grouped_extension) - allowed)
    if extras:
        errors.append(f"extension records outside frozen matrix: {extras[:5]}")
    if errors:
        raise ValueError("invalid seed extension:\n" + "\n".join(errors[:40]))


def validate_group_transfer_records(
    records: list[dict],
    config: dict,
    inventory: dict,
    *,
    tasks: list[str],
    families: list[str],
    budgets: list[int],
) -> None:
    """Require exact transfer coverage and complete-curve disjointness."""
    validate_completeness(
        records, config, inventory,
        tasks=tasks, families=families, budgets=budgets,
    )
    errors = []
    for index, record in enumerate(records):
        dataset = record.get("dataset", {})
        if dataset.get("split_mode") != "group_transfer":
            errors.append(f"record {index} lacks group_transfer split mode")
            continue
        train = set(dataset.get("train_curve_ids", []))
        validation = set(dataset.get("validation_curve_ids", []))
        test = set(dataset.get("test_curve_ids", []))
        if not train or not validation or len(test) != 1:
            errors.append(f"record {index} has incomplete curve roles")
        if train & validation or train & test or validation & test:
            errors.append(f"curve leakage in record {index}")
    if errors:
        raise ValueError("invalid group transfer:\n" + "\n".join(errors[:40]))


def collect_retry_audits(label: str, root: Path) -> list[dict]:
    """Hash explicit duplicate-retry audits without treating backups as evidence."""
    retry_root = Path(root) / "invalidated" / "retry_duplicates"
    audit_path = retry_root / "deduplication_audit.json"
    if not audit_path.exists():
        return []
    backups = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        for path in sorted(retry_root.rglob("*.jsonl"))
    ]
    return [{
        "evidence_root": label,
        "audit_path": str(audit_path.relative_to(root)),
        "audit_sha256": sha256_file(audit_path),
        "backups": backups,
    }]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--seed-extension-root", type=Path, required=True)
    parser.add_argument("--group-transfer-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--extension-config", type=Path, required=True)
    parser.add_argument("--group-transfer-config", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    base_config = json.loads(args.base_config.read_text())
    validate_base_evidence_unchanged(args.base_root)
    extension = load_extension(args.extension_config, args.base_config, args.base_root)
    extension["protocol_sha256"] = sha256_file(args.extension_config)
    group_config = json.loads(args.group_transfer_config.read_text())
    base_inventory = json.loads((args.base_root / "candidate_manifest.json").read_text())
    group_inventory = json.loads(
        (args.group_transfer_root / "candidate_manifest.json").read_text()
    )
    conventional = json.loads(
        (args.base_root / "conventional_selection.json").read_text()
    )["selected_tasks"]
    base_tasks = list(base_config["fixed_scientific_tasks"]) + list(conventional)
    base_families = list(base_config["families"])
    base_budgets = [int(value) for value in base_config["budgets"]]
    group_tasks = list(group_config["fixed_scientific_tasks"])
    group_families = list(group_config["families"])
    group_budgets = [int(value) for value in group_config["budgets"]]

    base_records = load_records(args.base_root)
    seed_records = load_records(args.seed_extension_root)
    group_records = load_records(args.group_transfer_root)
    validate_records(base_records + seed_records + group_records, args.schema)
    validate_seed_extension_records(
        base_records, seed_records, base_config, base_inventory, extension,
        args.base_root, tasks=base_tasks, families=base_families,
        budgets=base_budgets,
    )
    validate_group_transfer_records(
        group_records, group_config, group_inventory,
        tasks=group_tasks, families=group_families, budgets=group_budgets,
    )

    combined = base_records + seed_records
    derived = args.output_root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    within_summary = summarize_records(combined, int(base_config["training"]["max_steps"]))
    within_pairs = paired_comparisons(
        combined, base_config["primary_family"], int(base_config["primary_budget"])
    )
    group_summary = summarize_records(
        group_records, int(group_config["training"]["max_steps"])
    )
    group_pairs = paired_comparisons(
        group_records, group_config["primary_family"], int(group_config["primary_budget"])
    )
    outputs = {
        "within_curve_summary.json": within_summary,
        "within_curve_paired_comparisons.json": within_pairs,
        "group_transfer_summary.json": group_summary,
        "group_transfer_paired_comparisons.json": group_pairs,
    }
    for name, value in outputs.items():
        (derived / name).write_text(json.dumps(value, indent=2, sort_keys=True))
    _write_csv(derived / "within_curve_summary.csv", within_summary)
    _write_csv(derived / "within_curve_paired_comparisons.csv", within_pairs)
    _write_csv(derived / "group_transfer_summary.csv", group_summary)
    _write_csv(derived / "group_transfer_paired_comparisons.csv", group_pairs)
    (derived / "within_curve_results_macros.tex").write_text(
        render_latex_macros(within_summary, within_pairs)
    )
    (derived / "group_transfer_results_macros.tex").write_text(
        render_latex_macros(group_summary, group_pairs)
    )
    _parameter_curve_figure(
        within_summary, derived / "extended_within_curve_parameter_curves.pdf"
    )
    _parameter_curve_figure(
        group_summary, derived / "group_transfer_model_comparison.pdf", columns=3
    )
    render_primary_effects(
        within_pairs, int(base_config["primary_budget"]),
        derived / "extended_within_curve_primary_effects.pdf",
        columns=3,
    )
    render_primary_effects(
        group_pairs, int(group_config["primary_budget"]),
        derived / "group_transfer_primary_effects.pdf",
        columns=3,
    )
    raw_inputs = []
    retry_audits = []
    for label, root in (
        ("base", args.base_root),
        ("seed_extension", args.seed_extension_root),
        ("group_transfer", args.group_transfer_root),
    ):
        for path in sorted((root / "evaluation").rglob("*.jsonl")):
            raw_inputs.append({
                "evidence_root": label,
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            })
        retry_audits.extend(collect_retry_audits(label, root))
    manifest = {
        "command": __import__("sys").argv,
        "base_record_count": len(base_records),
        "seed_extension_record_count": len(seed_records),
        "within_curve_record_count": len(combined),
        "group_transfer_record_count": len(group_records),
        "base_config_sha256": sha256_file(args.base_config),
        "extension_config_sha256": sha256_file(args.extension_config),
        "group_transfer_config_sha256": sha256_file(args.group_transfer_config),
        "schema_sha256": sha256_file(args.schema),
        "raw_inputs": raw_inputs,
        "retry_duplicate_audits": retry_audits,
    }
    generated = sorted(path for path in derived.iterdir() if path.is_file())
    manifest["generated"] = [
        {"path": path.name, "sha256": sha256_file(path)}
        for path in generated if path.name != "artifact_manifest.json"
    ]
    (derived / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
