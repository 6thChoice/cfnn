"""Strict, provenance-bound artifacts for the frozen peak-sensitive protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib.pyplot as plt
import numpy as np

from peak_sensitive_result_validation import (
    CONFIG_PATH,
    FROZEN_CONFIG,
    FROZEN_CONFIG_SHA256,
    FROZEN_PROTOCOL_SHA256,
    PROVENANCE_ROOT,
    ROLE_MANIFEST_NAMES,
    SCHEMA_PATH,
    validate_peak_sensitive_record,
)


ROOT = Path(__file__).resolve().parent
BUILDER_PATH = Path(__file__).resolve()
VALIDATOR_PATH = ROOT / "peak_sensitive_result_validation.py"
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED_NAMESPACE = "peak-sensitive-artifacts-v2"
SIGN_FLIP_ENUMERATION_LIMIT = 20
SIGN_FLIP_MONTE_CARLO_DRAWS = 200_000
ARTIFACT_MODES = {"production", "test_fixture"}
FRF_PRIMARY_BASELINES = ("MLP", "KAN", "SIREN", "Vector fitting")
TABLE_OUTPUTS = (
    "summary.csv",
    "budget_curves.csv",
    "paired_effects.csv",
    "frf_density_interactions.csv",
    "failure_summary.csv",
    "latex_macros.tex",
)
FIGURE_PATHS = {
    "parameter_curves": "figures/parameter_curves.pdf",
    "eis_family_comparison": "figures/eis_family_comparison.pdf",
    "frf_density_interaction": "figures/frf_density_interaction.pdf",
    "paired_effects": "figures/paired_effects.pdf",
    "clipping_cap_rates": "figures/clipping_cap_rates.pdf",
    "representative_reconstructions": "figures/representative_reconstructions.pdf",
}
SOURCE_INPUTS = (
    ROOT / "peak_sensitive_results" / "provenance" / "source_audit.json",
    ROOT / "downstream_data" / "raw" / "source_manifest.json",
    ROOT / "downstream_data" / "raw" / "source_metadata_manifest.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL {path} line {line_number}: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL record must be an object: {path} line {line_number}")
        rows.append(row)
    return rows


def _seed(*parts: object) -> int:
    digest = hashlib.sha256(
        (BOOTSTRAP_SEED_NAMESPACE + ":" + ":".join(map(str, parts))).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _finite(value: object) -> bool:
    return value is not None and math.isfinite(float(value))


def _metric(row: dict, name: str) -> float | None:
    value = row["test_metrics"].get(name)
    return float(value) if _finite(value) else None


def _cell_key(row: dict) -> tuple[str, int, int | None, str]:
    return (
        str(row["task"]), int(row["selected_common_budget"]), row["frf_density"],
        str(row["family"]),
    )


def _instance_key(row: dict) -> tuple[str, int]:
    return str(row["dataset"]["curve_id"]), int(row["data_seed"])


def _pair_key(row: dict) -> tuple[tuple[str, int], int]:
    return _instance_key(row), int(row["init_seed"])


def _mode_value(value: dict, path: Path, artifact_mode: str) -> None:
    mode = value.get("artifact_mode")
    if artifact_mode == "test_fixture":
        if mode != "test_fixture":
            raise ValueError(f"fixture input lacks artifact_mode=test_fixture: {path}")
    elif mode == "test_fixture":
        raise ValueError(f"production artifact path rejects test_fixture input: {path}")
    elif mode not in {None, "production"}:
        raise ValueError(f"unsupported artifact mode in {path}")


def _resolve_lineage_path(input_root: Path, raw_path: str, artifact_mode: str) -> Path:
    candidate = Path(raw_path)
    alternatives = [candidate] if candidate.is_absolute() else [input_root / candidate, ROOT / candidate, candidate]
    path = next((item.resolve() for item in alternatives if item.exists()), None)
    if path is None:
        raise ValueError(f"frozen selection path does not exist: {raw_path}")
    if artifact_mode == "production" and input_root.resolve() not in path.parents:
        raise ValueError("production selection lineage escapes the input root")
    return path


def validate_scientific_record(
    record: dict, *, inventory: dict | None = None,
    candidate_manifest_sha256: str | None = None, artifact_mode: str = "production",
) -> None:
    """Run production validation after enforcing fixture/production separation."""
    if artifact_mode not in ARTIFACT_MODES:
        raise ValueError(f"unsupported artifact_mode {artifact_mode!r}")
    _mode_value(record, Path("evaluation record"), artifact_mode)
    clean_record = dict(record)
    clean_record.pop("artifact_mode", None)
    window = clean_record.get("window_spec", {})
    source_role = window.get("source_role") if isinstance(window, dict) else None
    detector = window.get("detector", {}) if isinstance(window, dict) else {}
    if source_role not in {"train_validation", "generator_metadata", "no_identifiable_feature", None}:
        raise ValueError("test-derived window is forbidden")
    if isinstance(detector, dict) and detector.get("test_response_used") is True:
        raise ValueError("test-derived window is forbidden")
    validate_peak_sensitive_record(
        clean_record, inventory=inventory,
        candidate_manifest_sha256=candidate_manifest_sha256,
    )
    if clean_record["stage"] != "peak_sensitive_evaluation":
        raise ValueError("old stage names or tuning records are not artifact inputs")
    if clean_record["window_spec"]["status"] == "no_identifiable_feature":
        if clean_record["test_metrics"]["resonance_window_complex_nrmse"] is not None:
            raise ValueError("no-window instance must explicitly omit the peak metric")


def _expected_scope(config: dict) -> dict:
    return {
        "tasks": list(config["fixed_scientific_tasks"]),
        "families": list(config["families"]),
        "budgets": [int(value) for value in config["budgets"]],
        "frf_densities": [int(value) for value in config["frf_densities"]],
    }


def _exclusion_key(row: dict) -> tuple[str, int, int | None]:
    return str(row.get("family")), int(row.get("budget", -1)), row.get("frf_density")


def _resolve_selection_candidate(selection: dict, inventory: dict, *, task: str,
                                 family: str, budget: int, inventory_hash: str) -> dict | None:
    """Resolve selected lineage payloads to the immutable candidate inventory."""
    required = {
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "candidate_manifest_sha256": inventory_hash,
    }
    for field, expected in required.items():
        if selection.get(field) != expected:
            raise ValueError(f"selection {field} differs from canonical lineage")
    if selection.get("status") != "selected":
        return None
    try:
        candidates = inventory["tasks"][task][family][str(budget)]
    except (KeyError, TypeError) as error:
        raise ValueError("selection scope is absent from the candidate inventory") from error
    candidate_payload = selection.get("candidate")
    if not isinstance(candidate_payload, dict):
        raise ValueError("selected lineage lacks a candidate payload")
    candidate_key = candidate_payload.get("candidate_key")
    matches = [candidate for candidate in candidates if candidate.get("candidate_key") == candidate_key]
    if len(matches) != 1:
        raise ValueError("selection candidate is not a unique candidate inventory member")
    candidate = matches[0]
    expected_candidate = {**candidate, "target_budget": int(budget)}
    if candidate_payload != expected_candidate:
        raise ValueError("selection candidate differs from the candidate inventory")
    if selection.get("optimizer") not in FROZEN_CONFIG["optimizer_grids"][family]:
        raise ValueError("selection optimizer is not a frozen family optimizer")
    for field in ("median_validation_resonance_nrmse", "median_validation_global_nrmse"):
        value = selection.get(field)
        if value is not None and (not _finite(value) or float(value) < 0.0):
            raise ValueError(f"selection {field} is not a finite non-negative summary")
    return candidate


def _has_hash_bound_exclusion(exclusions: list[dict], *, raw_path: str | None,
                              selection_hashes: dict[str, str]) -> bool:
    return bool(raw_path) and any(
        exclusion.get("selection_path") == raw_path and raw_path in selection_hashes
        for exclusion in exclusions
    )


def _load_selected_cells(
    input_root: Path, config: dict, *, artifact_mode: str = "production",
) -> tuple[dict, dict, dict, str, dict]:
    """Validate every canonical selection hash, returning only headline-budget cells."""
    if artifact_mode not in ARTIFACT_MODES:
        raise ValueError(f"unsupported artifact_mode {artifact_mode!r}")
    input_root = Path(input_root)
    frozen_path = input_root / "frozen_task_budgets.json"
    inventory_path = input_root / "candidate_manifest.json"
    audit_path = input_root / "tuning_audit.json"
    if not all(path.exists() for path in (frozen_path, inventory_path, audit_path)):
        raise ValueError("artifact input requires frozen budgets, tuning audit, and candidate manifest")
    frozen, inventory, tuning_audit = map(_read_json, (frozen_path, inventory_path, audit_path))
    for value, path in ((frozen, frozen_path), (inventory, inventory_path), (tuning_audit, audit_path)):
        _mode_value(value, path, artifact_mode)

    expected_scope = _expected_scope(config)
    inventory_hash = _sha256(inventory_path)
    required_frozen = {
        "protocol_key": config["protocol_key"],
        "protocol_mode": "production",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "scope": expected_scope,
        "scope_sha256": _canonical_hash(expected_scope),
        "candidate_manifest_sha256": inventory_hash,
        "tuning_audit_sha256": _sha256(audit_path),
    }
    for field, expected in required_frozen.items():
        if frozen.get(field) != expected:
            if field in {"scope", "scope_sha256"}:
                raise ValueError("frozen artifact does not have the exact six-task production scope")
            raise ValueError(f"frozen budget {field} mismatch")
    required_audit = {
        "status": "passed",
        "protocol_mode": "production",
        "scope": expected_scope,
        "scope_sha256": _canonical_hash(expected_scope),
        "candidate_manifest_sha256": inventory_hash,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "config_sha256": FROZEN_CONFIG_SHA256,
    }
    for field, expected in required_audit.items():
        if tuning_audit.get(field) != expected:
            raise ValueError(f"tuning audit {field} does not match exact canonical scope")
    if inventory.get("scope") != expected_scope:
        raise ValueError("candidate manifest scope does not match exact canonical scope")

    task_rows = frozen.get("tasks")
    if not isinstance(task_rows, dict) or set(task_rows) != set(expected_scope["tasks"]):
        raise ValueError("frozen artifact does not contain all exact six-task production scope entries")
    global_hashes = frozen.get("selection_file_sha256")
    if not isinstance(global_hashes, dict) or not global_hashes:
        raise ValueError("frozen artifact has no complete selection lineage")
    lineage: dict[str, dict] = {}
    per_task_union: dict[str, str] = {}
    cells: dict[tuple[str, int, int | None, str], dict] = {}
    canonical_task_fields = {
        "selected_common_budget", "losses", "relative_losses", "available_family_count",
        "pooled_median_relative_losses", "ineligible_budgets", "density_scope", "exclusions",
        "selection_file_sha256", "eligible_families_at_selected_budget",
    }

    for task in expected_scope["tasks"]:
        task_row = task_rows[task]
        missing_fields = canonical_task_fields - set(task_row)
        if missing_fields:
            raise ValueError(f"canonical frozen task entry missing fields for {task}: {sorted(missing_fields)}")
        budget = int(task_row["selected_common_budget"])
        if budget not in expected_scope["budgets"]:
            raise ValueError("selected common budget is outside the exact scope")
        densities = task_row["density_scope"]
        expected_densities = expected_scope["frf_densities"] if task == "aluminium_frf" else [None]
        if densities != expected_densities:
            raise ValueError(f"canonical density scope mismatch for {task}")
        task_hashes = task_row["selection_file_sha256"]
        if not isinstance(task_hashes, dict):
            raise ValueError("canonical task selection hashes must be an object")
        overlap = set(per_task_union).intersection(task_hashes)
        if overlap:
            raise ValueError("selection lineage is assigned to multiple tasks")
        per_task_union.update(task_hashes)
        exclusions = defaultdict(list)
        for exclusion in task_row["exclusions"]:
            exclusions[_exclusion_key(exclusion)].append(exclusion)
        selection_combinations = {}
        for raw_path, expected_hash in task_hashes.items():
            if global_hashes.get(raw_path) != expected_hash:
                raise ValueError("task/global selection lineage hashes disagree")
            path = _resolve_lineage_path(input_root, raw_path, artifact_mode)
            if _sha256(path) != expected_hash:
                raise ValueError("frozen selection hash does not match input artifact")
            selection = _read_json(path)
            _mode_value(selection, path, artifact_mode)
            family = selection.get("family")
            selection_budget = int(selection.get("target_budget", -1))
            density = selection.get("frf_density")
            if selection.get("task") != task:
                raise ValueError("selection task differs from its canonical task lineage")
            if family not in expected_scope["families"] or selection_budget not in expected_scope["budgets"]:
                raise ValueError("selection is outside exact family/budget scope")
            if density not in expected_densities:
                raise ValueError("selection density is outside exact task density scope")
            combination = (family, selection_budget, density)
            if combination in selection_combinations:
                raise ValueError("duplicate canonical selection combination")
            selection_combinations[combination] = raw_path
            candidate = _resolve_selection_candidate(
                selection, inventory, task=task, family=family, budget=selection_budget,
                inventory_hash=inventory_hash,
            )
            lineage[raw_path] = {
                "path": path, "sha256": expected_hash, "selection": selection,
                "candidate": candidate,
            }
        for family in expected_scope["families"]:
            for candidate_budget in expected_scope["budgets"]:
                for density in expected_densities:
                    combination = (family, candidate_budget, density)
                    if combination not in selection_combinations and not exclusions.get(combination):
                        raise ValueError(
                            f"canonical scope omission lacks explicit exclusion: {task}/{family}/{candidate_budget}/{density}"
                        )
        eligible = sorted(task_row["eligible_families_at_selected_budget"])
        expected_eligible = sorted(
            family for family, losses in task_row["losses"].items()
            if str(budget) in losses or budget in losses
        )
        if eligible != expected_eligible:
            raise ValueError(f"eligible family audit is not canonical for {task}")
        for family in eligible:
            for density in expected_densities:
                combination = (family, budget, density)
                raw_path = selection_combinations.get(combination)
                exclusion_rows = exclusions.get(combination, [])
                if raw_path is None:
                    if not _has_hash_bound_exclusion(
                        exclusion_rows, raw_path=None, selection_hashes=task_hashes,
                    ):
                        raise ValueError("missing headline selection lacks hash-bound canonical exclusion")
                    continue
                selection = lineage[raw_path]["selection"]
                if selection.get("status") != "selected":
                    if not _has_hash_bound_exclusion(
                        exclusion_rows, raw_path=raw_path, selection_hashes=task_hashes,
                    ):
                        raise ValueError("non-selected headline selection lacks hash-bound canonical exclusion")
                    continue
                key = (task, budget, density, family)
                cells[key] = lineage[raw_path]
    if per_task_union != global_hashes:
        raise ValueError("global selection lineage is not the exact union of task lineages")
    if not cells:
        raise ValueError("no realizable selected evaluation cells")
    return frozen, inventory, cells, inventory_hash, lineage


def _provenance_hashes(tasks: list[str]) -> dict[str, str]:
    paths = list(SOURCE_INPUTS)
    paths.extend(PROVENANCE_ROOT / ROLE_MANIFEST_NAMES[task] for task in tasks if task in ROLE_MANIFEST_NAMES)
    hashes = {}
    for path in paths:
        if not path.exists():
            raise ValueError(f"missing role/source provenance manifest: {path}")
        hashes[str(path.relative_to(ROOT))] = _sha256(path)
    return hashes


def _validate_fixture_manifest(input_root: Path, config: dict, artifact_mode: str) -> dict | None:
    path = input_root / "fixture_manifest.json"
    if artifact_mode == "production":
        if path.exists() and _read_json(path).get("artifact_mode") == "test_fixture":
            raise ValueError("production artifact path rejects test_fixture manifest")
        return None
    if not path.exists():
        raise ValueError("test_fixture input requires fixture_manifest.json")
    manifest = _read_json(path)
    _mode_value(manifest, path, artifact_mode)
    generator = manifest.get("generator", {})
    generator_path = ROOT / str(generator.get("path", ""))
    if not generator_path.exists() or _sha256(generator_path) != generator.get("sha256"):
        raise ValueError("fixture generator hash is not replayable")
    if not isinstance(generator.get("seed"), int) or not generator.get("version"):
        raise ValueError("fixture generator requires stable version and seed")
    raw_fixture = manifest.get("raw_fixture_inputs", {})
    spec_path = ROOT / str(raw_fixture.get("spec_path", ""))
    if not spec_path.exists() or _sha256(spec_path) != raw_fixture.get("spec_sha256"):
        raise ValueError("fixture spec hash is not replayable")
    spec = _read_json(spec_path)
    _mode_value(spec, spec_path, artifact_mode)
    expected_scope = _expected_scope(config)
    role_cells = sum(
        len(expected_scope["families"]) * len(
            expected_scope["frf_densities"] if task == "aluminium_frf" else [None]
        )
        for task in expected_scope["tasks"]
    )
    role_records = role_cells * len(config["evaluation_data_seeds"]) * len(config["evaluation_init_seeds"])
    if spec.get("scope") != expected_scope or manifest.get("scope") != expected_scope:
        raise ValueError("fixture scope does not match the exact canonical scope")
    if spec.get("expected_selected_cells") != role_cells or manifest.get("selected_cells") != role_cells:
        raise ValueError("fixture selected-cell count does not match frozen roles")
    if spec.get("record_count") != role_records or manifest.get("record_count") != role_records:
        raise ValueError("fixture record_count does not match frozen roles")
    manifest["_fixture_spec"] = spec
    return manifest


def audit_evaluation_records(
    input_root: Path, config: dict | None = None, *, artifact_mode: str = "production",
) -> dict:
    """Validate exact canonical scope, lineage, and 5-instance by 10-init cells."""
    input_root = Path(input_root)
    config = config or FROZEN_CONFIG
    if config != FROZEN_CONFIG:
        raise ValueError("artifact builder only accepts the frozen production configuration")
    fixture_manifest = _validate_fixture_manifest(input_root, config, artifact_mode)
    frozen, inventory, cells, inventory_hash, lineage = _load_selected_cells(
        input_root, config, artifact_mode=artifact_mode,
    )
    input_hashes = {
        "candidate_manifest.json": inventory_hash,
        "frozen_task_budgets.json": _sha256(input_root / "frozen_task_budgets.json"),
        "tuning_audit.json": _sha256(input_root / "tuning_audit.json"),
        str(CONFIG_PATH.relative_to(ROOT)): _sha256(CONFIG_PATH),
        str(SCHEMA_PATH.relative_to(ROOT)): _sha256(SCHEMA_PATH),
        str(BUILDER_PATH.relative_to(ROOT)): _sha256(BUILDER_PATH),
        str(VALIDATOR_PATH.relative_to(ROOT)): _sha256(VALIDATOR_PATH),
        **_provenance_hashes(config["fixed_scientific_tasks"]),
    }
    if fixture_manifest is not None:
        input_hashes["fixture_manifest.json"] = _sha256(input_root / "fixture_manifest.json")
        role_sources_path = input_root / "role_source_inputs.json"
        if not role_sources_path.exists():
            raise ValueError("fixture lacks replayable role/source input manifest")
        role_sources = _read_json(role_sources_path)
        _mode_value(role_sources, role_sources_path, artifact_mode)
        for row in role_sources.get("files", []):
            source_path = ROOT / row["path"]
            if not source_path.exists() or _sha256(source_path) != row["sha256"]:
                raise ValueError("fixture role/source manifest hash changed")
        input_hashes["role_source_inputs.json"] = _sha256(role_sources_path)
    input_hashes.update({raw: item["sha256"] for raw, item in lineage.items()})

    evaluation_root = input_root / "evaluation"
    jsonl_paths = sorted(evaluation_root.glob("**/*.jsonl")) if evaluation_root.exists() else []
    if not jsonl_paths:
        raise ValueError("no evaluation JSONL records found")
    records = []
    for path in jsonl_paths:
        input_hashes[str(path.relative_to(input_root))] = _sha256(path)
        for row in _read_jsonl(path):
            validate_scientific_record(
                row, inventory=inventory, candidate_manifest_sha256=inventory_hash,
                artifact_mode=artifact_mode,
            )
            key = _cell_key(row)
            if key not in cells:
                raise ValueError("evaluation record is outside selected frozen cells")
            selected = cells[key]
            record_selection = _resolve_lineage_path(input_root, row["selection_path"], artifact_mode)
            if record_selection != selected["path"] or row["selection_sha256"] != selected["sha256"]:
                raise ValueError("evaluation record selection hash/path mismatch")
            records.append(row)
    if {row["protocol_sha256"] for row in records} != {FROZEN_PROTOCOL_SHA256}:
        raise ValueError("mixed protocol hashes in evaluation records")
    if {row["config_sha256"] for row in records} != {FROZEN_CONFIG_SHA256}:
        raise ValueError("mixed configuration hashes in evaluation records")

    by_cell: dict[tuple, dict[tuple[tuple[str, int], int], dict]] = defaultdict(dict)
    for row in records:
        identity = _pair_key(row)
        cell = _cell_key(row)
        if identity in by_cell[cell]:
            raise ValueError("duplicate evaluation result key")
        by_cell[cell][identity] = row
    missing = []
    expected_pairs = len(config["evaluation_data_seeds"]) * len(config["evaluation_init_seeds"])
    for task, budget, density, family in sorted(cells):
        rows = by_cell[(task, budget, density, family)]
        if len(rows) != expected_pairs:
            present_seed_init = {(key[0][1], key[1]) for key in rows}
            for data_seed in config["evaluation_data_seeds"]:
                for init_seed in config["evaluation_init_seeds"]:
                    if (int(data_seed), int(init_seed)) not in present_seed_init:
                        missing.append(f"{task}/{family}/{budget}/seed:{data_seed}/init:{init_seed}")
    if missing:
        raise ValueError("missing evaluation keys: " + ", ".join(missing))
    expected_record_count = len(cells) * expected_pairs
    if len(records) != expected_record_count:
        raise ValueError("materialized evaluation record_count does not match canonical selected roles")
    if fixture_manifest is not None:
        fixture_spec = fixture_manifest["_fixture_spec"]
        if fixture_spec["expected_selected_cells"] != len(cells):
            raise ValueError("fixture spec selected-cell count does not match canonical selections")
        if fixture_spec["record_count"] != len(records) or fixture_manifest["record_count"] != len(records):
            raise ValueError("fixture record_count does not match materialized evaluation records")
    return {
        "frozen": frozen,
        "inventory": inventory,
        "cells": cells,
        "lineage": lineage,
        "records": records,
        "input_hashes": input_hashes,
        "fixture_manifest": fixture_manifest,
    }


def exact_sign_flip_p_value(
    instance_values: list[float], *, seed_parts: tuple[object, ...],
) -> float | None:
    values = np.asarray([float(value) for value in instance_values if _finite(value)], dtype=float)
    if len(values) == 0:
        return None
    observed = abs(float(np.mean(values)))
    tolerance = 1e-15
    if len(values) <= SIGN_FLIP_ENUMERATION_LIMIT:
        extreme = 0
        total = 1 << len(values)
        for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
            statistic = abs(float(np.mean(values * np.asarray(signs))))
            extreme += statistic + tolerance >= observed
        return float(extreme / total)
    rng = np.random.default_rng(_seed(*seed_parts, "sign_flip"))
    extreme = 0
    remaining = SIGN_FLIP_MONTE_CARLO_DRAWS
    while remaining:
        size = min(10_000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        statistics = np.abs(np.mean(signs * values[None, :], axis=1))
        extreme += int(np.sum(statistics + tolerance >= observed))
        remaining -= size
    return float((extreme + 1) / (SIGN_FLIP_MONTE_CARLO_DRAWS + 1))


def _not_estimable(reason: str = "no_paired_available_cases") -> dict:
    return {
        "state": "not_estimable",
        "reason": reason,
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "p_value": None,
        "median": None,
        "q1": None,
        "q3": None,
        "instance_count": 0,
        "init_count": 0,
    }


def hierarchical_paired_bootstrap(
    values: dict[object, dict[int, float]], *, seed_parts: tuple[object, ...],
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    """Weight instance means equally; resample paired init values within each instance."""
    cleaned = {
        instance: {int(init): float(value) for init, value in init_values.items() if _finite(value)}
        for instance, init_values in values.items()
    }
    cleaned = {instance: init_values for instance, init_values in cleaned.items() if init_values}
    instance_ids = sorted(cleaned, key=str)
    if not instance_ids:
        return _not_estimable()
    instance_means = np.asarray([
        np.mean(list(cleaned[instance].values())) for instance in instance_ids
    ], dtype=float)
    point = float(np.mean(instance_means))
    rng = np.random.default_rng(_seed(*seed_parts))
    draws = np.empty(replicates, dtype=float)
    for draw_index in range(replicates):
        sampled_indices = rng.integers(0, len(instance_ids), size=len(instance_ids))
        sampled_instance_means = []
        for sampled_index in sampled_indices:
            init_values = np.asarray(list(cleaned[instance_ids[int(sampled_index)]].values()), dtype=float)
            sampled_instance_means.append(float(np.mean(rng.choice(init_values, len(init_values), replace=True))))
        draws[draw_index] = float(np.mean(sampled_instance_means))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    q1, median, q3 = np.quantile(instance_means, [0.25, 0.5, 0.75])
    return {
        "state": "estimable",
        "reason": None,
        "estimate": point,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "p_value": exact_sign_flip_p_value(instance_means.tolist(), seed_parts=seed_parts),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "instance_count": len(instance_ids),
        "init_count": sum(len(values) for values in cleaned.values()),
    }


def frf_density_interaction(
    values: dict[int, dict[object, dict[int, float]]], *, seed_parts: tuple[object, ...],
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    """Fit slopes only on complete instance/init trajectories across all densities."""
    densities = sorted(values)
    if len(densities) < 2:
        return {
            "state": "not_estimable", "reason": "fewer_than_two_densities",
            "effect_per_log2_train_points": None, "intercept": None,
            "ci_low": None, "ci_high": None, "p_value": None,
            "median": None, "q1": None, "q3": None,
            "instance_count": 0, "trajectory_count": 0,
        }
    common_instances = set.intersection(*(set(values[density]) for density in densities))
    x = np.log2(np.asarray(densities, dtype=float))
    slopes: dict[object, dict[int, float]] = defaultdict(dict)
    intercepts = []
    for instance in sorted(common_instances, key=str):
        common_inits = set.intersection(*(set(values[density][instance]) for density in densities))
        for init in sorted(common_inits):
            trajectory = np.asarray([values[density][instance][init] for density in densities], dtype=float)
            if not np.all(np.isfinite(trajectory)):
                continue
            slope, intercept = np.polyfit(x, trajectory, 1)
            slopes[instance][int(init)] = float(slope)
            intercepts.append(float(intercept))
    stats = hierarchical_paired_bootstrap(slopes, seed_parts=seed_parts, replicates=replicates)
    if stats["state"] != "estimable":
        return {
            "state": "not_estimable", "reason": "no_complete_density_trajectories",
            "effect_per_log2_train_points": None, "intercept": None,
            "ci_low": None, "ci_high": None, "p_value": None,
            "median": None, "q1": None, "q3": None,
            "instance_count": 0, "trajectory_count": 0,
        }
    return {
        "state": "estimable",
        "reason": None,
        "effect_per_log2_train_points": stats["estimate"],
        "intercept": float(np.mean(intercepts)),
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "p_value": stats["p_value"],
        "median": stats["median"],
        "q1": stats["q1"],
        "q3": stats["q3"],
        "instance_count": stats["instance_count"],
        "trajectory_count": stats["init_count"],
    }


def _holm_adjust(
    rows: list[dict], *, p_value_field: str = "peak_p_value",
    output_field: str = "holm_peak_p_value",
) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["task"])].append(row)
    for task, subset in sorted(groups.items()):
        if len(subset) != 7 or len({row["baseline"] for row in subset}) != 7:
            raise ValueError(f"Holm adjustment requires one 7-contrast family for task {task}")
        for row in subset:
            row["holm_family_key"] = task
            row["holm_family_size"] = 7
            row[output_field] = None
        estimable = [row for row in subset if _finite(row.get(p_value_field))]
        ordered = sorted(estimable, key=lambda row: (float(row[p_value_field]), row["baseline"]))
        running = 0.0
        for index, row in enumerate(ordered):
            running = max(running, min(1.0, (7 - index) * float(row[p_value_field])))
            row[output_field] = running


def advantage_decision(
    peak_ci: tuple[float, float], global_relative_ci: tuple[float, float],
    failure_delta: float, favorable_instances: int,
) -> str:
    if peak_ci[1] >= 0.0:
        return "peak_superiority_not_supported"
    if global_relative_ci[1] > 0.10:
        return "peak_only_global_guardrail_failed"
    if failure_delta > 0.02:
        return "peak_only_failure_guardrail_failed"
    if favorable_instances < 4:
        return "peak_only_instance_direction_failed"
    return "supported"


def _prefixed_stats(prefix: str, stats: dict) -> dict:
    return {
        f"{prefix}_state": stats["state"],
        f"{prefix}_reason": stats["reason"],
        f"{prefix}_difference" if prefix == "peak" else f"{prefix}_relative_difference" if prefix == "global" else f"{prefix}_delta": stats["estimate"],
        f"{prefix}_ci_low": stats["ci_low"],
        f"{prefix}_ci_high": stats["ci_high"],
        f"{prefix}_p_value": stats["p_value"],
        f"{prefix}_median": stats["median"],
        f"{prefix}_q1": stats["q1"],
        f"{prefix}_q3": stats["q3"],
        f"{prefix}_instance_count": stats["instance_count"],
        f"{prefix}_init_count": stats["init_count"],
    }


def paired_effects(
    records: list[dict], cells: dict, *, replicates: int = BOOTSTRAP_REPLICATES,
) -> list[dict]:
    by_cell: dict[tuple, dict[tuple[tuple[str, int], int], dict]] = defaultdict(dict)
    for row in records:
        by_cell[_cell_key(row)][_pair_key(row)] = row
    primary = FROZEN_CONFIG["primary_family"]
    effects = []
    for task, budget, density, baseline in sorted(cells):
        if baseline == primary:
            continue
        primary_key = (task, budget, density, primary)
        baseline_key = (task, budget, density, baseline)
        if primary_key not in by_cell:
            raise ValueError(f"missing CFNN selected cell for {task}/{budget}/{density}")
        primary_pairs = by_cell[primary_key]
        baseline_pairs = by_cell[baseline_key]
        if set(primary_pairs) != set(baseline_pairs):
            raise ValueError("unpaired exact (instance, init) evaluation keys")
        peak_values: dict[object, dict[int, float]] = defaultdict(dict)
        global_values: dict[object, dict[int, float]] = defaultdict(dict)
        failure_values: dict[object, dict[int, float]] = defaultdict(dict)
        for (instance, init), cfnn in primary_pairs.items():
            other = baseline_pairs[(instance, init)]
            c_peak = _metric(cfnn, "resonance_window_complex_nrmse")
            b_peak = _metric(other, "resonance_window_complex_nrmse")
            if c_peak is not None and b_peak is not None:
                peak_values[instance][init] = c_peak - b_peak
            c_global = _metric(cfnn, "global_complex_nrmse")
            b_global = _metric(other, "global_complex_nrmse")
            if c_global is not None and b_global is not None and b_global != 0.0:
                global_values[instance][init] = (c_global - b_global) / b_global
            failure_values[instance][init] = float(
                cfnn["test_metrics"]["status"] == "numerical_failure"
            ) - float(other["test_metrics"]["status"] == "numerical_failure")
        peak_stats = hierarchical_paired_bootstrap(
            peak_values, seed_parts=(task, budget, density, baseline, "peak"),
            replicates=replicates,
        )
        global_stats = hierarchical_paired_bootstrap(
            global_values, seed_parts=(task, budget, density, baseline, "global"),
            replicates=replicates,
        )
        failure_stats = hierarchical_paired_bootstrap(
            failure_values, seed_parts=(task, budget, density, baseline, "failure"),
            replicates=replicates,
        )
        directions = [float(np.mean(list(values.values()))) for values in peak_values.values()]
        favorable = sum(value < 0.0 for value in directions)
        unfavorable = sum(value > 0.0 for value in directions)
        ties = len(directions) - favorable - unfavorable
        estimable = peak_stats["state"] == global_stats["state"] == "estimable"
        decision = (
            advantage_decision(
                (peak_stats["ci_low"], peak_stats["ci_high"]),
                (global_stats["ci_low"], global_stats["ci_high"]),
                failure_stats["estimate"], favorable,
            )
            if estimable else "not_estimable"
        )
        effects.append({
            "task": task,
            "selected_common_budget": budget,
            "frf_density": density,
            "baseline": baseline,
            **_prefixed_stats("peak", peak_stats),
            **_prefixed_stats("global", global_stats),
            **_prefixed_stats("failure", failure_stats),
            "favorable_instances": favorable,
            "unfavorable_instances": unfavorable,
            "tied_instances": ties,
            "advantage_state": "estimable" if estimable else "not_estimable",
            "advantage_decision": decision,
            "holm_peak_p_value": None,
            "holm_family_key": task if task != "aluminium_frf" else "aluminium_frf_slope",
            "holm_family_size": 7,
        })
    non_frf = [row for row in effects if row["task"] != "aluminium_frf"]
    for task in FROZEN_CONFIG["fixed_scientific_tasks"]:
        subset = [row for row in non_frf if row["task"] == task]
        if subset:
            _holm_adjust(subset)
    return effects


def frf_density_interactions(
    records: list[dict], effects: list[dict], *, replicates: int = BOOTSTRAP_REPLICATES,
) -> list[dict]:
    by_cell: dict[tuple, dict[tuple[tuple[str, int], int], dict]] = defaultdict(dict)
    for record in records:
        by_cell[_cell_key(record)][_pair_key(record)] = record
    primary = FROZEN_CONFIG["primary_family"]
    baselines = [family for family in FROZEN_CONFIG["families"] if family != primary]
    rows = []
    for baseline in baselines:
        paired: dict[int, dict[object, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        density_effects = sorted(
            (row for row in effects if row["task"] == "aluminium_frf" and row["baseline"] == baseline),
            key=lambda row: row["frf_density"],
        )
        for effect in density_effects:
            density = int(effect["frf_density"])
            primary_key = ("aluminium_frf", effect["selected_common_budget"], density, primary)
            baseline_key = ("aluminium_frf", effect["selected_common_budget"], density, baseline)
            for (instance, init), cfnn in by_cell[primary_key].items():
                other = by_cell[baseline_key].get((instance, init))
                c_peak = _metric(cfnn, "resonance_window_complex_nrmse")
                b_peak = _metric(other, "resonance_window_complex_nrmse") if other else None
                if c_peak is not None and b_peak is not None:
                    paired[density][instance][init] = c_peak - b_peak
        stats = frf_density_interaction(
            paired, seed_parts=("frf", baseline), replicates=replicates,
        )
        rows.append({
            "task": "aluminium_frf",
            "baseline": baseline,
            "analysis_role": "primary" if baseline in FRF_PRIMARY_BASELINES else "descriptive",
            "density_count": len(paired),
            **stats,
            "holm_p_value": None,
            "holm_family_key": "aluminium_frf",
            "holm_family_size": 7,
            "bootstrap_seed": _seed("frf", baseline),
        })
    _holm_adjust(rows, p_value_field="p_value", output_field="holm_p_value")
    rows.append({
        "task": "aluminium_frf",
        "baseline": "Vector fitting",
        "analysis_role": "primary",
        "density_count": 4,
        "state": "not_estimable",
        "reason": "excluded_non_realizable_runner_has_no_provenance_bound_vector_fitting_records",
        "effect_per_log2_train_points": None,
        "intercept": None,
        "ci_low": None,
        "ci_high": None,
        "p_value": None,
        "median": None,
        "q1": None,
        "q3": None,
        "instance_count": 0,
        "trajectory_count": 0,
        "holm_p_value": None,
        "holm_family_key": None,
        "holm_family_size": None,
        "bootstrap_seed": None,
    })
    return rows


def _instance_metric_summary(rows: list[dict], metric: str) -> tuple[float | None, float | None, float | None, int, int]:
    values: dict[object, list[float]] = defaultdict(list)
    for row in rows:
        value = _metric(row, metric)
        if value is not None:
            values[_instance_key(row)].append(value)
    instance_means = [float(np.mean(items)) for items in values.values() if items]
    if not instance_means:
        return None, None, None, 0, 0
    q1, median, q3 = np.quantile(instance_means, [0.25, 0.5, 0.75])
    return float(median), float(q1), float(q3), len(instance_means), sum(len(items) for items in values.values())


def _summary_rows(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        grouped[_cell_key(row)].append(row)
    summary = []
    for (task, budget, density, family), rows in sorted(grouped.items()):
        peak = _instance_metric_summary(rows, "resonance_window_complex_nrmse")
        global_stats = _instance_metric_summary(rows, "global_complex_nrmse")
        summary.append({
            "task": task,
            "selected_common_budget": budget,
            "frf_density": density,
            "family": family,
            "actual_parameters": int(np.median([row["actual_parameters"] for row in rows])),
            "record_count": len(rows),
            "peak_window_records": sum(_metric(row, "resonance_window_complex_nrmse") is not None for row in rows),
            "no_window_records": sum(row["window_spec"]["status"] == "no_identifiable_feature" for row in rows),
            "median_peak_nrmse": peak[0], "peak_q1": peak[1], "peak_q3": peak[2],
            "peak_instance_count": peak[3], "peak_init_count": peak[4],
            "median_global_nrmse": global_stats[0], "global_q1": global_stats[1], "global_q3": global_stats[2],
            "global_instance_count": global_stats[3], "global_init_count": global_stats[4],
            "median_clip_rate": float(np.median([float(row["gradient_clip_rate"]) for row in rows])),
            "training_cap_rate": float(np.mean([
                int(row["optimizer_steps"]) >= int(row["training_max_steps"]) for row in rows
            ])),
            "numerical_failure_rate": float(np.mean([
                row["test_metrics"]["status"] == "numerical_failure" for row in rows
            ])),
        })
    return summary


def _budget_rows(lineage: dict) -> list[dict]:
    rows = []
    for item in lineage.values():
        selection = item["selection"]
        candidate = item["candidate"]
        if selection.get("status") != "selected" or candidate is None:
            continue
        rows.append({
            "task": selection["task"],
            "family": selection["family"],
            "target_budget": int(selection["target_budget"]),
            "frf_density": selection.get("frf_density"),
            "actual_parameters": int(candidate["actual_parameters"]),
            "median_validation_peak_nrmse": selection.get("median_validation_resonance_nrmse"),
            "median_validation_global_nrmse": selection.get("median_validation_global_nrmse"),
            "selection_sha256": item["sha256"],
        })
    return sorted(rows, key=lambda row: (
        row["task"], row["family"], row["frf_density"] or -1, row["target_budget"],
    ))


def _failure_rows(effects: list[dict]) -> list[dict]:
    fields = (
        "task", "selected_common_budget", "frf_density", "baseline", "failure_state",
        "failure_delta", "failure_ci_low", "failure_ci_high", "failure_instance_count",
        "failure_init_count",
    )
    return [{field: row[field] for field in fields} for row in effects]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise ValueError(f"non-finite CSV value is forbidden in {path}")
            writer.writerow(row)


def _robust_limits(values: list[float], *, include_zero: bool = False) -> tuple[float, float]:
    finite = [float(value) for value in values if _finite(value)]
    if include_zero:
        finite.append(0.0)
    if not finite:
        return (0.0, 1.0)
    low, high = np.quantile(finite, [0.02, 0.98])
    padding = max(0.01, (high - low) * 0.12)
    return float(low - padding), float(high + padding)


def _save_pdf(path: Path, figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path, format="pdf",
        metadata={"Creator": "peak-sensitive-artifacts", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _draw_parameter_curves(path: Path, rows: list[dict]) -> None:
    tasks = FROZEN_CONFIG["fixed_scientific_tasks"]
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    for axis, task in zip(axes.flat, tasks):
        task_rows = [row for row in rows if row["task"] == task and _finite(row["median_validation_peak_nrmse"])]
        for family in FROZEN_CONFIG["families"]:
            family_rows = [row for row in task_rows if row["family"] == family]
            density_groups = sorted({row["frf_density"] for row in family_rows}, key=lambda value: value or -1)
            for density in density_groups:
                curve = sorted(
                    (row for row in family_rows if row["frf_density"] == density),
                    key=lambda row: row["actual_parameters"],
                )
                label = family if density is None else f"{family} ({density})"
                axis.plot(
                    [row["actual_parameters"] for row in curve],
                    [row["median_validation_peak_nrmse"] for row in curve],
                    marker="o", markersize=2.5, linewidth=0.8, label=label,
                )
        axis.set_title(task.replace("_", " "), fontsize=9)
        axis.set_xscale("log", base=2)
        axis.set_ylim(*_robust_limits([
            row["median_validation_peak_nrmse"] for row in task_rows
        ]))
        axis.set_xlabel("Actual parameters")
        axis.set_ylabel("Validation peak NRMSE")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6)
    _save_pdf(path, figure)


def _draw_eis(path: Path, rows: list[dict]) -> None:
    tasks = ("figshare_battery_eis", "hybrid_supercapacitor_eis")
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), constrained_layout=True)
    for axis, task in zip(axes, tasks):
        values = [row for row in rows if row["task"] == task and _finite(row["median_peak_nrmse"])]
        positions = np.arange(len(values))
        axis.bar(positions, [row["median_peak_nrmse"] for row in values], color="#4C78A8")
        axis.set_xticks(positions, [row["family"] for row in values], rotation=45, ha="right", fontsize=7)
        axis.set_ylim(*_robust_limits([row["median_peak_nrmse"] for row in values]))
        axis.set_title(task.replace("_", " "))
        axis.set_ylabel("Instance-median peak NRMSE")
    _save_pdf(path, figure)


def _draw_frf(path: Path, rows: list[dict]) -> None:
    values = [row for row in rows if row["task"] == "aluminium_frf" and _finite(row["peak_difference"])]
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for baseline in sorted({row["baseline"] for row in values}):
        curve = sorted((row for row in values if row["baseline"] == baseline), key=lambda row: row["frf_density"])
        axis.plot(
            [row["frf_density"] for row in curve], [row["peak_difference"] for row in curve],
            marker="o", linewidth=1.0, label=baseline,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(FROZEN_CONFIG["frf_densities"], FROZEN_CONFIG["frf_densities"])
    axis.set_ylim(*_robust_limits([row["peak_difference"] for row in values], include_zero=True))
    axis.set(xlabel="FRF training points", ylabel="CFNN minus baseline peak NRMSE")
    axis.legend(fontsize=7, ncol=2)
    _save_pdf(path, figure)


def _draw_effects(path: Path, rows: list[dict]) -> None:
    values = [row for row in rows if row["frf_density"] is None and row["peak_state"] == "estimable"]
    figure, axis = plt.subplots(figsize=(9.0, max(4.8, len(values) * 0.18)), constrained_layout=True)
    points = [row["peak_difference"] for row in values]
    errors = [
        [point - row["peak_ci_low"] for point, row in zip(points, values)],
        [row["peak_ci_high"] - point for point, row in zip(points, values)],
    ]
    axis.errorbar(points, range(len(values)), xerr=errors, fmt="o", markersize=3)
    axis.set_yticks(range(len(values)), [f"{row['task']}: {row['baseline']}" for row in values], fontsize=6)
    axis.set_xlim(*_robust_limits([
        *points, *(row["peak_ci_low"] for row in values), *(row["peak_ci_high"] for row in values),
    ], include_zero=True))
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("CFNN minus baseline peak NRMSE")
    _save_pdf(path, figure)


def _draw_clipping(path: Path, rows: list[dict]) -> None:
    tasks = FROZEN_CONFIG["fixed_scientific_tasks"]
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 6.5), constrained_layout=True)
    for axis, task in zip(axes.flat, tasks):
        values = [row for row in rows if row["task"] == task]
        positions = np.arange(len(values))
        axis.bar(positions - 0.2, [row["median_clip_rate"] for row in values], width=0.4, label="Clipping")
        axis.bar(positions + 0.2, [row["training_cap_rate"] for row in values], width=0.4, label="Step cap")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(task.replace("_", " "), fontsize=9)
        axis.set_xticks([])
        axis.set_ylabel("Rate")
    axes.flat[0].legend(fontsize=7)
    _save_pdf(path, figure)


def _load_reconstruction(input_root: Path, artifact_mode: str, records: list[dict]) -> tuple[dict | None, dict]:
    path = input_root / "reconstruction_inputs.json"
    metadata = {
        "path": FIGURE_PATHS["representative_reconstructions"],
        "source_fields": ["frequency", "truth_real", "truth_imag", "prediction_real", "prediction_imag"],
    }
    if not path.exists():
        return None, {**metadata, "applicability": "unavailable", "reason": "no_provenance_bound_reconstruction_arrays"}
    value = _read_json(path)
    _mode_value(value, path, artifact_mode)
    source = value.get("source", {})
    matching = [
        row for row in records
        if row["task"] == source.get("task") and row["family"] == source.get("family")
        and row["data_seed"] == source.get("data_seed") and row["init_seed"] == source.get("init_seed")
    ]
    if len(matching) != 1:
        raise ValueError("reconstruction arrays do not resolve to one evaluated source record")
    fields = metadata["source_fields"]
    lengths = {len(value.get(field, [])) for field in fields}
    if len(lengths) != 1 or next(iter(lengths), 0) < 2:
        raise ValueError("reconstruction arrays have inconsistent shapes")
    return value, {
        **metadata,
        "applicability": "generated",
        "reason": None,
        "source_record": source,
        "source_sha256": _sha256(path),
    }


def _draw_reconstruction(path: Path, value: dict) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True, constrained_layout=True)
    x = value["frequency"]
    for axis, component in zip(axes, ("real", "imag")):
        axis.plot(x, value[f"truth_{component}"], label="Reference", linewidth=1.2)
        axis.plot(x, value[f"prediction_{component}"], label="Reconstruction", linewidth=1.0)
        axis.set_ylabel(component.capitalize())
        axis.set_ylim(*_robust_limits([*value[f"truth_{component}"], *value[f"prediction_{component}"]]))
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Frequency coordinate")
    _save_pdf(path, figure)


def _build_figures(
    output_root: Path, input_root: Path, summary: list[dict], budget_rows: list[dict],
    effects: list[dict], records: list[dict], artifact_mode: str,
) -> tuple[dict, dict[str, str]]:
    specifications = {
        "parameter_curves": (
            bool(budget_rows), ["selection.candidate.actual_parameters", "selection.target_budget", "selection.median_validation_resonance_nrmse"],
            lambda path: _draw_parameter_curves(path, budget_rows), "no_budget_selection_rows",
        ),
        "eis_family_comparison": (
            any(row["task"].endswith("_eis") and _finite(row["median_peak_nrmse"]) for row in summary),
            ["summary.task", "summary.family", "summary.median_peak_nrmse"],
            lambda path: _draw_eis(path, summary), "no_estimable_eis_peak_rows",
        ),
        "frf_density_interaction": (
            any(row["task"] == "aluminium_frf" and _finite(row["peak_difference"]) for row in effects),
            ["paired_effects.frf_density", "paired_effects.baseline", "paired_effects.peak_difference"],
            lambda path: _draw_frf(path, effects), "no_estimable_frf_density_rows",
        ),
        "paired_effects": (
            any(row["frf_density"] is None and row["peak_state"] == "estimable" for row in effects),
            ["paired_effects.peak_difference", "paired_effects.peak_ci_low", "paired_effects.peak_ci_high"],
            lambda path: _draw_effects(path, effects), "no_estimable_non_frf_effects",
        ),
        "clipping_cap_rates": (
            bool(summary), ["summary.median_clip_rate", "summary.training_cap_rate"],
            lambda path: _draw_clipping(path, summary), "no_summary_rows",
        ),
    }
    figure_manifest, hashes = {}, {}
    for name, (applicable, source_fields, draw, reason) in specifications.items():
        relative = FIGURE_PATHS[name]
        if applicable:
            draw(output_root / relative)
            hashes[relative] = _sha256(output_root / relative)
        figure_manifest[name] = {
            "path": relative,
            "applicability": "generated" if applicable else "not_applicable",
            "reason": None if applicable else reason,
            "source_fields": source_fields,
        }
    reconstruction, reconstruction_metadata = _load_reconstruction(input_root, artifact_mode, records)
    if reconstruction is not None:
        relative = FIGURE_PATHS["representative_reconstructions"]
        _draw_reconstruction(output_root / relative, reconstruction)
        hashes[relative] = _sha256(output_root / relative)
    figure_manifest["representative_reconstructions"] = reconstruction_metadata
    return figure_manifest, hashes


_DIGIT_WORDS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
}


def _macro_token(value: object) -> str:
    output = []
    for character in str(value):
        if character.isalpha() and character.isascii():
            output.append(character)
        elif character in _DIGIT_WORDS:
            output.append(_DIGIT_WORDS[character])
        else:
            output.append("X")
    token = "".join(output).strip("X") or "None"
    return token


def _macro_value(value: object, *, state: str | None = None) -> str:
    if value is None or state == "not_estimable":
        return r"\textnormal{not estimable}"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite LaTeX macro value is forbidden")
        return f"{value:.4f}"
    return r"\detokenize{" + str(value) + "}"


def _write_macros(path: Path, effects: list[dict], interactions: list[dict]) -> None:
    lines = ["% Generated by build_peak_sensitive_artifacts.py; do not edit."]
    names = set()

    def define(name: str, value: str) -> None:
        if not name.isalpha() or name in names:
            raise ValueError(f"LaTeX macro name is not unique letter-only text: {name}")
        names.add(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for row in effects:
        identity = (
            "Task" + _macro_token(row["task"]) + "Baseline" + _macro_token(row["baseline"])
            + "Density" + _macro_token(row["frf_density"] if row["frf_density"] is not None else "none")
        )
        state = row["peak_state"]
        define("PeakState" + identity, _macro_value(state))
        define("PeakEffect" + identity, _macro_value(row["peak_difference"], state=state))
        define("PeakIntervalLow" + identity, _macro_value(row["peak_ci_low"], state=state))
        define("PeakIntervalHigh" + identity, _macro_value(row["peak_ci_high"], state=state))
        define("PeakMedian" + identity, _macro_value(row.get("peak_median"), state=state))
        define("PeakQuartileLow" + identity, _macro_value(row.get("peak_q1"), state=state))
        define("PeakQuartileHigh" + identity, _macro_value(row.get("peak_q3"), state=state))
        define("PeakPValue" + identity, _macro_value(row["peak_p_value"], state=state))
        define("PeakHolmPValue" + identity, _macro_value(row["holm_peak_p_value"], state=state))
        define("PeakDecision" + identity, _macro_value(row["advantage_decision"]))
        define("PeakContrast" + identity, _macro_value(row["baseline"]))
    for row in interactions:
        identity = "TaskAluminiumFrfBaseline" + _macro_token(row["baseline"])
        state = row["state"]
        define("FrfSlopeState" + identity, _macro_value(state))
        define("FrfSlopeEffect" + identity, _macro_value(row["effect_per_log2_train_points"], state=state))
        define("FrfSlopeIntervalLow" + identity, _macro_value(row["ci_low"], state=state))
        define("FrfSlopeIntervalHigh" + identity, _macro_value(row["ci_high"], state=state))
        define("FrfSlopePValue" + identity, _macro_value(row["p_value"], state=state))
        define("FrfSlopeHolmPValue" + identity, _macro_value(row["holm_p_value"], state=state))
    path.write_text("\n".join(lines) + "\n")


def _governing_analysis_scope() -> dict:
    return {
        "holm": {
            "family_definition": "one seven-neural-baseline contrast family per task",
            "family_size": 7,
            "aluminium_frf_test_statistic": "paired density interaction slope",
        },
        "frf_primary_comparisons": {
            baseline: (
                {"state": "excluded_non_realizable", "reason": "runner_produces_no_provenance_bound_vector_fitting_records"}
                if baseline == "Vector fitting" else {"state": "realizable", "source": "runner_evaluation_records"}
            )
            for baseline in FRF_PRIMARY_BASELINES
        },
    }


def build_artifacts(
    input_root: Path, output_root: Path, *, config_path: Path = CONFIG_PATH,
    schema_path: Path = SCHEMA_PATH, artifact_mode: str = "production",
) -> dict:
    """Audit results and write deterministic production or explicit fixture artifacts."""
    if artifact_mode not in ARTIFACT_MODES:
        raise ValueError(f"unsupported artifact_mode {artifact_mode!r}")
    if Path(config_path).resolve() != CONFIG_PATH.resolve() or Path(schema_path).resolve() != SCHEMA_PATH.resolve():
        raise ValueError("artifact builder requires the frozen production config and schema paths")
    input_root, output_root = Path(input_root), Path(output_root)
    audit = audit_evaluation_records(input_root, FROZEN_CONFIG, artifact_mode=artifact_mode)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = _summary_rows(audit["records"])
    budget_rows = _budget_rows(audit["lineage"])
    bootstrap_replicates = 200 if artifact_mode == "test_fixture" else BOOTSTRAP_REPLICATES
    effects = paired_effects(
        audit["records"], audit["cells"], replicates=bootstrap_replicates,
    )
    interactions = frf_density_interactions(
        audit["records"], effects, replicates=bootstrap_replicates,
    )
    failures = _failure_rows(effects)
    rows_by_path = {
        "summary.csv": summary,
        "budget_curves.csv": budget_rows,
        "paired_effects.csv": effects,
        "frf_density_interactions.csv": interactions,
        "failure_summary.csv": failures,
    }
    for relative, rows in rows_by_path.items():
        _write_csv(output_root / relative, rows)
    _write_macros(output_root / "latex_macros.tex", effects, interactions)
    figures, figure_hashes = _build_figures(
        output_root, input_root, summary, budget_rows, effects, audit["records"], artifact_mode,
    )
    output_hashes = {
        **{relative: _sha256(output_root / relative) for relative in TABLE_OUTPUTS},
        **figure_hashes,
    }
    fixture = audit["fixture_manifest"]
    provenance_input_hashes = audit["input_hashes"]
    materialized_input_digest = None
    materialized_input_file_count = None
    manifest_input_root = str(input_root.resolve())
    if fixture is not None:
        raw_fixture = fixture["raw_fixture_inputs"]
        spec_path = ROOT / raw_fixture["spec_path"]
        generator_path = ROOT / fixture["generator"]["path"]
        provenance_input_hashes = {
            raw_fixture["spec_path"]: _sha256(spec_path),
            fixture["generator"]["path"]: _sha256(generator_path),
            **_provenance_hashes(FROZEN_CONFIG["fixed_scientific_tasks"]),
            str(CONFIG_PATH.relative_to(ROOT)): _sha256(CONFIG_PATH),
            str(SCHEMA_PATH.relative_to(ROOT)): _sha256(SCHEMA_PATH),
            str(BUILDER_PATH.relative_to(ROOT)): _sha256(BUILDER_PATH),
            str(VALIDATOR_PATH.relative_to(ROOT)): _sha256(VALIDATOR_PATH),
        }
        materialized_input_digest = _canonical_hash(audit["input_hashes"])
        materialized_input_file_count = len(audit["input_hashes"])
        manifest_input_root = "replayable_compact_fixture_spec"
    manifest = {
        "artifact_mode": artifact_mode,
        "protocol_key": FROZEN_CONFIG["protocol_key"],
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed_namespace": BOOTSTRAP_SEED_NAMESPACE,
            "sign_flip_enumeration_limit": SIGN_FLIP_ENUMERATION_LIMIT,
            "sign_flip_monte_carlo_draws": SIGN_FLIP_MONTE_CARLO_DRAWS,
        },
        "statistical_code": {
            "builder_path": str(BUILDER_PATH.relative_to(ROOT)),
            "builder_sha256": _sha256(BUILDER_PATH),
            "validator_path": str(VALIDATOR_PATH.relative_to(ROOT)),
            "validator_sha256": _sha256(VALIDATOR_PATH),
        },
        "fixture_generator": fixture["generator"] if fixture else None,
        "raw_fixture_inputs": fixture["raw_fixture_inputs"] if fixture else None,
        "input_root": manifest_input_root,
        "input_hashes": provenance_input_hashes,
        "materialized_input_digest": materialized_input_digest,
        "materialized_input_file_count": materialized_input_file_count,
        "common_budget_sha256": audit["input_hashes"]["frozen_task_budgets.json"],
        "governing_analysis_scope": _governing_analysis_scope(),
        "figures": figures,
        "output_hashes": output_hashes,
        "selected_cells": len(audit["cells"]),
        "selection_lineage_files": len(audit["lineage"]),
        "evaluation_records": len(audit["records"]),
    }
    manifest_path = output_root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {**manifest, "output_hashes": {**output_hashes, "artifact_manifest.json": _sha256(manifest_path)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", default=CONFIG_PATH, type=Path)
    parser.add_argument("--schema", default=SCHEMA_PATH, type=Path)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    build_artifacts(
        args.input_root, args.output_root or args.input_root,
        config_path=args.config, schema_path=args.schema, artifact_mode="production",
    )


if __name__ == "__main__":
    main()
